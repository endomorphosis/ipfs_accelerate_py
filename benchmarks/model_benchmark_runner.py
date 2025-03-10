2025-03-06 21:07:08,168 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:08,168 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:08,169 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:07:08,169 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:07:08,170 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:08,170 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:07:08,171 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:07:08,171 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:07:08,171 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:07:08,171 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:07:08,171 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:07:08,204 - benchmark_db_api - ERROR - Error executing query: Parser Error: syntax error at or near "."
2025-03-06 21:07:08,209 - incremental_benchmarks - ERROR - Error querying database for incomplete benchmarks: Parser Error: syntax error at or near "."
2025-03-06 21:07:08,211 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:07:45,264 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:45,264 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:45,264 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:07:45,264 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:07:45,265 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:07:45,300 - benchmark_db_api - ERROR - Error executing query: Binder Error: Table "pr" does not have a column named "created_at"

Candidate bindings: : "average_latency_ms"

LINE 7:                     pr.created_at,
                            ^
2025-03-06 21:07:45,303 - incremental_benchmarks - ERROR - Error querying database for incomplete benchmarks: Binder Error: Table "pr" does not have a column named "created_at"

Candidate bindings: : "average_latency_ms"

LINE 7:                     pr.created_at,
                            ^
2025-03-06 21:07:45,307 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:08:26,180 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:08:26,180 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:08:26,197 - benchmark_db_api - ERROR - Error executing query: Binder Error: Table "pr" does not have a column named "created_at"

Candidate bindings: : "average_latency_ms"

LINE 7:                     COALESCE(pr.created_at, CURRENT_TIMESTAMP) as created_date,
                                     ^
2025-03-06 21:08:26,199 - incremental_benchmarks - ERROR - Error querying database for incomplete benchmarks: Binder Error: Table "pr" does not have a column named "created_at"

Candidate bindings: : "average_latency_ms"

LINE 7:                     COALESCE(pr.created_at, CURRENT_TIMESTAMP) as created_date,
                                     ^
2025-03-06 21:08:26,201 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:09:13,078 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:09:13,079 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:09:13,079 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:09:13,080 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:09:13,146 - incremental_benchmarks - INFO - Found 3 existing benchmark results in database
2025-03-06 21:09:13,149 - incremental_benchmarks - ERROR - Error querying database for incomplete benchmarks: can't compare offset-naive and offset-aware datetimes
2025-03-06 21:09:13,150 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:10:22,746 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:22,746 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:22,746 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:10:22,746 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:10:22,746 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:10:22,747 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:10:22,783 - incremental_benchmarks - INFO - Found 3 existing benchmark results in database
2025-03-06 21:10:22,785 - incremental_benchmarks - INFO - Found 0 benchmarks to run
2025-03-06 21:10:22,785 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:10:29,452 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:29,452 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:29,452 - incremental_benchmarks - INFO - No previous progress file found or error loading it. Starting fresh.
2025-03-06 21:10:29,452 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Hardware platforms: ['cuda']
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:10:29,453 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:10:29,500 - incremental_benchmarks - INFO - Found 3 existing benchmark results in database
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO - 
Running benchmark 1 of 2: bert-base-uncased:cuda:1
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO -   - Model: bert-base-uncased
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:10:29,502 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:10:29,503 - incremental_benchmarks - INFO -   - Priority: 2
2025-03-06 21:10:29,503 - incremental_benchmarks - INFO - Running benchmark: bert-base-uncased on cuda with batch size 1
2025-03-06 21:10:29,503 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for bert-base-uncased:cuda:1
2025-03-06 21:10:29,503 - incremental_benchmarks - INFO - Saved progress with 1 completed benchmarks
2025-03-06 21:10:30,503 - incremental_benchmarks - INFO - 
Running benchmark 2 of 2: bert-base-uncased:cuda:2
2025-03-06 21:10:30,504 - incremental_benchmarks - INFO -   - Model: bert-base-uncased
2025-03-06 21:10:30,504 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:10:30,504 - incremental_benchmarks - INFO -   - Batch size: 2
2025-03-06 21:10:30,504 - incremental_benchmarks - INFO -   - Priority: 2
2025-03-06 21:10:30,505 - incremental_benchmarks - INFO - Running benchmark: bert-base-uncased on cuda with batch size 2
2025-03-06 21:10:30,505 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for bert-base-uncased:cuda:2
2025-03-06 21:10:30,506 - incremental_benchmarks - INFO - Saved progress with 2 completed benchmarks
2025-03-06 21:10:30,507 - incremental_benchmarks - INFO - 
Completed 2 benchmarks successfully
2025-03-06 21:10:30,618 - benchmark_db_api - ERROR - Error executing query: Binder Error: Table "pr" does not have a column named "created_at"

Candidate bindings: : "average_latency_ms"

LINE 6:                         MAX(pr.created_at) as last_updated
                                    ^
2025-03-06 21:10:30,621 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211030.md
2025-03-06 21:10:45,563 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:45,563 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO - Loaded 2 completed benchmarks from progress file
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Hardware platforms: ['cuda']
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:10:45,564 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:10:45,609 - incremental_benchmarks - INFO - Found 3 existing benchmark results in database
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO - 
Running benchmark 1 of 2: bert-base-uncased:cuda:1
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO -   - Model: bert-base-uncased
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO -   - Priority: 2
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO - Running benchmark: bert-base-uncased on cuda with batch size 1
2025-03-06 21:10:45,611 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for bert-base-uncased:cuda:1
2025-03-06 21:10:45,612 - incremental_benchmarks - INFO - Saved progress with 2 completed benchmarks
2025-03-06 21:10:46,612 - incremental_benchmarks - INFO - 
Running benchmark 2 of 2: bert-base-uncased:cuda:2
2025-03-06 21:10:46,612 - incremental_benchmarks - INFO -   - Model: bert-base-uncased
2025-03-06 21:10:46,613 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:10:46,613 - incremental_benchmarks - INFO -   - Batch size: 2
2025-03-06 21:10:46,613 - incremental_benchmarks - INFO -   - Priority: 2
2025-03-06 21:10:46,613 - incremental_benchmarks - INFO - Running benchmark: bert-base-uncased on cuda with batch size 2
2025-03-06 21:10:46,613 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for bert-base-uncased:cuda:2
2025-03-06 21:10:46,614 - incremental_benchmarks - INFO - Saved progress with 2 completed benchmarks
2025-03-06 21:10:46,614 - incremental_benchmarks - INFO - 
Completed 2 benchmarks successfully
2025-03-06 21:10:46,712 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211046.md
2025-03-06 21:11:50,126 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:50,127 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO - Loaded 2 completed benchmarks from progress file
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Hardware platforms: ['cuda']
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Models: ['bert-base-uncased']
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:11:50,127 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:11:50,128 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:11:50,164 - incremental_benchmarks - INFO - Found 3 existing benchmark results in database
2025-03-06 21:11:50,166 - incremental_benchmarks - INFO - Found 0 benchmarks to run
2025-03-06 21:11:50,166 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:11:58,036 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:58,036 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:58,037 - incremental_benchmarks - INFO - Loaded 2 completed benchmarks from progress file
2025-03-06 21:11:58,037 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:11:58,037 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:11:58,037 - incremental_benchmarks - INFO -   - Hardware platforms: ['cuda']
2025-03-06 21:11:58,038 - incremental_benchmarks - INFO -   - Models: ['t5-small']
2025-03-06 21:11:58,038 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2]
2025-03-06 21:11:58,038 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:11:58,038 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:11:58,039 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:11:58,112 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:11:58,116 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:11:58,116 - incremental_benchmarks - INFO - Found 2 benchmarks to run
2025-03-06 21:11:58,116 - incremental_benchmarks - INFO - 
Running benchmark 1 of 2: t5-small:cuda:1
2025-03-06 21:11:58,117 - incremental_benchmarks - INFO -   - Model: t5-small
2025-03-06 21:11:58,117 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:11:58,118 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:11:58,118 - incremental_benchmarks - INFO -   - Priority: 1
2025-03-06 21:11:58,118 - incremental_benchmarks - INFO - Running benchmark: t5-small on cuda with batch size 1
2025-03-06 21:11:58,118 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for t5-small:cuda:1
2025-03-06 21:11:58,118 - incremental_benchmarks - INFO - Saved progress with 3 completed benchmarks
2025-03-06 21:11:59,118 - incremental_benchmarks - INFO - 
Running benchmark 2 of 2: t5-small:cuda:2
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO -   - Model: t5-small
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO -   - Hardware: cuda
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO -   - Batch size: 2
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO -   - Priority: 1
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO - Running benchmark: t5-small on cuda with batch size 2
2025-03-06 21:11:59,119 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for t5-small:cuda:2
2025-03-06 21:11:59,120 - incremental_benchmarks - INFO - Saved progress with 4 completed benchmarks
2025-03-06 21:11:59,120 - incremental_benchmarks - INFO - 
Completed 2 benchmarks successfully
2025-03-06 21:11:59,184 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211159.md
2025-03-06 21:12:50,227 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:12:50,227 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:12:50,227 - incremental_benchmarks - INFO - Loaded 4 completed benchmarks from progress file
2025-03-06 21:12:50,227 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:12:50,227 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:12:50,227 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:12:50,228 - incremental_benchmarks - INFO -   - Models: ['t5-small']
2025-03-06 21:12:50,228 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:12:50,228 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:12:50,228 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:12:50,228 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:12:50,262 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO - 
Running benchmark 1 of 1: t5-small:cpu:1
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO -   - Model: t5-small
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO -   - Priority: 1
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO - Running benchmark: t5-small on cpu with batch size 1
2025-03-06 21:12:50,264 - incremental_benchmarks - INFO - Executing command: /home/barberb/ipfs_accelerate_py/.venv/bin/python run_benchmark_with_db.py --model t5-small --hardware cpu --batch-sizes 1 --db /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb --simulate
2025-03-06 21:13:10,867 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:10,867 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO - Loaded 4 completed benchmarks from progress file
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO -   - Models: ['t5-small']
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:13:10,868 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:13:10,869 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:13:10,869 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:13:10,932 - incremental_benchmarks - INFO - Found 1 existing benchmark results in database
2025-03-06 21:13:10,935 - incremental_benchmarks - INFO - Found 0 benchmarks to run
2025-03-06 21:13:10,935 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:13:17,151 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:17,151 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:17,151 - incremental_benchmarks - INFO - Loaded 4 completed benchmarks from progress file
2025-03-06 21:13:17,151 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:13:17,151 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:17,151 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:13:17,152 - incremental_benchmarks - INFO -   - Models: ['wav2vec2-base']
2025-03-06 21:13:17,152 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:13:17,152 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:13:17,152 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:13:17,152 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:13:17,193 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:13:17,194 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:13:17,194 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO - 
Running benchmark 1 of 1: wav2vec2-base:cpu:1
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO -   - Model: wav2vec2-base
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO -   - Priority: 1
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO - Running benchmark: wav2vec2-base on cpu with batch size 1
2025-03-06 21:13:17,195 - incremental_benchmarks - INFO - Executing command: /home/barberb/ipfs_accelerate_py/.venv/bin/python run_benchmark_with_db.py --model wav2vec2-base --hardware cpu --batch-sizes 1 --db /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:21,525 - incremental_benchmarks - WARNING - Benchmark stderr:
2025-03-06 21:13:19,615 - benchmark_runner - WARNING - Improved hardware detection not available
2025-03-06 21:13:19,617 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:19,633 - benchmark_runner - INFO - Connected to database at /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:19,634 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:19,638 - benchmark_db_api - INFO - Added new model to database: wav2vec2-base (ID: 11)
2025-03-06 21:13:19,639 - benchmark_runner - INFO - Found or created model: wav2vec2-base (ID: 11, Family: audio)
2025-03-06 21:13:19,639 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:13:19,640 - benchmark_runner - INFO - Found or created hardware: cpu  (ID: 1)
2025-03-06 21:13:19,644 - benchmark_runner - INFO - Created new test run: benchmark_wav2vec2-base_cpu_20250306_211319 (ID: 1741316666)
2025-03-06 21:13:19,644 - benchmark_runner - INFO - Running benchmark: wav2vec2-base on cpu, embedding, batch_size=1
2025-03-06 21:13:19,644 - benchmark_runner - INFO - Running 10 warmup iterations
2025-03-06 21:13:19,844 - benchmark_runner - INFO - Running 100 benchmark iterations
2025-03-06 21:13:20,865 - benchmark_runner - INFO - Added performance result (ID: 273)
2025-03-06 21:13:20,865 - benchmark_runner - INFO -   - Test case: embedding, Batch size: 1
2025-03-06 21:13:20,865 - benchmark_runner - INFO -   - Latency: 31.46 ms, Throughput: 31.78 items/s
2025-03-06 21:13:20,865 - benchmark_runner - INFO -   - Memory: 3128.28 MB
2025-03-06 21:13:20,867 - benchmark_runner - INFO - Updated test run completion (ID: 1741316666, Execution time: 1.23s)
2025-03-06 21:13:20,892 - benchmark_runner - INFO - All benchmark results saved to database

2025-03-06 21:13:21,526 - incremental_benchmarks - INFO - Benchmark completed successfully: wav2vec2-base:cpu:1
2025-03-06 21:13:21,526 - incremental_benchmarks - INFO - Saved progress with 5 completed benchmarks
2025-03-06 21:13:21,526 - incremental_benchmarks - INFO - 
Completed 1 benchmarks successfully
2025-03-06 21:13:21,661 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211321.md
2025-03-06 21:14:03,957 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:14:03,957 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO - Loaded 5 completed benchmarks from progress file
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Models: ['wav2vec2-base']
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:14:03,958 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:14:04,005 - incremental_benchmarks - INFO - Found 1 existing benchmark results in database
2025-03-06 21:14:04,008 - incremental_benchmarks - INFO - Found 0 benchmarks to run
2025-03-06 21:14:04,009 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:16:24,549 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:24,549 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO - Loaded 5 completed benchmarks from progress file
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO -   - Models: ['whisper', 'clip', 't5', 'wav2vec2', 'vit', 'bert']
2025-03-06 21:16:24,550 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2, 4, 8, 16]
2025-03-06 21:16:24,551 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:16:24,551 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:16:24,551 - incremental_benchmarks - INFO -   - Max benchmarks: 5
2025-03-06 21:16:24,609 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:16:24,612 - incremental_benchmarks - INFO - Found 5 benchmarks to run
2025-03-06 21:16:24,612 - incremental_benchmarks - INFO - Found 5 benchmarks to run
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO - 
Running benchmark 1 of 5: whisper:cpu:1
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO -   - Model: whisper
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO - Running benchmark: whisper on cpu with batch size 1
2025-03-06 21:16:24,613 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for whisper:cpu:1
2025-03-06 21:16:24,614 - incremental_benchmarks - INFO - Saved progress with 6 completed benchmarks
2025-03-06 21:16:25,614 - incremental_benchmarks - INFO - 
Running benchmark 2 of 5: whisper:cpu:2
2025-03-06 21:16:25,614 - incremental_benchmarks - INFO -   - Model: whisper
2025-03-06 21:16:25,614 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:25,615 - incremental_benchmarks - INFO -   - Batch size: 2
2025-03-06 21:16:25,615 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:25,615 - incremental_benchmarks - INFO - Running benchmark: whisper on cpu with batch size 2
2025-03-06 21:16:25,615 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for whisper:cpu:2
2025-03-06 21:16:25,616 - incremental_benchmarks - INFO - Saved progress with 7 completed benchmarks
2025-03-06 21:16:26,616 - incremental_benchmarks - INFO - 
Running benchmark 3 of 5: whisper:cpu:4
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO -   - Model: whisper
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO -   - Batch size: 4
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO - Running benchmark: whisper on cpu with batch size 4
2025-03-06 21:16:26,617 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for whisper:cpu:4
2025-03-06 21:16:26,618 - incremental_benchmarks - INFO - Saved progress with 8 completed benchmarks
2025-03-06 21:16:27,618 - incremental_benchmarks - INFO - 
Running benchmark 4 of 5: whisper:cpu:8
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO -   - Model: whisper
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO -   - Batch size: 8
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO - Running benchmark: whisper on cpu with batch size 8
2025-03-06 21:16:27,619 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for whisper:cpu:8
2025-03-06 21:16:27,620 - incremental_benchmarks - INFO - Saved progress with 9 completed benchmarks
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO - 
Running benchmark 5 of 5: whisper:cpu:16
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO -   - Model: whisper
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO -   - Batch size: 16
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:28,620 - incremental_benchmarks - INFO - Running benchmark: whisper on cpu with batch size 16
2025-03-06 21:16:28,621 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for whisper:cpu:16
2025-03-06 21:16:28,621 - incremental_benchmarks - INFO - Saved progress with 10 completed benchmarks
2025-03-06 21:16:28,621 - incremental_benchmarks - INFO - 
Completed 5 benchmarks successfully
2025-03-06 21:16:28,675 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211628.md
2025-03-06 21:16:41,213 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:41,213 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO - Loaded 10 completed benchmarks from progress file
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO -   - Models: ['wav2vec2']
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO -   - Batch sizes: [1, 2, 4, 8, 16]
2025-03-06 21:16:41,214 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:16:41,215 - incremental_benchmarks - INFO -   - Priority only: True
2025-03-06 21:16:41,215 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:16:41,252 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:16:41,254 - incremental_benchmarks - INFO - Found 5 benchmarks to run
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO - Found 5 benchmarks to run
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO - 
Running benchmark 1 of 5: wav2vec2:cpu:1
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 1
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for wav2vec2:cpu:1
2025-03-06 21:16:41,255 - incremental_benchmarks - INFO - Saved progress with 11 completed benchmarks
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO - 
Running benchmark 2 of 5: wav2vec2:cpu:2
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO -   - Batch size: 2
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 2
2025-03-06 21:16:42,256 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for wav2vec2:cpu:2
2025-03-06 21:16:42,257 - incremental_benchmarks - INFO - Saved progress with 12 completed benchmarks
2025-03-06 21:16:43,257 - incremental_benchmarks - INFO - 
Running benchmark 3 of 5: wav2vec2:cpu:4
2025-03-06 21:16:43,257 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:16:43,257 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:43,257 - incremental_benchmarks - INFO -   - Batch size: 4
2025-03-06 21:16:43,258 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:43,258 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 4
2025-03-06 21:16:43,258 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for wav2vec2:cpu:4
2025-03-06 21:16:43,258 - incremental_benchmarks - INFO - Saved progress with 13 completed benchmarks
2025-03-06 21:16:44,258 - incremental_benchmarks - INFO - 
Running benchmark 4 of 5: wav2vec2:cpu:8
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO -   - Batch size: 8
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 8
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for wav2vec2:cpu:8
2025-03-06 21:16:44,259 - incremental_benchmarks - INFO - Saved progress with 14 completed benchmarks
2025-03-06 21:16:45,259 - incremental_benchmarks - INFO - 
Running benchmark 5 of 5: wav2vec2:cpu:16
2025-03-06 21:16:45,259 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO -   - Batch size: 16
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 16
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO - DRY RUN: Would run benchmark for wav2vec2:cpu:16
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO - Saved progress with 15 completed benchmarks
2025-03-06 21:16:45,260 - incremental_benchmarks - INFO - 
Completed 5 benchmarks successfully
2025-03-06 21:16:45,329 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211645.md
2025-03-06 21:16:56,992 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:56,992 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:56,992 - incremental_benchmarks - INFO - Loaded 15 completed benchmarks from progress file
2025-03-06 21:16:56,992 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:16:56,992 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Models: ['wav2vec2']
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:16:56,993 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:16:57,028 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:16:57,030 - incremental_benchmarks - INFO - Found 0 benchmarks to run
2025-03-06 21:16:57,030 - incremental_benchmarks - INFO - No incomplete benchmarks found. All done!
2025-03-06 21:17:01,813 - incremental_benchmarks - INFO - Using database path from environment: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:01,813 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:01,814 - incremental_benchmarks - INFO - Loaded 15 completed benchmarks from progress file
2025-03-06 21:17:01,814 - incremental_benchmarks - INFO - Initialized IncrementalBenchmarkRunner
2025-03-06 21:17:01,814 - incremental_benchmarks - INFO -   - Database: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:01,814 - incremental_benchmarks - INFO -   - Hardware platforms: ['cpu']
2025-03-06 21:17:01,815 - incremental_benchmarks - INFO -   - Models: ['wav2vec2']
2025-03-06 21:17:01,815 - incremental_benchmarks - INFO -   - Batch sizes: [1]
2025-03-06 21:17:01,815 - incremental_benchmarks - INFO -   - Refresh if older than 30 days
2025-03-06 21:17:01,815 - incremental_benchmarks - INFO -   - Priority only: False
2025-03-06 21:17:01,815 - incremental_benchmarks - INFO -   - Max benchmarks: 0
2025-03-06 21:17:01,863 - incremental_benchmarks - INFO - Found 0 existing benchmark results in database
2025-03-06 21:17:01,867 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:17:01,867 - incremental_benchmarks - INFO - Found 1 benchmarks to run
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO - 
Running benchmark 1 of 1: wav2vec2:cpu:1
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO -   - Model: wav2vec2
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO -   - Hardware: cpu
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO -   - Batch size: 1
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO -   - Priority: 0
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO - Running benchmark: wav2vec2 on cpu with batch size 1
2025-03-06 21:17:01,868 - incremental_benchmarks - INFO - Executing command: /home/barberb/ipfs_accelerate_py/.venv/bin/python run_benchmark_with_db.py --model wav2vec2 --hardware cpu --batch-sizes 1 --db /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb --simulate
2025-03-06 21:17:06,442 - incremental_benchmarks - WARNING - Benchmark stderr:
2025-03-06 21:17:05,195 - benchmark_runner - WARNING - Improved hardware detection not available
2025-03-06 21:17:05,198 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:05,211 - benchmark_runner - INFO - Connected to database at /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:05,212 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:05,215 - benchmark_db_api - INFO - Added new model to database: wav2vec2 (ID: 12)
2025-03-06 21:17:05,216 - benchmark_runner - INFO - Found or created model: wav2vec2 (ID: 12, Family: audio)
2025-03-06 21:17:05,216 - benchmark_db_api - INFO - Initialized BenchmarkDBAPI with DB: /home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb
2025-03-06 21:17:05,217 - benchmark_runner - INFO - Found or created hardware: cpu  (ID: 1)
2025-03-06 21:17:05,220 - benchmark_runner - INFO - Created new test run: benchmark_wav2vec2_cpu_20250306_211705 (ID: 1741316673)
2025-03-06 21:17:05,220 - benchmark_runner - INFO - Simulating benchmark: wav2vec2 on cpu, embedding, batch_size=1
2025-03-06 21:17:05,722 - benchmark_runner - INFO - Added performance result (ID: 280)
2025-03-06 21:17:05,722 - benchmark_runner - INFO -   - Test case: embedding, Batch size: 1
2025-03-06 21:17:05,722 - benchmark_runner - INFO -   - Latency: 26.60 ms, Throughput: 54.94 items/s
2025-03-06 21:17:05,722 - benchmark_runner - INFO -   - Memory: 4553.52 MB
2025-03-06 21:17:05,723 - benchmark_runner - INFO - Updated test run completion (ID: 1741316673, Execution time: 0.51s)
2025-03-06 21:17:05,738 - benchmark_runner - INFO - All benchmark results saved to database

2025-03-06 21:17:06,442 - incremental_benchmarks - INFO - Benchmark completed successfully: wav2vec2:cpu:1
2025-03-06 21:17:06,443 - incremental_benchmarks - INFO - Saved progress with 15 completed benchmarks
2025-03-06 21:17:06,443 - incremental_benchmarks - INFO - 
Completed 1 benchmarks successfully
2025-03-06 21:17:06,500 - incremental_benchmarks - INFO - Generated benchmark report: benchmark_report_20250306_211706.md
