#\!/bin/bash

# Create a backup
cp BENCHMARK_DATABASE_GUIDE.md BENCHMARK_DATABASE_GUIDE.md.bak

# Replace database-related paths
sed -i "s|python test/benchmark_db_|python duckdb_api/benchmark_db_|g" BENCHMARK_DATABASE_GUIDE.md
sed -i "s|python test/scripts/benchmark_db_|python duckdb_api/benchmark_db_|g" BENCHMARK_DATABASE_GUIDE.md
sed -i "s|python test/scripts/create_|python duckdb_api/create_|g" BENCHMARK_DATABASE_GUIDE.md
sed -i "s|python test/run_benchmark_with_db.py|python duckdb_api/run_benchmark_with_db.py|g" BENCHMARK_DATABASE_GUIDE.md
sed -i "s|python test/fixed_benchmark_db_query.py|python duckdb_api/fixed_benchmark_db_query.py|g" BENCHMARK_DATABASE_GUIDE.md
sed -i "s|python test/run_incremental_benchmarks.py|python duckdb_api/run_incremental_benchmarks.py|g" BENCHMARK_DATABASE_GUIDE.md

# Update generator paths
sed -i "s|python test/run_model_benchmarks.py|python generators/run_model_benchmarks.py|g" BENCHMARK_DATABASE_GUIDE.md

# Keep test files in test directory

