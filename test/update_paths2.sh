#\!/bin/bash

# Update CI benchmark integrator paths
sed -i "s|python test/scripts/ci_benchmark_integrator.py|python duckdb_api/ci_benchmark_integrator.py|g" BENCHMARK_DATABASE_GUIDE.md

# Update test_comprehensive_hardware_coverage.py paths
sed -i "s|python test/test_comprehensive_hardware_coverage.py|python duckdb_api/test_comprehensive_hardware_coverage.py|g" BENCHMARK_DATABASE_GUIDE.md

# Update other test paths
sed -i "s|python test/scripts/|python duckdb_api/|g" BENCHMARK_DATABASE_GUIDE.md

