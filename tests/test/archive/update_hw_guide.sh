#\!/bin/bash

# Create a backup
cp HARDWARE_BENCHMARKING_GUIDE.md HARDWARE_BENCHMARKING_GUIDE.md.bak

# Replace database-related paths
sed -i "s|python test/benchmark_db_|python duckdb_api/benchmark_db_|g" HARDWARE_BENCHMARKING_GUIDE.md

# Update test_comprehensive_hardware_coverage.py paths
sed -i "s|python test/test_comprehensive_hardware_coverage.py|python duckdb_api/test_comprehensive_hardware_coverage.py|g" HARDWARE_BENCHMARKING_GUIDE.md


