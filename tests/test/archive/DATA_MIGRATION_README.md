# IPFS Accelerate Test Results Migration Tool

This tool migrates legacy JSON test results for the IPFS Accelerate package to the DuckDB database. It provides comprehensive validation, deduplication, and reporting capabilities.

## Features

- **Smart File Discovery**: Automatically identifies IPFS-related test result files
- **Flexible Data Extraction**: Extracts test results from various JSON structures
- **Comprehensive Validation**: Validates JSON files for format and data integrity
- **Schema Validation**: Ensures test results conform to expected schema
- **Deduplication**: Prevents duplicate results in the database
- **Archiving**: Archives original JSON files after successful migration
- **Reporting**: Generates detailed migration reports
- **Configurable**: Supports different validation modes and options

## Installation

The migration tool is part of the IPFS Accelerate Python package test suite. To use it, you need to have the following dependencies installed:

```bash
pip install duckdb pandas
```

## Usage

### Basic Usage

```bash
# Migrate test results from specific directories
python migrate_ipfs_test_results.py --input-dirs ./test_results ./archived_results

# Migrate and archive original files
python migrate_ipfs_test_results.py --input-dirs ./test_results --archive

# Migrate, archive, and generate a report
python migrate_ipfs_test_results.py --input-dirs ./test_results --archive --report

# Use strict validation
python migrate_ipfs_test_results.py --input-dirs ./test_results --validate-strict
```

### Advanced Options

```bash
# Migrate test results, archive and delete original files after successful migration
python migrate_ipfs_test_results.py --input-dirs ./test_results --delete

# Specify database path and archive directory
python migrate_ipfs_test_results.py --input-dirs ./test_results --db-path ./custom_db.duckdb --archive-dir ./archived_results

# Create a compressed archive package of all processed files
python migrate_ipfs_test_results.py --input-dirs ./test_results --create-archive-package

# Enable verbose logging
python migrate_ipfs_test_results.py --input-dirs ./test_results --verbose
```

## Testing the Migration Tool

A dedicated test script is provided to verify the migration tool's functionality. It creates sample test files and tests the migration process.

```bash
# Create sample test files
python generators/models/test_ipfs_migration.py --create-samples

# Run migration on sample files
python generators/models/test_ipfs_migration.py --migrate

# Validate migration results
python generators/models/test_ipfs_migration.py --validate

# Run all steps: create samples, migrate, and validate
python generators/models/test_ipfs_migration.py --all

# Clean up after testing
python generators/models/test_ipfs_migration.py --all --cleanup
```

## Understanding the Migration Process

The migration process consists of several key steps:

1. **File Discovery**: The tool searches for JSON files in the specified directories and identifies those that appear to contain IPFS test results. It uses file naming patterns and content inspection to make this determination.

2. **File Validation**: Each identified file is validated to ensure it contains properly formatted JSON data. The tool can handle various JSON structures including single test results, arrays of test results, and nested dictionaries.

3. **Result Extraction**: Test results are extracted from the validated files. The tool can extract results from different structures and normalize them into a consistent format.

4. **Schema Validation**: Each extracted test result is validated against the expected schema. In strict mode, required fields must be present and have the correct types. In non-strict mode, the tool is more lenient but still checks field types where possible.

5. **Deduplication**: The tool creates a hash for each test result based on its content to identify duplicates. Only unique test results are migrated to the database.

6. **Database Storage**: Valid, unique test results are stored in the DuckDB database using the TestResultsDBHandler from the test_ipfs_accelerate.py module.

7. **Archiving**: If requested, the original JSON files are copied to an archive directory after successful migration.

8. **Deletion**: If requested and archiving was successful, the original JSON files are deleted.

9. **Reporting**: The tool generates a detailed migration report summarizing the process and results.

## Migration Schema

The migration tool validates test results against the following schema:

### Core Fields
- **test_name**: Name of the test (string)
- **status**: Test status (string: "success", "failure", "error", "skipped")
- **timestamp**: Test execution timestamp (string, integer, or float)
- **execution_time**: Test execution time in seconds (float or integer)

### IPFS-Specific Fields
- **cid**: Content identifier for IPFS operations (string)
- **add_time**: Time taken for add operation (float or integer)
- **get_time**: Time taken for get operation (float or integer)
- **file_size**: Size of file in bytes (integer or float)
- **checkpoint_loading_time**: Time taken to load checkpoint (float or integer)
- **dispatch_time**: Time taken for dispatching (float or integer)

### Performance Metric Fields
- **throughput**: Operations per second (float or integer)
- **latency**: Operation latency in seconds (float or integer)
- **memory_usage**: Memory usage in MB (float or integer)
- **batch_size**: Batch size used for testing (integer)

### Container Operation Fields
- **container_name**: Name of the container (string)
- **image**: Docker image used (string)
- **start_time**: Container start time (float or integer)
- **stop_time**: Container stop time (float or integer)
- **operation**: Operation type (string: "start", "stop")

### Configuration Test Fields
- **config_section**: Configuration section (string)
- **config_key**: Configuration key (string)
- **expected_value**: Expected configuration value (string, integer, float, or boolean)
- **actual_value**: Actual configuration value (string, integer, float, or boolean)

## Report Format

The migration tool generates a markdown report that includes:

- **Summary statistics**: Number of files processed, results found, results migrated, etc.
- **Migration success rate**: Percentage of results successfully migrated
- **Error list**: Details of any errors encountered during migration
- **Processed files**: List of files that were processed

The report can be written to a file specified with the `--report-file` option.

## Environment Variables

The tool uses the following environment variables:

- **BENCHMARK_DB_PATH**: Path to the DuckDB database (default: ./benchmark_db.duckdb)

## Database Schema

The tool stores test results in the database using the TestResultsDBHandler from the test_ipfs_accelerate.py module. The database schema includes tables for test results, hardware platforms, models, and performance metrics.

## Migration Logs

The tool logs all operations to both the console and a log file (`ipfs_migration.log`). The log includes detailed information about the migration process, including errors and warnings.

## Future Enhancements

Planned enhancements for the migration tool include:

1. **Incremental Migration**: Support for incremental migration of new test results
2. **Backup and Restore**: Built-in database backup and restore functionality
3. **Parallel Processing**: Parallel processing of large test result files
4. **Extended Schema Support**: Support for additional test result schemas
5. **Interactive Mode**: Interactive command-line interface for migration management

## Troubleshooting

If you encounter issues with the migration tool, check the following:

- **Missing Dependencies**: Ensure DuckDB and pandas are installed
- **Database Access**: Ensure the database file is accessible and not locked by another process
- **File Permissions**: Ensure the tool has permission to read, write, and delete files as needed
- **Log File**: Check the `ipfs_migration.log` file for detailed error information

## Contributing

Contributions to the migration tool are welcome. Please follow the standard development process:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

The IPFS Accelerate Test Results Migration Tool is licensed under the same license as the IPFS Accelerate Python package.