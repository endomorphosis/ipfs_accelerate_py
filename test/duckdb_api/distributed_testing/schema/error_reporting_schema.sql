-- Error Reporting Schema for the Distributed Testing Framework
-- This schema defines the tables for storing worker error reports and related data.

-- Worker Error Reports Table
CREATE TABLE IF NOT EXISTS worker_error_reports (
    id INTEGER PRIMARY KEY,
    worker_id VARCHAR NOT NULL,
    error_type VARCHAR NOT NULL,
    error_category VARCHAR NOT NULL, 
    message TEXT NOT NULL,
    traceback TEXT,
    timestamp TIMESTAMP NOT NULL,
    task_id VARCHAR,
    is_recurring BOOLEAN DEFAULT FALSE,
    is_critical BOOLEAN DEFAULT FALSE,
    system_context JSON,
    hardware_context JSON,
    error_frequency JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Error Categories Summary Table
CREATE TABLE IF NOT EXISTS error_categories_summary (
    id INTEGER PRIMARY KEY,
    category VARCHAR NOT NULL,
    count INTEGER NOT NULL,
    last_occurrence TIMESTAMP NOT NULL,
    time_range_hours INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Error Patterns Table
CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY,
    pattern VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    occurrences INTEGER NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Worker Error Statistics Table
CREATE TABLE IF NOT EXISTS worker_error_statistics (
    id INTEGER PRIMARY KEY,
    worker_id VARCHAR NOT NULL,
    error_count INTEGER NOT NULL,
    critical_error_count INTEGER NOT NULL,
    most_common_error VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    last_error_time TIMESTAMP NOT NULL,
    time_range_hours INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hardware Error Statistics Table
CREATE TABLE IF NOT EXISTS hardware_error_statistics (
    id INTEGER PRIMARY KEY,
    hardware_type VARCHAR NOT NULL,
    error_count INTEGER NOT NULL,
    error_rate FLOAT NOT NULL,
    most_common_error VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    overheating_count INTEGER DEFAULT 0,
    memory_pressure_count INTEGER DEFAULT 0,
    throttling_count INTEGER DEFAULT 0,
    time_range_hours INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_worker_error_timestamp ON worker_error_reports (timestamp);
CREATE INDEX IF NOT EXISTS idx_worker_error_worker_id ON worker_error_reports (worker_id);
CREATE INDEX IF NOT EXISTS idx_worker_error_category ON worker_error_reports (error_category);
CREATE INDEX IF NOT EXISTS idx_worker_error_task_id ON worker_error_reports (task_id);
CREATE INDEX IF NOT EXISTS idx_error_pattern_occurrences ON error_patterns (occurrences DESC);
CREATE INDEX IF NOT EXISTS idx_worker_error_stats_worker_id ON worker_error_statistics (worker_id);
CREATE INDEX IF NOT EXISTS idx_hardware_error_stats_type ON hardware_error_statistics (hardware_type);

-- Create a view for recurring errors
CREATE VIEW IF NOT EXISTS recurring_errors AS
SELECT *
FROM worker_error_reports
WHERE is_recurring = TRUE
ORDER BY timestamp DESC;

-- Create a view for critical errors
CREATE VIEW IF NOT EXISTS critical_errors AS
SELECT *
FROM worker_error_reports
WHERE is_critical = TRUE
ORDER BY timestamp DESC;

-- Create a view for hardware-related errors
CREATE VIEW IF NOT EXISTS hardware_errors AS
SELECT *
FROM worker_error_reports
WHERE error_category IN ('HARDWARE_AVAILABILITY_ERROR', 'HARDWARE_CAPABILITY_ERROR', 'HARDWARE_PERFORMANCE_ERROR')
ORDER BY timestamp DESC;

-- Create a view for resource-related errors
CREATE VIEW IF NOT EXISTS resource_errors AS
SELECT *
FROM worker_error_reports
WHERE error_category IN ('RESOURCE_ALLOCATION_ERROR', 'RESOURCE_CLEANUP_ERROR')
ORDER BY timestamp DESC;

-- Create a view for network-related errors
CREATE VIEW IF NOT EXISTS network_errors AS
SELECT *
FROM worker_error_reports
WHERE error_category IN ('NETWORK_CONNECTION_ERROR', 'NETWORK_TIMEOUT_ERROR')
ORDER BY timestamp DESC;

-- Create a view for worker-related errors
CREATE VIEW IF NOT EXISTS worker_errors AS
SELECT *
FROM worker_error_reports
WHERE error_category IN ('WORKER_DISCONNECTION_ERROR', 'WORKER_CRASH_ERROR', 'WORKER_OVERLOAD_ERROR')
ORDER BY timestamp DESC;

-- Create a view for test-related errors
CREATE VIEW IF NOT EXISTS test_errors AS
SELECT *
FROM worker_error_reports
WHERE error_category IN ('TEST_EXECUTION_ERROR', 'TEST_DEPENDENCY_ERROR', 'TEST_CONFIGURATION_ERROR')
ORDER BY timestamp DESC;

-- Create a view for error summary statistics
CREATE VIEW IF NOT EXISTS error_summary_statistics AS
SELECT 
    COUNT(*) AS total_errors,
    SUM(CASE WHEN is_recurring THEN 1 ELSE 0 END) AS recurring_errors,
    SUM(CASE WHEN is_critical THEN 1 ELSE 0 END) AS critical_errors,
    SUM(CASE WHEN error_category IN ('HARDWARE_AVAILABILITY_ERROR', 'HARDWARE_CAPABILITY_ERROR', 'HARDWARE_PERFORMANCE_ERROR') AND is_critical THEN 1 ELSE 0 END) AS critical_hardware_errors,
    SUM(CASE WHEN error_category IN ('RESOURCE_ALLOCATION_ERROR', 'RESOURCE_CLEANUP_ERROR') THEN 1 ELSE 0 END) AS resource_errors,
    SUM(CASE WHEN error_category IN ('NETWORK_CONNECTION_ERROR', 'NETWORK_TIMEOUT_ERROR') THEN 1 ELSE 0 END) AS network_errors,
    SUM(CASE WHEN error_category IN ('WORKER_DISCONNECTION_ERROR', 'WORKER_CRASH_ERROR', 'WORKER_OVERLOAD_ERROR') THEN 1 ELSE 0 END) AS worker_errors,
    SUM(CASE WHEN error_category IN ('TEST_EXECUTION_ERROR', 'TEST_DEPENDENCY_ERROR', 'TEST_CONFIGURATION_ERROR') THEN 1 ELSE 0 END) AS test_errors,
    MIN(timestamp) AS start_time,
    MAX(timestamp) AS end_time
FROM worker_error_reports
WHERE timestamp >= (CURRENT_TIMESTAMP - INTERVAL '24 HOURS');