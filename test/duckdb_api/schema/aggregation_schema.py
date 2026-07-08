#!/usr/bin/env python3
"""
Distributed Testing Framework - Result Aggregation Schema

This module defines the database schema for storing result aggregation data,
including anomalies, trends, and related information.
"""

# SQL schema for performance anomalies table
PERFORMANCE_ANOMALIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS performance_anomalies (
    anomaly_id INTEGER PRIMARY KEY,
    entity_type VARCHAR NOT NULL,  -- 'worker' or 'task_type'
    entity_id VARCHAR NOT NULL,    -- worker_id or task_type
    metric VARCHAR NOT NULL,       -- The metric name that showed anomalous behavior
    detected_at TIMESTAMP NOT NULL,  -- When the anomaly was detected
    anomaly_timestamp TIMESTAMP NOT NULL,  -- When the anomaly occurred
    value FLOAT NOT NULL,          -- The anomalous value
    baseline_mean FLOAT NOT NULL,  -- Mean of the baseline
    baseline_stdev FLOAT NOT NULL, -- Standard deviation of the baseline
    z_score FLOAT NOT NULL,        -- Z-score of the anomaly
    is_high BOOLEAN NOT NULL,      -- Whether it's a high (true) or low (false) anomaly
    
    -- Add indexes for better query performance
    INDEX idx_anomalies_entity (entity_type, entity_id),
    INDEX idx_anomalies_metric (metric),
    INDEX idx_anomalies_timestamp (anomaly_timestamp)
)
"""

# SQL schema for performance trends table
PERFORMANCE_TRENDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS performance_trends (
    trend_id INTEGER PRIMARY KEY,
    entity_type VARCHAR NOT NULL,  -- 'worker' or 'task_type'
    entity_id VARCHAR NOT NULL,    -- worker_id or task_type
    metric VARCHAR NOT NULL,       -- The metric name that showed a trend
    detected_at TIMESTAMP NOT NULL,  -- When the trend was detected
    slope FLOAT NOT NULL,          -- Slope of the trend line
    p_value FLOAT NOT NULL,        -- P-value of the trend
    r_squared FLOAT NOT NULL,      -- R-squared value
    is_significant BOOLEAN NOT NULL,  -- Whether the trend is significant
    direction VARCHAR NOT NULL,    -- 'increasing' or 'decreasing'
    forecast_values JSON,          -- JSON array of forecasted values
    
    -- Add indexes for better query performance
    INDEX idx_trends_entity (entity_type, entity_id),
    INDEX idx_trends_metric (metric),
    INDEX idx_trends_detected (detected_at)
)
"""

# SQL schema for aggregate results cache table
AGGREGATE_RESULTS_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS aggregate_results_cache (
    cache_id INTEGER PRIMARY KEY,
    cache_key VARCHAR NOT NULL UNIQUE,  -- Unique key for the cache entry
    result_type VARCHAR NOT NULL,       -- Type of results
    aggregation_level VARCHAR NOT NULL, -- Level of aggregation
    filter_params JSON,                -- JSON of filter parameters
    time_range_start TIMESTAMP,        -- Start of time range for filtering
    time_range_end TIMESTAMP,          -- End of time range for filtering
    results JSON NOT NULL,             -- JSON of aggregated results
    created_at TIMESTAMP NOT NULL,     -- When the cache entry was created
    expires_at TIMESTAMP NOT NULL,     -- When the cache entry expires
    
    -- Add index for better query performance
    INDEX idx_cache_key (cache_key),
    INDEX idx_cache_expires (expires_at)
)
"""

# Combined schema list
AGGREGATION_SCHEMA_LIST = [
    PERFORMANCE_ANOMALIES_SCHEMA,
    PERFORMANCE_TRENDS_SCHEMA,
    AGGREGATE_RESULTS_CACHE_SCHEMA
]

# Function to create all result aggregation tables
def create_aggregation_tables(conn):
    """Create all result aggregation tables in the database.
    
    Args:
        conn: DuckDB connection object
    """
    cursor = conn.cursor()
    
    for schema in AGGREGATION_SCHEMA_LIST:
        cursor.execute(schema)
        
    conn.commit()
    
    # Add any views or other schema objects
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS recent_anomalies AS
        SELECT *
        FROM performance_anomalies
        WHERE detected_at > CURRENT_TIMESTAMP - INTERVAL 7 DAY
        ORDER BY detected_at DESC
    """)
    
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS significant_trends AS
        SELECT *
        FROM performance_trends
        WHERE is_significant = TRUE
        ORDER BY detected_at DESC
    """)
    
    conn.commit()