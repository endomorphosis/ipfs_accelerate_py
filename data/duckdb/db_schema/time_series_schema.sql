-- Time-Series Performance Tracking Schema
-- Version: 1.0
-- Date: March 15, 2025

-- This schema extension adds time-series tracking capabilities to the benchmark database

-- 1. Add versioning to performance_results table
ALTER TABLE IF EXISTS performance_results
ADD COLUMN IF NOT EXISTS version_tag VARCHAR,
ADD COLUMN IF NOT EXISTS git_commit_hash VARCHAR,
ADD COLUMN IF NOT EXISTS environment_hash VARCHAR,
ADD COLUMN IF NOT EXISTS run_group_id VARCHAR;

-- 2. Create performance_baselines table to track baseline performance
CREATE TABLE IF NOT EXISTS performance_baselines (
    baseline_id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL,
    hardware_id INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    sequence_length INTEGER DEFAULT NULL,
    precision VARCHAR DEFAULT 'fp32',
    baseline_throughput FLOAT,
    baseline_latency FLOAT,
    baseline_memory FLOAT,
    baseline_power FLOAT,
    baseline_date TIMESTAMP,
    baseline_version_tag VARCHAR,
    baseline_git_commit VARCHAR,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id),
    UNIQUE (model_id, hardware_id, batch_size, sequence_length, precision)
);

-- 3. Create performance_regressions table to track detected regressions
CREATE TABLE IF NOT EXISTS performance_regressions (
    regression_id INTEGER PRIMARY KEY,
    performance_id INTEGER,
    baseline_id INTEGER,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    regression_type VARCHAR, -- 'throughput', 'latency', 'memory', 'power'
    severity FLOAT, -- percentage of regression
    status VARCHAR DEFAULT 'detected', -- 'detected', 'investigating', 'resolved', 'false_positive'
    issue_link VARCHAR, -- link to GitHub issue or ticket
    resolution TEXT,
    resolution_date TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (performance_id) REFERENCES performance_results(id),
    FOREIGN KEY (baseline_id) REFERENCES performance_baselines(baseline_id)
);

-- 4. Create performance_trends table to store trend analysis
CREATE TABLE IF NOT EXISTS performance_trends (
    trend_id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL,
    hardware_id INTEGER NOT NULL,
    metric_type VARCHAR NOT NULL, -- 'throughput', 'latency', 'memory', 'power'
    trend_start_date TIMESTAMP,
    trend_end_date TIMESTAMP,
    trend_direction VARCHAR, -- 'improving', 'degrading', 'stable', 'volatile'
    trend_magnitude FLOAT, -- percentage change over period
    trend_confidence FLOAT, -- confidence score for trend (0-1)
    trend_data JSON, -- Store detailed trend data points
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
);

-- 5. Create regression_notifications table to track notification status
CREATE TABLE IF NOT EXISTS regression_notifications (
    notification_id INTEGER PRIMARY KEY,
    regression_id INTEGER,
    notification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notification_type VARCHAR, -- 'email', 'slack', 'github_issue', 'webhook'
    notification_target TEXT, -- recipient or endpoint
    notification_status VARCHAR, -- 'queued', 'sent', 'failed'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    last_retry_date TIMESTAMP,
    response_data TEXT,
    FOREIGN KEY (regression_id) REFERENCES performance_regressions(regression_id)
);

-- 6. Create performance_comparisons view for easier analysis
CREATE VIEW IF NOT EXISTS performance_comparisons AS
SELECT 
    p.id as performance_id,
    m.model_name,
    m.model_type,
    h.hardware_type,
    p.batch_size,
    p.precision,
    p.throughput_items_per_second,
    p.latency_ms,
    p.memory_mb,
    p.power_watts,
    p.timestamp as test_date,
    p.version_tag,
    b.baseline_throughput,
    b.baseline_latency,
    b.baseline_memory,
    b.baseline_power,
    (p.throughput_items_per_second / NULLIF(b.baseline_throughput, 0) - 1) * 100 as throughput_change_pct,
    (p.latency_ms / NULLIF(b.baseline_latency, 0) - 1) * 100 as latency_change_pct,
    (p.memory_mb / NULLIF(b.baseline_memory, 0) - 1) * 100 as memory_change_pct,
    (p.power_watts / NULLIF(b.baseline_power, 0) - 1) * 100 as power_change_pct
FROM 
    performance_results p
JOIN 
    models m ON p.model_id = m.model_id
JOIN 
    hardware_platforms h ON p.hardware_id = h.hardware_id
LEFT JOIN 
    performance_baselines b ON p.model_id = b.model_id 
        AND p.hardware_id = b.hardware_id
        AND p.batch_size = b.batch_size
        AND (p.sequence_length = b.sequence_length OR (p.sequence_length IS NULL AND b.sequence_length IS NULL))
        AND p.precision = b.precision;

-- 7. Create performance_metrics_history view for time-series analysis
CREATE VIEW IF NOT EXISTS performance_metrics_history AS
SELECT 
    m.model_name,
    m.model_type,
    h.hardware_type,
    p.batch_size,
    p.precision,
    p.throughput_items_per_second,
    p.latency_ms,
    p.memory_mb,
    p.power_watts,
    p.timestamp as test_date,
    p.version_tag,
    p.git_commit_hash,
    p.environment_hash,
    p.run_group_id
FROM 
    performance_results p
JOIN 
    models m ON p.model_id = m.model_id
JOIN 
    hardware_platforms h ON p.hardware_id = h.hardware_id
ORDER BY 
    m.model_name, h.hardware_type, p.batch_size, p.precision, p.timestamp;

-- 8. Create indexes for faster time-series queries
CREATE INDEX IF NOT EXISTS idx_perf_results_timestamp ON performance_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_perf_results_version ON performance_results(version_tag);
CREATE INDEX IF NOT EXISTS idx_perf_results_model_hw_time ON performance_results(model_id, hardware_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_perf_baselines_model_hw ON performance_baselines(model_id, hardware_id);
CREATE INDEX IF NOT EXISTS idx_perf_regressions_date ON performance_regressions(detection_date);
CREATE INDEX IF NOT EXISTS idx_perf_trends_model_hw ON performance_trends(model_id, hardware_id);
CREATE INDEX IF NOT EXISTS idx_perf_trends_metric ON performance_trends(metric_type);
CREATE INDEX IF NOT EXISTS idx_perf_results_run_group ON performance_results(run_group_id);

-- 9. Create functions for common time-series operations

-- Function to set a new baseline from recent results
CREATE OR REPLACE FUNCTION set_performance_baseline(
    p_model_id INTEGER,
    p_hardware_id INTEGER,
    p_batch_size INTEGER,
    p_sequence_length INTEGER,
    p_precision VARCHAR,
    p_days_lookback INTEGER DEFAULT 7,
    p_min_samples INTEGER DEFAULT 3
) RETURNS INTEGER AS $$
DECLARE
    v_baseline_id INTEGER;
    v_count INTEGER;
    v_throughput FLOAT;
    v_latency FLOAT;
    v_memory FLOAT;
    v_power FLOAT;
    v_date TIMESTAMP;
    v_version_tag VARCHAR;
    v_git_commit VARCHAR;
BEGIN
    -- Check if we have enough samples
    SELECT COUNT(*) INTO v_count
    FROM performance_results
    WHERE model_id = p_model_id
      AND hardware_id = p_hardware_id
      AND batch_size = p_batch_size
      AND (sequence_length = p_sequence_length OR (sequence_length IS NULL AND p_sequence_length IS NULL))
      AND precision = p_precision
      AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_lookback;
      
    IF v_count < p_min_samples THEN
        RAISE EXCEPTION 'Not enough samples (%) for baseline, minimum required: %', v_count, p_min_samples;
    END IF;
    
    -- Calculate baseline metrics
    SELECT 
        AVG(throughput_items_per_second),
        AVG(latency_ms),
        AVG(memory_mb),
        AVG(power_watts),
        MAX(timestamp),
        MAX(version_tag),
        MAX(git_commit_hash)
    INTO 
        v_throughput,
        v_latency,
        v_memory,
        v_power,
        v_date,
        v_version_tag,
        v_git_commit
    FROM performance_results
    WHERE model_id = p_model_id
      AND hardware_id = p_hardware_id
      AND batch_size = p_batch_size
      AND (sequence_length = p_sequence_length OR (sequence_length IS NULL AND p_sequence_length IS NULL))
      AND precision = p_precision
      AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_lookback;
    
    -- Insert or update baseline
    INSERT INTO performance_baselines (
        model_id, hardware_id, batch_size, sequence_length, precision,
        baseline_throughput, baseline_latency, baseline_memory, baseline_power,
        baseline_date, baseline_version_tag, baseline_git_commit,
        notes, created_at
    ) VALUES (
        p_model_id, p_hardware_id, p_batch_size, p_sequence_length, p_precision,
        v_throughput, v_latency, v_memory, v_power,
        v_date, v_version_tag, v_git_commit,
        'Automatically generated baseline from ' || v_count || ' samples',
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (model_id, hardware_id, batch_size, sequence_length, precision)
    DO UPDATE SET
        baseline_throughput = EXCLUDED.baseline_throughput,
        baseline_latency = EXCLUDED.baseline_latency,
        baseline_memory = EXCLUDED.baseline_memory,
        baseline_power = EXCLUDED.baseline_power,
        baseline_date = EXCLUDED.baseline_date,
        baseline_version_tag = EXCLUDED.baseline_version_tag,
        baseline_git_commit = EXCLUDED.baseline_git_commit,
        notes = EXCLUDED.notes,
        created_at = CURRENT_TIMESTAMP
    RETURNING baseline_id INTO v_baseline_id;
    
    RETURN v_baseline_id;
END;
$$ LANGUAGE plpgsql;

-- Function to detect regressions based on thresholds
CREATE OR REPLACE FUNCTION detect_performance_regressions(
    p_model_id INTEGER DEFAULT NULL,
    p_hardware_id INTEGER DEFAULT NULL,
    p_days_lookback INTEGER DEFAULT 1,
    p_throughput_threshold FLOAT DEFAULT -5.0,  -- percentage degradation threshold (negative)
    p_latency_threshold FLOAT DEFAULT 5.0,      -- percentage degradation threshold (positive)
    p_memory_threshold FLOAT DEFAULT 5.0,       -- percentage degradation threshold (positive)
    p_power_threshold FLOAT DEFAULT 5.0         -- percentage degradation threshold (positive)
) RETURNS TABLE (
    regression_id INTEGER,
    model_name VARCHAR,
    hardware_type VARCHAR,
    batch_size INTEGER,
    precision VARCHAR,
    regression_type VARCHAR,
    severity FLOAT,
    test_date TIMESTAMP,
    baseline_date TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH recent_results AS (
        SELECT 
            pr.id,
            pr.model_id,
            m.model_name,
            pr.hardware_id,
            h.hardware_type,
            pr.batch_size,
            pr.sequence_length,
            pr.precision,
            pr.throughput_items_per_second,
            pr.latency_ms,
            pr.memory_mb,
            pr.power_watts,
            pr.timestamp,
            pb.baseline_id,
            pb.baseline_throughput,
            pb.baseline_latency,
            pb.baseline_memory,
            pb.baseline_power,
            pb.baseline_date
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms h ON pr.hardware_id = h.hardware_id
        LEFT JOIN 
            performance_baselines pb ON pr.model_id = pb.model_id 
                AND pr.hardware_id = pb.hardware_id
                AND pr.batch_size = pb.batch_size
                AND (pr.sequence_length = pb.sequence_length OR (pr.sequence_length IS NULL AND pb.sequence_length IS NULL))
                AND pr.precision = pb.precision
        WHERE 
            pr.timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_lookback
            AND (pr.model_id = p_model_id OR p_model_id IS NULL)
            AND (pr.hardware_id = p_hardware_id OR p_hardware_id IS NULL)
            AND pb.baseline_id IS NOT NULL
    )
    
    -- Detect throughput regressions (lower is worse)
    SELECT
        nextval('performance_regressions_regression_id_seq'::regclass),
        r.model_name,
        r.hardware_type,
        r.batch_size,
        r.precision,
        'throughput'::VARCHAR as regression_type,
        ((r.throughput_items_per_second / NULLIF(r.baseline_throughput, 0)) - 1.0) * 100 as severity,
        r.timestamp,
        r.baseline_date
    FROM 
        recent_results r
    WHERE 
        ((r.throughput_items_per_second / NULLIF(r.baseline_throughput, 0)) - 1.0) * 100 <= p_throughput_threshold
        AND r.baseline_throughput > 0
    
    UNION ALL
    
    -- Detect latency regressions (higher is worse)
    SELECT
        nextval('performance_regressions_regression_id_seq'::regclass),
        r.model_name,
        r.hardware_type,
        r.batch_size,
        r.precision,
        'latency'::VARCHAR as regression_type,
        ((r.latency_ms / NULLIF(r.baseline_latency, 0)) - 1.0) * 100 as severity,
        r.timestamp,
        r.baseline_date
    FROM 
        recent_results r
    WHERE 
        ((r.latency_ms / NULLIF(r.baseline_latency, 0)) - 1.0) * 100 >= p_latency_threshold
        AND r.baseline_latency > 0
    
    UNION ALL
    
    -- Detect memory regressions (higher is worse)
    SELECT
        nextval('performance_regressions_regression_id_seq'::regclass),
        r.model_name,
        r.hardware_type,
        r.batch_size,
        r.precision,
        'memory'::VARCHAR as regression_type,
        ((r.memory_mb / NULLIF(r.baseline_memory, 0)) - 1.0) * 100 as severity,
        r.timestamp,
        r.baseline_date
    FROM 
        recent_results r
    WHERE 
        ((r.memory_mb / NULLIF(r.baseline_memory, 0)) - 1.0) * 100 >= p_memory_threshold
        AND r.baseline_memory > 0
    
    UNION ALL
    
    -- Detect power regressions (higher is worse)
    SELECT
        nextval('performance_regressions_regression_id_seq'::regclass),
        r.model_name,
        r.hardware_type,
        r.batch_size,
        r.precision,
        'power'::VARCHAR as regression_type,
        ((r.power_watts / NULLIF(r.baseline_power, 0)) - 1.0) * 100 as severity,
        r.timestamp,
        r.baseline_date
    FROM 
        recent_results r
    WHERE 
        ((r.power_watts / NULLIF(r.baseline_power, 0)) - 1.0) * 100 >= p_power_threshold
        AND r.baseline_power > 0;
END;
$$ LANGUAGE plpgsql;