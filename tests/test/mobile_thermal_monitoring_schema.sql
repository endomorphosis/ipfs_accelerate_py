-- Mobile and Edge Device Thermal Monitoring Schema
-- April 2025

-- Thermal Events Table
-- Stores thermal events that occur during monitoring
CREATE TABLE IF NOT EXISTS thermal_events (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,  -- android, ios, etc.
    device_model VARCHAR,  -- Optional device model
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR NOT NULL,  -- NORMAL, WARNING, THROTTLING, CRITICAL, EMERGENCY
    zone_name VARCHAR NOT NULL,  -- CPU, GPU, battery, etc.
    temperature FLOAT NOT NULL,
    impact_score FLOAT,  -- 0.0-1.0 impact score
    actions_taken JSON  -- Actions taken in response to the event
);

-- Thermal Zones Table
-- Stores information about thermal zones
CREATE TABLE IF NOT EXISTS thermal_zones (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    device_model VARCHAR,
    zone_name VARCHAR NOT NULL,  -- CPU, GPU, battery, etc.
    sensor_type VARCHAR NOT NULL,
    warning_temp FLOAT NOT NULL,
    critical_temp FLOAT NOT NULL,
    path VARCHAR,  -- Path to the thermal zone (for real devices)
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Temperature Readings Table
-- Stores temperature readings from thermal zones
CREATE TABLE IF NOT EXISTS temperature_readings (
    id INTEGER PRIMARY KEY,
    zone_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    temperature FLOAT NOT NULL,
    status VARCHAR NOT NULL,  -- NORMAL, WARNING, THROTTLING, CRITICAL, EMERGENCY
    FOREIGN KEY (zone_id) REFERENCES thermal_zones(id)
);

-- Throttling Events Table
-- Stores throttling events that occur during monitoring
CREATE TABLE IF NOT EXISTS throttling_events (
    id INTEGER PRIMARY KEY,
    thermal_event_id INTEGER,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    throttling_level INTEGER NOT NULL,  -- 0-5 throttling level
    performance_impact FLOAT NOT NULL,  -- 0.0-1.0 performance impact
    duration_seconds FLOAT,  -- Duration of throttling
    device_type VARCHAR NOT NULL,
    device_model VARCHAR,
    FOREIGN KEY (thermal_event_id) REFERENCES thermal_events(id)
);

-- Thermal Profiles Table
-- Stores thermal profiles for different device types
CREATE TABLE IF NOT EXISTS thermal_profiles (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    description VARCHAR,
    device_type VARCHAR NOT NULL,
    profile_type VARCHAR NOT NULL,  -- default, battery_saving, performance, etc.
    thermal_zones JSON NOT NULL,  -- Thermal zone configurations
    cooling_policy JSON NOT NULL,  -- Cooling policy configuration
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Thermal Reports Table
-- Stores comprehensive thermal reports
CREATE TABLE IF NOT EXISTS thermal_reports (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    device_model VARCHAR,
    report_type VARCHAR NOT NULL,  -- regular, simulation, etc.
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration_seconds FLOAT,  -- Duration of monitoring or simulation
    thermal_status VARCHAR NOT NULL,  -- NORMAL, WARNING, THROTTLING, CRITICAL, EMERGENCY
    performance_impact FLOAT NOT NULL,  -- 0.0-1.0 performance impact
    temperatures JSON NOT NULL,  -- Current temperatures for all zones
    max_temperatures JSON NOT NULL,  -- Maximum temperatures for all zones
    avg_temperatures JSON,  -- Average temperatures for all zones
    trends JSON,  -- Temperature trends for all zones
    forecasts JSON,  -- Temperature forecasts for all zones
    events_count INTEGER,  -- Number of thermal events
    recommendations JSON,  -- Thermal management recommendations
    full_report JSON NOT NULL  -- Complete report
);

-- Mobile Device Metrics Table
-- Stores mobile-specific metrics
CREATE TABLE IF NOT EXISTS mobile_device_metrics (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    device_model VARCHAR,
    processor_type VARCHAR,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    battery_level FLOAT,  -- 0.0-1.0 battery level
    battery_temperature FLOAT,  -- Battery temperature in Celsius
    cpu_utilization FLOAT,  -- 0.0-1.0 CPU utilization
    gpu_utilization FLOAT,  -- 0.0-1.0 GPU utilization
    memory_utilization FLOAT,  -- 0.0-1.0 memory utilization
    network_utilization FLOAT,  -- 0.0-1.0 network utilization
    battery_discharge_rate FLOAT,  -- % per hour
    thermal_state JSON  -- Thermal state information
);

-- Thermal Simulation Configurations Table
-- Stores configurations for thermal simulations
CREATE TABLE IF NOT EXISTS thermal_simulation_configs (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    workload_pattern VARCHAR NOT NULL,  -- steady, increasing, pulsed, etc.
    duration_seconds INTEGER NOT NULL,
    cpu_workload_start FLOAT,  -- 0.0-1.0 starting CPU workload
    gpu_workload_start FLOAT,  -- 0.0-1.0 starting GPU workload
    cpu_workload_end FLOAT,  -- 0.0-1.0 ending CPU workload
    gpu_workload_end FLOAT,  -- 0.0-1.0 ending GPU workload
    thermal_profile_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thermal_profile_id) REFERENCES thermal_profiles(id)
);

-- Thermal Simulation Results Table
-- Stores results from thermal simulations
CREATE TABLE IF NOT EXISTS thermal_simulation_results (
    id INTEGER PRIMARY KEY,
    config_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    report_id INTEGER NOT NULL,
    max_cpu_temp FLOAT,
    max_gpu_temp FLOAT,
    max_throttling_level INTEGER,  -- 0-5 maximum throttling level
    avg_performance_impact FLOAT,  -- 0.0-1.0 average performance impact
    events_count INTEGER,  -- Number of thermal events
    FOREIGN KEY (config_id) REFERENCES thermal_simulation_configs(id),
    FOREIGN KEY (report_id) REFERENCES thermal_reports(id)
);

-- Qualcomm QNN Specific Metrics Table
-- Stores Qualcomm QNN specific metrics
CREATE TABLE IF NOT EXISTS qnn_thermal_metrics (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    device_model VARCHAR,
    soc_type VARCHAR,  -- e.g., Snapdragon 8 Gen 3
    qnn_version VARCHAR,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    npu_temperature FLOAT,  -- NPU temperature in Celsius
    npu_utilization FLOAT,  -- 0.0-1.0 NPU utilization
    dsp_temperature FLOAT,  -- DSP temperature in Celsius
    dsp_utilization FLOAT,  -- 0.0-1.0 DSP utilization
    cpu_big_temperature FLOAT,  -- CPU big cores temperature in Celsius
    cpu_little_temperature FLOAT,  -- CPU little cores temperature in Celsius
    gpu_temperature FLOAT,  -- GPU temperature in Celsius
    thermal_state VARCHAR NOT NULL,  -- NORMAL, WARNING, THROTTLING, CRITICAL, EMERGENCY
    thermal_throttling_active BOOLEAN,
    power_consumption_mw FLOAT,
    thermal_report_id INTEGER,
    FOREIGN KEY (thermal_report_id) REFERENCES thermal_reports(id)
);

-- WebGPU Specific Metrics Table
-- Stores WebGPU specific metrics
CREATE TABLE IF NOT EXISTS webgpu_thermal_metrics (
    id INTEGER PRIMARY KEY,
    device_type VARCHAR NOT NULL,
    browser_name VARCHAR NOT NULL,
    browser_version VARCHAR,
    gpu_vendor VARCHAR,
    gpu_name VARCHAR,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    gpu_temperature FLOAT,  -- GPU temperature in Celsius (if available)
    shader_compilation_time_ms FLOAT,
    pipeline_creation_time_ms FLOAT,
    compute_shader_execution_time_ms FLOAT,
    memory_usage_mb FLOAT,
    thermal_state VARCHAR NOT NULL,  -- NORMAL, WARNING, THROTTLING, CRITICAL, EMERGENCY
    thermal_throttling_active BOOLEAN,
    power_consumption_mw FLOAT,
    thermal_report_id INTEGER,
    FOREIGN KEY (thermal_report_id) REFERENCES thermal_reports(id)
);

-- Indexes for faster query performance
CREATE INDEX IF NOT EXISTS idx_thermal_events_timestamp ON thermal_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_thermal_events_event_type ON thermal_events(event_type);
CREATE INDEX IF NOT EXISTS idx_thermal_events_zone_name ON thermal_events(zone_name);
CREATE INDEX IF NOT EXISTS idx_temperature_readings_zone_id ON temperature_readings(zone_id);
CREATE INDEX IF NOT EXISTS idx_temperature_readings_timestamp ON temperature_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_throttling_events_timestamp ON throttling_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_thermal_reports_device_type ON thermal_reports(device_type);
CREATE INDEX IF NOT EXISTS idx_thermal_reports_timestamp ON thermal_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_thermal_profiles_device_type ON thermal_profiles(device_type);
CREATE INDEX IF NOT EXISTS idx_thermal_profiles_profile_type ON thermal_profiles(profile_type);
CREATE INDEX IF NOT EXISTS idx_mobile_device_metrics_device_type ON mobile_device_metrics(device_type);
CREATE INDEX IF NOT EXISTS idx_mobile_device_metrics_timestamp ON mobile_device_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_qnn_thermal_metrics_device_model ON qnn_thermal_metrics(device_model);
CREATE INDEX IF NOT EXISTS idx_qnn_thermal_metrics_timestamp ON qnn_thermal_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_webgpu_thermal_metrics_browser_name ON webgpu_thermal_metrics(browser_name);
CREATE INDEX IF NOT EXISTS idx_webgpu_thermal_metrics_timestamp ON webgpu_thermal_metrics(timestamp);

-- Views for common queries

-- View for latest thermal status by device
CREATE VIEW IF NOT EXISTS v_latest_thermal_status AS
SELECT
    device_type,
    device_model,
    MAX(timestamp) as last_updated,
    (SELECT event_type FROM thermal_events te2 
     WHERE te2.device_type = te.device_type AND te2.device_model = te.device_model 
     ORDER BY timestamp DESC LIMIT 1) as latest_status
FROM thermal_events te
GROUP BY device_type, device_model;

-- View for thermal zone statistics
CREATE VIEW IF NOT EXISTS v_thermal_zone_stats AS
SELECT
    tz.device_type,
    tz.device_model,
    tz.zone_name,
    COUNT(tr.id) as reading_count,
    AVG(tr.temperature) as avg_temperature,
    MAX(tr.temperature) as max_temperature,
    MIN(tr.temperature) as min_temperature,
    (SELECT COUNT(*) FROM temperature_readings tr2 
     WHERE tr2.zone_id = tz.id AND tr2.status != 'NORMAL') as abnormal_count
FROM thermal_zones tz
LEFT JOIN temperature_readings tr ON tr.zone_id = tz.id
GROUP BY tz.device_type, tz.device_model, tz.zone_name;

-- View for throttling statistics
CREATE VIEW IF NOT EXISTS v_throttling_stats AS
SELECT
    device_type,
    device_model,
    COUNT(id) as throttling_count,
    AVG(throttling_level) as avg_throttling_level,
    MAX(throttling_level) as max_throttling_level,
    AVG(performance_impact) as avg_performance_impact,
    SUM(duration_seconds) as total_throttling_seconds
FROM throttling_events
GROUP BY device_type, device_model;

-- View for simulation comparison
CREATE VIEW IF NOT EXISTS v_simulation_comparison AS
SELECT
    tsc.device_type,
    tsc.workload_pattern,
    COUNT(tsr.id) as simulation_count,
    AVG(tsr.max_cpu_temp) as avg_max_cpu_temp,
    AVG(tsr.max_gpu_temp) as avg_max_gpu_temp,
    AVG(tsr.max_throttling_level) as avg_max_throttling_level,
    AVG(tsr.avg_performance_impact) as avg_performance_impact,
    tp.name as profile_name
FROM thermal_simulation_configs tsc
JOIN thermal_simulation_results tsr ON tsr.config_id = tsc.id
LEFT JOIN thermal_profiles tp ON tp.id = tsc.thermal_profile_id
GROUP BY tsc.device_type, tsc.workload_pattern, tp.name;

-- Example function to calculate thermal trend
CREATE OR REPLACE FUNCTION calculate_thermal_trend(
    zone_id INTEGER,
    window_seconds INTEGER DEFAULT 60
) RETURNS TABLE (
    trend_celsius_per_minute FLOAT,
    min_temp FLOAT,
    max_temp FLOAT,
    avg_temp FLOAT,
    stable BOOLEAN
) AS $$
DECLARE
    window_start TIMESTAMP;
BEGIN
    window_start := CURRENT_TIMESTAMP - INTERVAL '1 second' * window_seconds;
    
    RETURN QUERY
    WITH window_readings AS (
        SELECT
            temperature,
            timestamp
        FROM temperature_readings
        WHERE zone_id = zone_id AND timestamp >= window_start
        ORDER BY timestamp
    ),
    stats AS (
        SELECT
            MIN(temperature) as min_temp,
            MAX(temperature) as max_temp,
            AVG(temperature) as avg_temp,
            COUNT(*) as reading_count
        FROM window_readings
    ),
    regression AS (
        SELECT
            CASE 
                WHEN COUNT(*) <= 1 THEN 0
                ELSE (COUNT(*) * SUM(EXTRACT(EPOCH FROM timestamp) * temperature) - SUM(EXTRACT(EPOCH FROM timestamp)) * SUM(temperature)) /
                     (COUNT(*) * SUM(EXTRACT(EPOCH FROM timestamp) * EXTRACT(EPOCH FROM timestamp)) - SUM(EXTRACT(EPOCH FROM timestamp)) * SUM(EXTRACT(EPOCH FROM timestamp)))
            END * 60 as slope  -- Convert to per minute
        FROM window_readings
    )
    SELECT
        r.slope as trend_celsius_per_minute,
        s.min_temp,
        s.max_temp,
        s.avg_temp,
        (s.max_temp - s.min_temp < 3.0 AND ABS(r.slope) < 0.5) as stable
    FROM regression r, stats s;
END;
$$ LANGUAGE plpgsql;