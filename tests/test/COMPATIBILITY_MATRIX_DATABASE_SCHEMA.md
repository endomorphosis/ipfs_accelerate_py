# Compatibility Matrix Database Schema

**Date: March 6, 2025**  
**Author: Claude**  
**Status: Implemented**

## Overview

This document outlines the database schema for the compatibility matrix, describing how cross-platform hardware compatibility data is stored in the DuckDB database as part of the IPFS Accelerate Python Framework.

## Schema

The compatibility matrix uses the following tables:

### cross_platform_compatibility

This is the primary table for compatibility data:

```sql
CREATE TABLE IF NOT EXISTS cross_platform_compatibility (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    model_type VARCHAR,
    model_size VARCHAR,
    cuda_support BOOLEAN,
    rocm_support BOOLEAN,
    mps_support BOOLEAN,
    openvino_support BOOLEAN,
    qualcomm_support BOOLEAN,
    webnn_support BOOLEAN,
    webgpu_support BOOLEAN,
    cpu_support BOOLEAN,
    recommended_platform VARCHAR, 
    compatibility_score FLOAT,
    last_updated TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
)
```

### compatibility_details

Additional details about specific compatibility:

```sql
CREATE TABLE IF NOT EXISTS compatibility_details (
    id INTEGER PRIMARY KEY,
    compatibility_id INTEGER,
    hardware_type VARCHAR,
    compatibility_level VARCHAR, -- 'high', 'medium', 'low', 'unsupported'
    limitations TEXT,
    workarounds TEXT,
    performance_notes TEXT,
    validation_status VARCHAR, -- 'validated', 'partially_validated', 'unvalidated'
    validator_id INTEGER,
    validation_date TIMESTAMP,
    FOREIGN KEY (compatibility_id) REFERENCES cross_platform_compatibility(id),
    FOREIGN KEY (validator_id) REFERENCES validators(validator_id)
)
```

### compatibility_history

Track changes to compatibility over time:

```sql
CREATE TABLE IF NOT EXISTS compatibility_history (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    hardware_type VARCHAR,
    previous_status VARCHAR,
    new_status VARCHAR,
    change_reason TEXT,
    change_date TIMESTAMP,
    change_version VARCHAR,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
)
```

### hardware_recommendations

Specific recommendations based on model and task:

```sql
CREATE TABLE IF NOT EXISTS hardware_recommendations (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    task_type VARCHAR,
    batch_size INTEGER,
    recommended_hardware VARCHAR,
    fallback_hardware VARCHAR,
    recommendation_reason TEXT,
    performance_ratio FLOAT, -- Ratio compared to best hardware
    memory_requirements_mb INTEGER,
    min_compute_units INTEGER,
    generated_at TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
)
```

### validators

People or processes that validate compatibility:

```sql
CREATE TABLE IF NOT EXISTS validators (
    validator_id INTEGER PRIMARY KEY,
    name VARCHAR,
    type VARCHAR, -- 'human', 'automated_test', 'ci_pipeline'
    email VARCHAR,
    organization VARCHAR,
    verification_level INTEGER, -- 1-3 indicating validation thoroughness
    created_at TIMESTAMP
)
```

## Query Examples

### Generate Basic Compatibility Matrix

```sql
SELECT 
    m.model_name,
    m.model_family,
    cpc.cpu_support,
    cpc.cuda_support,
    cpc.rocm_support,
    cpc.mps_support,
    cpc.openvino_support,
    cpc.qualcomm_support,
    cpc.webnn_support, 
    cpc.webgpu_support,
    cpc.recommended_platform
FROM models m
JOIN cross_platform_compatibility cpc ON m.model_id = cpc.model_id
ORDER BY m.model_family, m.model_name
```

### Filter by Model Type

```sql
SELECT 
    m.model_name,
    cpc.cpu_support,
    cpc.cuda_support,
    cpc.rocm_support,
    cpc.qualcomm_support,
    cpc.webgpu_support
FROM models m
JOIN cross_platform_compatibility cpc ON m.model_id = cpc.model_id
WHERE m.model_type = 'text_embedding'
ORDER BY m.model_name
```

### Get Detailed Compatibility Information

```sql
SELECT 
    m.model_name,
    cd.hardware_type,
    cd.compatibility_level,
    cd.limitations,
    cd.workarounds,
    cd.performance_notes,
    cd.validation_status,
    v.name as validator
FROM models m
JOIN cross_platform_compatibility cpc ON m.model_id = cpc.model_id
JOIN compatibility_details cd ON cpc.id = cd.compatibility_id
LEFT JOIN validators v ON cd.validator_id = v.validator_id
WHERE m.model_name = 'bert-base-uncased'
```

### Get Hardware Recommendations for Specific Task

```sql
SELECT 
    m.model_name,
    hr.task_type,
    hr.batch_size,
    hr.recommended_hardware,
    hr.fallback_hardware,
    hr.recommendation_reason,
    hr.performance_ratio,
    hr.memory_requirements_mb
FROM models m
JOIN hardware_recommendations hr ON m.model_id = hr.model_id
WHERE m.model_name LIKE '%t5%' AND hr.task_type = 'text_generation'
```

### Track Compatibility Changes Over Time

```sql
SELECT 
    m.model_name,
    ch.hardware_type,
    ch.previous_status,
    ch.new_status,
    ch.change_reason,
    ch.change_date,
    ch.change_version
FROM models m
JOIN compatibility_history ch ON m.model_id = ch.model_id
WHERE m.model_name = 'llama-7b'
ORDER BY ch.change_date DESC
```

## Integration with Visualization Tools

The database schema is designed to seamlessly integrate with visualization tools like Plotly or D3.js. The queries can be used to generate:

1. Interactive compatibility matrices with filtering
2. Hardware recommendation dashboards
3. Timeline visualizations of compatibility changes
4. Performance comparisons across hardware platforms

## Related Files

- **generate_enhanced_compatibility_matrix.py**: Script for generating the compatibility matrix
- **compatibility_matrix_template.html**: HTML template for visualizing the matrix
- **test_ipfs_accelerate.py**: Main test script that populates compatibility data

## Best Practices

1. **Regular Updates**: Update the compatibility matrix after new hardware tests
2. **Validation Process**: Follow a structured validation process for compatibility claims
3. **Performance Notes**: Include detailed performance notes for each hardware platform
4. **Version Tracking**: Track changes to compatibility through the history table
5. **CI/CD Integration**: Automatically update the matrix as part of CI/CD pipelines

## Future Enhancements

1. **Predictive Compatibility**: Use ML to predict compatibility for untested configurations
2. **Extended Metrics**: Add more detailed performance metrics beyond basic compatibility
3. **User Feedback**: Add system for collecting and incorporating user feedback
4. **External API**: Provide API for accessing compatibility data programmatically
5. **Mobile App**: Create mobile app for querying compatibility on the go