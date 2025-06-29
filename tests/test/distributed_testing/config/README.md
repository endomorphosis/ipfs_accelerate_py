# Configuration Files for Distributed Testing Framework

This directory contains configuration templates for the Dynamic Resource Manager and Performance Trend Analyzer components of the Distributed Testing Framework.

## Configuration Files

### Dynamic Resource Manager

- **resource_manager_config.json**: General configuration for the Dynamic Resource Manager, including scaling thresholds, strategies, and intervals.
- **worker_templates.json**: Templates for different types of workers that can be provisioned by the resource manager.
- **provider_config.json**: Configuration for different cloud providers (AWS, GCP, Azure) and local worker provisioning.

### Performance Trend Analyzer

- **performance_analyzer_config.json**: Configuration for the Performance Trend Analyzer, including anomaly detection settings, trend analysis parameters, and reporting options.

## Usage

1. Copy these configuration files to your working directory.
2. Modify them according to your environment and requirements.
3. Pass the configuration file paths when starting the components:

```bash
# Start the dynamic resource manager
python -m distributed_testing.dynamic_resource_manager \
  --coordinator http://localhost:8080 \
  --config path/to/resource_manager_config.json \
  --templates path/to/worker_templates.json \
  --providers path/to/provider_config.json

# Start the performance trend analyzer
python -m distributed_testing.performance_trend_analyzer \
  --coordinator http://localhost:8080 \
  --config path/to/performance_analyzer_config.json \
  --db-path path/to/metrics.db \
  --output-dir path/to/reports
```

## Configuration Guidelines

### Resource Manager

- **min_workers/max_workers**: Set according to your baseline workload and maximum capacity.
- **polling_interval**: How often to check workload metrics (in seconds).
- **thresholds**: Tune CPU, memory, and queue thresholds to match your workload patterns.
- **cooldown periods**: Prevent rapid scaling up and down by setting appropriate cooldown periods.
- **scaling_strategy**: Choose from STATIC, STEPWISE, PREDICTIVE, ADAPTIVE, or COST_OPTIMIZED.

### Worker Templates

- Define templates for different workload types (CPU, GPU, memory-intensive).
- Include all necessary software dependencies in the worker_config section.
- Set accurate cost information for proper cost optimization.
- Estimate startup times for better predictive scaling.

### Provider Configuration

- Enable/disable specific cloud regions based on where you want to deploy.
- Set appropriate regional limits to control costs.
- Configure region-specific settings like VPCs, subnets, and security groups.
- For local workers, configure Docker or process-based execution.

### Performance Analyzer

- **polling_interval**: How often to collect and analyze metrics.
- **history_window**: How far back to retain metrics for trend analysis.
- **anomaly_detection**: Tune algorithms based on the variability of your workloads.
- **trend_analysis**: Configure based on how quickly performance patterns typically emerge.
- **metrics_to_track**: Add or remove metrics based on what's important for your use case.
- **alert_thresholds**: Set warning and critical thresholds according to your SLAs.

## Security Notes

- The provider configuration files may contain sensitive information. Ensure they're properly secured.
- DO NOT commit configuration files with real credentials to your repository.
- Consider using environment variables or a secrets manager for sensitive values.
- For cloud credentials, use IAM roles with the minimum necessary permissions.

## Customization

These templates are starting points. You should customize them based on:

1. Your specific workload characteristics
2. Available hardware resources
3. Budget constraints
4. Performance requirements
5. Cloud provider preferences

Review and adjust the configurations regularly as your usage patterns evolve.