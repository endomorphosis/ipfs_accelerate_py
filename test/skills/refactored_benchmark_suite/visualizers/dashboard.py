"""
Interactive dashboard for visualizing benchmark results.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("benchmark.visualizers.dashboard")

def generate_dashboard(benchmark_results: List, output_dir: str = "benchmark_results") -> str:
    """
    Generate an interactive dashboard for benchmark results.
    
    Args:
        benchmark_results: List of BenchmarkResults instances
        output_dir: Directory to save the dashboard
        
    Returns:
        Path to the generated dashboard
    """
    try:
        import dash
        from dash import dcc, html
        import plotly.express as px
        import plotly.graph_objs as go
        import pandas as pd
        
        # Create dashboard directory
        dashboard_dir = os.path.join(output_dir, "dashboard")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Prepare data
        data = []
        
        for results in benchmark_results:
            for result in results.results:
                row = {
                    "model": results.config.model_id,
                    "hardware": result.hardware,
                    "batch_size": result.batch_size,
                    "sequence_length": result.sequence_length
                }
                
                # Add metrics
                for metric_name, metric_value in result.metrics.items():
                    row[metric_name] = metric_value
                
                data.append(row)
        
        if not data:
            logger.warning("No data available for dashboard")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save data for dashboard
        df.to_csv(os.path.join(dashboard_dir, "benchmark_data.csv"), index=False)
        
        # Create dashboard HTML file
        dashboard_html = os.path.join(dashboard_dir, "index.html")
        
        # Create dashboard template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HuggingFace Model Benchmark Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                .chart-container {
                    margin-top: 20px;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }
                .controls {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                select {
                    padding: 8px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                }
                .table-container {
                    margin-top: 20px;
                    max-height: 400px;
                    overflow-y: auto;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>HuggingFace Model Benchmark Dashboard</h1>
                
                <div class="controls">
                    <div>
                        <label for="modelSelect">Model:</label>
                        <select id="modelSelect">
                            <option value="all">All Models</option>
                            {model_options}
                        </select>
                    </div>
                    <div>
                        <label for="hardwareSelect">Hardware:</label>
                        <select id="hardwareSelect">
                            <option value="all">All Hardware</option>
                            {hardware_options}
                        </select>
                    </div>
                    <div>
                        <label for="metricSelect">Metric:</label>
                        <select id="metricSelect">
                            <option value="latency_ms">Latency (ms)</option>
                            <option value="latency_p90_ms">Latency p90 (ms)</option>
                            <option value="latency_p95_ms">Latency p95 (ms)</option>
                            <option value="latency_p99_ms">Latency p99 (ms)</option>
                            <option value="throughput_items_per_sec">Throughput (items/sec)</option>
                            <option value="memory_usage_mb">Memory Usage (MB)</option>
                            <option value="memory_peak_mb">Peak Memory (MB)</option>
                            <option value="memory_allocated_end_mb">Allocated Memory (MB)</option>
                            <option value="memory_reserved_end_mb">Reserved Memory (MB)</option>
                            <option value="cpu_memory_end_mb">CPU Memory (MB)</option>
                            <option value="flops">FLOPs</option>
                            <option value="gflops">GFLOPs</option>
                            <option value="tflops">TFLOPs</option>
                            <option value="hardware_efficiency">Hardware Efficiency (%)</option>
                            <option value="flops_per_token">FLOPs per Token</option>
                            <option value="flops_per_parameter">FLOPs per Parameter</option>
                            <option value="power_avg_watts">Power (W)</option>
                            <option value="energy_joules">Energy (J)</option>
                            <option value="gflops_per_watt">GFLOPs/Watt</option>
                            <option value="throughput_per_watt">Throughput/Watt</option>
                        </select>
                    </div>
                    <div>
                        <label for="chartTypeSelect">Chart Type:</label>
                        <select id="chartTypeSelect">
                            <option value="bar">Bar Chart</option>
                            <option value="line">Line Chart</option>
                            <option value="scatter">Scatter Plot</option>
                        </select>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div id="mainChart"></div>
                </div>
                
                <div class="advanced-charts" style="display: flex; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 50%;" class="chart-container">
                        <h3>Latency Percentiles</h3>
                        <div id="latencyPercentileChart"></div>
                    </div>
                    
                    <div style="flex: 1; min-width: 50%;" class="chart-container">
                        <h3>Memory Breakdown</h3>
                        <div id="memoryBreakdownChart"></div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Batch Size Scaling</h3>
                    <div id="batchSizeChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Hardware Efficiency Comparison</h3>
                    <div id="hardwareEfficiencyChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Hardware Performance Comparison</h3>
                    <div id="hardwareComparisonChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Power Efficiency</h3>
                    <div id="powerEfficiencyChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Memory Bandwidth Utilization</h3>
                    <div id="bandwidthUtilizationChart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Roofline Performance Model</h3>
                    <div id="rooflineModelChart"></div>
                </div>
                
                <div class="table-container">
                    <h2>Detailed Results</h2>
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Batch Size</th>
                                <th>Seq. Length</th>
                                <th>Latency (ms)</th>
                                <th>p90 (ms)</th>
                                <th>p99 (ms)</th>
                                <th>Throughput</th>
                                <th>Memory (MB)</th>
                                <th>Peak Memory (MB)</th>
                                <th>GFLOPs</th>
                                <th>HW Efficiency</th>
                                <th>Power (W)</th>
                                <th>Energy (J)</th>
                                <th>GFLOPs/Watt</th>
                                <th>Model Type</th>
                                <th>Tensor Core</th>
                                <th>CPU</th>
                                <th>CUDA</th>
                                <th>GPU</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // Data
                const benchmarkData = {data_json};
                
                // Initialize charts
                updateCharts();
                
                // Add event listeners
                document.getElementById('modelSelect').addEventListener('change', updateCharts);
                document.getElementById('hardwareSelect').addEventListener('change', updateCharts);
                document.getElementById('metricSelect').addEventListener('change', updateCharts);
                document.getElementById('chartTypeSelect').addEventListener('change', updateCharts);
                
                // Initialize all charts on load
                updateCharts();
                
                function updateCharts() {
                    const selectedModel = document.getElementById('modelSelect').value;
                    const selectedHardware = document.getElementById('hardwareSelect').value;
                    const selectedMetric = document.getElementById('metricSelect').value;
                    const selectedChartType = document.getElementById('chartTypeSelect').value;
                    
                    // Filter data
                    let filteredData = benchmarkData;
                    if (selectedModel !== 'all') {
                        filteredData = filteredData.filter(d => d.model === selectedModel);
                    }
                    if (selectedHardware !== 'all') {
                        filteredData = filteredData.filter(d => d.hardware === selectedHardware);
                    }
                    
                    // Update main chart with selected chart type
                    updateMainChart(filteredData, selectedMetric, selectedChartType);
                    
                    // Update advanced charts
                    updateLatencyPercentileChart(filteredData);
                    updateMemoryBreakdownChart(filteredData);
                    
                    // Update batch size chart
                    updateBatchSizeChart(filteredData, selectedMetric);
                    
                    // Update hardware efficiency chart
                    updateHardwareEfficiencyChart(filteredData);
                    
                    // Update hardware comparison chart
                    updateHardwareComparisonChart(filteredData);
                    
                    // Update power efficiency chart
                    updatePowerEfficiencyChart(filteredData);
                    
                    // Update bandwidth utilization chart
                    updateBandwidthUtilizationChart(filteredData);
                    
                    // Update roofline model chart
                    updateRooflineModelChart(filteredData);
                    
                    // Update table
                    updateTable(filteredData);
                }
                
                function updateMainChart(data, metric, chartType) {
                    const metricLabels = {
                        'latency_ms': 'Latency (ms)',
                        'latency_p90_ms': 'Latency p90 (ms)',
                        'latency_p95_ms': 'Latency p95 (ms)',
                        'latency_p99_ms': 'Latency p99 (ms)',
                        'throughput_items_per_sec': 'Throughput (items/sec)',
                        'memory_usage_mb': 'Memory Usage (MB)',
                        'memory_peak_mb': 'Peak Memory (MB)',
                        'memory_allocated_end_mb': 'Allocated Memory (MB)',
                        'cpu_memory_end_mb': 'CPU Memory (MB)',
                        'flops': 'FLOPs',
                        'gflops': 'GFLOPs',
                        'power_avg_watts': 'Power (W)',
                        'energy_joules': 'Energy (J)',
                        'gflops_per_watt': 'GFLOPs/Watt',
                        'throughput_per_watt': 'Throughput/Watt'
                    };
                    
                    // Check if we have data for this metric
                    const hasMetricData = data.some(d => d[metric] !== undefined);
                    if (!hasMetricData) {
                        // Display message in the chart container
                        document.getElementById('mainChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No data available for ${metricLabels[metric] || metric}
                            </div>`;
                        return;
                    }
                    
                    const traces = [];
                    
                    // Group by model and hardware
                    const models = [...new Set(data.map(d => d.model))];
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    const batchSizes = [...new Set(data.map(d => d.batch_size))].sort((a, b) => a - b);
                    
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const modelHardwareData = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                d[metric] !== undefined
                            );
                            
                            if (modelHardwareData.length > 0) {
                                // Sort by batch size
                                modelHardwareData.sort((a, b) => a.batch_size - b.batch_size);
                                
                                // Create trace based on chart type
                                let trace;
                                
                                switch (chartType) {
                                    case 'bar':
                                        trace = {
                                            x: modelHardwareData.map(d => `BS ${d.batch_size}`),
                                            y: modelHardwareData.map(d => d[metric]),
                                            type: 'bar',
                                            name: `${model} - ${hardware.toUpperCase()}`
                                        };
                                        break;
                                    
                                    case 'scatter':
                                        trace = {
                                            x: modelHardwareData.map(d => d.batch_size),
                                            y: modelHardwareData.map(d => d[metric]),
                                            type: 'scatter',
                                            mode: 'markers',
                                            marker: {
                                                size: 10
                                            },
                                            name: `${model} - ${hardware.toUpperCase()}`
                                        };
                                        break;
                                    
                                    case 'line':
                                    default:
                                        trace = {
                                            x: modelHardwareData.map(d => d.batch_size),
                                            y: modelHardwareData.map(d => d[metric]),
                                            type: 'scatter',
                                            mode: 'lines+markers',
                                            name: `${model} - ${hardware.toUpperCase()}`
                                        };
                                        break;
                                }
                                
                                traces.push(trace);
                            }
                        }
                    }
                    
                    // Define layout for the different chart types
                    let layout;
                    
                    if (chartType === 'bar') {
                        layout = {
                            title: `${metricLabels[metric] || metric} by Batch Size`,
                            xaxis: {
                                title: 'Batch Size',
                                type: 'category'
                            },
                            yaxis: {
                                title: metricLabels[metric] || metric
                            },
                            barmode: 'group',
                            legend: {
                                orientation: 'h',
                                yanchor: 'bottom',
                                y: -0.2
                            }
                        };
                    } else {
                        layout = {
                            title: `${metricLabels[metric] || metric} by Batch Size`,
                            xaxis: {
                                title: 'Batch Size'
                            },
                            yaxis: {
                                title: metricLabels[metric] || metric
                            },
                            legend: {
                                orientation: 'h',
                                yanchor: 'bottom',
                                y: -0.2
                            }
                        };
                    }
                    
                    Plotly.newPlot('mainChart', traces, layout);
                }
                
                function updateLatencyPercentileChart(data) {
                    // Check if we have percentile data
                    const hasPercentileData = data.some(d => 
                        d.latency_p90_ms !== undefined || 
                        d.latency_p95_ms !== undefined || 
                        d.latency_p99_ms !== undefined
                    );
                    
                    if (!hasPercentileData) {
                        // Display message in the chart container
                        document.getElementById('latencyPercentileChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No percentile data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    const traces = [];
                    
                    // Group by model and hardware
                    const models = [...new Set(data.map(d => d.model))];
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const modelHardwareData = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                (d.latency_ms !== undefined)
                            );
                            
                            if (modelHardwareData.length > 0) {
                                // Sort by batch size
                                modelHardwareData.sort((a, b) => a.batch_size - b.batch_size);
                                
                                // Create mean latency trace
                                traces.push({
                                    x: modelHardwareData.map(d => d.batch_size),
                                    y: modelHardwareData.map(d => d.latency_ms),
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: `${model} - ${hardware.toUpperCase()} (Mean)`,
                                    line: {
                                        width: 3
                                    },
                                    marker: {
                                        size: 8
                                    }
                                });
                                
                                // Add p90 percentile if available
                                if (modelHardwareData.some(d => d.latency_p90_ms !== undefined)) {
                                    traces.push({
                                        x: modelHardwareData.map(d => d.batch_size),
                                        y: modelHardwareData.map(d => d.latency_p90_ms !== undefined ? d.latency_p90_ms : null),
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        name: `${model} - ${hardware.toUpperCase()} (p90)`,
                                        line: {
                                            dash: 'dot',
                                            width: 2
                                        },
                                        marker: {
                                            size: 6
                                        }
                                    });
                                }
                                
                                // Add p95 percentile if available
                                if (modelHardwareData.some(d => d.latency_p95_ms !== undefined)) {
                                    traces.push({
                                        x: modelHardwareData.map(d => d.batch_size),
                                        y: modelHardwareData.map(d => d.latency_p95_ms !== undefined ? d.latency_p95_ms : null),
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        name: `${model} - ${hardware.toUpperCase()} (p95)`,
                                        line: {
                                            dash: 'dash',
                                            width: 2
                                        },
                                        marker: {
                                            size: 6
                                        }
                                    });
                                }
                                
                                // Add p99 percentile if available
                                if (modelHardwareData.some(d => d.latency_p99_ms !== undefined)) {
                                    traces.push({
                                        x: modelHardwareData.map(d => d.batch_size),
                                        y: modelHardwareData.map(d => d.latency_p99_ms !== undefined ? d.latency_p99_ms : null),
                                        type: 'scatter',
                                        mode: 'lines+markers',
                                        name: `${model} - ${hardware.toUpperCase()} (p99)`,
                                        line: {
                                            dash: 'dashdot',
                                            width: 2
                                        },
                                        marker: {
                                            size: 6
                                        }
                                    });
                                }
                            }
                        }
                    }
                    
                    const layout = {
                        title: 'Latency Percentiles by Batch Size',
                        xaxis: {
                            title: 'Batch Size'
                        },
                        yaxis: {
                            title: 'Latency (ms)'
                        },
                        legend: {
                            orientation: 'h',
                            yanchor: 'bottom',
                            y: -0.4
                        }
                    };
                    
                    Plotly.newPlot('latencyPercentileChart', traces, layout);
                }
                
                function updateMemoryBreakdownChart(data) {
                    // Check if we have detailed memory data
                    const hasDetailedMemory = data.some(d => 
                        d.memory_peak_mb !== undefined || 
                        d.memory_allocated_end_mb !== undefined || 
                        d.memory_reserved_end_mb !== undefined || 
                        d.cpu_memory_end_mb !== undefined
                    );
                    
                    if (!hasDetailedMemory) {
                        // Display message in the chart container
                        document.getElementById('memoryBreakdownChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No detailed memory data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    // Group by model, hardware, and batch size to create breakdown chart
                    const models = [...new Set(data.map(d => d.model))];
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    
                    // For simplicity, use the first batch size for memory breakdown
                    const modelHWData = [];
                    
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const matches = data.filter(d => d.model === model && d.hardware === hardware);
                            if (matches.length > 0) {
                                // Use the first batch size result for this model/hardware
                                modelHWData.push(matches[0]);
                            }
                        }
                    }
                    
                    // Create stacked bar chart for memory breakdown
                    const traces = [];
                    const memoryMetrics = {
                        'memory_allocated_end_mb': 'Allocated',
                        'memory_reserved_end_mb': 'Reserved',
                        'cpu_memory_end_mb': 'CPU'
                    };
                    
                    // Create a trace for each memory metric
                    for (const [metricKey, metricLabel] of Object.entries(memoryMetrics)) {
                        // Skip metrics that don't exist in the data
                        if (!modelHWData.some(d => d[metricKey] !== undefined)) {
                            continue;
                        }
                        
                        traces.push({
                            x: modelHWData.map(d => `${d.model} - ${d.hardware.toUpperCase()} (BS ${d.batch_size})`),
                            y: modelHWData.map(d => d[metricKey] !== undefined ? d[metricKey] : 0),
                            type: 'bar',
                            name: metricLabel
                        });
                    }
                    
                    // Add peak memory as a line
                    if (modelHWData.some(d => d.memory_peak_mb !== undefined)) {
                        traces.push({
                            x: modelHWData.map(d => `${d.model} - ${d.hardware.toUpperCase()} (BS ${d.batch_size})`),
                            y: modelHWData.map(d => d.memory_peak_mb !== undefined ? d.memory_peak_mb : 0),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Peak Memory',
                            marker: {
                                size: 8,
                                symbol: 'diamond'
                            },
                            line: {
                                color: 'red',
                                width: 2
                            }
                        });
                    }
                    
                    const layout = {
                        title: 'Memory Breakdown by Model/Hardware',
                        xaxis: {
                            title: 'Model / Hardware',
                            type: 'category'
                        },
                        yaxis: {
                            title: 'Memory (MB)'
                        },
                        barmode: 'stack',
                        legend: {
                            orientation: 'h',
                            yanchor: 'bottom',
                            y: -0.3
                        }
                    };
                    
                    Plotly.newPlot('memoryBreakdownChart', traces, layout);
                }
                
                function updateBatchSizeChart(data, metric) {
                    const metricLabels = {
                        'latency_ms': 'Latency (ms)',
                        'throughput_items_per_sec': 'Throughput (items/sec)',
                        'memory_usage_mb': 'Memory Usage (MB)'
                    };
                    
                    // Group by model, hardware and batch size
                    const groupedData = {};
                    
                    for (const d of data) {
                        const key = `${d.model} - ${d.hardware} - ${d.batch_size}`;
                        if (!groupedData[key]) {
                            groupedData[key] = {
                                model: d.model,
                                hardware: d.hardware,
                                batch_size: d.batch_size,
                                values: []
                            };
                        }
                        
                        if (d[metric] !== undefined) {
                            groupedData[key].values.push(d[metric]);
                        }
                    }
                    
                    // Calculate averages
                    const chartData = Object.values(groupedData).map(group => {
                        return {
                            model: group.model,
                            hardware: group.hardware,
                            batch_size: group.batch_size,
                            [metric]: group.values.length > 0 ? group.values.reduce((a, b) => a + b, 0) / group.values.length : 0
                        };
                    });
                    
                    // Create grouped bar chart
                    const batches = [...new Set(chartData.map(d => d.batch_size))].sort((a, b) => a - b);
                    const hardwares = [...new Set(chartData.map(d => d.hardware))];
                    const models = [...new Set(chartData.map(d => d.model))];
                    
                    const traces = [];
                    
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const filteredData = chartData.filter(d => d.model === model && d.hardware === hardware);
                            
                            if (filteredData.length > 0) {
                                traces.push({
                                    x: filteredData.map(d => d.batch_size),
                                    y: filteredData.map(d => d[metric]),
                                    type: 'bar',
                                    name: `${model} - ${hardware.toUpperCase()}`
                                });
                            }
                        }
                    }
                    
                    const layout = {
                        title: `${metricLabels[metric]} Comparison by Batch Size`,
                        xaxis: {
                            title: 'Batch Size',
                            type: 'category'
                        },
                        yaxis: {
                            title: metricLabels[metric]
                        },
                        barmode: 'group',
                        legend: {
                            orientation: 'h',
                            yanchor: 'bottom',
                            y: -0.2
                        }
                    };
                    
                    Plotly.newPlot('batchSizeChart', traces, layout);
                }
                
                function updateHardwareEfficiencyChart(data) {
                    // Check if we have hardware efficiency data
                    const hasEfficiencyData = data.some(d => d.hardware_efficiency !== undefined);
                    
                    if (!hasEfficiencyData) {
                        // Display message in the chart container
                        document.getElementById('hardwareEfficiencyChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No hardware efficiency data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    // Group by model and hardware
                    const modelHardwareData = {};
                    
                    for (const d of data) {
                        if (d.hardware_efficiency === undefined) continue;
                        
                        const key = `${d.model}`;
                        if (!modelHardwareData[key]) {
                            modelHardwareData[key] = [];
                        }
                        
                        modelHardwareData[key].push({
                            hardware: d.hardware,
                            efficiency: d.hardware_efficiency,
                            tensor_core_eligible: d.tensor_core_eligible,
                            model_type: d.model_type || 'unknown'
                        });
                    }
                    
                    // Create bar chart data
                    const chartData = [];
                    const models = Object.keys(modelHardwareData);
                    
                    for (const model of models) {
                        for (const hw of modelHardwareData[model]) {
                            chartData.push({
                                model: model,
                                hardware: hw.hardware.toUpperCase(),
                                efficiency: hw.efficiency,
                                model_type: hw.model_type,
                                tensor_core: hw.tensor_core_eligible ? 'Yes' : 'No'
                            });
                        }
                    }
                    
                    // Sort by hardware then by model
                    chartData.sort((a, b) => {
                        if (a.hardware !== b.hardware) return a.hardware.localeCompare(b.hardware);
                        return a.model.localeCompare(b.model);
                    });
                    
                    // Create chart
                    const trace = {
                        x: chartData.map(d => `${d.model} (${d.hardware})`),
                        y: chartData.map(d => d.efficiency),
                        type: 'bar',
                        marker: {
                            color: chartData.map(d => {
                                // Color bars by hardware
                                if (d.hardware === 'CUDA' || d.hardware === 'GPU') return 'rgba(50, 168, 82, 0.8)';
                                if (d.hardware === 'CPU') return 'rgba(54, 162, 235, 0.8)';
                                if (d.hardware === 'MPS') return 'rgba(255, 99, 132, 0.8)';
                                if (d.hardware === 'ROCM') return 'rgba(255, 159, 64, 0.8)';
                                return 'rgba(153, 102, 255, 0.8)';
                            })
                        },
                        text: chartData.map(d => {
                            // Add tensor core and model type info
                            let text = `${d.efficiency.toFixed(2)}%`;
                            if (d.tensor_core === 'Yes') text += '<br>Tensor Core';
                            return text;
                        }),
                        textposition: 'auto',
                        hovertemplate: '%{x}<br>Efficiency: %{y:.2f}%<br>Model Type: %{customdata}<extra></extra>',
                        customdata: chartData.map(d => d.model_type)
                    };
                    
                    const layout = {
                        title: 'Hardware Efficiency by Model and Platform',
                        xaxis: {
                            title: 'Model (Hardware)',
                            tickangle: -45
                        },
                        yaxis: {
                            title: 'Hardware Efficiency (%)',
                            range: [0, 105]  // Max 105% to ensure there's room for text at the top
                        }
                    };
                    
                    Plotly.newPlot('hardwareEfficiencyChart', [trace], layout);
                }
                
                function updateHardwareComparisonChart(data) {
                    // Check if we have data for multiple hardware platforms
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    if (hardwares.length <= 1) {
                        // Display message in the chart container
                        document.getElementById('hardwareComparisonChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                Hardware comparison requires data from multiple hardware platforms
                            </div>`;
                        return;
                    }
                    
                    // Group by model and batch size
                    const models = [...new Set(data.map(d => d.model))];
                    const batchSizes = [...new Set(data.map(d => d.batch_size))].sort((a, b) => a - b);
                    
                    // If we have multiple batch sizes, use the smallest one for fairest comparison
                    const batchSize = batchSizes[0];
                    
                    // Prepare data for the chart
                    const chartData = [];
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const modelHardwareData = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                d.batch_size === batchSize
                            );
                            
                            if (modelHardwareData.length > 0) {
                                // Use the first result for this model/hardware/batch_size combination
                                const result = modelHardwareData[0];
                                chartData.push({
                                    model: model,
                                    hardware: hardware.toUpperCase(),
                                    latency: result.latency_ms || 0,
                                    throughput: result.throughput_items_per_sec || 0,
                                    gflops: result.gflops || 0,
                                    efficiency: result.hardware_efficiency || 0
                                });
                            }
                        }
                    }
                    
                    // Create chart
                    const hardwareColors = {
                        'CPU': 'rgba(54, 162, 235, 0.8)',
                        'CUDA': 'rgba(50, 168, 82, 0.8)',
                        'GPU': 'rgba(50, 168, 82, 0.8)',
                        'MPS': 'rgba(255, 99, 132, 0.8)',
                        'ROCM': 'rgba(255, 159, 64, 0.8)'
                    };
                    
                    const selectedMetric = document.getElementById('metricSelect').value;
                    const metricToUse = (selectedMetric === 'latency_ms' || selectedMetric.includes('latency')) ? 'latency' :
                                        (selectedMetric === 'throughput_items_per_sec') ? 'throughput' :
                                        (selectedMetric.includes('flops')) ? 'gflops' : 'efficiency';
                    
                    const metricLabels = {
                        'latency': 'Latency (ms)',
                        'throughput': 'Throughput (items/sec)',
                        'gflops': 'GFLOPs',
                        'efficiency': 'Hardware Efficiency (%)'
                    };
                    
                    const traces = [];
                    for (const hardware of hardwares) {
                        const hwData = chartData.filter(d => d.hardware === hardware.toUpperCase());
                        traces.push({
                            x: hwData.map(d => d.model),
                            y: hwData.map(d => d[metricToUse]),
                            type: 'bar',
                            name: hardware.toUpperCase(),
                            marker: {
                                color: hardwareColors[hardware.toUpperCase()] || 'rgba(153, 102, 255, 0.8)'
                            }
                        });
                    }
                    
                    const layout = {
                        title: `Hardware Comparison (${metricLabels[metricToUse]})`,
                        xaxis: {
                            title: 'Model',
                            type: 'category'
                        },
                        yaxis: {
                            title: metricLabels[metricToUse]
                        },
                        barmode: 'group',
                        legend: {
                            orientation: 'h',
                            yanchor: 'bottom',
                            y: -0.2
                        }
                    };
                    
                    Plotly.newPlot('hardwareComparisonChart', traces, layout);
                }
                
                function updateBandwidthUtilizationChart(data) {
                    // Check if we have bandwidth data
                    const hasBandwidthData = data.some(d => 
                        d.avg_bandwidth_gbps !== undefined && 
                        d.peak_theoretical_bandwidth_gbps !== undefined &&
                        d.bandwidth_utilization_percent !== undefined
                    );
                    
                    if (!hasBandwidthData) {
                        // Display message in the chart container
                        document.getElementById('bandwidthUtilizationChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No bandwidth utilization data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    // Group by model and hardware to create a comparison chart
                    const models = [...new Set(data.map(d => d.model))];
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    
                    // Prepare traces
                    const traces = [];
                    
                    // Create a bar chart for bandwidth utilization
                    const utilizationTrace = {
                        type: 'bar',
                        x: [],
                        y: [],
                        name: 'Bandwidth Utilization (%)',
                        marker: {
                            color: '#00aaff',
                            colorscale: [
                                [0, 'rgb(255, 0, 0)'],
                                [0.5, 'rgb(255, 255, 0)'],
                                [1, 'rgb(0, 255, 0)']
                            ]
                        }
                    };
                    
                    // Create traces for actual bandwidth and peak bandwidth
                    const actualBandwidthTrace = {
                        type: 'bar',
                        x: [],
                        y: [],
                        name: 'Actual Bandwidth (GB/s)',
                        marker: {
                            color: '#00aa00'
                        }
                    };
                    
                    const peakBandwidthTrace = {
                        type: 'bar',
                        x: [],
                        y: [],
                        name: 'Peak Theoretical Bandwidth (GB/s)',
                        marker: {
                            color: '#aa0000',
                            opacity: 0.5
                        }
                    };
                    
                    // For each model and hardware, add data points
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const matches = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                d.avg_bandwidth_gbps !== undefined &&
                                d.peak_theoretical_bandwidth_gbps !== undefined
                            );
                            
                            if (matches.length > 0) {
                                // For simplicity, use the first batch size for this model/hardware
                                const result = matches[0];
                                
                                const label = `${model} - ${hardware.toUpperCase()}`
                                
                                utilizationTrace.x.push(label);
                                utilizationTrace.y.push(result.bandwidth_utilization_percent || 0);
                                
                                actualBandwidthTrace.x.push(label);
                                actualBandwidthTrace.y.push(result.avg_bandwidth_gbps || 0);
                                
                                peakBandwidthTrace.x.push(label);
                                peakBandwidthTrace.y.push(result.peak_theoretical_bandwidth_gbps || 0);
                            }
                        }
                    }
                    
                    // Check if we have enough data for each type of chart
                    if (utilizationTrace.x.length > 0) {
                        // Create utilization percentage chart
                        Plotly.newPlot('bandwidthUtilizationChart', [utilizationTrace], {
                            title: 'Memory Bandwidth Utilization (%)',
                            xaxis: {
                                title: 'Model / Hardware',
                                tickangle: -45
                            },
                            yaxis: {
                                title: 'Utilization (%)',
                                range: [0, 100]
                            }
                        });
                    } else if (actualBandwidthTrace.x.length > 0 && peakBandwidthTrace.x.length > 0) {
                        // Create comparative bandwidth chart
                        Plotly.newPlot('bandwidthUtilizationChart', [actualBandwidthTrace, peakBandwidthTrace], {
                            title: 'Actual vs. Peak Memory Bandwidth (GB/s)',
                            xaxis: {
                                title: 'Model / Hardware',
                                tickangle: -45
                            },
                            yaxis: {
                                title: 'Bandwidth (GB/s)'
                            },
                            barmode: 'group'
                        });
                    } else {
                        // Not enough data for either chart
                        document.getElementById('bandwidthUtilizationChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                Insufficient bandwidth data available for visualization
                            </div>`;
                    }
                }
                
                function updateRooflineModelChart(data) {
                    // Check if we have roofline data
                    const hasRooflineData = data.some(d => 
                        d.roofline_data !== undefined && 
                        d.roofline_data.peak_compute_flops !== undefined
                    );
                    
                    if (!hasRooflineData) {
                        // Display message in the chart container
                        document.getElementById('rooflineModelChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No roofline model data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    // Collect roofline data points
                    const rooflinePoints = [];
                    
                    for (const result of data) {
                        if (result.roofline_data) {
                            rooflinePoints.push({
                                model: result.model,
                                hardware: result.hardware,
                                batch_size: result.batch_size,
                                peak_compute: result.roofline_data.peak_compute_flops,
                                peak_bandwidth: result.roofline_data.peak_memory_bandwidth_bytes_per_sec,
                                ridge_point: result.roofline_data.ridge_point_flops_per_byte,
                                arithmetic_intensity: result.roofline_data.arithmetic_intensity_flops_per_byte,
                                actual_performance: result.roofline_data.actual_performance_flops,
                                is_compute_bound: result.roofline_data.is_compute_bound
                            });
                        }
                    }
                    
                    if (rooflinePoints.length === 0) {
                        // No valid data points
                        document.getElementById('rooflineModelChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                Insufficient roofline data available for visualization
                            </div>`;
                        return;
                    }
                    
                    // For simplicity, use the first data point to define the roofline
                    const firstPoint = rooflinePoints[0];
                    
                    // Create x-axis points for the roofline (log scale)
                    const xValues = [];
                    for (let i = -3; i <= 5; i++) {
                        xValues.push(Math.pow(10, i));
                    }
                    
                    // Create memory-bound part of the roofline
                    const memoryBoundX = xValues.filter(x => x <= firstPoint.ridge_point);
                    const memoryBoundY = memoryBoundX.map(x => x * (firstPoint.peak_bandwidth / 1e9));
                    
                    // Create compute-bound part of the roofline
                    const computeBoundX = xValues.filter(x => x >= firstPoint.ridge_point);
                    const computeBoundY = computeBoundX.map(() => firstPoint.peak_compute / 1e12);  // Convert to TFLOPS
                    
                    // Create roofline traces
                    const memoryBoundTrace = {
                        x: memoryBoundX,
                        y: memoryBoundY.map(y => y / 1e12),  // Convert to TFLOPS
                        mode: 'lines',
                        name: 'Memory Bound',
                        line: {
                            color: 'red',
                            width: 2
                        }
                    };
                    
                    const computeBoundTrace = {
                        x: computeBoundX,
                        y: computeBoundY,
                        mode: 'lines',
                        name: 'Compute Bound',
                        line: {
                            color: 'blue',
                            width: 2
                        }
                    };
                    
                    // Create data points for actual performance
                    const dataPointsTrace = {
                        x: rooflinePoints.map(p => p.arithmetic_intensity),
                        y: rooflinePoints.map(p => p.actual_performance / 1e12),  // Convert to TFLOPS
                        mode: 'markers',
                        name: 'Model Performance',
                        text: rooflinePoints.map(p => 
                            `${p.model} - ${p.hardware.toUpperCase()} (${p.is_compute_bound ? 'Compute Bound' : 'Memory Bound'})`
                        ),
                        marker: {
                            size: 10,
                            color: rooflinePoints.map(p => p.is_compute_bound ? 'blue' : 'red')
                        }
                    };
                    
                    // Plot the roofline model
                    Plotly.newPlot('rooflineModelChart', [memoryBoundTrace, computeBoundTrace, dataPointsTrace], {
                        title: 'Roofline Performance Model',
                        xaxis: {
                            title: 'Arithmetic Intensity (FLOP/Byte)',
                            type: 'log',
                            exponentformat: 'power'
                        },
                        yaxis: {
                            title: 'Performance (TFLOPS)',
                            type: 'log',
                            exponentformat: 'power'
                        },
                        legend: {
                            orientation: 'h',
                            yanchor: 'bottom',
                            y: -0.2
                        }
                    });
                }
                
                function updatePowerEfficiencyChart(data) {
                    // Check if we have power data
                    const hasPowerData = data.some(d => 
                        d.power_avg_watts !== undefined && 
                        d.gflops_per_watt !== undefined
                    );
                    
                    if (!hasPowerData) {
                        // Display message in the chart container
                        document.getElementById('powerEfficiencyChart').innerHTML = 
                            `<div style="text-align: center; padding: 50px; color: #666;">
                                No power efficiency data available for the selected models/hardware
                            </div>`;
                        return;
                    }
                    
                    // Group by model and hardware to create a comparison chart
                    const models = [...new Set(data.map(d => d.model))];
                    const hardwares = [...new Set(data.map(d => d.hardware))];
                    
                    // Prepare traces
                    const traces = [];
                    
                    // Create a scatter plot for power vs. GFLOPs
                    const powerTrace = {
                        x: [],
                        y: [],
                        text: [],
                        mode: 'markers',
                        type: 'scatter',
                        marker: {
                            size: 12,
                            color: []
                        },
                        name: 'GFLOPs vs Power'
                    };
                    
                    // Color scale for efficiency
                    const colors = ['#ff0000', '#ffa500', '#ffff00', '#008000', '#00ff00'];
                    
                    // Add data points for each model/hardware combination
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const matches = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                d.power_avg_watts !== undefined && 
                                d.gflops !== undefined
                            );
                            
                            if (matches.length > 0) {
                                // For simplicity, use the first batch size for this model/hardware
                                const result = matches[0];
                                
                                powerTrace.x.push(result.power_avg_watts);
                                powerTrace.y.push(result.gflops);
                                powerTrace.text.push(`${model} - ${hardware.toUpperCase()} (${result.gflops_per_watt?.toFixed(2) || 'N/A'} GFLOPs/W)`);
                                
                                // Determine color based on efficiency (higher is better)
                                const efficiencyValue = result.gflops_per_watt || 0;
                                const colorIndex = Math.min(Math.floor(efficiencyValue / 20), colors.length - 1);
                                powerTrace.marker.color.push(colors[colorIndex]);
                            }
                        }
                    }
                    
                    traces.push(powerTrace);
                    
                    // Add a bubble chart for comparing GFLOPs/Watt across hardware platforms
                    const efficiencyTrace = {
                        type: 'bar',
                        x: [],
                        y: [],
                        name: 'GFLOPs/Watt',
                        marker: {
                            color: '#00aaff'
                        }
                    };
                    
                    // For each model and hardware, add a bar representing GFLOPs/Watt
                    for (const model of models) {
                        for (const hardware of hardwares) {
                            const matches = data.filter(d => 
                                d.model === model && 
                                d.hardware === hardware && 
                                d.gflops_per_watt !== undefined
                            );
                            
                            if (matches.length > 0) {
                                // For simplicity, use the first batch size for this model/hardware
                                const result = matches[0];
                                
                                efficiencyTrace.x.push(`${model} - ${hardware.toUpperCase()}`);
                                efficiencyTrace.y.push(result.gflops_per_watt);
                            }
                        }
                    }
                    
                    // Add the second trace if we have data
                    if (efficiencyTrace.x.length > 0) {
                        // Create a new figure for the GFLOPs/Watt bar chart
                        Plotly.newPlot('powerEfficiencyChart', [efficiencyTrace], {
                            title: 'Power Efficiency (GFLOPs/Watt)',
                            xaxis: {
                                title: 'Model / Hardware',
                                tickangle: -45
                            },
                            yaxis: {
                                title: 'GFLOPs/Watt'
                            }
                        });
                    } else {
                        // If no GFLOPs/Watt data, plot the scatter
                        Plotly.newPlot('powerEfficiencyChart', traces, {
                            title: 'Performance vs. Power Consumption',
                            xaxis: {
                                title: 'Power (W)'
                            },
                            yaxis: {
                                title: 'GFLOPs'
                            }
                        });
                    }
                }
                
                function updateTable(data) {
                    const tbody = document.querySelector('#resultsTable tbody');
                    tbody.innerHTML = '';
                    
                    for (const d of data) {
                        const row = document.createElement('tr');
                        
                        // Format numbers with commas for thousands
                        const formatNumber = (num) => {
                            if (num === undefined) return 'N/A';
                            if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
                            if (num >= 1000) return (num / 1000).toFixed(2) + 'K';
                            return num.toFixed(2);
                        };
                        
                        // Create row with all the detailed metrics
                        row.innerHTML = `
                            <td>${d.model}</td>
                            <td>${d.hardware.toUpperCase()}</td>
                            <td>${d.batch_size}</td>
                            <td>${d.sequence_length}</td>
                            <td>${d.latency_ms !== undefined ? d.latency_ms.toFixed(2) : 'N/A'}</td>
                            <td>${d.latency_p90_ms !== undefined ? d.latency_p90_ms.toFixed(2) : 'N/A'}</td>
                            <td>${d.latency_p99_ms !== undefined ? d.latency_p99_ms.toFixed(2) : 'N/A'}</td>
                            <td>${d.throughput_items_per_sec !== undefined ? d.throughput_items_per_sec.toFixed(2) : 'N/A'}</td>
                            <td>${d.memory_usage_mb !== undefined ? d.memory_usage_mb.toFixed(2) : 'N/A'}</td>
                            <td>${d.memory_peak_mb !== undefined ? d.memory_peak_mb.toFixed(2) : 'N/A'}</td>
                            <td>${d.gflops !== undefined ? formatNumber(d.gflops) : (d.flops !== undefined ? formatNumber(d.flops / 1e9) : 'N/A')}</td>
                            <td>${d.hardware_efficiency !== undefined ? d.hardware_efficiency.toFixed(2) + '%' : 'N/A'}</td>
                            <td>${d.power_avg_watts !== undefined ? d.power_avg_watts.toFixed(2) : 'N/A'}</td>
                            <td>${d.energy_joules !== undefined ? d.energy_joules.toFixed(2) : 'N/A'}</td>
                            <td>${d.gflops_per_watt !== undefined ? formatNumber(d.gflops_per_watt) : 'N/A'}</td>
                            <td>${d.model_type || 'N/A'}</td>
                            <td>${d.tensor_core_eligible === true ? 'Yes' : (d.tensor_core_eligible === false ? 'No' : 'N/A')}</td>
                            <td>${d.hardware === 'cpu' ? 'Yes' : 'No'}</td>
                            <td>${d.hardware === 'cuda' ? 'Yes' : 'No'}</td>
                            <td>${d.hardware === 'gpu' ? 'Yes' : 'No'}</td>
                        `;
                        
                        tbody.appendChild(row);
                    }
                }
            </script>
        </body>
        </html>
        """
        
        # Generate model options
        model_options = "\n".join(
            f'<option value="{model}">{model}</option>'
            for model in df["model"].unique()
        )
        
        # Generate hardware options
        hardware_options = "\n".join(
            f'<option value="{hw}">{hw.upper()}</option>'
            for hw in df["hardware"].unique()
        )
        
        # Generate table rows
        table_rows = ""
        for _, row in df.iterrows():
            # Format GFLOPs
            if 'gflops' in row:
                gflops_value = row['gflops']
            elif 'flops' in row:
                gflops_value = row['flops'] / 1e9
            else:
                gflops_value = 'N/A'
                
            if isinstance(gflops_value, (int, float)):
                if gflops_value >= 1000:
                    gflops_formatted = f"{gflops_value/1000:.2f}K"
                else:
                    gflops_formatted = f"{gflops_value:.2f}"
            else:
                gflops_formatted = 'N/A'
            
            table_rows += f"""
            <tr>
                <td>{row['model']}</td>
                <td>{row['hardware'].upper()}</td>
                <td>{row['batch_size']}</td>
                <td>{row['sequence_length']}</td>
                <td>{row.get('latency_ms', 'N/A')}</td>
                <td>{row.get('latency_p90_ms', 'N/A')}</td>
                <td>{row.get('latency_p99_ms', 'N/A')}</td>
                <td>{row.get('throughput_items_per_sec', 'N/A')}</td>
                <td>{row.get('memory_usage_mb', 'N/A')}</td>
                <td>{row.get('memory_peak_mb', 'N/A')}</td>
                <td>{gflops_formatted}</td>
            </tr>
            """
        
        # Fill in template
        html_content = html_template.format(
            model_options=model_options,
            hardware_options=hardware_options,
            table_rows=table_rows,
            data_json=df.to_json(orient="records")
        )
        
        # Write dashboard HTML
        with open(dashboard_html, "w") as f:
            f.write(html_content)
        
        logger.info(f"Generated dashboard at {dashboard_html}")
        return dashboard_html
        
    except ImportError:
        logger.error("dash and plotly are required for dashboard generation. Install with 'pip install dash plotly pandas'")
        return None
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        return None

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard")
    parser.add_argument("--results-dir", type=str, default="benchmark_results",
                      help="Directory containing benchmark result files")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save dashboard (defaults to results-dir/dashboard)")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load benchmark results
    result_files = []
    for root, _, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith(".json") and file.startswith("benchmark_"):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        logger.error(f"No benchmark result files found in {args.results_dir}")
        sys.exit(1)
    
    # Load results
    results = []
    for file in result_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not results:
        logger.error("No valid benchmark results found")
        sys.exit(1)
    
    # Generate dashboard
    output_dir = args.output_dir or args.results_dir
    dashboard_path = generate_dashboard(results, output_dir)
    
    if dashboard_path:
        logger.info(f"Dashboard generated at {dashboard_path}")
        sys.exit(0)
    else:
        logger.error("Failed to generate dashboard")
        sys.exit(1)