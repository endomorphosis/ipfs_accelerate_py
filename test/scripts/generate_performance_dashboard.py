#!/usr/bin/env python
"""
Performance Dashboard Generator

This script generates an interactive performance dashboard for GitHub Pages from benchmark results
stored in the DuckDB database. It creates visualizations and reports for hardware compatibility,
performance metrics, and regression analysis.

Usage:
    python generate_performance_dashboard.py --db-path ./benchmark_db.duckdb
    python generate_performance_dashboard.py --output-dir ./dashboard
    python generate_performance_dashboard.py --run-id 20250306123456
    """

    import os
    import sys
    import json
    import argparse
    import logging
    import datetime
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Tuple

try:
    import duckdb
    import pandas as pd
    import numpy as np
    from jinja2 import Template
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    print())))))))))))))))"Missing dependencies. Please install with: pip install duckdb pandas numpy jinja2")
    print())))))))))))))))"Continuing in limited functionality mode...")

try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_VISUALIZATION = True
except ImportError:
    HAVE_VISUALIZATION = False
    print())))))))))))))))"Visualization libraries not found. Charts will be disabled.")
    print())))))))))))))))"Install with: pip install matplotlib plotly")

try:
    # Import dashboard configuration
    from ci_dashboard_config import ())))))))))))))))
    DASHBOARD_CONFIG,
    REPORT_TEMPLATES,
    CHART_CONFIGS,
    GITHUB_PAGES_CONFIG
    )
except ImportError:
    print())))))))))))))))"Dashboard configuration not found. Using default configuration.")
    # Set up basic configuration as fallback
    DASHBOARD_CONFIG = {}}}}}}}}}}}}}}}
    REPORT_TEMPLATES = {}}}}}}}}}}}}}}}
    CHART_CONFIGS = {}}}}}}}}}}}}}}}
    GITHUB_PAGES_CONFIG = {}}}}}}}}}}}}}}}

# Configure logging
    logging.basicConfig())))))))))))))))
    level=logging.INFO,
    format='%())))))))))))))))asctime)s [],%())))))))))))))))levelname)s] %())))))))))))))))message)s',
    handlers=[],
    logging.StreamHandler())))))))))))))))),
    logging.FileHandler())))))))))))))))"dashboard_generation.log")
    ]
    )
    logger = logging.getLogger())))))))))))))))__name__)

def connect_to_database())))))))))))))))db_path: Optional[],str] = None) -> Optional[],duckdb.DuckDBPyConnection]:
    """Connect to the benchmark database."""
    if not HAVE_DEPS:
    return None
        
    # Get database path from env var or parameter or default
    if db_path is None:
        db_path = os.environ.get())))))))))))))))"BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    try:
        # Connect to the database
        conn = duckdb.connect())))))))))))))))db_path)
        logger.info())))))))))))))))f"Connected to database: {}}}}}}}}}}}}}}db_path}")
        return conn
    except Exception as e:
        logger.error())))))))))))))))f"Error connecting to database: {}}}}}}}}}}}}}}e}")
        return None

def get_latest_run_id())))))))))))))))conn: duckdb.DuckDBPyConnection) -> Optional[],str]:
    """Get the latest run ID from the database."""
    try:
        # Query to get the latest run ID
        query = """
        SELECT run_id
        FROM performance_results
        ORDER BY test_timestamp DESC
        LIMIT 1
        """
        
        result = conn.execute())))))))))))))))query).fetchone()))))))))))))))))
        if result is None:
            logger.warning())))))))))))))))"No run IDs found in database")
        return None
        
        run_id = result[],0]
        logger.info())))))))))))))))f"Latest run ID: {}}}}}}}}}}}}}}run_id}")
    return run_id
    except Exception as e:
        logger.error())))))))))))))))f"Error getting latest run ID: {}}}}}}}}}}}}}}e}")
    return None

def get_compatibility_data())))))))))))))))conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Get compatibility data for hardware platforms and models."""
    try:
        # Query to get model compatibility with hardware platforms
        query = """
        SELECT 
        m.model_name,
        m.model_type,
        h.hardware_type,
        CASE WHEN MAX())))))))))))))))tr.success) = 1 THEN 1 ELSE 0 END as is_compatible
        FROM 
        models m
        CROSS JOIN 
        hardware_platforms h
        LEFT JOIN 
        test_results tr ON m.model_id = tr.model_id AND h.hardware_id = tr.hardware_id
        GROUP BY 
        m.model_name, m.model_type, h.hardware_type
        ORDER BY 
        m.model_type, m.model_name, h.hardware_type
        """
        
        df = conn.execute())))))))))))))))query).df()))))))))))))))))
        
        # Create a pivot table
        pivot_df = df.pivot())))))))))))))))index=[],'model_name', 'model_type'], 
        columns='hardware_type',
        values='is_compatible').fillna())))))))))))))))0)
        
        # Reset index to get a regular dataframe
        pivot_df = pivot_df.reset_index()))))))))))))))))
        
    return pivot_df
    except Exception as e:
        logger.error())))))))))))))))f"Error getting compatibility data: {}}}}}}}}}}}}}}e}")
    return pd.DataFrame()))))))))))))))))

def get_performance_data())))))))))))))))conn: duckdb.DuckDBPyConnection) -> Dict[],str, pd.DataFrame]:
    """Get performance data for hardware platforms and models."""
    try:
        # Query to get throughput data
        throughput_query = """
        SELECT 
        m.model_name,
        h.hardware_type,
        AVG())))))))))))))))pr.throughput_items_per_second) as throughput
        FROM 
        performance_results pr
        JOIN 
        models m ON pr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON pr.hardware_id = h.hardware_id
        GROUP BY 
        m.model_name, h.hardware_type
        ORDER BY 
        m.model_name, h.hardware_type
        """
        
        # Query to get latency data
        latency_query = """
        SELECT 
        m.model_name,
        h.hardware_type,
        AVG())))))))))))))))pr.average_latency_ms) as latency
        FROM 
        performance_results pr
        JOIN 
        models m ON pr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON pr.hardware_id = h.hardware_id
        GROUP BY 
        m.model_name, h.hardware_type
        ORDER BY 
        m.model_name, h.hardware_type
        """
        
        # Query to get memory data
        memory_query = """
        SELECT 
        m.model_name,
        h.hardware_type,
        AVG())))))))))))))))pr.memory_peak_mb) as memory
        FROM 
        performance_results pr
        JOIN 
        models m ON pr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON pr.hardware_id = h.hardware_id
        GROUP BY 
        m.model_name, h.hardware_type
        ORDER BY 
        m.model_name, h.hardware_type
        """
        
        # Execute queries
        throughput_df = conn.execute())))))))))))))))throughput_query).df()))))))))))))))))
        latency_df = conn.execute())))))))))))))))latency_query).df()))))))))))))))))
        memory_df = conn.execute())))))))))))))))memory_query).df()))))))))))))))))
        
        # Create pivot tables
        throughput_pivot = throughput_df.pivot())))))))))))))))index='model_name', 
        columns='hardware_type',
        values='throughput').fillna())))))))))))))))0)
        
        latency_pivot = latency_df.pivot())))))))))))))))index='model_name', 
        columns='hardware_type',
        values='latency').fillna())))))))))))))))0)
        
        memory_pivot = memory_df.pivot())))))))))))))))index='model_name', 
        columns='hardware_type',
        values='memory').fillna())))))))))))))))0)
        
        # Reset index
        throughput_pivot = throughput_pivot.reset_index()))))))))))))))))
        latency_pivot = latency_pivot.reset_index()))))))))))))))))
        memory_pivot = memory_pivot.reset_index()))))))))))))))))
        
    return {}}}}}}}}}}}}}}
    "throughput": throughput_pivot,
    "latency": latency_pivot,
    "memory": memory_pivot,
    "raw_throughput": throughput_df,
    "raw_latency": latency_df,
    "raw_memory": memory_df
    }
    except Exception as e:
        logger.error())))))))))))))))f"Error getting performance data: {}}}}}}}}}}}}}}e}")
    return {}}}}}}}}}}}}}}}

def get_regression_data())))))))))))))))conn: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """Get regression data for the specified run."""
    try:
        # Query to get regression data
        query = """
        WITH current_run AS ())))))))))))))))
        -- Current run results
        SELECT
        pr.model_id,
        pr.hardware_id,
        pr.throughput_items_per_second as current_throughput,
        pr.average_latency_ms as current_latency,
        pr.memory_peak_mb as current_memory
        FROM
        performance_results pr
        WHERE
        pr.run_id = ?
        ),
        previous_run AS ())))))))))))))))
        -- Previous run results
        SELECT
        pr.model_id,
        pr.hardware_id,
        pr.throughput_items_per_second as prev_throughput,
        pr.average_latency_ms as prev_latency,
        pr.memory_peak_mb as prev_memory
        FROM
        performance_results pr
        JOIN ())))))))))))))))
        -- Get previous run ID for each model-hardware combination
        SELECT
        pr2.model_id,
        pr2.hardware_id,
        MAX())))))))))))))))pr2.test_timestamp) as max_timestamp
        FROM
        performance_results pr2
        JOIN
        current_run cr ON pr2.model_id = cr.model_id AND pr2.hardware_id = cr.hardware_id
        WHERE
        pr2.run_id != ?
        GROUP BY
        pr2.model_id, pr2.hardware_id
        ) prev ON pr.model_id = prev.model_id AND pr.hardware_id = prev.hardware_id AND pr.test_timestamp = prev.max_timestamp
        )
        -- Combine current and previous runs to detect regressions
        SELECT 
        m.model_name,
        h.hardware_type,
        'throughput' as metric,
        cr.current_throughput as current_value,
        pr.prev_throughput as previous_value,
        ())))))))))))))))cr.current_throughput - pr.prev_throughput) / pr.prev_throughput as change_percent
        FROM 
        current_run cr
        JOIN 
        previous_run pr ON cr.model_id = pr.model_id AND cr.hardware_id = pr.hardware_id
        JOIN 
        models m ON cr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON cr.hardware_id = h.hardware_id
        WHERE 
        pr.prev_throughput > 0 AND cr.current_throughput < pr.prev_throughput * 0.9
        
        UNION ALL
        
        SELECT 
        m.model_name,
        h.hardware_type,
        'latency' as metric,
        cr.current_latency as current_value,
        pr.prev_latency as previous_value,
        ())))))))))))))))cr.current_latency - pr.prev_latency) / pr.prev_latency as change_percent
        FROM 
        current_run cr
        JOIN 
        previous_run pr ON cr.model_id = pr.model_id AND cr.hardware_id = pr.hardware_id
        JOIN 
        models m ON cr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON cr.hardware_id = h.hardware_id
        WHERE 
        pr.prev_latency > 0 AND cr.current_latency > pr.prev_latency * 1.1
        
        UNION ALL
        
        SELECT 
        m.model_name,
        h.hardware_type,
        'memory' as metric,
        cr.current_memory as current_value,
        pr.prev_memory as previous_value,
        ())))))))))))))))cr.current_memory - pr.prev_memory) / pr.prev_memory as change_percent
        FROM 
        current_run cr
        JOIN 
        previous_run pr ON cr.model_id = pr.model_id AND cr.hardware_id = pr.hardware_id
        JOIN 
        models m ON cr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON cr.hardware_id = h.hardware_id
        WHERE 
        pr.prev_memory > 0 AND cr.current_memory > pr.prev_memory * 1.1
        
        ORDER BY
        change_percent
        """
        
        # Execute query
        df = conn.execute())))))))))))))))query, [],run_id, run_id]).df()))))))))))))))))
        
        # Add severity column
        df[],'severity'] = df[],'change_percent'].apply())))))))))))))))lambda x: 'high' if abs())))))))))))))))x) > 0.2 else 'medium')
        
        return df:
    except Exception as e:
        logger.error())))))))))))))))f"Error getting regression data: {}}}}}}}}}}}}}}e}")
            return pd.DataFrame()))))))))))))))))

def get_historical_data())))))))))))))))conn: duckdb.DuckDBPyConnection, days: int = 30) -> pd.DataFrame:
    """Get historical performance data for the last N days."""
    try:
        # Query to get historical performance data
        query = """
        SELECT 
        DATE_TRUNC())))))))))))))))'day', pr.test_timestamp) as date,
        m.model_name,
        h.hardware_type,
        AVG())))))))))))))))pr.throughput_items_per_second) as throughput,
        AVG())))))))))))))))pr.average_latency_ms) as latency,
        AVG())))))))))))))))pr.memory_peak_mb) as memory
        FROM 
        performance_results pr
        JOIN 
        models m ON pr.model_id = m.model_id
        JOIN 
        hardware_platforms h ON pr.hardware_id = h.hardware_id
        WHERE 
        pr.test_timestamp >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY 
        DATE_TRUNC())))))))))))))))'day', pr.test_timestamp), m.model_name, h.hardware_type
        ORDER BY 
        date, m.model_name, h.hardware_type
        """
        
        # Execute query
        df = conn.execute())))))))))))))))query % days).df()))))))))))))))))
        
    return df
    except Exception as e:
        logger.error())))))))))))))))f"Error getting historical data: {}}}}}}}}}}}}}}e}")
    return pd.DataFrame()))))))))))))))))

def generate_compatibility_chart())))))))))))))))df: pd.DataFrame, output_dir: str) -> Optional[],str]:
    """Generate compatibility heatmap chart."""
    if not HAVE_VISUALIZATION or df.empty:
    return None
    
    try:
        # Create compatibility heatmap
        # First, convert DataFrame to format suitable for heatmap
        model_names = df[],'model_name'].tolist()))))))))))))))))
        model_types = df[],'model_type'].tolist()))))))))))))))))
        
        # Get hardware columns
        hw_columns = [],col for col in df.columns if col not in [],'model_name', 'model_type']]
        
        # Create data for heatmap
        z_data = df[],hw_columns].values.tolist()))))))))))))))))
        
        # Create heatmap using plotly
        fig = go.Figure())))))))))))))))data=go.Heatmap())))))))))))))))
        z=z_data,
        x=hw_columns,
        y=[],f"{}}}}}}}}}}}}}}model} ()))))))))))))))){}}}}}}}}}}}}}}mtype})" for model, mtype in zip())))))))))))))))model_names, model_types)],
        colorscale=[],[],0, 'red'], [],1, 'green']],
        zauto=False,
        zmin=0,
        zmax=1
        ))
        
        # Add title and labels
        fig.update_layout())))))))))))))))
        title='Model-Hardware Compatibility Matrix',
        xaxis_title='Hardware Platform',
        yaxis_title='Model',
        height=max())))))))))))))))500, len())))))))))))))))model_names) * 20)  # Adjust height based on number of models
        )
        
        # Save chart
        os.makedirs())))))))))))))))output_dir, exist_ok=True)
        output_file = os.path.join())))))))))))))))output_dir, 'compatibility_matrix.html')
        fig.write_html())))))))))))))))output_file)
        
        return output_file:
    except Exception as e:
        logger.error())))))))))))))))f"Error generating compatibility chart: {}}}}}}}}}}}}}}e}")
            return None

def generate_performance_charts())))))))))))))))perf_data: Dict[],str, pd.DataFrame], output_dir: str) -> List[],str]:
    """Generate performance comparison charts."""
    if not HAVE_VISUALIZATION or not perf_data:
    return [],]
    
    output_files = [],]
    
    try:
        # Create throughput bar chart
        if 'raw_throughput' in perf_data and not perf_data[],'raw_throughput'].empty:
            df = perf_data[],'raw_throughput']
            fig = px.bar())))))))))))))))
            df,
            x='model_name',
            y='throughput',
            color='hardware_type',
            title='Throughput Comparison by Hardware Platform',
            labels={}}}}}}}}}}}}}}
            'model_name': 'Model',
            'throughput': 'Throughput ())))))))))))))))items/s)',
            'hardware_type': 'Hardware Platform'
            },
            barmode='group'
            )
            
            # Save chart
            os.makedirs())))))))))))))))output_dir, exist_ok=True)
            output_file = os.path.join())))))))))))))))output_dir, 'throughput_comparison.html')
            fig.write_html())))))))))))))))output_file)
            output_files.append())))))))))))))))output_file)
        
        # Create latency bar chart
        if 'raw_latency' in perf_data and not perf_data[],'raw_latency'].empty:
            df = perf_data[],'raw_latency']
            fig = px.bar())))))))))))))))
            df,
            x='model_name',
            y='latency',
            color='hardware_type',
            title='Latency Comparison by Hardware Platform',
            labels={}}}}}}}}}}}}}}
            'model_name': 'Model',
            'latency': 'Latency ())))))))))))))))ms)',
            'hardware_type': 'Hardware Platform'
            },
            barmode='group'
            )
            
            # Save chart
            os.makedirs())))))))))))))))output_dir, exist_ok=True)
            output_file = os.path.join())))))))))))))))output_dir, 'latency_comparison.html')
            fig.write_html())))))))))))))))output_file)
            output_files.append())))))))))))))))output_file)
        
        # Create memory bar chart
        if 'raw_memory' in perf_data and not perf_data[],'raw_memory'].empty:
            df = perf_data[],'raw_memory']
            fig = px.bar())))))))))))))))
            df,
            x='model_name',
            y='memory',
            color='hardware_type',
            title='Memory Usage Comparison by Hardware Platform',
            labels={}}}}}}}}}}}}}}
            'model_name': 'Model',
            'memory': 'Memory Usage ())))))))))))))))MB)',
            'hardware_type': 'Hardware Platform'
            },
            barmode='group'
            )
            
            # Save chart
            os.makedirs())))))))))))))))output_dir, exist_ok=True)
            output_file = os.path.join())))))))))))))))output_dir, 'memory_comparison.html')
            fig.write_html())))))))))))))))output_file)
            output_files.append())))))))))))))))output_file)
        
            return output_files
    except Exception as e:
        logger.error())))))))))))))))f"Error generating performance charts: {}}}}}}}}}}}}}}e}")
            return output_files

def generate_regression_charts())))))))))))))))df: pd.DataFrame, output_dir: str) -> Optional[],str]:
    """Generate regression analysis chart."""
    if not HAVE_VISUALIZATION or df.empty:
    return None
    
    try:
        # Create regression chart
        fig = px.bar())))))))))))))))
        df,
        x='model_name',
        y='change_percent',
        color='hardware_type',
        facet_row='metric',
        title='Performance Regressions by Metric',
        labels={}}}}}}}}}}}}}}
        'model_name': 'Model',
        'change_percent': 'Change ())))))))))))))))%)',
        'hardware_type': 'Hardware Platform',
        'metric': 'Metric'
        },
        barmode='group',
        category_orders={}}}}}}}}}}}}}}"metric": [],"throughput", "latency", "memory"]}
        )
        
        # Customize appearance
        fig.update_layout())))))))))))))))height=800)
        fig.for_each_annotation())))))))))))))))lambda a: a.update())))))))))))))))text=a.text.split())))))))))))))))"=")[],1]))
        
        # Save chart
        os.makedirs())))))))))))))))output_dir, exist_ok=True)
        output_file = os.path.join())))))))))))))))output_dir, 'regressions.html')
        fig.write_html())))))))))))))))output_file)
        
    return output_file
    except Exception as e:
        logger.error())))))))))))))))f"Error generating regression chart: {}}}}}}}}}}}}}}e}")
    return None

def generate_historical_chart())))))))))))))))df: pd.DataFrame, output_dir: str) -> Optional[],str]:
    """Generate historical performance trend chart."""
    if not HAVE_VISUALIZATION or df.empty:
    return None
    
    try:
        # Filter to just a few representative models to avoid cluttered chart
        models = df[],'model_name'].unique()))))))))))))))))
        if len())))))))))))))))models) > 5:
            # Take first 5 models to avoid cluttered chart
            selected_models = models[],:5]
            df = df[],df[],'model_name'].isin())))))))))))))))selected_models)]
        
        # Create historical chart for throughput
            fig = px.line())))))))))))))))
            df, 
            x='date', 
            y='throughput', 
            color='model_name',
            line_dash='hardware_type',
            title='Throughput Trends by Model and Hardware',
            labels={}}}}}}}}}}}}}}
            'date': 'Date',
            'throughput': 'Throughput ())))))))))))))))items/s)',
            'model_name': 'Model',
            'hardware_type': 'Hardware Platform'
            }
            )
        
        # Save chart
            os.makedirs())))))))))))))))output_dir, exist_ok=True)
            output_file = os.path.join())))))))))))))))output_dir, 'historical_throughput.html')
            fig.write_html())))))))))))))))output_file)
        
        return output_file
    except Exception as e:
        logger.error())))))))))))))))f"Error generating historical chart: {}}}}}}}}}}}}}}e}")
        return None

def generate_compatibility_report())))))))))))))))df: pd.DataFrame, format: str = "markdown") -> Optional[],str]:
    """Generate hardware compatibility report."""
    if df.empty:
    return None
    
    try:
        template_key = f"compatibility_matrix"
        if template_key not in REPORT_TEMPLATES:
            logger.warning())))))))))))))))f"Template {}}}}}}}}}}}}}}template_key} not found in REPORT_TEMPLATES")
            # Generate simple report
            if format == "markdown":
            return generate_compatibility_markdown())))))))))))))))df)
            else:
            return generate_compatibility_html())))))))))))))))df)
        
            template_data = REPORT_TEMPLATES[],template_key]
            template_format = template_data.get())))))))))))))))"format", "markdown")
            template_content = template_data.get())))))))))))))))"template")
        
        if not template_content:
            logger.warning())))))))))))))))f"Template content for {}}}}}}}}}}}}}}template_key} is empty")
            # Generate simple report
            if format == "markdown":
            return generate_compatibility_markdown())))))))))))))))df)
            else:
            return generate_compatibility_html())))))))))))))))df)
        
        # Process data for template
            hardware_types = [],col for col in df.columns if col not in [],'model_name', 'model_type']]
        
        models_data = [],]:
        for _, row in df.iterrows())))))))))))))))):
            model_data = {}}}}}}}}}}}}}}
            "model_name": row[],'model_name'],
            "model_type": row[],'model_type'] or "unknown",
            "compatibilities": [],]
            }
            
            for hw in hardware_types:
                model_data[],"compatibilities"].append()))))))))))))))){}}}}}}}}}}}}}}
                "hardware_type": hw,
                "compatible": bool())))))))))))))))row[],hw])
                })
            
                models_data.append())))))))))))))))model_data)
        
        # Prepare recommendations by model type
                recommendations = [],]
        for model_type in df[],'model_type'].dropna())))))))))))))))).unique())))))))))))))))):
            # Find the best hardware for each model type based on compatibility
            model_type_df = df[],df[],'model_type'] == model_type]
            hw_counts = {}}}}}}}}}}}}}}hw: model_type_df[],hw].sum())))))))))))))))) for hw in hardware_types}:
                best_hw = max())))))))))))))))hw_counts.items())))))))))))))))), key=lambda x: x[],1])[],0] if hw_counts else "cpu"
            
            recommendations.append()))))))))))))))){}}}}}}}}}}}}}}:
                "model_type": model_type,
                "recommended_hardware": best_hw,
                "notes": f"Best compatibility for {}}}}}}}}}}}}}}model_type} models"
                })
        
        # Render template
                template = Template())))))))))))))))template_content)
                rendered = template.render())))))))))))))))
                generated_date=datetime.datetime.now())))))))))))))))).strftime())))))))))))))))"%Y-%m-%d %H:%M:%S"),
                hardware_types=hardware_types,
                models=models_data,
                recommendations=recommendations
                )
        
                return rendered
    except Exception as e:
        logger.error())))))))))))))))f"Error generating compatibility report: {}}}}}}}}}}}}}}e}")
        # Generate simple report as fallback
        if format == "markdown":
        return generate_compatibility_markdown())))))))))))))))df)
        else:
        return generate_compatibility_html())))))))))))))))df)

def generate_compatibility_markdown())))))))))))))))df: pd.DataFrame) -> str:
    """Generate simple markdown compatibility report."""
    markdown = [],]
    markdown.append())))))))))))))))"# Hardware Compatibility Matrix")
    markdown.append())))))))))))))))f"\nGenerated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}")
    
    # Add header row
    hardware_types = [],col for col in df.columns if col not in [],'model_name', 'model_type']]
    header = "| Model | Type | " + " | ".join())))))))))))))))hardware_types) + " |"
    markdown.append())))))))))))))))"\n" + header)
    
    # Add separator row
    separator = "|" + "---|" * ())))))))))))))))2 + len())))))))))))))))hardware_types))
    markdown.append())))))))))))))))separator)
    
    # Add data rows:
    for _, row in df.iterrows())))))))))))))))):
        model_name = row[],'model_name']
        model_type = row[],'model_type'] if not pd.isna())))))))))))))))row[],'model_type']) else ""
        
        # Create compatibility indicators
        compatibility = [],]:
        for hw in hardware_types:
            if hw in row and row[],hw]:
                compatibility.append())))))))))))))))"✅")  # Compatible
            else:
                compatibility.append())))))))))))))))"⚠️")  # Not compatible
        
                data_row = f"| {}}}}}}}}}}}}}}model_name} | {}}}}}}}}}}}}}}model_type} | " + " | ".join())))))))))))))))compatibility) + " |"
                markdown.append())))))))))))))))data_row)
    
                return "\n".join())))))))))))))))markdown)

def generate_compatibility_html())))))))))))))))df: pd.DataFrame) -> str:
    """Generate simple HTML compatibility report."""
    html = [],]
    html.append())))))))))))))))"<!DOCTYPE html>")
    html.append())))))))))))))))"<html>")
    html.append())))))))))))))))"<head>")
    html.append())))))))))))))))"    <title>Hardware Compatibility Matrix</title>")
    html.append())))))))))))))))"    <style>")
    html.append())))))))))))))))"        body {}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }")
    html.append())))))))))))))))"        table {}}}}}}}}}}}}}} border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    html.append())))))))))))))))"        th, td {}}}}}}}}}}}}}} border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append())))))))))))))))"        th {}}}}}}}}}}}}}} background-color: #f2f2f2; }")
    html.append())))))))))))))))"        tr:nth-child())))))))))))))))even) {}}}}}}}}}}}}}} background-color: #f9f9f9; }")
    html.append())))))))))))))))"        .compatible {}}}}}}}}}}}}}} color: green; }")
    html.append())))))))))))))))"        .not-compatible {}}}}}}}}}}}}}} color: orange; }")
    html.append())))))))))))))))"    </style>")
    html.append())))))))))))))))"</head>")
    html.append())))))))))))))))"<body>")
    
    html.append())))))))))))))))"<h1>Hardware Compatibility Matrix</h1>")
    html.append())))))))))))))))f"<p>Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add table
    html.append())))))))))))))))"<table>")
    
    # Add header row
    hardware_types = [],col for col in df.columns if col not in [],'model_name', 'model_type']]
    html.append())))))))))))))))"<tr>")
    html.append())))))))))))))))"    <th>Model</th>")
    html.append())))))))))))))))"    <th>Type</th>"):
    for hw in hardware_types:
        html.append())))))))))))))))f"    <th>{}}}}}}}}}}}}}}hw}</th>")
        html.append())))))))))))))))"</tr>")
    
    # Add data rows
    for _, row in df.iterrows())))))))))))))))):
        model_name = row[],'model_name']
        model_type = row[],'model_type'] if not pd.isna())))))))))))))))row[],'model_type']) else ""
        
        html.append())))))))))))))))"<tr>")
        html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}model_name}</td>")
        html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}model_type}</td>")
        
        # Add compatibility indicators:
        for hw in hardware_types:
            if hw in row and row[],hw]:
                html.append())))))))))))))))f"    <td class='compatible'>✅</td>")  # Compatible
            else:
                html.append())))))))))))))))f"    <td class='not-compatible'>⚠️</td>")  # Not compatible
        
                html.append())))))))))))))))"</tr>")
    
                html.append())))))))))))))))"</table>")
                html.append())))))))))))))))"</body>")
                html.append())))))))))))))))"</html>")
    
                return "\n".join())))))))))))))))html)

                def generate_performance_report())))))))))))))))perf_data: Dict[],str, pd.DataFrame],
                chart_files: List[],str] = None,
                              format: str = "html") -> Optional[],str]:
                                  """Generate performance report."""
    if not perf_data:
                                  return None
    
    try:
        template_key = f"performance_report"
        if template_key not in REPORT_TEMPLATES:
            logger.warning())))))))))))))))f"Template {}}}}}}}}}}}}}}template_key} not found in REPORT_TEMPLATES")
            # Generate simple report
        return generate_performance_html())))))))))))))))perf_data, chart_files)
        
        template_data = REPORT_TEMPLATES[],template_key]
        template_format = template_data.get())))))))))))))))"format", "html")
        template_content = template_data.get())))))))))))))))"template")
        
        if not template_content:
            logger.warning())))))))))))))))f"Template content for {}}}}}}}}}}}}}}template_key} is empty")
            # Generate simple report
        return generate_performance_html())))))))))))))))perf_data, chart_files)
        
        # Process data for template
        hardware_types = [],]
        throughput_data = [],]
        latency_data = [],]
        memory_data = [],]
        
        if 'throughput' in perf_data and not perf_data[],'throughput'].empty:
            for _, row in perf_data[],'throughput'].iterrows())))))))))))))))):
                model_data = {}}}}}}}}}}}}}}"model_name": row[],'model_name'], "values": [],]}
                for col in perf_data[],'throughput'].columns:
                    if col != 'model_name':
                        if col not in hardware_types:
                            hardware_types.append())))))))))))))))col)
                            model_data[],"values"].append())))))))))))))))f"{}}}}}}}}}}}}}}row[],col]:.2f}")
                            throughput_data.append())))))))))))))))model_data)
        
        if 'latency' in perf_data and not perf_data[],'latency'].empty:
            for _, row in perf_data[],'latency'].iterrows())))))))))))))))):
                model_data = {}}}}}}}}}}}}}}"model_name": row[],'model_name'], "values": [],]}
                for col in perf_data[],'latency'].columns:
                    if col != 'model_name':
                        model_data[],"values"].append())))))))))))))))f"{}}}}}}}}}}}}}}row[],col]:.2f}")
                        latency_data.append())))))))))))))))model_data)
        
        if 'memory' in perf_data and not perf_data[],'memory'].empty:
            for _, row in perf_data[],'memory'].iterrows())))))))))))))))):
                model_data = {}}}}}}}}}}}}}}"model_name": row[],'model_name'], "values": [],]}
                for col in perf_data[],'memory'].columns:
                    if col != 'model_name':
                        model_data[],"values"].append())))))))))))))))f"{}}}}}}}}}}}}}}row[],col]:.2f}")
                        memory_data.append())))))))))))))))model_data)
        
        # Render template
                        template = Template())))))))))))))))template_content)
                        rendered = template.render())))))))))))))))
                        generated_date=datetime.datetime.now())))))))))))))))).strftime())))))))))))))))"%Y-%m-%d %H:%M:%S"),
                        hardware_types=hardware_types,
                        throughput_data=throughput_data,
                        latency_data=latency_data,
                        memory_data=memory_data,
                        chart_files=chart_files or [],]
                        )
        
                    return rendered
    except Exception as e:
        logger.error())))))))))))))))f"Error generating performance report: {}}}}}}}}}}}}}}e}")
        # Generate simple report as fallback
                    return generate_performance_html())))))))))))))))perf_data, chart_files)

def generate_performance_html())))))))))))))))perf_data: Dict[],str, pd.DataFrame], chart_files: List[],str] = None) -> str:
    """Generate simple HTML performance report."""
    html = [],]
    html.append())))))))))))))))"<!DOCTYPE html>")
    html.append())))))))))))))))"<html>")
    html.append())))))))))))))))"<head>")
    html.append())))))))))))))))"    <title>Performance Report</title>")
    html.append())))))))))))))))"    <style>")
    html.append())))))))))))))))"        body {}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }")
    html.append())))))))))))))))"        table {}}}}}}}}}}}}}} border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    html.append())))))))))))))))"        th, td {}}}}}}}}}}}}}} border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append())))))))))))))))"        th {}}}}}}}}}}}}}} background-color: #f2f2f2; }")
    html.append())))))))))))))))"        tr:nth-child())))))))))))))))even) {}}}}}}}}}}}}}} background-color: #f9f9f9; }")
    html.append())))))))))))))))"        .chart {}}}}}}}}}}}}}} width: 100%; height: 600px; margin-bottom: 20px; border: none; }")
    html.append())))))))))))))))"    </style>")
    html.append())))))))))))))))"</head>")
    html.append())))))))))))))))"<body>")
    
    html.append())))))))))))))))"<h1>Performance Report</h1>")
    html.append())))))))))))))))f"<p>Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}</p>")
    
    # Add charts if available::
    if chart_files:
        html.append())))))))))))))))"<h2>Performance Charts</h2>")
        for chart_file in chart_files:
            chart_name = os.path.basename())))))))))))))))chart_file)
            html.append())))))))))))))))f"<iframe src='{}}}}}}}}}}}}}}chart_name}' class='chart'></iframe>")
    
    # Add throughput table
    if 'throughput' in perf_data and not perf_data[],'throughput'].empty:
        html.append())))))))))))))))"<h2>Throughput ())))))))))))))))items/second)</h2>")
        html.append())))))))))))))))"<table>")
        
        # Header row
        html.append())))))))))))))))"<tr>")
        html.append())))))))))))))))"    <th>Model</th>")
        for col in perf_data[],'throughput'].columns:
            if col != 'model_name':
                html.append())))))))))))))))f"    <th>{}}}}}}}}}}}}}}col}</th>")
                html.append())))))))))))))))"</tr>")
        
        # Data rows
        for _, row in perf_data[],'throughput'].iterrows())))))))))))))))):
            html.append())))))))))))))))"<tr>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'model_name']}</td>")
            for col in perf_data[],'throughput'].columns:
                if col != 'model_name':
                    html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],col]:.2f}</td>")
                    html.append())))))))))))))))"</tr>")
        
                    html.append())))))))))))))))"</table>")
    
    # Add latency table
    if 'latency' in perf_data and not perf_data[],'latency'].empty:
        html.append())))))))))))))))"<h2>Latency ())))))))))))))))ms)</h2>")
        html.append())))))))))))))))"<table>")
        
        # Header row
        html.append())))))))))))))))"<tr>")
        html.append())))))))))))))))"    <th>Model</th>")
        for col in perf_data[],'latency'].columns:
            if col != 'model_name':
                html.append())))))))))))))))f"    <th>{}}}}}}}}}}}}}}col}</th>")
                html.append())))))))))))))))"</tr>")
        
        # Data rows
        for _, row in perf_data[],'latency'].iterrows())))))))))))))))):
            html.append())))))))))))))))"<tr>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'model_name']}</td>")
            for col in perf_data[],'latency'].columns:
                if col != 'model_name':
                    html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],col]:.2f}</td>")
                    html.append())))))))))))))))"</tr>")
        
                    html.append())))))))))))))))"</table>")
    
    # Add memory table
    if 'memory' in perf_data and not perf_data[],'memory'].empty:
        html.append())))))))))))))))"<h2>Memory Usage ())))))))))))))))MB)</h2>")
        html.append())))))))))))))))"<table>")
        
        # Header row
        html.append())))))))))))))))"<tr>")
        html.append())))))))))))))))"    <th>Model</th>")
        for col in perf_data[],'memory'].columns:
            if col != 'model_name':
                html.append())))))))))))))))f"    <th>{}}}}}}}}}}}}}}col}</th>")
                html.append())))))))))))))))"</tr>")
        
        # Data rows
        for _, row in perf_data[],'memory'].iterrows())))))))))))))))):
            html.append())))))))))))))))"<tr>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'model_name']}</td>")
            for col in perf_data[],'memory'].columns:
                if col != 'model_name':
                    html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],col]:.2f}</td>")
                    html.append())))))))))))))))"</tr>")
        
                    html.append())))))))))))))))"</table>")
    
                    html.append())))))))))))))))"</body>")
                    html.append())))))))))))))))"</html>")
    
                return "\n".join())))))))))))))))html)

                def generate_regression_report())))))))))))))))df: pd.DataFrame,
                run_id: str,
                chart_file: Optional[],str] = None,
                             format: str = "html") -> Optional[],str]:
                                 """Generate regression report."""
    if df.empty:
        # Generate empty report
        if format == "html":
            html = [],]
            html.append())))))))))))))))"<!DOCTYPE html>")
            html.append())))))))))))))))"<html>")
            html.append())))))))))))))))"<head>")
            html.append())))))))))))))))"    <title>Regression Report</title>")
            html.append())))))))))))))))"    <style>")
            html.append())))))))))))))))"        body {}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }")
            html.append())))))))))))))))"    </style>")
            html.append())))))))))))))))"</head>")
            html.append())))))))))))))))"<body>")
            html.append())))))))))))))))"    <h1>Performance Regression Report</h1>")
            html.append())))))))))))))))f"    <p>Run ID: {}}}}}}}}}}}}}}run_id}</p>")
            html.append())))))))))))))))f"    <p>Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}</p>")
            html.append())))))))))))))))"    <p>No performance regressions detected.</p>")
            html.append())))))))))))))))"</body>")
            html.append())))))))))))))))"</html>")
        return "\n".join())))))))))))))))html)
        else:  # markdown
                return f"# Performance Regression Report\n\nRun ID: {}}}}}}}}}}}}}}run_id}\n\nGenerated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}\n\nNo performance regressions detected."
    
    try:
        template_key = f"regression_report"
        if template_key not in REPORT_TEMPLATES:
            logger.warning())))))))))))))))f"Template {}}}}}}}}}}}}}}template_key} not found in REPORT_TEMPLATES")
            # Generate simple report
            if format == "html":
            return generate_regression_html())))))))))))))))df, run_id, chart_file)
            else:
            return generate_regression_markdown())))))))))))))))df, run_id)
        
            template_data = REPORT_TEMPLATES[],template_key]
            template_format = template_data.get())))))))))))))))"format", "html")
            template_content = template_data.get())))))))))))))))"template")
        
        if not template_content:
            logger.warning())))))))))))))))f"Template content for {}}}}}}}}}}}}}}template_key} is empty")
            # Generate simple report
            if format == "html":
            return generate_regression_html())))))))))))))))df, run_id, chart_file)
            else:
            return generate_regression_markdown())))))))))))))))df, run_id)
        
        # Process data for template
            high_count = len())))))))))))))))df[],df[],'severity'] == 'high'])
            medium_count = len())))))))))))))))df[],df[],'severity'] == 'medium'])
        
        # Format regression data
            regressions = [],]
        for _, row in df.iterrows())))))))))))))))):
            # Format values
            current = f"{}}}}}}}}}}}}}}row[],'current_value']:.2f}" if not pd.isna())))))))))))))))row[],'current_value']) else "N/A":::
            previous = f"{}}}}}}}}}}}}}}row[],'previous_value']:.2f}" if not pd.isna())))))))))))))))row[],'previous_value']) else "N/A":::
                change = f"{}}}}}}}}}}}}}}row[],'change_percent'] * 100:.2f}%" if not pd.isna())))))))))))))))row[],'change_percent']) else "N/A"
            
            # Metric display name
            metric_display = row[],'metric'].capitalize())))))))))))))))):
            if row[],'metric'] == 'throughput':
                metric_display = 'Throughput'
            elif row[],'metric'] == 'latency':
                metric_display = 'Latency'
            elif row[],'metric'] == 'memory':
                metric_display = 'Memory'
            
                regression = {}}}}}}}}}}}}}}
                "model_name": row[],'model_name'],
                "hardware_type": row[],'hardware_type'],
                "metric": row[],'metric'],
                "metric_display_name": metric_display,
                "current_value": current,
                "historical_value": previous,
                "change_percent": change,
                "severity": row[],'severity']
                }
            
                regressions.append())))))))))))))))regression)
        
        # Render template
                template = Template())))))))))))))))template_content)
                rendered = template.render())))))))))))))))
                run_id=run_id,
                generated_date=datetime.datetime.now())))))))))))))))).strftime())))))))))))))))"%Y-%m-%d %H:%M:%S"),
                regression_count=len())))))))))))))))df),
                high_count=high_count,
                medium_count=medium_count,
                regressions=regressions,
                chart_file=os.path.basename())))))))))))))))chart_file) if chart_file else None
                )
        
        return rendered:
    except Exception as e:
        logger.error())))))))))))))))f"Error generating regression report: {}}}}}}}}}}}}}}e}")
        # Generate simple report as fallback
        if format == "html":
        return generate_regression_html())))))))))))))))df, run_id, chart_file)
        else:
        return generate_regression_markdown())))))))))))))))df, run_id)

def generate_regression_html())))))))))))))))df: pd.DataFrame, run_id: str, chart_file: Optional[],str] = None) -> str:
    """Generate simple HTML regression report."""
    html = [],]
    html.append())))))))))))))))"<!DOCTYPE html>")
    html.append())))))))))))))))"<html>")
    html.append())))))))))))))))"<head>")
    html.append())))))))))))))))"    <title>Regression Report</title>")
    html.append())))))))))))))))"    <style>")
    html.append())))))))))))))))"        body {}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }")
    html.append())))))))))))))))"        table {}}}}}}}}}}}}}} border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    html.append())))))))))))))))"        th, td {}}}}}}}}}}}}}} border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html.append())))))))))))))))"        th {}}}}}}}}}}}}}} background-color: #f2f2f2; }")
    html.append())))))))))))))))"        tr:nth-child())))))))))))))))even) {}}}}}}}}}}}}}} background-color: #f9f9f9; }")
    html.append())))))))))))))))"        .regression {}}}}}}}}}}}}}} color: red; }")
    html.append())))))))))))))))"        .high-severity {}}}}}}}}}}}}}} font-weight: bold; background-color: #ffcccc; }")
    html.append())))))))))))))))"        .medium-severity {}}}}}}}}}}}}}} background-color: #fff2cc; }")
    html.append())))))))))))))))"        .chart {}}}}}}}}}}}}}} width: 100%; height: 600px; margin-bottom: 20px; border: none; }")
    html.append())))))))))))))))"    </style>")
    html.append())))))))))))))))"</head>")
    html.append())))))))))))))))"<body>")
    
    html.append())))))))))))))))"<h1>Performance Regression Report</h1>")
    html.append())))))))))))))))f"<p>Run ID: {}}}}}}}}}}}}}}run_id}</p>")
    html.append())))))))))))))))f"<p>Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}</p>")
    
    # Summary
    high_count = len())))))))))))))))df[],df[],'severity'] == 'high'])
    medium_count = len())))))))))))))))df[],df[],'severity'] == 'medium'])
    
    html.append())))))))))))))))f"<p><strong>{}}}}}}}}}}}}}}len())))))))))))))))df)}</strong> performance regressions detected.</p>")
    html.append())))))))))))))))"<h2>Summary by Severity</h2>")
    html.append())))))))))))))))"<ul>")
    html.append())))))))))))))))f"    <li><strong>High Severity:</strong> {}}}}}}}}}}}}}}high_count} regressions</li>")
    html.append())))))))))))))))f"    <li><strong>Medium Severity:</strong> {}}}}}}}}}}}}}}medium_count} regressions</li>")
    html.append())))))))))))))))"</ul>")
    
    # Add chart if available::
    if chart_file:
        html.append())))))))))))))))"<h2>Regression Chart</h2>")
        chart_name = os.path.basename())))))))))))))))chart_file)
        html.append())))))))))))))))f"<iframe src='{}}}}}}}}}}}}}}chart_name}' class='chart'></iframe>")
    
    # Detailed regressions
        html.append())))))))))))))))"<h2>Detailed Regressions</h2>")
        html.append())))))))))))))))"<table>")
        html.append())))))))))))))))"<tr>")
        html.append())))))))))))))))"    <th>Model</th>")
        html.append())))))))))))))))"    <th>Hardware</th>")
        html.append())))))))))))))))"    <th>Metric</th>")
        html.append())))))))))))))))"    <th>Current Value</th>")
        html.append())))))))))))))))"    <th>Previous Value</th>")
        html.append())))))))))))))))"    <th>Change</th>")
        html.append())))))))))))))))"    <th>Severity</th>")
        html.append())))))))))))))))"</tr>")
    
    # Sort by severity and then by change percent
        sorted_df = df.sort_values())))))))))))))))[],'severity', 'change_percent'], ascending=[],True, False])
    
    for _, row in sorted_df.iterrows())))))))))))))))):
        # Format values
        current = f"{}}}}}}}}}}}}}}row[],'current_value']:.2f}" if not pd.isna())))))))))))))))row[],'current_value']) else "N/A":::
        previous = f"{}}}}}}}}}}}}}}row[],'previous_value']:.2f}" if not pd.isna())))))))))))))))row[],'previous_value']) else "N/A":::
            change = f"{}}}}}}}}}}}}}}row[],'change_percent'] * 100:.2f}%" if not pd.isna())))))))))))))))row[],'change_percent']) else "N/A"
        
        # CSS class for severity
            severity_class = row[],'severity'] + "-severity"
        
        # Metric display name
            metric_display = row[],'metric'].capitalize()))))))))))))))))
        
            html.append())))))))))))))))f"<tr class='{}}}}}}}}}}}}}}severity_class}'>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'model_name']}</td>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'hardware_type']}</td>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}metric_display}</td>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}current}</td>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}previous}</td>")
            html.append())))))))))))))))f"    <td class='regression'>{}}}}}}}}}}}}}}change}</td>")
            html.append())))))))))))))))f"    <td>{}}}}}}}}}}}}}}row[],'severity'].upper()))))))))))))))))}</td>")
            html.append())))))))))))))))"</tr>")
    
            html.append())))))))))))))))"</table>")
            html.append())))))))))))))))"</body>")
            html.append())))))))))))))))"</html>")
    
            return "\n".join())))))))))))))))html)
:
def generate_regression_markdown())))))))))))))))df: pd.DataFrame, run_id: str) -> str:
    """Generate simple markdown regression report."""
    markdown = [],]
    markdown.append())))))))))))))))"# Performance Regression Report")
    markdown.append())))))))))))))))f"Run ID: {}}}}}}}}}}}}}}run_id}")
    markdown.append())))))))))))))))f"Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    high_count = len())))))))))))))))df[],df[],'severity'] == 'high'])
    medium_count = len())))))))))))))))df[],df[],'severity'] == 'medium'])
    
    markdown.append())))))))))))))))f"**{}}}}}}}}}}}}}}len())))))))))))))))df)}** performance regressions detected.")
    markdown.append())))))))))))))))"\n## Summary by Severity")
    markdown.append())))))))))))))))f"- **High Severity:** {}}}}}}}}}}}}}}high_count} regressions")
    markdown.append())))))))))))))))f"- **Medium Severity:** {}}}}}}}}}}}}}}medium_count} regressions")
    
    # Detailed regressions
    markdown.append())))))))))))))))"\n## Detailed Regressions")
    markdown.append())))))))))))))))"| Model | Hardware | Metric | Current Value | Previous Value | Change | Severity |")
    markdown.append())))))))))))))))"|-------|----------|--------|---------------|----------------|--------|----------|")
    
    # Sort by severity and then by change percent
    sorted_df = df.sort_values())))))))))))))))[],'severity', 'change_percent'], ascending=[],True, False])
    
    for _, row in sorted_df.iterrows())))))))))))))))):
        # Format values
        current = f"{}}}}}}}}}}}}}}row[],'current_value']:.2f}" if not pd.isna())))))))))))))))row[],'current_value']) else "N/A":::
        previous = f"{}}}}}}}}}}}}}}row[],'previous_value']:.2f}" if not pd.isna())))))))))))))))row[],'previous_value']) else "N/A":::
            change = f"{}}}}}}}}}}}}}}row[],'change_percent'] * 100:.2f}%" if not pd.isna())))))))))))))))row[],'change_percent']) else "N/A"
        
        # Metric display name
            metric_display = row[],'metric'].capitalize()))))))))))))))))
        
            markdown.append())))))))))))))))f"| {}}}}}}}}}}}}}}row[],'model_name']} | {}}}}}}}}}}}}}}row[],'hardware_type']} | {}}}}}}}}}}}}}}metric_display} | {}}}}}}}}}}}}}}current} | {}}}}}}}}}}}}}}previous} | {}}}}}}}}}}}}}}change} | {}}}}}}}}}}}}}}row[],'severity'].upper()))))))))))))))))} |")
    
            return "\n".join())))))))))))))))markdown)
:
    def generate_index_page())))))))))))))))output_dir: str,
    compatibility_file: Optional[],str] = None,
    performance_file: Optional[],str] = None,
    regression_file: Optional[],str] = None,
                      historical_file: Optional[],str] = None) -> str:
                          """Generate index page for the dashboard."""
                          html = [],]
                          html.append())))))))))))))))"<!DOCTYPE html>")
                          html.append())))))))))))))))"<html>")
                          html.append())))))))))))))))"<head>")
                          html.append())))))))))))))))"    <title>IPFS Accelerate Performance Dashboard</title>")
                          html.append())))))))))))))))"    <style>")
                          html.append())))))))))))))))"        body {}}}}}}}}}}}}}} font-family: Arial, sans-serif; margin: 20px; }")
                          html.append())))))))))))))))"        h1 {}}}}}}}}}}}}}} color: #333; }")
                          html.append())))))))))))))))"        .nav {}}}}}}}}}}}}}} background-color: #f8f9fa; padding: 10px; margin-bottom: 20px; }")
                          html.append())))))))))))))))"        .nav a {}}}}}}}}}}}}}} margin-right: 15px; text-decoration: none; color: #007bff; }")
                          html.append())))))))))))))))"        .nav a:hover {}}}}}}}}}}}}}} text-decoration: underline; }")
                          html.append())))))))))))))))"        .card {}}}}}}}}}}}}}} border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }")
                          html.append())))))))))))))))"        .card h2 {}}}}}}}}}}}}}} margin-top: 0; }")
                          html.append())))))))))))))))"        .btn {}}}}}}}}}}}}}} display: inline-block; background-color: #007bff; color: white; padding: 8px 15px; ")
                          html.append())))))))))))))))"               text-decoration: none; border-radius: 5px; margin-top: 10px; }")
                          html.append())))))))))))))))"        .btn:hover {}}}}}}}}}}}}}} background-color: #0056b3; }")
                          html.append())))))))))))))))"    </style>")
                          html.append())))))))))))))))"</head>")
                          html.append())))))))))))))))"<body>")
    
    # Header
                          html.append())))))))))))))))"<h1>IPFS Accelerate Performance Dashboard</h1>")
                          html.append())))))))))))))))f"<p>Generated: {}}}}}}}}}}}}}}datetime.datetime.now())))))))))))))))).strftime())))))))))))))))'%Y-%m-%d %H:%M:%S')}</p>")
    
    # Navigation
                          html.append())))))))))))))))"<div class='nav'>")
                          html.append())))))))))))))))"    <a href='index.html'>Overview</a>")
    if compatibility_file:
        compatibility_name = os.path.basename())))))))))))))))compatibility_file)
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}compatibility_name}'>Compatibility Matrix</a>")
    if performance_file:
        performance_name = os.path.basename())))))))))))))))performance_file)
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}performance_name}'>Performance Report</a>")
    if regression_file:
        regression_name = os.path.basename())))))))))))))))regression_file)
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}regression_name}'>Regression Analysis</a>")
    if historical_file:
        historical_name = os.path.basename())))))))))))))))historical_file)
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}historical_name}'>Historical Trends</a>")
        html.append())))))))))))))))"</div>")
    
    # Card for Compatibility Matrix
    if compatibility_file:
        compatibility_name = os.path.basename())))))))))))))))compatibility_file)
        html.append())))))))))))))))"<div class='card'>")
        html.append())))))))))))))))"    <h2>Hardware Compatibility Matrix</h2>")
        html.append())))))))))))))))"    <p>View the compatibility status of models across different hardware platforms.</p>")
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}compatibility_name}' class='btn'>View Matrix</a>")
        html.append())))))))))))))))"</div>")
    
    # Card for Performance Report
    if performance_file:
        performance_name = os.path.basename())))))))))))))))performance_file)
        html.append())))))))))))))))"<div class='card'>")
        html.append())))))))))))))))"    <h2>Performance Report</h2>")
        html.append())))))))))))))))"    <p>Explore detailed performance metrics including throughput, latency, and memory usage.</p>")
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}performance_name}' class='btn'>View Report</a>")
        html.append())))))))))))))))"</div>")
    
    # Card for Regression Analysis
    if regression_file:
        regression_name = os.path.basename())))))))))))))))regression_file)
        html.append())))))))))))))))"<div class='card'>")
        html.append())))))))))))))))"    <h2>Regression Analysis</h2>")
        html.append())))))))))))))))"    <p>Identify performance regressions compared to previous test runs.</p>")
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}regression_name}' class='btn'>View Analysis</a>")
        html.append())))))))))))))))"</div>")
    
    # Card for Historical Trends
    if historical_file:
        historical_name = os.path.basename())))))))))))))))historical_file)
        html.append())))))))))))))))"<div class='card'>")
        html.append())))))))))))))))"    <h2>Historical Performance Trends</h2>")
        html.append())))))))))))))))"    <p>Track performance metrics over time to identify trends and improvements.</p>")
        html.append())))))))))))))))f"    <a href='{}}}}}}}}}}}}}}historical_name}' class='btn'>View Trends</a>")
        html.append())))))))))))))))"</div>")
    
    # Footer
        html.append())))))))))))))))"<hr>")
        html.append())))))))))))))))"<p>IPFS Accelerate Python Framework - Performance Dashboard</p>")
        html.append())))))))))))))))"<p>Generated as part of the CI/CD test results integration.</p>")
    
        html.append())))))))))))))))"</body>")
        html.append())))))))))))))))"</html>")
    
    # Save file
        index_file = os.path.join())))))))))))))))output_dir, 'index.html')
    with open())))))))))))))))index_file, 'w') as f:
        f.write())))))))))))))))"\n".join())))))))))))))))html))
    
        return index_file

def main())))))))))))))))):
    parser = argparse.ArgumentParser())))))))))))))))description='Performance Dashboard Generator')
    parser.add_argument())))))))))))))))'--db-path', help='Path to the DuckDB database')
    parser.add_argument())))))))))))))))'--output-dir', default='./dashboard', help='Output directory for dashboard files')
    parser.add_argument())))))))))))))))'--run-id', help='Specific run ID to analyze ())))))))))))))))default: latest run)')
    parser.add_argument())))))))))))))))'--days', type=int, default=30, help='Number of days for historical data ())))))))))))))))default: 30)')
    parser.add_argument())))))))))))))))'--formats', default='html,markdown', help='Output formats ())))))))))))))))comma-separated: html,markdown,json)')
    
    args = parser.parse_args()))))))))))))))))
    
    # Check for dependencies
    if not HAVE_DEPS:
        logger.error())))))))))))))))"Missing required dependencies. Please install with: pip install duckdb pandas numpy jinja2")
        sys.exit())))))))))))))))1)
    
    # Connect to database
        conn = connect_to_database())))))))))))))))args.db_path)
    if conn is None:
        logger.error())))))))))))))))"Could not connect to database.")
        sys.exit())))))))))))))))1)
    
    # Create output directory
        os.makedirs())))))))))))))))args.output_dir, exist_ok=True)
    
    # Get run ID
        run_id = args.run_id
    if run_id is None:
        run_id = get_latest_run_id())))))))))))))))conn)
        if run_id is None:
            logger.error())))))))))))))))"Could not determine run ID.")
            sys.exit())))))))))))))))1)
    
            logger.info())))))))))))))))f"Generating dashboard for run ID: {}}}}}}}}}}}}}}run_id}")
    
    # Get compatibility data
            logger.info())))))))))))))))"Getting compatibility data...")
            compatibility_df = get_compatibility_data())))))))))))))))conn)
    
    # Get performance data
            logger.info())))))))))))))))"Getting performance data...")
            performance_data = get_performance_data())))))))))))))))conn)
    
    # Get regression data
            logger.info())))))))))))))))"Getting regression data...")
            regression_df = get_regression_data())))))))))))))))conn, run_id)
    
    # Get historical data
            logger.info())))))))))))))))f"Getting historical data for the last {}}}}}}}}}}}}}}args.days} days...")
            historical_df = get_historical_data())))))))))))))))conn, args.days)
    
    # Generate charts
            charts_dir = os.path.join())))))))))))))))args.output_dir, 'charts')
            os.makedirs())))))))))))))))charts_dir, exist_ok=True)
    
            logger.info())))))))))))))))"Generating charts...")
            compatibility_chart = generate_compatibility_chart())))))))))))))))compatibility_df, charts_dir)
            performance_charts = generate_performance_charts())))))))))))))))performance_data, charts_dir)
            regression_chart = generate_regression_charts())))))))))))))))regression_df, charts_dir)
            historical_chart = generate_historical_chart())))))))))))))))historical_df, charts_dir)
    
    # Parse formats
    formats = [],f.strip())))))))))))))))).lower())))))))))))))))) for f in args.formats.split())))))))))))))))',')]:
    # Generate reports
        reports = {}}}}}}}}}}}}}}}
    
        logger.info())))))))))))))))"Generating reports...")
    for fmt in formats:
        if fmt == 'html':
            # Generate HTML reports
            compatibility_html = generate_compatibility_report())))))))))))))))compatibility_df, "html")
            performance_html = generate_performance_report())))))))))))))))performance_data, performance_charts, "html")
            regression_html = generate_regression_report())))))))))))))))regression_df, run_id, regression_chart, "html")
            
            # Save HTML reports
            if compatibility_html:
                compatibility_file = os.path.join())))))))))))))))args.output_dir, 'compatibility_matrix.html')
                with open())))))))))))))))compatibility_file, 'w') as f:
                    f.write())))))))))))))))compatibility_html)
                    reports[],'compatibility_html'] = compatibility_file
                    logger.info())))))))))))))))f"Saved compatibility matrix to {}}}}}}}}}}}}}}compatibility_file}")
            
            if performance_html:
                performance_file = os.path.join())))))))))))))))args.output_dir, 'performance_report.html')
                with open())))))))))))))))performance_file, 'w') as f:
                    f.write())))))))))))))))performance_html)
                    reports[],'performance_html'] = performance_file
                    logger.info())))))))))))))))f"Saved performance report to {}}}}}}}}}}}}}}performance_file}")
            
            if regression_html:
                regression_file = os.path.join())))))))))))))))args.output_dir, 'regression_report.html')
                with open())))))))))))))))regression_file, 'w') as f:
                    f.write())))))))))))))))regression_html)
                    reports[],'regression_html'] = regression_file
                    logger.info())))))))))))))))f"Saved regression report to {}}}}}}}}}}}}}}regression_file}")
            
        elif fmt == 'markdown':
            # Generate markdown reports
            compatibility_md = generate_compatibility_report())))))))))))))))compatibility_df, "markdown")
            regression_md = generate_regression_report())))))))))))))))regression_df, run_id, None, "markdown")
            
            # Save markdown reports
            if compatibility_md:
                compatibility_file = os.path.join())))))))))))))))args.output_dir, 'compatibility_matrix.md')
                with open())))))))))))))))compatibility_file, 'w') as f:
                    f.write())))))))))))))))compatibility_md)
                    reports[],'compatibility_md'] = compatibility_file
                    logger.info())))))))))))))))f"Saved compatibility matrix to {}}}}}}}}}}}}}}compatibility_file}")
            
            if regression_md:
                regression_file = os.path.join())))))))))))))))args.output_dir, 'regression_report.md')
                with open())))))))))))))))regression_file, 'w') as f:
                    f.write())))))))))))))))regression_md)
                    reports[],'regression_md'] = regression_file
                    logger.info())))))))))))))))f"Saved regression report to {}}}}}}}}}}}}}}regression_file}")
            
        elif fmt == 'json':
            # Generate JSON reports
            compatibility_json = json.dumps()))))))))))))))){}}}}}}}}}}}}}}
            "generated_at": datetime.datetime.now())))))))))))))))).isoformat())))))))))))))))),
            "models": compatibility_df.to_dict())))))))))))))))orient='records')
            }, indent=2)
            
            performance_json = json.dumps()))))))))))))))){}}}}}}}}}}}}}}
            "generated_at": datetime.datetime.now())))))))))))))))).isoformat())))))))))))))))),
            "throughput": performance_data.get())))))))))))))))'raw_throughput', pd.DataFrame()))))))))))))))))).to_dict())))))))))))))))orient='records'),
            "latency": performance_data.get())))))))))))))))'raw_latency', pd.DataFrame()))))))))))))))))).to_dict())))))))))))))))orient='records'),
            "memory": performance_data.get())))))))))))))))'raw_memory', pd.DataFrame()))))))))))))))))).to_dict())))))))))))))))orient='records')
            }, indent=2)
            
            regression_json = json.dumps()))))))))))))))){}}}}}}}}}}}}}}
            "generated_at": datetime.datetime.now())))))))))))))))).isoformat())))))))))))))))),
            "run_id": run_id,
            "regressions": regression_df.to_dict())))))))))))))))orient='records')
            }, indent=2)
            
            historical_json = json.dumps()))))))))))))))){}}}}}}}}}}}}}}
            "generated_at": datetime.datetime.now())))))))))))))))).isoformat())))))))))))))))),
            "days": args.days,
            "data": historical_df.to_dict())))))))))))))))orient='records')
            }, indent=2)
            
            # Save JSON reports
            compatibility_file = os.path.join())))))))))))))))args.output_dir, 'compatibility_matrix.json')
            with open())))))))))))))))compatibility_file, 'w') as f:
                f.write())))))))))))))))compatibility_json)
                reports[],'compatibility_json'] = compatibility_file
                logger.info())))))))))))))))f"Saved compatibility matrix to {}}}}}}}}}}}}}}compatibility_file}")
            
                performance_file = os.path.join())))))))))))))))args.output_dir, 'performance_report.json')
            with open())))))))))))))))performance_file, 'w') as f:
                f.write())))))))))))))))performance_json)
                reports[],'performance_json'] = performance_file
                logger.info())))))))))))))))f"Saved performance report to {}}}}}}}}}}}}}}performance_file}")
            
                regression_file = os.path.join())))))))))))))))args.output_dir, 'regression_report.json')
            with open())))))))))))))))regression_file, 'w') as f:
                f.write())))))))))))))))regression_json)
                reports[],'regression_json'] = regression_file
                logger.info())))))))))))))))f"Saved regression report to {}}}}}}}}}}}}}}regression_file}")
            
                historical_file = os.path.join())))))))))))))))args.output_dir, 'historical_data.json')
            with open())))))))))))))))historical_file, 'w') as f:
                f.write())))))))))))))))historical_json)
                reports[],'historical_json'] = historical_file
                logger.info())))))))))))))))f"Saved historical data to {}}}}}}}}}}}}}}historical_file}")
    
    # Generate index page
                logger.info())))))))))))))))"Generating index page...")
                index_file = generate_index_page())))))))))))))))
                args.output_dir,
                reports.get())))))))))))))))'compatibility_html'),
                reports.get())))))))))))))))'performance_html'),
                reports.get())))))))))))))))'regression_html'),
                historical_chart
                )
                logger.info())))))))))))))))f"Saved index page to {}}}}}}}}}}}}}}index_file}")
    
                logger.info())))))))))))))))"Dashboard generation complete!")
                logger.info())))))))))))))))f"Dashboard is available at: {}}}}}}}}}}}}}}os.path.abspath())))))))))))))))args.output_dir)}/index.html")

if __name__ == "__main__":
    main()))))))))))))))))