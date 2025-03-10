#!/usr/bin/env python
"""
Benchmark Database Analytics

This script provides advanced analytics on the benchmark database, including:
    - Performance trends over time
    - Hardware comparison across model families
    - Model comparison across hardware platforms
    - Performance prediction for untested configurations
    - Anomaly detection for performance regressions
    - Correlation analysis between model parameters and performance

Usage:
    python benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis performance-trends
    """

    import os
    import sys
    import json
    import argparse
    import logging
    import datetime
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use())))))))'Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import scipy.stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

# Add parent directory to path for module imports
    sys.path.append())))))))str())))))))Path())))))))__file__).parent.parent))

# Configure logging
    logging.basicConfig())))))))
    level=logging.INFO,
    format='%())))))))asctime)s - %())))))))name)s - %())))))))levelname)s - %())))))))message)s',
    handlers=[]]]]],,,,,logging.StreamHandler()))))))))],
    )
    logger = logging.getLogger())))))))"benchmark_analytics")

def parse_args())))))))):
    parser = argparse.ArgumentParser())))))))description="Advanced analytics on benchmark database")
    
    parser.add_argument())))))))"--db", type=str, default="./benchmark_db.duckdb", 
    help="Path to DuckDB database")
    
    # Analysis types
    analysis_group = parser.add_mutually_exclusive_group())))))))required=True)
    analysis_group.add_argument())))))))"--analysis", type=str, 
    choices=[]]]]],,,,,
    'performance-trends',
    'hardware-comparison',
    'model-comparison',
    'performance-prediction',
    'anomaly-detection',
    'correlation-analysis',
    'all'
    ],
    help="Type of analysis to perform")
    
    # Filters
    parser.add_argument())))))))"--model-family", type=str,
    help="Filter by model family ())))))))bert, t5, etc.)")
    parser.add_argument())))))))"--hardware-type", type=str,
    help="Filter by hardware type ())))))))cpu, cuda, etc.)")
    parser.add_argument())))))))"--batch-size", type=int,
    help="Filter by batch size")
    parser.add_argument())))))))"--since", type=str,
    help="Filter by date ())))))))YYYY-MM-DD)")
    parser.add_argument())))))))"--until", type=str,
    help="Filter by date ())))))))YYYY-MM-DD)")
    
    # Output options
    parser.add_argument())))))))"--output-dir", type=str, default="./analytics_output",
    help="Directory to save output files")
    parser.add_argument())))))))"--format", type=str, choices=[]]]]],,,,,'png', 'pdf', 'svg', 'html'], default='png',
    help="Output format for visualizations")
    parser.add_argument())))))))"--interactive", action="store_true",
    help="Generate interactive HTML visualizations ())))))))requires plotly)")
    
    # Advanced options
    parser.add_argument())))))))"--regression-threshold", type=float, default=0.1,
    help="Threshold for detecting performance regressions ())))))))as fraction)")
    parser.add_argument())))))))"--prediction-features", type=str,
    default="model_family,hardware_type,batch_size",
    help="Comma-separated list of features to use for prediction")
    parser.add_argument())))))))"--verbose", action="store_true",
    help="Enable verbose logging")
    
    return parser.parse_args()))))))))

def connect_to_db())))))))db_path):
    """Connect to the DuckDB database"""
    if not os.path.exists())))))))db_path):
        logger.error())))))))f"Database file not found: {}}}}}}}}}}}db_path}")
        sys.exit())))))))1)
        
    try:
        conn = duckdb.connect())))))))db_path)
        return conn
    except Exception as e:
        logger.error())))))))f"Error connecting to database: {}}}}}}}}}}}e}")
        sys.exit())))))))1)

def get_performance_data())))))))conn, args):
    """Get performance data from the database with filters"""
    # Build query with filters
    query = """
    SELECT 
    pr.result_id,
    m.model_name,
    m.model_family,
    m.modality,
    hp.hardware_type,
    hp.device_name,
    pr.test_case,
    pr.batch_size,
    pr.precision,
    pr.total_time_seconds,
    pr.average_latency_ms,
    pr.throughput_items_per_second,
    pr.memory_peak_mb,
    pr.created_at
    FROM 
    performance_results pr
    JOIN 
    models m ON pr.model_id = m.model_id
    JOIN 
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    """
    
    params = []]]]],,,,,]
    where_clauses = []]]]],,,,,]
    
    # Apply filters
    if args.model_family:
        where_clauses.append())))))))"m.model_family = ?")
        params.append())))))))args.model_family)
    
    if args.hardware_type:
        where_clauses.append())))))))"hp.hardware_type = ?")
        params.append())))))))args.hardware_type)
    
    if args.batch_size:
        where_clauses.append())))))))"pr.batch_size = ?")
        params.append())))))))args.batch_size)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime())))))))args.since, "%Y-%m-%d")
            where_clauses.append())))))))"pr.created_at >= ?")
            params.append())))))))since_date)
        except ValueError:
            logger.warning())))))))f"Invalid date format for --since: {}}}}}}}}}}}args.since}, expected YYYY-MM-DD")
    
    if args.until:
        try:
            until_date = datetime.datetime.strptime())))))))args.until, "%Y-%m-%d")
            where_clauses.append())))))))"pr.created_at <= ?")
            params.append())))))))until_date)
        except ValueError:
            logger.warning())))))))f"Invalid date format for --until: {}}}}}}}}}}}args.until}, expected YYYY-MM-DD")
    
    if where_clauses:
        query += " WHERE " + " AND ".join())))))))where_clauses)
    
        query += " ORDER BY pr.created_at"
    
    # Execute query
        df = conn.execute())))))))query, params).fetchdf()))))))))
    
    # Check if we got any data:
    if df.empty:
        logger.warning())))))))"No performance data found matching the criteria")
        return None
    
            return df

def analyze_performance_trends())))))))df, args):
    """Analyze performance trends over time"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing performance trends over time")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Extract date from timestamps for better grouping
    df[]]]]],,,,,'date'] = df[]]]]],,,,,'created_at'].dt.date
    
    # Group by model family, hardware type, and date
    group_cols = []]]]],,,,,'model_family', 'hardware_type', 'date']
    metrics = []]]]],,,,,'average_latency_ms', 'throughput_items_per_second', 'memory_peak_mb']
    
    # Get unique model families and hardware types
    model_families = df[]]]]],,,,,'model_family'].unique()))))))))
    hardware_types = df[]]]]],,,,,'hardware_type'].unique()))))))))
    
    # Create separate trend plots for each model family
    for model_family in model_families:
        for metric in metrics:
            # Filter data for this model family
            family_df = df[]]]]],,,,,df[]]]]],,,,,'model_family'] == model_family]
            
            # Create a plot
            plt.figure())))))))figsize=())))))))12, 6))
            
            # Plot each hardware type as a separate line
            for hw_type in hardware_types:
                hw_df = family_df[]]]]],,,,,family_df[]]]]],,,,,'hardware_type'] == hw_type]
                
                if not hw_df.empty:
                    # Group by date and get mean of the metric
                    trend_data = hw_df.groupby())))))))'date')[]]]]],,,,,metric].mean())))))))).reset_index()))))))))
                    
                    if not trend_data.empty:
                        plt.plot())))))))trend_data[]]]]],,,,,'date'], trend_data[]]]]],,,,,metric], marker='o', label=hw_type)
            
                        plt.title())))))))f'{}}}}}}}}}}}model_family.upper()))))))))} - {}}}}}}}}}}}metric} Trend')
                        plt.xlabel())))))))'Date')
                        plt.ylabel())))))))metric)
                        plt.legend()))))))))
                        plt.grid())))))))True, alpha=0.3)
            
            # Format date axis
                        plt.gcf())))))))).autofmt_xdate()))))))))
            
            # Adjust y axis to start from 0 for throughput
            if metric == 'throughput_items_per_second':
                plt.ylim())))))))bottom=0)
            
            # Save the plot
                metric_name = metric.replace())))))))'_', '-')
                output_path = os.path.join())))))))args.output_dir, f'trend_{}}}}}}}}}}}model_family}_{}}}}}}}}}}}metric_name}.{}}}}}}}}}}}args.format}')
                plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
                plt.close()))))))))
            
                logger.info())))))))f"Saved trend analysis for {}}}}}}}}}}}model_family} {}}}}}}}}}}}metric} to {}}}}}}}}}}}output_path}")
    
    # Create a summary trend plot with all model families for throughput
                plt.figure())))))))figsize=())))))))14, 8))
    
    # Use a different color palette with more distinct colors
                palette = sns.color_palette())))))))"husl", len())))))))model_families) * len())))))))hardware_types))
                color_idx = 0
    
    for model_family in model_families:
        for hw_type in hardware_types:
            filtered_df = df[]]]]],,,,,())))))))df[]]]]],,,,,'model_family'] == model_family) & ())))))))df[]]]]],,,,,'hardware_type'] == hw_type)]
            
            if not filtered_df.empty:
                # Group by date and get mean throughput
                trend_data = filtered_df.groupby())))))))'date')[]]]]],,,,,'throughput_items_per_second'].mean())))))))).reset_index()))))))))
                
                if not trend_data.empty:
                    plt.plot())))))))trend_data[]]]]],,,,,'date'], trend_data[]]]]],,,,,'throughput_items_per_second'], 
                    marker='o', label=f'{}}}}}}}}}}}model_family}-{}}}}}}}}}}}hw_type}',
                    color=palette[]]]]],,,,,color_idx])
                    color_idx += 1
    
                    plt.title())))))))'Throughput Trends by Model Family and Hardware')
                    plt.xlabel())))))))'Date')
                    plt.ylabel())))))))'Throughput ())))))))items/sec)')
                    plt.legend())))))))bbox_to_anchor=())))))))1.05, 1), loc='upper left')
                    plt.grid())))))))True, alpha=0.3)
                    plt.gcf())))))))).autofmt_xdate()))))))))
    
    # Save the summary plot
                    output_path = os.path.join())))))))args.output_dir, f'trend_all_throughput.{}}}}}}}}}}}args.format}')
                    plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
                    plt.close()))))))))
    
                    logger.info())))))))f"Saved combined throughput trend analysis to {}}}}}}}}}}}output_path}")
    
    # Generate interactive visualization if requested::::::
    if args.interactive:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create a DataFrame for plotly with all necessary columns
            plot_df = df.groupby())))))))[]]]]],,,,,'date', 'model_family', 'hardware_type'])[]]]]],,,,,'throughput_items_per_second'].mean())))))))).reset_index()))))))))
            
            # Create the interactive plot
            fig = px.line())))))))plot_df, x='date', y='throughput_items_per_second', 
            color='model_family', line_dash='hardware_type',
            labels={}}}}}}}}}}}
            'date': 'Date',
            'throughput_items_per_second': 'Throughput ())))))))items/sec)',
            'model_family': 'Model Family',
            'hardware_type': 'Hardware Type'
            },
            title='Interactive Throughput Trends by Model Family and Hardware')
            
            # Add hover data
            fig.update_traces())))))))mode='lines+markers')
            
            # Save as HTML
            output_path = os.path.join())))))))args.output_dir, 'interactive_trend_all_throughput.html')
            fig.write_html())))))))output_path)
            logger.info())))))))f"Saved interactive throughput trend visualization to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def analyze_hardware_comparison())))))))df, args):
    """Compare performance across hardware platforms"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing hardware performance comparison")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Get unique model families and hardware types
    model_families = df[]]]]],,,,,'model_family'].unique()))))))))
    hardware_types = df[]]]]],,,,,'hardware_type'].unique()))))))))
    
    # Create a figure for the hardware comparison
    plt.figure())))))))figsize=())))))))14, 8))
    
    # Group data by model family and hardware type, calculate mean throughput
    grouped_df = df.groupby())))))))[]]]]],,,,,'model_family', 'hardware_type'])[]]]]],,,,,'throughput_items_per_second'].mean())))))))).reset_index()))))))))
    
    # Pivot the data for plotting
    pivot_df = grouped_df.pivot())))))))index='model_family', columns='hardware_type', values='throughput_items_per_second')
    
    # Create the plot - if there are more than 1 model family, use a grouped bar chart:
    if len())))))))model_families) > 1:
        ax = pivot_df.plot())))))))kind='bar', figsize=())))))))14, 8))
        plt.title())))))))'Average Throughput by Model Family and Hardware')
        plt.xlabel())))))))'Model Family')
        plt.ylabel())))))))'Throughput ())))))))items/sec)')
        plt.legend())))))))title='Hardware Type')
        plt.grid())))))))True, alpha=0.3, axis='y')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label())))))))container, fmt='%.1f', fontsize=8)
    else:
        # If only one model family, create a simpler bar chart
        hardware_comparison = grouped_df.set_index())))))))'hardware_type')[]]]]],,,,,'throughput_items_per_second']
        hardware_comparison.plot())))))))kind='bar', figsize=())))))))10, 6))
        plt.title())))))))f'Average Throughput for {}}}}}}}}}}}model_families[]]]]],,,,,0]} by Hardware')
        plt.xlabel())))))))'Hardware Type')
        plt.ylabel())))))))'Throughput ())))))))items/sec)')
        plt.grid())))))))True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate())))))))hardware_comparison):
            plt.text())))))))i, v + 0.5, f'{}}}}}}}}}}}v:.1f}', ha='center')
    
    # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'hardware_comparison_throughput.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved hardware comparison analysis to {}}}}}}}}}}}output_path}")
    
    # Create a heatmap of hardware performance
            plt.figure())))))))figsize=())))))))12, 8))
    
    # Create a pivot table normalized by the best performance for each model family
    # This shows relative performance of each hardware type
            normalized_pivot = pivot_df.copy()))))))))
    for idx in normalized_pivot.index:
        max_val = normalized_pivot.loc[]]]]],,,,,idx].max()))))))))
        if max_val > 0:
            normalized_pivot.loc[]]]]],,,,,idx] = normalized_pivot.loc[]]]]],,,,,idx] / max_val
    
    # Create the heatmap
            sns.heatmap())))))))normalized_pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.5, fmt='.2f', cbar_kws={}}}}}}}}}}}'label': 'Relative Performance'})
    
            plt.title())))))))'Relative Throughput Performance by Hardware ())))))))1.0 = Best)')
            plt.ylabel())))))))'Model Family')
            plt.xlabel())))))))'Hardware Type')
    
    # Save the heatmap
            output_path = os.path.join())))))))args.output_dir, f'hardware_heatmap.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved hardware performance heatmap to {}}}}}}}}}}}output_path}")
    
    # Generate interactive visualization if requested::::::
    if args.interactive:
        try:
            import plotly.express as px
            
            # Create a DataFrame for plotly with all necessary columns
            fig = px.bar())))))))grouped_df, x='model_family', y='throughput_items_per_second',
            color='hardware_type', barmode='group',
            labels={}}}}}}}}}}}
            'model_family': 'Model Family',
            'throughput_items_per_second': 'Throughput ())))))))items/sec)',
            'hardware_type': 'Hardware Type'
            },
            title='Interactive Hardware Performance Comparison by Model Family')
            
            # Add hover data
            fig.update_traces())))))))
            hovertemplate="<b>%{}}}}}}}}}}}x}</b><br>Hardware: %{}}}}}}}}}}}color}<br>Throughput: %{}}}}}}}}}}}y:.2f} items/sec"
            )
            
            # Save as HTML
            output_path = os.path.join())))))))args.output_dir, 'interactive_hardware_comparison.html')
            fig.write_html())))))))output_path)
            logger.info())))))))f"Saved interactive hardware comparison to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def analyze_model_comparison())))))))df, args):
    """Compare performance across model families"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing model family performance comparison")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Get unique batch sizes
    batch_sizes = df[]]]]],,,,,'batch_size'].unique()))))))))
    batch_sizes.sort()))))))))
    
    # Create metrics comparison across model families for each hardware type
    for hardware_type in df[]]]]],,,,,'hardware_type'].unique())))))))):
        # Filter data for this hardware type
        hw_df = df[]]]]],,,,,df[]]]]],,,,,'hardware_type'] == hardware_type]
        
        # Create throughput comparison by batch size
        plt.figure())))))))figsize=())))))))14, 8))
        
        # Group by model family and batch size
        for model_family in hw_df[]]]]],,,,,'model_family'].unique())))))))):
            family_df = hw_df[]]]]],,,,,hw_df[]]]]],,,,,'model_family'] == model_family]
            
            # Group by batch size and calculate mean throughput
            batch_throughput = family_df.groupby())))))))'batch_size')[]]]]],,,,,'throughput_items_per_second'].mean()))))))))
            
            # Plot a line for each model family
            plt.plot())))))))batch_throughput.index, batch_throughput.values, marker='o', label=model_family)
        
            plt.title())))))))f'Throughput by Batch Size for {}}}}}}}}}}}hardware_type.upper()))))))))}')
            plt.xlabel())))))))'Batch Size')
            plt.ylabel())))))))'Throughput ())))))))items/sec)')
            plt.legend()))))))))
            plt.grid())))))))True, alpha=0.3)
        
        # Set x ticks to match batch sizes
            plt.xticks())))))))batch_sizes)
        
        # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'model_comparison_{}}}}}}}}}}}hardware_type}_throughput.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
        
            logger.info())))))))f"Saved model comparison for {}}}}}}}}}}}hardware_type} to {}}}}}}}}}}}output_path}")
        
        # Create memory usage comparison
            plt.figure())))))))figsize=())))))))14, 8))
        
        # Group by model family and calculate mean memory usage
        for model_family in hw_df[]]]]],,,,,'model_family'].unique())))))))):
            family_df = hw_df[]]]]],,,,,hw_df[]]]]],,,,,'model_family'] == model_family]
            
            # Group by batch size and calculate mean memory usage
            batch_memory = family_df.groupby())))))))'batch_size')[]]]]],,,,,'memory_peak_mb'].mean()))))))))
            
            # Plot a line for each model family
            plt.plot())))))))batch_memory.index, batch_memory.values, marker='o', label=model_family)
        
            plt.title())))))))f'Memory Usage by Batch Size for {}}}}}}}}}}}hardware_type.upper()))))))))}')
            plt.xlabel())))))))'Batch Size')
            plt.ylabel())))))))'Memory Peak ())))))))MB)')
            plt.legend()))))))))
            plt.grid())))))))True, alpha=0.3)
        
        # Set x ticks to match batch sizes
            plt.xticks())))))))batch_sizes)
        
        # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'model_comparison_{}}}}}}}}}}}hardware_type}_memory.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
        
            logger.info())))))))f"Saved memory usage comparison for {}}}}}}}}}}}hardware_type} to {}}}}}}}}}}}output_path}")
    
    # Generate scaling efficiency plot - how throughput scales with batch size
    for hardware_type in df[]]]]],,,,,'hardware_type'].unique())))))))):
        plt.figure())))))))figsize=())))))))14, 8))
        
        hw_df = df[]]]]],,,,,df[]]]]],,,,,'hardware_type'] == hardware_type]
        
        for model_family in hw_df[]]]]],,,,,'model_family'].unique())))))))):
            family_df = hw_df[]]]]],,,,,hw_df[]]]]],,,,,'model_family'] == model_family]
            
            # Get baseline throughput ())))))))batch size = 1)
            baseline_df = family_df[]]]]],,,,,family_df[]]]]],,,,,'batch_size'] == 1]
            if baseline_df.empty:
            continue
                
            baseline_throughput = baseline_df[]]]]],,,,,'throughput_items_per_second'].mean()))))))))
            
            # Calculate scaling efficiency for each batch size
            scaling_data = []]]]],,,,,]
            for batch_size in sorted())))))))family_df[]]]]],,,,,'batch_size'].unique()))))))))):
                batch_df = family_df[]]]]],,,,,family_df[]]]]],,,,,'batch_size'] == batch_size]
                if batch_df.empty:
                continue
                    
                batch_throughput = batch_df[]]]]],,,,,'throughput_items_per_second'].mean()))))))))
                
                # Efficiency = actual_throughput / ())))))))baseline_throughput * batch_size)
                efficiency = batch_throughput / ())))))))baseline_throughput * batch_size)
                
                scaling_data.append())))))))())))))))batch_size, efficiency))
            
            if scaling_data:
                batch_sizes, efficiencies = zip())))))))*scaling_data)
                plt.plot())))))))batch_sizes, efficiencies, marker='o', label=model_family)
        
                plt.title())))))))f'Batch Scaling Efficiency for {}}}}}}}}}}}hardware_type.upper()))))))))}')
                plt.xlabel())))))))'Batch Size')
                plt.ylabel())))))))'Scaling Efficiency ())))))))1.0 = linear scaling)')
                plt.axhline())))))))y=1.0, color='gray', linestyle='--', alpha=0.7)
                plt.legend()))))))))
                plt.grid())))))))True, alpha=0.3)
        
        # Set y-axis to start from 0
                plt.ylim())))))))bottom=0)
        
        # Save the plot
                output_path = os.path.join())))))))args.output_dir, f'scaling_efficiency_{}}}}}}}}}}}hardware_type}.{}}}}}}}}}}}args.format}')
                plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
                plt.close()))))))))
        
                logger.info())))))))f"Saved scaling efficiency analysis for {}}}}}}}}}}}hardware_type} to {}}}}}}}}}}}output_path}")
    
    # Generate interactive visualization if requested::::::
    if args.interactive:
        try:
            import plotly.express as px
            
            # Create a DataFrame with scaling efficiency data
            scaling_data = []]]]],,,,,]
            
            for hardware_type in df[]]]]],,,,,'hardware_type'].unique())))))))):
                hw_df = df[]]]]],,,,,df[]]]]],,,,,'hardware_type'] == hardware_type]
                
                for model_family in hw_df[]]]]],,,,,'model_family'].unique())))))))):
                    family_df = hw_df[]]]]],,,,,hw_df[]]]]],,,,,'model_family'] == model_family]
                    
                    # Get baseline throughput ())))))))batch size = 1)
                    baseline_df = family_df[]]]]],,,,,family_df[]]]]],,,,,'batch_size'] == 1]
                    if baseline_df.empty:
                    continue
                        
                    baseline_throughput = baseline_df[]]]]],,,,,'throughput_items_per_second'].mean()))))))))
                    
                    # Calculate scaling efficiency for each batch size
                    for batch_size in sorted())))))))family_df[]]]]],,,,,'batch_size'].unique()))))))))):
                        batch_df = family_df[]]]]],,,,,family_df[]]]]],,,,,'batch_size'] == batch_size]
                        if batch_df.empty:
                        continue
                            
                        batch_throughput = batch_df[]]]]],,,,,'throughput_items_per_second'].mean()))))))))
                        
                        # Efficiency = actual_throughput / ())))))))baseline_throughput * batch_size)
                        if baseline_throughput > 0 and batch_size > 0:
                            efficiency = batch_throughput / ())))))))baseline_throughput * batch_size)
                            
                            scaling_data.append()))))))){}}}}}}}}}}}
                            'hardware_type': hardware_type,
                            'model_family': model_family,
                            'batch_size': batch_size,
                            'efficiency': efficiency,
                            'throughput': batch_throughput
                            })
            
            if scaling_data:
                scaling_df = pd.DataFrame())))))))scaling_data)
                
                # Create interactive plot
                fig = px.line())))))))scaling_df, x='batch_size', y='efficiency', 
                color='model_family', line_dash='hardware_type',
                hover_data=[]]]]],,,,,'throughput'],
                labels={}}}}}}}}}}}
                'batch_size': 'Batch Size',
                'efficiency': 'Scaling Efficiency',
                'model_family': 'Model Family',
                'hardware_type': 'Hardware Type',
                'throughput': 'Throughput ())))))))items/sec)'
                },
                title='Interactive Batch Scaling Efficiency Analysis')
                
                # Add a horizontal line at y=1.0
                fig.add_shape())))))))
                type='line',
                x0=scaling_df[]]]]],,,,,'batch_size'].min())))))))),
                y0=1.0,
                x1=scaling_df[]]]]],,,,,'batch_size'].max())))))))),
                y1=1.0,
                line=dict())))))))
                color='gray',
                dash='dash',
                )
                )
                
                # Add hover data
                fig.update_traces())))))))mode='lines+markers')
                
                # Save as HTML
                output_path = os.path.join())))))))args.output_dir, 'interactive_scaling_efficiency.html')
                fig.write_html())))))))output_path)
                logger.info())))))))f"Saved interactive scaling efficiency visualization to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def analyze_performance_prediction())))))))df, args):
    """Predict performance for untested configurations"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing performance prediction for untested configurations")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Prepare data for prediction
    # Select relevant features specified in args.prediction_features
    features = []]]]],,,,,f.strip())))))))) for f in args.prediction_features.split())))))))',')]:
    # Verify all features exist in the dataframe
    missing_features = []]]]],,,,,f for f in features if f not in df.columns]:
    if missing_features:
        logger.warning())))))))f"Missing features: {}}}}}}}}}}}missing_features}")
        features = []]]]],,,,,f for f in features if f in df.columns]
        :
    if not features:
        logger.error())))))))"No valid features available for prediction")
            return
    
    # Add batch_size if not already included ())))))))important for predictions):
    if 'batch_size' not in features:
        features.append())))))))'batch_size')
    
    # Target variables to predict
        targets = []]]]],,,,,'throughput_items_per_second', 'average_latency_ms', 'memory_peak_mb']
    
    # Prepare categorical and numerical features
        categorical_features = []]]]],,,,,]
        numerical_features = []]]]],,,,,]
    
    for feature in features:
        if df[]]]]],,,,,feature].dtype == 'object' or df[]]]]],,,,,feature].dtype == 'category':
            categorical_features.append())))))))feature)
        else:
            numerical_features.append())))))))feature)
    
            prediction_results = {}}}}}}}}}}}}
    
    for target in targets:
        # Prepare complete dataset
        X = df[]]]]],,,,,features].copy()))))))))
        y = df[]]]]],,,,,target].copy()))))))))
        
        # Handle missing values
        X = X.fillna())))))))X.mode())))))))).iloc[]]]]],,,,,0])
        y = y.fillna())))))))y.mean())))))))))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split())))))))X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer())))))))
        transformers=[]]]]],,,,,
        ())))))))'num', Pipeline())))))))[]]]]],,,,,
        ())))))))'imputer', SimpleImputer())))))))strategy='mean')),
        ())))))))'scaler', StandardScaler())))))))))
        ]), numerical_features),
        ())))))))'cat', Pipeline())))))))[]]]]],,,,,
        ())))))))'imputer', SimpleImputer())))))))strategy='most_frequent')),
        ())))))))'onehot', OneHotEncoder())))))))handle_unknown='ignore'))
        ]), categorical_features)
        ]
        )
        
        # Create and train model
        model = Pipeline())))))))[]]]]],,,,,
        ())))))))'preprocessor', preprocessor),
        ())))))))'regressor', RandomForestRegressor())))))))n_estimators=100, random_state=42))
        ])
        
        model.fit())))))))X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict())))))))X_test)
        mse = mean_squared_error())))))))y_test, y_pred)
        r2 = r2_score())))))))y_test, y_pred)
        
        logger.info())))))))f"Model performance for {}}}}}}}}}}}target}: MSE={}}}}}}}}}}}mse:.4f}, R²={}}}}}}}}}}}r2:.4f}")
        
        # Store results
        prediction_results[]]]]],,,,,target] = {}}}}}}}}}}}
        'mse': mse,
        'r2': r2,
        'model': model,
        'feature_importance': None  # Will be set below
        }
        
        # Get feature importance
        if isinstance())))))))model[]]]]],,,,,'regressor'], RandomForestRegressor):
            # For random forest, extract feature names after preprocessing
            preprocessor = model[]]]]],,,,,'preprocessor']
            cat_features = preprocessor.transformers_[]]]]],,,,,1][]]]]],,,,,1][]]]]],,,,,'onehot'].get_feature_names_out())))))))categorical_features)
            all_features = numerical_features + list())))))))cat_features)
            
            # Extract feature importance
            feature_importance = model[]]]]],,,,,'regressor'].feature_importances_
            
            # Check if lengths match:
            if len())))))))all_features) == len())))))))feature_importance):
                # Create a dataframe with feature importance
                importance_df = pd.DataFrame()))))))){}}}}}}}}}}}
                'feature': all_features,
                'importance': feature_importance
                }).sort_values())))))))'importance', ascending=False)
                
                prediction_results[]]]]],,,,,target][]]]]],,,,,'feature_importance'] = importance_df
                
                # Plot feature importance
                plt.figure())))))))figsize=())))))))10, 8))
                sns.barplot())))))))x='importance', y='feature', data=importance_df.head())))))))15))
                plt.title())))))))f'Feature Importance for {}}}}}}}}}}}target} Prediction')
                plt.tight_layout()))))))))
                
                # Save the plot
                output_path = os.path.join())))))))args.output_dir, f'feature_importance_{}}}}}}}}}}}target}.{}}}}}}}}}}}args.format}')
                plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
                plt.close()))))))))
                
                logger.info())))))))f"Saved feature importance for {}}}}}}}}}}}target} prediction to {}}}}}}}}}}}output_path}")
    
    # Generate prediction summary for all hardware-model-batch size combinations
    # Create a grid of possible combinations using existing values
                model_families = df[]]]]],,,,,'model_family'].unique()))))))))
                hardware_types = df[]]]]],,,,,'hardware_type'].unique()))))))))
                batch_sizes = sorted())))))))df[]]]]],,,,,'batch_size'].unique())))))))))
    
    # Create all possible combinations
                combinations = []]]]],,,,,]
    for model_family in model_families:
        for hardware_type in hardware_types:
            for batch_size in batch_sizes:
                combinations.append()))))))){}}}}}}}}}}}
                'model_family': model_family,
                'hardware_type': hardware_type,
                'batch_size': batch_size
                })
    
                prediction_grid = pd.DataFrame())))))))combinations)
    
    # Add additional required features to the prediction grid
    for feature in features:
        if feature not in prediction_grid.columns:
            # Use the most common value for each combination
            for i, row in prediction_grid.iterrows())))))))):
                subset = df[]]]]],,,,,())))))))df[]]]]],,,,,'model_family'] == row[]]]]],,,,,'model_family']) & 
                ())))))))df[]]]]],,,,,'hardware_type'] == row[]]]]],,,,,'hardware_type'])]
                if not subset.empty and feature in subset.columns:
                    # Use most common value for this combination
                    prediction_grid.at[]]]]],,,,,i, feature] = subset[]]]]],,,,,feature].mode())))))))).iloc[]]]]],,,,,0]
                else:
                    # Use overall most common value
                    prediction_grid.at[]]]]],,,,,i, feature] = df[]]]]],,,,,feature].mode())))))))).iloc[]]]]],,,,,0]
    
    # Make predictions for each target
    for target, result_data in prediction_results.items())))))))):
        model = result_data[]]]]],,,,,'model']
        prediction_grid[]]]]],,,,,target] = model.predict())))))))prediction_grid[]]]]],,,,,features])
    
    # Save the prediction grid
        output_path = os.path.join())))))))args.output_dir, 'performance_predictions.csv')
        prediction_grid.to_csv())))))))output_path, index=False)
        logger.info())))))))f"Saved performance predictions to {}}}}}}}}}}}output_path}")
    
    # Create heatmap of predicted throughput
        plt.figure())))))))figsize=())))))))14, 8))
    
    # Create pivot table of predicted throughput
        pivot_df = prediction_grid.pivot_table())))))))
        values='throughput_items_per_second', 
        index='model_family',
        columns=[]]]]],,,,,'hardware_type', 'batch_size']
        )
    
    # Create heatmap
        sns.heatmap())))))))pivot_df, annot=True, fmt='.1f', cmap='viridis')
        plt.title())))))))'Predicted Throughput by Model Family, Hardware Type, and Batch Size')
        plt.tight_layout()))))))))
    
    # Save the plot
        output_path = os.path.join())))))))args.output_dir, f'predicted_throughput_heatmap.{}}}}}}}}}}}args.format}')
        plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
        plt.close()))))))))
    
        logger.info())))))))f"Saved predicted throughput heatmap to {}}}}}}}}}}}output_path}")
    
    # Generate interactive visualization if requested::::::
    if args.interactive:
        try:
            import plotly.express as px
            
            # Create interactive heatmap for throughput predictions
            fig = px.density_heatmap())))))))prediction_grid, 
            x='batch_size',
            y='model_family',
            z='throughput_items_per_second',
            facet_col='hardware_type',
            labels={}}}}}}}}}}}
            'batch_size': 'Batch Size',
            'model_family': 'Model Family',
            'throughput_items_per_second': 'Predicted Throughput',
            'hardware_type': 'Hardware Type'
            },
            title='Interactive Predicted Throughput Heatmap')
            
            # Save as HTML
            output_path = os.path.join())))))))args.output_dir, 'interactive_predicted_throughput.html')
            fig.write_html())))))))output_path)
            logger.info())))))))f"Saved interactive throughput prediction to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def analyze_anomaly_detection())))))))df, args):
    """Detect anomalies and performance regressions"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing for performance anomalies and regressions")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Convert timestamp to date for grouping
    df[]]]]],,,,,'date'] = df[]]]]],,,,,'created_at'].dt.date
    
    # Group data by model family, hardware type, batch size
    group_cols = []]]]],,,,,'model_family', 'hardware_type', 'batch_size', 'date']
    metrics = []]]]],,,,,'throughput_items_per_second', 'average_latency_ms', 'memory_peak_mb']
    
    # Dictionary to store detected anomalies
    anomalies = []]]]],,,,,]
    
    # Process each combination
    for name, group in df.groupby())))))))[]]]]],,,,,'model_family', 'hardware_type', 'batch_size']):
        model_family, hardware_type, batch_size = name
        
        # Skip if not enough data points for a trend:
        if len())))))))group) < 3:
        continue
            
        # Get date-ordered metrics
        date_metrics = group.sort_values())))))))'date').groupby())))))))'date')[]]]]],,,,,metrics].mean()))))))))
        
        # For each metric, detect anomalies using different methods
        for metric in metrics:
            values = date_metrics[]]]]],,,,,metric].values
            dates = date_metrics.index
            
            if len())))))))values) < 3:
            continue
                
            # 1. Moving average method
            window_size = min())))))))3, len())))))))values)-1)
            moving_avg = np.convolve())))))))values, np.ones())))))))window_size)/window_size, mode='valid')
            
            # Add padding to match original length
            padding = np.array())))))))[]]]]],,,,,values[]]]]],,,,,0]] * ())))))))len())))))))values) - len())))))))moving_avg)))
            moving_avg = np.concatenate())))))))())))))))padding, moving_avg))
            
            # Calculate deviations from moving average
            deviations = values - moving_avg
            
            # Detect significant deviations ())))))))using threshold from args)
            threshold = args.regression_threshold * np.mean())))))))values)
            
            for i in range())))))))1, len())))))))values)):
                is_regression = False
                
                # Check if value significantly decreased from moving average ())))))))for throughput):
                if metric == 'throughput_items_per_second' and deviations[]]]]],,,,,i] < -threshold:
                    is_regression = True
                
                # Check if value significantly increased from moving average ())))))))for latency and memory):
                elif metric in []]]]],,,,,'average_latency_ms', 'memory_peak_mb'] and deviations[]]]]],,,,,i] > threshold:
                    is_regression = True
                
                if is_regression:
                    # Calculate percent change
                    percent_change = ())))))))values[]]]]],,,,,i] - moving_avg[]]]]],,,,,i]) / moving_avg[]]]]],,,,,i] * 100
                    
                    anomalies.append()))))))){}}}}}}}}}}}
                    'model_family': model_family,
                    'hardware_type': hardware_type,
                    'batch_size': batch_size,
                    'metric': metric,
                    'date': dates[]]]]],,,,,i],
                    'value': values[]]]]],,,,,i],
                    'expected': moving_avg[]]]]],,,,,i],
                    'deviation': deviations[]]]]],,,,,i],
                    'percent_change': percent_change,
                    'detection_method': 'moving_average'
                    })
    
    # Create anomaly summary
    if anomalies:
        anomaly_df = pd.DataFrame())))))))anomalies)
        
        # Save the anomalies to CSV
        output_path = os.path.join())))))))args.output_dir, 'anomalies.csv')
        anomaly_df.to_csv())))))))output_path, index=False)
        logger.info())))))))f"Saved {}}}}}}}}}}}len())))))))anomalies)} detected anomalies to {}}}}}}}}}}}output_path}")
        
        # Create anomaly visualization
        plt.figure())))))))figsize=())))))))14, 8))
        
        # Group anomalies by model family and hardware type
        anomaly_counts = anomaly_df.groupby())))))))[]]]]],,,,,'model_family', 'hardware_type']).size())))))))).reset_index()))))))))
        anomaly_counts.columns = []]]]],,,,,'model_family', 'hardware_type', 'count']
        
        # Create pivot table
        pivot_df = anomaly_counts.pivot())))))))index='model_family', columns='hardware_type', values='count').fillna())))))))0)
        
        # Create heatmap
        sns.heatmap())))))))pivot_df, annot=True, fmt='g', cmap='Reds')
        plt.title())))))))'Anomaly Counts by Model Family and Hardware Type')
        plt.tight_layout()))))))))
        
        # Save the plot
        output_path = os.path.join())))))))args.output_dir, f'anomaly_heatmap.{}}}}}}}}}}}args.format}')
        plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
        plt.close()))))))))
        
        logger.info())))))))f"Saved anomaly heatmap to {}}}}}}}}}}}output_path}")
        
        # Plot some example anomalies
        for metric in metrics:
            metric_anomalies = anomaly_df[]]]]],,,,,anomaly_df[]]]]],,,,,'metric'] == metric]
            
            if not metric_anomalies.empty:
                # Take top 5 anomalies by percent change
                top_anomalies = metric_anomalies.sort_values())))))))'percent_change', ascending=())))))))metric == 'throughput_items_per_second')).head())))))))5)
                
                for _, anomaly in top_anomalies.iterrows())))))))):
                    # Get the full time series for this configuration
                    config_df = df[]]]]],,,,,())))))))df[]]]]],,,,,'model_family'] == anomaly[]]]]],,,,,'model_family']) & 
                    ())))))))df[]]]]],,,,,'hardware_type'] == anomaly[]]]]],,,,,'hardware_type']) &
                    ())))))))df[]]]]],,,,,'batch_size'] == anomaly[]]]]],,,,,'batch_size'])]
                    
                    # Group by date
                    date_series = config_df.groupby())))))))'date')[]]]]],,,,,metric].mean()))))))))
                    
                    # Plot the time series
                    plt.figure())))))))figsize=())))))))10, 6))
                    plt.plot())))))))date_series.index, date_series.values, marker='o', label='Actual')
                    
                    # Highlight the anomaly
                    anomaly_date = anomaly[]]]]],,,,,'date']
                    if anomaly_date in date_series.index:
                        anomaly_value = date_series[]]]]],,,,,anomaly_date]
                        plt.scatter())))))))[]]]]],,,,,anomaly_date], []]]]],,,,,anomaly_value], c='red', s=100, zorder=5, label='Anomaly')
                        
                        # Add text annotation
                        plt.annotate())))))))f"{}}}}}}}}}}}anomaly[]]]]],,,,,'percent_change']:.1f}%", 
                        ())))))))anomaly_date, anomaly_value),
                        xytext=())))))))0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontweight='bold')
                    
                    # Create title with configuration details
                        title = f"Anomaly: {}}}}}}}}}}}anomaly[]]]]],,,,,'model_family']} on {}}}}}}}}}}}anomaly[]]]]],,,,,'hardware_type']} ())))))))Batch={}}}}}}}}}}}anomaly[]]]]],,,,,'batch_size']})"
                        plt.title())))))))title)
                        plt.xlabel())))))))'Date')
                        plt.ylabel())))))))metric)
                        plt.grid())))))))True, alpha=0.3)
                        plt.legend()))))))))
                    
                    # Format date axis
                        plt.gcf())))))))).autofmt_xdate()))))))))
                    
                    # Generate filename
                        filename = f"anomaly_{}}}}}}}}}}}anomaly[]]]]],,,,,'model_family']}_{}}}}}}}}}}}anomaly[]]]]],,,,,'hardware_type']}_{}}}}}}}}}}}anomaly[]]]]],,,,,'batch_size']}_{}}}}}}}}}}}metric}.{}}}}}}}}}}}args.format}"
                        output_path = os.path.join())))))))args.output_dir, filename)
                        plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
                        plt.close()))))))))
                    
                        logger.info())))))))f"Saved anomaly plot to {}}}}}}}}}}}output_path}")
    else:
        logger.info())))))))"No anomalies detected with the current threshold")
        
        # Create a file indicating no anomalies
        output_path = os.path.join())))))))args.output_dir, 'no_anomalies.txt')
        with open())))))))output_path, 'w') as f:
            f.write())))))))f"No anomalies detected with threshold {}}}}}}}}}}}args.regression_threshold}\n")
    
    # Generate interactive visualization if requested::::::
    if args.interactive and anomalies:
        try:
            import plotly.express as px
            
            # Create interactive time series with anomalies
            # For each unique combination with anomalies, create a time series
            unique_configs = anomaly_df[]]]]],,,,,[]]]]],,,,,'model_family', 'hardware_type', 'batch_size']].drop_duplicates()))))))))
            
            for _, config in unique_configs.iterrows())))))))):
                model_family = config[]]]]],,,,,'model_family']
                hardware_type = config[]]]]],,,,,'hardware_type']
                batch_size = config[]]]]],,,,,'batch_size']
                
                # Get the full time series for this configuration
                config_df = df[]]]]],,,,,())))))))df[]]]]],,,,,'model_family'] == model_family) & 
                ())))))))df[]]]]],,,,,'hardware_type'] == hardware_type) &
                ())))))))df[]]]]],,,,,'batch_size'] == batch_size)]
                
                if config_df.empty:
                continue
                
                # Group by date
                date_metrics = config_df.groupby())))))))'date')[]]]]],,,,,metrics].mean())))))))).reset_index()))))))))
                
                # Convert date to string for plotly
                date_metrics[]]]]],,,,,'date_str'] = date_metrics[]]]]],,,,,'date'].astype())))))))str)
                
                # Create the figure
                fig = px.line())))))))date_metrics, x='date_str', y=metrics, 
                labels={}}}}}}}}}}}
                'date_str': 'Date',
                'value': 'Value',
                'variable': 'Metric'
                },
                title=f'Metrics for {}}}}}}}}}}}model_family} on {}}}}}}}}}}}hardware_type} ())))))))Batch={}}}}}}}}}}}batch_size})')
                
                # Add anomaly points
                config_anomalies = anomaly_df[]]]]],,,,,())))))))anomaly_df[]]]]],,,,,'model_family'] == model_family) & 
                ())))))))anomaly_df[]]]]],,,,,'hardware_type'] == hardware_type) &
                ())))))))anomaly_df[]]]]],,,,,'batch_size'] == batch_size)]
                
                for _, anomaly in config_anomalies.iterrows())))))))):
                    date_str = anomaly[]]]]],,,,,'date'].strftime())))))))'%Y-%m-%d')
                    metric = anomaly[]]]]],,,,,'metric']
                    
                    # Find the corresponding data point
                    date_row = date_metrics[]]]]],,,,,date_metrics[]]]]],,,,,'date_str'] == date_str]
                    if not date_row.empty:
                        # Add annotation
                        fig.add_annotation())))))))
                        x=date_str,
                        y=date_row[]]]]],,,,,metric].values[]]]]],,,,,0],
                        text=f"{}}}}}}}}}}}anomaly[]]]]],,,,,'percent_change']:.1f}%",
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor="red",
                        arrowsize=1,
                        arrowwidth=2,
                        bgcolor="rgba())))))))255, 0, 0, 0.2)"
                        )
                
                # Save as HTML
                        filename = f"interactive_anomaly_{}}}}}}}}}}}model_family}_{}}}}}}}}}}}hardware_type}_{}}}}}}}}}}}batch_size}.html"
                        output_path = os.path.join())))))))args.output_dir, filename)
                        fig.write_html())))))))output_path)
                        logger.info())))))))f"Saved interactive anomaly plot to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def analyze_correlation())))))))df, args):
    """Analyze correlations between model parameters and performance"""
    if df is None or df.empty:
    return
        
    logger.info())))))))"Analyzing correlations between model parameters and performance")
    
    # Create output directory
    os.makedirs())))))))args.output_dir, exist_ok=True)
    
    # Analyze the relationship between batch size and performance metrics
    metrics = []]]]],,,,,'throughput_items_per_second', 'average_latency_ms', 'memory_peak_mb']
    
    # Create a figure with subplots for each metric
    fig, axs = plt.subplots())))))))1, len())))))))metrics), figsize=())))))))18, 6))
    
    for i, metric in enumerate())))))))metrics):
        # Create scatter plot of batch size vs metric for each hardware type
        for hardware_type in df[]]]]],,,,,'hardware_type'].unique())))))))):
            hw_df = df[]]]]],,,,,df[]]]]],,,,,'hardware_type'] == hardware_type]
            
            # Group by batch size and calculate mean
            batch_data = hw_df.groupby())))))))'batch_size')[]]]]],,,,,metric].mean()))))))))
            
            # Plot scatter with line
            axs[]]]]],,,,,i].plot())))))))batch_data.index, batch_data.values, marker='o', label=hardware_type)
        
            axs[]]]]],,,,,i].set_title())))))))f'{}}}}}}}}}}}metric} vs Batch Size')
            axs[]]]]],,,,,i].set_xlabel())))))))'Batch Size')
            axs[]]]]],,,,,i].set_ylabel())))))))metric)
            axs[]]]]],,,,,i].grid())))))))True, alpha=0.3)
            axs[]]]]],,,,,i].legend()))))))))
    
            plt.tight_layout()))))))))
    
    # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'batch_size_correlation.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved batch size correlation analysis to {}}}}}}}}}}}output_path}")
    
    # Analyze correlation between all metrics
            numeric_df = df[]]]]],,,,,[]]]]],,,,,'batch_size', 'average_latency_ms', 'throughput_items_per_second', 'memory_peak_mb']].dropna()))))))))
    
    # Calculate correlation matrix
            corr_matrix = numeric_df.corr()))))))))
    
    # Create heatmap of correlations
            plt.figure())))))))figsize=())))))))10, 8))
            sns.heatmap())))))))corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title())))))))'Correlation Matrix of Performance Metrics')
            plt.tight_layout()))))))))
    
    # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'correlation_matrix.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved correlation matrix to {}}}}}}}}}}}output_path}")
    
    # Analyze the correlation between throughput and hardware/model family
    # First, create a pivot table for average throughput by hardware and model family
            pivot_throughput = df.pivot_table())))))))
            values='throughput_items_per_second',
            index='model_family',
            columns='hardware_type',
            aggfunc='mean'
            )
    
    # Calculate correlation between hardware platforms
            hardware_corr = pivot_throughput.corr()))))))))
    
            plt.figure())))))))figsize=())))))))10, 8))
            sns.heatmap())))))))hardware_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title())))))))'Correlation of Throughput Between Hardware Platforms')
            plt.tight_layout()))))))))
    
    # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'hardware_correlation.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved hardware correlation matrix to {}}}}}}}}}}}output_path}")
    
    # Analyze scaling behavior for different batch sizes
    # Create a scatter plot of throughput vs batch size with regression lines
            plt.figure())))))))figsize=())))))))12, 8))
    
    for hardware_type in df[]]]]],,,,,'hardware_type'].unique())))))))):
        hw_df = df[]]]]],,,,,df[]]]]],,,,,'hardware_type'] == hardware_type]
        
        # Plot scatter points
        plt.scatter())))))))hw_df[]]]]],,,,,'batch_size'], hw_df[]]]]],,,,,'throughput_items_per_second'], 
        label=f'{}}}}}}}}}}}hardware_type} ())))))))actual)', alpha=0.6)
        
        # Fit regression line
        if len())))))))hw_df) > 1:
            x = hw_df[]]]]],,,,,'batch_size'].values.reshape())))))))-1, 1)
            y = hw_df[]]]]],,,,,'throughput_items_per_second'].values
            
            model = LinearRegression()))))))))
            model.fit())))))))x, y)
            
            # Plot regression line
            x_range = np.linspace())))))))min())))))))x), max())))))))x), 100).reshape())))))))-1, 1)
            y_pred = model.predict())))))))x_range)
            
            plt.plot())))))))x_range, y_pred, '--', 
            label=f'{}}}}}}}}}}}hardware_type} ())))))))trend, slope={}}}}}}}}}}}model.coef_[]]]]],,,,,0]:.2f})')
            
            # Calculate and display R²
            r2 = r2_score())))))))y, model.predict())))))))x))
            plt.text())))))))max())))))))x), model.predict())))))))[]]]]],,,,,[]]]]],,,,,max())))))))x)]])[]]]]],,,,,0], 
            f'R²={}}}}}}}}}}}r2:.2f}',
            ha='right', va='bottom')
    
            plt.title())))))))'Throughput vs Batch Size with Trend Lines')
            plt.xlabel())))))))'Batch Size')
            plt.ylabel())))))))'Throughput ())))))))items/sec)')
            plt.legend()))))))))
            plt.grid())))))))True, alpha=0.3)
    
    # Save the plot
            output_path = os.path.join())))))))args.output_dir, f'batch_scaling_regression.{}}}}}}}}}}}args.format}')
            plt.savefig())))))))output_path, dpi=300, bbox_inches='tight')
            plt.close()))))))))
    
            logger.info())))))))f"Saved batch scaling regression analysis to {}}}}}}}}}}}output_path}")
    
    # Generate interactive visualization if requested::::::
    if args.interactive:
        try:
            import plotly.express as px
            
            # Create interactive scatter plot with regression lines
            fig = px.scatter())))))))df, x='batch_size', y='throughput_items_per_second', 
            color='hardware_type', facet_col='model_family',
            trendline='ols',
            labels={}}}}}}}}}}}
            'batch_size': 'Batch Size',
            'throughput_items_per_second': 'Throughput ())))))))items/sec)',
            'hardware_type': 'Hardware Type',
            'model_family': 'Model Family'
            },
            title='Interactive Throughput vs Batch Size with Trend Lines')
            
            # Save as HTML
            output_path = os.path.join())))))))args.output_dir, 'interactive_batch_scaling.html')
            fig.write_html())))))))output_path)
            logger.info())))))))f"Saved interactive batch scaling visualization to {}}}}}}}}}}}output_path}")
            
        except ImportError:
            logger.warning())))))))"Plotly is not installed, skipping interactive visualization")

def main())))))))):
    args = parse_args()))))))))
    
    # Set logging level
    if args.verbose:
        logger.setLevel())))))))logging.DEBUG)
    
    # Connect to the database
        conn = connect_to_db())))))))args.db)
    
    try:
        # Get performance data with filters
        df = get_performance_data())))))))conn, args)
        
        if df is None or df.empty:
            logger.error())))))))"No data available for analysis with the given filters")
        return
        
        logger.info())))))))f"Found {}}}}}}}}}}}len())))))))df)} performance records for analysis")
        
        # Run requested analysis
        if args.analysis == 'performance-trends' or args.analysis == 'all':
            analyze_performance_trends())))))))df, args)
            
        if args.analysis == 'hardware-comparison' or args.analysis == 'all':
            analyze_hardware_comparison())))))))df, args)
            
        if args.analysis == 'model-comparison' or args.analysis == 'all':
            analyze_model_comparison())))))))df, args)
            
        if args.analysis == 'performance-prediction' or args.analysis == 'all':
            analyze_performance_prediction())))))))df, args)
            
        if args.analysis == 'anomaly-detection' or args.analysis == 'all':
            analyze_anomaly_detection())))))))df, args)
            
        if args.analysis == 'correlation-analysis' or args.analysis == 'all':
            analyze_correlation())))))))df, args)
            
            logger.info())))))))"Analysis complete")
        
    except Exception as e:
        logger.error())))))))f"Error during analysis: {}}}}}}}}}}}e}")
            raise
    
    finally:
        # Close the database connection
        conn.close()))))))))

if __name__ == "__main__":
    main()))))))))