#!/usr/bin/env python
"""
Benchmark Regression Detector for CI/CD Integration

This script analyzes benchmark results from the database to detect performance regressions.
It compares recent benchmarks with historical data to identify significant performance degradation
and can automatically create GitHub issues for regressions that meet specific thresholds.

Part of Phase 16 of the IPFS Accelerate project.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import requests
import statistics
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig()))))
level=logging.INFO,
format='%()))))asctime)s - %()))))name)s - %()))))levelname)s - %()))))message)s'
)
logger = logging.getLogger()))))"regression_detector")

def parse_args()))))):
    parser = argparse.ArgumentParser()))))description="Detect performance regressions in benchmark results")
    
    parser.add_argument()))))"--db", type=str, default="./benchmark_db.duckdb", 
    help="Path to DuckDB database")
    parser.add_argument()))))"--run-id", type=str,
    help="Run ID to analyze")
    parser.add_argument()))))"--model", type=str,
    help="Filter analysis to specific model")
    parser.add_argument()))))"--hardware", type=str,
    help="Filter analysis to specific hardware")
    parser.add_argument()))))"--threshold", type=float, default=0.1,
    help="Regression threshold ()))))e.g., 0.1 = 10%% degradation)")
    parser.add_argument()))))"--window", type=int, default=5,
    help="Number of previous runs to compare against")
    parser.add_argument()))))"--metrics", type=str, default="throughput,latency",
    help="Comma-separated list of metrics to analyze")
    parser.add_argument()))))"--create-issues", action="store_true",
    help="Create GitHub issues for regressions")
    parser.add_argument()))))"--github-token", type=str,
    help="GitHub token for creating issues")
    parser.add_argument()))))"--github-repo", type=str,
    help="GitHub repository ()))))owner/repo)")
    parser.add_argument()))))"--output", type=str,
    help="Output file for regression report")
    parser.add_argument()))))"--format", type=str, default="json",
    choices=[],"json", "markdown", "html"],
    help="Output format")
    parser.add_argument()))))"--verbose", action="store_true",
    help="Enable verbose logging")
    
return parser.parse_args())))))

def connect_to_db()))))db_path):
    """Connect to the DuckDB database"""
    if not os.path.exists()))))db_path):
        logger.error()))))f"Database file not found: {}}}db_path}")
        sys.exit()))))1)
        
    try:
        conn = duckdb.connect()))))db_path)
        return conn
    except Exception as e:
        logger.error()))))f"Error connecting to database: {}}}e}")
        sys.exit()))))1)

def get_latest_run_id()))))conn):
    """Get the latest run ID if not provided""":
    try:
        result = conn.execute()))))"""
        SELECT run_id
        FROM performance_results
        ORDER BY timestamp DESC
        LIMIT 1
        """).fetchone())))))
        
        if result and result[],0],:,
        return result[],0],
        else:
            logger.error()))))"No run IDs found in database")
            sys.exit()))))1)
    except Exception as e:
        logger.error()))))f"Error getting latest run ID: {}}}e}")
        sys.exit()))))1)

def get_recent_results()))))conn, run_id, model=None, hardware=None, window=5):
    """Get results from the current run and historical runs for comparison"""
    model_filter = f"AND model_name = '{}}}model}'" if model else ""
    hardware_filter = f"AND hardware_type = '{}}}hardware}'" if hardware else ""
    
    # Get current run results
    current_query = f"""
    SELECT 
    pr.id,
    m.model_name,
    h.hardware_type,
    pr.batch_size,
    pr.throughput AS throughput,
    pr.latency AS latency,
    pr.memory_peak AS memory,
    pr.timestamp,
    pr.run_id
    FROM 
    performance_results pr
    JOIN
    models m ON pr.model_id = m.id
    JOIN
    hardware_platforms h ON pr.hardware_id = h.id
    WHERE
    pr.run_id = '{}}}run_id}'
    {}}}model_filter}
    {}}}hardware_filter}
    """
    
    # Get historical results for comparison ()))))excluding current run)
    historical_query = f"""
    WITH recent_runs AS ()))))
    SELECT DISTINCT
    run_id,
    timestamp
    FROM
    performance_results
    WHERE
    run_id != '{}}}run_id}'
    ORDER BY
    timestamp DESC
    LIMIT {}}}window}
    )
    SELECT 
    pr.id,
    m.model_name,
    h.hardware_type,
    pr.batch_size,
    pr.throughput AS throughput,
    pr.latency AS latency,
    pr.memory_peak AS memory,
    pr.timestamp,
    pr.run_id
    FROM 
    performance_results pr
    JOIN
    models m ON pr.model_id = m.id
    JOIN
    hardware_platforms h ON pr.hardware_id = h.id
    JOIN
    recent_runs rr ON pr.run_id = rr.run_id
    WHERE
    1=1
    {}}}model_filter}
    {}}}hardware_filter}
    """
    :
    try:
        current_results = conn.execute()))))current_query).fetchdf())))))
        historical_results = conn.execute()))))historical_query).fetchdf())))))
        
        return current_results, historical_results
    except Exception as e:
        logger.error()))))f"Error fetching results: {}}}e}")
        sys.exit()))))1)

def detect_regressions()))))current_results, historical_results, threshold=0.1, metrics=None):
    """
    Detect performance regressions by comparing current results with historical data.
    Returns a list of regression details.
    """
    if metrics is None:
        metrics = [],"throughput", "latency"]
        ,
    if current_results.empty:
        logger.warning()))))"No current results to analyze")
        return [],]
        ,,,
    if historical_results.empty:
        logger.warning()))))"No historical results to compare against")
        return [],]
        ,,,
        regressions = [],]
        ,,,
    # Group by model, hardware, batch size
        for ()))))model, hardware, batch_size), current_group in current_results.groupby()))))[],'model_name', 'hardware_type', 'batch_size']):,
        # Find matching historical results
        historical_group = historical_results[],
        ()))))historical_results[],'model_name'] == model) &
        ()))))historical_results[],'hardware_type'] == hardware) &
        ()))))historical_results[],'batch_size'] == batch_size)
        ]
        
        if historical_group.empty:
            logger.info()))))f"No historical data for {}}}model} on {}}}hardware} with batch size {}}}batch_size}")
        continue
        
        # Analyze each metric
        for metric in metrics:
            if metric not in current_group.columns or metric not in historical_group.columns:
            continue
                
            current_value = current_group[],metric].mean())))))
            historical_values = historical_group[],metric].tolist())))))
            historical_mean = statistics.mean()))))historical_values)
            
            # For latency, lower is better; for throughput, higher is better
            if metric == "latency":
                change_ratio = current_value / historical_mean - 1
                regression = change_ratio > threshold
            else:  # throughput, memory
            change_ratio = 1 - current_value / historical_mean
            regression = change_ratio > threshold
            
            if regression:
                # Calculate z-score for statistical significance
                if len()))))historical_values) > 1:
                    historical_stddev = statistics.stdev()))))historical_values)
                    if historical_stddev > 0:
                        z_score = abs()))))current_value - historical_mean) / historical_stddev
                    else:
                        z_score = float()))))'inf')
                else:
                    z_score = float()))))'inf')
                
                # Only report statistically significant regressions ()))))z-score > 2)
                if z_score > 2:
                    regression_info = {}}}
                    "model": model,
                    "hardware": hardware,
                    "batch_size": batch_size,
                    "metric": metric,
                    "current_value": float()))))current_value),
                    "historical_mean": float()))))historical_mean),
                    "change_percentage": float()))))change_ratio * 100),
                    "z_score": float()))))z_score),
                    "run_id": current_group[],'run_id'].iloc[],0],,
                        "timestamp": current_group[],'timestamp'].iloc[],0],.isoformat()))))) if isinstance()))))current_group[],'timestamp'].iloc[],0],, pd.Timestamp) else current_group[],'timestamp'].iloc[],0],,:
                            "historical_runs": historical_group[],'run_id'].unique()))))).tolist()))))),
                            "severity": "high" if change_ratio > 0.2 else "medium" if change_ratio > 0.1 else "low"
                            }
                    
                            regressions.append()))))regression_info)
    
                    return regressions
:
def format_regression_report()))))regressions, format_type="json"):
    """Format the regression report in the specified format"""
    if format_type == "json":
    return json.dumps()))))regressions, indent=2)
    
    elif format_type == "markdown":
        if not regressions:
        return "## Performance Regression Report\n\nNo regressions detected."
        
        markdown = "## Performance Regression Report\n\n"
        markdown += f"**Date:** {}}}datetime.datetime.now()))))).strftime()))))'%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"**Total Regressions:** {}}}len()))))regressions)}\n\n"
        
        markdown += "### Summary of Regressions\n\n"
        markdown += "| Model | Hardware | Batch Size | Metric | Change | Severity |\n"
        markdown += "|-------|----------|------------|--------|--------|----------|\n"
        
        for reg in regressions:
            markdown += f"| {}}}reg[],'model']} | {}}}reg[],'hardware']} | {}}}reg[],'batch_size']} | {}}}reg[],'metric']} | {}}}reg[],'change_percentage']:.2f}% | {}}}reg[],'severity'].upper())))))} |\n"
        
            markdown += "\n### Detailed Regression Information\n\n"
        
        for i, reg in enumerate()))))regressions, 1):
            markdown += f"#### Regression #{}}}i}: {}}}reg[],'model']} on {}}}reg[],'hardware']}\n\n"
            markdown += f"- **Metric:** {}}}reg[],'metric']}\n"
            markdown += f"- **Batch Size:** {}}}reg[],'batch_size']}\n"
            markdown += f"- **Current Value:** {}}}reg[],'current_value']:.4f}\n"
            markdown += f"- **Historical Mean:** {}}}reg[],'historical_mean']:.4f}\n"
            markdown += f"- **Change:** {}}}reg[],'change_percentage']:.2f}%\n"
            markdown += f"- **Statistical Significance ()))))Z-Score):** {}}}reg[],'z_score']:.2f}\n"
            markdown += f"- **Severity:** {}}}reg[],'severity'].upper())))))}\n"
            markdown += f"- **Run ID:** {}}}reg[],'run_id']}\n"
            markdown += f"- **Timestamp:** {}}}reg[],'timestamp']}\n\n"
        
            return markdown
    
    elif format_type == "html":
        if not regressions:
        return """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Performance Regression Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
        <div class="container mt-4">
        <h1>Performance Regression Report</h1>
        <p>No regressions detected.</p>
        </div>
        </body>
        </html>
        """
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Performance Regression Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
        <div class="container mt-4">
        <h1>Performance Regression Report</h1>
        <p><strong>Date:</strong> {}}}date}</p>
        <p><strong>Total Regressions:</strong> {}}}total}</p>
                
        <h2>Summary of Regressions</h2>
        <table class="table table-striped">
        <thead>
        <tr>
        <th>Model</th>
        <th>Hardware</th>
        <th>Batch Size</th>
        <th>Metric</th>
        <th>Change</th>
        <th>Severity</th>
        </tr>
        </thead>
        <tbody>
        """.format()))))
        date=datetime.datetime.now()))))).strftime()))))'%Y-%m-%d %H:%M:%S'),
        total=len()))))regressions)
        )
        
        for reg in regressions:
            html += f"""
            <tr>
            <td>{}}}reg[],'model']}</td>
            <td>{}}}reg[],'hardware']}</td>
            <td>{}}}reg[],'batch_size']}</td>
            <td>{}}}reg[],'metric']}</td>
            <td>{}}}reg[],'change_percentage']:.2f}%</td>
            <td><span class="badge bg-{}}}'danger' if reg[],'severity']=='high' else 'warning' if reg[],'severity']=='medium' else 'info'}">{}}}reg[],'severity'].upper())))))}</span></td>
            </tr>
            """
        
            html += """
            </tbody>
            </table>
                
            <h2>Detailed Regression Information</h2>
            <div class="accordion" id="regressionAccordion">
            """
        :
        for i, reg in enumerate()))))regressions, 1):
            html += f"""
            <div class="accordion-item">
            <h2 class="accordion-header" id="heading{}}}i}">
                        <button class="accordion-button {}}}'collapsed' if i > 1 else ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{}}}i}" aria-expanded="{}}}i==1}" aria-controls="collapse{}}}i}">:
                            Regression #{}}}i}: {}}}reg[],'model']} on {}}}reg[],'hardware']} ())))){}}}reg[],'metric']})
                            </button>
                            </h2>
                            <div id="collapse{}}}i}" class="accordion-collapse collapse {}}}'show' if i==1 else ''}" aria-labelledby="heading{}}}i}" data-bs-parent="#regressionAccordion">
                            <div class="accordion-body">
                            <ul class="list-group">:
                                <li class="list-group-item"><strong>Metric:</strong> {}}}reg[],'metric']}</li>
                                <li class="list-group-item"><strong>Batch Size:</strong> {}}}reg[],'batch_size']}</li>
                                <li class="list-group-item"><strong>Current Value:</strong> {}}}reg[],'current_value']:.4f}</li>
                                <li class="list-group-item"><strong>Historical Mean:</strong> {}}}reg[],'historical_mean']:.4f}</li>
                                <li class="list-group-item"><strong>Change:</strong> {}}}reg[],'change_percentage']:.2f}%</li>
                                <li class="list-group-item"><strong>Statistical Significance ()))))Z-Score):</strong> {}}}reg[],'z_score']:.2f}</li>
                                <li class="list-group-item"><strong>Severity:</strong> {}}}reg[],'severity'].upper())))))}</li>
                                <li class="list-group-item"><strong>Run ID:</strong> {}}}reg[],'run_id']}</li>
                                <li class="list-group-item"><strong>Timestamp:</strong> {}}}reg[],'timestamp']}</li>
                                </ul>
                                </div>
                                </div>
                                </div>
                                """
        
                                html += """
                                </div>
                                </div>
                                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
                                </body>
                                </html>
                                """
        
                            return html
    
    else:
        logger.error()))))f"Unsupported format: {}}}format_type}")
                            return json.dumps()))))regressions)

def create_github_issue()))))regressions, token, repo):
    """Create a GitHub issue for performance regressions"""
    if not regressions:
        logger.info()))))"No regressions to report")
    return None
    
    # Group regressions by severity
    high_severity = [],r for r in regressions if r[],'severity'] == 'high']
    medium_severity = [],r for r in regressions if r[],'severity'] == 'medium']
    low_severity = [],r for r in regressions if r[],'severity'] == 'low']
    
    # Only create issues for medium and high severity:
    if not high_severity and not medium_severity:
        logger.info()))))"No medium or high severity regressions to report")
    return None
    
    # Prepare issue content
    title = f"Performance Regression Alert: {}}}len()))))high_severity)} high, {}}}len()))))medium_severity)} medium severity issues"
    body = format_regression_report()))))regressions, format_type="markdown")
    
    # Create GitHub issue
    url = f"https://api.github.com/repos/{}}}repo}/issues"
    headers = {}}}
    "Authorization": f"token {}}}token}",
    "Accept": "application/vnd.github.v3+json"
    }
    data = {}}}
    "title": title,
    "body": body,
    "labels": [],"performance", "regression", "automated"]
    }
    
    try:
        response = requests.post()))))url, headers=headers, json=data)
        response.raise_for_status())))))
        issue = response.json())))))
        logger.info()))))f"Created GitHub issue #{}}}issue[],'number']}: {}}}issue[],'html_url']}")
    return issue
    except Exception as e:
        logger.error()))))f"Error creating GitHub issue: {}}}e}")
    return None

def main()))))):
    args = parse_args())))))
    
    # Set logging level
    if args.verbose:
        logger.setLevel()))))logging.DEBUG)
    
    # Connect to database
        conn = connect_to_db()))))args.db)
    
    # Get run ID if not provided
    run_id = args.run_id or get_latest_run_id()))))conn):
        logger.info()))))f"Analyzing run ID: {}}}run_id}")
    
    # Get metrics to analyze
        metrics = args.metrics.split()))))',') if args.metrics else [],"throughput", "latency"]
        ,
    # Get current and historical results
        current_results, historical_results = get_recent_results()))))
        conn, run_id, args.model, args.hardware, args.window
        )
    
    # Detect regressions
        regressions = detect_regressions()))))
        current_results, historical_results, args.threshold, metrics
        )
    
    # Generate report
        report = format_regression_report()))))regressions, args.format)
    
    # Write report to file if output specified:
    if args.output:
        with open()))))args.output, 'w') as f:
            f.write()))))report)
            logger.info()))))f"Regression report written to {}}}args.output}")
    else:
        print()))))report)
    
    # Create GitHub issue if requested:
    if args.create_issues and ()))))args.github_token and args.github_repo):
        issue = create_github_issue()))))regressions, args.github_token, args.github_repo)
        if issue:
            logger.info()))))f"Created GitHub issue: {}}}issue[],'html_url']}")
    elif args.create_issues:
        logger.warning()))))"Cannot create GitHub issue: token or repo not provided")
    
    # Return exit code based on regression severity
    if any()))))r[],'severity'] == 'high' for r in regressions):
        logger.warning()))))"High severity regressions detected")
        return 2
    elif any()))))r[],'severity'] == 'medium' for r in regressions):
        logger.warning()))))"Medium severity regressions detected")
        return 1
    else:
        logger.info()))))"No significant regressions detected")
        return 0

if __name__ == "__main__":
    sys.exit()))))main()))))))