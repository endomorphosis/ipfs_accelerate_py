#!/usr/bin/env python3
"""
Run Error Recovery Visualization Tests

This script runs tests for the error recovery visualization system, generating
the example visualizations referenced in the documentation.
"""

import os
import sys
import logging
import asyncio
import argparse
import random
import time
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_test_error_recovery_visualization")


async def run_visualization_test(output_dir=None, generate_sample_data=False, format="png"):
    """
    Run the error recovery visualization tests.
    
    Args:
        output_dir: Directory to save visualizations
        generate_sample_data: Whether to generate sample data in the database
        format: Output file format (png, pdf, svg)
    """
    # Import necessary modules
    import duckdb
    from error_recovery_visualization_integration import create_visualization_integration
    
    # Create a mock implementation of the error recovery module
    class PerformanceBasedErrorRecovery:
        def __init__(self, error_handler, recovery_manager, coordinator=None, db_connection=None):
            self.error_handler = error_handler
            self.recovery_manager = recovery_manager
            self.coordinator = coordinator
            self.db_connection = db_connection
            
        def get_performance_metrics(self):
            """Mock implementation to get performance metrics."""
            if self.db_connection:
                # Fetch strategy stats
                strategy_stats = {}
                
                strategies_result = self.db_connection.execute("""
                SELECT DISTINCT strategy_id, strategy_name 
                FROM recovery_strategy_scores
                """).fetchall()
                
                for strategy_id, strategy_name in strategies_result:
                    # Get overall stats for this strategy
                    result = self.db_connection.execute("""
                    SELECT 
                        AVG(success_rate) as success_rate,
                        AVG(average_recovery_time) as avg_recovery_time,
                        AVG(resource_efficiency) as resource_efficiency,
                        AVG(impact_score) as impact_score,
                        AVG(stability_score) as stability_score,
                        AVG(task_recovery_rate) as task_recovery_rate,
                        AVG(overall_score) as overall_score,
                        SUM(sample_count) as total_samples
                    FROM recovery_strategy_scores
                    WHERE strategy_id = ?
                    """, (strategy_id,)).fetchone()
                    
                    success_rate, avg_recovery_time, resource_efficiency, impact_score, stability_score, task_recovery_rate, overall_score, total_samples = result
                    
                    # Get by_error_type data
                    by_error_type = {}
                    error_types_result = self.db_connection.execute("""
                    SELECT 
                        error_type,
                        success_rate,
                        average_recovery_time,
                        resource_efficiency,
                        impact_score,
                        stability_score,
                        task_recovery_rate,
                        overall_score,
                        sample_count,
                        last_used,
                        metrics
                    FROM recovery_strategy_scores
                    WHERE strategy_id = ?
                    """, (strategy_id,)).fetchall()
                    
                    for et_result in error_types_result:
                        error_type = et_result[0]
                        by_error_type[error_type] = {
                            "success_rate": et_result[1],
                            "avg_recovery_time": et_result[2],
                            "resource_efficiency": et_result[3],
                            "impact_score": et_result[4],
                            "stability_score": et_result[5],
                            "task_recovery_rate": et_result[6],
                            "overall_score": et_result[7],
                            "sample_count": et_result[8],
                            "last_used": et_result[9],
                            "metrics": et_result[10]
                        }
                    
                    strategy_stats[strategy_id] = {
                        "name": strategy_name,
                        "total_samples": total_samples,
                        "success_rate": success_rate,
                        "avg_recovery_time": avg_recovery_time,
                        "resource_efficiency": resource_efficiency,
                        "impact_score": impact_score,
                        "stability_score": stability_score,
                        "task_recovery_rate": task_recovery_rate,
                        "overall_score": overall_score,
                        "by_error_type": by_error_type
                    }
                
                # Get top strategies by error type
                top_strategies = {}
                error_types_result = self.db_connection.execute("""
                SELECT DISTINCT error_type FROM recovery_strategy_scores
                """).fetchall()
                
                for (error_type,) in error_types_result:
                    best_strategy_result = self.db_connection.execute("""
                    SELECT strategy_id, strategy_name, overall_score
                    FROM recovery_strategy_scores
                    WHERE error_type = ?
                    ORDER BY overall_score DESC
                    LIMIT 1
                    """, (error_type,)).fetchone()
                    
                    if best_strategy_result:
                        top_strategies[error_type] = {
                            "strategy_id": best_strategy_result[0],
                            "strategy_name": best_strategy_result[1],
                            "score": best_strategy_result[2]
                        }
                
                # Calculate summary stats
                summary = {
                    "total_strategies": len(strategy_stats),
                    "total_error_types": len(error_types_result),
                    "average_success_rate": sum(s["success_rate"] for s in strategy_stats.values()) / len(strategy_stats) if strategy_stats else 0.0,
                    "average_recovery_time": sum(s["avg_recovery_time"] for s in strategy_stats.values()) / len(strategy_stats) if strategy_stats else 0.0
                }
                
                return {
                    "strategy_stats": strategy_stats,
                    "top_strategies": top_strategies,
                    "summary": summary
                }
            else:
                return {"strategy_stats": {}, "top_strategies": {}, "summary": {}}
        
        def get_progressive_recovery_history(self, error_id=None):
            """Mock implementation to get progressive recovery history."""
            if not self.db_connection:
                return {"summary": {}}
                
            if error_id:
                # Get history for specific error
                history_result = self.db_connection.execute("""
                SELECT error_id, recovery_level, strategy_id, strategy_name, timestamp, success, details
                FROM progressive_recovery_history
                WHERE error_id = ?
                ORDER BY recovery_level
                """, (error_id,)).fetchall()
                
                history = []
                for row in history_result:
                    history.append({
                        "error_id": row[0],
                        "recovery_level": row[1],
                        "strategy_id": row[2],
                        "strategy_name": row[3],
                        "timestamp": row[4],
                        "success": row[5],
                        "details": row[6]
                    })
                
                # Get current level
                current_level_result = self.db_connection.execute("""
                SELECT MAX(recovery_level) FROM progressive_recovery_history WHERE error_id = ?
                """, (error_id,)).fetchone()
                
                current_level = current_level_result[0] if current_level_result and current_level_result[0] else 1
                
                return {
                    "error_id": error_id,
                    "history": history,
                    "current_level": current_level
                }
            else:
                # Get summary of all errors
                level_counts_result = self.db_connection.execute("""
                SELECT 
                    COUNT(DISTINCT CASE WHEN max_level = 1 THEN error_id END) as level_1_count,
                    COUNT(DISTINCT CASE WHEN max_level = 2 THEN error_id END) as level_2_count,
                    COUNT(DISTINCT CASE WHEN max_level = 3 THEN error_id END) as level_3_count,
                    COUNT(DISTINCT CASE WHEN max_level = 4 THEN error_id END) as level_4_count,
                    COUNT(DISTINCT CASE WHEN max_level = 5 THEN error_id END) as level_5_count
                FROM (
                    SELECT error_id, MAX(recovery_level) as max_level
                    FROM progressive_recovery_history
                    GROUP BY error_id
                ) t
                """).fetchone()
                
                success_counts_result = self.db_connection.execute("""
                SELECT 
                    SUM(CASE WHEN last_success THEN 1 ELSE 0 END) as successful_recoveries,
                    SUM(CASE WHEN NOT last_success THEN 1 ELSE 0 END) as failed_recoveries
                FROM (
                    SELECT 
                        error_id, 
                        MAX(recovery_level) as max_level,
                        (SELECT success FROM progressive_recovery_history p2 
                         WHERE p2.error_id = p1.error_id 
                         ORDER BY recovery_level DESC LIMIT 1) as last_success
                    FROM progressive_recovery_history p1
                    GROUP BY error_id
                ) t
                """).fetchone()
                
                errors_result = self.db_connection.execute("""
                SELECT error_id, MAX(recovery_level) as current_level, COUNT(*) as attempts, 
                      (SELECT success FROM progressive_recovery_history p2 
                       WHERE p2.error_id = p1.error_id 
                       ORDER BY recovery_level DESC LIMIT 1) as last_success,
                      (SELECT timestamp FROM progressive_recovery_history p2 
                       WHERE p2.error_id = p1.error_id 
                       ORDER BY recovery_level DESC LIMIT 1) as last_timestamp
                FROM progressive_recovery_history p1
                GROUP BY error_id
                """).fetchall()
                
                errors = []
                for row in errors_result:
                    errors.append({
                        "error_id": row[0],
                        "current_level": row[1],
                        "attempts": row[2],
                        "last_attempt_success": row[3],
                        "last_attempt_time": row[4]
                    })
                
                return {
                    "errors": errors,
                    "summary": {
                        "level_1_count": level_counts_result[0] if level_counts_result else 0,
                        "level_2_count": level_counts_result[1] if level_counts_result else 0,
                        "level_3_count": level_counts_result[2] if level_counts_result else 0,
                        "level_4_count": level_counts_result[3] if level_counts_result else 0,
                        "level_5_count": level_counts_result[4] if level_counts_result else 0,
                        "successful_recoveries": success_counts_result[0] if success_counts_result else 0,
                        "failed_recoveries": success_counts_result[1] if success_counts_result else 0
                    }
                }
        
        def get_strategy_recommendations(self, error_type):
            """Mock implementation to get strategy recommendations."""
            if not self.db_connection:
                return []
                
            recommendations_result = self.db_connection.execute("""
            SELECT 
                strategy_id, strategy_name, overall_score, success_rate, average_recovery_time,
                sample_count, last_used, metrics
            FROM recovery_strategy_scores
            WHERE error_type = ?
            ORDER BY overall_score DESC
            """, (error_type,)).fetchall()
            
            recommendations = []
            for row in recommendations_result:
                recommendations.append({
                    "strategy_id": row[0],
                    "strategy_name": row[1],
                    "score": row[2],
                    "success_rate": row[3],
                    "avg_recovery_time": row[4],
                    "sample_count": row[5],
                    "last_used": row[6],
                    "metrics": row[7]
                })
                
            return recommendations
            
        async def analyze_recovery_performance(self, days=30):
            """Mock implementation to analyze recovery performance."""
            if not self.db_connection:
                return {"time_series": []}
                
            # Get time series data
            # Get time series data - use strftime to format dates
            time_series_result = self.db_connection.execute("""
            WITH days_data AS (
                SELECT 
                    strftime(timestamp, '%Y-%m-%d') as date,
                    COUNT(*) as total_recoveries,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_recoveries,
                    AVG(execution_time_seconds) as avg_execution_time
                FROM recovery_performance
                GROUP BY strftime(timestamp, '%Y-%m-%d')
                ORDER BY strftime(timestamp, '%Y-%m-%d')
            )
            SELECT 
                date,
                total_recoveries,
                successful_recoveries,
                avg_execution_time,
                CASE 
                    WHEN total_recoveries > 0 THEN CAST(successful_recoveries AS FLOAT) / total_recoveries 
                    ELSE 0 
                END as success_rate
            FROM days_data
            """).fetchall()
            
            time_series = []
            for row in time_series_result:
                time_series.append({
                    "date": row[0],
                    "total_recoveries": row[1],
                    "successful_recoveries": row[2],
                    "avg_execution_time": row[3],
                    "success_rate": row[4]
                })
                
            # Calculate metrics
            overall_metrics_result = self.db_connection.execute("""
            SELECT 
                COUNT(*) as total_recoveries,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_recoveries,
                AVG(execution_time_seconds) as avg_execution_time,
                AVG(impact_score) as avg_impact_score,
                AVG(post_recovery_stability) as avg_stability_score,
                CASE 
                    WHEN COUNT(*) > 0 THEN CAST(SUM(CASE WHEN success THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) 
                    ELSE 0 
                END as success_rate
            FROM recovery_performance
            """).fetchone()
            
            if overall_metrics_result:
                overall = {
                    "total_recoveries": overall_metrics_result[0],
                    "successful_recoveries": overall_metrics_result[1],
                    "avg_execution_time": overall_metrics_result[2],
                    "avg_impact_score": overall_metrics_result[3],
                    "avg_stability_score": overall_metrics_result[4],
                    "success_rate": overall_metrics_result[5]
                }
            else:
                overall = {
                    "total_recoveries": 0,
                    "successful_recoveries": 0,
                    "avg_execution_time": 0,
                    "avg_impact_score": 0,
                    "avg_stability_score": 0,
                    "success_rate": 0
                }
                
            # Calculate metrics by error type
            error_type_metrics_result = self.db_connection.execute("""
            SELECT 
                error_type,
                COUNT(*) as total_recoveries,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_recoveries,
                AVG(execution_time_seconds) as avg_execution_time,
                AVG(impact_score) as avg_impact_score,
                AVG(post_recovery_stability) as avg_stability_score,
                CASE 
                    WHEN COUNT(*) > 0 THEN CAST(SUM(CASE WHEN success THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) 
                    ELSE 0 
                END as success_rate
            FROM recovery_performance
            GROUP BY error_type
            """).fetchall()
            
            error_type_metrics = {}
            for row in error_type_metrics_result:
                error_type_metrics[row[0]] = {
                    "total_recoveries": row[1],
                    "successful_recoveries": row[2],
                    "avg_execution_time": row[3],
                    "avg_impact_score": row[4],
                    "avg_stability_score": row[5],
                    "success_rate": row[6]
                }
                
            # Calculate metrics by strategy
            strategy_metrics_result = self.db_connection.execute("""
            SELECT 
                strategy_id,
                strategy_name,
                COUNT(*) as total_recoveries,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_recoveries,
                AVG(execution_time_seconds) as avg_execution_time,
                AVG(impact_score) as avg_impact_score,
                AVG(post_recovery_stability) as avg_stability_score,
                CASE 
                    WHEN COUNT(*) > 0 THEN CAST(SUM(CASE WHEN success THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) 
                    ELSE 0 
                END as success_rate
            FROM recovery_performance
            GROUP BY strategy_id, strategy_name
            """).fetchall()
            
            strategy_metrics = {}
            for row in strategy_metrics_result:
                strategy_metrics[row[0]] = {
                    "name": row[1],
                    "total_recoveries": row[2],
                    "successful_recoveries": row[3],
                    "avg_execution_time": row[4],
                    "avg_impact_score": row[5],
                    "avg_stability_score": row[6],
                    "success_rate": row[7]
                }
                
            return {
                "overall": overall,
                "by_error_type": error_type_metrics,
                "by_strategy": strategy_metrics,
                "time_series": time_series,
                "record_count": overall["total_recoveries"],
                "period_days": days
            }
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to database - use in-memory for tests
    db_connection = duckdb.connect(":memory:")
    
    # Create tables
    db_connection.execute("""
    CREATE SEQUENCE IF NOT EXISTS seq_recovery_performance_id;
    CREATE TABLE IF NOT EXISTS recovery_performance (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_recovery_performance_id'),
        strategy_id VARCHAR,
        strategy_name VARCHAR,
        error_type VARCHAR,
        execution_time_seconds FLOAT,
        success BOOLEAN,
        timestamp TIMESTAMP,
        affected_tasks INTEGER,
        task_recovery_success INTEGER,
        resource_usage JSON,
        impact_score FLOAT,
        post_recovery_stability FLOAT,
        metrics JSON,
        context JSON
    )
    """)
    
    db_connection.execute("""
    CREATE TABLE IF NOT EXISTS recovery_strategy_scores (
        strategy_id VARCHAR,
        strategy_name VARCHAR,
        error_type VARCHAR,
        success_rate FLOAT,
        average_recovery_time FLOAT,
        resource_efficiency FLOAT,
        impact_score FLOAT,
        stability_score FLOAT,
        task_recovery_rate FLOAT,
        overall_score FLOAT,
        sample_count INTEGER,
        last_used TIMESTAMP,
        metrics JSON,
        PRIMARY KEY (strategy_id, error_type)
    )
    """)
    
    db_connection.execute("""
    CREATE TABLE IF NOT EXISTS recovery_timeouts (
        error_type VARCHAR,
        strategy_id VARCHAR,
        timeout_seconds FLOAT,
        timestamp TIMESTAMP,
        success BOOLEAN,
        PRIMARY KEY (error_type, strategy_id, timestamp)
    )
    """)
    
    db_connection.execute("""
    CREATE TABLE IF NOT EXISTS progressive_recovery_history (
        error_id VARCHAR,
        recovery_level INTEGER,
        strategy_id VARCHAR,
        strategy_name VARCHAR,
        timestamp TIMESTAMP,
        success BOOLEAN,
        details JSON,
        PRIMARY KEY (error_id, recovery_level)
    )
    """)
    
    # Generate sample data
    if generate_sample_data:
        await generate_performance_data(db_connection)
    
    # Create dummy coordinator
    class DummyCoordinator:
        def __init__(self):
            self.db = db_connection
    
    # Create a mock recovery manager
    class MockRecoveryManager:
        def __init__(self):
            self.strategies = {
                "retry": MockStrategy("Retry Strategy", "low"),
                "worker_recovery": MockStrategy("Worker Recovery Strategy", "medium"),
                "database_recovery": MockStrategy("Database Recovery Strategy", "medium"),
                "coordinator_recovery": MockStrategy("Coordinator Recovery Strategy", "high"),
                "system_recovery": MockStrategy("System Recovery Strategy", "critical")
            }
            
    class MockStrategy:
        def __init__(self, name, level):
            self.name = name
            self.level = MockLevel(level)
            
        async def execute(self, error_info):
            return True
            
    class MockLevel:
        def __init__(self, value):
            self.value = value
            
    class MockErrorHandler:
        def __init__(self):
            pass
            
        def register_error_hook(self, error_type, hook):
            pass
    
    # Create recovery system
    recovery_system = PerformanceBasedErrorRecovery(
        error_handler=MockErrorHandler(),
        recovery_manager=MockRecoveryManager(),
        coordinator=DummyCoordinator(),
        db_connection=db_connection
    )
    
    # Create visualization integration
    integration = create_visualization_integration(
        recovery_system=recovery_system,
        db_connection=db_connection,
        output_dir=output_dir
    )
    
    # Generate dashboard
    logger.info("Generating dashboard...")
    dashboard_path = await integration.generate_dashboard(days=30, file_format=format)
    logger.info(f"Dashboard generated at: {dashboard_path}")
    
    # Generate individual visualizations with specific filenames to match documentation
    logger.info("Generating individual visualizations...")
    
    # Create visualizer with format
    visualizer = integration.visualizer
    
    # Generate the specific visualizations mentioned in documentation
    performance_path = visualizer.visualize_strategy_performance(
        performance_data=recovery_system.get_performance_metrics(),
        filename="strategy_performance_dashboard"
    )
    logger.info(f"  - Strategy performance dashboard: {performance_path}")
    
    heatmap_path = visualizer.visualize_error_recovery_heatmap(
        performance_data=recovery_system.get_performance_metrics(),
        filename="error_recovery_heatmap"
    )
    logger.info(f"  - Error recovery heatmap: {heatmap_path}")
    
    analysis_data = await recovery_system.analyze_recovery_performance(days=30)
    trends_path = visualizer.visualize_performance_trends(
        time_series_data=analysis_data.get("time_series", []),
        filename="performance_trend_graphs"
    )
    logger.info(f"  - Performance trend graphs: {trends_path}")
    
    vis_paths = {
        "strategy_performance_dashboard": performance_path,
        "error_recovery_heatmap": heatmap_path,
        "performance_trend_graphs": trends_path,
    }
    
    # Generate visualization for specific error type
    error_types = ["NETWORK", "TIMEOUT", "DATABASE", "SYSTEM"]
    for error_type in error_types:
        logger.info(f"Generating visualization for error type {error_type}...")
        vis_path = await integration.generate_visualization_for_error_type(error_type, file_format=format)
        logger.info(f"  - Error type visualization for {error_type}: {vis_path}")
    
    logger.info(f"All visualizations generated in {output_dir}")
    
    return {
        "dashboard": dashboard_path,
        "individual": vis_paths,
        "error_types": error_types
    }


async def generate_performance_data(db_connection):
    """
    Generate sample performance data for testing.
    
    Args:
        db_connection: Database connection
    """
    logger.info("Generating sample performance data...")
    
    # Define sample strategies
    strategies = [
        {"id": "retry", "name": "Retry Strategy"},
        {"id": "worker_recovery", "name": "Worker Recovery Strategy"},
        {"id": "database_recovery", "name": "Database Recovery Strategy"},
        {"id": "coordinator_recovery", "name": "Coordinator Recovery Strategy"},
        {"id": "system_recovery", "name": "System Recovery Strategy"}
    ]
    
    # Define sample error types
    error_types = ["NETWORK", "TIMEOUT", "DATABASE", "SYSTEM", "RESOURCE", "VALIDATION"]
    
    # Generate recovery performance records
    now = datetime.now()
    
    for i in range(200):  # Generate 200 performance records
        strategy = random.choice(strategies)
        error_type = random.choice(error_types)
        
        # Determine success probability based on strategy and error type
        base_success_prob = 0.75  # 75% base success rate
        if error_type == "NETWORK" and strategy["id"] == "retry":
            success_prob = 0.9  # Retry works well for network errors
        elif error_type == "DATABASE" and strategy["id"] == "database_recovery":
            success_prob = 0.85  # Database recovery works well for database errors
        elif error_type == "SYSTEM" and strategy["id"] == "system_recovery":
            success_prob = 0.8  # System recovery works well for system errors
        else:
            success_prob = base_success_prob
            
        success = random.random() < success_prob
        
        # Generate timestamp in the past 30 days
        days_ago = random.randint(0, 29)
        timestamp = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Generate execution time
        base_time = 5.0  # Base execution time in seconds
        if success:
            execution_time = base_time * (0.5 + random.random())  # 0.5x to 1.5x base time
        else:
            execution_time = base_time * (1.5 + random.random() * 2.0)  # 1.5x to 3.5x base time
            
        # Generate random number of affected tasks and recovered tasks
        affected_tasks = random.randint(1, 10)
        task_recovery_success = affected_tasks if success else random.randint(0, affected_tasks - 1)
        
        # Generate resource usage
        resource_usage = {
            "cpu_percent": random.uniform(10.0, 50.0),
            "memory_percent": random.uniform(5.0, 30.0),
            "process_cpu": random.uniform(5.0, 20.0),
            "process_memory_mb": random.uniform(50.0, 200.0)
        }
        
        # Generate impact score
        impact_score = 0.2 + random.random() * 0.6  # 0.2 to 0.8
        
        # Generate stability score
        stability_score = 0.9 if success else 0.2 + random.random() * 0.4  # 0.2 to 0.6 if failed, 0.9 if success
        
        # Generate metrics
        metrics = {
            "success_rate": 1.0 if success else 0.0,
            "recovery_time": execution_time,
            "resource_usage": sum(resource_usage.values()),
            "impact_score": impact_score,
            "stability": stability_score,
            "task_recovery": task_recovery_success / affected_tasks if affected_tasks > 0 else 0.0
        }
        
        # Generate context
        context = {
            "error_id": f"error_{i}",
            "recovery_level": random.randint(1, 5),
            "component": random.choice(["worker", "coordinator", "database", "network"]),
            "operation": random.choice(["assign", "execute", "query", "connect"])
        }
        
        # Insert into database
        db_connection.execute("""
        INSERT INTO recovery_performance (
            strategy_id, strategy_name, error_type, execution_time_seconds,
            success, timestamp, affected_tasks, task_recovery_success,
            resource_usage, impact_score, post_recovery_stability,
            metrics, context
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy["id"],
            strategy["name"],
            error_type,
            execution_time,
            success,
            timestamp,
            affected_tasks,
            task_recovery_success,
            json.dumps(resource_usage),
            impact_score,
            stability_score,
            json.dumps(metrics),
            json.dumps(context)
        ))
    
    # Calculate strategy scores
    for strategy in strategies:
        for error_type in error_types:
            # Get performance records for this strategy and error type
            result = db_connection.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(execution_time_seconds) as avg_time,
                AVG(impact_score) as avg_impact,
                AVG(post_recovery_stability) as avg_stability,
                SUM(affected_tasks) as total_affected_tasks,
                SUM(task_recovery_success) as total_recovered_tasks
            FROM recovery_performance
            WHERE strategy_id = ? AND error_type = ?
            """, (strategy["id"], error_type)).fetchone()
            
            if result[0] > 0:  # If we have records
                total = result[0]
                successes = result[1]
                avg_time = result[2]
                avg_impact = result[3]
                avg_stability = result[4]
                total_affected_tasks = result[5]
                total_recovered_tasks = result[6]
                
                # Calculate success rate
                success_rate = successes / total
                
                # Calculate resource efficiency (random for demo)
                resource_efficiency = 0.6 + random.random() * 0.3  # 0.6 to 0.9
                
                # Calculate task recovery rate
                task_recovery_rate = total_recovered_tasks / total_affected_tasks if total_affected_tasks > 0 else 1.0
                
                # Calculate overall score
                overall_score = (
                    success_rate * 0.4 +
                    (1.0 - (avg_time / 30.0)) * 0.15 +  # Normalize time (lower is better)
                    resource_efficiency * 0.1 +
                    (1.0 - avg_impact) * 0.1 +  # Lower impact is better
                    avg_stability * 0.1 +
                    task_recovery_rate * 0.15
                )
                
                # Cap the score at 1.0
                overall_score = min(overall_score, 1.0)
                
                # Insert into database
                db_connection.execute("""
                INSERT INTO recovery_strategy_scores (
                    strategy_id, strategy_name, error_type, success_rate,
                    average_recovery_time, resource_efficiency, impact_score,
                    stability_score, task_recovery_rate, overall_score,
                    sample_count, last_used, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy["id"],
                    strategy["name"],
                    error_type,
                    success_rate,
                    avg_time,
                    resource_efficiency,
                    avg_impact,
                    avg_stability,
                    task_recovery_rate,
                    overall_score,
                    total,
                    datetime.now(),
                    json.dumps({
                        "success_rate": success_rate,
                        "recovery_time": avg_time,
                        "recovery_time_score": 1.0 - (avg_time / 30.0),
                        "resource_efficiency": resource_efficiency,
                        "impact_score": avg_impact,
                        "stability_score": avg_stability,
                        "task_recovery_rate": task_recovery_rate
                    })
                ))
    
    # Generate progressive recovery history
    error_ids = [f"error_{i}" for i in range(20)]  # Use 20 error IDs
    
    for error_id in error_ids:
        # Randomly assign current recovery level
        max_level = random.randint(1, 5)
        
        # Generate a random success/failure path up to max_level
        for level in range(1, max_level + 1):
            strategy = random.choice(strategies)
            
            # Earlier levels more likely to fail than later levels
            success = level == max_level or random.random() > 0.6
            
            # Insert progression entry
            db_connection.execute("""
            INSERT INTO progressive_recovery_history (
                error_id, recovery_level, strategy_id, strategy_name,
                timestamp, success, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                error_id,
                level,
                strategy["id"],
                strategy["name"],
                datetime.now() - timedelta(minutes=level * 5),  # 5 minutes between levels
                success,
                json.dumps({
                    "old_level": level,
                    "new_level": level + 1 if not success and level < 5 else level,
                    "execution_time": random.uniform(1.0, 10.0)
                })
            ))
            
            # If success at this level, stop the progression
            if success:
                break
    
    logger.info("Sample performance data generated successfully")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Error Recovery Visualization Tests")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate sample performance data")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output file format")
    args = parser.parse_args()
    
    # Run the visualization test
    asyncio.run(run_visualization_test(
        output_dir=args.output_dir,
        generate_sample_data=args.generate_sample_data,
        format=args.format
    ))


if __name__ == "__main__":
    main()