#!/usr/bin/env python3
"""
Interactive Performance Dashboard for IPFS Accelerate Python

This module provides an interactive web-based dashboard for monitoring
and analyzing performance benchmarks, system status, and optimization
recommendations in real-time.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
from datetime import datetime, timedelta

# Safe imports
try:
    from .safe_imports import safe_import, get_import_summary
    from .production_validation import run_production_validation, ValidationLevel
    from .advanced_benchmarking import run_quick_benchmark, AdvancedBenchmarkSuite
    from .model_compatibility import get_optimal_hardware
    from ..hardware_detection import HardwareDetector
except ImportError:
    # Fallback imports for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.safe_imports import safe_import, get_import_summary
    from utils.production_validation import run_production_validation, ValidationLevel
    from utils.advanced_benchmarking import run_quick_benchmark, AdvancedBenchmarkSuite
    from utils.model_compatibility import get_optimal_hardware
    from hardware_detection import HardwareDetector

# Optional web framework
flask = safe_import('flask')
if flask:
    from flask import Flask, render_template_string, jsonify, request

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Interactive web dashboard for performance monitoring."""
    
    def __init__(self, port: int = 8080, update_interval: int = 30):
        self.port = port
        self.update_interval = update_interval
        self.detector = HardwareDetector()
        self.benchmark_suite = AdvancedBenchmarkSuite()
        
        # Dashboard data
        self.dashboard_data = {
            "system_status": {},
            "recent_benchmarks": [],
            "performance_trends": [],
            "optimization_recommendations": [],
            "last_updated": None
        }
        
        # Background update thread
        self.update_thread = None
        self.running = False
        
        # Initialize Flask app if available
        if flask:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            self.app = None
            logger.warning("Flask not available - dashboard will run in CLI mode only")
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_template(), **self.dashboard_data)
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/run_benchmark', methods=['POST'])
        def api_run_benchmark():
            try:
                # Run quick benchmark
                benchmark_run = run_quick_benchmark()
                
                # Update dashboard data
                self._update_benchmark_data(benchmark_run)
                
                return jsonify({
                    "success": True,
                    "run_id": benchmark_run.run_id,
                    "duration": benchmark_run.duration_seconds
                })
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/validate_system')
        def api_validate_system():
            try:
                # Run system validation
                report = run_production_validation("production")
                
                # Update dashboard data
                self.dashboard_data["system_status"] = {
                    "overall_score": report.overall_score,
                    "recommendations": report.deployment_recommendations,
                    "last_validation": datetime.now().isoformat()
                }
                
                return jsonify({
                    "success": True,
                    "score": report.overall_score,
                    "recommendations": report.deployment_recommendations
                })
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/hardware_info')
        def api_hardware_info():
            try:
                available_hardware = self.detector.get_available_hardware()
                best_hardware = self.detector.get_best_available_hardware()
                
                hardware_details = {}
                for hw in available_hardware:
                    try:
                        info = self.detector.get_hardware_info(hw)
                        hardware_details[hw] = info
                    except Exception:
                        hardware_details[hw] = {"status": "unavailable"}
                
                return jsonify({
                    "available_hardware": available_hardware,
                    "best_hardware": best_hardware,
                    "hardware_details": hardware_details
                })
            except Exception as e:
                logger.error(f"Hardware info failed: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def start_dashboard(self, background: bool = False):
        """Start the performance dashboard."""
        
        logger.info(f"Starting performance dashboard on port {self.port}")
        
        # Start background update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._background_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Initial data update
        self._update_dashboard_data()
        
        if self.app and flask:
            if background:
                # Run in background thread
                def run_flask():
                    self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
                
                flask_thread = threading.Thread(target=run_flask)
                flask_thread.daemon = True
                flask_thread.start()
                
                logger.info(f"Dashboard running at http://localhost:{self.port}")
            else:
                # Run in foreground
                self.app.run(host='0.0.0.0', port=self.port, debug=False)
        else:
            # CLI mode
            logger.info("Running dashboard in CLI mode (Flask not available)")
            self._run_cli_dashboard()
    
    def stop_dashboard(self):
        """Stop the dashboard and cleanup."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def _background_updates(self):
        """Background thread for periodic data updates."""
        
        while self.running:
            try:
                self._update_dashboard_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Background update failed: {e}")
                time.sleep(self.update_interval)
    
    def _update_dashboard_data(self):
        """Update dashboard data with current system status."""
        
        try:
            # System status
            available_hardware = self.detector.get_available_hardware()
            best_hardware = self.detector.get_best_available_hardware()
            
            self.dashboard_data["system_status"] = {
                "available_hardware": available_hardware,
                "best_hardware": best_hardware,
                "hardware_count": len(available_hardware),
                "timestamp": datetime.now().isoformat()
            }
            
            # Performance trends (mock data for now)
            self.dashboard_data["performance_trends"] = self._generate_mock_trends()
            
            # Optimization recommendations
            self.dashboard_data["optimization_recommendations"] = [
                f"System detected {len(available_hardware)} hardware platforms",
                f"Recommended hardware: {best_hardware}",
                "Consider running benchmark for detailed performance analysis"
            ]
            
            self.dashboard_data["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")
    
    def _update_benchmark_data(self, benchmark_run):
        """Update dashboard with new benchmark results."""
        
        # Add to recent benchmarks
        benchmark_summary = {
            "run_id": benchmark_run.run_id,
            "timestamp": benchmark_run.timestamp,
            "duration": benchmark_run.duration_seconds,
            "total_benchmarks": benchmark_run.summary.get("total_benchmarks", 0),
            "success_rate": benchmark_run.summary.get("statistics", {}).get("overall", {}).get("success_rate", 0)
        }
        
        self.dashboard_data["recent_benchmarks"].insert(0, benchmark_summary)
        
        # Keep only last 10 benchmarks
        self.dashboard_data["recent_benchmarks"] = self.dashboard_data["recent_benchmarks"][:10]
        
        # Update optimization recommendations
        if "optimization_recommendations" in benchmark_run.summary:
            self.dashboard_data["optimization_recommendations"] = benchmark_run.summary["optimization_recommendations"]
    
    def _generate_mock_trends(self) -> List[Dict[str, Any]]:
        """Generate mock performance trend data."""
        
        trends = []
        
        # Generate sample data points over the last 24 hours
        now = datetime.now()
        for i in range(24):
            timestamp = now - timedelta(hours=i)
            
            # Mock performance metrics
            trends.append({
                "timestamp": timestamp.isoformat(),
                "latency_ms": 50 + (i % 10) * 5 + (i % 3) * 2,
                "throughput": 100 - (i % 8) * 3,
                "memory_mb": 2000 + (i % 15) * 50,
                "cpu_usage": 30 + (i % 20) * 2
            })
        
        return list(reversed(trends))  # Chronological order
    
    def _run_cli_dashboard(self):
        """Run dashboard in command-line interface mode."""
        
        print("\n" + "="*80)
        print("🖥️  IPFS ACCELERATE PYTHON - PERFORMANCE DASHBOARD (CLI MODE)")
        print("="*80)
        
        while self.running:
            try:
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display dashboard
                self._print_cli_dashboard()
                
                # Wait for input or timeout
                print("\nPress 'q' to quit, 'b' to run benchmark, 'v' to validate system, or wait for auto-update...")
                
                # Simple input with timeout simulation
                import select
                import sys
                
                if sys.stdin in select.select([sys.stdin], [], [], 10):
                    choice = input().strip().lower()
                    
                    if choice == 'q':
                        break
                    elif choice == 'b':
                        self._cli_run_benchmark()
                    elif choice == 'v':
                        self._cli_validate_system()
                else:
                    # Auto-update timeout
                    continue
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"CLI dashboard error: {e}")
                time.sleep(5)
    
    def _print_cli_dashboard(self):
        """Print dashboard information to CLI."""
        
        data = self.dashboard_data
        
        print(f"\n🕐 Last Updated: {data.get('last_updated', 'Never')}")
        
        # System Status
        system = data.get("system_status", {})
        print(f"\n🖥️  System Status:")
        print(f"   Hardware Available: {system.get('hardware_count', 0)} platforms")
        print(f"   Recommended: {system.get('best_hardware', 'Unknown')}")
        print(f"   Platforms: {', '.join(system.get('available_hardware', []))}")
        
        # Recent Benchmarks
        benchmarks = data.get("recent_benchmarks", [])
        print(f"\n📊 Recent Benchmarks ({len(benchmarks)}):")
        
        if benchmarks:
            for i, bench in enumerate(benchmarks[:5], 1):
                timestamp = bench.get("timestamp", 0)
                duration = bench.get("duration", 0)
                success_rate = bench.get("success_rate", 0)
                
                print(f"   {i}. {time.ctime(timestamp)} - {duration:.1f}s ({success_rate:.0f}% success)")
        else:
            print("   No benchmarks run yet")
        
        # Optimization Recommendations
        recommendations = data.get("optimization_recommendations", [])
        print(f"\n💡 Recommendations:")
        for rec in recommendations[:5]:
            print(f"   • {rec}")
        
        # Performance Trends (simplified)
        trends = data.get("performance_trends", [])
        if trends:
            latest = trends[-1] if trends else {}
            print(f"\n📈 Current Performance:")
            print(f"   Latency: {latest.get('latency_ms', 0):.1f}ms")
            print(f"   Throughput: {latest.get('throughput', 0):.0f} req/sec")
            print(f"   Memory: {latest.get('memory_mb', 0):.0f}MB")
    
    def _cli_run_benchmark(self):
        """Run benchmark from CLI."""
        
        print("\n🚀 Running benchmark...")
        
        try:
            benchmark_run = run_quick_benchmark()
            self._update_benchmark_data(benchmark_run)
            
            print(f"✅ Benchmark completed in {benchmark_run.duration_seconds:.2f}s")
            print(f"   Results: {benchmark_run.summary.get('successful_benchmarks', 0)} successful")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            input("\nPress Enter to continue...")
    
    def _cli_validate_system(self):
        """Run system validation from CLI."""
        
        print("\n🔍 Running system validation...")
        
        try:
            report = run_production_validation("production")
            
            print(f"✅ Validation completed")
            print(f"   Overall Score: {report.overall_score:.1f}/100")
            print(f"   Recommendations: {len(report.deployment_recommendations)}")
            
            for rec in report.deployment_recommendations[:3]:
                print(f"   • {rec}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            input("\nPress Enter to continue...")
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for web dashboard."""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPFS Accelerate Python - Performance Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #333;
            margin: 0;
        }
        .header .subtitle {
            color: #666;
            margin-top: 5px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }
        .actions {
            text-align: center;
            margin-top: 30px;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .loading {
            display: none;
            color: #666;
        }
        .timestamp {
            color: #888;
            font-size: 12px;
            text-align: right;
            margin-top: 20px;
        }
    </style>
    <script>
        function runBenchmark() {
            const btn = document.getElementById('benchmarkBtn');
            const loading = document.getElementById('benchmarkLoading');
            
            btn.disabled = true;
            loading.style.display = 'block';
            
            fetch('/api/run_benchmark', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Benchmark completed in ${data.duration.toFixed(2)}s`);
                        location.reload();
                    } else {
                        alert(`Benchmark failed: ${data.error}`);
                    }
                })
                .catch(error => {
                    alert(`Error: ${error}`);
                })
                .finally(() => {
                    btn.disabled = false;
                    loading.style.display = 'none';
                });
        }
        
        function validateSystem() {
            const btn = document.getElementById('validateBtn');
            const loading = document.getElementById('validateLoading');
            
            btn.disabled = true;
            loading.style.display = 'block';
            
            fetch('/api/validate_system')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`System validation completed. Score: ${data.score.toFixed(1)}/100`);
                        location.reload();
                    } else {
                        alert(`Validation failed: ${data.error}`);
                    }
                })
                .catch(error => {
                    alert(`Error: ${error}`);
                })
                .finally(() => {
                    btn.disabled = false;
                    loading.style.display = 'none';
                });
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IPFS Accelerate Python</h1>
            <div class="subtitle">Performance Dashboard</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🖥️ System Status</h3>
                {% if system_status %}
                <div class="metric">
                    <span>Hardware Platforms:</span>
                    <span class="status-good">{{ system_status.get('hardware_count', 0) }}</span>
                </div>
                <div class="metric">
                    <span>Recommended:</span>
                    <span>{{ system_status.get('best_hardware', 'Unknown') }}</span>
                </div>
                <div class="metric">
                    <span>Available:</span>
                    <span>{{ ', '.join(system_status.get('available_hardware', [])) }}</span>
                </div>
                {% else %}
                <p>System status not available</p>
                {% endif %}
            </div>
            
            <div class="card">
                <h3>📊 Recent Benchmarks</h3>
                {% if recent_benchmarks %}
                {% for benchmark in recent_benchmarks[:5] %}
                <div class="metric">
                    <span>{{ benchmark.get('run_id', 'Unknown')[:12] }}...</span>
                    <span class="status-good">{{ benchmark.get('success_rate', 0)|round }}%</span>
                </div>
                {% endfor %}
                {% else %}
                <p>No benchmarks run yet</p>
                {% endif %}
            </div>
            
            <div class="card">
                <h3>💡 Recommendations</h3>
                {% if optimization_recommendations %}
                {% for rec in optimization_recommendations[:5] %}
                <div style="margin: 10px 0; padding: 8px; background: white; border-radius: 4px;">
                    • {{ rec }}
                </div>
                {% endfor %}
                {% else %}
                <p>Run benchmark for recommendations</p>
                {% endif %}
            </div>
            
            <div class="card">
                <h3>📈 Performance Trends</h3>
                {% if performance_trends %}
                {% set latest = performance_trends[-1] %}
                <div class="metric">
                    <span>Latency:</span>
                    <span>{{ latest.get('latency_ms', 0)|round }}ms</span>
                </div>
                <div class="metric">
                    <span>Throughput:</span>
                    <span>{{ latest.get('throughput', 0)|round }} req/sec</span>
                </div>
                <div class="metric">
                    <span>Memory:</span>
                    <span>{{ latest.get('memory_mb', 0)|round }}MB</span>
                </div>
                {% else %}
                <p>Performance trends not available</p>
                {% endif %}
            </div>
        </div>
        
        <div class="actions">
            <button id="benchmarkBtn" class="btn" onclick="runBenchmark()">
                🚀 Run Benchmark
            </button>
            <div id="benchmarkLoading" class="loading">Running benchmark...</div>
            
            <button id="validateBtn" class="btn btn-success" onclick="validateSystem()">
                🔍 Validate System
            </button>
            <div id="validateLoading" class="loading">Validating system...</div>
        </div>
        
        <div class="timestamp">
            Last updated: {{ last_updated or 'Never' }}
        </div>
    </div>
</body>
</html>
        """

def start_performance_dashboard(port: int = 8080, background: bool = False) -> PerformanceDashboard:
    """Start the performance dashboard."""
    
    dashboard = PerformanceDashboard(port)
    dashboard.start_dashboard(background)
    return dashboard

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPFS Accelerate Python Performance Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port for web dashboard")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode only")
    parser.add_argument("--update-interval", type=int, default=30, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        dashboard = PerformanceDashboard(args.port, args.update_interval)
        
        if args.cli or not flask:
            # Force CLI mode
            dashboard.app = None
            dashboard.start_dashboard()
        else:
            print(f"🚀 Starting web dashboard at http://localhost:{args.port}")
            dashboard.start_dashboard()
            
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        sys.exit(1)