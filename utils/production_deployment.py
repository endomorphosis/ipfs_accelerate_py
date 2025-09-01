#!/usr/bin/env python3
"""
Production Deployment Tools for IPFS Accelerate Python

This module provides comprehensive tools for production deployment including
automated deployment scripts, monitoring setup, and maintenance utilities.
"""

import os
import sys
import json
import time
import logging
import subprocess
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile

# Safe imports
try:
    from .safe_imports import safe_import, get_import_summary
    from .production_validation import run_production_validation, ValidationLevel
    from .advanced_benchmarking import run_quick_benchmark
    from ..hardware_detection import HardwareDetector
except ImportError:
    # Fallback imports for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.safe_imports import safe_import, get_import_summary
    from utils.production_validation import run_production_validation, ValidationLevel
    from utils.advanced_benchmarking import run_quick_benchmark
    from hardware_detection import HardwareDetector

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    environment: str  # development, staging, production
    install_mode: str  # minimal, full, webnn, all
    enable_monitoring: bool
    enable_benchmarking: bool
    enable_dashboard: bool
    port: int
    log_level: str
    optimization_target: str  # speed, memory, power, balanced

@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    success: bool
    environment: str
    timestamp: float
    validation_score: float
    installed_components: List[str]
    configuration: Dict[str, Any]
    recommendations: List[str]
    errors: List[str]
    log_file: Optional[str]

class ProductionDeploymentManager:
    """Comprehensive production deployment manager."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.deployment_dir = self.base_dir / "deployments"
        self.deployment_dir.mkdir(exist_ok=True)
        
        self.detector = HardwareDetector()
        self.deployment_history = []
        
    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy IPFS Accelerate Python with specified configuration."""
        
        logger.info(f"Starting deployment for {config.environment} environment")
        start_time = time.time()
        
        errors = []
        installed_components = []
        recommendations = []
        
        # Create deployment directory
        deployment_id = f"deploy_{config.environment}_{int(start_time)}"
        deploy_dir = self.deployment_dir / deployment_id
        deploy_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = deploy_dir / "deployment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, config.log_level.upper()))
        logger.addHandler(file_handler)
        
        try:
            # Step 1: Pre-deployment validation
            logger.info("Step 1: Running pre-deployment validation")
            validation_level = ValidationLevel.ENTERPRISE if config.environment == "production" else ValidationLevel.PRODUCTION
            validation_report = run_production_validation(validation_level.value)
            
            if validation_report.overall_score < 70:
                errors.append(f"System validation score too low: {validation_report.overall_score:.1f}/100")
                recommendations.extend(validation_report.deployment_recommendations)
            
            # Step 2: Install dependencies
            logger.info("Step 2: Installing dependencies")
            install_success = self._install_dependencies(config.install_mode, deploy_dir)
            if install_success:
                installed_components.append(f"dependencies_{config.install_mode}")
            else:
                errors.append("Failed to install dependencies")
            
            # Step 3: Configure environment
            logger.info("Step 3: Configuring environment")
            env_config = self._configure_environment(config, deploy_dir)
            if env_config:
                installed_components.append("environment_configuration")
            else:
                errors.append("Failed to configure environment")
            
            # Step 4: Setup monitoring (if enabled)
            if config.enable_monitoring:
                logger.info("Step 4: Setting up monitoring")
                monitoring_success = self._setup_monitoring(config, deploy_dir)
                if monitoring_success:
                    installed_components.append("monitoring")
                else:
                    errors.append("Failed to setup monitoring")
            
            # Step 5: Setup dashboard (if enabled)
            if config.enable_dashboard:
                logger.info("Step 5: Setting up dashboard")
                dashboard_success = self._setup_dashboard(config, deploy_dir)
                if dashboard_success:
                    installed_components.append("dashboard")
                else:
                    errors.append("Failed to setup dashboard")
            
            # Step 6: Run benchmark (if enabled)
            if config.enable_benchmarking:
                logger.info("Step 6: Running deployment benchmark")
                benchmark_success = self._run_deployment_benchmark(deploy_dir)
                if benchmark_success:
                    installed_components.append("benchmark")
                else:
                    errors.append("Failed to run deployment benchmark")
            
            # Step 7: Generate deployment documentation
            logger.info("Step 7: Generating deployment documentation")
            self._generate_deployment_docs(config, deploy_dir, validation_report)
            installed_components.append("documentation")
            
            # Step 8: Create deployment scripts
            logger.info("Step 8: Creating deployment scripts")
            self._create_deployment_scripts(config, deploy_dir)
            installed_components.append("deployment_scripts")
            
            success = len(errors) == 0
            
            if success:
                logger.info("✅ Deployment completed successfully")
            else:
                logger.warning(f"⚠️  Deployment completed with {len(errors)} errors")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            errors.append(f"Deployment exception: {e}")
            success = False
        
        finally:
            # Remove file handler
            logger.removeHandler(file_handler)
        
        # Create deployment result
        result = DeploymentResult(
            success=success,
            environment=config.environment,
            timestamp=start_time,
            validation_score=validation_report.overall_score if 'validation_report' in locals() else 0.0,
            installed_components=installed_components,
            configuration=asdict(config),
            recommendations=recommendations,
            errors=errors,
            log_file=str(log_file) if log_file.exists() else None
        )
        
        # Save deployment result
        self._save_deployment_result(result, deploy_dir)
        self.deployment_history.append(result)
        
        return result
    
    def _install_dependencies(self, install_mode: str, deploy_dir: Path) -> bool:
        """Install dependencies based on install mode."""
        
        try:
            # Create requirements file based on mode
            requirements_file = deploy_dir / "requirements.txt"
            
            if install_mode == "minimal":
                requirements = [
                    "aiohttp>=3.8.1",
                    "duckdb>=0.7.0",
                    "tqdm>=4.64.0",
                    "numpy>=1.23.0"
                ]
            elif install_mode == "full":
                requirements = [
                    "aiohttp>=3.8.1",
                    "duckdb>=0.7.0",
                    "tqdm>=4.64.0",
                    "numpy>=1.23.0",
                    "torch>=2.1.0",
                    "transformers>=4.46.0",
                    "uvicorn>=0.27.0",
                    "fastapi>=0.110.0"
                ]
            elif install_mode == "webnn":
                requirements = [
                    "aiohttp>=3.8.1",
                    "duckdb>=0.7.0",
                    "tqdm>=4.64.0",
                    "numpy>=1.23.0",
                    "onnxruntime-web>=1.15.0"
                ]
            else:  # all
                requirements = [
                    "aiohttp>=3.8.1",
                    "duckdb>=0.7.0",
                    "tqdm>=4.64.0",
                    "numpy>=1.23.0",
                    "torch>=2.1.0",
                    "transformers>=4.46.0",
                    "uvicorn>=0.27.0",
                    "fastapi>=0.110.0",
                    "onnxruntime-web>=1.15.0",
                    "flask>=2.3.0"
                ]
            
            requirements_file.write_text("\n".join(requirements))
            
            # Create virtual environment
            venv_dir = deploy_dir / "venv"
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_dir)
            ], check=True)
            
            # Install requirements
            pip_path = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip"
            subprocess.run([
                str(pip_path), "install", "-r", str(requirements_file)
            ], check=True)
            
            logger.info(f"✅ Dependencies installed successfully ({install_mode} mode)")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Dependency installation error: {e}")
            return False
    
    def _configure_environment(self, config: DeploymentConfig, deploy_dir: Path) -> bool:
        """Configure environment variables and settings."""
        
        try:
            # Create environment configuration file
            env_config = {
                "IPFS_ACCELERATE_ENV": config.environment,
                "IPFS_ACCELERATE_LOG_LEVEL": config.log_level.upper(),
                "IPFS_ACCELERATE_OPTIMIZATION_TARGET": config.optimization_target,
                "IPFS_ACCELERATE_PORT": str(config.port),
                "IPFS_ACCELERATE_ENABLE_MONITORING": str(config.enable_monitoring).lower(),
                "IPFS_ACCELERATE_ENABLE_DASHBOARD": str(config.enable_dashboard).lower(),
            }
            
            # Add hardware-specific configuration
            available_hardware = self.detector.get_available_hardware()
            best_hardware = self.detector.get_best_available_hardware()
            
            env_config["IPFS_ACCELERATE_AVAILABLE_HARDWARE"] = ",".join(available_hardware)
            env_config["IPFS_ACCELERATE_BEST_HARDWARE"] = best_hardware
            
            # Create .env file
            env_file = deploy_dir / ".env"
            env_content = []
            for key, value in env_config.items():
                env_content.append(f"{key}={value}")
            
            env_file.write_text("\n".join(env_content))
            
            # Create Python configuration file
            config_file = deploy_dir / "config.py"
            config_content = f'''#!/usr/bin/env python3
"""
IPFS Accelerate Python - Production Configuration
Generated automatically during deployment.
"""

import os

# Environment
ENVIRONMENT = "{config.environment}"
LOG_LEVEL = "{config.log_level.upper()}"
OPTIMIZATION_TARGET = "{config.optimization_target}"

# Server
PORT = {config.port}

# Features  
ENABLE_MONITORING = {config.enable_monitoring}
ENABLE_DASHBOARD = {config.enable_dashboard}
ENABLE_BENCHMARKING = {config.enable_benchmarking}

# Hardware
AVAILABLE_HARDWARE = {available_hardware}
BEST_HARDWARE = "{best_hardware}"

# Override with environment variables
for key, value in os.environ.items():
    if key.startswith("IPFS_ACCELERATE_"):
        config_key = key.replace("IPFS_ACCELERATE_", "").lower()
        globals()[config_key.upper()] = value
'''
            
            config_file.write_text(config_content)
            
            logger.info("✅ Environment configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment configuration failed: {e}")
            return False
    
    def _setup_monitoring(self, config: DeploymentConfig, deploy_dir: Path) -> bool:
        """Setup monitoring and logging infrastructure."""
        
        try:
            # Create monitoring directory
            monitoring_dir = deploy_dir / "monitoring"
            monitoring_dir.mkdir(exist_ok=True)
            
            # Create logging configuration
            logging_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    },
                    "detailed": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
                    }
                },
                "handlers": {
                    "console": {
                        "level": config.log_level.upper(),
                        "class": "logging.StreamHandler",
                        "formatter": "standard"
                    },
                    "file": {
                        "level": "INFO",
                        "class": "logging.FileHandler",
                        "filename": str(monitoring_dir / "application.log"),
                        "formatter": "detailed"
                    }
                },
                "loggers": {
                    "": {
                        "handlers": ["console", "file"],
                        "level": config.log_level.upper(),
                        "propagate": False
                    }
                }
            }
            
            logging_config_file = monitoring_dir / "logging.json"
            with open(logging_config_file, 'w') as f:
                json.dump(logging_config, f, indent=2)
            
            # Create monitoring script
            monitoring_script = monitoring_dir / "monitor.py"
            monitoring_script_content = '''#!/usr/bin/env python3
"""
IPFS Accelerate Python - Production Monitoring Script
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Import production validation
try:
    from utils.production_validation import run_production_validation
    from utils.advanced_benchmarking import run_quick_benchmark
except ImportError:
    print("Warning: Production modules not available")
    run_production_validation = None
    run_quick_benchmark = None

def monitor_system():
    """Monitor system performance and health."""
    
    while True:
        try:
            timestamp = datetime.now().isoformat()
            
            # Run quick validation
            if run_production_validation:
                report = run_production_validation("basic")
                
                monitor_data = {
                    "timestamp": timestamp,
                    "validation_score": report.overall_score,
                    "system_info": report.system_info,
                    "recommendations": report.deployment_recommendations[:3]
                }
                
                # Save monitoring data
                monitor_file = Path("monitoring_data.jsonl")
                with open(monitor_file, "a") as f:
                    f.write(json.dumps(monitor_data) + "\\n")
                
                print(f"[{timestamp}] System score: {report.overall_score:.1f}/100")
            
            # Wait 5 minutes
            time.sleep(300)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_system()
'''
            
            monitoring_script.write_text(monitoring_script_content)
            monitoring_script.chmod(0o755)
            
            logger.info("✅ Monitoring setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False
    
    def _setup_dashboard(self, config: DeploymentConfig, deploy_dir: Path) -> bool:
        """Setup performance dashboard."""
        
        try:
            # Create dashboard directory
            dashboard_dir = deploy_dir / "dashboard"
            dashboard_dir.mkdir(exist_ok=True)
            
            # Create dashboard startup script
            dashboard_script = dashboard_dir / "start_dashboard.py"
            dashboard_script_content = f'''#!/usr/bin/env python3
"""
IPFS Accelerate Python - Production Dashboard Startup
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.performance_dashboard import start_performance_dashboard
    
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        
        print(f"🚀 Starting IPFS Accelerate dashboard on port {config.port}")
        dashboard = start_performance_dashboard(port={config.port})
        
except ImportError as e:
    print(f"Dashboard not available: {{e}}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\\n⏹️  Dashboard stopped")
except Exception as e:
    print(f"Dashboard failed: {{e}}")
    sys.exit(1)
'''
            
            dashboard_script.write_text(dashboard_script_content)
            dashboard_script.chmod(0o755)
            
            logger.info("✅ Dashboard setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard setup failed: {e}")
            return False
    
    def _run_deployment_benchmark(self, deploy_dir: Path) -> bool:
        """Run deployment benchmark to validate performance."""
        
        try:
            # Create benchmark directory
            benchmark_dir = deploy_dir / "benchmarks"
            benchmark_dir.mkdir(exist_ok=True)
            
            # Run quick benchmark
            benchmark_run = run_quick_benchmark()
            
            # Save benchmark results
            benchmark_file = benchmark_dir / "deployment_benchmark.json"
            with open(benchmark_file, 'w') as f:
                json.dump(asdict(benchmark_run), f, indent=2, default=str)
            
            # Create benchmark summary
            summary = benchmark_run.summary
            success_rate = summary.get("statistics", {}).get("overall", {}).get("success_rate", 0)
            
            if success_rate >= 80:
                logger.info(f"✅ Deployment benchmark passed ({success_rate:.0f}% success rate)")
                return True
            else:
                logger.warning(f"⚠️  Deployment benchmark concerns ({success_rate:.0f}% success rate)")
                return True  # Don't fail deployment for benchmark issues
                
        except Exception as e:
            logger.error(f"Deployment benchmark failed: {e}")
            return False
    
    def _generate_deployment_docs(self, config: DeploymentConfig, deploy_dir: Path, validation_report) -> None:
        """Generate deployment documentation."""
        
        docs_dir = deploy_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_content = f'''# IPFS Accelerate Python - Production Deployment

## Deployment Information

- **Environment**: {config.environment}
- **Install Mode**: {config.install_mode}  
- **Deployment Date**: {time.ctime()}
- **Validation Score**: {validation_report.overall_score:.1f}/100

## Configuration

- **Port**: {config.port}
- **Log Level**: {config.log_level}
- **Optimization Target**: {config.optimization_target}
- **Monitoring Enabled**: {config.enable_monitoring}
- **Dashboard Enabled**: {config.enable_dashboard}
- **Benchmarking Enabled**: {config.enable_benchmarking}

## System Information

- **Platform**: {validation_report.system_info.get('platform', 'Unknown')}
- **Python Version**: {validation_report.python_info.get('version', 'Unknown').split()[0]}
- **Available Hardware**: {', '.join(validation_report.hardware_capabilities.get('available_hardware', []))}
- **Recommended Hardware**: {validation_report.hardware_capabilities.get('best_hardware', 'Unknown')}

## Quick Start

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate     # Windows
   ```

2. Start the application:
   ```bash
   python -m ipfs_accelerate_py
   ```

3. (Optional) Start monitoring:
   ```bash
   python monitoring/monitor.py &
   ```

4. (Optional) Start dashboard:
   ```bash
   python dashboard/start_dashboard.py
   ```

## Deployment Recommendations

{chr(10).join(f"- {rec}" for rec in validation_report.deployment_recommendations)}

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in the virtual environment
2. **Port Conflicts**: Change the port in config.py if {config.port} is in use
3. **Hardware Issues**: Check hardware compatibility with the benchmark suite
4. **Performance Issues**: Review optimization recommendations and adjust configuration

### Support

- Check the logs in `monitoring/application.log`
- Run validation: `python -c "from utils.production_validation import *; run_production_validation('production')"`
- Run benchmark: `python -c "from utils.advanced_benchmarking import *; run_quick_benchmark()"`

## Maintenance

### Regular Tasks

1. **Weekly**: Review monitoring logs and performance trends
2. **Monthly**: Update dependencies and run full validation
3. **Quarterly**: Review hardware optimization and update configuration

### Updates

To update the deployment:

1. Backup current configuration
2. Pull latest code changes
3. Run new deployment with same configuration
4. Compare validation scores and performance
'''
        
        readme_file = docs_dir / "README.md"
        readme_file.write_text(readme_content)
        
        # Create operational runbook
        runbook_content = f'''# IPFS Accelerate Python - Operational Runbook

## Daily Operations

### Health Checks

```bash
# Quick system validation
python -c "from utils.production_validation import *; print(run_production_validation('basic').overall_score)"

# Check application status
curl http://localhost:{config.port}/api/status

# Review recent logs
tail -n 50 monitoring/application.log
```

### Performance Monitoring

```bash
# Quick benchmark
python -c "from utils.advanced_benchmarking import *; run_quick_benchmark()"

# Check hardware utilization
python -c "from hardware_detection import *; print(HardwareDetector().get_available_hardware())"
```

## Incident Response

### High Latency

1. Check hardware utilization
2. Review recent configuration changes
3. Run performance benchmark
4. Consider hardware optimization

### Memory Issues

1. Check current memory usage
2. Review model loading patterns
3. Consider precision optimization (fp16/int8)
4. Restart services if necessary

### Hardware Failures

1. Check hardware detection status
2. Fallback to CPU if accelerators fail  
3. Update hardware configuration
4. Monitor performance impact

## Maintenance Procedures

### Weekly Maintenance

```bash
# Clean old logs (keep last 7 days)
find monitoring/ -name "*.log" -mtime +7 -delete

# Update performance baseline
python -c "from utils.advanced_benchmarking import *; run_quick_benchmark()"
```

### Monthly Maintenance

```bash
# Full system validation
python utils/production_validation.py --level enterprise

# Dependency audit
pip list --outdated

# Performance regression check
python utils/advanced_benchmarking.py --full
```

## Emergency Procedures

### System Down

1. Check process status: `ps aux | grep python`
2. Review error logs: `tail monitoring/application.log`
3. Restart services: `./scripts/restart.sh`
4. Validate recovery: `python utils/production_validation.py`

### Data Corruption

1. Stop all services
2. Restore from backup
3. Validate data integrity
4. Restart services
5. Monitor for stability

## Contact Information

- **Deployment Date**: {time.ctime()}
- **Environment**: {config.environment}
- **Configuration**: See config.py for current settings
'''
        
        runbook_file = docs_dir / "RUNBOOK.md"
        runbook_file.write_text(runbook_content)
    
    def _create_deployment_scripts(self, config: DeploymentConfig, deploy_dir: Path) -> None:
        """Create deployment and management scripts."""
        
        scripts_dir = deploy_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Create startup script
        startup_script = scripts_dir / "start.sh"
        startup_script_content = f'''#!/bin/bash
# IPFS Accelerate Python - Production Startup Script

set -e

echo "🚀 Starting IPFS Accelerate Python ({config.environment})"

# Load environment variables
if [ -f .env ]; then
    source .env
    echo "✅ Environment variables loaded"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Start monitoring (if enabled)
if [ "{config.enable_monitoring}" = "True" ]; then
    echo "📊 Starting monitoring..."
    python monitoring/monitor.py &
    MONITOR_PID=$!
    echo $MONITOR_PID > monitoring.pid
    echo "✅ Monitoring started (PID: $MONITOR_PID)"
fi

# Start dashboard (if enabled)  
if [ "{config.enable_dashboard}" = "True" ]; then
    echo "🖥️  Starting dashboard..."
    python dashboard/start_dashboard.py &
    DASHBOARD_PID=$!
    echo $DASHBOARD_PID > dashboard.pid
    echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
fi

# Start main application
echo "🚀 Starting main application..."
python -m ipfs_accelerate_py

echo "🛑 Application stopped"
'''
        
        startup_script.write_text(startup_script_content)
        startup_script.chmod(0o755)
        
        # Create shutdown script
        shutdown_script = scripts_dir / "stop.sh"
        shutdown_script_content = '''#!/bin/bash
# IPFS Accelerate Python - Production Shutdown Script

echo "🛑 Stopping IPFS Accelerate Python services"

# Stop monitoring
if [ -f monitoring.pid ]; then
    MONITOR_PID=$(cat monitoring.pid)
    echo "📊 Stopping monitoring (PID: $MONITOR_PID)"
    kill $MONITOR_PID 2>/dev/null || echo "  Monitoring process not found"
    rm -f monitoring.pid
fi

# Stop dashboard
if [ -f dashboard.pid ]; then
    DASHBOARD_PID=$(cat dashboard.pid)
    echo "🖥️  Stopping dashboard (PID: $DASHBOARD_PID)"
    kill $DASHBOARD_PID 2>/dev/null || echo "  Dashboard process not found"
    rm -f dashboard.pid
fi

echo "✅ All services stopped"
'''
        
        shutdown_script.write_text(shutdown_script_content)
        shutdown_script.chmod(0o755)
        
        # Create restart script
        restart_script = scripts_dir / "restart.sh"
        restart_script_content = '''#!/bin/bash
# IPFS Accelerate Python - Production Restart Script

echo "🔄 Restarting IPFS Accelerate Python"

./scripts/stop.sh
sleep 2
./scripts/start.sh
'''
        
        restart_script.write_text(restart_script_content)
        restart_script.chmod(0o755)
        
        # Create health check script
        health_script = scripts_dir / "health_check.sh"
        health_script_content = f'''#!/bin/bash
# IPFS Accelerate Python - Health Check Script

echo "🔍 IPFS Accelerate Python Health Check"
echo "========================================"

# Check virtual environment
if [ -f venv/bin/activate ]; then
    echo "✅ Virtual environment: Available"
    source venv/bin/activate
else
    echo "❌ Virtual environment: Missing"
    exit 1
fi

# Check Python imports
python -c "import sys; print(f'✅ Python {{sys.version.split()[0]}}: Available')" 2>/dev/null || {{
    echo "❌ Python: Import failed"
    exit 1
}}

# Check core modules
python -c "from hardware_detection import HardwareDetector; print('✅ Core modules: Available')" 2>/dev/null || {{
    echo "❌ Core modules: Import failed"  
    exit 1
}}

# Check hardware detection
python -c "from hardware_detection import HardwareDetector; hw=HardwareDetector().get_available_hardware(); print(f'✅ Hardware detection: {{len(hw)}} platforms')" 2>/dev/null || {{
    echo "❌ Hardware detection: Failed"
    exit 1
}}

# Check port availability (if dashboard enabled)
if [ "{config.enable_dashboard}" = "True" ]; then
    if netstat -tuln | grep -q ":{config.port} "; then
        echo "✅ Dashboard port {config.port}: In use"
    else
        echo "⚠️  Dashboard port {config.port}: Available"
    fi
fi

echo "========================================"
echo "✅ Health check completed successfully"
'''
        
        health_script.write_text(health_script_content)
        health_script.chmod(0o755)
    
    def _save_deployment_result(self, result: DeploymentResult, deploy_dir: Path) -> None:
        """Save deployment result to file."""
        
        try:
            result_file = deploy_dir / "deployment_result.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # Also save to deployment history
            history_file = self.deployment_dir / "deployment_history.jsonl"
            with open(history_file, 'a') as f:
                f.write(json.dumps(asdict(result), default=str) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save deployment result: {e}")

def create_deployment_config(
    environment: str = "production",
    install_mode: str = "full",
    enable_monitoring: bool = True,
    enable_dashboard: bool = True,
    enable_benchmarking: bool = True,
    port: int = 8080,
    log_level: str = "INFO",
    optimization_target: str = "balanced"
) -> DeploymentConfig:
    """Create deployment configuration with sensible defaults."""
    
    return DeploymentConfig(
        environment=environment,
        install_mode=install_mode,
        enable_monitoring=enable_monitoring,
        enable_dashboard=enable_dashboard,
        enable_benchmarking=enable_benchmarking,
        port=port,
        log_level=log_level,
        optimization_target=optimization_target
    )

def deploy_production(
    environment: str = "production",
    install_mode: str = "full",
    base_dir: Optional[str] = None
) -> DeploymentResult:
    """Deploy IPFS Accelerate Python for production use."""
    
    config = create_deployment_config(
        environment=environment,
        install_mode=install_mode
    )
    
    manager = ProductionDeploymentManager(base_dir)
    return manager.deploy(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPFS Accelerate Python Production Deployment")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="production", help="Deployment environment")
    parser.add_argument("--install-mode", choices=["minimal", "full", "webnn", "all"],
                       default="full", help="Installation mode")
    parser.add_argument("--base-dir", help="Base directory for deployment")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    parser.add_argument("--no-benchmarking", action="store_true", help="Disable benchmarking")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        config = DeploymentConfig(
            environment=args.environment,
            install_mode=args.install_mode,
            enable_monitoring=not args.no_monitoring,
            enable_dashboard=not args.no_dashboard,
            enable_benchmarking=not args.no_benchmarking,
            port=args.port,
            log_level=args.log_level,
            optimization_target="balanced"
        )
        
        manager = ProductionDeploymentManager(args.base_dir)
        result = manager.deploy(config)
        
        if result.success:
            print(f"\n✅ Deployment successful!")
            print(f"📁 Deployment directory: {manager.deployment_dir}")
            print(f"📊 Validation score: {result.validation_score:.1f}/100")
            print(f"🔧 Installed components: {', '.join(result.installed_components)}")
            
            if result.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in result.recommendations[:3]:
                    print(f"  • {rec}")
        else:
            print(f"\n❌ Deployment failed!")
            print(f"🐛 Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"  • {error}")
            
        if result.log_file:
            print(f"\n📋 Full log: {result.log_file}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        logging.exception("Deployment error details:")
        sys.exit(1)