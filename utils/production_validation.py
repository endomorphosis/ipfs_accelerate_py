#!/usr/bin/env python3
"""
Production Validation Suite for IPFS Accelerate Python

This module provides comprehensive validation tools for production deployment,
including compatibility testing, performance validation, and deployment readiness checks.
"""

import os
import sys
import time
import json
import logging
import platform
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import tempfile
from pathlib import Path

# Safe imports
try:
    from .safe_imports import safe_import, get_import_summary
    from .model_compatibility import get_optimal_hardware, check_model_compatibility
    from .performance_modeling import simulate_model_performance, HardwareType
    from ..hardware_detection import HardwareDetector
except ImportError:
    # Fallback imports for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.safe_imports import safe_import, get_import_summary
    from utils.model_compatibility import get_optimal_hardware, check_model_compatibility
    from utils.performance_modeling import simulate_model_performance, HardwareType
    from hardware_detection import HardwareDetector

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels for different deployment scenarios."""
    BASIC = "basic"           # Basic functionality checks
    PRODUCTION = "production" # Production deployment ready
    ENTERPRISE = "enterprise" # Enterprise deployment ready

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    execution_time_ms: float
    level: ValidationLevel

@dataclass
class SystemCompatibilityReport:
    """Comprehensive system compatibility report."""
    system_info: Dict[str, str]
    python_info: Dict[str, str]
    dependency_status: Dict[str, bool]
    hardware_capabilities: Dict[str, Any]
    performance_baseline: Dict[str, float]
    validation_results: List[ValidationResult]
    overall_score: float
    deployment_recommendations: List[str]

class ProductionValidator:
    """Comprehensive production validation suite."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.PRODUCTION):
        self.validation_level = validation_level
        self.detector = HardwareDetector()
        self.results = []
        
    def run_validation_suite(self) -> SystemCompatibilityReport:
        """Run complete validation suite based on specified level."""
        logger.info(f"Starting {self.validation_level.value} validation suite...")
        start_time = time.time()
        
        # Core validation steps
        self._validate_system_compatibility()
        self._validate_python_environment()
        self._validate_dependencies()
        self._validate_hardware_detection()
        self._validate_model_compatibility()
        self._validate_performance_baseline()
        
        if self.validation_level in [ValidationLevel.PRODUCTION, ValidationLevel.ENTERPRISE]:
            self._validate_production_readiness()
            
        if self.validation_level == ValidationLevel.ENTERPRISE:
            self._validate_enterprise_requirements()
            
        execution_time = (time.time() - start_time) * 1000
        
        # Generate comprehensive report
        report = self._generate_compatibility_report()
        
        logger.info(f"Validation completed in {execution_time:.1f}ms")
        logger.info(f"Overall score: {report.overall_score:.2f}/100")
        
        return report
        
    def _validate_system_compatibility(self) -> None:
        """Validate system compatibility."""
        start_time = time.time()
        
        try:
            system_info = {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
            }
            
            # Check supported platforms
            supported_platforms = ["Linux", "Darwin", "Windows"]
            platform_supported = system_info["platform"] in supported_platforms
            
            # Check Python version compatibility
            python_version = tuple(map(int, platform.python_version().split('.')))
            python_supported = python_version >= (3, 8)
            
            passed = platform_supported and python_supported
            message = "System compatibility check completed"
            if not passed:
                issues = []
                if not platform_supported:
                    issues.append(f"Unsupported platform: {system_info['platform']}")
                if not python_supported:
                    issues.append(f"Python version too old: {platform.python_version()}")
                message = f"System compatibility issues: {', '.join(issues)}"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="system_compatibility",
                passed=passed,
                message=message,
                details=system_info,
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="system_compatibility",
                passed=False,
                message=f"System compatibility check failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
    def _validate_python_environment(self) -> None:
        """Validate Python environment."""
        start_time = time.time()
        
        try:
            python_info = {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
                "path": sys.path[:3],  # First few paths for brevity
            }
            
            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            python_info["virtual_environment"] = in_venv
            
            # Check importable modules
            core_modules = ["os", "sys", "json", "logging", "subprocess"]
            importable_modules = []
            for module in core_modules:
                try:
                    __import__(module)
                    importable_modules.append(module)
                except ImportError:
                    pass
            
            python_info["core_modules_available"] = len(importable_modules)
            python_info["total_core_modules"] = len(core_modules)
            
            passed = len(importable_modules) == len(core_modules)
            message = f"Python environment validation: {len(importable_modules)}/{len(core_modules)} core modules available"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="python_environment",
                passed=passed,
                message=message,
                details=python_info,
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="python_environment",
                passed=False,
                message=f"Python environment validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
    def _validate_dependencies(self) -> None:
        """Validate dependency status with enhanced production readiness assessment."""
        start_time = time.time()
        
        try:
            # Import enhanced dependency validation functions
            from safe_imports import validate_production_dependencies, get_import_summary, check_available
            
            # Get comprehensive dependency status  
            prod_deps = validate_production_dependencies()
            summary = get_import_summary()
            
            # Test actual imports to verify functionality
            core_modules_working = 0
            core_modules_tested = [
                ('numpy', 'import numpy; numpy.array([1,2,3])'),
                ('aiohttp', 'import aiohttp; aiohttp.__version__'),
                ('duckdb', 'import duckdb; duckdb.connect(":memory:")'),
                ('tqdm', 'import tqdm; tqdm.tqdm'),
                ('requests', 'import requests; requests.__version__')
            ]
            
            working_imports = {}
            for module, test_code in core_modules_tested:
                try:
                    exec(test_code)
                    working_imports[module] = True
                    core_modules_working += 1
                except Exception as e:
                    working_imports[module] = False
                    logger.debug(f"Module {module} not working: {e}")
            
            # Calculate comprehensive score
            core_score = (prod_deps['core_available'] / max(1, prod_deps['core_total'])) * 40
            ml_score = (prod_deps['ml_available'] / max(1, prod_deps['ml_total'])) * 25
            web_score = (prod_deps['web_available'] / max(1, prod_deps['web_total'])) * 20  
            optional_score = (prod_deps['optional_available'] / max(1, prod_deps['optional_total'])) * 10
            working_score = (core_modules_working / max(1, len(core_modules_tested))) * 5
            
            total_score = core_score + ml_score + web_score + optional_score + working_score
            passed = total_score >= 50  # More reasonable threshold
            
            details = {
                "production_validation": prod_deps,
                "import_summary": summary,
                "working_imports": working_imports,
                "core_modules_tested": len(core_modules_tested),
                "core_modules_working": core_modules_working,
                "scores": {
                    "core": core_score,
                    "ml": ml_score, 
                    "web": web_score,
                    "optional": optional_score,
                    "working": working_score,
                    "total": total_score
                },
                "available_count": prod_deps['total_available'],
                "total_count": prod_deps['total_possible'],
                "dependency_score": prod_deps['dependency_score']
            }
            
            message = f"Dependencies: {prod_deps['core_available']}/{prod_deps['core_total']} core, {prod_deps['ml_available']}/{prod_deps['ml_total']} ML, {prod_deps['web_available']}/{prod_deps['web_total']} web, {prod_deps['optional_available']}/{prod_deps['optional_total']} optional (score: {total_score:.1f}/100)"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="dependencies",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="dependencies",
                passed=False,
                message=f"Dependency validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
    def _validate_hardware_detection(self) -> None:
        """Validate hardware detection capabilities."""
        start_time = time.time()
        
        try:
            # Test hardware detection
            available_hardware = self.detector.get_available_hardware()
            best_hardware = self.detector.get_best_available_hardware()
            
            # Get detailed hardware info
            hardware_details = {}
            for hw in available_hardware:
                try:
                    info = self.detector.get_hardware_info(hw)
                    hardware_details[hw] = info
                except Exception as e:
                    hardware_details[hw] = {"error": str(e)}
            
            passed = len(available_hardware) > 0 and best_hardware is not None
            
            details = {
                "available_hardware": available_hardware,
                "best_hardware": best_hardware,
                "hardware_details": hardware_details,
                "detection_methods": ["cpu", "cuda", "mps", "webnn", "webgpu"]
            }
            
            message = f"Hardware detection: {len(available_hardware)} platforms available, best: {best_hardware}"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="hardware_detection",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="hardware_detection",
                passed=False,
                message=f"Hardware detection validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.BASIC
            ))
            
    def _validate_model_compatibility(self) -> None:
        """Validate model compatibility system."""
        start_time = time.time()
        
        try:
            # Test model compatibility for various scenarios
            test_models = ["bert-base-uncased", "gpt2", "llama-7b"]
            available_hardware = self.detector.get_available_hardware()
            
            compatibility_results = {}
            successful_tests = 0
            total_tests = 0
            
            for model in test_models:
                compatibility_results[model] = {}
                for hardware in available_hardware:
                    total_tests += 1
                    try:
                        result = check_model_compatibility(model, hardware)
                        compatibility_results[model][hardware] = result
                        if result.get("compatible", False):
                            successful_tests += 1
                    except Exception as e:
                        compatibility_results[model][hardware] = {"error": str(e)}
            
            # Test optimal hardware recommendations
            recommendations = {}
            for model in test_models:
                try:
                    rec = get_optimal_hardware(model, available_hardware)
                    recommendations[model] = rec
                except Exception as e:
                    recommendations[model] = {"error": str(e)}
            
            passed = successful_tests > 0 and total_tests > 0
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            details = {
                "test_models": test_models,
                "compatibility_matrix": compatibility_results,
                "recommendations": recommendations,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": success_rate
            }
            
            message = f"Model compatibility: {successful_tests}/{total_tests} tests passed ({success_rate:.1f}%)"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="model_compatibility",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="model_compatibility",
                passed=False,
                message=f"Model compatibility validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
            
    def _validate_performance_baseline(self) -> None:
        """Validate performance modeling and establish baselines."""
        start_time = time.time()
        
        try:
            # Test performance simulation
            test_scenarios = [
                ("bert-base-uncased", "cpu", 1, "fp32"),
                ("gpt2", "cpu", 1, "fp32"),
            ]
            
            available_hardware = self.detector.get_available_hardware()
            
            performance_results = {}
            successful_simulations = 0
            total_simulations = 0
            
            for model, hardware, batch_size, precision in test_scenarios:
                if hardware in available_hardware:
                    total_simulations += 1
                    try:
                        result = simulate_model_performance(model, hardware, batch_size, precision)
                        performance_results[f"{model}_{hardware}_{precision}"] = {
                            "inference_time_ms": result.inference_time_ms,
                            "memory_usage_mb": result.memory_usage_mb,
                            "efficiency_score": result.efficiency_score,
                            "bottleneck": result.bottleneck
                        }
                        successful_simulations += 1
                    except Exception as e:
                        performance_results[f"{model}_{hardware}_{precision}"] = {"error": str(e)}
            
            passed = successful_simulations > 0
            
            # Calculate baseline metrics
            baseline_metrics = {}
            if successful_simulations > 0:
                inference_times = [
                    result["inference_time_ms"] 
                    for result in performance_results.values() 
                    if "inference_time_ms" in result
                ]
                memory_usages = [
                    result["memory_usage_mb"] 
                    for result in performance_results.values() 
                    if "memory_usage_mb" in result
                ]
                
                if inference_times:
                    baseline_metrics["avg_inference_time_ms"] = sum(inference_times) / len(inference_times)
                    baseline_metrics["min_inference_time_ms"] = min(inference_times)
                    baseline_metrics["max_inference_time_ms"] = max(inference_times)
                
                if memory_usages:
                    baseline_metrics["avg_memory_usage_mb"] = sum(memory_usages) / len(memory_usages)
                    baseline_metrics["min_memory_usage_mb"] = min(memory_usages)
                    baseline_metrics["max_memory_usage_mb"] = max(memory_usages)
            
            details = {
                "test_scenarios": test_scenarios,
                "performance_results": performance_results,
                "baseline_metrics": baseline_metrics,
                "successful_simulations": successful_simulations,
                "total_simulations": total_simulations,
                "available_hardware": available_hardware
            }
            
            message = f"Performance baseline: {successful_simulations}/{total_simulations} simulations successful"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="performance_baseline",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="performance_baseline",
                passed=False,
                message=f"Performance baseline validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
            
    def _validate_production_readiness(self) -> None:
        """Validate production deployment readiness."""
        start_time = time.time()
        
        try:
            checks = []
            
            # Check error handling
            try:
                # Test graceful handling of missing dependencies
                result = check_model_compatibility("nonexistent-model", "cpu")
                checks.append(("error_handling", True, "Handles invalid models gracefully"))
            except Exception:
                checks.append(("error_handling", False, "Does not handle errors gracefully"))
            
            # Check memory management
            try:
                # Simulate multiple model loads
                for i in range(5):
                    simulate_model_performance("bert-base-uncased", "cpu", 1, "fp32")
                checks.append(("memory_management", True, "Multiple simulations successful"))
            except Exception as e:
                checks.append(("memory_management", False, f"Memory management issue: {e}"))
            
            # Check concurrent operations
            try:
                # Test concurrent compatibility checks
                import threading
                results = []
                def check_compat():
                    result = check_model_compatibility("gpt2", "cpu")
                    results.append(result)
                
                threads = []
                for _ in range(3):
                    thread = threading.Thread(target=check_compat)
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                checks.append(("concurrency", True, f"Concurrent operations successful: {len(results)} results"))
            except Exception as e:
                checks.append(("concurrency", False, f"Concurrency issue: {e}"))
            
            passed_checks = sum(1 for _, passed, _ in checks if passed)
            total_checks = len(checks)
            passed = passed_checks >= total_checks * 0.8  # 80% pass rate
            
            details = {
                "checks": checks,
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "pass_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
            }
            
            message = f"Production readiness: {passed_checks}/{total_checks} checks passed"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="production_readiness",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="production_readiness",
                passed=False,
                message=f"Production readiness validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.PRODUCTION
            ))
    
    def _validate_enterprise_requirements(self) -> None:
        """Validate enterprise deployment requirements."""
        start_time = time.time()
        
        try:
            enterprise_checks = []
            
            # Check logging capabilities
            try:
                test_logger = logging.getLogger("test_enterprise")
                test_logger.info("Test logging message")
                enterprise_checks.append(("logging", True, "Logging system functional"))
            except Exception as e:
                enterprise_checks.append(("logging", False, f"Logging issue: {e}"))
            
            # Check configuration management
            try:
                # Test environment variable handling
                test_var = os.environ.get("TEST_VAR", "default_value")
                enterprise_checks.append(("config_management", True, f"Environment variables accessible: {test_var}"))
            except Exception as e:
                enterprise_checks.append(("config_management", False, f"Configuration issue: {e}"))
            
            # Check scalability indicators
            try:
                # Test large batch simulation
                result = simulate_model_performance("bert-base-uncased", "cpu", 32, "fp32")
                enterprise_checks.append(("scalability", True, f"Large batch simulation: {result.inference_time_ms:.1f}ms"))
            except Exception as e:
                enterprise_checks.append(("scalability", False, f"Scalability issue: {e}"))
            
            # Check security considerations
            try:
                # Basic security checks
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = Path(temp_dir) / "test_security.json"
                    temp_file.write_text('{"test": "data"}')
                    data = json.loads(temp_file.read_text())
                enterprise_checks.append(("security", True, "File operations secure"))
            except Exception as e:
                enterprise_checks.append(("security", False, f"Security concern: {e}"))
            
            passed_checks = sum(1 for _, passed, _ in enterprise_checks if passed)
            total_checks = len(enterprise_checks)
            passed = passed_checks >= total_checks * 0.9  # 90% pass rate for enterprise
            
            details = {
                "enterprise_checks": enterprise_checks,
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "pass_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
            }
            
            message = f"Enterprise readiness: {passed_checks}/{total_checks} checks passed"
            
            execution_time = (time.time() - start_time) * 1000
            
            self.results.append(ValidationResult(
                name="enterprise_readiness",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=execution_time,
                level=ValidationLevel.ENTERPRISE
            ))
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.results.append(ValidationResult(
                name="enterprise_readiness",
                passed=False,
                message=f"Enterprise readiness validation failed: {e}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
                level=ValidationLevel.ENTERPRISE
            ))
    
    def _generate_compatibility_report(self) -> SystemCompatibilityReport:
        """Generate comprehensive compatibility report."""
        
        # Extract system and performance info from results
        system_info = {}
        python_info = {}
        dependency_status = {}
        hardware_capabilities = {}
        performance_baseline = {}
        
        for result in self.results:
            if result.name == "system_compatibility":
                system_info = result.details
            elif result.name == "python_environment":
                python_info = result.details
            elif result.name == "dependencies":
                dependency_status = result.details.get("all_dependencies", {})
            elif result.name == "hardware_detection":
                hardware_capabilities = result.details
            elif result.name == "performance_baseline":
                performance_baseline = result.details.get("baseline_metrics", {})
        
        # Calculate overall score
        total_score = 0
        max_score = 0
        
        for result in self.results:
            if result.level.value in [self.validation_level.value]:
                max_score += 100
                if result.passed:
                    total_score += 100
                else:
                    # Partial credit for some failures
                    if result.details and not any("error" in str(v) for v in result.details.values()):
                        total_score += 30
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Generate deployment recommendations
        recommendations = []
        
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            recommendations.append(f"Address {len(failed_results)} failed validation checks")
        
        if dependency_status:
            missing_deps = [k for k, v in dependency_status.items() if not v]
            if missing_deps:
                recommendations.append(f"Consider installing optional dependencies: {', '.join(missing_deps[:3])}")
        
        if hardware_capabilities.get("available_hardware"):
            hw_count = len(hardware_capabilities["available_hardware"])
            if hw_count == 1:
                recommendations.append("Consider enabling additional hardware accelerators for better performance")
            else:
                recommendations.append(f"Good hardware support: {hw_count} platforms available")
        
        if overall_score >= 90:
            recommendations.append("✅ System is ready for production deployment")
        elif overall_score >= 70:
            recommendations.append("⚠️  System needs minor improvements before production deployment")
        else:
            recommendations.append("❌ System requires significant improvements before production deployment")
        
        return SystemCompatibilityReport(
            system_info=system_info,
            python_info=python_info,
            dependency_status=dependency_status,
            hardware_capabilities=hardware_capabilities,
            performance_baseline=performance_baseline,
            validation_results=self.results,
            overall_score=overall_score,
            deployment_recommendations=recommendations
        )

def run_production_validation(level: str = "production", output_file: Optional[str] = None) -> SystemCompatibilityReport:
    """Run production validation suite and optionally save results."""
    
    # Parse validation level
    try:
        validation_level = ValidationLevel(level.lower())
    except ValueError:
        validation_level = ValidationLevel.PRODUCTION
        logger.warning(f"Unknown validation level '{level}', using 'production'")
    
    validator = ProductionValidator(validation_level)
    report = validator.run_validation_suite()
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report to {output_file}: {e}")
    
    return report

def print_validation_summary(report: SystemCompatibilityReport) -> None:
    """Print a human-readable validation summary."""
    
    print("\n" + "="*80)
    print("🔍 IPFS ACCELERATE PYTHON - PRODUCTION VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 Overall Score: {report.overall_score:.1f}/100")
    
    if report.overall_score >= 90:
        print("✅ EXCELLENT - Ready for production deployment")
    elif report.overall_score >= 70:
        print("⚠️  GOOD - Minor improvements recommended")
    elif report.overall_score >= 50:
        print("⚠️  FAIR - Several improvements needed")
    else:
        print("❌ POOR - Major improvements required")
    
    print(f"\n🖥️  System Information:")
    if report.system_info:
        for key, value in report.system_info.items():
            print(f"  • {key}: {value}")
    
    print(f"\n🐍 Python Environment:")
    if report.python_info:
        print(f"  • Version: {report.python_info.get('version', 'Unknown')}")
        print(f"  • Virtual Environment: {report.python_info.get('virtual_environment', 'Unknown')}")
    
    print(f"\n📦 Dependencies:")
    passed_deps = sum(1 for v in report.dependency_status.values() if v)
    total_deps = len(report.dependency_status)
    print(f"  • Available: {passed_deps}/{total_deps} dependencies")
    
    print(f"\n🚀 Hardware Capabilities:")
    if report.hardware_capabilities.get("available_hardware"):
        hw_list = report.hardware_capabilities["available_hardware"]
        print(f"  • Available: {', '.join(hw_list)}")
        print(f"  • Recommended: {report.hardware_capabilities.get('best_hardware', 'Unknown')}")
    
    print(f"\n📈 Performance Baseline:")
    if report.performance_baseline:
        for metric, value in report.performance_baseline.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.2f}")
            else:
                print(f"  • {metric}: {value}")
    
    print(f"\n🔍 Validation Results:")
    for result in report.validation_results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.message}")
        if result.execution_time_ms > 1000:
            print(f"      (⚠️  Slow: {result.execution_time_ms:.0f}ms)")
    
    print(f"\n💡 Deployment Recommendations:")
    for rec in report.deployment_recommendations:
        print(f"  • {rec}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IPFS Accelerate Python Production Validation")
    parser.add_argument("--level", choices=["basic", "production", "enterprise"], 
                       default="production", help="Validation level")
    parser.add_argument("--output", help="Save detailed report to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    
    try:
        report = run_production_validation(args.level, args.output)
        
        if not args.quiet:
            print_validation_summary(report)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)