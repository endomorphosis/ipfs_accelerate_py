#!/usr/bin/env python3
"""
Comprehensive Dependency Installer for IPFS Accelerate Python

This module provides robust dependency installation with graceful failure handling
for optional components like Playwright, browser engines, and other AI/ML libraries.
"""

import os
import sys
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dependency_installer")

class DependencyInstaller:
    """Comprehensive dependency installer with graceful failure handling."""
    
    def __init__(self):
        """Initialize the dependency installer."""
        self.installation_log = []
        self.failed_installations = []
        self.successful_installations = []
        self.repo_root = Path(__file__).resolve().parents[1]
        self.external_dir = self.repo_root / "external"
        self.local_packages = [
            "ipfs_kit_py",
            "ipfs_model_manager_py",
            "ipfs_transformers_py",
        ]
        
    def check_dependency(self, module_name: str, import_name: Optional[str] = None) -> bool:
        """
        Check if a dependency is available.
        
        Args:
            module_name: Name of the module to check
            import_name: Alternative import name if different from module_name
            
        Returns:
            True if dependency is available, False otherwise
        """
        try:
            importlib.import_module(import_name or module_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_name: str, 
                       import_name: Optional[str] = None,
                       pip_args: Optional[List[str]] = None) -> bool:
        """
        Install a package using pip.
        
        Args:
            package_name: Name of package to install
            import_name: Import name to verify installation
            pip_args: Additional pip arguments
            
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Construct pip command
            cmd = [sys.executable, '-m', 'pip', 'install']
            if pip_args:
                cmd.extend(pip_args)
            cmd.append(package_name)
            
            logger.info(f"Installing {package_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Verify installation
                if self.check_dependency(import_name or package_name):
                    self.successful_installations.append(package_name)
                    self.installation_log.append(f"✅ {package_name} installed successfully")
                    logger.info(f"✅ {package_name} installed successfully")
                    return True
                else:
                    self.failed_installations.append(package_name)
                    self.installation_log.append(f"❌ {package_name} installed but import failed")
                    logger.warning(f"❌ {package_name} installed but import failed")
                    return False
            else:
                self.failed_installations.append(package_name)
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.installation_log.append(f"❌ {package_name} installation failed: {error_msg}")
                logger.error(f"❌ {package_name} installation failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            self.failed_installations.append(package_name)
            self.installation_log.append(f"❌ {package_name} installation timed out")
            logger.error(f"❌ {package_name} installation timed out")
            return False
        except Exception as e:
            self.failed_installations.append(package_name)
            self.installation_log.append(f"❌ {package_name} installation error: {str(e)}")
            logger.error(f"❌ {package_name} installation error: {str(e)}")
            return False
    
    def install_playwright_with_browsers(self) -> bool:
        """
        Install Playwright with browser engines.
        
        Returns:
            True if installation successful, False otherwise
        """
        logger.info("🎭 Installing Playwright with browser support...")
        
        # First install the Python package
        if not self.install_package("playwright"):
            return False
        
        # Install browser engines
        try:
            logger.info("📥 Installing Playwright browser engines...")
            result = subprocess.run([
                sys.executable, '-m', 'playwright', 'install'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.installation_log.append("✅ Playwright browser engines installed")
                logger.info("✅ Playwright browser engines installed")
                return True
            else:
                # Try installing just Chromium
                logger.info("⚠️ Full browser install failed, trying Chromium only...")
                result = subprocess.run([
                    sys.executable, '-m', 'playwright', 'install', 'chromium'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.installation_log.append("✅ Playwright Chromium browser installed")
                    logger.info("✅ Playwright Chromium browser installed")
                    return True
                else:
                    self.installation_log.append("❌ Playwright browser installation failed")
                    logger.error("❌ Playwright browser installation failed")
                    return False
                    
        except subprocess.TimeoutExpired:
            self.installation_log.append("❌ Playwright browser installation timed out")
            logger.error("❌ Playwright browser installation timed out")
            return False
        except Exception as e:
            self.installation_log.append(f"❌ Playwright browser installation error: {str(e)}")
            logger.error(f"❌ Playwright browser installation error: {str(e)}")
            return False
    
    def install_ai_ml_dependencies(self) -> Dict[str, bool]:
        """
        Install common AI/ML dependencies.
        
        Returns:
            Dictionary of package names and their installation status
        """
        logger.info("🤖 Installing AI/ML dependencies...")
        
        ai_packages = {
            # Core packages
            "transformers": "transformers",
            "torch": "torch", 
            "numpy": "numpy",
            "scipy": "scipy",
            "scikit-learn": "sklearn",
            "pynvml": "pynvml",
            
            # Audio processing
            "librosa": "librosa",
            "soundfile": "soundfile",
            "faster-whisper": "faster_whisper",
            
            # Image processing  
            "Pillow": "PIL",
            "opencv-python": "cv2",
            
            # Web and API
            "requests": "requests",
            "aiohttp": "aiohttp",
            "flask": "flask",
            
            # Data science
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            
            # Optional advanced packages
            "fastmcp": "fastmcp",
            "duckdb": "duckdb",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "aiohttp": "aiohttp",
            "websockets": "websockets",
            "sseclient-py": "sseclient",
        }
        
        results = {}
        for package_name, import_name in ai_packages.items():
            if self.check_dependency(import_name):
                results[package_name] = True
                self.installation_log.append(f"✅ {package_name} already available")
            else:
                results[package_name] = self.install_package(package_name, import_name)
        
        return results

    def install_local_packages(self) -> Dict[str, bool]:
        """Install bundled local packages from external/ when present."""
        results: Dict[str, bool] = {}
        if not self.external_dir.exists():
            logger.info("No external directory found; skipping local package installs")
            return results

        for package in self.local_packages:
            package_path = self.external_dir / package
            if not package_path.exists():
                results[package] = False
                continue

            try:
                cmd = [sys.executable, "-m", "pip", "install", "-e", str(package_path)]
                logger.info(f"Installing local package {package} from {package_path}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    self.successful_installations.append(package)
                    self.installation_log.append(f"✅ {package} installed from {package_path}")
                    results[package] = True
                else:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    self.failed_installations.append(package)
                    self.installation_log.append(f"❌ {package} local install failed: {error_msg}")
                    logger.error(f"❌ {package} local install failed: {error_msg}")
                    results[package] = False
            except Exception as e:
                self.failed_installations.append(package)
                self.installation_log.append(f"❌ {package} local install error: {str(e)}")
                logger.error(f"❌ {package} local install error: {str(e)}")
                results[package] = False

        return results
    
    def run_comprehensive_installation(self) -> Dict[str, Any]:
        """
        Run comprehensive dependency installation.
        
        Returns:
            Installation summary and results
        """
        logger.info("🚀 Starting comprehensive dependency installation...")
        
        # Install AI/ML dependencies
        ai_results = self.install_ai_ml_dependencies()

        # Install local external packages when available
        local_results = self.install_local_packages()
        
        # Install Playwright if needed
        playwright_available = self.check_dependency("playwright")
        if not playwright_available:
            playwright_success = self.install_playwright_with_browsers()
        else:
            playwright_success = True
            self.installation_log.append("✅ Playwright already available")
        
        # Generate summary
        total_packages = len(ai_results) + len(local_results) + (0 if playwright_available else 1)
        successful_count = len(self.successful_installations) + (1 if playwright_success else 0)
        failed_count = len(self.failed_installations)
        
        summary = {
            "total_packages": total_packages,
            "successful_installations": successful_count,
            "failed_installations": failed_count,
            "success_rate": (successful_count / total_packages) * 100 if total_packages > 0 else 0,
            "ai_packages": ai_results,
            "local_packages": local_results,
            "playwright_available": playwright_success,
            "installation_log": self.installation_log,
            "failed_packages": self.failed_installations
        }
        
        logger.info(f"📊 Installation Summary:")
        logger.info(f"   • Total packages: {total_packages}")
        logger.info(f"   • Successful: {successful_count}")
        logger.info(f"   • Failed: {failed_count}")
        logger.info(f"   • Success rate: {summary['success_rate']:.1f}%")
        
        return summary
    
    def create_fallback_config(self, installation_summary: Dict[str, Any]) -> Dict[str, bool]:
        """
        Create fallback configuration based on available dependencies.
        
        Args:
            installation_summary: Results from run_comprehensive_installation()
            
        Returns:
            Configuration dictionary for enabling/disabling features
        """
        config = {
            "HAVE_PLAYWRIGHT": installation_summary["playwright_available"],
            "HAVE_TRANSFORMERS": self.check_dependency("transformers"),
            "HAVE_TORCH": self.check_dependency("torch"),
            "HAVE_REQUESTS": self.check_dependency("requests"),
            "HAVE_FLASK": self.check_dependency("flask"),
            "HAVE_FASTMCP": self.check_dependency("fastmcp"),
            "HAVE_DUCKDB": self.check_dependency("duckdb"),
            "HAVE_LIBROSA": self.check_dependency("librosa"),
            "HAVE_PIL": self.check_dependency("PIL"),
            "HAVE_CV2": self.check_dependency("cv2"),
            "HAVE_AIOHTTP": self.check_dependency("aiohttp"),
        }
        
        # Save configuration to file
        config_path = Path("dependency_config.json")
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Dependency configuration saved to {config_path}")
        
        return config


def install_dependencies_with_fallbacks() -> Dict[str, Any]:
    """
    Main function to install dependencies with fallback support.
    
    Returns:
        Installation summary and configuration
    """
    installer = DependencyInstaller()
    
    try:
        # Run comprehensive installation
        summary = installer.run_comprehensive_installation()
        
        # Create fallback configuration
        config = installer.create_fallback_config(summary)
        
        # Print final status
        print("\n" + "="*60)
        print("🎯 DEPENDENCY INSTALLATION COMPLETE")
        print("="*60)
        
        for log_entry in installer.installation_log[-10:]:  # Show last 10 entries
            print(f"  {log_entry}")
        
        if installer.failed_installations:
            print(f"\n⚠️ Some packages failed to install: {', '.join(installer.failed_installations)}")
            print("🔄 The system will continue with available dependencies")
        
        print(f"\n✅ System ready with {summary['success_rate']:.1f}% dependency coverage")
        
        return {
            "summary": summary,
            "config": config,
            "installer": installer
        }
        
    except Exception as e:
        logger.error(f"💥 Critical error during dependency installation: {e}")
        return {
            "summary": {"success_rate": 0, "error": str(e)},
            "config": {},
            "installer": installer
        }


if __name__ == "__main__":
    result = install_dependencies_with_fallbacks()
    exit_code = 0 if result["summary"]["success_rate"] > 50 else 1
    sys.exit(exit_code)