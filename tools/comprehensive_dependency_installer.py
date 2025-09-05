#!/usr/bin/env python3
"""
Comprehensive Dependency Installer for IPFS Accelerate Python

This module provides robust dependency installation with graceful failure handling
for ALL optional components including FastMCP, Playwright, browser engines, and other AI/ML libraries.
"""

import os
import sys
import subprocess
import logging
import platform
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import importlib.util
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_dependency_installer")

class ComprehensiveDependencyInstaller:
    """Comprehensive dependency installer with graceful failure handling."""
    
    def __init__(self, log_file: str = "dependency_installation.log"):
        """Initialize the comprehensive dependency installer."""
        self.installation_log = []
        self.failed_installations = []
        self.successful_installations = []
        self.log_file = log_file
        self.system_info = self._get_system_info()
        
        # Comprehensive dependency definitions
        self.dependencies = {
            # Core MCP and Web Framework Dependencies
            "fastmcp": {
                "pip_name": "fastmcp",
                "import_name": "fastmcp",
                "description": "FastMCP for Model Control Protocol server",
                "category": "mcp",
                "critical": True,
                "fallback_packages": ["mcp", "uvicorn", "fastapi"]
            },
            "flask": {
                "pip_name": "flask",
                "import_name": "flask",
                "description": "Flask web framework for Kitchen Sink interface",
                "category": "web",
                "critical": True
            },
            "flask-cors": {
                "pip_name": "flask-cors",
                "import_name": "flask_cors",
                "description": "Flask CORS support",
                "category": "web",
                "critical": True
            },
            
            # Browser Automation and Testing
            "playwright": {
                "pip_name": "playwright",
                "import_name": "playwright",
                "description": "Playwright for browser automation and screenshots",
                "category": "testing",
                "critical": False,
                "post_install": ["python", "-m", "playwright", "install", "chromium"]
            },
            "selenium": {
                "pip_name": "selenium",
                "import_name": "selenium",
                "description": "Selenium as fallback for browser automation",
                "category": "testing",
                "critical": False
            },
            
            # AI/ML Core Libraries
            "transformers": {
                "pip_name": "transformers",
                "import_name": "transformers",
                "description": "HuggingFace Transformers library",
                "category": "ai",
                "critical": False
            },
            "torch": {
                "pip_name": "torch",
                "import_name": "torch",
                "description": "PyTorch for neural networks",
                "category": "ai",
                "critical": False,
                "pip_args": ["--index-url", "https://download.pytorch.org/whl/cpu"]
            },
            "tensorflow": {
                "pip_name": "tensorflow",
                "import_name": "tensorflow",
                "description": "TensorFlow for neural networks",
                "category": "ai",
                "critical": False
            },
            "numpy": {
                "pip_name": "numpy",
                "import_name": "numpy",
                "description": "NumPy for numerical computing",
                "category": "core",
                "critical": True
            },
            "scipy": {
                "pip_name": "scipy", 
                "import_name": "scipy",
                "description": "SciPy for scientific computing",
                "category": "core",
                "critical": False
            },
            
            # Database and Storage
            "duckdb": {
                "pip_name": "duckdb",
                "import_name": "duckdb",
                "description": "DuckDB for model metadata storage",
                "category": "database",
                "critical": False
            },
            "sqlite3": {
                "pip_name": None,  # Built-in
                "import_name": "sqlite3",
                "description": "SQLite3 for fallback storage",
                "category": "database",
                "critical": True,
                "builtin": True
            },
            
            # Vector Search and Embeddings
            "sentence-transformers": {
                "pip_name": "sentence-transformers",
                "import_name": "sentence_transformers",
                "description": "Sentence Transformers for embeddings",
                "category": "ai",
                "critical": False
            },
            "faiss-cpu": {
                "pip_name": "faiss-cpu",
                "import_name": "faiss",
                "description": "FAISS for vector search",
                "category": "ai",
                "critical": False
            },
            
            # Web and Network
            "requests": {
                "pip_name": "requests",
                "import_name": "requests",
                "description": "HTTP requests library",
                "category": "web",
                "critical": True
            },
            "aiohttp": {
                "pip_name": "aiohttp",
                "import_name": "aiohttp",
                "description": "Async HTTP client/server",
                "category": "web",
                "critical": False
            },
            "httpx": {
                "pip_name": "httpx",
                "import_name": "httpx",
                "description": "Modern HTTP client",
                "category": "web", 
                "critical": False
            },
            
            # IPFS and Content Addressing
            "ipfshttpclient": {
                "pip_name": "ipfshttpclient",
                "import_name": "ipfshttpclient",
                "description": "IPFS HTTP client",
                "category": "ipfs",
                "critical": False
            },
            "multiformats": {
                "pip_name": "multiformats",
                "import_name": "multiformats",
                "description": "Multiformats for content addressing",
                "category": "ipfs",
                "critical": False
            },
            
            # Image and Audio Processing
            "pillow": {
                "pip_name": "pillow",
                "import_name": "PIL",
                "description": "Python Imaging Library",
                "category": "media",
                "critical": False
            },
            "opencv-python": {
                "pip_name": "opencv-python",
                "import_name": "cv2",
                "description": "OpenCV for computer vision",
                "category": "media",
                "critical": False
            },
            "librosa": {
                "pip_name": "librosa",
                "import_name": "librosa",
                "description": "Audio processing library",
                "category": "media",
                "critical": False
            },
            
            # Data Processing
            "pandas": {
                "pip_name": "pandas",
                "import_name": "pandas",
                "description": "Data manipulation library",
                "category": "data",
                "critical": False
            },
            "pyarrow": {
                "pip_name": "pyarrow",
                "import_name": "pyarrow",
                "description": "Apache Arrow for data processing",
                "category": "data",
                "critical": False
            },
            
            # Async and Concurrency
            "asyncio": {
                "pip_name": None,  # Built-in
                "import_name": "asyncio",
                "description": "Asyncio for concurrent programming",
                "category": "core",
                "critical": True,
                "builtin": True
            },
            "uvloop": {
                "pip_name": "uvloop",
                "import_name": "uvloop",
                "description": "Fast asyncio event loop",
                "category": "performance",
                "critical": False,
                "platform_specific": ["linux", "darwin"]  # Not available on Windows
            },
            
            # Development and Testing
            "pytest": {
                "pip_name": "pytest",
                "import_name": "pytest",
                "description": "Testing framework",
                "category": "testing",
                "critical": False
            },
            "black": {
                "pip_name": "black",
                "import_name": "black",
                "description": "Code formatter",
                "category": "dev",
                "critical": False
            },
            "flake8": {
                "pip_name": "flake8",
                "import_name": "flake8",
                "description": "Code linter",
                "category": "dev",
                "critical": False
            }
        }
        
        logger.info(f"Initialized comprehensive installer for {len(self.dependencies)} dependencies")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for platform-specific installations."""
        return {
            "platform": platform.platform(),
            "system": platform.system().lower(),
            "architecture": platform.architecture()[0],
            "python_version": platform.python_version(),
            "python_executable": sys.executable
        }
    
    def check_dependency(self, module_name: str, import_name: Optional[str] = None) -> bool:
        """Check if a dependency is available."""
        try:
            importlib.import_module(import_name or module_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_name: str, 
                       pip_name: Optional[str] = None,
                       pip_args: Optional[List[str]] = None,
                       post_install: Optional[List[str]] = None) -> bool:
        """Install a package with comprehensive error handling."""
        if pip_name is None:
            logger.info(f"Skipping {package_name} (built-in module)")
            return True
            
        try:
            # Prepare installation command
            cmd = [sys.executable, "-m", "pip", "install"]
            
            # Add pip arguments if specified
            if pip_args:
                cmd.extend(pip_args)
            
            cmd.append(pip_name)
            
            logger.info(f"Installing {package_name} ({pip_name})...")
            
            # Run installation with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package_name}")
                
                # Run post-installation commands if specified
                if post_install:
                    try:
                        logger.info(f"Running post-install command for {package_name}")
                        post_result = subprocess.run(
                            post_install,
                            capture_output=True,
                            text=True,
                            timeout=600  # 10 minute timeout for browser installation
                        )
                        if post_result.returncode == 0:
                            logger.info(f"✅ Post-install completed for {package_name}")
                        else:
                            logger.warning(f"⚠️ Post-install failed for {package_name}: {post_result.stderr}")
                    except Exception as e:
                        logger.warning(f"⚠️ Post-install error for {package_name}: {e}")
                
                self.successful_installations.append(package_name)
                self.installation_log.append({
                    "package": package_name,
                    "status": "success",
                    "pip_name": pip_name,
                    "timestamp": time.time()
                })
                return True
            else:
                logger.error(f"❌ Failed to install {package_name}: {result.stderr}")
                self.failed_installations.append({
                    "package": package_name,
                    "error": result.stderr,
                    "returncode": result.returncode
                })
                self.installation_log.append({
                    "package": package_name,
                    "status": "failed",
                    "error": result.stderr,
                    "timestamp": time.time()
                })
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Installation timeout for {package_name}")
            self.failed_installations.append({
                "package": package_name,
                "error": "Installation timeout",
                "returncode": -1
            })
            return False
        except Exception as e:
            logger.error(f"❌ Installation error for {package_name}: {e}")
            self.failed_installations.append({
                "package": package_name,
                "error": str(e),
                "returncode": -1
            })
            return False
    
    def install_fallback_packages(self, main_package: str, fallback_packages: List[str]) -> bool:
        """Install fallback packages if main package fails."""
        logger.info(f"Attempting fallback installations for {main_package}")
        
        for fallback in fallback_packages:
            if self.install_package(f"{main_package}_fallback_{fallback}", fallback):
                logger.info(f"✅ Successfully installed fallback {fallback} for {main_package}")
                return True
        
        logger.warning(f"⚠️ All fallback installations failed for {main_package}")
        return False
    
    def is_platform_compatible(self, package_name: str) -> bool:
        """Check if package is compatible with current platform."""
        dep = self.dependencies.get(package_name, {})
        platform_specific = dep.get("platform_specific")
        
        if platform_specific:
            return self.system_info["system"] in platform_specific
        
        return True
    
    def install_all_dependencies(self, 
                                categories: Optional[List[str]] = None,
                                critical_only: bool = False) -> Dict[str, Any]:
        """Install all dependencies with comprehensive error handling."""
        
        logger.info("🚀 Starting comprehensive dependency installation...")
        logger.info(f"System: {self.system_info['platform']}")
        logger.info(f"Python: {self.system_info['python_version']}")
        
        # Filter dependencies based on criteria
        deps_to_install = {}
        for name, dep in self.dependencies.items():
            # Check category filter
            if categories and dep.get("category") not in categories:
                continue
            
            # Check critical filter
            if critical_only and not dep.get("critical", False):
                continue
            
            # Check platform compatibility
            if not self.is_platform_compatible(name):
                logger.info(f"⏭️ Skipping {name} (not compatible with {self.system_info['system']})")
                continue
            
            deps_to_install[name] = dep
        
        logger.info(f"Installing {len(deps_to_install)} dependencies...")
        
        # Installation statistics
        total_deps = len(deps_to_install)
        installed_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Install dependencies by category priority
        category_order = ["core", "mcp", "web", "database", "ai", "testing", "dev", "performance", "media", "data", "ipfs"]
        
        for category in category_order:
            category_deps = {name: dep for name, dep in deps_to_install.items() 
                           if dep.get("category") == category}
            
            if not category_deps:
                continue
                
            logger.info(f"\n📦 Installing {category} dependencies ({len(category_deps)} packages)")
            
            for name, dep in category_deps.items():
                # Check if already available
                import_name = dep.get("import_name", name)
                if self.check_dependency(name, import_name):
                    logger.info(f"✅ {name} already available")
                    skipped_count += 1
                    continue
                
                # Skip built-in modules that aren't available (shouldn't happen)
                if dep.get("builtin") and not self.check_dependency(name, import_name):
                    logger.error(f"❌ Built-in module {name} not available!")
                    failed_count += 1
                    continue
                
                # Install the package
                success = self.install_package(
                    name,
                    dep.get("pip_name"),
                    dep.get("pip_args"),
                    dep.get("post_install")
                )
                
                if success:
                    installed_count += 1
                elif dep.get("fallback_packages"):
                    # Try fallback packages
                    if self.install_fallback_packages(name, dep["fallback_packages"]):
                        installed_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
        
        # Final verification
        self._verify_installations()
        
        # Save installation log
        self._save_installation_log()
        
        # Generate report
        report = {
            "total_dependencies": total_deps,
            "installed": installed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "success_rate": (installed_count + skipped_count) / total_deps * 100,
            "critical_dependencies_status": self._check_critical_dependencies(),
            "system_info": self.system_info,
            "failed_installations": self.failed_installations,
            "successful_installations": self.successful_installations
        }
        
        logger.info(f"\n🎯 Installation Summary:")
        logger.info(f"   Total: {total_deps}")
        logger.info(f"   Installed: {installed_count}")
        logger.info(f"   Already Available: {skipped_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info(f"   Success Rate: {report['success_rate']:.1f}%")
        
        return report
    
    def _verify_installations(self):
        """Verify that installed packages can be imported."""
        logger.info("\n🔍 Verifying installations...")
        
        verification_results = {}
        for name, dep in self.dependencies.items():
            import_name = dep.get("import_name", name)
            if self.check_dependency(name, import_name):
                verification_results[name] = True
                logger.debug(f"✅ {name} verified")
            else:
                verification_results[name] = False
                logger.warning(f"⚠️ {name} not available after installation")
        
        return verification_results
    
    def _check_critical_dependencies(self) -> Dict[str, bool]:
        """Check status of critical dependencies."""
        critical_status = {}
        for name, dep in self.dependencies.items():
            if dep.get("critical", False):
                import_name = dep.get("import_name", name)
                critical_status[name] = self.check_dependency(name, import_name)
        
        return critical_status
    
    def _save_installation_log(self):
        """Save installation log to file."""
        try:
            log_data = {
                "timestamp": time.time(),
                "system_info": self.system_info,
                "installation_log": self.installation_log,
                "successful_installations": self.successful_installations,
                "failed_installations": self.failed_installations
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"📝 Installation log saved to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save installation log: {e}")
    
    def create_mock_modules(self):
        """Create mock modules for failed dependencies to prevent import errors."""
        logger.info("🔧 Creating mock modules for failed dependencies...")
        
        mock_modules = {}
        for failed in self.failed_installations:
            package_name = failed["package"]
            dep = self.dependencies.get(package_name, {})
            import_name = dep.get("import_name", package_name)
            
            # Create a simple mock module
            mock_module = type(sys)("mock_" + import_name)
            mock_module.__file__ = f"<mock {import_name}>"
            mock_module.__path__ = []
            
            # Add basic mock functions/classes
            if "transformers" in import_name:
                mock_module.AutoTokenizer = type("MockAutoTokenizer", (), {})
                mock_module.AutoModel = type("MockAutoModel", (), {})
            elif "torch" in import_name:
                mock_module.tensor = lambda x: x
                mock_module.nn = type("MockNN", (), {})
            elif "fastmcp" in import_name:
                mock_module.FastMCP = type("MockFastMCP", (), {"tool": lambda self: lambda f: f})
            
            sys.modules[import_name] = mock_module
            mock_modules[import_name] = mock_module
            logger.info(f"🔧 Created mock module for {import_name}")
        
        return mock_modules
    
    def install_browser_dependencies(self) -> bool:
        """Install browser dependencies for screenshot functionality."""
        logger.info("🌐 Installing browser dependencies...")
        
        # Try Playwright first
        if self.install_package("playwright", "playwright", post_install=["python", "-m", "playwright", "install", "chromium"]):
            return True
        
        # Try Selenium as fallback
        if self.install_package("selenium", "selenium"):
            # Try to install ChromeDriver
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                # Install ChromeDriver
                subprocess.run([sys.executable, "-m", "pip", "install", "webdriver-manager"], 
                             check=False, capture_output=True)
                
                logger.info("✅ Selenium and ChromeDriver setup completed")
                return True
            except Exception as e:
                logger.warning(f"⚠️ ChromeDriver setup failed: {e}")
        
        logger.warning("⚠️ Browser automation dependencies not available")
        return False

def install_all_dependencies(categories: Optional[List[str]] = None, 
                           critical_only: bool = False) -> Dict[str, Any]:
    """
    Install all dependencies for the IPFS Accelerate Python project.
    
    Args:
        categories: List of categories to install (e.g., ['core', 'ai', 'web'])
        critical_only: If True, only install critical dependencies
        
    Returns:
        Installation report dictionary
    """
    installer = ComprehensiveDependencyInstaller()
    return installer.install_all_dependencies(categories, critical_only)

def install_browser_dependencies() -> bool:
    """Install browser dependencies for screenshot functionality."""
    installer = ComprehensiveDependencyInstaller()
    return installer.install_browser_dependencies()

def create_mock_modules() -> Dict[str, Any]:
    """Create mock modules for failed dependencies."""
    installer = ComprehensiveDependencyInstaller()
    return installer.create_mock_modules()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Dependency Installer")
    parser.add_argument("--categories", nargs="+", 
                       choices=["core", "mcp", "web", "ai", "testing", "dev", "database", "ipfs", "media", "data", "performance"],
                       help="Categories to install")
    parser.add_argument("--critical-only", action="store_true",
                       help="Install only critical dependencies")
    parser.add_argument("--browser-deps", action="store_true",
                       help="Install browser automation dependencies")
    parser.add_argument("--create-mocks", action="store_true",
                       help="Create mock modules for failed dependencies")
    
    args = parser.parse_args()
    
    if args.browser_deps:
        success = install_browser_dependencies()
        if success:
            print("✅ Browser dependencies installed successfully")
        else:
            print("❌ Browser dependencies installation failed")
    
    if args.create_mocks:
        mocks = create_mock_modules()
        print(f"🔧 Created {len(mocks)} mock modules")
    
    # Install main dependencies
    report = install_all_dependencies(args.categories, args.critical_only)
    
    print(f"\n🎯 Final Report:")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Critical Dependencies: {sum(report['critical_dependencies_status'].values())}/{len(report['critical_dependencies_status'])} available")
    
    if report["failed_installations"]:
        print(f"\n❌ Failed installations:")
        for failed in report["failed_installations"]:
            print(f"  - {failed['package']}: {failed['error'][:100]}...")