#!/usr/bin/env python3
"""
Safe import utilities for graceful dependency handling.

This module provides utilities for safely importing optional dependencies
with appropriate fallbacks and user-friendly error messages.
"""

import logging
import importlib.util
from typing import Optional, Any, Dict, List, Callable
import warnings
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class ImportError(Exception):
    """Custom import error with installation suggestions."""
    
    def __init__(self, message: str, package_name: str, install_command: str = None):
        self.package_name = package_name
        self.install_command = install_command or f"pip install {package_name}"
        super().__init__(f"{message}\n\nTo install: {self.install_command}")

class SafeImporter:
    """
    Safe import manager for optional dependencies.
    
    Provides graceful fallbacks and installation guidance for missing packages.
    """
    
    def __init__(self):
        self._import_cache: Dict[str, Any] = {}
        self._failed_imports: Dict[str, str] = {}
        
    def safe_import(self, 
                   module_name: str, 
                   fallback: Any = None,
                   optional: bool = True,
                   install_command: str = None,
                   min_version: str = None) -> Any:
        """
        Safely import a module with fallback handling.
        
        Args:
            module_name: Name of the module to import
            fallback: Value to return if import fails (None by default)
            optional: If True, log as info; if False, raise ImportError
            install_command: Custom installation command suggestion
            min_version: Minimum required version (not enforced, just logged)
            
        Returns:
            Imported module or fallback value
        """
        
        # Check cache first
        if module_name in self._import_cache:
            return self._import_cache[module_name]
            
        # Check if already failed
        if module_name in self._failed_imports:
            if not optional:
                raise ImportError(
                    f"Required module '{module_name}' not available",
                    module_name,
                    install_command
                )
            return fallback
            
        try:
            module = importlib.import_module(module_name)
            self._import_cache[module_name] = module
            
            # Log successful import with version if available
            version = getattr(module, '__version__', 'unknown')
            logger.debug(f"Successfully imported {module_name} version {version}")
            
            if min_version and hasattr(module, '__version__'):
                logger.debug(f"{module_name} version {version} (minimum required: {min_version})")
                
            return module
            
        except (ImportError, ModuleNotFoundError) as e:
            error_msg = str(e)
            self._failed_imports[module_name] = error_msg
            
            if optional:
                logger.info(f"Optional dependency '{module_name}' not available: {error_msg}")
                self._import_cache[module_name] = fallback
                return fallback
            else:
                raise ImportError(
                    f"Required module '{module_name}' not available: {error_msg}",
                    module_name,
                    install_command
                )
    
    def check_module_available(self, module_name: str) -> bool:
        """Check if a module is available without importing it."""
        if module_name in self._import_cache:
            return self._import_cache[module_name] is not None
        
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    
    def get_import_summary(self) -> Dict[str, Any]:
        """Get summary of import status for all checked modules."""
        return {
            'successful_imports': {k: v for k, v in self._import_cache.items() if v is not None},
            'failed_imports': self._failed_imports,
            'available_modules': [k for k, v in self._import_cache.items() if v is not None]
        }

# Global importer instance
_importer = SafeImporter()

def safe_import(module_name: str, 
                fallback: Any = None,
                optional: bool = True,
                install_command: str = None,
                min_version: str = None) -> Any:
    """
    Convenience function for safe importing.
    
    See SafeImporter.safe_import for full documentation.
    """
    return _importer.safe_import(
        module_name=module_name,
        fallback=fallback,
        optional=optional,
        install_command=install_command,
        min_version=min_version
    )

def check_available(module_name: str) -> bool:
    """Check if a module is available."""
    return _importer.check_module_available(module_name)

def get_import_summary() -> Dict[str, Any]:
    """Get import summary."""
    return _importer.get_import_summary()

# Common dependency patterns
def import_torch(optional: bool = True):
    """Import PyTorch with graceful fallback."""
    return safe_import(
        'torch',
        fallback=None,
        optional=optional,
        install_command='pip install torch',
        min_version='2.1'
    )

def import_transformers(optional: bool = True):
    """Import Transformers with graceful fallback."""
    return safe_import(
        'transformers',
        fallback=None,
        optional=optional,
        install_command='pip install transformers>=4.46',
        min_version='4.46'
    )

def import_fastapi(optional: bool = True):
    """Import FastAPI with graceful fallback."""
    return safe_import(
        'fastapi',
        fallback=None,
        optional=optional,
        install_command='pip install fastapi>=0.110.0',
        min_version='0.110.0'
    )

def import_uvicorn(optional: bool = True):
    """Import Uvicorn with graceful fallback."""
    return safe_import(
        'uvicorn',
        fallback=None,
        optional=optional,
        install_command='pip install uvicorn>=0.27.0',
        min_version='0.27.0'
    )

def import_numpy(optional: bool = False):
    """Import NumPy (usually required)."""
    return safe_import(
        'numpy',
        fallback=None,
        optional=optional,
        install_command='pip install numpy>=1.24.0',
        min_version='1.24.0'
    )

def import_requests(optional: bool = False):
    """Import requests (usually required)."""
    return safe_import(
        'requests',
        fallback=None,
        optional=optional,
        install_command='pip install requests>=2.28.0',
        min_version='2.28.0'
    )

def import_aiohttp(optional: bool = True):
    """Import aiohttp with graceful fallback."""
    return safe_import(
        'aiohttp',
        fallback=None,
        optional=optional,
        install_command='pip install aiohttp>=3.8.1',
        min_version='3.8.1'
    )

class MockModule:
    """Mock module that provides basic interface for testing."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        logger.info(f"Using mock module for {module_name}")
    
    def __getattr__(self, name):
        logger.debug(f"Mock {self.module_name}.{name} called")
        return lambda *args, **kwargs: None
    
    def __call__(self, *args, **kwargs):
        logger.debug(f"Mock {self.module_name}() called")
        return None

def create_mock_torch():
    """Create mock PyTorch module for testing without GPU."""
    
    class MockTorch:
        """Mock PyTorch that simulates basic interface."""
        
        @property
        def cuda(self):
            return MockCuda()
        
        @property
        def backends(self):
            return MockBackends()
        
        def device(self, device_str):
            return MockDevice(device_str)
        
        def tensor(self, data):
            return MockTensor(data)
        
        def no_grad(self):
            return MockNoGrad()
        
        def __version__(self):
            return "2.1.0+mock"
    
    class MockCuda:
        def is_available(self):
            return False
        
        def device_count(self):
            return 0
    
    class MockBackends:
        @property
        def mps(self):
            return MockMPS()
        
        @property
        def cudnn(self):
            return MockCudnn()
    
    class MockMPS:
        def is_available(self):
            return False
    
    class MockCudnn:
        @property 
        def enabled(self):
            return False
    
    class MockDevice:
        def __init__(self, device_str):
            self.device_str = device_str
    
    class MockTensor:
        def __init__(self, data):
            self.data = data
            
        def to(self, device):
            return self
        
        def cuda(self):
            return self
    
    class MockNoGrad:
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
    
    return MockTorch()

# Dependency groups for different use cases
DEPENDENCY_GROUPS = {
    'minimal': [
        ('numpy', 'import_numpy'),
        ('requests', 'import_requests'), 
        ('aiohttp', 'import_aiohttp'),
    ],
    'ml_core': [
        ('torch', 'import_torch'),
        ('transformers', 'import_transformers'),
    ],
    'web_server': [
        ('fastapi', 'import_fastapi'),
        ('uvicorn', 'import_uvicorn'),
    ],
    'testing': [
        ('pytest', lambda: safe_import('pytest', install_command='pip install pytest>=8.0.0')),
    ]
}

def check_dependency_group(group_name: str) -> Dict[str, bool]:
    """Check availability of dependencies in a group."""
    if group_name not in DEPENDENCY_GROUPS:
        raise ValueError(f"Unknown dependency group: {group_name}")
    
    results = {}
    dependencies = DEPENDENCY_GROUPS[group_name]
    
    for dep_name, import_func in dependencies:
        if isinstance(import_func, str):
            # Get function from globals
            import_func = globals().get(import_func)
        
        try:
            result = import_func() if callable(import_func) else safe_import(dep_name)
            results[dep_name] = result is not None
        except Exception as e:
            logger.debug(f"Error checking {dep_name}: {e}")
            results[dep_name] = False
    
    return results

def get_production_dependencies() -> Dict[str, List[str]]:
    """Get production dependency requirements for validation."""
    return {
        'core': ['numpy', 'duckdb', 'aiohttp', 'tqdm', 'psutil', 'requests'],
        'ml': ['torch', 'transformers', 'onnx', 'onnxruntime'],
        'web': ['fastapi', 'uvicorn', 'websockets'],
        'optional': ['matplotlib', 'plotly', 'pandas', 'scikit-learn']
    }

def validate_production_dependencies() -> Dict[str, Any]:
    """Validate dependencies for production readiness."""
    deps = get_production_dependencies()
    results = {
        'core_available': 0,
        'core_total': len(deps['core']),
        'ml_available': 0, 
        'ml_total': len(deps['ml']),
        'web_available': 0,
        'web_total': len(deps['web']),
        'optional_available': 0,
        'optional_total': len(deps['optional']),
        'details': {}
    }
    
    for category, modules in deps.items():
        available_count = 0
        category_details = {}
        
        for module in modules:
            is_available = check_available(module)
            category_details[module] = is_available
            if is_available:
                available_count += 1
        
        results[f'{category}_available'] = available_count
        results['details'][category] = category_details
    
    # Calculate overall scores
    total_available = (results['core_available'] + results['ml_available'] + 
                      results['web_available'] + results['optional_available'])
    total_possible = (results['core_total'] + results['ml_total'] + 
                     results['web_total'] + results['optional_total'])
    
    results['total_available'] = total_available
    results['total_possible'] = total_possible
    results['dependency_score'] = (total_available / total_possible * 100) if total_possible > 0 else 0
    
    return results

def print_dependency_status():
    """Print status of all dependencies."""
    print("\n🔍 Dependency Status Report")
    print("=" * 50)
    
    for group_name in DEPENDENCY_GROUPS.keys():
        print(f"\n📦 {group_name.upper()} Dependencies:")
        status = check_dependency_group(group_name)
        
        for dep_name, available in status.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {dep_name}")
    
    # Print production validation
    prod_deps = validate_production_dependencies()
    print(f"\n📊 Production Dependency Validation:")
    print(f"  🔧 Core: {prod_deps['core_available']}/{prod_deps['core_total']}")
    print(f"  🤖 ML: {prod_deps['ml_available']}/{prod_deps['ml_total']}")
    print(f"  🌐 Web: {prod_deps['web_available']}/{prod_deps['web_total']}")
    print(f"  📊 Optional: {prod_deps['optional_available']}/{prod_deps['optional_total']}")
    print(f"  📈 Overall Score: {prod_deps['dependency_score']:.1f}%")
    
    # Print summary
    summary = get_import_summary()
    print(f"\n📊 Import Summary:")
    print(f"  ✅ Available: {len(summary['available_modules'])}")
    print(f"  ❌ Failed: {len(summary['failed_imports'])}")
    
    if summary['failed_imports']:
        print(f"\n⚠️  Failed imports:")
        for module, error in summary['failed_imports'].items():
            print(f"  - {module}: {error}")

if __name__ == "__main__":
    # Demo/test the safe import system
    print_dependency_status()