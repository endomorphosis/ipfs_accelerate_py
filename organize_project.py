#!/usr/bin/env python3
"""
Project Organization Script for IPFS Accelerate

This script organizes the project structure by:
1. Moving files to appropriate directories
2. Cleaning up logs and temporary files
3. Organizing tests and documentation
4. Creating proper directory structure
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectOrganizer:
    """Organizes the IPFS Accelerate project structure."""
    
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.moved_files = []
        self.cleaned_files = []
        
        # Define target directory structure
        self.directories = {
            'tests': 'tests',
            'logs': 'logs',
            'scripts': 'scripts', 
            'configs': 'configs',
            'results': 'test-results',
            'tools': 'tools',
            'docs': 'docs',
            'examples': 'examples',
            'docker': 'docker',
            'archived': 'archived'
        }
        
        # File categorization patterns
        self.file_patterns = {
            'test_files': [
                'test_*.py', 'test_*.sh', '*_test.py', '*_test.sh', 'test.*'
            ],
            'log_files': [
                '*.log', '*_output.txt', 'debug_output.txt', '*_results.json',
                'diagnostic*.txt', 'verification_results.json'
            ],
            'script_files': [
                'run_*.sh', 'start_*.sh', 'stop_*.sh', 'restart_*.sh',
                'fix_*.sh', 'verify_*.sh', 'install_*.sh'
            ],
            'config_files': [
                '*.json', 'pixi.lock', '*_settings.json', '*.yml', '*.yaml'
            ],
            'docker_files': [
                'Dockerfile*', 'docker-compose*.yml', '*.dockerfile'
            ],
            'documentation': [
                '*.md', '*.rst', '*.txt'
            ],
            'python_scripts': [
                '*.py'
            ]
        }
    
    def create_directories(self):
        """Create the target directory structure."""
        logger.info("Creating directory structure...")
        
        for name, path in self.directories.items():
            full_path = self.root / path
            full_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"✅ Created directory: {path}")
        
        # Create specialized subdirectories
        specialized_dirs = [
            'tests/unit',
            'tests/integration', 
            'tests/e2e',
            'tests/mojo',
            'logs/mcp',
            'logs/tests',
            'logs/archived',
            'configs/mcp',
            'configs/docker',
            'scripts/mcp',
            'scripts/setup',
            'docs/api',
            'docs/tutorials',
            'docs/architecture',
            'docker/mojo',
            'archived/old_scripts',
            'archived/old_tests'
        ]
        
        for dir_path in specialized_dirs:
            full_path = self.root / dir_path
            full_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"✅ Created specialized directory: {dir_path}")
    
    def move_test_files(self):
        """Move and organize test files."""
        logger.info("Organizing test files...")
        
        test_files = list(self.root.glob('test_*.py'))
        
        for test_file in test_files:
            if test_file.name.startswith('test_mojo') or test_file.name.startswith('test_real'):
                target = self.root / 'tests' / 'mojo' / test_file.name
            elif 'integration' in test_file.name or 'e2e' in test_file.name:
                target = self.root / 'tests' / 'e2e' / test_file.name
            elif 'mcp' in test_file.name:
                target = self.root / 'tests' / 'integration' / test_file.name
            else:
                target = self.root / 'tests' / 'unit' / test_file.name
            
            self._move_file(test_file, target)
        
        # Move test scripts
        test_scripts = list(self.root.glob('test_*.sh'))
        for script in test_scripts:
            target = self.root / 'tests' / script.name
            self._move_file(script, target)
    
    def move_log_files(self):
        """Move log files and results."""
        logger.info("Organizing log files...")
        
        # Move log files
        log_patterns = ['*.log', '*_output.txt', '*_results.json', 'debug_output.txt']
        for pattern in log_patterns:
            for log_file in self.root.glob(pattern):
                if 'mcp' in log_file.name:
                    target = self.root / 'logs' / 'mcp' / log_file.name
                else:
                    target = self.root / 'logs' / log_file.name
                self._move_file(log_file, target)
        
        # Move test results
        result_patterns = ['*_test_results*', '*_diagnostics*', 'verification_results*']
        for pattern in result_patterns:
            for result_file in self.root.glob(pattern):
                target = self.root / 'test-results' / result_file.name
                self._move_file(result_file, target)
    
    def move_script_files(self):
        """Move script files."""
        logger.info("Organizing script files...")
        
        script_patterns = ['run_*.sh', 'start_*.sh', 'stop_*.sh', 'restart_*.sh', 
                          'fix_*.sh', 'verify_*.sh', 'install_*.sh']
        
        for pattern in script_patterns:
            for script in self.root.glob(pattern):
                if 'mcp' in script.name:
                    target = self.root / 'scripts' / 'mcp' / script.name
                else:
                    target = self.root / 'scripts' / script.name
                self._move_file(script, target)
        
        # Move Python scripts that are tools
        tool_scripts = [
            'analyze_*.py', 'check_*.py', 'clean_*.py', 'clear_*.py',
            'compare_*.py', 'connect_*.py', 'debug_*.py', 'diagnose_*.py',
            'fix_*.py', 'generate_*.py', 'install_*.py', 'register_*.py',
            'setup_*.py', 'update_*.py', 'validate_*.py', 'verify_*.py'
        ]
        
        for pattern in tool_scripts:
            for script in self.root.glob(pattern):
                target = self.root / 'tools' / script.name
                self._move_file(script, target)
    
    def move_config_files(self):
        """Move configuration files."""
        logger.info("Organizing configuration files...")
        
        config_files = [
            '*_settings.json', 'mcp_*.json', 'hardware_detection.json',
            'vscode_*.json', 'post_*.json', 'pre_*.json'
        ]
        
        for pattern in config_files:
            for config in self.root.glob(pattern):
                if 'mcp' in config.name:
                    target = self.root / 'configs' / 'mcp' / config.name
                else:
                    target = self.root / 'configs' / config.name
                self._move_file(config, target)
        
        # Move Docker files
        docker_files = ['Dockerfile*', 'docker-compose*.yml']
        for pattern in docker_files:
            for docker_file in self.root.glob(pattern):
                target = self.root / 'docker' / docker_file.name
                self._move_file(docker_file, target)
    
    def move_mcp_files(self):
        """Move MCP-related files."""
        logger.info("Organizing MCP files...")
        
        # Move standalone MCP servers to archived
        mcp_servers = [
            'minimal_mcp_server.py', 'simple_mcp_server.py', 'enhanced_mcp_server.py',
            'improved_mcp_server.py', 'robust_mcp_server.py', 'standalone_mcp_server.py',
            'unified_mcp_server.py', 'working_mcp_server.py', 'clean_mcp_server.py'
        ]
        
        for server in mcp_servers:
            server_path = self.root / server
            if server_path.exists():
                target = self.root / 'archived' / 'old_scripts' / server
                self._move_file(server_path, target)
        
        # Keep only the final MCP server
        final_server = self.root / 'final_mcp_server.py'
        if final_server.exists():
            target = self.root / 'src' / 'mcp' / 'final_mcp_server.py'
            self._move_file(final_server, target)
    
    def clean_temp_files(self):
        """Clean up temporary and cache files."""
        logger.info("Cleaning temporary files...")
        
        # Remove cache directories
        cache_dirs = ['__pycache__', '.pytest_cache']
        for cache_dir in cache_dirs:
            cache_path = self.root / cache_dir
            if cache_path.exists():
                shutil.rmtree(cache_path)
                self.cleaned_files.append(str(cache_path))
                logger.info(f"🗑️ Removed cache directory: {cache_dir}")
        
        # Remove temporary files
        temp_patterns = ['*.pyc', '*.pyo', '*.tmp', '*.bak', '*.pid']
        for pattern in temp_patterns:
            for temp_file in self.root.glob(pattern):
                temp_file.unlink()
                self.cleaned_files.append(str(temp_file))
                logger.info(f"🗑️ Removed temp file: {temp_file.name}")
    
    def create_documentation_index(self):
        """Create documentation index files."""
        logger.info("Creating documentation index...")
        
        # Move existing markdown files to docs
        md_files = list(self.root.glob('*.md'))
        for md_file in md_files:
            if md_file.name.lower() not in ['readme.md', 'license.md']:
                target = self.root / 'docs' / md_file.name
                self._move_file(md_file, target)
    
    def _move_file(self, source: Path, target: Path):
        """Safely move a file."""
        try:
            if source.exists() and source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    logger.warning(f"Target exists, skipping: {target}")
                    return
                shutil.move(str(source), str(target))
                self.moved_files.append(f"{source} -> {target}")
                logger.info(f"📁 Moved: {source.name} -> {target.relative_to(self.root)}")
        except Exception as e:
            logger.error(f"Failed to move {source} to {target}: {e}")
    
    def generate_report(self):
        """Generate organization report."""
        report = {
            "organization_date": "2025-06-29",
            "moved_files": len(self.moved_files),
            "cleaned_files": len(self.cleaned_files),
            "directories_created": list(self.directories.values()),
            "file_moves": self.moved_files[:20],  # First 20 moves
            "cleaned_items": self.cleaned_files
        }
        
        report_path = self.root / 'ORGANIZATION_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Organization report saved to: {report_path}")
    
    def organize(self):
        """Run the complete organization process."""
        logger.info("🎯 Starting project organization...")
        
        try:
            self.create_directories()
            self.move_test_files()
            self.move_log_files() 
            self.move_script_files()
            self.move_config_files()
            self.move_mcp_files()
            self.create_documentation_index()
            self.clean_temp_files()
            self.generate_report()
            
            logger.info("✅ Project organization completed successfully!")
            logger.info(f"📊 Moved {len(self.moved_files)} files")
            logger.info(f"🗑️ Cleaned {len(self.cleaned_files)} temporary items")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Organization failed: {e}")
            return False

def main():
    """Main organization function."""
    root_path = Path(__file__).parent
    organizer = ProjectOrganizer(root_path)
    
    success = organizer.organize()
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
