#!/usr/bin/env python3
"""
Update all model generators to support Mojo/MAX targets.
This script systematically updates all generators in the IPFS Accelerate framework
to ensure they can generate models targeting Mojo/MAX architectures.
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MojoMaxGeneratorUpdater:
    """Updates generators to support Mojo/MAX targets."""
    
    def __init__(self, root_dir: str = "."):
        """Initialize the updater."""
        self.root_dir = Path(root_dir)
        self.updated_files = []
        self.skipped_files = []
        self.errors = []
    
    def update_all_generators(self):
        """Update all generator files in the project."""
        logger.info("Starting Mojo/MAX generator update process...")
        
        # Find all generator files
        generator_files = self._find_generator_files()
        logger.info(f"Found {len(generator_files)} generator files to update")
        
        # Update each file
        for file_path in generator_files:
            try:
                self._update_generator_file(file_path)
                self.updated_files.append(file_path)
                logger.info(f"Updated: {file_path}")
            except Exception as e:
                self.errors.append((file_path, str(e)))
                logger.error(f"Error updating {file_path}: {e}")
        
        # Update hardware detection files
        self._update_hardware_detection_files()
        
        # Update API server files
        self._update_api_server_files()
        
        # Generate summary
        self._generate_update_summary()
    
    def _find_generator_files(self) -> List[Path]:
        """Find all generator-related files."""
        generator_files = []
        
        # Pattern matches for generator files
        patterns = [
            "**/generator*.py",
            "**/skill_*.py",
            "**/model_*.py",
            "**/generate_*.py"
        ]
        
        for pattern in patterns:
            generator_files.extend(self.root_dir.glob(pattern))
        
        # Filter out duplicates and test files that shouldn't be modified
        filtered_files = []
        skip_patterns = [
            "test_",
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "mojo_max_support.py"  # Skip our support file
        ]
        
        for file_path in set(generator_files):
            if file_path.is_file() and file_path.suffix == '.py':
                skip = False
                for pattern in skip_patterns:
                    if pattern in str(file_path):
                        skip = True
                        break
                if not skip:
                    filtered_files.append(file_path)
        
        return filtered_files
    
    def _update_generator_file(self, file_path: Path):
        """Update a single generator file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already updated
        if "mojo_max_support" in content or "MojoMaxTargetMixin" in content:
            self.skipped_files.append(file_path)
            return
        
        # Check if this is a skill file that needs updating
        if self._is_skill_file(content):
            content = self._update_skill_file_content(content, file_path)
        elif self._is_generator_core_file(content):
            content = self._update_generator_core_content(content)
        elif self._is_hardware_detection_file(content):
            content = self._update_hardware_detection_content(content)
        else:
            # Generic update for other generator files
            content = self._update_generic_generator_content(content)
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _is_skill_file(self, content: str) -> bool:
        """Check if this is a skill file."""
        return ("class " in content and "Skill" in content and 
                ("get_default_device" in content or "device" in content))
    
    def _is_generator_core_file(self, content: str) -> bool:
        """Check if this is a generator core file."""
        return ("class GeneratorCore" in content or 
                "hardware_info" in content and "context" in content)
    
    def _is_hardware_detection_file(self, content: str) -> bool:
        """Check if this is a hardware detection file."""
        return ("detect" in content and "hardware" in content and 
                ("cuda" in content or "rocm" in content))
    
    def _update_skill_file_content(self, content: str, file_path: Path) -> str:
        """Update a skill file to support Mojo/MAX."""
        # Add import for MojoMaxTargetMixin
        import_pattern = r"(import os\nimport sys.*?\n)"
        if "from .mojo_max_support import MojoMaxTargetMixin" not in content:
            content = re.sub(
                import_pattern,
                r"\1from .mojo_max_support import MojoMaxTargetMixin\n",
                content,
                flags=re.DOTALL
            )
        
        # Update class definition to inherit from MojoMaxTargetMixin
        class_pattern = r"class (\w+Skill):"
        if "MojoMaxTargetMixin" not in content:
            content = re.sub(
                class_pattern,
                r"class \1(MojoMaxTargetMixin):",
                content
            )
        
        # Update __init__ method to call super().__init__()
        init_pattern = r"(def __init__\(self[^)]*\):\s*\n\s*)(\"\"\".*?\"\"\"\s*\n\s*)?"
        if "super().__init__()" not in content:
            content = re.sub(
                init_pattern,
                r"\1\2super().__init__()\n        ",
                content,
                flags=re.DOTALL
            )
        
        # Update get_default_device method
        device_pattern = r"def get_default_device\(self\):.*?return \"cpu\""
        if "get_default_device_with_mojo_max" not in content:
            replacement = '''def get_default_device(self):
        """Get the best available device (legacy method, use get_default_device_with_mojo_max)."""
        return self.get_default_device_with_mojo_max()'''
            content = re.sub(device_pattern, replacement, content, flags=re.DOTALL)
        
        # Update process method to handle Mojo/MAX
        process_pattern = r"def process\(self, ([^)]+)\):\s*\n\s*(\"\"\".*?\"\"\"\s*\n\s*)?"
        if "process_with_mojo_max" not in content:
            mojo_max_check = '''def process(self, \\1):
        \\2# Check for Mojo/MAX target
        if self.device in ["mojo_max", "max", "mojo"]:
            return self.process_with_mojo_max(\\1, self.model_id)
        
        '''
            content = re.sub(process_pattern, mojo_max_check, content, flags=re.DOTALL)
        
        # Update device checks in load_model and process methods
        content = re.sub(
            r'if self\.device != "cpu":',
            'if self.device not in ["cpu", "mojo_max", "max", "mojo"]:',
            content
        )
        
        return content
    
    def _update_generator_core_content(self, content: str) -> str:
        """Update generator core file to include Mojo/MAX hardware flags."""
        # Add hardware availability flags for Mojo/MAX
        hardware_flags_pattern = r'("has_webgpu": hardware_info\.get\("webgpu", \{\}\)\.get\("available", False\))'
        replacement = r'''\1,
            "has_mojo": hardware_info.get("mojo", {}).get("available", False),
            "has_max": hardware_info.get("max", {}).get("available", False),
            "has_mojo_max": hardware_info.get("mojo", {}).get("available", False) or hardware_info.get("max", {}).get("available", False)'''
        
        if "has_mojo" not in content:
            content = re.sub(hardware_flags_pattern, replacement, content)
        
        return content
    
    def _update_hardware_detection_content(self, content: str) -> str:
        """Update hardware detection to include Mojo/MAX."""
        # Add mojo and max to hardware info dictionary
        hardware_dict_pattern = r'("webgpu": False,)\s*("system":)'
        replacement = r'''\1
        "mojo": False,
        "max": False,
        \2'''
        
        if '"mojo": False' not in content:
            content = re.sub(hardware_dict_pattern, replacement, content)
        
        # Add check_mojo_max call
        check_calls_pattern = r'(check_webnn_webgpu\(hardware_info\))'
        if "check_mojo_max" not in content:
            content = re.sub(check_calls_pattern, r'\1\n    check_mojo_max(hardware_info)', content)
        
        # Update best_available device logic
        best_available_pattern = r'(# Determine best available device\s*\n\s*if hardware_info\.get\("cuda", False\):)'
        replacement = r'''# Determine best available device
    if hardware_info.get("max", False):
        hardware_info["best_available"] = "max"
        hardware_info["torch_device"] = "cpu"  # MAX can use CPU as fallback
    elif hardware_info.get("mojo", False):
        hardware_info["best_available"] = "mojo"
        hardware_info["torch_device"] = "cpu"  # Mojo can use CPU as fallback
    elif hardware_info.get("cuda", False):'''
        
        if 'hardware_info.get("max", False)' not in content:
            content = re.sub(best_available_pattern, replacement, content, flags=re.DOTALL)
        
        return content
    
    def _update_generic_generator_content(self, content: str) -> str:
        """Update generic generator files."""
        # Add environment variable checks for Mojo/MAX targets
        if "USE_MOJO_MAX_TARGET" not in content and "target" in content.lower():
            # Add environment variable support
            env_check = '''
# Check for Mojo/MAX target environment variable
if os.environ.get("USE_MOJO_MAX_TARGET", "").lower() in ("1", "true", "yes"):
    target_backend = "mojo_max"
'''
            # Insert after imports
            import_end = content.find('\n\n')
            if import_end > 0:
                content = content[:import_end] + env_check + content[import_end:]
        
        return content
    
    def _update_hardware_detection_files(self):
        """Update specific hardware detection files."""
        # Already handled in the main loop, but this could add specific updates
        pass
    
    def _update_api_server_files(self):
        """Update API server files to include Mojo/MAX hardware in responses."""
        api_files = list(self.root_dir.glob("**/generator_api_server.py"))
        
        for file_path in api_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add Mojo/MAX to hardware support
                if "has_mojo" not in content:
                    hardware_pattern = r'(hardware: List\[str\] = \["cpu"\])'
                    content = re.sub(
                        hardware_pattern,
                        r'hardware: List[str] = ["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu", "mojo", "max"]',
                        content
                    )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.updated_files.append(file_path)
                logger.info(f"Updated API server: {file_path}")
                
            except Exception as e:
                self.errors.append((file_path, str(e)))
                logger.error(f"Error updating API server {file_path}: {e}")
    
    def _generate_update_summary(self):
        """Generate a summary of the update process."""
        summary = f"""
=== Mojo/MAX Generator Update Summary ===

Updated Files: {len(self.updated_files)}
Skipped Files: {len(self.skipped_files)}
Errors: {len(self.errors)}

Updated Files:
"""
        for file_path in self.updated_files:
            summary += f"  - {file_path}\n"
        
        if self.skipped_files:
            summary += "\nSkipped Files (already updated or not applicable):\n"
            for file_path in self.skipped_files:
                summary += f"  - {file_path}\n"
        
        if self.errors:
            summary += "\nErrors:\n"
            for file_path, error in self.errors:
                summary += f"  - {file_path}: {error}\n"
        
        summary += f"""
=== Changes Made ===
1. Added MojoMaxTargetMixin inheritance to skill classes
2. Updated device detection to include Mojo/MAX targets
3. Added environment variable support (USE_MOJO_MAX_TARGET)
4. Updated hardware availability flags in generator contexts
5. Modified device handling in model loading and processing
6. Added fallback mechanisms for Mojo/MAX unavailability

=== Next Steps ===
1. Test updated generators with Mojo/MAX targets
2. Verify environment variable functionality
3. Validate fallback behavior when Mojo/MAX is not available
4. Run comprehensive tests with USE_MOJO_MAX_TARGET=1
"""
        
        # Write summary to file
        with open(self.root_dir / "MOJO_MAX_GENERATOR_UPDATE_SUMMARY.md", 'w') as f:
            f.write(summary)
        
        logger.info("Update complete! See MOJO_MAX_GENERATOR_UPDATE_SUMMARY.md for details")
        print(summary)

def main():
    """Main entry point."""
    updater = MojoMaxGeneratorUpdater()
    updater.update_all_generators()

if __name__ == "__main__":
    main()
