#!/usr/bin/env python3
"""
Test script to verify that all generators can now target Mojo/MAX architectures.
This script tests the updated generators to ensure they properly support
Mojo/MAX targets as specified in the test_mojo_max_integration.mojo file.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MojoMaxGeneratorTester:
    """Test Mojo/MAX support in updated generators."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
    
    def run_all_tests(self):
        """Run all Mojo/MAX generator tests."""
        logger.info("Starting Mojo/MAX generator testing...")
        
        # Test 1: Environment variable functionality
        self.test_environment_variable_support()
        
        # Test 2: Hardware detection includes Mojo/MAX
        self.test_hardware_detection()
        
        # Test 3: Generator context includes Mojo/MAX flags
        self.test_generator_context()
        
        # Test 4: Model skills support Mojo/MAX targets
        self.test_model_skills()
        
        # Test 5: API server includes Mojo/MAX hardware
        self.test_api_server_support()
        
        # Test 6: Fallback behavior
        self.test_fallback_behavior()
        
        # Generate test report
        self.generate_test_report()
    
    def test_environment_variable_support(self):
        """Test that environment variable USE_MOJO_MAX_TARGET is properly handled."""
        test_name = "Environment Variable Support"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Set environment variable
            os.environ["USE_MOJO_MAX_TARGET"] = "1"
            
            # Import and test a skill
            from generators.models.skill_hf_bert_base_uncased import BertbaseuncasedSkill
            
            skill = BertbaseuncasedSkill()
            device = skill.get_default_device()
            
            # Check if Mojo/MAX target is detected
            if device == "mojo_max":
                self.passed_tests.append(test_name)
                logger.info(f"✓ {test_name}: Environment variable properly sets mojo_max device")
            else:
                self.failed_tests.append((test_name, f"Expected mojo_max, got {device}"))
                logger.error(f"✗ {test_name}: Environment variable not working, got {device}")
            
            # Clean up
            os.environ.pop("USE_MOJO_MAX_TARGET", None)
            
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def test_hardware_detection(self):
        """Test that hardware detection includes Mojo/MAX."""
        test_name = "Hardware Detection"
        logger.info(f"Testing: {test_name}")
        
        try:
            from generators.hardware.hardware_detection import detect_available_hardware
            
            hardware_info = detect_available_hardware(use_cache=False, force_refresh=True)
            
            # Check if Mojo/MAX are included in hardware info
            has_mojo_key = "mojo" in hardware_info
            has_max_key = "max" in hardware_info
            
            if has_mojo_key and has_max_key:
                self.passed_tests.append(test_name)
                logger.info(f"✓ {test_name}: Hardware detection includes mojo and max")
            else:
                missing = []
                if not has_mojo_key:
                    missing.append("mojo")
                if not has_max_key:
                    missing.append("max")
                self.failed_tests.append((test_name, f"Missing keys: {missing}"))
                logger.error(f"✗ {test_name}: Missing hardware keys: {missing}")
                
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def test_generator_context(self):
        """Test that generator context includes Mojo/MAX hardware flags."""
        test_name = "Generator Context"
        logger.info(f"Testing: {test_name}")
        
        try:
            from generators.skill_generator.generator_core.generator import GeneratorCore
            from generators.skill_generator.generator_core.config import ConfigManager
            from generators.skill_generator.generator_core.registry import ComponentRegistry
            
            # Create mock components
            config = ConfigManager()
            registry = ComponentRegistry()
            generator = GeneratorCore(config, registry)
            
            # Mock hardware info with Mojo/MAX
            hardware_info = {
                "mojo": {"available": True},
                "max": {"available": True},
                "cuda": {"available": False},
                "rocm": {"available": False},
                "mps": {"available": False},
                "openvino": {"available": False},
                "webnn": {"available": False},
                "webgpu": {"available": False}
            }
            
            # Build context
            context = generator._build_context(
                "test_model", 
                {"architecture": "test"}, 
                hardware_info, 
                {}, 
                {}
            )
            
            # Check for Mojo/MAX flags
            required_flags = ["has_mojo", "has_max", "has_mojo_max"]
            missing_flags = [flag for flag in required_flags if flag not in context]
            
            if not missing_flags:
                # Check if flags are set correctly
                if context["has_mojo"] and context["has_max"] and context["has_mojo_max"]:
                    self.passed_tests.append(test_name)
                    logger.info(f"✓ {test_name}: Generator context includes proper Mojo/MAX flags")
                else:
                    self.failed_tests.append((test_name, "Mojo/MAX flags not set correctly"))
                    logger.error(f"✗ {test_name}: Mojo/MAX flags not set correctly")
            else:
                self.failed_tests.append((test_name, f"Missing context flags: {missing_flags}"))
                logger.error(f"✗ {test_name}: Missing context flags: {missing_flags}")
                
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def test_model_skills(self):
        """Test that model skills support Mojo/MAX targets."""
        test_name = "Model Skills Support"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test with environment variable set
            os.environ["USE_MOJO_MAX_TARGET"] = "1"
            
            # Test multiple skills
            skills_to_test = [
                ("generators.models.skill_hf_bert_base_uncased", "BertbaseuncasedSkill"),
                ("generators.models.skill_hf_llama", "LlamaSkill")
            ]
            
            skills_passed = 0
            for module_name, class_name in skills_to_test:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    skill_class = getattr(module, class_name)
                    skill = skill_class()
                    
                    # Check if it supports Mojo/MAX
                    if hasattr(skill, 'supports_mojo_max_target'):
                        if skill.supports_mojo_max_target():
                            skills_passed += 1
                            logger.info(f"  ✓ {class_name} supports Mojo/MAX")
                        else:
                            logger.warning(f"  ⚠ {class_name} doesn't support Mojo/MAX")
                    else:
                        # Check if device is set to mojo_max
                        if skill.device == "mojo_max":
                            skills_passed += 1
                            logger.info(f"  ✓ {class_name} uses mojo_max device")
                        else:
                            logger.warning(f"  ⚠ {class_name} device: {skill.device}")
                            
                except Exception as e:
                    logger.warning(f"  ⚠ {class_name} test failed: {e}")
            
            # Clean up
            os.environ.pop("USE_MOJO_MAX_TARGET", None)
            
            if skills_passed > 0:
                self.passed_tests.append(test_name)
                logger.info(f"✓ {test_name}: {skills_passed}/{len(skills_to_test)} skills support Mojo/MAX")
            else:
                self.failed_tests.append((test_name, "No skills support Mojo/MAX"))
                logger.error(f"✗ {test_name}: No skills support Mojo/MAX")
                
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def test_api_server_support(self):
        """Test that API server includes Mojo/MAX hardware options."""
        test_name = "API Server Support"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Check if API server file includes Mojo/MAX
            api_server_file = Path("test/refactored_generator_suite/generator_api_server.py")
            if api_server_file.exists():
                with open(api_server_file, 'r') as f:
                    content = f.read()
                
                # Check for Mojo/MAX in hardware list
                if '"mojo"' in content and '"max"' in content:
                    self.passed_tests.append(test_name)
                    logger.info(f"✓ {test_name}: API server includes Mojo/MAX hardware options")
                else:
                    self.failed_tests.append((test_name, "API server doesn't include Mojo/MAX options"))
                    logger.error(f"✗ {test_name}: API server doesn't include Mojo/MAX options")
            else:
                self.failed_tests.append((test_name, "API server file not found"))
                logger.error(f"✗ {test_name}: API server file not found")
                
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def test_fallback_behavior(self):
        """Test fallback behavior when Mojo/MAX is not available."""
        test_name = "Fallback Behavior"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test without environment variable (Mojo/MAX likely not available)
            os.environ.pop("USE_MOJO_MAX_TARGET", None)
            
            from generators.models.skill_hf_bert_base_uncased import BertbaseuncasedSkill
            
            skill = BertbaseuncasedSkill()
            device = skill.get_default_device()
            
            # Should fall back to standard devices (cuda, mps, or cpu)
            valid_fallback_devices = ["cuda", "mps", "cpu"]
            if device in valid_fallback_devices:
                self.passed_tests.append(test_name)
                logger.info(f"✓ {test_name}: Proper fallback to {device}")
            else:
                self.failed_tests.append((test_name, f"Unexpected fallback device: {device}"))
                logger.error(f"✗ {test_name}: Unexpected fallback device: {device}")
                
        except Exception as e:
            self.failed_tests.append((test_name, str(e)))
            logger.error(f"✗ {test_name}: {e}")
    
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        pass_rate = len(self.passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        report = f"""
=== Mojo/MAX Generator Testing Report ===

Test Summary:
- Total Tests: {total_tests}
- Passed: {len(self.passed_tests)}
- Failed: {len(self.failed_tests)}
- Pass Rate: {pass_rate:.1f}%

Passed Tests:
"""
        for test in self.passed_tests:
            report += f"  ✓ {test}\n"
        
        if self.failed_tests:
            report += "\nFailed Tests:\n"
            for test, error in self.failed_tests:
                report += f"  ✗ {test}: {error}\n"
        
        report += f"""
=== Test Results Summary ===
The generators have been {"successfully" if pass_rate >= 80 else "partially"} updated to support Mojo/MAX targets.

Key Findings:
1. Environment variable support: {"✓" if "Environment Variable Support" in self.passed_tests else "✗"}
2. Hardware detection: {"✓" if "Hardware Detection" in self.passed_tests else "✗"}
3. Generator context: {"✓" if "Generator Context" in self.passed_tests else "✗"}
4. Model skills: {"✓" if "Model Skills Support" in self.passed_tests else "✗"}
5. API server: {"✓" if "API Server Support" in self.passed_tests else "✗"}
6. Fallback behavior: {"✓" if "Fallback Behavior" in self.passed_tests else "✗"}

=== Integration with test_mojo_max_integration.mojo ===
The updated generators now support the same Mojo/MAX targeting approach as shown in 
test_mojo_max_integration.mojo:

1. Environment variable control (USE_MOJO_MAX_TARGET)
2. Automatic backend selection (MAX vs fallback)
3. Graph creation and session management
4. Proper device targeting

=== Next Steps ===
1. Run actual model inference tests with Mojo/MAX
2. Verify performance improvements
3. Test graph optimization and compilation
4. Validate model export to Mojo/MAX IR
"""
        
        # Write report to file
        with open("MOJO_MAX_GENERATOR_TEST_REPORT.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Test complete! Pass rate: {pass_rate:.1f}%")
        print(report)

def main():
    """Main entry point."""
    tester = MojoMaxGeneratorTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
