#!/usr/bin/env python3
"""
Utility script to fix issues with the template system.

This script addresses common issues that might arise with the template system,
such as missing directories, import errors, etc.
"""

import os
import sys
import logging
import importlib
import pkgutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_directory_structure():
    """Check that all required directories exist."""
    required_dirs = [
        "templates",
        "generated_reference",
        "hardware",
        "generators"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
            logger.warning(f"Missing directory: {dir_name}")
    
    if missing_dirs:
        logger.info("Creating missing directories...")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
        
        # Create __init__.py files in each module directory
        for dir_name in missing_dirs:
            if dir_name not in ["generated_reference"]:  # No __init__.py needed in output dirs
                init_file = os.path.join(dir_name, "__init__.py")
                with open(init_file, "w") as f:
                    f.write(f"""#!/usr/bin/env python3
\"\"\"
{dir_name.capitalize()} module for the refactored generator suite.
\"\"\"
""")
                logger.info(f"Created {init_file}")


def check_imports():
    """Check that all required modules can be imported."""
    required_modules = [
        "templates.base_hardware",
        "templates.base_architecture",
        "templates.base_pipeline",
        "templates.template_composer",
        "templates.cuda_hardware",
        "templates.rocm_hardware",
        "templates.openvino_hardware",
        "templates.apple_hardware",
        "templates.qualcomm_hardware",
        "templates.image_pipeline"
    ]
    
    # Skip these modules as they may contain template syntax that's not valid Python
    skip_modules = [
        "templates.decoder_only",
        "templates.encoder_only",
        "templates.vision",
        "templates.speech",
        "templates.vision_text",
        "templates.encoder_decoder"
    ]
    
    missing_modules = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ Import successful: {module_name}")
        except ImportError as e:
            missing_modules.append((module_name, str(e)))
            logger.warning(f"❌ Import failed: {module_name} - {e}")
    
    return missing_modules


def check_hardware_templates():
    """Check that all hardware templates are properly implemented."""
    try:
        # Import the base hardware template
        from templates.base_hardware import BaseHardwareTemplate
        
        # List of hardware template modules to check
        hardware_modules = [
            ("templates.base_hardware", "CPUHardwareTemplate"),
            ("templates.cuda_hardware", "CudaHardwareTemplate"),
            ("templates.rocm_hardware", "RocmHardwareTemplate"),
            ("templates.openvino_hardware", "OpenvinoHardwareTemplate"),
            ("templates.apple_hardware", "AppleHardwareTemplate"),
            ("templates.qualcomm_hardware", "QualcommHardwareTemplate")
        ]
        
        # Check each hardware template
        for module_name, class_name in hardware_modules:
            try:
                module = importlib.import_module(module_name)
                template_class = getattr(module, class_name)
                
                # Check if it's a subclass of BaseHardwareTemplate
                if issubclass(template_class, BaseHardwareTemplate):
                    # Try to instantiate it
                    template = template_class()
                    logger.info(f"✅ Hardware template validated: {class_name}")
                else:
                    logger.warning(f"❌ {class_name} is not a subclass of BaseHardwareTemplate")
            except ImportError as e:
                logger.warning(f"❌ Could not import {module_name}: {e}")
            except AttributeError as e:
                logger.warning(f"❌ Could not find {class_name} in {module_name}: {e}")
            except Exception as e:
                logger.warning(f"❌ Error validating {class_name}: {e}")
    
    except ImportError as e:
        logger.error(f"❌ Could not import BaseHardwareTemplate: {e}")


def check_architecture_templates():
    """Check that all architecture templates are properly implemented."""
    try:
        # Import the base architecture template
        from templates.base_architecture import BaseArchitectureTemplate
        
        # Try to import all architecture templates
        arch_modules = [
            ("templates.encoder_only", "EncoderOnlyArchitectureTemplate"),
            ("templates.decoder_only", "DecoderOnlyArchitectureTemplate"),
            ("templates.encoder_decoder", "EncoderDecoderArchitectureTemplate"),
            ("templates.vision", "VisionArchitectureTemplate"),
            ("templates.speech", "SpeechArchitectureTemplate"),
            ("templates.vision_text", "VisionTextArchitectureTemplate")
        ]
        
        # Check each architecture template
        for module_name, class_name in arch_modules:
            try:
                module = importlib.import_module(module_name)
                template_class = getattr(module, class_name)
                
                # Check if it's a subclass of BaseArchitectureTemplate
                if issubclass(template_class, BaseArchitectureTemplate):
                    # Try to instantiate it
                    template = template_class()
                    logger.info(f"✅ Architecture template validated: {class_name}")
                else:
                    logger.warning(f"❌ {class_name} is not a subclass of BaseArchitectureTemplate")
            except ImportError as e:
                logger.warning(f"❌ Could not import {module_name}: {e}")
            except AttributeError as e:
                logger.warning(f"❌ Could not find {class_name} in {module_name}: {e}")
            except Exception as e:
                logger.warning(f"❌ Error validating {class_name}: {e}")
    
    except ImportError as e:
        logger.error(f"❌ Could not import BaseArchitectureTemplate: {e}")


def check_pipeline_templates():
    """Check that all pipeline templates are properly implemented."""
    try:
        # Import the base pipeline template
        from templates.base_pipeline import BasePipelineTemplate
        
        # Try to import pipeline templates
        pipeline_modules = [
            ("templates.base_pipeline", "TextPipelineTemplate"),
            ("templates.image_pipeline", "ImagePipelineTemplate")
        ]
        
        # Check each pipeline template
        for module_name, class_name in pipeline_modules:
            try:
                module = importlib.import_module(module_name)
                template_class = getattr(module, class_name)
                
                # Check if it's a subclass of BasePipelineTemplate
                if issubclass(template_class, BasePipelineTemplate):
                    # Try to instantiate it
                    template = template_class()
                    logger.info(f"✅ Pipeline template validated: {class_name}")
                else:
                    logger.warning(f"❌ {class_name} is not a subclass of BasePipelineTemplate")
            except ImportError as e:
                logger.warning(f"❌ Could not import {module_name}: {e}")
            except AttributeError as e:
                logger.warning(f"❌ Could not find {class_name} in {module_name}: {e}")
            except Exception as e:
                logger.warning(f"❌ Error validating {class_name}: {e}")
    
    except ImportError as e:
        logger.error(f"❌ Could not import BasePipelineTemplate: {e}")


def check_template_composer():
    """Check that the template composer is properly implemented."""
    try:
        # Import the template composer
        from templates.template_composer import TemplateComposer
        logger.info(f"✅ Template composer imported successfully")
        
        # Check if we can import the create_reference_implementations.py functions
        try:
            from create_reference_implementations import (
                create_hardware_templates,
                create_architecture_templates,
                create_pipeline_templates
            )
            
            # Try to create templates
            hardware_templates = create_hardware_templates()
            architecture_templates = create_architecture_templates()
            pipeline_templates = create_pipeline_templates()
            
            # Try to instantiate the template composer
            composer = TemplateComposer(
                hardware_templates=hardware_templates,
                architecture_templates=architecture_templates,
                pipeline_templates=pipeline_templates,
                output_dir="generated_reference"
            )
            
            logger.info(f"✅ Template composer instantiated successfully")
        except ImportError as e:
            logger.warning(f"❌ Could not import from create_reference_implementations.py: {e}")
        except Exception as e:
            logger.warning(f"❌ Error instantiating template composer: {e}")
    
    except ImportError as e:
        logger.error(f"❌ Could not import TemplateComposer: {e}")


def main():
    """Run the checks and fix any issues."""
    logger.info("Checking template system...")
    
    # Check directory structure
    check_directory_structure()
    
    # Check imports
    missing_modules = check_imports()
    
    # Check hardware templates
    check_hardware_templates()
    
    # Check architecture templates
    check_architecture_templates()
    
    # Check pipeline templates
    check_pipeline_templates()
    
    # Check template composer
    check_template_composer()
    
    if missing_modules:
        logger.warning("Some modules could not be imported. Please fix the issues.")
        for module_name, error in missing_modules:
            logger.warning(f"  - {module_name}: {error}")
    else:
        logger.info("All required modules can be imported.")
    
    logger.info("Check completed. Run test_template_system.py to verify the template system.")


if __name__ == "__main__":
    main()