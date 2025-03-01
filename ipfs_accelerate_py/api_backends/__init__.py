# Import modules with try/except blocks to handle missing dependencies
import logging
from importlib import import_module

logger = logging.getLogger(__name__)

# Dictionary to store imported modules
api_modules = {}

# List of modules to try importing
module_names = [
    "chat_format",
    "openai_api",
    "s3_kit",
    "hf_tei",
    "hf_tgi", 
    "groq",
    "ovms",
    "llvm",
    "ollama",
    "opea",
    "apis",
    # "api_models_registry"
]

# Import each module, handling import errors
for module_name in module_names:
    try:
        module = import_module(f".{module_name}", package="ipfs_accelerate_py.api_backends")
        # Get the class with the same name as the module
        if hasattr(module, module_name):
            class_obj = getattr(module, module_name)
            # Add to globals to support "from .module import module" syntax
            globals()[module_name] = class_obj
            # Store in dictionary
            api_modules[module_name] = class_obj
        else:
            logger.warning(f"Module {module_name} imported but class {module_name} not found in it")
    except ImportError as e:
        logger.debug(f"Could not import module {module_name}: {e}")
        # Create a placeholder that will raise a more helpful error if used
        class ModuleNotAvailable:
            def __init__(self, *args, **kwargs):
                raise ImportError(f"The {module_name} module is not available: {e}")
        
        globals()[module_name] = ModuleNotAvailable
        api_modules[module_name] = ModuleNotAvailable