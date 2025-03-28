# ipfs_accelerate_py.api_backends module initialization
# This module exposes all the API backend classes directly

import os
import logging
logger = logging.getLogger(__name__)

# Make sure model_list directory exists
model_list_dir = os.path.join(os.path.dirname(__file__), "model_list")
if not os.path.exists(model_list_dir):
    try:
        os.makedirs(model_list_dir, exist_ok=True)
        logger.info(f"Created model_list directory at {model_list_dir}")
    except Exception as e:
        logger.warning(f"Could not create model_list directory: {e}")

try:
    from .claude import claude
except ImportError as e:
    logger.debug(f"Failed to import claude: {e}")
    claude = None

try:
    from .openai_api import openai_api
except ImportError as e:
    logger.debug(f"Failed to import openai_api: {e}")
    openai_api = None

try:
    from .groq import groq
except ImportError as e:
    logger.debug(f"Failed to import groq: {e}")
    groq = None

try:
    from .gemini import gemini
except ImportError as e:
    logger.debug(f"Failed to import gemini: {e}")
    gemini = None

try:
    from .ollama import ollama
except ImportError as e:
    logger.debug(f"Failed to import ollama: {e}")
    ollama = None

try:
    from .hf_tgi import hf_tgi
except ImportError as e:
    logger.debug(f"Failed to import hf_tgi: {e}")
    hf_tgi = None

try:
    from .hf_tei import hf_tei
except ImportError as e:
    logger.debug(f"Failed to import hf_tei: {e}")
    hf_tei = None

try:
    from .llvm import llvm
except ImportError as e:
    logger.debug(f"Failed to import llvm: {e}")
    llvm = None

try:
    from .opea import opea
except ImportError as e:
    logger.debug(f"Failed to import opea: {e}")
    opea = None

try:
    from .ovms import ovms
except ImportError as e:
    logger.debug(f"Failed to import ovms: {e}")
    ovms = None

try:
    from .s3_kit import s3_kit
except ImportError as e:
    logger.debug(f"Failed to import s3_kit: {e}")
    s3_kit = None

# List of all backend classes
__all__ = [
    "claude", "openai_api", "groq", "gemini", "ollama", "hf_tgi", "hf_tei", "llvm", "opea", "ovms", "s3_kit"
]
