"""
Configuration utilities for generators.
"""

class GeneratorConfig:
    """Common configuration for all generators."""
    
    # Default paths
    DEFAULT_TEMPLATE_DB_PATH = "templates/model_templates.duckdb"
    DEFAULT_OUTPUT_DIR = "generated"
    
    # Default hardware backends
    @staticmethod
    def get_default_hardware_backends():
        """Return the default hardware backends to support."""
        return ["cpu", "cuda", "qualcomm", "openvino", "webgpu", "webnn"]
    
    # Default model types
    @staticmethod
    def get_key_model_types():
        """Return the key model types to support."""
        return [
            "bert", "t5", "llama", "whisper", 
            "clap", "vit", "clip", "wav2vec2"
        ]
