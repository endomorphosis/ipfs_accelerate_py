#!/usr/bin/env python3

# Import hardware detection capabilities if available:
try:
    from scripts.generators.hardware.hardware_detection import ()
    HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
    detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    """
    Test Registry Fixer for Hugging Face models

    This script fixes the MODEL_REGISTRY in test_generator.py by adding the missing model families
    and then generates the test files for them.

Usage:
    python test_registry_fixer.py
    """

    import os
    import sys
    import re
    import subprocess
    from pathlib import Path

def get_missing_model_families()):
    """Get the list of missing model families based on huggingface_model_types.json and existing tests."""
    # Models identified as missing from previous analysis
    missing_models = [],
    'dino',
    'qdqbert',
    'flan',
    'stablelm',
    'open-llama',
    'mpt',
    'bloom-7b1',
    'auto',
    'falcon-7b',
    'galactica',
    'qwen3_vl'
    ]
    
    # Check which tests actually exist
    existing_tests = [],]
    for file in os.listdir()'.'):
        if file.startswith()'test_hf_') and file.endswith()'.py'):
            model_name = file[],8:-3]  # Remove 'test_hf_' and '.py'
            existing_tests.append()model_name)
    
            missing_test_files = [],]
    for model in missing_models:
        normalized_model = model.replace()'-', '_').lower())
        if normalized_model not in existing_tests:
            missing_test_files.append()model)
    
        return missing_test_files

def create_model_registry_entries()):
    """Create registry entries for missing models."""
    registry_entries = {}}}}}}}}}}}}}}}
    'dino': {}}}}}}}}}}}}}}}
    'family_name': 'DINO',
    'description': 'DINO vision models for self-supervised learning',
    'default_model': 'facebook/dino-vitb16',
    'class': 'DinoForImageClassification',
    'task': 'image-classification',
    'inputs': {}}}}}}}}}}}}}}}'image_url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
    'dependencies': [],'transformers', 'pillow', 'requests'],
    'task_specific_args': {}}}}}}}}}}}}}}}}
    },
    'qdqbert': {}}}}}}}}}}}}}}}
    'family_name': 'QDQBERT',
    'description': 'Quantized-Dequantized BERT models',
    'default_model': 'bert-base-uncased-qdq',
    'class': 'QDQBertForMaskedLM',
    'task': 'fill-mask',
    'inputs': {}}}}}}}}}}}}}}}'text': 'The quick brown fox jumps over the [],MASK] dog.'},
    'dependencies': [],'transformers', 'tokenizers'],
    'task_specific_args': {}}}}}}}}}}}}}}}'top_k': 5}
    },
    'flan': {}}}}}}}}}}}}}}}
    'family_name': 'FLAN',
    'description': 'FLAN instruction-tuned models',
    'default_model': 'google/flan-t5-small',
    'class': 'FlanT5ForConditionalGeneration',
    'task': 'text2text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'Translate to French: How are you?'},
    'dependencies': [],'transformers', 'tokenizers', 'sentencepiece'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 50}
    },
    'stablelm': {}}}}}}}}}}}}}}}
    'family_name': 'StableLM',
    'description': 'StableLM causal language models',
    'default_model': 'stabilityai/stablelm-base-alpha-7b',
    'class': 'StableLmForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'StableLM is a language model that'},
    'dependencies': [],'transformers', 'tokenizers', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'open-llama': {}}}}}}}}}}}}}}}
    'family_name': 'Open-LLaMA',
    'description': 'Open-LLaMA causal language models',
    'default_model': 'openlm-research/open_llama_7b',
    'class': 'OpenLlamaForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'Open-LLaMA is a model that'},
    'dependencies': [],'transformers', 'tokenizers', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'mpt': {}}}}}}}}}}}}}}}
    'family_name': 'MPT',
    'description': 'MPT causal language models',
    'default_model': 'mosaicml/mpt-7b',
    'class': 'MptForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'MPT is a language model that'},
    'dependencies': [],'transformers', 'tokenizers', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'bloom-7b1': {}}}}}}}}}}}}}}}
    'family_name': 'BLOOM-7B1',
    'description': 'BLOOM-7B1 language model',
    'default_model': 'bigscience/bloom-7b1',
    'class': 'BloomForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'BLOOM is a language model that'},
    'dependencies': [],'transformers', 'tokenizers', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'auto': {}}}}}}}}}}}}}}}
    'family_name': 'Auto',
    'description': 'Auto-detected model classes',
    'default_model': 'bert-base-uncased',
    'class': 'AutoModel',
    'task': 'feature-extraction',
    'inputs': {}}}}}}}}}}}}}}}'text': 'This is a test input for Auto model classes.'},
    'dependencies': [],'transformers', 'tokenizers'],
    'task_specific_args': {}}}}}}}}}}}}}}}}
    },
    'falcon-7b': {}}}}}}}}}}}}}}}
    'family_name': 'Falcon-7B',
    'description': 'Falcon-7B causal language model',
    'default_model': 'tiiuae/falcon-7b',
    'class': 'FalconForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'Falcon is a language model that'},
    'dependencies': [],'transformers', 'tokenizers', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'galactica': {}}}}}}}}}}}}}}}
    'family_name': 'Galactica',
    'description': 'Galactica scientific language models',
    'default_model': 'facebook/galactica-125m',
    'class': 'OPTForCausalLM',
    'task': 'text-generation',
    'inputs': {}}}}}}}}}}}}}}}'text': 'The theory of relativity states that'},
    'dependencies': [],'transformers', 'tokenizers'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100, 'min_length': 30}
    },
    'qwen3_vl': {}}}}}}}}}}}}}}}
    'family_name': 'Qwen3-VL',
    'description': 'Qwen3 vision-language models',
    'default_model': 'Qwen/Qwen3-VL-7B',
    'class': 'Qwen3VLForConditionalGeneration',
    'task': 'image-to-text',
    'inputs': {}}}}}}}}}}}}}}}
    'image_url': 'http://images.cocodataset.org/val2017/000000039769.jpg',
    'text': 'What do you see in this image?'
    },
    'dependencies': [],'transformers', 'pillow', 'requests', 'accelerate'],
    'task_specific_args': {}}}}}}}}}}}}}}}'max_length': 100}
    }
    }
    
        return registry_entries

def generate_registry_entry()model_id, model_info):
    """Generate a registry entry string for test_generator.py."""
    family_name = model_info[],'family_name']
    description = model_info[],'description']
    default_model = model_info[],'default_model']
    class_name = model_info[],'class']
    task = model_info[],'task']
    inputs = model_info[],'inputs']
    dependencies = model_info[],'dependencies']
    task_specific_args = model_info[],'task_specific_args']
    
    # Format inputs
    inputs_str = "{}}}}}}}}}}}}}}}\n"
    for k, v in inputs.items()):
        if isinstance()v, str):
            inputs_str += f'            "{}}}}}}}}}}}}}}}k}": "{}}}}}}}}}}}}}}}v}",\n'
        else:
            inputs_str += f'            "{}}}}}}}}}}}}}}}k}": {}}}}}}}}}}}}}}}v},\n'
            inputs_str += "        }"
    
    # Create models dictionary
            models_str = "{}}}}}}}}}}}}}}}\n"
            models_str += f'            "{}}}}}}}}}}}}}}}default_model}": {}}}}}}}}}}}}}}}{}}}}}}}}}}}}}}}\n'
            models_str += f'                "description": "{}}}}}}}}}}}}}}}family_name} model",\n'
            models_str += f'                "class": "{}}}}}}}}}}}}}}}class_name}"\n'
            models_str += '            }\n'
            models_str += "        }"
    
    # Format task specific args
            task_args_str = "{}}}}}}}}}}}}}}}\n"
    for k, v in task_specific_args.items()):
        if isinstance()v, str):
            task_args_str += f'                "{}}}}}}}}}}}}}}}k}": "{}}}}}}}}}}}}}}}v}",\n'
        else:
            task_args_str += f'                "{}}}}}}}}}}}}}}}k}": {}}}}}}}}}}}}}}}v},\n'
            task_args_str += "            }"
    
            entry = f"""
            "{}}}}}}}}}}}}}}}model_id}": {}}}}}}}}}}}}}}}{}}}}}}}}}}}}}}}
            "family_name": "{}}}}}}}}}}}}}}}family_name}",
            "description": "{}}}}}}}}}}}}}}}description}",
            "default_model": "{}}}}}}}}}}}}}}}default_model}",
            "class": "{}}}}}}}}}}}}}}}class_name}",
            "test_class": "Test{}}}}}}}}}}}}}}}family_name.replace()'-', '')}Models",
            "module_name": "test_hf_{}}}}}}}}}}}}}}}model_id.lower()).replace()'-', '_')}",
            "tasks": [],"{}}}}}}}}}}}}}}}task}"],
            "inputs": {}}}}}}}}}}}}}}}inputs_str},
            "dependencies": {}}}}}}}}}}}}}}}str()dependencies)},
            "task_specific_args": {}}}}}}}}}}}}}}}{}}}}}}}}}}}}}}}
            "{}}}}}}}}}}}}}}}task}": {}}}}}}}}}}}}}}}task_args_str}
            }},
            "models": {}}}}}}}}}}}}}}}models_str}
            }}"""
    
    return entry:
def update_test_generator()):
    """Update the test_generator.py file to add missing model families."""
    # Read test_generator.py
    with open()'test_generator.py', 'r') as f:
        content = f.read())
    
    # Locate MODEL_REGISTRY definition
        model_registry_match = re.search()r"MODEL_REGISTRY\s*=\s*\{}}}}}}}}}}}}}}}()[],^}]+)\}", content, re.DOTALL)
    if not model_registry_match:
        print()"Could not find MODEL_REGISTRY in test_generator.py")
        return False
    
    # Get missing models and create registry entries
        missing_models = get_missing_model_families())
        registry_entries = create_model_registry_entries())
    
    # Create new registry entries for each missing model
        new_entries = ""
    for model_id in missing_models:
        if model_id in registry_entries:
            new_entries += generate_registry_entry()model_id, registry_entries[],model_id])
    
    # Add new entries to MODEL_REGISTRY
            old_registry = model_registry_match.group()0)
            new_registry = old_registry.replace()"}", new_entries + "\n}")
    
    # Update test_generator.py
            new_content = content.replace()old_registry, new_registry)
    
    # Create a backup
    with open()'test_generator.py.bak', 'w') as f:
        f.write()content)
    
    # Write updated file
    with open()'test_generator.py', 'w') as f:
        f.write()new_content)
    
        print()f"Added {}}}}}}}}}}}}}}}len()missing_models)} new model families to MODEL_REGISTRY")
        return missing_models

def generate_test_files()missing_models):
    """Generate test files for missing models using test_generator.py."""
    for model_id in missing_models:
        print()f"Generating test file for {}}}}}}}}}}}}}}}model_id}...")
        model_id_normalized = model_id.lower()).replace()'-', '_')
        
        # Run test_generator.py to generate the test file
        command = f"python test_generator.py --generate {}}}}}}}}}}}}}}}model_id}"
        result = subprocess.run()command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print()f"Successfully generated test_hf_{}}}}}}}}}}}}}}}model_id_normalized}.py")
        else:
            print()f"Failed to generate test for {}}}}}}}}}}}}}}}model_id}: {}}}}}}}}}}}}}}}result.stderr}")
    
            return True

def main()):
    """Main function to fix the registry and generate test files."""
    print()"Starting Test Registry Fixer...")
    
    # Update test_generator.py with missing model families
    missing_models = update_test_generator())
    if not missing_models:
        print()"Failed to update test_generator.py")
    return
    
    # Generate test files for missing models
    if generate_test_files()missing_models):
        print()f"Successfully added {}}}}}}}}}}}}}}}len()missing_models)} new model families and generated test files")
        
        # List the newly generated files
        print()"\nNewly generated test files:")
        for model_id in missing_models:
            model_id_normalized = model_id.lower()).replace()'-', '_')
            print()f"test_hf_{}}}}}}}}}}}}}}}model_id_normalized}.py")
    else:
        print()"Failed to generate some test files")

if __name__ == "__main__":
    main())