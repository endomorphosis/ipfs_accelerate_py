#!/usr/bin/env python3

"""
Replace the problematic MODEL_REGISTRY in test_generator_fixed.py with a clean version.
"""

import os
import sys
import time
import json
import shutil

def replace_model_registry(file_path):
    """Replace the MODEL_REGISTRY with a clean, well-structured version."""
    print(f"Fixing {file_path}...")
    
    # Create backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Load the model registry data from huggingface_model_types.json
    registry_file = os.path.join(os.path.dirname(file_path), 'huggingface_model_types.json')
    registry_data = {}
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            print(f"Loaded registry data from {registry_file}")
        except Exception as e:
            print(f"Error loading registry file: {e}")
    
    # Define a clean MODEL_REGISTRY
    # This dictionary structure matches the original but fixes indentation and avoids duplication
    clean_registry = {
        "bert": {
            "family_name": "BERT",
            "description": "BERT-family masked language models",
            "default_model": registry_data.get("bert", {}).get("default_model", "bert-base-uncased"),
            "class": "BertForMaskedLM",
            "test_class": "TestBertModels",
            "module_name": "test_hf_bert",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "The quick brown fox jumps over the [MASK] dog.",
            },
        },
        "gpt2": {
            "family_name": "GPT-2",
            "description": "GPT-2 autoregressive language models",
            "default_model": registry_data.get("gpt2", {}).get("default_model", "gpt2"),
            "class": "GPT2LMHeadModel",
            "test_class": "TestGPT2Models",
            "module_name": "test_hf_gpt2",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "GPT-2 is a transformer model that",
            },
            "task_specific_args": {
                "text-generation": {
                    "max_length": 50,
                    "min_length": 20,
                },
            },
        },
        "t5": {
            "family_name": "T5",
            "description": "T5 encoder-decoder models",
            "default_model": registry_data.get("t5", {}).get("default_model", "t5-small"),
            "class": "T5ForConditionalGeneration",
            "test_class": "TestT5Models",
            "module_name": "test_hf_t5",
            "tasks": ['text2text-generation'],
            "inputs": {
                "text": "translate English to German: The house is wonderful.",
            },
            "task_specific_args": {
                "text2text-generation": {
                    "max_length": 50,
                },
            },
        },
        "vit": {
            "family_name": "ViT",
            "description": "Vision Transformer models",
            "default_model": registry_data.get("vit", {}).get("default_model", "google/vit-base-patch16-224"),
            "class": "ViTForImageClassification",
            "test_class": "TestVitModels",
            "module_name": "test_hf_vit",
            "tasks": ['image-classification'],
            "inputs": {
            },
        },
        "gpt-j": {
            "family_name": "GPT-J",
            "description": "GPT-J autoregressive language models",
            "default_model": registry_data.get("gpt-j", {}).get("default_model", "EleutherAI/gpt-j-6b"),
            "class": "GPTJForCausalLM",
            "test_class": "TestGPTJModels",
            "module_name": "test_hf_gpt_j",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "GPT-J is a transformer model that",
            },
            "task_specific_args": {
                "text-generation": {
                    "max_length": 50,
                },
            },
        },
        "gpt-neo": {
            "family_name": "GPT-Neo",
            "description": "GPT-Neo autoregressive language models",
            "default_model": registry_data.get("gpt-neo", {}).get("default_model", "EleutherAI/gpt-neo-1.3B"),
            "class": "GPTNeoForCausalLM",
            "test_class": "TestGPTNeoModels",
            "module_name": "test_hf_gpt_neo",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "GPT-Neo is a transformer model that",
            },
        },
        "xlm-roberta": {
            "family_name": "XLM-RoBERTa",
            "description": "XLM-RoBERTa masked language models for cross-lingual understanding",
            "default_model": registry_data.get("xlm-roberta", {}).get("default_model", "xlm-roberta-base"),
            "class": "XLMRobertaForMaskedLM",
            "test_class": "TestXLMRobertaModels",
            "module_name": "test_hf_xlm_roberta",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "XLM-RoBERTa is a <mask> language model.",
            },
        },
        "roberta": {
            "family_name": "RoBERTa",
            "description": "RoBERTa masked language models",
            "default_model": registry_data.get("roberta", {}).get("default_model", "roberta-base"),
            "class": "RobertaForMaskedLM",
            "test_class": "TestRobertaModels",
            "module_name": "test_hf_roberta",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "RoBERTa is a <mask> language model.",
            },
        },
        "distilbert": {
            "family_name": "DistilBERT",
            "description": "DistilBERT masked language models",
            "default_model": registry_data.get("distilbert", {}).get("default_model", "distilbert-base-uncased"),
            "class": "DistilBertForMaskedLM",
            "test_class": "TestDistilBertModels",
            "module_name": "test_hf_distilbert",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "DistilBERT is a <mask> language model.",
            },
        },
        "albert": {
            "family_name": "ALBERT",
            "description": "ALBERT (A Lite BERT) masked language models",
            "default_model": registry_data.get("albert", {}).get("default_model", "albert-base-v2"),
            "class": "AlbertForMaskedLM",
            "test_class": "TestAlbertModels",
            "module_name": "test_hf_albert",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "ALBERT is a <mask> language model.",
            },
        },
        "electra": {
            "family_name": "ELECTRA",
            "description": "ELECTRA discriminator models",
            "default_model": registry_data.get("electra", {}).get("default_model", "google/electra-small-discriminator"),
            "class": "ElectraForMaskedLM",
            "test_class": "TestElectraModels",
            "module_name": "test_hf_electra",
            "tasks": ['fill-mask'],
            "inputs": {
                "text": "ELECTRA is a <mask> language model.",
            },
        },
        "bart": {
            "family_name": "BART",
            "description": "BART sequence-to-sequence models",
            "default_model": registry_data.get("bart", {}).get("default_model", "facebook/bart-base"),
            "class": "BartForConditionalGeneration",
            "test_class": "TestBartModels",
            "module_name": "test_hf_bart",
            "tasks": ['summarization', 'translation'],
            "inputs": {
                "text": "BART is a denoising autoencoder for pretraining sequence-to-sequence models.",
            },
        },
        "mbart": {
            "family_name": "mBART",
            "description": "Multilingual BART sequence-to-sequence models",
            "default_model": registry_data.get("mbart", {}).get("default_model", "facebook/mbart-large-cc25"),
            "class": "MBartForConditionalGeneration",
            "test_class": "TestMBartModels",
            "module_name": "test_hf_mbart",
            "tasks": ['translation'],
            "inputs": {
                "text": "mBART is a multilingual sequence-to-sequence model.",
            },
        },
        "pegasus": {
            "family_name": "Pegasus",
            "description": "Pegasus summarization models",
            "default_model": registry_data.get("pegasus", {}).get("default_model", "google/pegasus-xsum"),
            "class": "PegasusForConditionalGeneration",
            "test_class": "TestPegasusModels",
            "module_name": "test_hf_pegasus",
            "tasks": ['summarization'],
            "inputs": {
                "text": "Pegasus is a model for abstractive summarization optimized for ROUGE.",
            },
        },
        "mt5": {
            "family_name": "mT5",
            "description": "Multilingual T5 models",
            "default_model": registry_data.get("mt5", {}).get("default_model", "google/mt5-small"),
            "class": "MT5ForConditionalGeneration",
            "test_class": "TestMT5Models",
            "module_name": "test_hf_mt5",
            "tasks": ['translation'],
            "inputs": {
                "text": "translate English to German: The house is wonderful.",
            },
        },
        "clip": {
            "family_name": "CLIP",
            "description": "Contrastive Language-Image Pre-training models",
            "default_model": registry_data.get("clip", {}).get("default_model", "openai/clip-vit-base-patch32"),
            "class": "CLIPModel",
            "test_class": "TestCLIPModels",
            "module_name": "test_hf_clip",
            "tasks": ['zero-shot-image-classification'],
            "inputs": {
            },
        },
        "blip": {
            "family_name": "BLIP",
            "description": "Bootstrapping Language-Image Pre-training models",
            "default_model": registry_data.get("blip", {}).get("default_model", "Salesforce/blip-image-captioning-base"),
            "class": "BlipForConditionalGeneration",
            "test_class": "TestBlipModels",
            "module_name": "test_hf_blip",
            "tasks": ['image-to-text'],
            "inputs": {
            },
        },
        "llava": {
            "family_name": "LLaVA",
            "description": "Large Language and Vision Assistant",
            "default_model": registry_data.get("llava", {}).get("default_model", "llava-hf/llava-1.5-7b-hf"),
            "class": "LlavaForConditionalGeneration",
            "test_class": "TestLlavaModels",
            "module_name": "test_hf_llava",
            "tasks": ['visual-question-answering'],
            "inputs": {
            },
        },
        "whisper": {
            "family_name": "Whisper",
            "description": "Speech recognition models",
            "default_model": registry_data.get("whisper", {}).get("default_model", "openai/whisper-base.en"),
            "class": "WhisperForConditionalGeneration",
            "test_class": "TestWhisperModels",
            "module_name": "test_hf_whisper",
            "tasks": ['automatic-speech-recognition'],
            "inputs": {
            },
        },
        "wav2vec2": {
            "family_name": "Wav2Vec2",
            "description": "Speech representation models",
            "default_model": registry_data.get("wav2vec2", {}).get("default_model", "facebook/wav2vec2-base"),
            "class": "Wav2Vec2ForCTC",
            "test_class": "TestWav2Vec2Models",
            "module_name": "test_hf_wav2vec2",
            "tasks": ['automatic-speech-recognition'],
            "inputs": {
            },
        },
        "hubert": {
            "family_name": "HuBERT",
            "description": "Hidden-Unit BERT speech models",
            "default_model": registry_data.get("hubert", {}).get("default_model", "facebook/hubert-base-ls960"),
            "class": "HubertForCTC",
            "test_class": "TestHubertModels",
            "module_name": "test_hf_hubert",
            "tasks": ['automatic-speech-recognition'],
            "inputs": {
            },
        },
        "llama": {
            "family_name": "LLaMA",
            "description": "Large Language Model Meta AI",
            "default_model": registry_data.get("llama", {}).get("default_model", "meta-llama/Llama-2-7b-hf"),
            "class": "LlamaForCausalLM",
            "test_class": "TestLlamaModels",
            "module_name": "test_hf_llama",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "LLaMA is a foundational language model that",
            },
        },
        "opt": {
            "family_name": "OPT",
            "description": "Open Pre-trained Transformer language models",
            "default_model": registry_data.get("opt", {}).get("default_model", "facebook/opt-125m"),
            "class": "OPTForCausalLM",
            "test_class": "TestOPTModels",
            "module_name": "test_hf_opt",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "OPT is an open-source language model that",
            },
        },
        "bloom": {
            "family_name": "BLOOM",
            "description": "BigScience Large Open-science Open-access Multilingual language models",
            "default_model": registry_data.get("bloom", {}).get("default_model", "bigscience/bloom-560m"),
            "class": "BloomForCausalLM",
            "test_class": "TestBloomModels",
            "module_name": "test_hf_bloom",
            "tasks": ['text-generation'],
            "inputs": {
                "text": "BLOOM is a multilingual language model that",
            },
        }
    }
    
    # Convert the clean registry to Python code
    registry_str = "MODEL_REGISTRY = {\n"
    for model_type, config in clean_registry.items():
        registry_str += f'    "{model_type}": {{\n'
        for key, value in config.items():
            if isinstance(value, str):
                registry_str += f'        "{key}": "{value}",\n'
            elif isinstance(value, list):
                # Handle lists like tasks
                items = ', '.join([f"'{item}'" for item in value])
                registry_str += f'        "{key}": [{items}],\n'
            elif isinstance(value, dict):
                # Handle nested dictionaries like inputs and task_specific_args
                registry_str += f'        "{key}": {{\n'
                for k, v in value.items():
                    if isinstance(v, str):
                        registry_str += f'            "{k}": "{v}",\n'
                    elif isinstance(v, dict):
                        # Handle doubly nested dictionaries
                        registry_str += f'            "{k}": {{\n'
                        for inner_k, inner_v in v.items():
                            registry_str += f'                "{inner_k}": {inner_v},\n'
                        registry_str += '            },\n'
                    else:
                        registry_str += f'            "{k}": {v},\n'
                registry_str += '        },\n'
        registry_str += '    },\n'
    registry_str += "}\n\n"
    
    # Read the original file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find MODEL_REGISTRY section beginning and CLASS_NAME_FIXES section
    registry_start = content.find("MODEL_REGISTRY = {")
    if registry_start == -1:
        print("Error: MODEL_REGISTRY not found")
        return False
        
    fixes_start = content.find("# Class name capitalization fixes", registry_start)
    if fixes_start == -1:
        fixes_start = content.find("CLASS_NAME_FIXES = {", registry_start)
    
    if fixes_start == -1:
        print("Error: Could not find section after MODEL_REGISTRY")
        return False
    
    # Replace the MODEL_REGISTRY with our clean version
    new_content = content[:registry_start] + registry_str + content[fixes_start:]
    
    # Write the content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    # Verify the file
    try:
        compile(new_content, file_path, 'exec')
        print("✅ Fixed file is syntactically valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
        print("Failed to fix file. Restoring backup.")
        shutil.copy2(backup_path, file_path)
        return False

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "test_generator_fixed.py"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return 1
    
    success = replace_model_registry(file_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())