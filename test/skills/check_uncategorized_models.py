#!/usr/bin/env python3
"""
Script to check for any uncategorized models and look them up in the Transformers documentation.
"""

import os
import re
import glob
import subprocess
from collections import defaultdict

# Directory where test files are located
TEST_DIR = "/home/barberb/ipfs_accelerate_py/test/skills"
# Transformers docs directory
TRANSFORMERS_DOCS_DIR = "/home/barberb/ipfs_accelerate_py/test/transformers_docs_build"

def get_implemented_models():
    """Get a list of all implemented models from test files."""
    test_files = glob.glob(os.path.join(TEST_DIR, "test_hf_*.py"))
    
    # Filter out files in subdirectories
    test_files = [f for f in test_files if "/" not in os.path.relpath(f, TEST_DIR)]
    
    # Extract model names from file names
    model_names = []
    for test_file in test_files:
        base_name = os.path.basename(test_file)
        model_name = base_name.replace("test_hf_", "").replace(".py", "")
        model_names.append(model_name)
    
    return model_names

def categorize_models(models):
    """Categorize models by architecture type."""
    categories = defaultdict(list)
    
    # Dictionary mapping categories to their keyword patterns
    category_patterns = {
        "encoder_only": [
            # BERT family and related encoder-only models
            r"^bert", r"albert", r"camembert", r"canine", r"deberta", r"distilbert", 
            r"electra", r"ernie", r"flaubert", r"funnel", r"layoutlm", r"longformer", 
            r"mpnet", r"rembert", r"roberta", r"roformer", r"xlm", r"xlmroberta", r"xlnet",
            r"nezha", r"bigbird(?!_pegasus)", r"megatron_bert", r"mobilebert", r"luke", r"tapas",
            r"markuplm", r"squeezebert", r"convbert", r"data2vec_text", r"dpr", r"esm", r"qdqbert",
            r"roc_bert", r"bros", r"retribert", r"splinter", r"ibert", r"big_bird", r"lilt"
        ],
        "decoder_only": [
            # GPT family and related decoder-only models
            r"^gpt", r"bloom", r"llama", r"mistral", r"phi", r"falcon", r"mpt", r"neo", 
            r"neox", r"opt", r"gemma", r"codegen", r"stablelm", r"pythia", r"xglm", 
            r"codellama", r"olmo", r"transfo_xl", r"ctrl", r"mamba", r"rwkv", r"biogpt",
            r"starcoder", r"tinyllama", r"baichuan", r"blenderbot", r"qwen", r"open_llama",
            r"persimmon", r"openai_gpt", r"orca", r"xmoe", r"mixtral", r"cohere", r"command_r",
            r"claude3", r"glm", r"cm3", r"nemotron", r"dbrx", r"deepseek", r"deepseek_coder",
            r"deepseek_distil", r"granite", r"granitemoe", r"jamba", r"jetmoe", r"cpmant",
            r"deepseek_r1", r"deepseek_r1_distil", r"pixtral", r"moshi"
        ],
        "encoder_decoder": [
            # T5 family and related encoder-decoder models
            r"^t5", r"bart", r"pegasus", r"prophetnet", r"led", r"mbart", r"longt5", 
            r"bigbird_pegasus", r"nllb", r"pegasus_x", r"umt5", r"flan", r"m2m", r"plbart", 
            r"mt5", r"switch_transformers", r"m2m_100", r"fsmt", r"mvp", r"nat",
            r"encoder_decoder", r"nougat", r"rag", r"marian", r"trocr", r"udop"
        ],
        "vision": [
            # Vision models
            r"vit", r"swin", r"deit", r"resnet", r"convnext", r"beit", r"segformer", r"detr", 
            r"mask2former", r"yolos", r"sam", r"dinov2", r"mobilevit", r"cvt", r"levit", 
            r"swinv2", r"perceiver", r"poolformer", r"efficientnet", r"regnet", r"dpt",
            r"glpn", r"mobilenet", r"bit", r"van", r"swiftformer", r"pvt", r"convnextv2",
            r"data2vec_vision", r"focalnet", r"seggpt", r"upernet", r"vitdet", r"mlp_mixer",
            r"timm", r"hiera", r"dino", r"florence", r"donut", r"table_transformer",
            r"conditional_detr", r"depth_anything", r"efficientformer", r"deepseek_vision",
            r"deta", r"omnivore", r"maskformer", r"oneformer", r"superpoint", r"videomae",
            r"vqgan", r"yoso", r"zoedepth", r"owlv2", r"omdet_turbo"
        ],
        "multimodal": [
            # Multimodal models
            r"clip", r"blip", r"llava", r"flava", r"git", r"idefics", r"paligemma", 
            r"imagebind", r"vilt", r"chinese_clip", r"instructblip", r"owlvit", r"siglip", 
            r"groupvit", r"xclip", r"align", r"altclip", r"bridgetower", r"blip_2",
            r"kosmos", r"flamingo", r"pix2struct", r"vision_encoder_decoder", 
            r"vision_text_dual_encoder", r"vision_t5", r"fuyu", r"tvlt", r"visual_bert",
            r"tvp", r"vipllava", r"ulip", r"lxmert", r"imagegpt", r"clvp", r"cogvlm2",
            r"mgp_str", r"chameleon"
        ],
        "audio": [
            # Audio models
            r"whisper", r"wav2vec2", r"hubert", r"sew", r"unispeech", r"clap", r"musicgen", 
            r"encodec", r"wavlm", r"data2vec_audio", r"audioldm2", r"speecht5", r"bark",
            r"mctct", r"univnet", r"vits", r"audio_spectrogram", r"speech_to_text",
            r"pop2piano", r"seamless_m4t", r"fastspeech", r"speech_encoder_decoder",
            r"timesformer", r"jukebox", r"mimi"
        ],
        "language_modeling": [
            # General language modeling
            r"mra", r"reformer", r"mega", r"realm", r"informer", r"autoformer", r"time_series",
            r"patchtst", r"xmod", r"fnet", r"nystromformer", r"zamba"
        ],
        "structured_data": [
            # Structured data models
            r"graphormer", r"graphsage", r"trajectory", r"decision", r"patchtsmixer", r"dac"
        ],
        "utility": [
            # Utility models or special cases
            r"__help", r"__list_only", r"__model", r"\\"
        ]
    }
    
    # Process each model
    uncategorized_models = []
    
    for model in models:
        model_lower = model.lower().replace("-", "_")
        assigned = False
        
        # Try to match against each category
        for category, patterns in category_patterns.items():
            if assigned:
                break
                
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    categories[category].append(model)
                    assigned = True
                    break
        
        # If not assigned to any specific category
        if not assigned:
            uncategorized_models.append(model)
            categories["uncategorized"].append(model)
    
    return categories, uncategorized_models

def search_transformers_docs(model_name):
    """Search for model information in the Transformers documentation."""
    try:
        result = subprocess.run(
            ["find", TRANSFORMERS_DOCS_DIR, "-name", f"*{model_name}*", "-type", "f"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []

def get_model_details_from_docs(model_name):
    """Extract model details from the Transformers documentation."""
    doc_files = search_transformers_docs(model_name)
    
    if not doc_files or len(doc_files) == 0 or doc_files[0] == '':
        return {
            "model_name": model_name,
            "architecture_type": "Unknown",
            "tasks": [],
            "doc_files": []
        }
    
    # Try to determine architecture type and tasks based on file locations
    architecture_type = "Unknown"
    tasks = []
    
    for file in doc_files:
        file_lower = file.lower()
        
        # Check for architecture type clues in the file path
        if "bert" in file_lower or "roberta" in file_lower or "albert" in file_lower or "encoder-only" in file_lower:
            architecture_type = "encoder-only"
        elif "gpt" in file_lower or "llama" in file_lower or "bloom" in file_lower or "decoder-only" in file_lower:
            architecture_type = "decoder-only"
        elif "t5" in file_lower or "bart" in file_lower or "encoder-decoder" in file_lower:
            architecture_type = "encoder-decoder"
        elif "vit" in file_lower or "resnet" in file_lower or "vision" in file_lower:
            architecture_type = "vision"
        elif "clip" in file_lower or "blip" in file_lower or "multimodal" in file_lower:
            architecture_type = "multimodal"
        elif "wav2vec" in file_lower or "whisper" in file_lower or "audio" in file_lower:
            architecture_type = "audio"
        
        # Check for task clues in the file path
        if "classification" in file_lower:
            tasks.append("classification")
        if "generation" in file_lower:
            tasks.append("text-generation")
        if "fill-mask" in file_lower:
            tasks.append("fill-mask")
        if "translation" in file_lower:
            tasks.append("translation")
        if "summarization" in file_lower:
            tasks.append("summarization")
        if "question-answering" in file_lower:
            tasks.append("question-answering")
        if "speech" in file_lower:
            tasks.append("speech")
        if "image" in file_lower:
            tasks.append("image")
    
    return {
        "model_name": model_name,
        "architecture_type": architecture_type,
        "tasks": list(set(tasks)),
        "doc_files": doc_files
    }

def manually_classify_remaining_models(uncategorized_models):
    """Manually classify models that couldn't be categorized automatically."""
    # These classifications are based on domain knowledge and transformers documentation
    manual_classifications = {
        "omdet_turbo": {"architecture_type": "vision", "description": "Omnivore Detection Turbo model - a vision model for object detection."}
    }
    
    classified_models = []
    still_uncategorized = []
    
    for model in uncategorized_models:
        if model in manual_classifications:
            classified_models.append({
                "model_name": model,
                "architecture_type": manual_classifications[model]["architecture_type"],
                "description": manual_classifications[model]["description"],
                "source": "manual"
            })
        else:
            # Try to find in the transformers docs
            model_details = get_model_details_from_docs(model)
            if model_details["architecture_type"] != "Unknown":
                classified_models.append({
                    "model_name": model,
                    "architecture_type": model_details["architecture_type"],
                    "tasks": model_details["tasks"],
                    "doc_files": model_details["doc_files"],
                    "source": "transformers_docs"
                })
            else:
                still_uncategorized.append(model)
    
    return classified_models, still_uncategorized

def main():
    """Main function to check for uncategorized models and classify them."""
    print("Checking for uncategorized models...")
    
    # Get implemented models
    all_models = get_implemented_models()
    print(f"Found {len(all_models)} implemented models")
    
    # Categorize models
    categories, uncategorized_models = categorize_models(all_models)
    
    if not uncategorized_models:
        print("All models are already categorized!")
        return
    
    print(f"\nFound {len(uncategorized_models)} uncategorized models:")
    for model in uncategorized_models:
        print(f"- {model}")
    
    # Try to classify uncategorized models
    print("\nAttempting to classify uncategorized models...")
    classified_models, still_uncategorized = manually_classify_remaining_models(uncategorized_models)
    
    if classified_models:
        print(f"\nSuccessfully classified {len(classified_models)} models:")
        for model in classified_models:
            print(f"- {model['model_name']}: {model['architecture_type']} (Source: {model['source']})")
            if "doc_files" in model and model["doc_files"]:
                print(f"  Documentation files:")
                for doc_file in model["doc_files"][:3]:  # Show max 3 files
                    print(f"  - {doc_file}")
                if len(model["doc_files"]) > 3:
                    print(f"  - ... and {len(model['doc_files']) - 3} more files")
    
    if still_uncategorized:
        print(f"\n{len(still_uncategorized)} models still uncategorized:")
        for model in still_uncategorized:
            print(f"- {model}")
    else:
        print("\nAll models have been successfully categorized!")

if __name__ == "__main__":
    main()