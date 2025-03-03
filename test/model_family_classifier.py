"""
Model family classification module for the IPFS Accelerate framework.
This module analyzes model capabilities and classifies models into families.
"""

import os
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Model family definitions
MODEL_FAMILIES = {
    "embedding": {
        "description": "Text embedding and representation models like BERT",
        "keywords": ["bert", "embedding", "roberta", "distilbert", "camembert", "albert", "xlm", "sentence", "ernie", "luke"],
        "tasks": ["feature-extraction", "fill-mask", "sentence-similarity", "token-classification", "text-classification"],
        "methods": ["embed", "encode", "get_embedding", "get_sentence_embedding", "get_vector", "get_representation"]
    },
    "text_generation": {
        "description": "Text generation and language models like GPT, LLaMA, T5, etc.",
        "keywords": ["gpt", "t5", "llama", "opt", "bloom", "phi", "falcon", "mistral", "mamba", "transformer", "mixtral", "gemma"],
        "tasks": ["text-generation", "text2text-generation", "summarization", "translation", "question-answering"],
        "methods": ["generate", "predict", "generate_text", "complete", "predict_next", "chat", "summarize"]
    },
    "vision": {
        "description": "Vision and image processing models like ViT, CLIP (vision part), etc.",
        "keywords": ["vit", "vision", "resnet", "efficientnet", "swin", "deit", "beit", "convnext", "sam", "yolo", "detr"],
        "tasks": ["image-classification", "object-detection", "image-segmentation", "depth-estimation", "zero-shot-image-classification"],
        "methods": ["predict", "classify", "detect", "segment", "process_image"]
    },
    "audio": {
        "description": "Audio processing models like Whisper, Wav2Vec2, etc.",
        "keywords": ["whisper", "wav2vec", "hubert", "audio", "speech", "clap", "unispeech", "wavlm", "encodec", "speecht5"],
        "tasks": ["automatic-speech-recognition", "audio-classification", "audio-to-audio", "text-to-audio", "audio-frame-classification"],
        "methods": ["transcribe", "recognize", "speech_to_text", "process_audio", "audio_to_text"]
    },
    "multimodal": {
        "description": "Models that combine multiple modalities like LLaVA, BLIP, etc.",
        "keywords": ["llava", "blip", "clip", "flava", "vision-text", "flamingo", "multimodal", "idefics", "fuyu", "pali"],
        "tasks": ["image-to-text", "visual-question-answering", "document-question-answering", "text-to-image", "image-text-to-text"],
        "methods": ["vision_to_text", "image_text_generation", "visual_question_answering", "process_image_text", "describe_image"]
    }
}

# Specialized subfamilies within each family
MODEL_SUBFAMILIES = {
    "embedding": [
        {"name": "masked_lm", "keywords": ["bert", "roberta", "distilbert", "albert", "masked"], "tasks": ["fill-mask", "masked-lm"]},
        {"name": "sentence_transformer", "keywords": ["sentence", "sbert", "simcse"], "tasks": ["sentence-similarity"]},
        {"name": "token_classifier", "keywords": ["ner", "token", "sequence-classification"], "tasks": ["token-classification"]}
    ],
    "text_generation": [
        {"name": "causal_lm", "keywords": ["gpt", "llama", "falcon", "mixtral", "phi", "gemma", "causal"], "tasks": ["text-generation"]},
        {"name": "seq2seq", "keywords": ["t5", "bart", "mbart", "pegasus", "seq2seq"], "tasks": ["text2text-generation", "translation", "summarization"]},
        {"name": "chat_model", "keywords": ["chat", "instruct", "assistant", "dialog"], "tasks": ["text-generation"], "methods": ["chat"]}
    ],
    "vision": [
        {"name": "image_classifier", "keywords": ["classifier", "classification", "vit", "resnet"], "tasks": ["image-classification"]},
        {"name": "object_detector", "keywords": ["yolo", "detr", "detection", "detector"], "tasks": ["object-detection"]},
        {"name": "segmentation", "keywords": ["segmentation", "mask", "sam", "segment"], "tasks": ["image-segmentation"]},
        {"name": "depth_estimation", "keywords": ["depth", "estimation", "dpt", "glpn"], "tasks": ["depth-estimation"]}
    ],
    "audio": [
        {"name": "speech_recognition", "keywords": ["asr", "speech", "recognition", "whisper", "wav2vec"], "tasks": ["automatic-speech-recognition"]},
        {"name": "audio_classifier", "keywords": ["audio", "classification", "clap"], "tasks": ["audio-classification"]},
        {"name": "text_to_speech", "keywords": ["tts", "text-to-speech", "speech-synthesis"], "tasks": ["text-to-audio"]}
    ],
    "multimodal": [
        {"name": "vision_language", "keywords": ["llava", "blip", "vqa", "visual-question"], "tasks": ["visual-question-answering"]},
        {"name": "image_text_encoder", "keywords": ["clip", "siglip", "contrastive"], "tasks": ["zero-shot-image-classification"]},
        {"name": "document_qa", "keywords": ["document", "ocr", "layout", "donut"], "tasks": ["document-question-answering"]}
    ]
}

class ModelFamilyClassifier:
    """
    Analyzes model capabilities and classifies models into families.
    """
    
    def __init__(self, 
                 model_family_defs: Optional[Dict[str, Dict[str, Any]]] = None,
                 model_subfamily_defs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 model_db_path: Optional[str] = None):
        """
        Initialize model family classifier.
        
        Args:
            model_family_defs: Dictionary of model family definitions
            model_subfamily_defs: Dictionary of model subfamily definitions
            model_db_path: Path to model database JSON file
        """
        self.families = model_family_defs or MODEL_FAMILIES
        self.subfamilies = model_subfamily_defs or MODEL_SUBFAMILIES
        self.model_db = {}
        self.model_db_path = model_db_path
        
        # Load model database if provided
        if model_db_path and os.path.exists(model_db_path):
            self.load_model_db(model_db_path)
    
    def load_model_db(self, db_path: str):
        """Load model metadata database from JSON file"""
        try:
            with open(db_path, 'r') as f:
                self.model_db = json.load(f)
                logger.info(f"Loaded model database with {len(self.model_db)} entries from {db_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load model database from {db_path}: {str(e)}")
            self.model_db = {}
    
    def analyze_model_name(self, model_name: str) -> Dict[str, Any]:
        """
        Analyze the model name to extract family information.
        
        Args:
            model_name: The model name to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Normalize model name for analysis
        normalized_name = model_name.lower().replace('-', '_')
        
        # Check if model is in the database
        if model_name in self.model_db:
            # Return known information
            return {
                "model_name": model_name,
                "family": self.model_db[model_name].get("family"),
                "subfamily": self.model_db[model_name].get("subfamily"),
                "tasks": self.model_db[model_name].get("tasks", []),
                "source": "model_db"
            }
        
        # Initialize scores for each family
        family_scores = {family: 0 for family in self.families}
        
        # Score each family based on keyword matches in the name
        for family, info in self.families.items():
            for keyword in info["keywords"]:
                if keyword.lower() in normalized_name:
                    family_scores[family] += 1
        
        # Determine the best matching family
        best_family = max(family_scores.items(), key=lambda x: x[1]) if family_scores else None
        
        # If no clear match, return inconclusive
        if not best_family or best_family[1] == 0:
            return {
                "model_name": model_name,
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "name_analysis"
            }
        
        # Determine possible subfamily with improved matching
        subfamily = None
        subfamily_confidence = 0
        
        if best_family[0] in self.subfamilies:
            subfamily_scores = {}
            for subfam in self.subfamilies[best_family[0]]:
                score = 0
                keyword_count = len(subfam["keywords"])
                
                # Skip subfamilies with no keywords (avoid division by zero)
                if keyword_count == 0:
                    continue
                    
                # Look for exact and partial keyword matches
                for keyword in subfam["keywords"]:
                    if keyword.lower() in normalized_name:
                        score += 1
                    # Add partial match with lower weight
                    elif any(part in normalized_name for part in keyword.lower().split('-')):
                        score += 0.5
                        
                # Normalize by keyword count to handle different subfamily definitions fairly
                subfamily_scores[subfam["name"]] = score / keyword_count
            
            # Get best matching subfamily
            best_subfamily = max(subfamily_scores.items(), key=lambda x: x[1]) if subfamily_scores else None
            if best_subfamily and best_subfamily[1] > 0:
                subfamily = best_subfamily[0]
                subfamily_confidence = best_subfamily[1]  # Already normalized to 0-1 range
        
        return {
            "model_name": model_name,
            "family": best_family[0],
            "subfamily": subfamily,
            "confidence": best_family[1] / len(self.families[best_family[0]]["keywords"]),
            "subfamily_confidence": subfamily_confidence,
            "source": "name_analysis"
        }
    
    def analyze_model_tasks(self, tasks: List[str]) -> Dict[str, Any]:
        """
        Analyze the model tasks to determine family.
        
        Args:
            tasks: List of task names the model supports
            
        Returns:
            Dictionary with analysis results
        """
        if not tasks:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "task_analysis"
            }
        
        # Initialize scores for each family
        family_scores = {family: 0 for family in self.families}
        family_tasks_count = {family: len(info["tasks"]) for family, info in self.families.items()}
        
        # Normalize tasks (remove hyphens, lowercase)
        normalized_tasks = [task.lower().replace('-', '_') for task in tasks]
        
        # Score each family based on task matches with better matching logic
        for family, info in self.families.items():
            family_task_count = len(info["tasks"])
            if family_task_count == 0:
                continue
                
            # Normalize family tasks
            normalized_family_tasks = [t.lower().replace('-', '_') for t in info["tasks"]]
            
            # Count exact matches
            exact_matches = sum(1 for task in normalized_tasks if task in normalized_family_tasks)
            
            # Count partial matches (e.g., "text-generation" vs "text_generation")
            partial_matches = 0
            for task in normalized_tasks:
                # Skip tasks that were exact matches
                if task in normalized_family_tasks:
                    continue
                    
                # Check for partial matches using words in the task
                task_words = set(task.split('_'))
                for family_task in normalized_family_tasks:
                    family_task_words = set(family_task.split('_'))
                    common_words = task_words.intersection(family_task_words)
                    if common_words and len(common_words) >= min(2, len(task_words) / 2):
                        partial_matches += 0.5
                        break
            
            # Calculate score with both exact and partial matches
            family_scores[family] = exact_matches + partial_matches
        
        # Determine the best matching family
        best_family = max(family_scores.items(), key=lambda x: x[1]) if family_scores else (None, 0)
        
        # If no clear match, return inconclusive
        if not best_family or best_family[1] == 0:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "task_analysis"
            }
        
        # Calculate confidence, normalizing by the number of tasks in the family
        family_confidence = best_family[1] / max(1, family_tasks_count[best_family[0]])
        
        # Analyze for subfamily with improved matching
        subfamily = None
        subfamily_confidence = 0
        
        if best_family[0] in self.subfamilies:
            subfamily_scores = {}
            for subfam in self.subfamilies[best_family[0]]:
                if "tasks" not in subfam or not subfam["tasks"]:
                    continue
                
                # Normalize subfamily tasks
                normalized_subfam_tasks = [t.lower().replace('-', '_') for t in subfam["tasks"]]
                
                # Count exact matches
                exact_matches = sum(1 for task in normalized_tasks if task in normalized_subfam_tasks)
                
                # Count partial matches
                partial_matches = 0
                for task in normalized_tasks:
                    # Skip tasks that were exact matches
                    if task in normalized_subfam_tasks:
                        continue
                        
                    # Check for partial matches
                    task_words = set(task.split('_'))
                    for subfam_task in normalized_subfam_tasks:
                        subfam_task_words = set(subfam_task.split('_'))
                        common_words = task_words.intersection(subfam_task_words)
                        if common_words and len(common_words) >= min(2, len(task_words) / 2):
                            partial_matches += 0.5
                            break
                
                # Calculate score with normalization
                subfamily_scores[subfam["name"]] = (exact_matches + partial_matches) / len(subfam["tasks"])
            
            # Get best matching subfamily
            best_subfamily = max(subfamily_scores.items(), key=lambda x: x[1]) if subfamily_scores else None
            if best_subfamily and best_subfamily[1] > 0:
                subfamily = best_subfamily[0]
                subfamily_confidence = best_subfamily[1]  # Already normalized
        
        return {
            "family": best_family[0],
            "subfamily": subfamily,
            "confidence": family_confidence,
            "subfamily_confidence": subfamily_confidence,
            "source": "task_analysis"
        }
    
    def analyze_model_methods(self, methods: List[str]) -> Dict[str, Any]:
        """
        Analyze the model methods to determine family.
        
        Args:
            methods: List of method names implemented by the model
            
        Returns:
            Dictionary with analysis results
        """
        if not methods:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "method_analysis"
            }
        
        # Initialize scores for each family
        family_scores = {family: 0 for family in self.families}
        
        # Score each family based on method matches
        for family, info in self.families.items():
            for method in methods:
                for pattern in info.get("methods", []):
                    if pattern.lower() in method.lower():
                        family_scores[family] += 1
                        break
        
        # Determine the best matching family
        best_family = max(family_scores.items(), key=lambda x: x[1]) if family_scores else (None, 0)
        
        # If no clear match, return inconclusive
        if not best_family or best_family[1] == 0:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "method_analysis"
            }
        
        # Analyze for subfamily
        subfamily = None
        subfamily_confidence = 0
        
        if best_family[0] in self.subfamilies:
            subfamily_scores = {}
            for subfam in self.subfamilies[best_family[0]]:
                score = 0
                for method in methods:
                    for pattern in subfam.get("methods", []):
                        if pattern and pattern.lower() in method.lower():
                            score += 1
                            break
                subfamily_scores[subfam["name"]] = score
            
            # Get best matching subfamily
            best_subfamily = max(subfamily_scores.items(), key=lambda x: x[1]) if subfamily_scores else None
            if best_subfamily and best_subfamily[1] > 0:
                subfamily = best_subfamily[0]
                subfamily_confidence = best_subfamily[1] / max(1, len(methods))
        
        return {
            "family": best_family[0],
            "subfamily": subfamily,
            "confidence": best_family[1] / max(1, len(methods)),
            "subfamily_confidence": subfamily_confidence,
            "source": "method_analysis"
        }
    
    def analyze_model_class(self, class_name: str) -> Dict[str, Any]:
        """
        Analyze the model class name to determine family.
        
        Args:
            class_name: The model class name (e.g., "BertModel", "T5ForConditionalGeneration")
            
        Returns:
            Dictionary with analysis results
        """
        if not class_name:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "class_analysis"
            }
        
        # Normalize class name
        normalized_class = class_name.lower()
        
        # Initialize scores for each family
        family_scores = {family: 0 for family in self.families}
        
        # Score each family based on keyword matches in class name
        for family, info in self.families.items():
            for keyword in info["keywords"]:
                if keyword.lower() in normalized_class:
                    family_scores[family] += 1
        
        # Check for specific class patterns
        for_pattern = re.search(r'For(\w+)', class_name)
        if for_pattern:
            task_suffix = for_pattern.group(1)
            
            # Map common task suffixes to families
            task_family_map = {
                "ConditionalGeneration": "text_generation",
                "CausalLM": "text_generation",
                "MaskedLM": "embedding",
                "SequenceClassification": "embedding",
                "TokenClassification": "embedding",
                "QuestionAnswering": "text_generation",
                "ImageClassification": "vision",
                "ObjectDetection": "vision",
                "AudioClassification": "audio",
                "SpeechSeq2Seq": "audio",
                "VisionTextDualEncoder": "multimodal",
                "VisualQuestionAnswering": "multimodal"
            }
            
            if task_suffix in task_family_map:
                family_scores[task_family_map[task_suffix]] += 2  # Give extra weight to explicit task markers
        
        # Determine the best matching family
        best_family = max(family_scores.items(), key=lambda x: x[1]) if family_scores else (None, 0)
        
        # If no clear match, return inconclusive
        if not best_family or best_family[1] == 0:
            return {
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "class_analysis"
            }
        
        # Determine possible subfamily
        subfamily = None
        subfamily_confidence = 0
        
        if best_family[0] in self.subfamilies:
            subfamily_scores = {}
            for subfam in self.subfamilies[best_family[0]]:
                score = 0
                for keyword in subfam["keywords"]:
                    if keyword.lower() in normalized_class:
                        score += 1
                subfamily_scores[subfam["name"]] = score
            
            # Get best matching subfamily
            best_subfamily = max(subfamily_scores.items(), key=lambda x: x[1]) if subfamily_scores else None
            if best_subfamily and best_subfamily[1] > 0:
                subfamily = best_subfamily[0]
                subfamily_confidence = best_subfamily[1] / len(subfam["keywords"])
        
        return {
            "family": best_family[0],
            "subfamily": subfamily,
            "confidence": best_family[1] / len(self.families[best_family[0]]["keywords"]),
            "subfamily_confidence": subfamily_confidence,
            "source": "class_analysis"
        }
    
    def analyze_hardware_compatibility(self, hw_compatibility: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze hardware compatibility to infer model family.
        
        Args:
            hw_compatibility: Dictionary with hardware compatibility information
            
        Returns:
            Dictionary with analysis results
        """
        if not hw_compatibility:
            return {
                "family": None,
                "confidence": 0,
                "source": "hardware_analysis"
            }
        
        # Store all the compatibility hints and their confidence scores
        family_hints = []
        
        # 1. Check for hardware-specific incompatibilities that indicate model families
        
        # Models incompatible with MPS (Apple Silicon) are often multimodal or complex vision models
        mps_compatible = hw_compatibility.get("mps", {}).get("compatible", True)
        if not mps_compatible:
            family_hints.append({
                "family": "multimodal",
                "confidence": 0.5,
                "reason": "MPS incompatibility suggests multimodal model"
            })
            
            # Additional evidence: CUDA compatibility with high memory
            cuda_compatible = hw_compatibility.get("cuda", {}).get("compatible", False)
            if cuda_compatible:
                # If CUDA compatible but MPS incompatible, strengthen multimodal hypothesis
                family_hints[-1]["confidence"] = 0.6
                family_hints[-1]["reason"] += " (CUDA compatible, MPS incompatible)"
        
        # 2. Check memory requirements across different hardware
        memory_factors = {}
        
        # Analyze GPU memory requirements
        for hw_type in ["cuda", "rocm"]:
            memory_usage = hw_compatibility.get(hw_type, {}).get("memory_usage", {}).get("peak", 0)
            if memory_usage > 5000:  # More than 5GB suggests large LLM
                family_hints.append({
                    "family": "text_generation",
                    "confidence": 0.7,
                    "subfamily": "causal_lm",
                    "subfamily_confidence": 0.65,
                    "reason": f"Very high memory usage ({memory_usage/1024:.1f}GB) suggests large language model"
                })
                memory_factors["high"] = True
            elif memory_usage > 2000:  # 2-5GB suggests medium-sized generation model
                family_hints.append({
                    "family": "text_generation",
                    "confidence": 0.6,
                    "subfamily": "seq2seq",
                    "subfamily_confidence": 0.5,
                    "reason": f"High memory usage ({memory_usage/1024:.1f}GB) suggests text generation model"
                })
                memory_factors["medium"] = True
        
        # 3. Check hardware acceleration patterns
        
        # Models that work well with OpenVINO but not with WebNN are often vision models
        openvino_compatible = hw_compatibility.get("openvino", {}).get("compatible", False) 
        webnn_compatible = hw_compatibility.get("webnn", {}).get("compatible", False)
        
        if openvino_compatible and not webnn_compatible:
            family_hints.append({
                "family": "vision",
                "confidence": 0.5,
                "reason": "OpenVINO compatible but WebNN incompatible suggests vision model"
            })
        
        # 4. Audio models have specific hardware patterns
        rocm_compatible = hw_compatibility.get("rocm", {}).get("compatible", False)
        if not rocm_compatible and mps_compatible:
            # Audio models often work on MPS but not on ROCm
            family_hints.append({
                "family": "audio",
                "confidence": 0.4,
                "reason": "ROCm incompatible but MPS compatible suggests audio model"
            })
        
        # 5. Check for pattern of CUDA incompatibilities that suggest vision-language models
        if not hw_compatibility.get("cuda", {}).get("compatible", True) and not mps_compatible:
            family_hints.append({
                "family": "multimodal",
                "confidence": 0.6,
                "subfamily": "vision_language",
                "subfamily_confidence": 0.5,
                "reason": "CUDA and MPS incompatibility suggests complex vision-language model"
            })
        
        # If we have hints, return the one with highest confidence
        if family_hints:
            best_hint = max(family_hints, key=lambda x: x["confidence"])
            return {
                "family": best_hint["family"],
                "confidence": best_hint["confidence"],
                "subfamily": best_hint.get("subfamily"),
                "subfamily_confidence": best_hint.get("subfamily_confidence", 0),
                "source": "hardware_analysis",
                "reason": best_hint["reason"]
            }
        
        # No clear indicators
        return {
            "family": None,
            "confidence": 0,
            "source": "hardware_analysis"
        }
    
    def classify_model(self, 
                       model_name: str,
                       model_class: Optional[str] = None,
                       tasks: Optional[List[str]] = None,
                       methods: Optional[List[str]] = None,
                       hw_compatibility: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify a model into a family based on all available information.
        
        Args:
            model_name: The model name
            model_class: Optional model class name
            tasks: Optional list of tasks the model supports
            methods: Optional list of methods implemented by the model
            hw_compatibility: Optional hardware compatibility information
            
        Returns:
            Dictionary with classification results
        """
        # Collect results from various analyses
        analyses = []
        
        # Analyze model name
        name_analysis = self.analyze_model_name(model_name)
        analyses.append(name_analysis)
        
        # Analyze model class
        if model_class:
            class_analysis = self.analyze_model_class(model_class)
            analyses.append(class_analysis)
        
        # Analyze tasks
        if tasks:
            task_analysis = self.analyze_model_tasks(tasks)
            analyses.append(task_analysis)
        
        # Analyze methods
        if methods:
            method_analysis = self.analyze_model_methods(methods)
            analyses.append(method_analysis)
        
        # Analyze hardware compatibility
        if hw_compatibility:
            hw_analysis = self.analyze_hardware_compatibility(hw_compatibility)
            if hw_analysis["family"]:
                analyses.append(hw_analysis)
        
        # Filter for analyses with valid family assignments
        valid_analyses = [a for a in analyses if a.get("family")]
        
        if not valid_analyses:
            return {
                "model_name": model_name,
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "combined_analysis",
                "analyses": analyses
            }
        
        # Define analysis method weights for more accurate classification
        method_weights = {
            "name_analysis": 0.7,
            "class_analysis": 0.9,
            "task_analysis": 1.0,
            "method_analysis": 0.8,
            "hardware_analysis": 0.5
        }
        
        # Count family assignments across all analyses with weighting
        family_scores = defaultdict(float)
        family_confidences = defaultdict(list)
        subfamily_assignments = defaultdict(lambda: defaultdict(float))
        subfamily_confidences = defaultdict(lambda: defaultdict(list))
        
        for analysis in valid_analyses:
            source = analysis.get("source", "unknown")
            weight = method_weights.get(source, 0.5)
            
            family = analysis.get("family")
            confidence = analysis.get("confidence", 0)
            
            if family:
                # Add weighted score
                family_scores[family] += confidence * weight
                family_confidences[family].append(confidence)
            
            subfamily = analysis.get("subfamily")
            subfamily_confidence = analysis.get("subfamily_confidence", 0)
            
            if subfamily and family:
                # Add weighted score for subfamily
                subfamily_assignments[family][subfamily] += subfamily_confidence * weight
                subfamily_confidences[family][subfamily].append(subfamily_confidence)
        
        # Determine the most likely family by highest weighted score
        if not family_scores:
            return {
                "model_name": model_name,
                "family": None,
                "subfamily": None,
                "confidence": 0,
                "source": "combined_analysis",
                "analyses": analyses
            }
            
        best_family = max(family_scores.items(), key=lambda x: x[1])
        family_name = best_family[0]
        
        # Calculate combined confidence score - average of raw confidences
        raw_confidences = family_confidences[family_name]
        family_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0
        
        # Determine the most likely subfamily with improved approach
        subfamily_name = None
        subfamily_confidence = 0
        
        if family_name in subfamily_assignments and subfamily_assignments[family_name]:
            best_subfamily = max(subfamily_assignments[family_name].items(), key=lambda x: x[1])
            subfamily_name = best_subfamily[0]
            
            # Average raw subfamily confidences for the chosen subfamily
            raw_sub_confidences = subfamily_confidences[family_name][subfamily_name]
            if raw_sub_confidences:
                subfamily_confidence = sum(raw_sub_confidences) / len(raw_sub_confidences)
        
        # Create final classification result
        return {
            "model_name": model_name,
            "family": family_name,
            "subfamily": subfamily_name,
            "confidence": family_confidence,
            "subfamily_confidence": subfamily_confidence,
            "source": "combined_analysis",
            "analyses": analyses
        }
    
    def get_template_for_family(self, family: str, subfamily: Optional[str] = None) -> str:
        """
        Get the appropriate template filename for a given model family.
        
        Args:
            family: The model family
            subfamily: Optional subfamily for more specific template
            
        Returns:
            Template filename for the family
        """
        family_template_map = {
            "embedding": "hf_embedding_template.py",
            "text_generation": "hf_text_generation_template.py",
            "vision": "hf_vision_template.py",
            "audio": "hf_audio_template.py",
            "multimodal": "hf_multimodal_template.py"
        }
        
        if family in family_template_map:
            return family_template_map[family]
        
        # If family not recognized or None, use base template
        return "hf_template.py"
    
    def update_model_db(self, model_name: str, classification: Dict[str, Any]) -> None:
        """
        Update the model database with classification results.
        
        Args:
            model_name: The model name
            classification: Dictionary with classification results
        """
        if not classification.get("family"):
            return
        
        # Create or update model entry
        if model_name not in self.model_db:
            self.model_db[model_name] = {}
        
        # Update family information
        self.model_db[model_name]["family"] = classification["family"]
        
        if classification.get("subfamily"):
            self.model_db[model_name]["subfamily"] = classification["subfamily"]
        
        if classification.get("confidence"):
            self.model_db[model_name]["confidence"] = classification["confidence"]
        
        # Save updated database if path is set
        if self.model_db_path:
            try:
                with open(self.model_db_path, 'w') as f:
                    json.dump(self.model_db, f, indent=2)
                    logger.info(f"Updated model database with classification for {model_name}")
            except IOError as e:
                logger.warning(f"Failed to save model database: {str(e)}")

def classify_model(model_name: str, 
                  model_class: Optional[str] = None,
                  tasks: Optional[List[str]] = None,
                  methods: Optional[List[str]] = None,
                  hw_compatibility: Optional[Dict[str, Any]] = None,
                  model_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify a model into a family based on all available information.
    
    Args:
        model_name: The model name
        model_class: Optional model class name
        tasks: Optional list of tasks the model supports
        methods: Optional list of methods implemented by the model
        hw_compatibility: Optional hardware compatibility information
        model_db_path: Optional path to model database
        
    Returns:
        Dictionary with classification results
    """
    classifier = ModelFamilyClassifier(model_db_path=model_db_path)
    result = classifier.classify_model(
        model_name=model_name,
        model_class=model_class,
        tasks=tasks,
        methods=methods,
        hw_compatibility=hw_compatibility
    )
    
    # Update model database if path is provided
    if model_db_path:
        classifier.update_model_db(model_name, result)
    
    return result


if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Classify a model into a family')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--class', dest='model_class', type=str, help='Model class name')
    parser.add_argument('--tasks', type=str, help='Comma-separated list of tasks')
    parser.add_argument('--methods', type=str, help='Comma-separated list of methods')
    parser.add_argument('--db', type=str, help='Path to model database')
    args = parser.parse_args()
    
    # Parse tasks and methods if provided
    tasks = args.tasks.split(',') if args.tasks else None
    methods = args.methods.split(',') if args.methods else None
    
    # Classify model
    result = classify_model(
        model_name=args.model,
        model_class=args.model_class,
        tasks=tasks,
        methods=methods,
        model_db_path=args.db
    )
    
    # Print results
    print(f"\n=== Model Classification Results for {args.model} ===")
    print(f"Family: {result['family']}")
    print(f"Subfamily: {result.get('subfamily')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    if result.get('analyses'):
        print("\nDetailed Analysis:")
        for analysis in result['analyses']:
            if analysis.get('family'):
                source = analysis.get('source', 'unknown')
                confidence = analysis.get('confidence', 0)
                print(f"  - {source}: family={analysis['family']} (confidence={confidence:.2f})")
    
    # Print template recommendation
    classifier = ModelFamilyClassifier()
    template = classifier.get_template_for_family(result['family'], result.get('subfamily'))
    print(f"\nRecommended Template: {template}")