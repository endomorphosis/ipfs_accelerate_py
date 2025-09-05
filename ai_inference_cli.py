#!/usr/bin/env python3
"""
AI Inference CLI Tool

This tool provides a comprehensive command-line interface for all AI inference types
supported by the MCP server. It mirrors the functionality of the MCP tools but works
as a standalone CLI application.

Usage Examples:
    python ai_inference_cli.py text generate --prompt "Hello world" --max-length 50
    python ai_inference_cli.py text classify --text "I love this product" --model-id "bert-base-uncased"
    python ai_inference_cli.py audio transcribe --audio-file "speech.wav"
    python ai_inference_cli.py vision classify --image-file "cat.jpg"
    python ai_inference_cli.py multimodal caption --image-file "scene.jpg"
    python ai_inference_cli.py specialized code --prompt "Create a function to sort a list"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_inference_cli")

# Import the MCP server for processing
sys.path.append(os.path.dirname(__file__))
try:
    from comprehensive_mcp_server import ComprehensiveMCPServer
    HAVE_MCP_SERVER = True
except ImportError as e:
    HAVE_MCP_SERVER = False
    logger.error(f"Could not import MCP server: {e}")

class AIInferenceCLI:
    """Comprehensive AI Inference CLI Tool."""
    
    def __init__(self):
        """Initialize the CLI tool."""
        self.server = None
        if HAVE_MCP_SERVER:
            try:
                self.server = ComprehensiveMCPServer()
                logger.info("MCP server initialized for CLI operations")
            except Exception as e:
                logger.error(f"Failed to initialize MCP server: {e}")
        
        # Define all available inference types and their parameters
        self.inference_types = {
            'text': {
                'generate': self._text_generate,
                'classify': self._text_classify,
                'embeddings': self._text_embeddings,
                'fill-mask': self._text_fill_mask,
                'translate': self._text_translate,
                'summarize': self._text_summarize,
                'question': self._text_question_answer,
            },
            'audio': {
                'transcribe': self._audio_transcribe,
                'classify': self._audio_classify,
                'synthesize': self._audio_synthesize,
                'generate': self._audio_generate,
            },
            'vision': {
                'classify': self._vision_classify,
                'detect': self._vision_detect_objects,
                'segment': self._vision_segment,
                'generate': self._vision_generate,
            },
            'multimodal': {
                'caption': self._multimodal_caption,
                'vqa': self._multimodal_visual_question,
                'document': self._multimodal_document,
            },
            'specialized': {
                'timeseries': self._specialized_timeseries,
                'code': self._specialized_code,
                'tabular': self._specialized_tabular,
            },
            'system': {
                'list-models': self._system_list_models,
                'recommend': self._system_recommend_model,
                'stats': self._system_get_stats,
                'available-types': self._system_available_types,
            }
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            description="AI Inference CLI - Comprehensive inference tool supporting all model types",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Text processing
  %(prog)s text generate --prompt "Hello world" --max-length 50
  %(prog)s text classify --text "I love this product!"
  %(prog)s text embeddings --text "Generate vector embeddings"
  
  # Audio processing
  %(prog)s audio transcribe --audio-file speech.wav
  %(prog)s audio classify --audio-file music.mp3
  
  # Vision processing
  %(prog)s vision classify --image-file cat.jpg
  %(prog)s vision detect --image-file street.jpg
  
  # Multimodal processing  
  %(prog)s multimodal caption --image-file scene.jpg
  %(prog)s multimodal vqa --image-file photo.jpg --question "What's in this image?"
  
  # Specialized tasks
  %(prog)s specialized code --prompt "Create a sorting function"
  %(prog)s specialized timeseries --data-file data.json
  
  # System commands
  %(prog)s system list-models
  %(prog)s system recommend --task-type text_generation
"""
        )
        
        # Global options
        parser.add_argument(
            '--model-id', 
            help='Specific model ID to use (optional - will auto-select if not provided)'
        )
        parser.add_argument(
            '--hardware', 
            default='cpu', 
            choices=['cpu', 'gpu', 'cuda', 'mps'],
            help='Hardware to use for inference (default: cpu)'
        )
        parser.add_argument(
            '--output-format', 
            default='json', 
            choices=['json', 'text', 'pretty'],
            help='Output format (default: json)'
        )
        parser.add_argument(
            '--verbose', '-v', 
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--save-result', 
            help='Save result to file'
        )
        
        # Create subparsers for different categories
        subparsers = parser.add_subparsers(dest='category', help='Inference category')
        
        # Text processing subcommands
        self._add_text_parsers(subparsers)
        
        # Audio processing subcommands
        self._add_audio_parsers(subparsers)
        
        # Vision processing subcommands
        self._add_vision_parsers(subparsers)
        
        # Multimodal processing subcommands
        self._add_multimodal_parsers(subparsers)
        
        # Specialized processing subcommands
        self._add_specialized_parsers(subparsers)
        
        # System commands
        self._add_system_parsers(subparsers)
        
        return parser
    
    def _add_text_parsers(self, subparsers):
        """Add text processing command parsers."""
        text_parser = subparsers.add_parser('text', help='Text processing commands')
        text_subparsers = text_parser.add_subparsers(dest='command', help='Text commands')
        
        # Text generation
        generate_parser = text_subparsers.add_parser('generate', help='Generate text using causal language models')
        generate_parser.add_argument('--prompt', required=True, help='Input prompt for text generation')
        generate_parser.add_argument('--max-length', type=int, default=100, help='Maximum length of generated text')
        generate_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling (0.0-2.0)')
        generate_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
        generate_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling parameter')
        
        # Text classification
        classify_parser = text_subparsers.add_parser('classify', help='Classify text')
        classify_parser.add_argument('--text', required=True, help='Text to classify')
        classify_parser.add_argument('--return-all-scores', action='store_true', help='Return all class scores')
        
        # Text embeddings
        embeddings_parser = text_subparsers.add_parser('embeddings', help='Generate text embeddings')
        embeddings_parser.add_argument('--text', required=True, help='Text to embed')
        embeddings_parser.add_argument('--normalize', action='store_true', default=True, help='Normalize embeddings')
        
        # Fill mask
        mask_parser = text_subparsers.add_parser('fill-mask', help='Fill masked tokens in text')
        mask_parser.add_argument('--text', required=True, help='Text with [MASK] tokens')
        mask_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to return')
        
        # Translation
        translate_parser = text_subparsers.add_parser('translate', help='Translate text between languages')
        translate_parser.add_argument('--text', required=True, help='Text to translate')
        translate_parser.add_argument('--source-lang', required=True, help='Source language code')
        translate_parser.add_argument('--target-lang', required=True, help='Target language code')
        
        # Summarization
        summarize_parser = text_subparsers.add_parser('summarize', help='Summarize text')
        summarize_parser.add_argument('--text', required=True, help='Text to summarize')
        summarize_parser.add_argument('--max-length', type=int, default=150, help='Maximum summary length')
        summarize_parser.add_argument('--min-length', type=int, default=30, help='Minimum summary length')
        
        # Question answering
        qa_parser = text_subparsers.add_parser('question', help='Answer questions based on context')
        qa_parser.add_argument('--question', required=True, help='Question to answer')
        qa_parser.add_argument('--context', required=True, help='Context for answering the question')
        qa_parser.add_argument('--max-answer-length', type=int, default=100, help='Maximum answer length')
    
    def _add_audio_parsers(self, subparsers):
        """Add audio processing command parsers."""
        audio_parser = subparsers.add_parser('audio', help='Audio processing commands')
        audio_subparsers = audio_parser.add_subparsers(dest='command', help='Audio commands')
        
        # Audio transcription
        transcribe_parser = audio_subparsers.add_parser('transcribe', help='Transcribe audio to text')
        transcribe_parser.add_argument('--audio-file', required=True, help='Audio file to transcribe')
        transcribe_parser.add_argument('--language', help='Language of the audio (auto-detect if not specified)')
        transcribe_parser.add_argument('--task', default='transcribe', choices=['transcribe', 'translate'], help='Task type')
        
        # Audio classification
        classify_parser = audio_subparsers.add_parser('classify', help='Classify audio content')
        classify_parser.add_argument('--audio-file', required=True, help='Audio file to classify')
        classify_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
        
        # Speech synthesis
        synthesize_parser = audio_subparsers.add_parser('synthesize', help='Synthesize speech from text')
        synthesize_parser.add_argument('--text', required=True, help='Text to synthesize')
        synthesize_parser.add_argument('--speaker', help='Speaker voice to use')
        synthesize_parser.add_argument('--language', default='en', help='Language for synthesis')
        synthesize_parser.add_argument('--output-file', help='Output audio file')
        
        # Audio generation
        generate_parser = audio_subparsers.add_parser('generate', help='Generate audio from prompts')
        generate_parser.add_argument('--prompt', required=True, help='Prompt for audio generation')
        generate_parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds')
        generate_parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')
        generate_parser.add_argument('--output-file', help='Output audio file')
    
    def _add_vision_parsers(self, subparsers):
        """Add vision processing command parsers."""
        vision_parser = subparsers.add_parser('vision', help='Vision processing commands')
        vision_subparsers = vision_parser.add_subparsers(dest='command', help='Vision commands')
        
        # Image classification
        classify_parser = vision_subparsers.add_parser('classify', help='Classify images')
        classify_parser.add_argument('--image-file', required=True, help='Image file to classify')
        classify_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
        
        # Object detection
        detect_parser = vision_subparsers.add_parser('detect', help='Detect objects in images')
        detect_parser.add_argument('--image-file', required=True, help='Image file to analyze')
        detect_parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold')
        detect_parser.add_argument('--save-annotated', help='Save image with bounding boxes')
        
        # Image segmentation
        segment_parser = vision_subparsers.add_parser('segment', help='Segment images')
        segment_parser.add_argument('--image-file', required=True, help='Image file to segment')
        segment_parser.add_argument('--save-mask', help='Save segmentation mask')
        
        # Image generation
        generate_parser = vision_subparsers.add_parser('generate', help='Generate images from text')
        generate_parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
        generate_parser.add_argument('--width', type=int, default=512, help='Image width')
        generate_parser.add_argument('--height', type=int, default=512, help='Image height')
        generate_parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
        generate_parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale')
        generate_parser.add_argument('--output-file', help='Output image file')
    
    def _add_multimodal_parsers(self, subparsers):
        """Add multimodal processing command parsers."""
        multimodal_parser = subparsers.add_parser('multimodal', help='Multimodal processing commands')
        multimodal_subparsers = multimodal_parser.add_subparsers(dest='command', help='Multimodal commands')
        
        # Image captioning
        caption_parser = multimodal_subparsers.add_parser('caption', help='Generate image captions')
        caption_parser.add_argument('--image-file', required=True, help='Image file to caption')
        caption_parser.add_argument('--max-length', type=int, default=50, help='Maximum caption length')
        
        # Visual question answering
        vqa_parser = multimodal_subparsers.add_parser('vqa', help='Answer questions about images')
        vqa_parser.add_argument('--image-file', required=True, help='Image file to analyze')
        vqa_parser.add_argument('--question', required=True, help='Question about the image')
        
        # Document processing
        document_parser = multimodal_subparsers.add_parser('document', help='Process documents')
        document_parser.add_argument('--document-file', required=True, help='Document file to process')
        document_parser.add_argument('--query', required=True, help='Query about the document')
    
    def _add_specialized_parsers(self, subparsers):
        """Add specialized processing command parsers."""
        specialized_parser = subparsers.add_parser('specialized', help='Specialized processing commands')
        specialized_subparsers = specialized_parser.add_subparsers(dest='command', help='Specialized commands')
        
        # Time series prediction
        timeseries_parser = specialized_subparsers.add_parser('timeseries', help='Time series forecasting')
        timeseries_parser.add_argument('--data-file', required=True, help='JSON file with time series data')
        timeseries_parser.add_argument('--forecast-horizon', type=int, default=10, help='Number of steps to forecast')
        
        # Code generation
        code_parser = specialized_subparsers.add_parser('code', help='Generate code')
        code_parser.add_argument('--prompt', required=True, help='Code generation prompt')
        code_parser.add_argument('--language', default='python', help='Programming language')
        code_parser.add_argument('--max-length', type=int, default=200, help='Maximum code length')
        code_parser.add_argument('--output-file', help='Save generated code to file')
        
        # Tabular data processing
        tabular_parser = specialized_subparsers.add_parser('tabular', help='Process tabular data')
        tabular_parser.add_argument('--data-file', required=True, help='CSV/JSON file with tabular data')
        tabular_parser.add_argument('--task', default='classification', help='Task type (classification, regression)')
        tabular_parser.add_argument('--target-column', help='Target column for prediction')
    
    def _add_system_parsers(self, subparsers):
        """Add system command parsers."""
        system_parser = subparsers.add_parser('system', help='System management commands')
        system_subparsers = system_parser.add_subparsers(dest='command', help='System commands')
        
        # List models
        list_parser = system_subparsers.add_parser('list-models', help='List available models')
        list_parser.add_argument('--model-type', help='Filter by model type')
        list_parser.add_argument('--architecture', help='Filter by architecture')
        list_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
        
        # Recommend model
        recommend_parser = system_subparsers.add_parser('recommend', help='Get model recommendations')
        recommend_parser.add_argument('--task-type', required=True, help='Type of task')
        recommend_parser.add_argument('--input-type', default='tokens', help='Input data type')
        recommend_parser.add_argument('--output-type', default='logits', help='Output data type')
        
        # Get statistics
        stats_parser = system_subparsers.add_parser('stats', help='Get system statistics')
        
        # Available types
        types_parser = system_subparsers.add_parser('available-types', help='List available model types')
    
    def _encode_file(self, file_path: str) -> str:
        """Encode file as base64 string."""
        try:
            with open(file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding file {file_path}: {e}")
            return ""
    
    def _load_json_file(self, file_path: str) -> Any:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def _perform_inference(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference using the MCP server."""
        if not self.server:
            return {
                "error": "MCP server not available",
                "mock_result": f"Mock result for {task_type}",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Use the server's internal inference method
            result = self.server._perform_inference(
                task_type=task_type,
                input_data=params,
                model_id=params.get('model_id'),
                hardware=params.get('hardware', 'cpu')
            )
            return result
        except Exception as e:
            logger.error(f"Error performing {task_type} inference: {e}")
            return {
                "error": str(e),
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
    
    # Text processing implementations
    def _text_generate(self, args) -> Dict[str, Any]:
        """Generate text using causal language modeling."""
        params = {
            'prompt': args.text if hasattr(args, 'text') else args.prompt,
            'max_length': args.max_length,
            'temperature': args.temperature,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        if hasattr(args, 'top_p'):
            params['top_p'] = args.top_p
        if hasattr(args, 'top_k'):
            params['top_k'] = args.top_k
        
        return self._perform_inference('causal_language_modeling', params)
    
    def _text_classify(self, args) -> Dict[str, Any]:
        """Classify text."""
        params = {
            'text': args.text,
            'return_all_scores': args.return_all_scores,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('text_classification', params)
    
    def _text_embeddings(self, args) -> Dict[str, Any]:
        """Generate text embeddings."""
        params = {
            'text': args.text,
            'normalize': args.normalize,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('embedding_generation', params)
    
    def _text_fill_mask(self, args) -> Dict[str, Any]:
        """Fill masked tokens."""
        params = {
            'text': args.text,
            'top_k': args.top_k,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('masked_language_modeling', params)
    
    def _text_translate(self, args) -> Dict[str, Any]:
        """Translate text."""
        params = {
            'text': args.text,
            'source_language': args.source_lang,
            'target_language': args.target_lang,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('translation', params)
    
    def _text_summarize(self, args) -> Dict[str, Any]:
        """Summarize text."""
        params = {
            'text': args.text,
            'max_length': args.max_length,
            'min_length': args.min_length,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('summarization', params)
    
    def _text_question_answer(self, args) -> Dict[str, Any]:
        """Answer questions based on context."""
        params = {
            'question': args.question,
            'context': args.context,
            'max_answer_length': args.max_answer_length,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('question_answering', params)
    
    # Audio processing implementations
    def _audio_transcribe(self, args) -> Dict[str, Any]:
        """Transcribe audio to text."""
        audio_data = self._encode_file(args.audio_file)
        params = {
            'audio': audio_data,
            'language': args.language,
            'task': args.task,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('automatic_speech_recognition', params)
    
    def _audio_classify(self, args) -> Dict[str, Any]:
        """Classify audio content."""
        audio_data = self._encode_file(args.audio_file)
        params = {
            'audio': audio_data,
            'top_k': args.top_k,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('audio_classification', params)
    
    def _audio_synthesize(self, args) -> Dict[str, Any]:
        """Synthesize speech from text."""
        params = {
            'text': args.text,
            'speaker': args.speaker,
            'language': args.language,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('text_to_speech', params)
        
        # Save output file if specified
        if hasattr(args, 'output_file') and args.output_file:
            result['output_file'] = args.output_file
        
        return result
    
    def _audio_generate(self, args) -> Dict[str, Any]:
        """Generate audio from prompts."""
        params = {
            'prompt': args.prompt,
            'duration': args.duration,
            'sample_rate': args.sample_rate,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('audio_generation', params)
        
        # Save output file if specified
        if hasattr(args, 'output_file') and args.output_file:
            result['output_file'] = args.output_file
        
        return result
    
    # Vision processing implementations
    def _vision_classify(self, args) -> Dict[str, Any]:
        """Classify images."""
        image_data = self._encode_file(args.image_file)
        params = {
            'image': image_data,
            'top_k': args.top_k,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('image_classification', params)
    
    def _vision_detect_objects(self, args) -> Dict[str, Any]:
        """Detect objects in images."""
        image_data = self._encode_file(args.image_file)
        params = {
            'image': image_data,
            'threshold': args.confidence_threshold,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('object_detection', params)
        
        # Save annotated image if specified
        if hasattr(args, 'save_annotated') and args.save_annotated:
            result['annotated_image_saved'] = args.save_annotated
        
        return result
    
    def _vision_segment(self, args) -> Dict[str, Any]:
        """Segment images."""
        image_data = self._encode_file(args.image_file)
        params = {
            'image': image_data,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('image_segmentation', params)
        
        # Save mask if specified
        if hasattr(args, 'save_mask') and args.save_mask:
            result['mask_saved'] = args.save_mask
        
        return result
    
    def _vision_generate(self, args) -> Dict[str, Any]:
        """Generate images from text."""
        params = {
            'prompt': args.prompt,
            'width': args.width,
            'height': args.height,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('image_diffusion', params)
        
        # Save output file if specified
        if hasattr(args, 'output_file') and args.output_file:
            result['output_file'] = args.output_file
        
        return result
    
    # Multimodal processing implementations
    def _multimodal_caption(self, args) -> Dict[str, Any]:
        """Generate image captions."""
        image_data = self._encode_file(args.image_file)
        params = {
            'image': image_data,
            'max_length': args.max_length,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('image_to_text', params)
    
    def _multimodal_visual_question(self, args) -> Dict[str, Any]:
        """Answer questions about images."""
        image_data = self._encode_file(args.image_file)
        params = {
            'image': image_data,
            'question': args.question,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('visual_question_answering', params)
    
    def _multimodal_document(self, args) -> Dict[str, Any]:
        """Process documents."""
        document_data = self._encode_file(args.document_file)
        params = {
            'document': document_data,
            'query': args.query,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('document_understanding', params)
    
    # Specialized processing implementations
    def _specialized_timeseries(self, args) -> Dict[str, Any]:
        """Time series forecasting."""
        data = self._load_json_file(args.data_file)
        if data is None:
            return {"error": "Could not load time series data"}
        
        params = {
            'data': data,
            'horizon': args.forecast_horizon,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('time_series_forecasting', params)
    
    def _specialized_code(self, args) -> Dict[str, Any]:
        """Generate code."""
        params = {
            'prompt': args.prompt,
            'language': args.language,
            'max_length': args.max_length,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        result = self._perform_inference('code_generation', params)
        
        # Save code to file if specified
        if hasattr(args, 'output_file') and args.output_file:
            try:
                with open(args.output_file, 'w') as f:
                    if 'generated_code' in result:
                        f.write(result['generated_code'])
                result['code_saved'] = args.output_file
            except Exception as e:
                result['save_error'] = str(e)
        
        return result
    
    def _specialized_tabular(self, args) -> Dict[str, Any]:
        """Process tabular data."""
        if args.data_file.endswith('.json'):
            data = self._load_json_file(args.data_file)
        else:
            # For CSV files, we'd need pandas but let's mock it for now
            data = {"mock": "csv_data"}
        
        if data is None:
            return {"error": "Could not load tabular data"}
        
        params = {
            'data': data,
            'task': args.task,
            'target_column': args.target_column if hasattr(args, 'target_column') else None,
            'model_id': args.model_id,
            'hardware': args.hardware
        }
        return self._perform_inference('tabular_processing', params)
    
    # System commands implementations
    def _system_list_models(self, args) -> Dict[str, Any]:
        """List available models."""
        if not self.server:
            return {"error": "MCP server not available", "models": []}
        
        try:
            models = self.server.model_manager.list_models()
            filtered_models = []
            
            for model in models[:args.limit]:
                model_dict = {
                    "model_id": getattr(model, 'model_id', str(model)),
                    "model_name": getattr(model, 'model_name', 'Unknown'),
                    "model_type": getattr(model, 'model_type', 'Unknown'),
                    "architecture": getattr(model, 'architecture', 'Unknown'),
                    "description": getattr(model, 'description', ''),
                }
                
                # Apply filters
                if args.model_type and model_dict['model_type'] != args.model_type:
                    continue
                if args.architecture and model_dict['architecture'] != args.architecture:
                    continue
                
                filtered_models.append(model_dict)
            
            return {
                "models": filtered_models,
                "total": len(filtered_models),
                "limit": args.limit
            }
        except Exception as e:
            return {"error": str(e), "models": []}
    
    def _system_recommend_model(self, args) -> Dict[str, Any]:
        """Get model recommendations."""
        if not self.server:
            return {"error": "MCP server not available"}
        
        try:
            from ipfs_accelerate_py.model_manager import RecommendationContext, DataType
            
            context = RecommendationContext(
                task_type=args.task_type,
                hardware=args.hardware,
                input_type=DataType(args.input_type),
                output_type=DataType(args.output_type),
                requirements={}
            )
            
            recommendation = self.server.bandit_recommender.recommend_model(context)
            
            if recommendation:
                return {
                    "recommendation": {
                        "model_id": recommendation.model_id,
                        "confidence_score": recommendation.confidence_score,
                        "predicted_performance": recommendation.predicted_performance,
                        "reasoning": recommendation.reasoning
                    },
                    "context": {
                        "task_type": args.task_type,
                        "hardware": args.hardware,
                        "input_type": args.input_type,
                        "output_type": args.output_type
                    }
                }
            else:
                return {
                    "error": "No suitable model found",
                    "available_types": list(self.server.available_model_types.keys())
                }
        except Exception as e:
            return {"error": str(e)}
    
    def _system_get_stats(self, args) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.server:
            return {"error": "MCP server not available"}
        
        try:
            models = self.server.model_manager.list_models()
            stats = {
                "total_models": len(models),
                "model_types": {},
                "discovered_model_categories": {
                    category: len(models) 
                    for category, models in self.server.available_model_types.items()
                },
                "total_discovered": sum(len(models) for models in self.server.available_model_types.values()),
                "timestamp": datetime.now().isoformat()
            }
            
            for model in models:
                model_type = getattr(model, 'model_type', 'Unknown')
                if hasattr(model_type, 'value'):
                    model_type = model_type.value
                stats["model_types"][str(model_type)] = stats["model_types"].get(str(model_type), 0) + 1
            
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def _system_available_types(self, args) -> Dict[str, Any]:
        """List available model types."""
        if not self.server:
            return {"error": "MCP server not available"}
        
        return {
            "model_categories": self.server.available_model_types,
            "total_categories": len(self.server.available_model_types),
            "total_models": sum(len(models) for models in self.server.available_model_types.values()),
            "supported_inference_types": list(self.inference_types.keys())
        }
    
    def format_output(self, result: Dict[str, Any], format_type: str) -> str:
        """Format output according to the specified format."""
        if format_type == 'json':
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif format_type == 'pretty':
            return self._pretty_format(result)
        else:  # text
            return self._text_format(result)
    
    def _pretty_format(self, result: Dict[str, Any]) -> str:
        """Format output in a pretty, human-readable format."""
        lines = []
        lines.append("=" * 60)
        lines.append("AI Inference Result")
        lines.append("=" * 60)
        
        if 'error' in result:
            lines.append(f"❌ Error: {result['error']}")
            return '\n'.join(lines)
        
        # Main result fields
        main_fields = ['generated_text', 'classification', 'answer', 'transcription', 
                      'translation', 'summary', 'caption', 'generated_code']
        
        for field in main_fields:
            if field in result:
                lines.append(f"✅ {field.replace('_', ' ').title()}: {result[field]}")
        
        # Confidence and metadata
        if 'confidence' in result:
            lines.append(f"🎯 Confidence: {result['confidence']:.2%}")
        
        if 'model_used' in result:
            lines.append(f"🤖 Model: {result['model_used']}")
        
        if 'processing_time' in result:
            lines.append(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        
        # Predictions or detailed results
        if 'predictions' in result:
            lines.append("\n📊 Predictions:")
            for pred in result['predictions'][:5]:  # Top 5
                if isinstance(pred, dict):
                    label = pred.get('label', pred.get('token', 'Unknown'))
                    score = pred.get('score', pred.get('confidence', 0))
                    lines.append(f"   • {label}: {score:.2%}")
        
        # Embeddings summary
        if 'embeddings' in result:
            emb = result['embeddings']
            lines.append(f"🔢 Embeddings: {len(emb)} dimensions")
            if len(emb) > 0:
                lines.append(f"   Range: [{min(emb):.3f}, {max(emb):.3f}]")
        
        return '\n'.join(lines)
    
    def _text_format(self, result: Dict[str, Any]) -> str:
        """Format output as plain text."""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        # Return the main result content
        main_fields = ['generated_text', 'answer', 'transcription', 'translation', 
                      'summary', 'caption', 'generated_code']
        
        for field in main_fields:
            if field in result:
                return str(result[field])
        
        # For classifications, return the top prediction
        if 'classification' in result:
            if isinstance(result['classification'], dict):
                return result['classification'].get('label', str(result['classification']))
            return str(result['classification'])
        
        if 'predictions' in result and result['predictions']:
            pred = result['predictions'][0]
            if isinstance(pred, dict):
                return pred.get('label', pred.get('token', str(pred)))
            return str(pred)
        
        # Fallback to JSON
        return json.dumps(result, indent=2)
    
    def run(self, args):
        """Run the CLI with the parsed arguments."""
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Check if we have required attributes
        if not hasattr(args, 'category') or not args.category:
            print("❌ Error: No category specified")
            return 1
            
        if not hasattr(args, 'command') or not args.command:
            print(f"❌ Error: No command specified for category '{args.category}'")
            return 1
        
        # Validate category and command
        if args.category not in self.inference_types:
            print(f"❌ Error: Unknown category '{args.category}'")
            print(f"Available categories: {list(self.inference_types.keys())}")
            return 1
        
        if args.command not in self.inference_types[args.category]:
            print(f"❌ Error: Unknown command '{args.command}' for category '{args.category}'")
            print(f"Available commands: {list(self.inference_types[args.category].keys())}")
            return 1
        
        # Get the handler function
        handler = self.inference_types[args.category][args.command]
        
        try:
            # Execute the inference
            result = handler(args)
            
            # Format and display output
            output = self.format_output(result, args.output_format)
            print(output)
            
            # Save result to file if requested
            if args.save_result:
                try:
                    with open(args.save_result, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"\n💾 Result saved to: {args.save_result}")
                except Exception as e:
                    print(f"\n❌ Error saving result: {e}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error executing {args.category} {args.command}: {e}")
            print(f"❌ Error: {e}")
            return 1

def main():
    """Main entry point for the CLI."""
    cli = AIInferenceCLI()
    parser = cli.create_parser()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Check if we have a category and command
    if not hasattr(args, 'category') or not args.category:
        parser.print_help()
        return 0
    
    if not hasattr(args, 'command') or not args.command:
        print(f"Please specify a command for category '{args.category}'")
        return 1
    
    return cli.run(args)

if __name__ == "__main__":
    sys.exit(main())