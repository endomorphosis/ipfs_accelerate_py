"""
Additional models to be added to the MODEL_REGISTRY in enhanced_generator.py
to achieve 100% coverage of the models tracked in HF_MODEL_COVERAGE_ROADMAP.md.
"""

# Dictionary of additional models to add to MODEL_REGISTRY
ADDITIONAL_MODELS = {
    # Decoder-only Models (High Priority)
    "gemma2": {
        "default_model": "google/gemma2-9b",
        "task": "text-generation",
        "class": "Gemma2ForCausalLM",
        "test_input": "Once upon a time"
    },
    "gemma3": {
        "default_model": "google/gemma3-7b",
        "task": "text-generation",
        "class": "Gemma3ForCausalLM",
        "test_input": "Once upon a time"
    },
    "llama-3": {
        "default_model": "meta-llama/Llama-3-8b-hf",
        "task": "text-generation",
        "class": "Llama3ForCausalLM",
        "test_input": "Once upon a time"
    },
    "phi3": {
        "default_model": "microsoft/phi-3-mini-4k-instruct",
        "task": "text-generation",
        "class": "Phi3ForCausalLM",
        "test_input": "Once upon a time"
    },
    "phi4": {
        "default_model": "microsoft/phi-4-yi",
        "task": "text-generation",
        "class": "Phi4ForCausalLM",
        "test_input": "Once upon a time"
    },
    "mistral-next": {
        "default_model": "mistralai/Mistral-Next-8x7b",
        "task": "text-generation",
        "class": "MistralNextForCausalLM",
        "test_input": "Once upon a time"
    },
    "nemotron": {
        "default_model": "nvidia/nemotron-4-340b-instruct",
        "task": "text-generation",
        "class": "NemotronForCausalLM",
        "test_input": "Once upon a time"
    },
    "persimmon": {
        "default_model": "adept/persimmon-8b-base",
        "task": "text-generation",
        "class": "PersimmonForCausalLM",
        "test_input": "Once upon a time"
    },
    "recurrent-gemma": {
        "default_model": "google/recurrent-gemma-2b",
        "task": "text-generation",
        "class": "RecurrentGemmaForCausalLM", 
        "test_input": "Once upon a time"
    },
    "rwkv": {
        "default_model": "RWKV/rwkv-4-430m-pile",
        "task": "text-generation",
        "class": "RwkvForCausalLM",
        "test_input": "Once upon a time"
    },
    "command-r": {
        "default_model": "CohereForAI/c4ai-command-r",
        "task": "text-generation",
        "class": "CommandRForCausalLM",
        "test_input": "Once upon a time"
    },
    "codegen": {
        "default_model": "Salesforce/codegen-350M-mono",
        "task": "text-generation",
        "class": "CodeGenForCausalLM",
        "test_input": "def fibonacci(n):"
    },
    "starcoder2": {
        "default_model": "bigcode/starcoder2-3b",
        "task": "text-generation", 
        "class": "Starcoder2ForCausalLM",
        "test_input": "def fibonacci(n):"
    },
    "openai-gpt": {
        "default_model": "openai-gpt",
        "task": "text-generation",
        "class": "OpenAIGPTLMHeadModel",
        "test_input": "Once upon a time"
    },
    
    # Encoder-decoder Models (High Priority)
    "m2m-100": {
        "default_model": "facebook/m2m100_418M",
        "task": "translation",
        "class": "M2M100ForConditionalGeneration",
        "test_input": "Hello, how are you?"
    },
    "seamless-m4t": {
        "default_model": "facebook/seamless-m4t-medium",
        "task": "translation",
        "class": "SeamlessM4TForTextToSpeech",
        "test_input": "Hello, how are you?"
    },
    "switch-transformers": {
        "default_model": "google/switch-base-8",
        "task": "text2text-generation",
        "class": "SwitchTransformersForConditionalGeneration",
        "test_input": "translate English to German: Hello, how are you?"
    },
    
    # Encoder-only Models (High Priority)
    "luke": {
        "default_model": "studio-ousia/luke-base",
        "task": "fill-mask",
        "class": "LukeForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "bigbird": {
        "default_model": "google/bigbird-roberta-base",
        "task": "fill-mask",
        "class": "BigBirdForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "convbert": {
        "default_model": "YituTech/conv-bert-base",
        "task": "fill-mask",
        "class": "ConvBertForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "esm": {
        "default_model": "facebook/esm2_t33_650M_UR50D",
        "task": "fill-mask",
        "class": "EsmForMaskedLM",
        "test_input": "GCTVED[MASK]LYGV"
    },
    "ibert": {
        "default_model": "kssteven/ibert-roberta-base",
        "task": "fill-mask", 
        "class": "IBertForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "megatron-bert": {
        "default_model": "nvidia/megatron-bert-uncased-345m",
        "task": "fill-mask",
        "class": "MegatronBertForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "mobilebert": {
        "default_model": "google/mobilebert-uncased",
        "task": "fill-mask",
        "class": "MobileBertForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "nystromformer": {
        "default_model": "EleutherAI/gpt-neox-1.3B",  # Placeholder
        "task": "fill-mask",
        "class": "NystromformerForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "splinter": {
        "default_model": "tau/splinter-base",
        "task": "fill-mask",
        "class": "SplinterForPreTraining",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "roformer": {
        "default_model": "junnyu/roformer_chinese_base",
        "task": "fill-mask",
        "class": "RoFormerForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "xlm": {
        "default_model": "xlm-mlm-en-2048",
        "task": "fill-mask",
        "class": "XLMWithLMHeadModel",
        "test_input": "The quick brown fox jumps over the <mask> dog."
    },
    "xmod": {
        "default_model": "facebook/xmod-base",
        "task": "fill-mask",
        "class": "XmodForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "canine": {
        "default_model": "google/canine-s",
        "task": "token-classification",
        "class": "CanineForTokenClassification",
        "test_input": "The quick brown fox jumps over the lazy dog."
    },
    
    # Speech Models (High Priority)
    "speech-to-text": {
        "default_model": "facebook/s2t-small-librispeech-asr",
        "task": "automatic-speech-recognition",
        "class": "Speech2TextForConditionalGeneration",
        "test_input": "test.mp3"
    },
    "speech-to-text-2": {
        "default_model": "facebook/s2t-wav2vec2-large-en",
        "task": "automatic-speech-recognition",
        "class": "Speech2Text2ForCausalLM",
        "test_input": "test.mp3"
    },
    "wav2vec2-conformer": {
        "default_model": "facebook/wav2vec2-conformer-rel-pos-large-960h",
        "task": "automatic-speech-recognition",
        "class": "Wav2Vec2ConformerForCTC",
        "test_input": "test.wav"
    },
    
    # Vision Models (High Priority)
    "beit3": {
        "default_model": "microsoft/beit-3-base",
        "task": "image-classification",
        "class": "Beit3ForImageClassification",
        "test_input": "test.jpg"
    },
    "conditional-detr": {
        "default_model": "microsoft/conditional-detr-resnet-50",
        "task": "object-detection",
        "class": "ConditionalDetrForObjectDetection",
        "test_input": "test.jpg"
    },
    "dino": {
        "default_model": "facebook/dino-vits16",
        "task": "image-classification",
        "class": "DinoForImageClassification",
        "test_input": "test.jpg"
    },
    "imagegpt": {
        "default_model": "openai/imagegpt-small",
        "task": "image-generation",
        "class": "ImageGPTForCausalImageModeling",
        "test_input": "test.jpg"
    },
    "mobilenet-v1": {
        "default_model": "google/mobilenet_v1_1.0_224",
        "task": "image-classification",
        "class": "MobileNetV1ForImageClassification",
        "test_input": "test.jpg"
    },
    "dinat": {
        "default_model": "microsoft/dinat-mini-224",
        "task": "image-classification",
        "class": "DinatForImageClassification",
        "test_input": "test.jpg"
    },
    "depth-anything": {
        "default_model": "LiheYoung/depth-anything-small",
        "task": "depth-estimation",
        "class": "DepthAnythingForDepthEstimation",
        "test_input": "test.jpg"
    },
    "vitdet": {
        "default_model": "facebook/vit-det-base",
        "task": "object-detection",
        "class": "VitDetForObjectDetection",
        "test_input": "test.jpg"
    },
    "van": {
        "default_model": "Visual-Attention-Network/van-base",
        "task": "image-classification",
        "class": "VanForImageClassification", 
        "test_input": "test.jpg"
    },
    
    # Vision-text Models (High Priority)
    "vision-encoder-decoder": {
        "default_model": "nlpconnect/vit-gpt2-image-captioning",
        "task": "image-to-text",
        "class": "VisionEncoderDecoderModel",
        "test_input": "test.jpg"
    },
    "instructblip": {
        "default_model": "Salesforce/instructblip-vicuna-7b",
        "task": "image-to-text",
        "class": "InstructBlipForConditionalGeneration",
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "xclip": {
        "default_model": "microsoft/xclip-base-patch32",
        "task": "video-classification",
        "class": "XCLIPModel",
        "test_input": ["test.mp4", ["a video of a cat", "a video of a dog", "a video of a person"]]
    },

    # Multimodal Models (High Priority)
    "idefics2": {
        "default_model": "HuggingFaceM4/idefics2-8b",
        "task": "image-to-text",
        "class": "Idefics2ForConditionalGeneration", 
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "idefics3": {
        "default_model": "HuggingFaceM4/idefics3-8b",
        "task": "image-to-text",
        "class": "Idefics3ForConditionalGeneration",
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "llava-next-video": {
        "default_model": "llava-hf/llava-next-video-7b-hf",
        "task": "video-to-text",
        "class": "LlavaNextVideoForConditionalGeneration",
        "test_input": ["test.mp4", "What is happening in this video?"]
    },
    "mllama": {
        "default_model": "DAMO-NLP/mllama-1.5-7b",
        "task": "image-to-text",
        "class": "MLlamaForConditionalGeneration",
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "qwen2-vl": {
        "default_model": "Qwen/Qwen2-VL-7B",
        "task": "image-to-text",
        "class": "Qwen2VLForConditionalGeneration",
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "qwen3-vl": {
        "default_model": "Qwen/Qwen3-VL-7B",
        "task": "image-to-text",
        "class": "Qwen3VLForConditionalGeneration",
        "test_input": ["test.jpg", "What is in this image?"]
    },
    "siglip": {
        "default_model": "google/siglip-base-patch16-224",
        "task": "zero-shot-image-classification",
        "class": "SiglipModel",
        "test_input": ["test.jpg", ["a photo of a cat", "a photo of a dog", "a photo of a person"]]
    },
    
    # Additional models found in "Unknown" section but missing proper architecture categories
    "stablelm": {
        "default_model": "stabilityai/stablelm-3b-4e1t",
        "task": "text-generation",
        "class": "StableLmForCausalLM",
        "test_input": "Once upon a time"
    },
    "data2vec-text": {
        "default_model": "facebook/data2vec-text-base",
        "task": "fill-mask",
        "class": "Data2VecTextForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "vision-text-dual-encoder": {
        "default_model": "google/vit-base-patch16-224",
        "task": "zero-shot-image-classification",
        "class": "VisionTextDualEncoderModel",
        "test_input": ["test.jpg", ["a photo of a cat", "a photo of a dog", "a photo of a person"]]
    },
    "umt5": {
        "default_model": "google/umt5-small",
        "task": "text2text-generation",
        "class": "UMT5ForConditionalGeneration",
        "test_input": "translate English to German: Hello, how are you?"
    },
    "xlm-roberta-xl": {
        "default_model": "facebook/xlm-roberta-xl",
        "task": "fill-mask",
        "class": "XLMRobertaXLForMaskedLM",
        "test_input": "The quick brown fox jumps over the <mask> dog."
    },
    "nezha": {
        "default_model": "sijunhe/nezha-cn-base",
        "task": "fill-mask",
        "class": "NezhaForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "mra": {
        "default_model": "uw-madison/mra-base-512-4l",
        "task": "fill-mask",
        "class": "MraForMaskedLM",
        "test_input": "The quick brown fox jumps over the [MASK] dog."
    },
    "flaubert": {
        "default_model": "flaubert/flaubert_small_cased",
        "task": "fill-mask",
        "class": "FlaubertForMaskedLM",
        "test_input": "Le <special:mask> est tombé dans la rivière."
    },
    "olmo": {
        "default_model": "allenai/OLMo-7B",
        "task": "text-generation",
        "class": "OLMoForCausalLM",
        "test_input": "Once upon a time"
    },
    "olmoe": {
        "default_model": "allenai/OLMoE-8B",
        "task": "text-generation",
        "class": "OLMoEForCausalLM",
        "test_input": "Once upon a time"
    },
    "convnextv2": {
        "default_model": "facebook/convnextv2-tiny-1k-224",
        "task": "image-classification",
        "class": "ConvNextV2ForImageClassification",
        "test_input": "test.jpg"
    },
    "swinv2": {
        "default_model": "microsoft/swinv2-tiny-patch4-window8-256",
        "task": "image-classification",
        "class": "Swinv2ForImageClassification",
        "test_input": "test.jpg"
    },
    "cvt": {
        "default_model": "microsoft/cvt-13",
        "task": "image-classification",
        "class": "CvtForImageClassification",
        "test_input": "test.jpg"
    }
}

# Additional architecture mappings to add to ARCHITECTURE_TYPES
ADDITIONAL_ARCHITECTURE_MAPPINGS = {
    "encoder-only": [
        "bigbird", "canine", "convbert", "data2vec-text", "esm", "flaubert", 
        "ibert", "luke", "megatron-bert", "mobilebert", "nystromformer", 
        "roformer", "splinter", "xlm", "xmod", "xlm-roberta-xl", "nezha", "mra"
    ],
    "decoder-only": [
        "codegen", "command-r", "gemma2", "gemma3", "llama-3", "mamba", 
        "mistral-next", "nemotron", "olmo", "olmoe", "openai-gpt", "persimmon", 
        "phi3", "phi4", "recurrent-gemma", "rwkv", "starcoder2", "stablelm"
    ],
    "encoder-decoder": [
        "m2m-100", "seamless-m4t", "switch-transformers", "umt5"
    ],
    "vision": [
        "beit3", "conditional-detr", "convnextv2", "depth-anything", "dinat",
        "dino", "imagegpt", "mobilenet-v1", "van", "vitdet", "swinv2", "cvt"
    ],
    "vision-text": [
        "instructblip", "vision-encoder-decoder", "xclip", "vision-text-dual-encoder"
    ],
    "speech": [
        "speech-to-text", "speech-to-text-2", "wav2vec2-conformer"
    ],
    "multimodal": [
        "idefics2", "idefics3", "llava-next-video", "mllama",
        "qwen2-vl", "qwen3-vl", "siglip"
    ]
}