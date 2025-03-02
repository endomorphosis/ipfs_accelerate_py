# Hugging Face Model Categorization

## 1. Text Generation

Models used primarily for text generation with transformer-based architectures.

### Models:
- albert
- bart
- bert-generation
- biogpt
- bloom
- camembert
- code_llama
- codegen
- codellama
- cohere
- ctrl
- deberta
- deberta-v2
- deepseek
- deepseek-coder
- deepseek-distil
- deepseek-r1
- deepseek-r1-distil
- distilbert
- electra
- ernie
- ernie_m
- falcon
- falcon_mamba
- flaubert
- funnel
- gemma
- gemma2
- gemma3
- glm
- gpt-sw3
- gpt2
- gpt_bigcode
- gpt_neo
- gpt_neox
- gpt_neox_japanese
- gptj
- gptsan-japanese
- granite
- granitemoe
- ibert
- llama
- mamba
- mamba2
- megatron-bert
- mistral
- mistral-nemo
- mistral-next
- mixtral
- mobilebert
- mpnet
- mpt
- nezha
- nystromformer
- olmo
- olmoe
- open-llama
- openai-gpt
- opt
- persimmon
- phi
- phi3
- phi4
- phimoe
- qwen2
- qwen3
- qwen3_moe
- roberta
- roberta-prelayernorm
- roc_bert
- roformer
- rwkv
- stablelm
- starcoder2
- t5
- transfo-xl
- xlm
- xlm-roberta
- xlm-roberta-xl
- xlnet

### Transformers Methods:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# For decoder-only models (GPT, LLama, etc.)
tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModelForCausalLM.from_pretrained("model_name")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Your prompt here", max_length=100)

# For encoder-decoder models (T5, BART)
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("model_name")
seq2seq = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
output = seq2seq("Your input here")
```

## 2. Vision Models

Models for computer vision tasks including image classification, object detection, etc.

### Models:
- beit
- bit
- convnext
- convnextv2
- cvt
- deit
- detr
- depth_anything
- dinov2
- dpt
- efficientformer
- efficientnet
- focalnet
- glpn
- hiera
- imagegpt
- levit
- mobilenet_v1
- mobilenet_v2
- mobilevit
- mobilevitv2
- poolformer
- pvt
- pvt_v2
- regnet
- resnet
- rt_detr
- rt_detr_resnet
- sam
- segformer
- seggpt
- superpoint
- swiftformer
- swin
- swin2sr
- swinv2
- upernet
- van
- vit
- vit_hybrid
- vit_mae
- vit_msn
- vitdet
- vitmatte
- yolos

### Transformers Methods:
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# Image classification
processor = AutoImageProcessor.from_pretrained("model_name")
model = AutoModelForImageClassification.from_pretrained("model_name")
classifier = pipeline("image-classification", model=model, feature_extractor=processor)
output = classifier("path/to/image.jpg")

# Object detection
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained("model_name")
detector = pipeline("object-detection", model=model, feature_extractor=processor)
output = detector("path/to/image.jpg")

# Image segmentation
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained("model_name")
segmenter = pipeline("image-segmentation", model=model, feature_extractor=processor)
output = segmenter("path/to/image.jpg")
```

## 3. Multimodal Models

Models that handle multiple modalities like text and images together.

### Models:
- align
- altclip
- blip
- blip-2
- bridgetower
- chinese_clip
- chinese_clip_vision_model
- clip
- clip_text_model
- clip_vision_model
- clipseg
- donut-swin
- flava
- fuyu
- git
- groupvit
- idefics
- idefics2
- idefics3
- instructblip
- instructblipvideo
- kosmos-2
- llava
- llava_next
- llava_next_video
- llava_onevision
- mllama
- omdet-turbo
- owlv2
- owlvit
- paligemma
- pix2struct
- qwen2_vl
- qwen3_vl
- siglip
- siglip_vision_model
- tvlt
- vilt
- video_llava
- videomae
- vipllava
- vision-encoder-decoder
- vision-text-dual-encoder
- visual_bert
- vivit
- xclip

### Transformers Methods:
```python
from transformers import AutoProcessor, AutoModel

# CLIP
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")
outputs = model(**inputs)
image_features = outputs.image_embeds
text_features = outputs.text_embeds

# LLaVA and similar models
from transformers import AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")
inputs = processor(text="Describe this image:", images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
```

## 4. Audio Models

Models for processing audio including speech recognition, classification, etc.

### Models:
- audio-spectrogram-transformer
- bark
- clap
- clvp
- data2vec-audio
- encodec
- fastspeech2_conformer
- hubert
- jukebox
- mctct
- musicgen
- musicgen_melody
- pop2piano
- qwen2_audio
- qwen2_audio_encoder
- sew
- sew-d
- speech-encoder-decoder
- speech_to_text
- speech_to_text_2
- speecht5
- unispeech
- unispeech-sat
- univnet
- vits
- wav2vec2
- wav2vec2-bert
- wav2vec2-conformer
- wavlm
- whisper

### Transformers Methods:
```python
from transformers import AutoProcessor, AutoModelForAudioClassification, pipeline

# Audio classification
processor = AutoProcessor.from_pretrained("model_name")
model = AutoModelForAudioClassification.from_pretrained("model_name")
classifier = pipeline("audio-classification", model=model, feature_extractor=processor)
output = classifier("path/to/audio.wav")

# Automatic speech recognition
from transformers import AutoModelForSpeechSeq2Seq
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
processor = AutoProcessor.from_pretrained("openai/whisper-small")
transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)
output = transcriber("path/to/audio.wav")
```

## 5. Seq2Seq Models

Models designed for sequence-to-sequence tasks like translation, summarization, etc.

### Models:
- bart
- big_bird
- bigbird_pegasus
- blenderbot
- blenderbot-small
- cpmant
- dbrx
- dbrx-instruct
- encoder-decoder
- fsmt
- jamba
- led
- lilt
- longformer
- longt5
- m2m_100
- marian
- mbart
- mega
- mt5
- mvp
- nat
- nemotron
- nllb-moe
- pegasus
- pegasus_x
- plbart
- prophetnet
- rag
- reformer
- switch_transformers
- t5
- trocr
- umt5
- xlm-prophetnet

### Transformers Methods:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Translation
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
translator = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
output = translator("This is a test")

# Summarization
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
output = summarizer("Your long text to summarize...")
```

## 6. Token Classification Models

Models for tasks like Named Entity Recognition, POS tagging, etc.

### Models:
- bert
- bros
- canine
- convbert
- data2vec-text
- dpr
- fnet
- layoutlm
- layoutlmv2
- layoutlmv3
- luke
- lxmert
- markuplm
- retribert
- splinter
- squeezebert
- tapas
- udop

### Transformers Methods:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Named Entity Recognition
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
output = ner("My name is Jean and I live in New York")

# Part-of-speech tagging
pos_tagger = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
output = pos_tagger("This is a simple test sentence")
```

## 7. Time Series Models

Models for time series forecasting and analysis.

### Models:
- autoformer
- informer
- patchtsmixer
- patchtst
- time_series_transformer
- timesformer
- trajectory_transformer

### Transformers Methods:
```python
from transformers import AutoModelForTimeSeriesPrediction

# Time series forecasting
model = AutoModelForTimeSeriesPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")
outputs = model(past_values=past_values, past_time_features=past_time_features, 
               future_time_features=future_time_features)
```

## 8. Specialized Models

Models with specific purposes that don't fit neatly into the above categories.

### Models:
- chameleon
- conditional_detr
- dac
- data2vec-vision
- decision_transformer
- deepseek-vision
- deformable_detr
- deta
- dinat
- esm
- graphormer
- grounding-dino
- jetmoe
- mask2former
- maskformer
- maskformer-swin
- mgp-str
- mimi
- mra
- nougat
- oneformer
- perceiver
- pixtral
- realm
- recurrent_gemma
- table-transformer
- timm_backbone
- tvp
- xglm
- xmod
- yoso
- zamba
- zoedepth
- optimized_model

### Transformers Methods:
```python
# These vary greatly by model type - examples:

# For ESM (protein models)
from transformers import EsmForProteinFolding, EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForProteinFolding.from_pretrained("facebook/esm2_t33_650M_UR50D")
outputs = model(input_ids)

# For graphormer
from transformers import AutoModelForGraphClassification
model = AutoModelForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")
outputs = model(input_data)
```

## 9. General Pipeline Usage

For most models, regardless of type, this general pattern can be used:

```python
from transformers import pipeline, AutoTokenizer, AutoModel

# For unknown model types, try general auto classes first
model_name = "model_name_or_path"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Or use the pipeline API which will select the appropriate classes
pipe = pipeline(task="appropriate_task", model=model_name)
result = pipe("Your input")
```