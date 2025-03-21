<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLIP

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP
(Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be
instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing
for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

The abstract from the paper is the following:

*State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This
restricted form of supervision limits their generality and usability since additional labeled data is needed to specify
any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a
much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes
with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400
million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference
learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study
the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks
such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The
model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need
for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot
without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained
model weights at this https URL.*

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/openai/CLIP).

## Usage tips and example

CLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image
classification. CLIP uses a ViT like transformer to get visual features and a causal language model to get the text
features. Both the text and visual features are then projected to a latent space with identical dimension. The dot
product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors
also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
The [`CLIPImageProcessor`] can be used to resize (or rescale) and normalize images for the model.

The [`CLIPTokenizer`] is used to encode the text. The [`CLIPProcessor`] wraps
[`CLIPImageProcessor`] and [`CLIPTokenizer`] into a single instance to both
encode the text and prepare the images. The following example shows how to get the image-text similarity scores using
[`CLIPProcessor`] and [`CLIPModel`].


```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```


### Combining CLIP and Flash Attention 2

First, make sure to install the latest version of Flash Attention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of flash-attn repository. Make also sure to load your model in half-precision (e.g. `torch.float16`)

<Tip warning={true}>

For small batch sizes, you might notice a slowdown in your model when using flash attention. Refer to the section [Expected speedups with Flash Attention and SDPA](#Expected-speedups-with-Flash-Attention-and-SDPA) below and select an appropriate attention implementation.

</Tip>

To load and run a model using Flash Attention 2, refer to the snippet below:

```python
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import CLIPProcessor, CLIPModel

>>> device = "cuda"
>>> torch_dtype = torch.float16

>>> model = CLIPModel.from_pretrained(
...     "openai/clip-vit-base-patch32",
...     attn_implementation="flash_attention_2",
...     device_map=device,
...     torch_dtype=torch_dtype,
... )
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
>>> inputs.to(device)

>>> with torch.no_grad():
...     with torch.autocast(device):
...         outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
>>> print(probs)
tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)
```


### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```python
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16, attn_implementation="sdpa")
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

### Expected speedups with Flash Attention and SDPA

On a local benchmark (NVIDIA A10G, PyTorch 2.3.1+cu121) with `float16`, we saw the following speedups during inference for `"openai/clip-vit-large-patch14"` checkpoint ([code](https://gist.github.com/qubvel/ac691a54e54f9fae8144275f866a7ff8)):

#### CLIPTextModel

|   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                 4 |            0.009 |          0.012 |         0.737 |           0.007 |          1.269 |
|                16 |            0.009 |          0.014 |         0.659 |           0.008 |          1.187 |
|                32 |            0.018 |          0.021 |         0.862 |           0.016 |          1.142 |
|                64 |            0.034 |          0.034 |         1.001 |           0.03  |          1.163 |
|               128 |            0.063 |          0.058 |         1.09  |           0.054 |          1.174 |

![clip_text_model_viz_3](https://github.com/user-attachments/assets/e9826b43-4e66-4f4c-952b-af4d90bd38eb)

#### CLIPVisionModel

|   Image batch size |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |            0.016 |          0.013 |         1.247 |           0.012 |          1.318 |
|                  4 |            0.025 |          0.021 |         1.198 |           0.021 |          1.202 |
|                 16 |            0.093 |          0.075 |         1.234 |           0.075 |          1.24  |
|                 32 |            0.181 |          0.147 |         1.237 |           0.146 |          1.241 |

![clip_image_model_viz_3](https://github.com/user-attachments/assets/50a36206-e3b9-4adc-ac8e-926b8b071d63)

#### CLIPModel

|   Image batch size |   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |                 4 |            0.025 |          0.026 |         0.954 |           0.02  |          1.217 |
|                  1 |                16 |            0.026 |          0.028 |         0.918 |           0.02  |          1.287 |
|                  1 |                64 |            0.042 |          0.046 |         0.906 |           0.036 |          1.167 |
|                  4 |                 4 |            0.028 |          0.033 |         0.849 |           0.024 |          1.189 |
|                  4 |                16 |            0.034 |          0.035 |         0.955 |           0.029 |          1.169 |
|                  4 |                64 |            0.059 |          0.055 |         1.072 |           0.05  |          1.179 |
|                 16 |                 4 |            0.096 |          0.088 |         1.091 |           0.078 |          1.234 |
|                 16 |                16 |            0.102 |          0.09  |         1.129 |           0.083 |          1.224 |
|                 16 |                64 |            0.127 |          0.11  |         1.157 |           0.105 |          1.218 |
|                 32 |                 4 |            0.185 |          0.159 |         1.157 |           0.149 |          1.238 |
|                 32 |                16 |            0.19  |          0.162 |         1.177 |           0.154 |          1.233 |
|                 32 |                64 |            0.216 |          0.181 |         1.19  |           0.176 |          1.228 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIP.

- [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd), a blog post about how to fine-tune CLIP with [RSICD dataset](https://github.com/201528014227051/RSICD_optimal) and comparison of performance changes due to data augmentation.
- This [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) shows how to train a CLIP-like vision-text dual encoder model using a pre-trained vision and text encoder using [COCO dataset](https://cocodataset.org/#home).

<PipelineTag pipeline="image-to-text"/>

- A [notebook](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing) on how to use a pretrained CLIP for inference with beam search for image captioning. 🌎

**Image retrieval**

- A [notebook](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing) on image retrieval using pretrained CLIP and computing MRR(Mean Reciprocal Rank) score. 🌎
- A [notebook](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb) on image retrieval and showing the similarity score. 🌎
- A [notebook](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing) on how to map images and texts to the same vector space using Multilingual CLIP. 🌎 
- A [notebook](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR) on how to run CLIP on semantic image search using [Unsplash](https://unsplash.com) and [TMDB](https://www.themoviedb.org/) datasets. 🌎

**Explainability**

- A [notebook](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) on how to visualize similarity between input token and image segment. 🌎

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

## CLIPConfig

[API documentation placeholder]

## CLIPTextConfig

[API documentation placeholder]

## CLIPVisionConfig

[API documentation placeholder]

## CLIPTokenizer

[API documentation placeholder]

## CLIPTokenizerFast

[API documentation placeholder]

## CLIPImageProcessor

[API documentation placeholder]

## CLIPImageProcessorFast

[API documentation placeholder]

## CLIPFeatureExtractor

[API documentation placeholder]

## CLIPProcessor

[API documentation placeholder]

<frameworkcontent>
<pt>

## CLIPModel

[API documentation placeholder]

## CLIPTextModel

[API documentation placeholder]

## CLIPTextModelWithProjection

[API documentation placeholder]

## CLIPVisionModelWithProjection

[API documentation placeholder]

## CLIPVisionModel

[API documentation placeholder]

## CLIPForImageClassification

[API documentation placeholder]

</pt>
<tf>

## TFCLIPModel

[API documentation placeholder]

## TFCLIPTextModel

[API documentation placeholder]

## TFCLIPVisionModel

[API documentation placeholder]

</tf>
<jax>

## FlaxCLIPModel

[API documentation placeholder]

## FlaxCLIPTextModel

[API documentation placeholder]

## FlaxCLIPTextModelWithProjection

[API documentation placeholder]

## FlaxCLIPVisionModel

[API documentation placeholder]

</jax>
</frameworkcontent>
