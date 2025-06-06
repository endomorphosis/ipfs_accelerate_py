<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn't be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren't supported in Transformers can be added with the [`HfQuantizer`] class.

<Tip>

Learn how to quantize models in the [Quantization](../quantization) guide.

</Tip>

## QuantoConfig

[API documentation placeholder]

## AqlmConfig

[API documentation placeholder]

## VptqConfig

[API documentation placeholder]

## AwqConfig

[API documentation placeholder]

## EetqConfig
[API documentation placeholder]

## GPTQConfig

[API documentation placeholder]

## BitsAndBytesConfig

[API documentation placeholder]

## HfQuantizer

[API documentation placeholder]

## HiggsConfig

[API documentation placeholder]

## HqqConfig

[API documentation placeholder]

## FbgemmFp8Config

[API documentation placeholder]

## CompressedTensorsConfig

[API documentation placeholder]

## TorchAoConfig

[API documentation placeholder]

## BitNetConfig

[API documentation placeholder]

## SpQRConfig

[API documentation placeholder]

## FineGrainedFP8Config

[API documentation placeholder]