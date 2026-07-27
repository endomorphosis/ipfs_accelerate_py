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

# Backbone

A backbone is a model used for feature extraction for higher level computer vision tasks such as object detection and image classification. Transformers provides an [`AutoBackbone`] class for initializing a Transformers backbone from pretrained model weights, and two utility classes:

* [`~utils.BackboneMixin`] enables initializing a backbone from Transformers or [timm](https://hf.co/docs/timm/index) and includes functions for returning the output features and indices.
* [`~utils.BackboneConfigMixin`] sets the output features and indices of the backbone configuration.

[timm](https://hf.co/docs/timm/index) models are loaded with the [`TimmBackbone`] and [`TimmBackboneConfig`] classes.

Backbones are supported for the following models:

* [BEiT](../model_doc/beit.md)
* [BiT](../model_doc/bit.md)
* [ConvNext](../model_doc/convnext.md)
* [ConvNextV2](../model_doc/convnextv2.md)
* [DiNAT](../model_doc/dinat.md)
* [DINOV2](../model_doc/dinov2.md)
* [FocalNet](../model_doc/focalnet.md)
* [MaskFormer](../model_doc/maskformer.md)
* [NAT](../model_doc/nat.md)
* [ResNet](../model_doc/resnet.md)
* [Swin Transformer](../model_doc/swin.md)
* [Swin Transformer v2](../model_doc/swinv2.md)
* [ViTDet](../model_doc/vitdet.md)

## AutoBackbone

[API documentation placeholder]

## BackboneMixin

[API documentation placeholder]

## BackboneConfigMixin

[API documentation placeholder]

## TimmBackbone

[API documentation placeholder]

## TimmBackboneConfig

[API documentation placeholder]
