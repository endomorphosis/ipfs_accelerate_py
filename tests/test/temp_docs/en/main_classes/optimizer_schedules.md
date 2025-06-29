<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Optimization

The `.optimization` module provides:

- an optimizer with weight decay fixed that can be used to fine-tuned models, and
- several schedules in the form of schedule objects that inherit from `_LRSchedule`:
- a gradient accumulation class to accumulate the gradients of multiple batches

## AdamW (PyTorch)

[API documentation placeholder]

## AdaFactor (PyTorch)

[API documentation placeholder]

## AdamWeightDecay (TensorFlow)

[API documentation placeholder]

[API documentation placeholder]

## Schedules

### Learning Rate Schedules (PyTorch)

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[API documentation placeholder]

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[API documentation placeholder]

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[API documentation placeholder]

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

### Warmup (TensorFlow)

[API documentation placeholder]

## Gradient Strategies

### GradientAccumulator (TensorFlow)

[API documentation placeholder]