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

# Utilities for Generation

This page lists all the utility functions used by [`~generation.GenerationMixin.generate`].

## Generate Outputs

The output of [`~generation.GenerationMixin.generate`] is an instance of a subclass of
[`~utils.ModelOutput`]. This output is a data structure containing all the information returned
by [`~generation.GenerationMixin.generate`], but that can also be used as tuple or dictionary.

Here's an example:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

The `generation_output` object is a [`~generation.GenerateDecoderOnlyOutput`], as we can
see in the documentation of that class below, it means it has the following attributes:

- `sequences`: the generated sequences of tokens
- `scores` (optional): the prediction scores of the language modelling head, for each generation step
- `hidden_states` (optional): the hidden states of the model, for each generation step
- `attentions` (optional): the attention weights of the model, for each generation step

Here we have the `scores` since we passed along `output_scores=True`, but we don't have `hidden_states` and
`attentions` because we didn't pass `output_hidden_states=True` or `output_attentions=True`.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `generation_output.scores` are all the generated prediction scores of the
language modeling head, and `generation_output.attentions` is `None`.

When using our `generation_output` object as a tuple, it only keeps the attributes that don't have `None` values.
Here, for instance, it has two elements, `loss` then `logits`, so

```python
generation_output[:2]
```

will return the tuple `(generation_output.sequences, generation_output.scores)` for instance.

When using our `generation_output` object as a dictionary, it only keeps the attributes that don't have `None`
values. Here, for instance, it has two keys that are `sequences` and `scores`.

We document here all output types.


### PyTorch

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

### TensorFlow

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

### FLAX

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## LogitsProcessor

A [`LogitsProcessor`] can be used to modify the prediction scores of a language model head for
generation.

### PyTorch

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]


### TensorFlow

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

### FLAX

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## StoppingCriteria

A [`StoppingCriteria`] can be used to change when to stop generation (other than EOS token). Please note that this is exclusively available to our PyTorch implementations.

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## Constraints

A [`Constraint`] can be used to force the generation to include specific tokens or sequences in the output. Please note that this is exclusively available to our PyTorch implementations.

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## BeamSearch

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## Streamers

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## Caches

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## Watermark Utils

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

## Compile Utils

[API documentation placeholder]

