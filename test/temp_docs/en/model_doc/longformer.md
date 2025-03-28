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

# Longformer

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

## Overview

The Longformer model was presented in [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf) by Iz Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA.*

This model was contributed by [beltagy](https://huggingface.co/beltagy). The Authors' code can be found [here](https://github.com/allenai/longformer).

## Usage tips

- Since the Longformer is based on RoBERTa, it doesn't have `token_type_ids`. You don't need to indicate which
  token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or
  `</s>`).
- A transformer model replacing the attention matrices by sparse matrices to go faster. Often, the local context (e.g., what are the two tokens left and right?) is enough to take action for a given token. Some preselected input tokens are still given global attention, but the attention matrix has way less parameters, resulting in a speed-up. See the local attention section for more information.

## Longformer Self Attention

Longformer self attention employs self attention on both a "local" context and a "global" context. Most tokens only
attend "locally" to each other meaning that each token attends to its \\(\frac{1}{2} w\\) previous tokens and
\\(\frac{1}{2} w\\) succeeding tokens with \\(w\\) being the window length as defined in
`config.attention_window`. Note that `config.attention_window` can be of type `List` to define a
different \\(w\\) for each layer. A selected few tokens attend "globally" to all other tokens, as it is
conventionally done for all tokens in `BertSelfAttention`.

Note that "locally" and "globally" attending tokens are projected by different query, key and value matrices. Also note
that every "locally" attending token not only attends to tokens within its window \\(w\\), but also to all "globally"
attending tokens so that global attention is *symmetric*.

The user can define which tokens attend "locally" and which tokens attend "globally" by setting the tensor
`global_attention_mask` at run-time appropriately. All Longformer models employ the following logic for
`global_attention_mask`:

- 0: the token attends "locally",
- 1: the token attends "globally".

For more information please also refer to [`~LongformerModel.forward`] method.

Using Longformer self attention, the memory and time complexity of the query-key matmul operation, which usually
represents the memory and time bottleneck, can be reduced from \\(\mathcal{O}(n_s \times n_s)\\) to
\\(\mathcal{O}(n_s \times w)\\), with \\(n_s\\) being the sequence length and \\(w\\) being the average window
size. It is assumed that the number of "globally" attending tokens is insignificant as compared to the number of
"locally" attending tokens.

For more information, please refer to the official [paper](https://arxiv.org/pdf/2004.05150.pdf).


## Training

[`LongformerForMaskedLM`] is trained the exact same way [`RobertaForMaskedLM`] is
trained and should be used as follows:

```python
input_ids = tokenizer.encode("This is a sentence from [MASK] training data", return_tensors="pt")
mlm_labels = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")

loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## LongformerConfig

[API documentation placeholder]

## LongformerTokenizer

[API documentation placeholder]

## LongformerTokenizerFast

[API documentation placeholder]

## Longformer specific outputs

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

<frameworkcontent>
<pt>

## LongformerModel

[API documentation placeholder]

## LongformerForMaskedLM

[API documentation placeholder]

## LongformerForSequenceClassification

[API documentation placeholder]

## LongformerForMultipleChoice

[API documentation placeholder]

## LongformerForTokenClassification

[API documentation placeholder]

## LongformerForQuestionAnswering

[API documentation placeholder]

</pt>
<tf>

## TFLongformerModel

[API documentation placeholder]

## TFLongformerForMaskedLM

[API documentation placeholder]

## TFLongformerForQuestionAnswering

[API documentation placeholder]

## TFLongformerForSequenceClassification

[API documentation placeholder]

## TFLongformerForTokenClassification

[API documentation placeholder]

## TFLongformerForMultipleChoice

[API documentation placeholder]

</tf>
</frameworkcontent>
