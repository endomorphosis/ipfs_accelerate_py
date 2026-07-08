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

# Model outputs

All models have outputs that are instances of subclasses of [`~utils.ModelOutput`]. Those are
data structures containing all the information returned by the model, but that can also be used as tuples or
dictionaries.

Let's see how this looks in an example:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
```

The `outputs` object is a [`~modeling_outputs.SequenceClassifierOutput`], as we can see in the
documentation of that class below, it means it has an optional `loss`, a `logits`, an optional `hidden_states` and
an optional `attentions` attribute. Here we have the `loss` since we passed along `labels`, but we don't have
`hidden_states` and `attentions` because we didn't pass `output_hidden_states=True` or
`output_attentions=True`.

<Tip>

When passing `output_hidden_states=True` you may expect the `outputs.hidden_states[-1]` to match `outputs.last_hidden_state` exactly.
However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it's returned.

</Tip>


You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get `None`. Here for instance `outputs.loss` is the loss computed by the model, and `outputs.attentions` is
`None`.

When considering our `outputs` object as tuple, it only considers the attributes that don't have `None` values.
Here for instance, it has two elements, `loss` then `logits`, so

```python
outputs[:2]
```

will return the tuple `(outputs.loss, outputs.logits)` for instance.

When considering our `outputs` object as dictionary, it only considers the attributes that don't have `None`
values. Here for instance, it has two keys that are `loss` and `logits`.

We document here the generic model outputs that are used by more than one model type. Specific output types are
documented on their corresponding model page.

## ModelOutput

[API documentation placeholder]

## BaseModelOutput

[API documentation placeholder]

## BaseModelOutputWithPooling

[API documentation placeholder]

## BaseModelOutputWithCrossAttentions

[API documentation placeholder]

## BaseModelOutputWithPoolingAndCrossAttentions

[API documentation placeholder]

## BaseModelOutputWithPast

[API documentation placeholder]

## BaseModelOutputWithPastAndCrossAttentions

[API documentation placeholder]

## Seq2SeqModelOutput

[API documentation placeholder]

## CausalLMOutput

[API documentation placeholder]

## CausalLMOutputWithCrossAttentions

[API documentation placeholder]

## CausalLMOutputWithPast

[API documentation placeholder]

## MaskedLMOutput

[API documentation placeholder]

## Seq2SeqLMOutput

[API documentation placeholder]

## NextSentencePredictorOutput

[API documentation placeholder]

## SequenceClassifierOutput

[API documentation placeholder]

## Seq2SeqSequenceClassifierOutput

[API documentation placeholder]

## MultipleChoiceModelOutput

[API documentation placeholder]

## TokenClassifierOutput

[API documentation placeholder]

## QuestionAnsweringModelOutput

[API documentation placeholder]

## Seq2SeqQuestionAnsweringModelOutput

[API documentation placeholder]

## Seq2SeqSpectrogramOutput

[API documentation placeholder]

## SemanticSegmenterOutput

[API documentation placeholder]

## ImageClassifierOutput

[API documentation placeholder]

## ImageClassifierOutputWithNoAttention

[API documentation placeholder]

## DepthEstimatorOutput

[API documentation placeholder]

## Wav2Vec2BaseModelOutput

[API documentation placeholder]

## XVectorOutput

[API documentation placeholder]

## Seq2SeqTSModelOutput

[API documentation placeholder]

## Seq2SeqTSPredictionOutput

[API documentation placeholder]

## SampleTSPredictionOutput

[API documentation placeholder]

## TFBaseModelOutput

[API documentation placeholder]

## TFBaseModelOutputWithPooling

[API documentation placeholder]

## TFBaseModelOutputWithPoolingAndCrossAttentions

[API documentation placeholder]

## TFBaseModelOutputWithPast

[API documentation placeholder]

## TFBaseModelOutputWithPastAndCrossAttentions

[API documentation placeholder]

## TFSeq2SeqModelOutput

[API documentation placeholder]

## TFCausalLMOutput

[API documentation placeholder]

## TFCausalLMOutputWithCrossAttentions

[API documentation placeholder]

## TFCausalLMOutputWithPast

[API documentation placeholder]

## TFMaskedLMOutput

[API documentation placeholder]

## TFSeq2SeqLMOutput

[API documentation placeholder]

## TFNextSentencePredictorOutput

[API documentation placeholder]

## TFSequenceClassifierOutput

[API documentation placeholder]

## TFSeq2SeqSequenceClassifierOutput

[API documentation placeholder]

## TFMultipleChoiceModelOutput

[API documentation placeholder]

## TFTokenClassifierOutput

[API documentation placeholder]

## TFQuestionAnsweringModelOutput

[API documentation placeholder]

## TFSeq2SeqQuestionAnsweringModelOutput

[API documentation placeholder]

## FlaxBaseModelOutput

[API documentation placeholder]

## FlaxBaseModelOutputWithPast

[API documentation placeholder]

## FlaxBaseModelOutputWithPooling

[API documentation placeholder]

## FlaxBaseModelOutputWithPastAndCrossAttentions

[API documentation placeholder]

## FlaxSeq2SeqModelOutput

[API documentation placeholder]

## FlaxCausalLMOutputWithCrossAttentions

[API documentation placeholder]

## FlaxMaskedLMOutput

[API documentation placeholder]

## FlaxSeq2SeqLMOutput

[API documentation placeholder]

## FlaxNextSentencePredictorOutput

[API documentation placeholder]

## FlaxSequenceClassifierOutput

[API documentation placeholder]

## FlaxSeq2SeqSequenceClassifierOutput

[API documentation placeholder]

## FlaxMultipleChoiceModelOutput

[API documentation placeholder]

## FlaxTokenClassifierOutput

[API documentation placeholder]

## FlaxQuestionAnsweringModelOutput

[API documentation placeholder]

## FlaxSeq2SeqQuestionAnsweringModelOutput

[API documentation placeholder]