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

# MobileBERT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

## Overview

The MobileBERT model was proposed in [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) by Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny
Zhou. It's a bidirectional transformer based on the BERT model, which is compressed and accelerated using several
approaches.

The abstract from the paper is the following:

*Natural Language Processing (NLP) has recently achieved great success by using huge pre-trained models with hundreds
of millions of parameters. However, these models suffer from heavy model sizes and high latency such that they cannot
be deployed to resource-limited mobile devices. In this paper, we propose MobileBERT for compressing and accelerating
the popular BERT model. Like the original BERT, MobileBERT is task-agnostic, that is, it can be generically applied to
various downstream NLP tasks via simple fine-tuning. Basically, MobileBERT is a thin version of BERT_LARGE, while
equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks.
To train MobileBERT, we first train a specially designed teacher model, an inverted-bottleneck incorporated BERT_LARGE
model. Then, we conduct knowledge transfer from this teacher to MobileBERT. Empirical studies show that MobileBERT is
4.3x smaller and 5.5x faster than BERT_BASE while achieving competitive results on well-known benchmarks. On the
natural language inference tasks of GLUE, MobileBERT achieves a GLUEscore o 77.7 (0.6 lower than BERT_BASE), and 62 ms
latency on a Pixel 4 phone. On the SQuAD v1.1/v2.0 question answering task, MobileBERT achieves a dev F1 score of
90.0/79.2 (1.5/2.1 higher than BERT_BASE).*

This model was contributed by [vshampor](https://huggingface.co/vshampor). The original code can be found [here](https://github.com/google-research/google-research/tree/master/mobilebert).

## Usage tips

- MobileBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather
  than the left.
- MobileBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore
  efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained
  with a causal language modeling (CLM) objective are better in that regard.


## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## MobileBertConfig

[API documentation placeholder]

## MobileBertTokenizer

[API documentation placeholder]

## MobileBertTokenizerFast

[API documentation placeholder]

## MobileBert specific outputs

[API documentation placeholder]

[API documentation placeholder]

<frameworkcontent>
<pt>

## MobileBertModel

[API documentation placeholder]

## MobileBertForPreTraining

[API documentation placeholder]

## MobileBertForMaskedLM

[API documentation placeholder]

## MobileBertForNextSentencePrediction

[API documentation placeholder]

## MobileBertForSequenceClassification

[API documentation placeholder]

## MobileBertForMultipleChoice

[API documentation placeholder]

## MobileBertForTokenClassification

[API documentation placeholder]

## MobileBertForQuestionAnswering

[API documentation placeholder]

</pt>
<tf>

## TFMobileBertModel

[API documentation placeholder]

## TFMobileBertForPreTraining

[API documentation placeholder]

## TFMobileBertForMaskedLM

[API documentation placeholder]

## TFMobileBertForNextSentencePrediction

[API documentation placeholder]

## TFMobileBertForSequenceClassification

[API documentation placeholder]

## TFMobileBertForMultipleChoice

[API documentation placeholder]

## TFMobileBertForTokenClassification

[API documentation placeholder]

## TFMobileBertForQuestionAnswering

[API documentation placeholder]

</tf>
</frameworkcontent>
