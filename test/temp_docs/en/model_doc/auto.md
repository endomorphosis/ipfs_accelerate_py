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

# Auto Classes

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you
are supplying to the `from_pretrained()` method. AutoClasses are here to do this job for you so that you
automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.

Instantiating one of [`AutoConfig`], [`AutoModel`], and
[`AutoTokenizer`] will directly create a class of the relevant architecture. For instance


```python
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

will create a model that is an instance of [`BertModel`].

There is one class of `AutoModel` for each task, and for each backend (PyTorch, TensorFlow, or Flax).

## Extending the Auto Classes

Each of the auto classes has a method to be extended with your custom classes. For instance, if you have defined a
custom class of model `NewModel`, make sure you have a `NewModelConfig` then you can add those to the auto
classes like this:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

You will then be able to use the auto classes like you would usually do!

<Tip warning={true}>

If your `NewModelConfig` is a subclass of [`~transformers.PretrainedConfig`], make sure its
`model_type` attribute is set to the same key you use when registering the config (here `"new-model"`).

Likewise, if your `NewModel` is a subclass of [`PreTrainedModel`], make sure its
`config_class` attribute is set to the same class you use when registering the model (here
`NewModelConfig`).

</Tip>

## AutoConfig

[API documentation placeholder]

## AutoTokenizer

[API documentation placeholder]

## AutoFeatureExtractor

[API documentation placeholder]

## AutoImageProcessor

[API documentation placeholder]

## AutoProcessor

[API documentation placeholder]

## Generic model classes

The following auto classes are available for instantiating a base model class without a specific head.

### AutoModel

[API documentation placeholder]

### TFAutoModel

[API documentation placeholder]

### FlaxAutoModel

[API documentation placeholder]

## Generic pretraining classes

The following auto classes are available for instantiating a model with a pretraining head.

### AutoModelForPreTraining

[API documentation placeholder]

### TFAutoModelForPreTraining

[API documentation placeholder]

### FlaxAutoModelForPreTraining

[API documentation placeholder]

## Natural Language Processing

The following auto classes are available for the following natural language processing tasks.

### AutoModelForCausalLM

[API documentation placeholder]

### TFAutoModelForCausalLM

[API documentation placeholder]

### FlaxAutoModelForCausalLM

[API documentation placeholder]

### AutoModelForMaskedLM

[API documentation placeholder]

### TFAutoModelForMaskedLM

[API documentation placeholder]

### FlaxAutoModelForMaskedLM

[API documentation placeholder]

### AutoModelForMaskGeneration

[API documentation placeholder]

### TFAutoModelForMaskGeneration

[API documentation placeholder]

### AutoModelForSeq2SeqLM

[API documentation placeholder]

### TFAutoModelForSeq2SeqLM

[API documentation placeholder]

### FlaxAutoModelForSeq2SeqLM

[API documentation placeholder]

### AutoModelForSequenceClassification

[API documentation placeholder]

### TFAutoModelForSequenceClassification

[API documentation placeholder]

### FlaxAutoModelForSequenceClassification

[API documentation placeholder]

### AutoModelForMultipleChoice

[API documentation placeholder]

### TFAutoModelForMultipleChoice

[API documentation placeholder]

### FlaxAutoModelForMultipleChoice

[API documentation placeholder]

### AutoModelForNextSentencePrediction

[API documentation placeholder]

### TFAutoModelForNextSentencePrediction

[API documentation placeholder]

### FlaxAutoModelForNextSentencePrediction

[API documentation placeholder]

### AutoModelForTokenClassification

[API documentation placeholder]

### TFAutoModelForTokenClassification

[API documentation placeholder]

### FlaxAutoModelForTokenClassification

[API documentation placeholder]

### AutoModelForQuestionAnswering

[API documentation placeholder]

### TFAutoModelForQuestionAnswering

[API documentation placeholder]

### FlaxAutoModelForQuestionAnswering

[API documentation placeholder]

### AutoModelForTextEncoding

[API documentation placeholder]

### TFAutoModelForTextEncoding

[API documentation placeholder]

## Computer vision

The following auto classes are available for the following computer vision tasks.

### AutoModelForDepthEstimation

[API documentation placeholder]

### AutoModelForImageClassification

[API documentation placeholder]

### TFAutoModelForImageClassification

[API documentation placeholder]

### FlaxAutoModelForImageClassification

[API documentation placeholder]

### AutoModelForVideoClassification

[API documentation placeholder]

### AutoModelForKeypointDetection

[API documentation placeholder]

### AutoModelForMaskedImageModeling

[API documentation placeholder]

### TFAutoModelForMaskedImageModeling

[API documentation placeholder]

### AutoModelForObjectDetection

[API documentation placeholder]

### AutoModelForImageSegmentation

[API documentation placeholder]

### AutoModelForImageToImage

[API documentation placeholder]

### AutoModelForSemanticSegmentation

[API documentation placeholder]

### TFAutoModelForSemanticSegmentation

[API documentation placeholder]

### AutoModelForInstanceSegmentation

[API documentation placeholder]

### AutoModelForUniversalSegmentation

[API documentation placeholder]

### AutoModelForZeroShotImageClassification

[API documentation placeholder]

### TFAutoModelForZeroShotImageClassification

[API documentation placeholder]

### AutoModelForZeroShotObjectDetection

[API documentation placeholder]

## Audio

The following auto classes are available for the following audio tasks.

### AutoModelForAudioClassification

[API documentation placeholder]

### AutoModelForAudioFrameClassification

[API documentation placeholder]

### TFAutoModelForAudioFrameClassification

[API documentation placeholder]

### AutoModelForCTC

[API documentation placeholder]

### AutoModelForSpeechSeq2Seq

[API documentation placeholder]

### TFAutoModelForSpeechSeq2Seq

[API documentation placeholder]

### FlaxAutoModelForSpeechSeq2Seq

[API documentation placeholder]

### AutoModelForAudioXVector

[API documentation placeholder]

### AutoModelForTextToSpectrogram

[API documentation placeholder]

### AutoModelForTextToWaveform

[API documentation placeholder]

## Multimodal

The following auto classes are available for the following multimodal tasks.

### AutoModelForTableQuestionAnswering

[API documentation placeholder]

### TFAutoModelForTableQuestionAnswering

[API documentation placeholder]

### AutoModelForDocumentQuestionAnswering

[API documentation placeholder]

### TFAutoModelForDocumentQuestionAnswering

[API documentation placeholder]

### AutoModelForVisualQuestionAnswering

[API documentation placeholder]

### AutoModelForVision2Seq

[API documentation placeholder]

### TFAutoModelForVision2Seq

[API documentation placeholder]

### FlaxAutoModelForVision2Seq

[API documentation placeholder]

### AutoModelForImageTextToText

[API documentation placeholder]