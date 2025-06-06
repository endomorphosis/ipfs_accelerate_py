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

# Agents & Tools

<Tip warning={true}>

Transformers Agents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

To learn more about agents and tools make sure to read the [introductory guide](../transformers_agents). This page
contains the API docs for the underlying classes.

## Agents

We provide two types of agents, based on the main [`Agent`] class:
- [`CodeAgent`] acts in one shot, generating code to solve the task, then executes it at once.
- [`ReactAgent`] acts step by step, each step consisting of one thought, then one tool call and execution. It has two classes:
  - [`ReactJsonAgent`] writes its tool calls in JSON.
  - [`ReactCodeAgent`] writes its tool calls in Python code.

### Agent

[API documentation placeholder]

### CodeAgent

[API documentation placeholder]

### React agents

[API documentation placeholder]

[API documentation placeholder]

[API documentation placeholder]

### ManagedAgent

[API documentation placeholder]

## Tools

### load_tool

[API documentation placeholder]

### tool

[API documentation placeholder]

### Tool

[API documentation placeholder]

### Toolbox

[API documentation placeholder]

### PipelineTool

[API documentation placeholder]

### launch_gradio_demo

[API documentation placeholder]

### stream_to_gradio

[API documentation placeholder]

### ToolCollection

[API documentation placeholder]

## Engines

You're free to create and use your own engines to be usable by the Agents framework.
These engines have the following specification:
1. Follow the [messages format](../chat_templating.md) for its input (`List[Dict[str, str]]`) and return a string.
2. Stop generating outputs *before* the sequences passed in the argument `stop_sequences`

### TransformersEngine

For convenience, we have added a `TransformersEngine` that implements the points above, taking a pre-initialized `Pipeline` as input.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TransformersEngine

>>> model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> model = AutoModelForCausalLM.from_pretrained(model_name)

>>> pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

>>> engine = TransformersEngine(pipe)
>>> engine([{"role": "user", "content": "Ok!"}], stop_sequences=["great"])

"What a "
```

[API documentation placeholder]

### HfApiEngine

The `HfApiEngine` is an engine that wraps an [HF Inference API](https://huggingface.co/docs/api-inference/index) client for the execution of the LLM.

```python
>>> from transformers import HfApiEngine

>>> messages = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "No need to help, take it easy."},
... ]

>>> HfApiEngine()(messages, stop_sequences=["conversation"])

"That's very kind of you to say! It's always nice to have a relaxed "
```

[API documentation placeholder]


## Agent Types

Agents can handle any type of object in-between tools; tools, being completely multimodal, can accept and return
text, image, audio, video, among other types. In order to increase compatibility between tools, as well as to 
correctly render these returns in ipython (jupyter, colab, ipython notebooks, ...), we implement wrapper classes
around these types.

The wrapped objects should continue behaving as initially; a text object should still behave as a string, an image
object should still behave as a `PIL.Image`.

These types have three specific purposes:

- Calling `to_raw` on the type should return the underlying object
- Calling `to_string` on the type should return the object as a string: that can be the string in case of an `AgentText`
  but will be the path of the serialized version of the object in other instances
- Displaying it in an ipython kernel should display the object correctly

### AgentText

[API documentation placeholder]

### AgentImage

[API documentation placeholder]

### AgentAudio

[API documentation placeholder]