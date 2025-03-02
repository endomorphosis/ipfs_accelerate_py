# Hugging Face Model Testing Progress

*Updated: 2025-03-01 16:47:46*

## Summary

<div style='display: flex; flex-wrap: wrap; gap: 10px;'>

<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>
<h3>Test Coverage</h3>
<p><b>7</b> of 47 models</p>
<p><b>14.9%</b> coverage</p>
</div>

<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>
<h3>Remote Code Models</h3>
<p><b>3</b> models requiring remote code</p>
<p><b>42.9%</b> of tested models</p>
</div>

<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1;'>
<h3>Dependencies</h3>
<p><b>10</b> unique dependencies</p>
<p><b>Top dependencies:</b></p>
<ul>
<li>sentencepiece: 5 models</li>
<li>tokenizers: 4 models</li>
<li>accelerate: 3 models</li>
</ul>
</div>

</div>

## Overall Progress

- **Total Models**: 7
- **Successfully Tested**: 7 (100.0%)
- **Failed Models**: 0

## Coverage by Priority

| Priority | Total | Tested | Coverage |
|----------|-------|--------|----------|
| High | 7 | 7 | 100.0% |

## Coverage by Task

| Task | Total | Tested | Coverage |
|------|-------|--------|----------|
| fill-mask | 2 | 2 | 100.0% |
| text-generation | 4 | 4 | 100.0% |
| text2text-generation | 1 | 1 | 100.0% |

## Recently Tested Models

- **bert-base-uncased**
  - Dependencies: tokenizers>=0.11.0, sentencepiece
- **gpt2**
  - Dependencies: regex
- **t5-small**
  - Dependencies: sentencepiece, tokenizers
- **distilroberta-base**
  - Dependencies: tokenizers>=0.11.0, sentencepiece
- **meta-llama/Llama-2-7b-hf** (requires remote code)
  - Dependencies: sentencepiece, tokenizers>=0.13.3, accelerate>=0.20.3
- **mistralai/Mistral-7B-v0.1** (requires remote code)
  - Dependencies: einops, accelerate>=0.18.0, safetensors>=0.3.2
- **google/gemma-2b** (requires remote code)
  - Dependencies: sentencepiece, accelerate>=0.21.0, safetensors>=0.3.2
