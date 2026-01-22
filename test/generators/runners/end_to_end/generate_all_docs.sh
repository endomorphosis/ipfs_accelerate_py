#!/bin/bash
# Generate documentation for all model-hardware combinations

# Set working directory
cd "$(dirname "$0")"

# Generate documentation for bert-base-uncased
BERT_MODEL="bert-base-uncased"
BERT_FAMILY="text_embedding"
for HW in cpu cuda rocm mps openvino qnn webnn webgpu; do
  echo "Generating documentation for $BERT_MODEL on $HW"
  python manual_doc_test.py --model "$BERT_MODEL" --family "$BERT_FAMILY" --hardware "$HW"
done

# Generate documentation for gpt2
GPT_MODEL="gpt2"
GPT_FAMILY="text_generation"
for HW in cpu cuda rocm mps openvino qnn webnn webgpu; do
  echo "Generating documentation for $GPT_MODEL on $HW"
  python manual_doc_test.py --model "$GPT_MODEL" --family "$GPT_FAMILY" --hardware "$HW"
done

# Generate documentation for vit-base-patch16-224
VIT_MODEL="vit-base-patch16-224"
VIT_FAMILY="vision"
for HW in cpu cuda rocm mps openvino qnn webnn webgpu; do
  echo "Generating documentation for $VIT_MODEL on $HW"
  python manual_doc_test.py --model "$VIT_MODEL" --family "$VIT_FAMILY" --hardware "$HW"
done

# Generate documentation for whisper-tiny
WHISPER_MODEL="whisper-tiny"
WHISPER_FAMILY="audio"
for HW in cpu cuda rocm mps openvino qnn webnn webgpu; do
  echo "Generating documentation for $WHISPER_MODEL on $HW"
  python manual_doc_test.py --model "$WHISPER_MODEL" --family "$WHISPER_FAMILY" --hardware "$HW"
done

# Generate documentation for clip
CLIP_MODEL="openai/clip-vit-base-patch32"
CLIP_FAMILY="multimodal"
for HW in cpu cuda rocm mps openvino qnn webnn webgpu; do
  echo "Generating documentation for $CLIP_MODEL on $HW"
  python manual_doc_test.py --model "$CLIP_MODEL" --family "$CLIP_FAMILY" --hardware "$HW"
done

echo "Documentation generation completed"