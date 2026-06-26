#!/bin/bash

MODEL_PATH="/your/path/to/the/model/Qwen3.6-35B-A3B-Q8_0.gguf"
BUILD_BIN="/your/path/to/llama.cpp/build/bin"
CUDA_LIB="/your/path/to/cuda"

export LD_LIBRARY_PATH="$BUILD_BIN:$CUDA_LIB:$LD_LIBRARY_PATH"
export GGML_CUDA_DISABLE_GRAPHS=1 # Prevent CUDA crashes due to mixed GPU generations

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,3,4

# --n-gpu-layers 99 | Force all model layers off the CPU and lock to VRAM
# --split-mode      | Split the model's sequential layers across the GPUs to prevent out of memory errors
# -c                | Context window size
# --flash-attn on   | Reduces self-attention memory overhead to speed up prefill
# -ctk / -ctv       | Quantize key/value cache tensors to 8-bit instead of FP16
# --cache-prompt    | Track static context blocks so the model doesn't reprocess them on every turn
# --cache-ram       | Dedicated host memory for prompt cache swapping
# -t 8 / -tb 8      | Restrict generations/batch pre-fills to 8 threads to avoid CPU efficiency cores
# -np 1             | Enforce single-slot execution to avoid GPU memory thrashing

$BUILD_BIN/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --split-mode layer \
  -c 262144 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv q8_0 \
  --cache-prompt \
  --cache-ram 16384 \
  -t 8 \
  -tb 8 \
  -np 1 \
  --metrics
