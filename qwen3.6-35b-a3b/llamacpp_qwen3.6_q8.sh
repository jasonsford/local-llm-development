#!/bin/bash

MODEL_PATH="/your/path/to/the/model/Qwen3.6-35B-A3B-Q8_0.gguf"
BUILD_BIN="/your/path/to/llama.cpp/build/bin"
CUDA_LIB="/usr/local/cuda-13.1/lib64"

export LD_LIBRARY_PATH="$BUILD_BIN:$CUDA_LIB:$LD_LIBRARY_PATH"
export GGML_CUDA_DISABLE_GRAPHS=1 #Prevent CUDA crashes since I'm using a combination of 4000 and 5000 series cards

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,3,4

$BUILD_BIN/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \  # Force all model layers off the CPU and stick to VRAM
  --split-mode layer \ # Split the sequential layers across the GPUs to prevent out of memory errors
  -c 262144 \          # Context window size
  --flash-attn on \    # Reduces self-attention memory overhead to speed up prefill
  -ctk q8_0 \          # Quantize key cache tensors to 8-bit instead of FP16
  -ctv q8_0 \          # Quantize value cache to 8-bit instead of FP16
  --cache-prompt \     # Turn on tracking for static context blocks so the model doesn't reprocess them on every turn
  --cache-ram 16384 \  # Dedicated host memory for prompt cache swapping
  -t 8 \               # Restrict generations to 8 threads so llama-server doesn't use the CPU efficiency cores
  -tb 8 \              # Restrict batch pre-fills to 8 threats so llama-server doesn't use the CPU efficiency cores
  -np 1 \              # Enforce single-slot execution to avoid GPU memory thrashing
  --metrics
