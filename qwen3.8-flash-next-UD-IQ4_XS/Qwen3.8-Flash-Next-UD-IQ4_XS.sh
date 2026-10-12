#!/usr/bin/env bash
# display_name: Qwen3.8 Flash Next UD IQ4 XS

LLAMA_BIN=/your/path/to/llama.cpp/build/bin/llama-server
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

exec "$LLAMA_BIN" \
  -m /your/path/to/Qwen3.8-Flash-Next-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 \
  --port 8081 \
  --ctx-size 262144 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --gpu-layers 999 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  -fa on \
  --split-mode layer \
  -np 1 \
  --tensor-split 40,29,30,30,42,32 \
  --reasoning-format deepseek \
  --reasoning-budget 2048 \
  --cache-prompt \
  --metrics \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --repeat-penalty 1.0 \
  --fit off \
  --jinja \
  --ui-mcp-proxy \
  --override-kv qwen4exp.attention.indexer.top_k=int:4096 \
  --min-p 0.0
