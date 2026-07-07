$MODEL_PATH = "\your\path\to\Qwen3.6-27B-Q4_K_M.gguf"
$BUILD_BIN   = "\your\path\to\llama.cpp\build\bin"

$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:Path

$env:GGML_CUDA_DISABLE_GRAPHS = "1"
$env:CUDA_DEVICE_ORDER         = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES      = "0,1"

& "$BUILD_BIN\llama-server.exe" `
  -m $MODEL_PATH `
  --host 0.0.0.0 `
  --port 8080 `
  --n-gpu-layers 99 `
  --split-mode layer `
  -c 32768 `
  --flash-attn on `
  --cache-prompt `
  --cache-ram 2048 `
  -t 8 `
  -tb 8 `
  -np 1 `
  --metrics