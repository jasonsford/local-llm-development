# Model Profile: Qwen3.6-27B-Q4_K_M

This directory contains the deployment configuration for running [Qwen3.6-27B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) for single-agent engineering tasks with [Pi](https://pi.dev/).

## Hardware Topology

The host environment uses a dual-GPU layout to provide a local coding agent using budget consumer video cards.

### Core Specs

- **CPU:** Intel Core i7-14700K (8 P-Cores / 12 E-Cores | 28 Threads)
- **Motherboard:** ASUS PRIME B760M-A AX
- **System RAM:** 64GB Dual-Channel DDR5 @ 4800 MT/s

### Physical PCIe Routing Layout

| Cluster Mapping | Hardware Target | VRAM | Physical Connection Type |
| :--- | :--- | :--- | :--- |
| **CUDA 0** | NVIDIA GeForce RTX 3060 | 12GB | Direct PCIe Slot |
| **CUDA 1** | NVIDIA GeForce RTX 3060 | 12GB | Direct PCIe Slot |

## Performance

The table below contrasts the metrics gathered using the wrapped LM Studio environment against a native, production-compiled `llama.cpp` server backend on Windows utilizing a CUDA 13.3 + Ninja build toolchain.

| Metric | LM Studio (Bundled Engine) | Native `llama.cpp` (CUDA 13.3 + Ninja) | Performance Delta / Realized Impact |
| :--- | :--- | :--- | :--- |
| **Prompt Processing** *(Prefill)* | 439 to 700 tokens/sec | **802 to 850 tokens/sec** | **~21% to 82% Acceleration** |
| **Text Generation** *(Decode)* | **16.7 to 17.6 tokens/sec** | 15.7 to 16.5 tokens/sec | ~6% Variance (Bound by physical VRAM bus bandwidth) |
| **Time-to-First-Token** *(TTFT)* | 3.6s *(Baseline Context)* | **3.06s** *(~1.5k tokens)*<br>**5.57s** *(~4.2k tokens)*<br>**12.52s** *(~9.6k tokens)* | **~15% Faster Ingestion** on cold turns;<br>**Near-Instant (0s)** on context-cached turns. |

## Model Serving Configuration

### 1. LM Studio Settings
- **Context Length:** `32768` (32k tokens)
- **Parallel Requests:** `1`
- **Hardware Acceleration:** `CUDA with Max GPU Offload`
- **Flash Attention:** `Enabled`
- **Temperature:** `0.6`

### 2. Native llama.cpp Configuration
For zero-overhead execution, a custom-compiled `llama-server.exe` instance is launched directly out of the build binaries directory. This layer utilizes upstream architectural optimizations and explicit execution thread bounds to bypass Intel efficiency core throttling.

The automation script driving this optimized orchestration is stored in this repository at:
[llama_cpp_qwen3.6_27b_q4.ps1](https://github.com/jasonsford/local-llm-development/blob/main/qwen3.6-27b-Q4_K_M/llama_cpp_qwen3.6_27b_q4.ps1)

### 3. Pi Agent Harness Configuration (`~/.pi/agent/models.json`)
Note - You'll need to update the port on localhost depending on if you're running this using LM Studio (port 1234) vs llama-server (port 8080).
```json
{
  "providers": {
    "lm-studio": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "lm-studio",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "qwen/qwen3.6-27b",
          "name": "Qwen 3.6 27B",
          "reasoning": true,
		  "contextWindow": 32768
        }
      ]
    }
  }
}
