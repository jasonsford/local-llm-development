# Model Profile: Qwen3.6-27B-Q4_K_M

This directory contains the deployment configuration for running [Qwen3.6-27B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) in LM Studio for single-agent engineering tasks with [Pi](https://pi.dev/).

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

A few metrics gathered from LM Studio during active multi-turn agent sessions using Pi:

* **Prompt Processing (Prefill Speed):** **439 to 700 tokens/sec**
* **Text Generation (Eval Speed):** **16.7 to 17.6 tokens/sec**
* **Time-to-First-Token (TTFT):** **3.6s**

## Model Serving Configuration

### 1. LM Studio Settings
- **Context Length:** `32768` (32k tokens)
- **Parallel Requests:** `1`
- **Hardware Acceleration:** `CUDA with Max GPU Offload`
- **Flash Attention:** `Enabled`
- **Temperature:** `0.6`

### 2. Pi Agent Harness Configuration (`~/.pi/agent/models.json`)
```json
{
  "providers": {
    "lm-studio": {
      "baseUrl": "http://localhost:1234/v1",
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
