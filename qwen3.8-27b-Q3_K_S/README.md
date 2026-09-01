# Model Profile: Qwen3.8-27B-Q3_K_S

This directory contains the deployment configuration for running [Qwen3.8-27B-Q3_K_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) via llama.cpp server, benchmarked across two backends on the same physical machine.

## Hardware Topology

The host environment is a single laptop-class workstation, contrasting the CPU's integrated Arc GPU (unified system memory, no PCIe transfer cost) against a Thunderbolt-attached discrete GPU (real PCIe bus transfer cost per token).

### Core Specs

- **CPU:** Intel Core Ultra 7 366H (4 P-Cores / 8 E-Cores | 16 Threads)
- **Chassis:** Dell Pro 5 14 (P514260)
- **System RAM:** 32GB

### Physical PCIe Routing Layout

| Cluster Mapping | Hardware Target | VRAM | Physical Connection Type |
| :--- | :--- | :--- | :--- |
| **Vulkan0** | Intel Arc iGPU (integrated) | Shared, up to 32GB system RAM | On-die, unified memory (no PCIe transfer) |
| **CUDA0** | NVIDIA GeForce RTX 5050 | 8GB dedicated | [Negent Dockr](https://negent.tech/pages/egpu-dock) Thunderbolt 4 eGPU dock (40Gbps) |

## Performance

The table below contrasts the Vulkan/iGPU config (`--n-gpu-layers 99`, self-speculative MTP decoding) against the tuned CUDA/eGPU config (`-ngl 28`, no MTP — every MTP variant tested hard-locked this card/dock/driver combination).

| Metric | Vulkan (Intel iGPU) | CUDA (RTX 5050 eGPU) | Performance Delta / Realized Impact |
| :--- | :--- | :--- | :--- |
| **Prompt Processing** *(Prefill)* | 29.57 tokens/sec | **67.70 tokens/sec** | **~129% Acceleration** — CUDA has working kernels for this model's DeltaNet/SSM layers that Vulkan lacks, which otherwise silently fall back to CPU |
| **Text Generation** *(Decode)* | **6.01 tokens/sec** | 5.26 tokens/sec | **~12% slower on CUDA** — every generated token pays a real PCIe/Thunderbolt boundary-crossing cost that the iGPU's unified memory never incurs |
| **GPU Layer Offload** | 99 (full — no partial-offload needed) | 28 of 65 (partial, contiguous split; VRAM-limited to 8GB) | N/A |
| **VRAM Utilization** | N/A (shared system RAM) | 6,285MiB / 8,151MiB (77%) | Pushing to `-ngl 29` regressed generation and tripped a "failed to fit params" warning — 28 is the validated ceiling |

**Takeaway:** these two backends now serve different profiles for this model rather than one clearly beating the other. Use Vulkan/iGPU for generation-heavy sessions; use CUDA/eGPU for prompt-eval-heavy work (large-context ingestion, RAG, long documents) where faster prefill matters more.
