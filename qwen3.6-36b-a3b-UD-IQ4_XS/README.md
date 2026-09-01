# Model Profile: Qwen3.6-35B-A3B-UD-IQ4_XS

This directory contains the deployment configuration for running [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) via llama.cpp, benchmarked across two backends on the same physical machine.

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

The table below contrasts the Vulkan/iGPU config (`--cpu-moe`, `-ngl auto`, `-ub 128`) against the final tuned CUDA/eGPU config (`--n-cpu-moe 32`, `-ub 384`, `--load-mode none`). Unlike the dense model, CUDA wins decisively on both metrics here — MoE's sparse `--n-cpu-moe` split lets the GPU handle only the small, high-value slice of expert weights instead of a large contiguous layer block.

| Metric | Vulkan (Intel iGPU) | CUDA (RTX 5050 eGPU) | Performance Delta / Realized Impact |
| :--- | :--- | :--- | :--- |
| **Prompt Processing** *(Prefill)* | 57.74 tokens/sec | **76.13 tokens/sec** | **~32% Acceleration** |
| **Text Generation** *(Decode)* | 10.83 tokens/sec | **~20.7–21.0 tokens/sec** | **~91% Acceleration** — CUDA's working DeltaNet/SSM kernels plus GPU-resident experts for the last 8 layers roughly double throughput |
| **VRAM Utilization** | N/A (shared system RAM) | ~5,863MiB / 8,151MiB (measured at `--n-cpu-moe 32`, `-ub 128`; not re-captured at the final `-ub 384`) | `--n-cpu-moe 31` (one layer more on GPU) loaded but performed *worse* on both metrics — 32 is the validated sweet spot, not just the safe floor |
