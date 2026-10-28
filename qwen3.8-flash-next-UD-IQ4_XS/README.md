# Model Profile: Qwen3.8-Flash-Next-UD-IQ4_XS

This directory contains the deployment configuration for running **Qwen3.8-Flash-Next-UD-IQ4_XS** across a 6-GPU workstation with a 256k context window for long-turn engineering and agentic workflows.

## Hardware Topology

The host environment utilizes an asymmetric multi-GPU cluster balancing Ampere, Ada Lovelace, and Blackwell GPUs over primary PCIe slots and high-speed OCuLink storage adapters.

### Core Specs

- **CPU:** Intel Core i7-14700K (8 P-Cores / 12 E-Cores | 28 Threads)
- **Motherboard:** MSI PRO Z790-P WIFI (MS-7E06)
- **System RAM:** 96GB DDR5
- **Total Dedicated VRAM Pool:** 104 GB (spread across 6 GPUs)

### Physical PCIe & OCuLink Routing Layout

| Cluster Mapping | Hardware Target | Architecture | VRAM | Physical Connection Type |
| :--- | :--- | :--- | :--- | :--- |
| **CUDA 0** | NVIDIA GeForce RTX 3080 20GB Mod | Ampere | 20GB | Direct PCIe Slot (Primary CPU Link, x16) |
| **CUDA 1** | NVIDIA GeForce RTX 4060 Ti | Ada Lovelace | 16GB | Direct PCIe Slot (Chipset Link, x8) |
| **CUDA 2** | NVIDIA GeForce RTX 4060 Ti | Ada Lovelace | 16GB | OCuLink via M.2 Storage Slot (`M2_1`) |
| **CUDA 3** | NVIDIA GeForce RTX 5060 Ti | Blackwell | 16GB | Direct PCIe Slot (Chipset Link, x8) |
| **CUDA 4** | NVIDIA GeForce RTX 3080 20GB Mod | Ampere | 20GB | OCuLink via M.2 Storage Slot (`M2_3`) |
| **CUDA 5** | NVIDIA GeForce RTX 4060 Ti | Ada Lovelace | 16GB | OCuLink via PCIe Add-In Card |

## Lessons Learned & Tuning Insights

### 1. Scaling Ingestion Speed (Prefill Starvation)
Initially, running `--batch-size 1024` and `--ubatch-size 1024` starved the tensor cores during prompt evaluation, dragging ingestion down to **~43.1 tokens/sec** and creating an unacceptable ~23-second lag on a simple 1k-token prompt. 

Expanding to `--batch-size 2048` with `--ubatch-size 1024` unlocked tensor core saturation during prefill without demanding excessive scratchpad memory, rocketing prompt evaluation to **522.5 tokens/sec (a 12.1x speedup)**.

### 2. Guarding the 20GB VRAM Boundary
Attempting to push `--ubatch-size` all the way to 2048 triggered an out-of-memory error on the 20GB RTX 3080 cards (`20GB needed, only 19.6GB available`). In `llama.cpp`, the scratch buffer and compute graph workspace scale directly with micro-batch dimensions. 

By holding `--ubatch-size` at 1024 and rebalancing `--tensor-split` from `43,29,30,30,43,29` to `40,29,30,30,42,32`, we shift roughly 1.2 GB of static layer weights off CUDA 0 onto CUDA 5. This maintains the full 256k context slot while securing a comfortable 1.7+ GB headroom cushion across all cards.

### 3. Sparse Indexer Override for Long-Context Coherence
Qwen3.8 Flash Next implements a hybrid attention structure pairing Gated DeltaNet with Qwen Sparse Attention (QSA). At extended context lengths beyond 128k, the stock token retrieval ceiling can degrade reasoning chains. Passing:
```bash
--override-kv qwen4exp.attention.indexer.top_k=int:4096
```
expands the sparse attention indexer budget to 4096 tokens, ensuring robust retrieval across deep multi-turn sessions with virtually zero latency penalty.

### 4. Sampling Constraints in Thinking Mode
When running native DeepSeek-style reasoning (`--reasoning-format deepseek`), strict repetition penalties and tight `top-p` limits prematurely suppress chain-of-thought exploration. Setting `--temp 1.0`, `--top-p 0.95`, `--min-p 0.0`, and resetting `--repeat-penalty 1.0` avoids repetitive degradation loops while allowing thinking traces to explore full solution trees cleanly.

### 5. GA102 Firmware & PCIe Optimizations (RTX 3080 20GB Mods)
Because the two RTX 3080 20GB cards host ~40% of the model's active weights (`--tensor-split 40,29,30,30,42,32`), optimizing their compute issue rate and host memory aperture unlocked significant throughput gains without altering VRAM capacity:
- **FP16/BF16 Issue Rate Unlock ([Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)):** Loaded as an early-initramfs kernel module, this bypasses the artificial GeForce instruction-issue clamp on the GA102 cores by reprogramming the `SS0`/`SS1` rate selectors prior to driver attachment. This boosted dense GEMM and FlashAttention throughput across CUDA 0 and CUDA 4, driving steady-state prefill from **522.50 t/s up to 702.92 t/s (+34.5%)**.
- **Dynamic 32 GiB Resizable BAR ([noon-at-cgn/3080-20gb-rebar](https://github.com/noon-at-cgn/3080-20gb-rebar)):** Memory-modded 20GB cards are locked to a 256 MiB BAR1 aperture. Running `rebar-arm` at early boot overrides the PCIe configuration space, expanding BAR1 to **32 GiB (32,768 MiB)** on both cards to eliminate host staging bounce buffers and map the full framebuffer directly into host physical address space.

## Performance

Captured directly from `llama-server` logs during sustained 256k-context inference runs comparing stock baseline vs. unlocked GA102 configurations:

| Metric | Stock Baseline | Unlocked (ReBAR + Zagitta Kmod) | Notes |
| :--- | :--- | :--- | :--- |
| **Prompt Processing (Prefill Speed)** | 522.50 tokens/sec | **702.92 tokens/sec** *(+34.5%)* | 1,019 tokens in 1.45s warm (`ubatch 1024`, `batch 2048`); 339.46 t/s cold graph compilation |
| **Text Generation (Eval Speed)** | 40.50 tokens/sec | **39.65 tokens/sec** | Bound by GDDR6X memory bus bandwidth; flat across runs within margin of error |
| **Allocated Context Depth** | 262,144 tokens (256k) | **262,144 tokens (256k)** | Fully resident in VRAM using `q4_0` / `q4_0` KV cache quantization |
| **BAR1 Aperture (CUDA 0 & 4)** | 256 MiB | **32,768 MiB (32 GiB)** | Expanded via [`noon-at-cgn/3080-20gb-rebar`](https://github.com/noon-at-cgn/3080-20gb-rebar) |
| **Tensor Core Issue Rate** | Stock GeForce Clamp | **Full Rate (Unlocked)** | Enabled via [`Zagitta/gpu-unlock`](https://github.com/Zagitta/gpu-unlock) early initramfs module |
| **CUDA Graph Stability** | 1,463 reused / 0 reallocs | **1,107 reused / 0 reallocs** | 100% graph reuse efficiency with zero memory thrashing |

## Model Artifacts & Launch Configuration

- **Execution Script:** Refer to [`Qwen3.8-Flash-Next-UD-IQ4_XS.sh`](./Qwen3.8-Flash-Next-UD-IQ4_XS.sh) in this directory for the complete server parameters.
