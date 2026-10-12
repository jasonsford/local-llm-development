# Model Profile: Qwen3.8-Flash-Next-UD-IQ4_XS

This directory contains the deployment configuration for running **Qwen3.8-Flash-Next-UD-IQ4_XS** across a 6-GPU workstation with a 256k context window for long-turn engineering and agentic workflows.

## Hardware Topology

The host environment utilizes an asymmetric multi-GPU cluster balancing Ampere, Ada Lovelace, and Blackwell GPUs over primary PCIe slots and high-speed OCuLink storage adapters.

### Core Specs

- **CPU:** Intel Core i7-14700K (8 P-Cores / 12 E-Cores | 28 Threads)
- **Motherboard:** MSI PRO Z790-P WIFI (MS-7E06)
- **System RAM:** 96GB DDR5
- **Total Dedicated VRAM Pool:** 106.4 GB (spread across 6 GPUs)

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

## Performance

Captured directly from `llama-server` during sustained multi-token benchmarks:

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Prompt Processing (Prefill Speed)** | **522.50 tokens/sec** | 1,002 tokens ingested in 1.91s (`ubatch: 1024`, `batch: 2048`) |
| **Text Generation (Eval Speed)** | **40.50 to 40.96 tokens/sec** | Steady across 1,500+ tokens without pipeline stalls |
| **Allocated Context Depth** | **262,144 tokens (256k)** | Fully resident in VRAM using `q4_0` / `q4_0` KV cache |
| **CUDA Graph Stability** | **1,463 reused / 0 reallocs** | 100% graph reuse efficiency with zero memory thrashing |

## Model Artifacts & Launch Configuration

- **Execution Script:** Refer to [`Qwen3.8-Flash-Next-UD-IQ4_XS.sh`](./Qwen3.8-Flash-Next-UD-IQ4_XS.sh) in this directory for the complete server parameters.
