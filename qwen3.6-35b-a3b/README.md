# Model Profile: Qwen3.6-35B-A3B-Q8_0

This directory contains the deployment configuration for running Qwen3.6-35B-A3B-Q8_0 for long-context, single-agent engineering tasks.

## Hardware Topology

The host environment uses a hybrid layout to maximize lane efficiency without multi-GPU bandwidth throttling.

### Core Specs

- **CPU:** Intel Core i7-14700K (8 P-Cores / 12 E-Cores | 28 Threads)
- **Motherboard:** MSI PRO Z790-P WIFI (MS-7E06)
- **System RAM:** 32GB Dual-Channel G.Skill DDR5 @ 4800 MT/s

### Physical PCIe Routing Layout

| Cluster Mapping | Hardware Target | VRAM | Physical Connection Type |
| :--- | :--- | :--- | :--- |
| **CUDA 0** | NVIDIA GeForce RTX 4060 Ti | 16GB | Direct PCIe Slot (Primary CPU Link) |
| **CUDA 1** | NVIDIA GeForce RTX 4060 Ti | 16GB | Oculink via Primary M.2 Storage Slot |
| **CUDA 2** *Omitted* | NVIDIA GeForce RTX 4070 Super | 12GB | Oculink via Secondary M.2 Storage Slot |
| **CUDA 3** | NVIDIA GeForce RTX 5060 Ti | 16GB | Direct PCIe Slot (Secondary Chipset Link) |
| **CUDA 4** | NVIDIA GeForce RTX 4060 Ti | 16GB | Oculink via PCIe Add-In Card |

## Lessons Learned

### 1. Bypassing the Chipset Bifurcation Wall
Many consumer desktop chipsets don't support PCIe slot bifurcation ($x4/x4/x4/x4$). I initially attempted to use a [PCIe to Oculink slot splitter](https://www.amazon.com/dp/B0F291T2L4), but the board failed to enumerate all but one connected device no matter how the PCIe lanes were configured in BIOS.

To overcome this, I used the motherboard's high-speed internal M.2 NVMe slots with [M.2 to Oculink Host Adapters](https://www.amazon.com/dp/B0DPJT7D3G). Because these slots carry 4 dedicated PCIe lanes straight from the CPU/Chipset, they provide isolated, un-throttled $x4$ bandwidth out to my [external GPU riser docks](https://www.amazon.com/dp/B0F9FBN5P5).

### 2. The Asymmetrical VRAM Trap (Why the 4070 is Excluded)
You'll notice that the shell script ignores the RTX 4070 with `export CUDA_VISIBLE_DEVICES=0,1,3,4`. One of the things I learned through trial and error is that in a layer-sliced or tensor-parallel cluster, the cumulative capacity is bottlenecked by the card with the lowest amount of VRAM. Including a 12GB card alongside four 16GB cards forces llama.cpp to clip memory allocation ceilings across all the active lanes, so dropping the 12GB card provides an identical pool of four 16GB cards ($4 \times 16\text{GB} = 64\text{GB}$ VRAM). This also allows me to use the 4070 card to run other, smaller models in parallel for other tasks.

### 3. The MoE Bandwidth Advantage
The MoE (Mixture of Experts) nature of the 35B-A3B model means that while it has a total size of 35 billion parameters, it only triggers 3 billion parameters per token. This provides an incredible architectural advantage for running the model on a split-GPU consumer platform because only a fraction of the weights travel across the bus per forward pass.

## Performance

A few metrics gathered from `llama-server` during active multi-turn agent sessions using OpenCode:

* **Prompt Processing (Prefill Speed):** **3,400 to 4,100+ tokens/sec**
* **Text Generation (Eval Speed):** **51 to 62 tokens/sec**
* **Time-to-First-Token (TTFT):** **< 350ms**
