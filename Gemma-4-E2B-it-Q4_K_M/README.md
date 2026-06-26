# Model Profile: Gemma-4-E2B-it-Q4_K_M

This directory contains the edge infrastructure profile and hardware configuration parameters for deploying [Gemma-4-E2B-it-Q4_K_M](https://huggingface.co/google/gemma-4-E2B-it) on the [NVIDIA Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/).

## Hardware Topology

The Jetson uses a low-power, unified memory layout capable of serving up small LLMs.

### Core Specs
- **Compute Module:** NVIDIA Jetson Orin Nano Super Developer Kit
- **Architecture:** 6-core ARM Cortex-A78AE v8.2 64-bit CPU
- **GPU:** 1024-core NVIDIA Ampere Architecture with 32 Tensor Cores
- **System Memory:** 8GB Shared LPDDR5 @ 2133 MHz (7.4GiB usable UMA block)
- **Storage:** PCIe Gen 3 NVMe SSD (Running at 8.0 GT/s un-throttled storage bandwidth)
- **OS:** Ubuntu 22.04.5 LTS (Kernel: 5.15.185-tegra aarch64)
- **Software Stack:** Linux for Tegra (L4T) R36.5.0 / CUDA Toolchain v12.6.68

## Infrastructure Lessons Learned

### 1. Exploiting the Unified Memory Architecture (UMA)
On a traditional PC, loading a model requires copying weights over the PCIe bus from system RAM into GPU VRAM. The Jetson eliminates this step entirely by using a unified memory architecture similar to the what Apple uses in the Mac Pro.

Because the CPU and GPU share the same LPDDR5 memory lines, the inference engine maps directly to the embedded GPU without copying weights across an external bus.

### 2. Preventing Edge Out-Of-Memory (OOM) via Memory Mapping
With only 7.4GiB of total usable system memory available for the OS, network stack, background utilities, and the LLM runtime, memory optimization is incredibly tight.

Memory-mapping the model file allows the system to read the weights lazily straight from the storage cache as needed by the GPU, keeping total system memory use safely around 2.7GB during idle states and leaving a safe buffer for handling LLM runtime contexts.

### 3. The NVMe Swap File Safety Net
Running even a small LLM on an 8GB board means that you're likely going to run up against memory limits pretty quickly. Make sure you have an adequate swap partition on a fast M.2 NVMe SSD is crucial to help with buffering context window expansions.

I'd recommend configuring a minimum of a 4GB swap file on an M.2 drive. **Don't store the swap on a micro-SD card! It WILL cause severe kernel panics!** (Ask me how I know...)

## Model Loading & Runtime Parameters

I used the following settings in `llama-cpp-python` to ensure safe memory limits and GPU offloading:

```python
# Core Inference Configurations
MAX_CONTEXT_TOKENS = 4096
N_BATCH = 512
MAX_TOKENS = 256
TEMPERATURE = 0.1

# -1 fully offloads the entire graph onto the Jetson's GPU
N_GPU_LAYERS = -1

# Reads weights directly from storage lines lazily
USE_MMAP = True
