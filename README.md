# Local LLM Development & Infrastructure Engineering

This is my central repository for things I've learned about Local LLM deployments and architecture.

## Why This Repository Exists

There are a lot of introductory guides to get started with running Local LLMs. One thing I found is that many of them treat anything beyond using LMStudio as a simple "plug-and-play" task. The reality is that while anyone can download llama.cpp and compile it, getting it to run a model efficiently requires a lot of trial and error.

My goal is to share what I've learned about getting the most value out of consumer hardware to run models locally.

## Repository Architecture

Each subdirectory in this repository represents a different model deployment profile that I've tested in my home development lab. Each folder contains a shell script along with an explanation of why certain flags and settings were used.

### Active Configurations

| Deployment Directory | Primary Target Model | Focus Workload | Core Infrastructure Strategy |
| :--- | :--- | :--- | :--- |
| **[`/qwen3.6-35b-a3b`](./qwen3.6-35b-a3b/)** | Qwen3.6-35B-A3B-Q8_0 | Agentic Software Engineering | 8-bit KV Quantization, Intel P-Core Pinning |
| **[`/Gemma-4-E2B-it-Q4_K_M`](./Gemma-4-E2B-it-Q4_K_M/)** | Gemma-4-E2B-it-Q4_K_M | Edge Text Summarization | 4-bit Quantization, Resource Allocation |
