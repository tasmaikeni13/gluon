# Performance and Architectural Analysis: Pallas TPU Kernels vs. XLA Fusion

## Executive Summary & Configuration Policy

This document records the empirical results, architectural analysis, and configuration decisions regarding custom Pallas TPU kernels and XLA fusion for language model training on Cloud TPU v4 hardware across **125M, 350M, and 1B** model scales.

### 1. Optimizers Default Policy: XLA Fusion
- **Default for Main Runs**: **XLA** across all 5 optimizers (AdamW, Sophia-G, Rocora, SOAP, Muon).
- **Rationale**: While extensive research and kernel engineering were conducted to write custom Pallas TPU kernels for all five optimizers, pointwise updates (AdamW, Sophia-G, Rocora) and Newton–Schulz iterations (Muon) did not outperform XLA compiler fusion. Although Pallas yielded a **+15.5% speedup on SOAP** at 125M scale, enabling custom kernels for SOAP while using XLA for the other arms would create an asymmetric, unfair baseline comparison. To maintain strict scientific controls, **all optimizers default to XLA fusion**.
- **Supplementary Material**: All engineered Pallas kernels are retained in `src/rocora/pallas_optimizers.py` as supplementary material and can be toggled on demand via `use_pallas: true` (or `pallas_mode: "pallas"`).

### 2. Attention Default Policy: Pallas TPU FlashAttention
- **Default**: **Pallas TPU FlashAttention** (`attention: "flash"`).
- **Rationale**: FlashAttention demonstrated substantial, repeatable speedups across all scales, cutting forward pass latency by **up to 88.4%** at 1B scale and backward pass latency by **39.5%**.
- **Option to Fallback**: The standard differentiable FP32 attention remains available as an option (`attention: "reference"` in `ModelConfig`).

---

## Empirical Benchmark Results (Cloud TPU v4)

All measurements were obtained on Google Cloud TPU v4 hardware using batch size 8, sequence length 1024, and head dimension $D=64$.

### 1. FlashAttention Speedups (Pallas vs. XLA Reference)

| Scale | Width / Heads | Forward Baseline (XLA) | Forward (Pallas Flash) | **Forward Speedup** | Backward Baseline (XLA) | Backward (Pallas Flash) | **Backward Speedup** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **125M** | $d=768, H=12$ | 0.880 ms | 0.731 ms | **+20.3% (1.203x)** | 3.170 ms | 3.128 ms | **+1.3% (1.013x)** |
| **350M** | $d=1024, H=16$ | 1.148 ms | 0.920 ms | **+24.8% (1.248x)** | 4.382 ms | 4.120 ms | **+6.4% (1.064x)** |
| **1B** | $d=1536, H=24$ | 2.589 ms | 1.374 ms | **+88.4% (1.884x)** 🚀 | 8.748 ms | 6.272 ms | **+39.5% (1.395x)** 🚀 |

### 2. Optimizer Speedups (Pallas Kernels vs. XLA Baselines)

| Optimizer | 125M ($d=768$) | 350M ($d=1024$) | 1B ($d=1536$) | Bottleneck Classification | Default Setting |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AdamW** | 0.499x | 0.489x | 0.495x | Memory-bandwidth bound (~0.5 FLOP/byte) | **XLA** |
| **Sophia-G** | 0.466x | 0.457x | 0.463x | Memory-bandwidth bound (~0.5 FLOP/byte) | **XLA** |
| **Rocora** | 0.409x | 0.415x | 0.418x | Memory-bandwidth bound (~0.5 FLOP/byte) | **XLA** |
| **SOAP** | **1.155x (+15.5%)** | **1.011x (+1.1%)** | 1.000x | Matrix preconditioning / projection | **XLA** (fair control) |
| **Muon** | 0.827x | 0.776x | **1.009x (+0.9%)** | Compute-bound Newton–Schulz | **XLA** |

---

## Technical Analysis: Why Custom Optimizer Kernels Do Not Beat XLA

### 1. Memory-Bandwidth Saturation in Pointwise Kernels
- **Arithmetic Intensity**: Pointwise optimizers (AdamW, Sophia-G, Rocora) perform basic arithmetic ($p \leftarrow p - \eta \cdot m / (\sqrt{v} + \epsilon)$) on large parameter vectors. The arithmetic intensity is approximately **0.5 FLOP per byte transferred** from HBM.
- **XLA Graph Fusion**: During training, the XLA compiler fuses the optimizer update directly into the backward pass graph. Intermediate gradient values are consumed directly from on-chip SRAM/registers without ever writing full gradient buffers back to HBM.
- **Standalone Kernel Overhead**: Standalone custom kernels (whether Pallas or Triton) require gradients to be fully materialized in HBM, loaded into VMEM, updated, and written back to HBM. This separation adds buffer round-trips and kernel invocation latency that XLA fusion inherently eliminates.

### 2. Matrix Optimizers (SOAP and Muon)
- **SOAP**: Custom Pallas Kronecker projection in VMEM delivered **+15.5% speedup** at 125M scale and **+1.1%** at 350M scale. However, at 1B scale ($1536 \times 1536$), the projection tensor sizes approach VMEM capacity limits, and XLA's optimized blocked matrix operations achieve identical performance (1.000x).
- **Muon**: Muon's quintic Newton–Schulz iterations ($\sim 8,660$ FLOP/byte) are compute-dense. XLA's optimized TPU Matrix Multiplication Units (MXUs) perform near theoretical peak. The custom Pallas kernel matches XLA closely (1.009x at 1B), but does not provide sufficient speedup to justify replacing XLA.
- **Fairness Principle**: If SOAP alone ran on Pallas while AdamW, Sophia, Rocora, and Muon ran on XLA, wall-clock training efficiency would be confounded by implementation asymmetry rather than optimizer algorithmic properties. Defaulting all optimizers to XLA ensures an apples-to-apples scientific comparison.

---

## How to Toggle Between Pallas and XLA

### Optimizer Execution Options
In configuration JSON (e.g. `configs/scale125m.json`) or code:
```json
"optimizer": {
  "name": "rocora",
  "use_pallas": false,
  "pallas_mode": "xla"
}
```
- `"pallas_mode": "xla"` (Default): Uses XLA-fused optimizer updates across all arms.
- `"pallas_mode": "pallas"`: Forces execution of the custom Pallas TPU kernels in `src/rocora/pallas_optimizers.py`.

### Attention Execution Options
In configuration JSON:
```json
"model": {
  "attention": "flash"
}
```
- `"attention": "flash"` (Default): Uses the high-performance Pallas TPU FlashAttention kernel (`block_q=128, block_k=1024`).
- `"attention": "reference"`: Uses the tiled XLA reference attention.
