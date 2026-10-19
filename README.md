# tinygrad vs MLX vs PyTorch on Apple Silicon

How does [tinygrad](https://github.com/tinygrad/tinygrad) (Tiny Corp's open-source framework) compare with what I already
use on my Mac: Apple's [MLX](https://github.com/ml-explore/mlx) and PyTorch on the Apple GPU (MPS)?

Two tests, on the same machine, with each framework's own fast path:

1. **Raw matrix multiply**, N × N, float32 and float16.
2. **Training the same model**: a 1.84M-parameter character-level transformer on *Don Quijote* (Spanish), with
   **identical starting weights, identical batches in the same order, identical hyperparameters**. All three start at
   exactly the same loss (4.4469) and end within 0.004 of each other, so the speed comparison is apples to apples.

Machine: Mac Studio, Apple M1 Max, 64 GB, macOS 26.3.1. Python 3.13.13, tinygrad 0.14.0, MLX 0.32.4, PyTorch 2.14.1,
NumPy 2.5.4. Run on 19 October 2026.

## Results

### Training (600 steps, batch 64 × 128 characters)

| Framework | ms / step ↓ | first step (compile) ↓ | total 600 steps ↓ | validation loss |
|---|---|---|---|---|
| **MLX** (`mx.compile`) | **34.6** | **0.9 s** | **21.9 s** | 1.647 |
| tinygrad + BEAM=2 (`TinyJit`, kernel search cached) | 41.8 | 6.5 s | 33.4 s | 1.646 |
| PyTorch MPS (eager, fused SDPA) | 45.2 | 3.4 s | 30.5 s | 1.647 |
| tinygrad (`TinyJit`) | 56.7 | 3.5 s | 39.6 s | 1.646 |

ms/step is the median of steps 100–600 (after compilation). The **first** BEAM=2 run spent **290 s** in its first
step searching for the fastest kernels (317 s total); tinygrad caches the search on disk, so later runs start in 6.5 s.

Loss at each checkpoint (same in all three to about 0.003):

| step | 0 | 100 | 200 | 300 | 400 | 500 | 599 |
|---|---|---|---|---|---|---|---|
| loss | 4.447 | 2.397 | 2.297 | 2.094 | 1.923 | 1.786 | 1.638 |

### Matrix multiply (TFLOPS ↑, median of 20 runs)

| Framework | fp32 1024 | fp32 2048 | fp32 4096 | fp16 1024 | fp16 2048 | fp16 4096 |
|---|---|---|---|---|---|---|
| MLX | 2.72 | 7.06 | 8.28 | 3.78 | 5.85 | 9.23 |
| PyTorch (MPS) | 2.70 | 7.12 | 8.40 | 3.73 | 6.25 | 9.18 |
| tinygrad (plain) | 0.84 | 3.59 | 5.52 | 0.82 | 4.40 | 6.66 |
| tinygrad + TinyJit | 2.88 | 5.63 | 5.77 | 3.64 | 8.03 | 9.46 |
| tinygrad + TinyJit + BEAM=2 | 2.60 | **7.60** | **8.59** | 2.62 | 6.46 | **9.54** |
| NumPy (CPU, Accelerate) | 2.10 | 1.85 | 1.66 | | | |

## What it says

- **Same math, same answers.** With identical weights and batches, the three frameworks agree to 3–4 decimals. The one
  setting that broke this: **MLX's `Adam` has `bias_correction=False` by default**; PyTorch and tinygrad always correct.
  With MLX's default, the same run ends at loss 2.32 instead of 1.64. `train_mlx.py` sets it to `True`.
- **MLX is the fastest for training** on this Mac: 17% faster per step than tinygrad with BEAM, 23% faster than PyTorch,
  and almost instant to compile.
- **tinygrad's plain mode pays Python overhead** on every call (scheduling the graph again). `TinyJit` removes it: at
  1024 × 1024 it goes from 0.84 to 2.88 TFLOPS.
- **With BEAM search, tinygrad's own generated kernels match Apple's.** At 4096 it reaches 8.59 TFLOPS (fp32) and 9.54
  (fp16), slightly ahead of MLX and PyTorch. The price is search time once (minutes), then it's cached.
- PyTorch here is plain eager mode; `torch.compile` was not tried.

## Run it

```sh
python3.13 -m venv .venv && .venv/bin/pip install "tinygrad==0.14.0" mlx torch numpy
.venv/bin/python train_mlx.py
.venv/bin/python train_torch.py
.venv/bin/python train_tinygrad.py
BEAM=2 .venv/bin/python train_tinygrad.py     # first run searches kernels (minutes), later runs use the cache
.venv/bin/python bench_matmul.py
BEAM=2 .venv/bin/python bench_matmul.py
```

The text downloads from Project Gutenberg on first run into `data/` (not in git). Every run writes `results/*.json`.

| File | What |
|---|---|
| `common.py` | data, the batch plan, hyperparameters, and the shared starting weights (generated once in NumPy) |
| `train_mlx.py`, `train_torch.py`, `train_tinygrad.py` | the same model and loop in each framework |
| `bench_matmul.py` | the matrix multiply benchmark |
| `results/` | the numbers above |

## How much to trust this

- **One machine, one run each.** The matmul numbers move by several percent between runs (look at fp16 2048).
  Differences under ~10% are noise.
- **A small model.** 1.84M parameters favours low per-step overhead; bigger models shift the balance toward raw kernel
  speed, where tinygrad + BEAM is competitive.
- **Each framework's documented fast path**, not hand-tuned code. Other settings could change the order.
