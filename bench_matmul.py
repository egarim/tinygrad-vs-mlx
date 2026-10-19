"""Raw matrix multiply speed: C = A @ B, N x N, float32 and float16. TFLOPS = 2*N^3 / time."""
import json, os, time
import numpy as np

SIZES, REPS, WARM = [1024, 2048, 4096], 20, 3


def bench(run, sync):
    for _ in range(WARM): run(); sync()
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter(); run(); sync(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def mlx_case(n, dt):
    import mlx.core as mx
    d = {'float32': mx.float32, 'float16': mx.float16}[dt]
    a, b = mx.random.normal((n, n)).astype(d), mx.random.normal((n, n)).astype(d); mx.eval(a, b)
    out = {}
    return bench(lambda: out.__setitem__('c', a @ b), lambda: mx.eval(out['c']))


def torch_case(n, dt):
    import torch
    d = {'float32': torch.float32, 'float16': torch.float16}[dt]
    a, b = torch.randn(n, n, device='mps', dtype=d), torch.randn(n, n, device='mps', dtype=d)
    return bench(lambda: a @ b, torch.mps.synchronize)


def tinygrad_case(n, dt):
    from tinygrad import Tensor, Device, dtypes
    d = {'float32': dtypes.float32, 'float16': dtypes.float16}[dt]
    a, b = Tensor.randn(n, n, dtype=d).realize(), Tensor.randn(n, n, dtype=d).realize()
    return bench(lambda: (a @ b).realize(), Device[Device.DEFAULT].synchronize)


def tinygrad_jit_case(n, dt):
    from tinygrad import Tensor, TinyJit, Device, dtypes
    d = {'float32': dtypes.float32, 'float16': dtypes.float16}[dt]
    a, b = Tensor.randn(n, n, dtype=d).realize(), Tensor.randn(n, n, dtype=d).realize()
    mm = TinyJit(lambda x, y: (x @ y).realize())   # plan once, replay the captured kernel
    return bench(lambda: mm(a, b), Device[Device.DEFAULT].synchronize)


def numpy_case(n, dt):
    a, b = np.random.randn(n, n).astype(dt), np.random.randn(n, n).astype(dt)
    return bench(lambda: a @ b, lambda: None)


if __name__ == '__main__':
    import sys
    cases = [('MLX', mlx_case), ('PyTorch (MPS)', torch_case), ('tinygrad', tinygrad_case), ('tinygrad + TinyJit', tinygrad_jit_case),
             ('NumPy (CPU)', numpy_case)]
    if os.getenv('BEAM'): cases = [(f"tinygrad + TinyJit + BEAM={os.getenv('BEAM')}", tinygrad_jit_case)]
    out = 'results/matmul_beam.json' if os.getenv('BEAM') else 'results/matmul.json'
    rows = []
    for name, fn in cases:
        for dt in ['float32', 'float16']:
            if name == 'NumPy (CPU)' and dt == 'float16': continue   # NumPy has no fast fp16 matmul
            for n in SIZES:
                s = fn(n, dt); tf = 2 * n ** 3 / s / 1e12
                rows.append({'framework': name, 'dtype': dt, 'n': n, 'ms': round(s * 1000, 3), 'tflops': round(tf, 2)})
                print(f'{name:14} {dt:8} {n:5}  {s*1000:8.3f} ms  {tf:6.2f} TFLOPS', flush=True)
    os.makedirs('results', exist_ok=True)
    json.dump(rows, open(out, 'w'), indent=1)
