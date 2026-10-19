"""Shared setup so every framework trains on exactly the same data, batches and settings."""
import json, os, re, time, urllib.request
import numpy as np

B, T, D, L, H = 64, 128, 192, 4, 4      # batch, context, width, layers, heads
STEPS, LR = 600, 1e-3                   # Adam, constant learning rate, no weight decay
LOG_EVERY = 100
TIMED_FROM = 100                         # ms/step = median over steps TIMED_FROM..STEPS (after compile/warm-up)
VALID_BATCHES = 20

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'quijote_es.txt')
URL = 'https://www.gutenberg.org/cache/epub/2000/pg2000.txt'   # Don Quijote (Spanish), public domain


def load_text():
    if not os.path.exists(DATA):
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    raw = open(DATA, encoding='utf-8').read().replace('\r', '')
    raw = raw.split('*** START OF THE PROJECT GUTENBERG EBOOK')[1].split('\n', 1)[1]
    raw = raw.split('*** END OF THE PROJECT GUTENBERG EBOOK')[0]
    return re.sub(r'[ \t]+', ' ', re.sub(r'\n{3,}', '\n\n', raw)).strip()


def dataset():
    """Encoded train/valid arrays, vocab size, the batch index plan and the fixed validation batches."""
    text = load_text()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    cut = int(len(data) * 0.9); train, valid = data[:cut], data[cut:]
    rng = np.random.default_rng(0)
    plan = rng.integers(0, len(train) - T - 1, size=(STEPS, B))            # identical batches for every framework
    vrng = np.random.default_rng(123)
    vplan = vrng.integers(0, len(valid) - T - 1, size=(VALID_BATCHES, B))
    return train, valid, len(chars), plan, vplan


def get_batch(src, idx):
    x = np.stack([src[i:i + T] for i in idx]); y = np.stack([src[i + 1:i + T + 1] for i in idx])
    return x, y


def init_weights(shape, seed):
    """Normal(0, 0.02), generated in numpy so all three frameworks start from the SAME numbers."""
    return (np.random.default_rng(seed).standard_normal(shape) * 0.02).astype(np.float32)


def save(name, result):
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    json.dump(result, open(os.path.join(ROOT, 'results', f'{name}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'loss_curve'}, indent=1))


class Timer:
    def __init__(self): self.t = []; self.last = time.perf_counter()
    def lap(self):
        now = time.perf_counter(); self.t.append(now - self.last); self.last = now


def param_spec(V):
    """Every parameter (name -> shape, kind). Linear weights are (out, in) in all three frameworks."""
    spec = {'tok': ((V, D), 'w'), 'pos': ((T, D), 'w')}
    for i in range(L):
        for n in 'qkvo': spec[f'b{i}.{n}'] = ((D, D), 'w')
        spec[f'b{i}.up'] = ((4 * D, D), 'w'); spec[f'b{i}.up_b'] = ((4 * D,), 'zero')
        spec[f'b{i}.down'] = ((D, 4 * D), 'w'); spec[f'b{i}.down_b'] = ((D,), 'zero')
        for ln in ('ln1', 'ln2'): spec[f'b{i}.{ln}_w'] = ((D,), 'one'); spec[f'b{i}.{ln}_b'] = ((D,), 'zero')
    spec['ln_w'] = ((D,), 'one'); spec['ln_b'] = ((D,), 'zero')
    spec['head'] = ((V, D), 'w'); spec['head_b'] = ((V,), 'zero')
    return spec


def initial_params(V):
    import zlib
    out = {}
    for name, (shape, kind) in param_spec(V).items():
        out[name] = (init_weights(shape, zlib.crc32(name.encode())) if kind == 'w'
                     else np.ones(shape, np.float32) if kind == 'one' else np.zeros(shape, np.float32))
    return out
