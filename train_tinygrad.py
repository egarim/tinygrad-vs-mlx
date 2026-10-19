"""Train the char model with tinygrad. Fast path: TinyJit over the whole step (the pattern of examples/beautiful_mnist.py)."""
import time
import numpy as np
from tinygrad import Tensor, TinyJit, Context, Device, nn
import common as c

train, valid, V, plan, vplan = c.dataset()
P = {k: Tensor(v).contiguous().realize() for k, v in c.initial_params(V).items()}
HD = c.D // c.H


def ln(x, w, b): return x.layernorm(eps=1e-5) * w + b


def forward(p, idx):
    b, t = idx.shape
    x = p['tok'][idx] + p['pos'][:t]
    for i in range(c.L):
        g = lambda n: p[f'b{i}.{n}']
        h = ln(x, g('ln1_w'), g('ln1_b'))
        q, k, v = ((h @ g(n).T).reshape(b, t, c.H, HD).transpose(1, 2) for n in 'qkv')
        a = q.scaled_dot_product_attention(k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(b, t, c.D) @ g('o').T
        h = ln(x, g('ln2_w'), g('ln2_b'))
        x = x + (h @ g('up').T + g('up_b')).gelu() @ g('down').T + g('down_b')   # tinygrad's gelu is the tanh one
    return ln(x, p['ln_w'], p['ln_b']) @ p['head'].T + p['head_b']


def loss_fn(x, y): return forward(P, x).reshape(-1, V).sparse_categorical_crossentropy(y.reshape(-1))


opt = nn.optim.Adam(list(P.values()), lr=c.LR)


@TinyJit
@Context(TRAINING=1)
def step(x, y):
    opt.zero_grad()
    loss = loss_fn(x, y).backward()
    return loss.realize(*opt.schedule_step())


@TinyJit
def eval_loss(x, y): return loss_fn(x, y).realize()


curve, tm = [], c.Timer()
t0 = time.perf_counter()
for s in range(c.STEPS):
    x, y = c.get_batch(train, plan[s])
    lv = step(Tensor(x).realize(), Tensor(y).realize()).item()   # .item() synchronizes the GPU
    tm.lap()
    if s % c.LOG_EVERY == 0 or s == c.STEPS - 1: curve.append([s, round(lv, 4)]); print(s, round(lv, 4), flush=True)
wall = time.perf_counter() - t0
vl = float(np.mean([eval_loss(*(Tensor(a).realize() for a in c.get_batch(valid, vplan[i]))).item()
                    for i in range(c.VALID_BATCHES)]))
import importlib.metadata as md
import os
beam = os.getenv('BEAM')
c.save('tinygrad_beam' if beam else 'tinygrad', {'framework': f'tinygrad + BEAM={beam}' if beam else 'tinygrad', 'version': md.version('tinygrad'), 'device': Device.DEFAULT,
                    'first_step_s': round(tm.t[0], 3), 'ms_per_step': round(1000 * float(np.median(tm.t[c.TIMED_FROM:])), 2),
                    'wall_s': round(wall, 1), 'valid_loss': round(vl, 4), 'loss_curve': curve})
