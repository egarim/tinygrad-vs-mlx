"""Train the char model with MLX (Apple's framework). Fast path: mx.compile over the whole step."""
import time
from functools import partial
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import common as c

train, valid, V, plan, vplan = c.dataset()
P = {k: mx.array(v) for k, v in c.initial_params(V).items()}
HD = c.D // c.H


def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) * mx.rsqrt(v + 1e-5) * w + b


def forward(p, idx):
    b, t = idx.shape
    x = p['tok'][idx] + p['pos'][:t]
    for i in range(c.L):
        g = lambda n: p[f'b{i}.{n}']
        h = ln(x, g('ln1_w'), g('ln1_b'))
        q, k, v = ((h @ g(n).T).reshape(b, t, c.H, HD).transpose(0, 2, 1, 3) for n in 'qkv')
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=HD ** -0.5, mask='causal')
        x = x + a.transpose(0, 2, 1, 3).reshape(b, t, c.D) @ g('o').T
        h = ln(x, g('ln2_w'), g('ln2_b'))
        x = x + nn.gelu_approx(h @ g('up').T + g('up_b')) @ g('down').T + g('down_b')
    return ln(x, p['ln_w'], p['ln_b']) @ p['head'].T + p['head_b']


def loss_fn(p, x, y):
    return nn.losses.cross_entropy(forward(p, x).reshape(-1, V), y.reshape(-1), reduction='mean')


opt = optim.Adam(learning_rate=c.LR, bias_correction=True)   # MLX defaults to False; PyTorch and tinygrad always correct
opt.init(P)
grad_fn = mx.value_and_grad(loss_fn)
state = [P, opt.state]


@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = grad_fn(P, x, y)
    P.update(opt.apply_gradients(grads, P))
    return loss


eval_loss = mx.compile(loss_fn)
curve, tm = [], c.Timer()
t0 = time.perf_counter()
for s in range(c.STEPS):
    x, y = c.get_batch(train, plan[s])
    loss = step(mx.array(x), mx.array(y)); mx.eval(state); lv = loss.item()
    tm.lap()
    if s % c.LOG_EVERY == 0 or s == c.STEPS - 1: curve.append([s, round(lv, 4)]); print(s, round(lv, 4), flush=True)
wall = time.perf_counter() - t0
vl = float(np.mean([eval_loss(P, *map(mx.array, c.get_batch(valid, vplan[i]))).item() for i in range(c.VALID_BATCHES)]))
c.save('mlx', {'framework': 'MLX', 'version': mx.__version__, 'device': str(mx.default_device()),
               'first_step_s': round(tm.t[0], 3), 'ms_per_step': round(1000 * float(np.median(tm.t[c.TIMED_FROM:])), 2),
               'wall_s': round(wall, 1), 'valid_loss': round(vl, 4), 'loss_curve': curve})
