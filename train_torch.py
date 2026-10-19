"""Train the char model with PyTorch on Apple's GPU (MPS). Eager mode, fused SDPA attention."""
import time
import numpy as np
import torch, torch.nn.functional as F
import common as c

dev = torch.device('mps')
train, valid, V, plan, vplan = c.dataset()
P = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in c.initial_params(V).items()}
HD = c.D // c.H


def forward(p, idx):
    b, t = idx.shape
    x = p['tok'][idx] + p['pos'][:t]
    for i in range(c.L):
        g = lambda n: p[f'b{i}.{n}']
        h = F.layer_norm(x, (c.D,), g('ln1_w'), g('ln1_b'), 1e-5)
        q, k, v = ((h @ g(n).T).reshape(b, t, c.H, HD).transpose(1, 2) for n in 'qkv')
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(b, t, c.D) @ g('o').T
        h = F.layer_norm(x, (c.D,), g('ln2_w'), g('ln2_b'), 1e-5)
        x = x + F.gelu(h @ g('up').T + g('up_b'), approximate='tanh') @ g('down').T + g('down_b')
    return F.layer_norm(x, (c.D,), p['ln_w'], p['ln_b'], 1e-5) @ p['head'].T + p['head_b']


def loss_fn(x, y):
    return F.cross_entropy(forward(P, x).reshape(-1, V), y.reshape(-1).long())


opt = torch.optim.Adam(P.values(), lr=c.LR)
curve, tm = [], c.Timer()
t0 = time.perf_counter()
for s in range(c.STEPS):
    x, y = (torch.from_numpy(a).to(dev) for a in c.get_batch(train, plan[s]))
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(x, y); loss.backward(); opt.step()
    lv = loss.item()   # .item() synchronizes the GPU, as in the other two
    tm.lap()
    if s % c.LOG_EVERY == 0 or s == c.STEPS - 1: curve.append([s, round(lv, 4)]); print(s, round(lv, 4), flush=True)
wall = time.perf_counter() - t0
with torch.no_grad():
    vl = float(np.mean([loss_fn(*(torch.from_numpy(a).to(dev) for a in c.get_batch(valid, vplan[i]))).item()
                        for i in range(c.VALID_BATCHES)]))
c.save('torch', {'framework': 'PyTorch (MPS)', 'version': torch.__version__, 'device': 'mps',
                 'first_step_s': round(tm.t[0], 3), 'ms_per_step': round(1000 * float(np.median(tm.t[c.TIMED_FROM:])), 2),
                 'wall_s': round(wall, 1), 'valid_loss': round(vl, 4), 'loss_curve': curve})
