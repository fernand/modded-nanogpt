import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import math
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)

########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    # a, b, c = 2, -1.5, 0.5
    # for _ in range(12):
    #     A = X @ X.mT
    #     B = b * A + c * A @ A
    #     X = a * X + B @ X
    a, b, c = 3.4445, -4.7750,  2.0315
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    # Original fast path. This mutates grad via lerp_, like the baseline.
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.compile
def muon_update_preserve_grad(grad, momentum, mu=0.95, nesterov=True):
    # Probe path. Same update, but does not mutate grad, so we can compute <g, u>.
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    # Fixed constants, not intended as tuning knobs.
    _CF_TARGET_RHO = 1.60 # below EOS=2, leaving stochastic-training margin
    _CF_EMA = 0.90
    _CF_MIN_MULT = 0.60 # conservative guardrails around the tuned base LR
    _CF_MAX_MULT = 1.10
    _CF_RHO_CAP = 4.00
    _CF_RHO_FLOOR = 0.05

    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, cf_probe_interval=0):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,

            # One public extra setting:
            # 0 disables central-flow probing; e.g. 50 probes every 50 steps.
            cf_probe_interval=cf_probe_interval,

            # Dynamic central-flow state.
            cf_lr_mult=1.0,
            cf_rho_ema=self._CF_TARGET_RHO,
            cf_last_rho=float("nan"),
        )
        super().__init__(params, defaults)
        self.cf_last_pred_decrease = None

    @staticmethod
    def _clamp(x, lo, hi):
        return min(max(x, lo), hi)

    def want_probe(self, step: int) -> bool:
        interval = self.param_groups[0]["cf_probe_interval"]
        return interval > 0 and step > 0 and step % interval == 0

    @torch.no_grad()
    def observe_loss(self, loss_before, loss_after):
        pred = self.cf_last_pred_decrease
        if pred is None:
            return None
        if pred.item() <= 1e-12:
            return None

        rho = 2 * (loss_after.detach() - loss_before.detach() + pred) / pred.clamp_min(1e-12)
        if not torch.isfinite(rho):
            return None

        rho_f = float(rho.clamp(0.0, self._CF_RHO_CAP).item())

        for group in self.param_groups:
            group["cf_last_rho"] = rho_f

            rho_ema = (
                self._CF_EMA * group["cf_rho_ema"]
                + (1 - self._CF_EMA) * rho_f
            )
            group["cf_rho_ema"] = rho_ema

            # Direct thermostat:
            # locally rho is approximately linear in LR, so sqrt gives a
            # damped one-step correction without exposing a gain hyperparameter.
            correction = math.sqrt(self._CF_TARGET_RHO / max(rho_ema, self._CF_RHO_FLOOR))
            group["cf_lr_mult"] = self._clamp(
                group["cf_lr_mult"] * correction,
                self._CF_MIN_MULT,
                self._CF_MAX_MULT,
            )

        return rho

    @torch.no_grad()
    def step(self, collect_stats=False):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        stats = None
        if collect_stats:
            # Same units as the training loss: summed CE over tokens and ranks.
            # stats[0] = first-order predicted decrease from Muon + Muon weight decay.
            stats = torch.zeros(1, device=self.param_groups[0]["params"][0].device, dtype=torch.float32)

        for group in self.param_groups:
            params = group["params"]

            pad = (-len(params)) % world_size
            params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad)]

            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["mu"]

            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]

                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)

                    if collect_stats:
                        update = muon_update_preserve_grad(p.grad, state["momentum"], mu=mu)

                        # First-order predicted loss decrease from:
                        #   p <- p - lr * update
                        stats[0].add_((p.grad.float() * update.float()).sum(), alpha=lr)

                        # First-order predicted loss decrease from decoupled WD:
                        #   p <- p * (1 - lr * wd)
                        if wd != 0:
                            stats[0].add_((p.grad.float() * p.float()).sum(), alpha=lr * wd)
                    else:
                        update = muon_update(p.grad, state["momentum"], mu=mu)

                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

                dist.all_gather(
                    params_pad[base_i:base_i + world_size],
                    params_pad[base_i + rank],
                )

        if collect_stats:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            self.cf_last_pred_decrease = stats[0].detach().clone()

########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 16
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):
    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3375

    # initialize model parameters
    for name, p in model.named_parameters():
        if name.endswith("weight"):
            if "proj" in name:
                p.data.zero_()
            elif "embed" in name:
                p.data.normal_()  # default torch init
            else:
                p.data.normal_(std=0.33**0.5 / p.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            p.data.zero_()
        elif name.endswith("gains"):
            p.data.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = Muon(
        [p for p in model.blocks.parameters() if p.ndim >= 2],
        lr=0.025, weight_decay=0.025,
        cf_probe_interval=50,   # the only new practical setting; 0 disables
    )
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta * group.get("cf_lr_mult", 1.0)

    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):
        # --------------- VALIDATION SECTION -----------------
        if step == train_steps or step % 125 == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            cf = optimizer2.param_groups[0]
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                    + f" step_avg:{step_avg:.2f}s"
                    + f" cf_mult:{cf['cf_lr_mult']:.3f}"
                    + f" cf_rho:{cf['cf_rho_ema']:.2f}"
                    + f" cf_last:{cf['cf_last_rho']:.2f}", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)

        do_cf_probe = optimizer2.want_probe(step)

        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0

        train_loss_before = torch.zeros((), device=device) if do_cf_probe else None

        for i in range(len(inputs) // mbs):
            loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            if do_cf_probe:
                train_loss_before += loss.detach()
            loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        if do_cf_probe:
            dist.all_reduce(train_loss_before, op=dist.ReduceOp.SUM)

        set_hparams(step)

        # Muon and AdamW touch disjoint parameters.
        # Step Muon first so the probe isolates the Muon-only loss change.
        optimizer2.step(collect_stats=do_cf_probe)

        if do_cf_probe:
            muon_loss_after = torch.zeros((), device=device)
            with torch.no_grad():
                for i in range(len(inputs) // mbs):
                    muon_loss_after += model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(muon_loss_after, op=dist.ReduceOp.SUM)
            optimizer2.observe_loss(train_loss_before, muon_loss_after)

        optimizer1.step()

        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{approx_training_time/(step + 1):.2f}s", console=True, log=False)

dist.destroy_process_group()
