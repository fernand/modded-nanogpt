import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch_sparse
from fast_hadamard_transform import hadamard_transform

with open(sys.argv[0]) as f:
    code = f.read()

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :] # type: ignore

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_up = nn.Parameter(torch.empty(4 * config.n_embd, config.N))
        nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))
        self.w_down = nn.Parameter(torch.empty(config.N, 4 * config.n_embd))
        nn.init.kaiming_uniform_(self.w_down, a=math.sqrt(5))

    def forward(self, x, random_sign, proj_indices, proj_values):
        up = torch_sparse.spmm(
            proj_indices, proj_values,
            self.config.n_embd, self.config.N, hadamard_transform(random_sign * self.w_up, 1 / math.sqrt(self.config.N)).T
        )
        x = x @ up
        x = F.gelu(x)
        down = torch_sparse.spmm(
            proj_indices, proj_values,
            self.config.n_embd, self.config.N, hadamard_transform(random_sign * self.w_down.T, 1 / math.sqrt(self.config.N)).T
        )
        return x @ down.T

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x, random_sign, proj_indices, proj_values):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x), random_sign, proj_indices, proj_values)
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    N: int = 1024

class GPT(nn.Module):
    def create_sparse_proj(self, n_embd, N, k, q):
        flat_indices = torch.randperm(n_embd * N, dtype=torch.long)[:k]
        row_indices = flat_indices // N
        col_indices = flat_indices % N
        indices = torch.stack([row_indices, col_indices], dim=0)
        values = torch.randn((k,)) / q
        indices, values = torch_sparse.coalesce(indices, values, n_embd, N)
        return indices, values

    def __init__(self, config):
        super().__init__()
        self.config = config
        q = min(math.pow((math.log(4 * config.n_embd)), 2) / config.N, 1)
        k = round(q * config.n_embd * config.N / 16) * 16
        self.register_buffer('random_sign', torch.randint(0, 2, (config.N,)).float() * 2 - 1)
        proj_indices, proj_values = self.create_sparse_proj(config.n_embd, config.N, k, q)
        self.register_buffer('proj_indices', proj_indices)
        self.register_buffer('proj_values', proj_values)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x, self.random_sign, self.proj_indices, self.proj_values)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas) # type: ignore
        return optimizer

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="d12", help="d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="how many batches to accumulate per node before doing an optimizer step")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--warmdown_iters", type=int, default=0, help="learning rate warmdown iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--save_every", type=int, default=5000, help="every how many steps to save the checkpoint")
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "CUDA not available"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

    if master_process:
        tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.sequence_length
        print(f"tokens per optimizer step will be: {tokens_per_iter:,}")

    tokens_per_fwdbwd = B * T * ddp_world_size

    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768, N=1024),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    torch.set_float32_matmul_precision('high')
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device)

    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    timings = []
    dimensionalities = []
    for step in range(args.num_iterations + 1):
        t0 = time.perf_counter()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= args.val_max_steps
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1).to(device)
        for micro_step in range(args.gradient_accumulation_steps):
            # In DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss += loss.detach() / args.gradient_accumulation_steps
            x, y = train_loader.next_batch()
            loss.backward()

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item() # Mean loss across GPUs and micro steps
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | {(t1-t0)*1000:.0f} ms")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs('logs/%s' % run_id, exist_ok=True)
            torch.save(log, 'logs/%s/model_step%06d.pt' % (run_id, step))

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs('logs/%s' % run_id, exist_ok=True)
        torch.save(log, 'logs/%s/final.pt' % run_id)

    destroy_process_group()
