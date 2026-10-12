"""Train GLUON vs AdamW on two matched 39M testbeds:

  LM:     39M GPT on WikiText-103   (val cross-entropy)
  vision: 39M ViT on CIFAR-100      (test top-1 / cross-entropy)

Usage:
  python3 src/train/train.py --domain lm --optimizer adamw --lr 1e-3 --steps 40000 --run-name lm_adamw
  python3 src/train/train.py --domain lm --optimizer gluon --lr 4e-4 --gamma 0.02 --steps 40000 --run-name lm_gluon
  python3 src/train/train.py --domain vision --optimizer adamw --lr 1e-3 --steps 40000 --run-name vit_adamw
  python3 src/train/train.py --domain vision --optimizer gluon --lr 4e-4 --steps 40000 --run-name vit_gluon

Logs: runs/<run-name>/log.jsonl ; checkpoint auto-resume from runs/<run-name>/ckpt.pt
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.gluon.optimizer import Gluon
from src.gluon.probes import ProbeGenerator
from src.gluon.split import fisher_diag_sample, residual_probe
from src.model.gpt import GPT, GPTConfig
from src.model.vit import ViT, ViTConfig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["lm", "vision"], required=True)
    p.add_argument("--optimizer", choices=["adamw", "gluon"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=None)  # per-domain default
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=None)  # per-domain default
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    # vision recipe
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--label-smooth", type=float, default=0.1)
    # gluon specific
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--hvp-interval", type=int, default=16)
    p.add_argument("--core-interval", type=int, default=16)
    p.add_argument("--core-offset", type=int, default=8)
    p.add_argument("--cycle-m", type=int, default=16)
    p.add_argument("--hvp-batch", type=int, default=None)  # per-domain default
    p.add_argument("--core-batch", type=int, default=None)  # default: hvp-batch
    p.add_argument("--beta-core", type=float, default=0.95)
    p.add_argument("--r-weight", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=60)
    p.add_argument("--ckpt-interval", type=int, default=2500)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.domain == "lm" else 256
    if args.weight_decay is None:
        args.weight_decay = 0.1 if args.domain == "lm" else 0.05
    if args.hvp_batch is None:
        args.hvp_batch = 2 if args.domain == "lm" else 32
    if args.core_batch is None:
        args.core_batch = args.hvp_batch
    if args.data_dir is None:
        args.data_dir = os.path.join(
            ROOT, "data_bin" if args.domain == "lm" else "data_vision")
    return args


# ---------------- LM data ----------------

class LMBatches:
    """Random training crops / deterministic eval windows from a uint16 memmap."""

    def __init__(self, path, block, batch, seed, device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.device = block, batch, device
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self):
        ix = torch.randint(len(self.data) - self.block - 1, (self.batch,),
                           generator=self.gen)
        x = torch.stack([torch.from_numpy(
            self.data[i:i + self.block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.data[i + 1:i + 1 + self.block].astype(np.int64)) for i in ix])
        return (x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True))

    def eval_windows(self, n_batches):
        stride, per = self.block, self.batch
        for b in range(n_batches):
            xs, ys = [], []
            for j in range(per):
                s = (b * per + j) * stride
                if s + self.block + 1 > len(self.data):
                    break
                xs.append(torch.from_numpy(self.data[s:s + self.block].astype(np.int64)))
                ys.append(torch.from_numpy(
                    self.data[s + 1:s + 1 + self.block].astype(np.int64)))
            if not xs:
                return
            yield (torch.stack(xs).to(self.device), torch.stack(ys).to(self.device))


# ---------------- vision data ----------------

class VisionBatches:
    """CIFAR-100 on-GPU sampler with pad-crop, hflip, mixup (own RNG stream so all
    optimizer arms see identical batch + augmentation sequences)."""

    def __init__(self, data_dir, batch, seed, device):
        self.x = torch.from_numpy(
            np.load(os.path.join(data_dir, "train_x.npy"))).to(device)   # uint8 BHWC
        self.y = torch.from_numpy(
            np.load(os.path.join(data_dir, "train_y.npy"))).to(device)
        self.tx = torch.from_numpy(
            np.load(os.path.join(data_dir, "test_x.npy"))).to(device)
        self.ty = torch.from_numpy(
            np.load(os.path.join(data_dir, "test_y.npy"))).to(device)
        self.batch, self.device = batch, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    def _normalize(self, u8_bhwc):
        x = u8_bhwc.permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def sample(self, mixup_alpha):
        B = self.batch
        ix = torch.randint(self.x.shape[0], (B,), generator=self.gen,
                           device=self.device)
        x = self._normalize(self.x[ix])
        y = self.y[ix]
        # pad-4 reflect + random crop (two gathers)
        xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
        oy = torch.randint(0, 9, (B,), generator=self.gen, device=self.device)
        ox = torch.randint(0, 9, (B,), generator=self.gen, device=self.device)
        ar = torch.arange(32, device=self.device)
        iy = (oy.view(B, 1) + ar.view(1, 32)).view(B, 1, 32, 1).expand(B, 3, 32, 40)
        xp = xp.gather(2, iy)
        jx = (ox.view(B, 1) + ar.view(1, 32)).view(B, 1, 1, 32).expand(B, 3, 32, 32)
        x = xp.gather(3, jx)
        # hflip
        fl = torch.rand(B, generator=self.gen, device=self.device) < 0.5
        x[fl] = x[fl].flip(-1)
        # mixup (batch-level lam)
        if mixup_alpha > 0:
            lam = float(torch._standard_gamma(torch.tensor(
                [mixup_alpha], device=self.device)).item())
            lam2 = float(torch._standard_gamma(torch.tensor(
                [mixup_alpha], device=self.device)).item())
            lam = lam / (lam + lam2 + 1e-12)
            perm = torch.randperm(B, generator=self.gen, device=self.device)
            x = lam * x + (1 - lam) * x[perm]
            return x, (y, y[perm], lam)
        return x, (y, y, 1.0)

    def eval_batches(self, bs=500):
        for s in range(0, self.tx.shape[0], bs):
            yield self._normalize(self.tx[s:s + bs]), self.ty[s:s + bs]


def vision_loss(logits, ytup, label_smooth):
    ya, yb, lam = ytup
    la = F.cross_entropy(logits, ya, label_smoothing=label_smooth)
    if lam == 1.0:
        return la
    lb = F.cross_entropy(logits, yb, label_smoothing=label_smooth)
    return lam * la + (1 - lam) * lb


# ---------------- shared ----------------

def lr_at(step, args):
    peak, mn = args.lr, args.lr * args.min_lr_frac
    if step < args.warmup:
        return peak * (step + 1) / args.warmup
    t = (step - args.warmup) / max(args.steps - args.warmup, 1)
    return mn + 0.5 * (peak - mn) * (1 + math.cos(math.pi * t))


def param_groups(model, weight_decay):
    seen, decay, nodecay = set(), [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else nodecay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate_lm(model, val, n_batches):
    model.eval()
    tot, ntok = 0.0, 0
    for X, Y in val.eval_windows(n_batches):
        loss = model(X, Y)
        tot += loss.item() * Y.numel()
        ntok += Y.numel()
    model.train()
    return {"val_loss": tot / max(ntok, 1),
            "val_ppl": math.exp(min(tot / max(ntok, 1), 20))}


@torch.no_grad()
def evaluate_vision(model, data):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for X, Y in data.eval_batches():
        logits = model(X)
        tot += F.cross_entropy(logits, Y, reduction="sum").item()
        correct += (logits.argmax(-1) == Y).sum().item()
        n += Y.numel()
    model.train()
    return {"val_loss": tot / n, "val_acc": correct / n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    run_dir = os.path.join(ROOT, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    if args.domain == "lm":
        model = GPT(GPTConfig(block_size=args.block_size)).to(device)
        train = LMBatches(os.path.join(args.data_dir, "train.bin"),
                          args.block_size, args.batch_size, args.seed, device)
        val = LMBatches(os.path.join(args.data_dir, "val.bin"),
                        args.block_size, 8, args.seed, device)
    else:
        model = ViT(ViTConfig()).to(device)
        train = VisionBatches(args.data_dir, args.batch_size, args.seed, device)
        val = train
    n_params = model.num_params()

    groups = param_groups(model, args.weight_decay)
    if args.optimizer == "adamw":
        b1 = args.beta1 if args.beta1 is not None else 0.9
        b2 = args.beta2 if args.beta2 is not None else \
            (0.95 if args.domain == "lm" else 0.999)
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(b1, b2), eps=1e-8,
                                fused=True)
        probe_gen, all_params = None, None
    else:
        b1 = args.beta1 if args.beta1 is not None else 0.96
        b2 = args.beta2 if args.beta2 is not None else 0.95
        opt = Gluon(groups, lr=args.lr, betas=(b1, b2), beta_core=args.beta_core,
                    gamma=args.gamma, cycle_m=args.cycle_m,
                    weight_decay=args.weight_decay, r_weight=args.r_weight)
        # weight decay handled inside Gluon per group; groups carry their own wd
        all_params = [p for g in groups for p in g["params"]]
        probe_gen = ProbeGenerator(all_params, m=args.cycle_m, seed=args.seed + 99)

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(torch.zeros((), device=device))

    start_step, t_train_cum, seen_items = 0, 0.0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if args.domain == "lm":
            train.gen.set_state(ck["data_gen"])
        else:
            train.gen.set_state(ck["data_gen"].to(train.gen.device)
                                if hasattr(ck["data_gen"], "to") else ck["data_gen"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start_step = ck["step"] + 1
        t_train_cum = ck["t_train_cum"]
        seen_items = ck["seen_items"]
        print(f"resumed {args.run_name} at step {start_step}", flush=True)

    log_f = open(log_path, "a", buffering=1)

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")

    if start_step == 0:
        log({"type": "meta", "run": args.run_name, "domain": args.domain,
             "optimizer": args.optimizer, "n_params": n_params, "args": vars(args),
             "gpu": torch.cuda.get_device_name(0)})
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, domain={args.domain}, "
          f"opt={args.optimizer}", flush=True)

    model.train()
    dt_acc, hvp_acc, loss_acc, nacc = 0.0, 0.0, 0.0, 0
    for step in range(start_step, args.steps):
        t0 = time.perf_counter()
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if args.domain == "lm":
            X, Y = train.sample()
            loss_closure_targets = Y
        else:
            X, ytup = train.sample(args.mixup)
            loss_closure_targets = ytup

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if args.domain == "lm":
                loss = model(X, Y)
            else:
                logits = model(X)
                loss = vision_loss(logits, ytup, args.label_smooth)
        scaler.scale(loss).backward()

        hvp_ms = 0.0
        if probe_gen is not None:
            th = time.perf_counter()
            did = False
            if (step + args.core_offset) % args.core_interval == 0:
                cs = fisher_diag_sample(model, all_params, X[: args.core_batch])
                opt.accumulate_core(cs)
                did = True
            if step % args.hvp_interval == 0:
                cyc = opt.total_probes // args.cycle_m
                r = opt.total_probes % args.cycle_m
                probes = probe_gen.probe(cyc, r)
                Xp = X[: args.hvp_batch]
                if args.domain == "lm":
                    Yp = loss_closure_targets[: args.hvp_batch]

                    def lf(lg):
                        return F.cross_entropy(
                            lg.view(-1, lg.size(-1)), Yp.reshape(-1))
                else:
                    ya, yb, lam = loss_closure_targets
                    ya_p, yb_p = ya[: args.hvp_batch], yb[: args.hvp_batch]

                    def lf(lg):
                        la = F.cross_entropy(lg, ya_p,
                                             label_smoothing=args.label_smooth)
                        lb = F.cross_entropy(lg, yb_p,
                                             label_smoothing=args.label_smooth)
                        return lam * la + (1 - lam) * lb
                prods = residual_probe(model, all_params, Xp, lf, probes)
                opt.accumulate_residual(prods)
                did = True
            if did:
                torch.cuda.synchronize()
                hvp_ms = (time.perf_counter() - th) * 1e3

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_train_cum += dt
        seen_items += X.shape[0] * (args.block_size if args.domain == "lm" else 1)
        dt_acc += dt; hvp_acc += hvp_ms; loss_acc += loss.item(); nacc += 1

        if (step + 1) % args.log_interval == 0 or step == args.steps - 1:
            rec = {"type": "train", "step": step + 1, "loss": loss_acc / nacc,
                   "lr": lr, "grad_norm": float(gnorm),
                   "dt_ms": dt_acc / nacc * 1e3, "hvp_ms": hvp_acc / nacc,
                   "t_train_s": t_train_cum, "items": seen_items,
                   "mem_mib": torch.cuda.max_memory_allocated() / 2**20}
            if probe_gen is not None and (step + 1) % (args.log_interval * 10) == 0:
                rec.update(opt.stats())
            log(rec)
            dt_acc = hvp_acc = loss_acc = 0.0; nacc = 0

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            ev = evaluate_lm(model, val, args.eval_batches) if args.domain == "lm" \
                else evaluate_vision(model, val)
            ev.update({"type": "eval", "step": step + 1, "t_train_s": t_train_cum,
                       "items": seen_items})
            log(ev)
            msg = (f"val {ev['val_loss']:.4f}"
                   + (f" acc {ev['val_acc']*100:.2f}%" if "val_acc" in ev else ""))
            print(f"[{args.run_name}] step {step+1} {msg} "
                  f"({t_train_cum/60:.1f} min)", flush=True)

        if (step + 1) % args.ckpt_interval == 0 or step == args.steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "data_gen": train.gen.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "step": step, "t_train_cum": t_train_cum,
                        "seen_items": seen_items, "args": vars(args)}, ckpt_path)

    log_f.close()
    print(f"[{args.run_name}] DONE", flush=True)


if __name__ == "__main__":
    main()
