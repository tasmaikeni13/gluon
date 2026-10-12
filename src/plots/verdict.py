"""Breakthrough-bar verdict for a (baseline, gluon) pair of runs.

Bars:
  bar 1: equal-step win on the primary metric (val loss; vision also reports top-1)
         by more than seed noise (0.005 nats / 0.15% acc)
  bar 2: equal-wall-clock win (interpolated at the earlier arm's finish time)
  bar 3: step-time overhead < 20%

Usage: python3 src/plots/verdict.py <baseline_run> <gluon_run>
"""

import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load(run):
    tr, ev, meta = [], [], None
    for line in open(os.path.join(ROOT, "runs", run, "log.jsonl")):
        r = json.loads(line)
        if r.get("type") == "eval":
            ev.append(r)
        elif r.get("type") == "train":
            tr.append(r)
        elif r.get("type") == "meta":
            meta = r
    return tr, ev, meta


def interp(ev, t, key):
    xs = [r["t_train_s"] / 60 for r in ev]
    ys = [r[key] for r in ev]
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= t:
            w = (t - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] * (1 - w) + ys[i] * w


def main():
    a_run, g_run = sys.argv[1], sys.argv[2]
    (a_tr, a_ev, a_meta), (g_tr, g_ev, _) = load(a_run), load(g_run)
    vision = a_meta and a_meta.get("domain") == "vision"
    a_fin, g_fin = a_ev[-1], g_ev[-1]
    a_dt = sum(r["dt_ms"] for r in a_tr) / len(a_tr)
    g_dt = sum(r["dt_ms"] for r in g_tr) / len(g_tr)
    overhead = (g_dt - a_dt) / a_dt * 100

    a_t, g_t = a_fin["t_train_s"] / 60, g_fin["t_train_s"] / 60
    t_common = min(a_t, g_t)
    a_at = interp(a_ev, t_common, "val_loss")
    g_at = interp(g_ev, t_common, "val_loss")
    d_steps = a_fin["val_loss"] - g_fin["val_loss"]
    d_time = a_at - g_at

    print(f"steps: {a_run} {a_fin['step']} vs {g_run} {g_fin['step']}")
    print(f"EQUAL STEPS   val loss: base {a_fin['val_loss']:.4f}  "
          f"gluon {g_fin['val_loss']:.4f}  Δ = {d_steps:+.4f} nats")
    if not vision:
        print(f"              ppl: {math.exp(a_fin['val_loss']):.2f} → "
              f"{math.exp(g_fin['val_loss']):.2f}")
    if vision:
        d_acc = g_fin["val_acc"] - a_fin["val_acc"]
        a_acc_t = interp(a_ev, t_common, "val_acc")
        g_acc_t = interp(g_ev, t_common, "val_acc")
        print(f"              top-1: base {a_fin['val_acc']*100:.2f}%  "
              f"gluon {g_fin['val_acc']*100:.2f}%  Δ = {d_acc*100:+.2f}%")
    print(f"WALL CLOCK    base {a_t:.1f} min, gluon {g_t:.1f} min "
          f"(overhead {overhead:+.1f}%: {a_dt:.1f} → {g_dt:.1f} ms/step)")
    print(f"EQUAL TIME    @{t_common:.1f} min val loss: base {a_at:.4f}  "
          f"gluon {g_at:.4f}  Δ = {d_time:+.4f}")
    if vision:
        print(f"              top-1 @{t_common:.1f} min: base {a_acc_t*100:.2f}%  "
              f"gluon {g_acc_t*100:.2f}%  Δ = {(g_acc_t-a_acc_t)*100:+.2f}%")

    if vision:
        bar1 = d_acc > 0.0015 and d_steps > 0
        bar2 = (g_acc_t - a_acc_t) > 0 and d_time > 0
    else:
        bar1 = d_steps > 0.005
        bar2 = d_time > 0.0
    bar3 = overhead < 20
    print(f"bar1 equal-step win   : {'PASS' if bar1 else 'FAIL'}")
    print(f"bar2 equal-time win   : {'PASS' if bar2 else 'FAIL'}")
    print(f"bar3 overhead < 20%   : {'PASS' if bar3 else 'FAIL'}")
    ok = bar1 and bar2 and bar3
    print(f"VERDICT: {'BREAKTHROUGH BAR MET' if ok else 'FAILED — iterate'}")


if __name__ == "__main__":
    main()
