"""Figures for the GLUON benchmark report.

Usage: python3 src/plots/make_plots.py <prefix> <baseline_run> <gluon_run>
Writes figures/<prefix>_{curves,gap,wallclock}.png (+ _acc for vision).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BASE_C, GLUON_C = "#7f7f7f", "#1f77b4"


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


def series(ev, key):
    return ([r["step"] for r in ev if key in r],
            [r[key] for r in ev if key in r])


def main():
    prefix, a_run, g_run = sys.argv[1], sys.argv[2], sys.argv[3]
    (a_tr, a_ev, a_meta), (g_tr, g_ev, _) = load(a_run), load(g_run)
    vision = a_meta and a_meta.get("domain") == "vision"

    lo = min(min(series(a_ev, "val_loss")[1]), min(series(g_ev, "val_loss")[1]))

    # 1. val-loss curves (steps)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for run, ev, c in ((a_run, a_ev, BASE_C), (g_run, g_ev, GLUON_C)):
        s, v = series(ev, "val_loss")
        ax.plot(s, v, color=c, lw=1.8, label=run)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(lo - 0.05, lo + 1.2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{prefix}_curves.png")
    plt.close(fig)

    # 2. gap trajectory (baseline − gluon at matched steps)
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=140)
    sa, va = series(a_ev, "val_loss")
    sg, vg = series(g_ev, "val_loss")
    common = sorted(set(sa) & set(sg))
    da, dg = dict(zip(sa, va)), dict(zip(sg, vg))
    ax.plot(common, [da[s] - dg[s] for s in common], color=GLUON_C, lw=1.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("val-loss gap (base − gluon)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{prefix}_gap.png")
    plt.close(fig)

    # 3. wall-clock curves
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for run, ev, c in ((a_run, a_ev, BASE_C), (g_run, g_ev, GLUON_C)):
        t = [r["t_train_s"] / 60 for r in ev]
        v = [r["val_loss"] for r in ev]
        ax.plot(t, v, color=c, lw=1.8, label=run)
    ax.set_xlabel("wall-clock (min)")
    ax.set_ylabel("val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(lo - 0.05, lo + 1.2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{prefix}_wallclock.png")
    plt.close(fig)

    # 4. vision: top-1 curves
    if vision:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        for run, ev, c in ((a_run, a_ev, BASE_C), (g_run, g_ev, GLUON_C)):
            s, v = series(ev, "val_acc")
            ax.plot(s, [x * 100 for x in v], color=c, lw=1.8, label=run)
        ax.set_xlabel("step")
        ax.set_ylabel("test top-1 (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{FIG}/{prefix}_acc.png")
        plt.close(fig)

    print(f"figures written: {FIG}/{prefix}_*.png")


if __name__ == "__main__":
    main()
