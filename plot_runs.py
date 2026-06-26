#!/usr/bin/env python3
"""Compare training runs: eval returns, training loss, and (optionally) the
per-variable gradient norms of one run.

Reads the CSV outputs that ModelTrain writes under ./data/ for each run label
(``results_<run>.csv``, ``loss_<run>.csv``, ``pgradients_<run>.csv``) and saves
a stacked comparison figure.

Examples:
    python plot_runs.py LL_53 LL_54 LL_55
    python plot_runs.py LL_53 LL_54 LL_55 --grad LL_55
    python plot_runs.py LL_50 LL_51 --no-grad --out data/LL50_vs_LL51.png

Notes:
    Eval returns are placed at ``index * eval_interval`` (the first point is the
    pre-training eval at step 0). Each run also logs one extra final eval beyond
    its last interval boundary, so the final marker's step is approximate.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_returns(data, run, eval_interval):
    path = os.path.join(data, f"results_{run}.csv")
    vals = [float(x) for x in open(path) if x.strip()]
    steps = [i * eval_interval for i in range(len(vals))]
    return steps, vals


def load_loss(data, run):
    steps, loss = [], []
    for line in open(os.path.join(data, f"loss_{run}.csv")):
        if line.startswith("Step") or not line.strip():
            continue
        s, l = line.split(",")
        steps.append(float(s))
        loss.append(float(l))
    return steps, loss


def load_grads(data, run):
    lines = [l for l in open(os.path.join(data, f"pgradients_{run}.csv")) if l.strip()]
    header = lines[0].strip().split(",")
    cols = {h: [] for h in header}
    for line in lines[1:]:
        for h, v in zip(header, line.strip().split(",")):
            cols[h].append(float(v))
    # Drop the step-0 row: it is a one-shot pre-training gradient snapshot whose
    # magnitude is off-scale relative to the training-time norms.
    if cols["Step"] and cols["Step"][0] == 0:
        for k in cols:
            cols[k] = cols[k][1:]
    return header, cols


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run labels, e.g. LL_53 LL_54 LL_55")
    p.add_argument("--data", default="data", help="directory holding the CSVs (default: data)")
    p.add_argument("--grad", default=None,
                   help="run whose per-variable gradient norms to plot (default: last run)")
    p.add_argument("--no-grad", action="store_true", help="omit the gradient-norm panel")
    p.add_argument("--eval-interval", type=int, default=20000,
                   help="steps between eval returns (default: 20000)")
    p.add_argument("--clip", type=float, default=2.0,
                   help="gradient-clip ceiling to draw on the grad panel (default: 2.0)")
    p.add_argument("--out", default=None, help="output PNG path (default: derived from run names)")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    grad_run = None if args.no_grad else (args.grad or args.runs[-1])
    npanels = 3 if grad_run else 2

    fig, axes = plt.subplots(npanels, 1, figsize=(11, 4 + 3.5 * npanels), sharex=True)
    ax1, ax2 = axes[0], axes[1]

    palette = plt.get_cmap("tab10")
    colors = {run: palette(i % 10) for i, run in enumerate(args.runs)}

    # Panel 1: eval returns
    for run in args.runs:
        rs, rv = load_returns(args.data, run, args.eval_interval)
        ax1.plot(rs, rv, marker="o", ms=3, lw=1.6, color=colors[run],
                 label=f"{run} (best {max(rv):.0f})")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.axhline(200, color="green", lw=0.8, ls=":", label="solved (+200)")
    ax1.set_ylabel("Avg eval return")
    ax1.set_title(" vs ".join(args.runs))
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: training loss
    for run in args.runs:
        ls, ll = load_loss(args.data, run)
        ax2.plot(ls, ll, lw=1.6, color=colors[run], label=run)
    ax2.set_ylabel("Training loss")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3 (optional): per-variable gradient norms for one run
    if grad_run:
        ax3 = axes[2]
        header, cols = load_grads(args.data, grad_run)
        gsteps = cols["Step"]
        names = header[1:]
        gpalette = plt.get_cmap("tab20")
        for i, name in enumerate(names):
            is_total = name == "Total"
            ax3.plot(gsteps, cols[name],
                     lw=2.0 if is_total else 1.3,
                     color="black" if is_total else gpalette(i % 20),
                     label=name.replace("QNet/", "").replace(":0", ""))
        ax3.axhline(args.clip, color="gray", lw=1.0, ls="--",
                    label=f"clip ceiling ({args.clip:g})")
        top = max(args.clip + 2.0, 1.1 * max(cols.get("Total", [0]) or [0]))
        ax3.set_ylim(0, top)
        ax3.set_ylabel(f"{grad_run} grad norm (per var)")
        ax3.legend(loc="upper left", fontsize=8, ncol=2)
        ax3.grid(alpha=0.3)

    axes[-1].set_xlabel("Training step")
    plt.tight_layout()

    out = args.out or os.path.join(
        args.data, "return_loss_" + "_vs_".join(args.runs) + ".png")
    plt.savefig(out, dpi=args.dpi)
    print("saved", out)


if __name__ == "__main__":
    main()
