"""
Plot the dynamic learning-rate schedule used in ModelTrain.init_agent()
(warmup -> cosine decay -> floor hold), for chosen combinations of
peak learning rate, cosine_decay_steps, and alpha.

Mirrors tf.keras.optimizers.schedules.CosineDecay(warmup_target=..., warmup_steps=...):
  warmup_steps = decay_steps * 0.1
  step < warmup_steps:  lr = 0.1*peak + 0.9*peak * (step / warmup_steps)
  step >= warmup_steps: f = (step - warmup_steps) / decay_steps
                        lr = peak * ((1 - alpha) * 0.5*(1 + cos(pi*f)) + alpha)
  step > warmup_steps + decay_steps: lr holds at peak * alpha

Usage examples:
  python plot_lr_schedule.py
  python plot_lr_schedule.py --peak 0.000025 0.000015 0.00005 --decay-steps 200000 --alpha 0.02
  python plot_lr_schedule.py --vary alpha --peak 0.000025 --decay-steps 200000 --alpha 0.1 0.02 -o lr_alpha.png
"""

import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt


def learning_rate(step, peak, decay_steps, alpha):
    step = np.asarray(step, dtype=float)
    initial = peak * 0.1
    warmup_steps = decay_steps * 0.1
    total_steps = decay_steps + warmup_steps

    step = np.minimum(step, total_steps)
    warmup = initial + (peak - initial) * (step / warmup_steps)

    f = (step - warmup_steps) / decay_steps
    cosine_decayed = 0.5 * (1 + np.cos(np.pi * f))
    decayed = peak * ((1 - alpha) * cosine_decayed + alpha)

    return np.where(step < warmup_steps, warmup, decayed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peak", type=float, nargs="+", default=[0.000025],
                    help="Peak learning rate(s), i.e. cfg.lrn_rate")
    p.add_argument("--decay-steps", type=int, nargs="+", default=[200000],
                    help="cfg.cosine_decay_steps value(s)")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.02],
                    help="Floor fraction of peak (LR floor = peak * alpha)")
    p.add_argument("--points", type=int, default=400, help="Samples along the x-axis")
    p.add_argument("-o", "--output", default="lr_schedule.png", help="Output PNG path")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    combos = list(itertools.product(args.peak, args.decay_steps, args.alpha))

    max_decay = max(args.decay_steps)
    x_max = max_decay * 1.1 * 1.25
    steps = np.linspace(0, x_max, args.points)

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for peak, decay_steps, alpha in combos:
        lr = learning_rate(steps, peak, decay_steps, alpha)
        label = f"peak={peak:g}, decay_steps={decay_steps:,}, alpha={alpha:g}"
        ax.plot(steps, lr, linewidth=2, label=label)

    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Dynamic learning rate schedule (warmup + cosine decay)")
    ax.ticklabel_format(axis="x", style="plain")
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output} ({len(combos)} curve(s))")


if __name__ == "__main__":
    main()
