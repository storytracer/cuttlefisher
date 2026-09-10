#!/usr/bin/env python
"""Overlay the training curves of several runs into one PNG.

    uv run plot_curves.py deliver/curves.png runs/detect/runs/s1024 runs/detect/runs/s1280 ...
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out, runs = sys.argv[1], sys.argv[2:]
panels = [("metrics/mAP50(B)", "val mAP50"), ("metrics/mAP50-95(B)", "val mAP50-95"),
          ("train/box_loss", "train box loss"), ("val/cls_loss", "val cls loss")]
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
for r in runs:
    df = pd.read_csv(Path(r) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(df["epoch"], df[col], label=Path(r).name, lw=1.2)
for ax, (col, title) in zip(axes, panels):
    ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout()
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print("wrote", out)
