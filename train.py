#!/usr/bin/env python
"""Train a YOLO26 block detector on FINLAM. Thin wrapper so every run has the same recipe.

    uv run train.py --model s --imgsz 1024 --name s1024
    uv run train.py --model m --imgsz 1280 --name m1280 --data data/finlam_plus_laliberte.yaml

Logs go to runs/<name>/ (ultralytics) and runs/<name>.log (stdout, when launched with nohup).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="s", help="yolo26 size: n/s/m/l/x, or a .pt path")
ap.add_argument("--data", default="data/finlam.yaml")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--batch", type=int, default=64, help="total across GPUs; 8 per GPU keeps ~10 steps/epoch on 623 pages")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--device", default="0,1,2,3,4,5,6,7")
ap.add_argument("--name", required=True)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--cache", default="ram", help="ram (fine for 623 pages) | disk | False. With DDP every rank "
                "caches the whole set: 8 ranks x 25 GB for the 8.6k-page phase-2 set, and forked workers copy on "
                "write on top — use False there, JPEG decoding is not the bottleneck")
ap.add_argument("--extra", nargs="*", default=[], help="extra k=v overrides for ultralytics")
a = ap.parse_args()

weights = a.model if a.model.endswith(".pt") else f"weights/yolo26{a.model}.pt"
extra = {}
for kv in a.extra:
    k, v = kv.split("=", 1)
    try:
        v = float(v) if "." in v else int(v)
    except ValueError:
        v = {"True": True, "False": False}.get(v, v)
    extra[k] = v

YOLO(weights).train(
    data=a.data,
    imgsz=a.imgsz,
    batch=a.batch,
    epochs=a.epochs,
    patience=a.patience,
    device=a.device,
    workers=8,
    cache={"False": False, "false": False}.get(a.cache, a.cache),
    cos_lr=True,
    max_det=600,  # pages have up to 524 boxes; the default 300 caps recall
    project=str(Path(__file__).resolve().parent / "runs"),  # relative "runs" gets nested under runs/detect/runs
    name=a.name,
    exist_ok=True,
    fraction=a.fraction,
    plots=True,
    **extra,
)
