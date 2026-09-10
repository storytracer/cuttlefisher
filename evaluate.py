#!/usr/bin/env python
"""Per-class evaluation of a weights file on one split, single GPU.

    uv run evaluate.py runs/s1024/weights/best.pt --split test --imgsz 1024 [--md deliver/x.md]

Prints precision, recall, AP50 and AP50-95 per class, title classes first, plus the mean over
the four title classes ("title mAP50") used to choose between runs. Never trust the last table
of a DDP training log: it double counts instances across ranks.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from convert import CLASSES

TITLE_CLASSES = ["ARTICLE-TITLE", "SECTION-TITLE", "ARTICLE-SUBTITLE", "ARTICLE-INSIDEHEADING"]
ORDER = TITLE_CLASSES + [c for c in CLASSES if c not in TITLE_CLASSES]


def count_instances(data_yaml: str, split: str) -> Counter:
    root = Path([l for l in Path(data_yaml).read_text().splitlines() if l.startswith("path:")][0].split(":", 1)[1].strip())
    n = Counter()
    for f in (root / "labels" / split).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                n[CLASSES[int(line.split()[0])]] += 1
    return n


def evaluate(weights, data, split, imgsz, conf, iou, device, md=None, name=None):
    model = YOLO(weights)
    m = model.val(data=data, split=split, imgsz=imgsz, conf=conf, iou=iou, max_det=600, device=device,
                  batch=8, plots=False, verbose=False, project=str(Path(__file__).resolve().parent / "runs" / "val"),
                  name=name or Path(weights).parts[-3], exist_ok=True)
    n = count_instances(data, split)
    per = {}
    for j, ci in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(j)
        per[CLASSES[int(ci)]] = dict(p=float(p), r=float(r), ap50=float(ap50), ap=float(ap), n=n[CLASSES[int(ci)]])

    rows = [f"| class | instances | P | R | AP50 | AP50-95 |", "|---|---|---|---|---|---|"]
    for c in ORDER:
        if c in per:
            d = per[c]
            rows.append(f"| {c} | {d['n']} | {d['p']:.3f} | {d['r']:.3f} | {d['ap50']:.3f} | {d['ap']:.3f} |")
        else:
            rows.append(f"| {c} | {n[c]} | – | – | – | – |")
    titles = [per[c]["ap50"] for c in TITLE_CLASSES if c in per]
    title_map = sum(titles) / len(titles) if titles else float("nan")
    rows.append(f"| **all ({len(per)} classes)** | {sum(n.values())} | {m.box.mp:.3f} | {m.box.mr:.3f} | **{m.box.map50:.3f}** | {m.box.map:.3f} |")
    rows.append(f"| **title classes ({len(titles)})** | | | | **{title_map:.3f}** | |")
    table = "\n".join(rows)
    header = f"weights `{weights}`, split `{split}`, imgsz {imgsz}, conf {conf}, iou {iou}, max_det 600\n\n"
    print(header + table)
    if md:
        Path(md).parent.mkdir(parents=True, exist_ok=True)
        Path(md).write_text(header + table + "\n")
        Path(md).with_suffix(".json").write_text(json.dumps(dict(
            weights=str(weights), split=split, imgsz=imgsz, conf=conf, iou=iou,
            map50=float(m.box.map50), map=float(m.box.map), title_map50=title_map, per_class=per), indent=1))
    return m.box.map50, title_map, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--data", default="data/finlam.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--md", help="write the table to this markdown file (+ .json)")
    ap.add_argument("--name")
    a = ap.parse_args()
    evaluate(a.weights, a.data, a.split, a.imgsz, a.conf, a.iou, a.device, a.md, a.name)
