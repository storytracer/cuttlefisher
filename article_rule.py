#!/usr/bin/env python
"""The article rule with predicted classes (step 6).

For each test page: take the ground-truth zones in ground-truth reading order, give each zone
the class of the best-overlapping prediction (IoU >= 0.5, else ARTICLE-TEXT), start a new
article at every run of ARTICLE-TITLE / SECTION-TITLE zones, and score the grouping against
the true article ids with pairwise F1 over same-article zone pairs. The same rule on the
ground-truth classes is the ceiling; the gap is what the detector costs.

    uv run article_rule.py runs/s1024/weights/best.pt --imgsz 1024 --conf 0.25 --iou 0.6 [--md deliver/article_rule.md]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from convert import CLASSES, CLASS_ID

TITLE_IDS = {CLASS_ID["ARTICLE-TITLE"], CLASS_ID["SECTION-TITLE"]}
TEXT_ID = CLASS_ID["ARTICLE-TEXT"]


def iou_matrix(a, b):
    """a: (n,4), b: (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def cut_articles(classes_in_order):
    """Article index per zone: a new article starts at each run of title zones."""
    art, cur, prev_title = [], -1, False
    for c in classes_in_order:
        is_title = c in TITLE_IDS
        if is_title and not prev_title:
            cur += 1
        if cur < 0:  # page starts without a title
            cur = 0
        art.append(cur)
        prev_title = is_title
    return art


def pairwise_prf(pred_ids, true_ids):
    tp = fp = fn = 0
    for i, j in combinations(range(len(true_ids)), 2):
        t = true_ids[i] == true_ids[j]
        p = pred_ids[i] == pred_ids[j]
        tp += t and p; fp += p and not t; fn += t and not p
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--root", default="data/finlam")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--md")
    a = ap.parse_args()

    model = YOLO(a.weights)
    root = Path(a.root)
    metas = sorted((root / "meta" / a.split).glob("*.json"))
    tot = {"pred": [0, 0, 0], "gt": [0, 0, 0]}
    per_page = {"pred": [], "gt": []}
    title_conf = [0, 0, 0, 0]  # zone-level: tp, fp, fn, tn for "is title" after assignment

    for mp in metas:
        meta = json.loads(mp.read_text())
        zones = sorted(meta["zones"], key=lambda z: z["order"])
        gt_cls = [z["cls"] for z in zones]
        true_art = [z["article"] for z in zones]
        boxes = np.array([z["bbox"] for z in zones], dtype=float)

        r = model.predict(str(root / "images" / a.split / f"{mp.stem}.jpg"), imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                          max_det=600, device=a.device, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.cls.cpu().numpy().astype(int)
        M = iou_matrix(boxes, pb)
        pred_cls = []
        for i in range(len(zones)):
            if M.shape[1] and M[i].max() >= 0.5:
                pred_cls.append(int(pc[M[i].argmax()]))
            else:
                pred_cls.append(TEXT_ID)

        for g, p in zip(gt_cls, pred_cls):
            gt_t, pr_t = g in TITLE_IDS, p in TITLE_IDS
            title_conf[0 if gt_t and pr_t else 1 if pr_t else 2 if gt_t else 3] += 1

        for key, cls in (("pred", pred_cls), ("gt", gt_cls)):
            tp, fp, fn = pairwise_prf(cut_articles(cls), true_art)
            for k, v in enumerate((tp, fp, fn)):
                tot[key][k] += v
            per_page[key].append(f1(tp, fp, fn)[2])

    rows = ["| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |", "|---|---|---|---|---|"]
    for key, label in (("gt", "ground truth"), ("pred", f"predicted (`{Path(a.weights).parts[-3]}`, imgsz {a.imgsz}, conf {a.conf})")):
        p, r_, f = f1(*tot[key])
        rows.append(f"| {label} | {p:.3f} | {r_:.3f} | **{f:.3f}** | {np.mean(per_page[key]):.3f} |")
    tp, fp, fn, tn = title_conf
    zp, zr, zf = f1(tp, fp, fn)
    rows.append("")
    rows.append(f"Zone-level title assignment (ARTICLE-TITLE or SECTION-TITLE, after IoU ≥ 0.5 matching): "
                f"P {zp:.3f}, R {zr:.3f}, F1 {zf:.3f} ({tp} tp, {fp} fp, {fn} fn over {tp+fp+fn+tn} zones on {len(metas)} pages).")
    out = "\n".join(rows)
    print(out)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(out + "\n")


if __name__ == "__main__":
    main()
