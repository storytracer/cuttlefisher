#!/usr/bin/env python
"""Render pages with their YOLO labels (or predictions) for eyeballing.

    uv run draw.py data/finlam test --n 5 --out checks/
    uv run draw.py data/finlam test --n 6 --out deliver/pages --weights runs/x/weights/best.pt --imgsz 1280
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from convert import CLASSES

# one colour per class; titles in warm saturated colours so they jump out
COLORS = {
    "HEADER-TITLE": (255, 0, 255), "HEADER-TEXT": (180, 120, 255), "ARTICLE-ILLUSTRATION": (0, 200, 200),
    "ADVERTISEMENT": (255, 140, 0), "ANNOUNCEMENT": (200, 200, 0), "ARTICLE-TITLE": (255, 0, 0),
    "ARTICLE-TEXT": (0, 160, 0), "ARTICLE-SUBTITLE": (255, 90, 90), "ARTICLE-INSIDEHEADING": (220, 0, 120),
    "CAPTION": (0, 90, 255), "AUTHOR": (120, 60, 0), "ARTICLE-TABLE": (0, 120, 120), "SECTION-TITLE": (255, 0, 120),
}


def draw_boxes(im, boxes, font, label_all=False):
    """boxes: list of (cls, x0, y0, x1, y1[, conf]) in pixels."""
    d = ImageDraw.Draw(im)
    for b in boxes:
        cls, x0, y0, x1, y1 = b[:5]
        name = CLASSES[int(cls)]
        col = COLORS[name]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        if label_all or name != "ARTICLE-TEXT":
            txt = name if len(b) == 5 else f"{name} {b[5]:.2f}"
            d.text((x0 + 2, max(0, y0 - 12)), txt, fill=col, font=font)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("split")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="checks")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((root / "images" / a.split).glob("*.jpg"))
    random.Random(a.seed).shuffle(imgs)
    imgs = imgs[: a.n]
    font = ImageFont.load_default(size=11)

    model = None
    if a.weights:
        from ultralytics import YOLO
        model = YOLO(a.weights)

    for p in imgs:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if model is None:
            boxes = []
            for line in (root / "labels" / a.split / f"{p.stem}.txt").read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            draw_boxes(im, boxes, font).save(out / f"{p.stem}_gt.jpg", quality=85)
        else:
            r = model.predict(str(p), imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=1000, verbose=False, device=1)[0]
            boxes = [(int(c), *xyxy, float(cf)) for c, xyxy, cf in
                     zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            draw_boxes(im, boxes, font).save(out / f"{p.stem}_pred.jpg", quality=85)
            # ground truth alongside, for comparison
            gt = []
            for line in (root / "labels" / a.split / f"{p.stem}.txt").read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                gt.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            draw_boxes(Image.open(p).convert("RGB"), gt, font).save(out / f"{p.stem}_gt.jpg", quality=85)
        print(p.stem, len(boxes), "boxes")


if __name__ == "__main__":
    main()
