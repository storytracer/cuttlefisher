#!/usr/bin/env python
"""Convert the FINLAM parquet datasets to YOLO detection format.

Output layout (git-ignored):
    data/<name>/images/<split>/<stem>.jpg
    data/<name>/labels/<split>/<stem>.txt     class cx cy w h  (normalised)
    data/<name>/meta/<split>/<stem>.json      zones with class, bbox, order, article id
    data/<name>.yaml

Polygons are rectangles in practice; we take the bounding box. In the small set the
coordinates are percentages (0-100) of the image size; the script checks the range on the
first pages and refuses to guess if they are not.

Usage:
    uv run convert.py finlam            # Teklia/Newspapers-finlam       -> data/finlam
    uv run convert.py laliberte         # Teklia/Newspapers-finlam-La-Liberte -> data/laliberte (13-class schema)
"""
import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent

# Target schema: the 13 classes of Teklia/Newspapers-finlam, in ClassLabel order.
CLASSES = [
    "HEADER-TITLE",
    "HEADER-TEXT",
    "ARTICLE-ILLUSTRATION",
    "ADVERTISEMENT",
    "ANNOUNCEMENT",
    "ARTICLE-TITLE",
    "ARTICLE-TEXT",
    "ARTICLE-SUBTITLE",
    "ARTICLE-INSIDEHEADING",
    "CAPTION",
    "AUTHOR",
    "ARTICLE-TABLE",
    "SECTION-TITLE",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# La Liberté's 16 classes onto the 13. Verified on rendered pages, see NOTES.md.
LALIBERTE_MAP = {
    "HEADER-TITLE": "HEADER-TITLE",
    "HEADER-TEXT": "HEADER-TEXT",
    "SECTION-TITLE": "SECTION-TITLE",
    "ILLUSTRATION": "ARTICLE-ILLUSTRATION",
    "ADVERTISEMENT": "ADVERTISEMENT",
    "ANNOUNCEMENT": "ANNOUNCEMENT",
    "TITLE": "ARTICLE-TITLE",
    "TEXT": "ARTICLE-TEXT",
    "SUBTITLE": "ARTICLE-SUBTITLE",
    "INSIDEHEADING": "ARTICLE-INSIDEHEADING",
    "CAPTION": "CAPTION",
    "AUTHOR": "AUTHOR",
    "TABLE": "ARTICLE-TABLE",
    "TABLECONTENT": "ARTICLE-TABLE",
    "ILLUSTRATEDTEXT": "ARTICLE-TEXT",
    "ASIDE": "ARTICLE-TEXT",
}

DATASETS = {
    "finlam": dict(src=ROOT / "data/hf/finlam/data", article_key="article_id"),
    "laliberte": dict(src=ROOT / "data/hf/laliberte/data", article_key="zone_article_ids"),
}


def class_names_from_schema(pf: pq.ParquetFile) -> list[str]:
    """Read the ClassLabel names of zone_classes from the HF metadata in the parquet."""
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    feat = meta["info"]["features"]["zone_classes"]
    inner = feat.get("feature") or feat.get("sequence")
    names = inner["names"]
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    return names


def bbox(poly, w, h, scale):
    """scale: 1.0 for fractions (La Liberté), 100.0 for percentages (FINLAM), None for pixels."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if scale:
        x0, x1 = x0 / scale * w, x1 / scale * w
        y0, y1 = y0 / scale * h, y1 / scale * h
    x0, x1 = max(0.0, min(w, x0)), max(0.0, min(w, x1))
    y0, y1 = max(0.0, min(h, y0)), max(0.0, min(h, y1))
    return x0, y0, x1, y1


def convert(name: str, limit: int | None = None):
    cfg = DATASETS[name]
    src, out = cfg["src"], ROOT / "data" / name
    files = sorted(src.glob("*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {src}; download the dataset first")

    src_names = class_names_from_schema(pq.ParquetFile(files[0]))
    if name == "finlam":
        assert src_names == CLASSES, src_names
        mapping = {c: c for c in CLASSES}
    else:
        missing = set(src_names) - set(LALIBERTE_MAP)
        assert not missing, f"unmapped classes: {missing}"
        mapping = LALIBERTE_MAP
    print(f"source classes: {src_names}")

    counts = {s: Counter() for s in ("train", "val", "test")}
    pages = Counter()
    scale = "unset"  # decided on the first page: 1.0 fractions, 100.0 percentages, None pixels
    for f in files:
        split = f.name.split("-")[0]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "meta" / split).mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if limit and pages[split] >= limit:
                    break
                stem = row["page_arkindex_id"]
                img_path = out / "images" / split / f"{stem}.jpg"
                raw = row["page_image"]["bytes"]
                im = Image.open(io.BytesIO(raw))
                w, h = im.size
                if not img_path.exists():
                    if im.format == "JPEG":
                        img_path.write_bytes(raw)
                    else:
                        im.convert("RGB").save(img_path, "JPEG", quality=92)

                polys = row["zone_polygons"]
                if scale == "unset" and polys:
                    mx = max(max(max(p) for p in poly) for poly in polys)
                    scale = 1.0 if mx <= 1.0 else 100.0 if mx <= 100.0 else None
                    print(f"coordinates look like {dict([(1.0, 'fractions'), (100.0, 'percentages'), (None, 'pixels')])[scale]} (max {mx:.3f})")

                lines, zones = [], []
                for i, poly in enumerate(polys):
                    cname = mapping[src_names[row["zone_classes"][i]]]
                    cid = CLASS_ID[cname]
                    x0, y0, x1, y1 = bbox(poly, w, h, scale)
                    bw, bh = x1 - x0, y1 - y0
                    if bw < 1 or bh < 1:
                        continue
                    counts[split][cname] += 1
                    lines.append(f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                    zones.append(dict(
                        cls=cid, bbox=[round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        order=row["zone_orders"][i], article=row[cfg["article_key"]][i],
                        text=row["zone_texts"][i],
                    ))
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                (out / "meta" / split / f"{stem}.json").write_text(json.dumps(dict(
                    newspaper=row.get("newspaper_name", name), page_index=row["page_index"], width=w, height=h, zones=zones,
                ), ensure_ascii=False))
                pages[split] += 1
        print(f"{f.name}: done, pages so far {dict(pages)}")

    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (ROOT / "data" / f"{name}.yaml").write_text("\n".join(yaml) + "\n")
    (ROOT / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    print("\npages:", dict(pages))
    print(f"\n{'class':24s}" + "".join(f"{s:>8s}" for s in counts))
    for c in CLASSES:
        print(f"{c:24s}" + "".join(f"{counts[s][c]:8d}" for s in counts))
    print(f"{'TOTAL':24s}" + "".join(f"{sum(counts[s].values()):8d}" for s in counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=DATASETS)
    ap.add_argument("--limit", type=int, help="pages per split, for quick tests")
    a = ap.parse_args()
    convert(a.name, a.limit)
