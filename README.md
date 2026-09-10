# cuttlefisher

A YOLO detector that finds and classifies the blocks of a historical newspaper page,
above all **article titles**, trained on the Teklia / LITIS *FINLAM* datasets.

## Why

Downstream, [SquiddleOCR](https://github.com/storytracer/squiddleocr) cuts a page into articles with one rule: *a new article starts at
every title in reading order*. That is the bottom-up pipeline of Mocaër et al. (ICDAR-HIP
2026): YOLO26 for blocks and classes, LayoutReader for order, then the cut. We already have
layout regions and reading order from eynollah; what we lack is a reliable title class.
The authors released no weights, only the data, so this repository trains the detector.

## Data

Both datasets are on the Hugging Face Hub under the MIT licence:

| dataset | pages (train/val/test) | scope | phase |
|---|---|---|---|
| [Teklia/Newspapers-finlam](https://huggingface.co/datasets/Teklia/Newspapers-finlam) | 721 (623/50/48) | 149 French and English newspapers, 19th–20th c., 13 classes | 1 |
| [Teklia/Newspapers-finlam-La-Liberte](https://huggingface.co/datasets/Teklia/Newspapers-finlam-La-Liberte) | 8,836 (7957/446/433) | *La Liberté*, 1925–1928, 16 classes | 2 |

The target schema is the 13 classes of the small set (see `classes.txt`); the 16 La Liberté
classes are mapped onto it in `convert.py`.

## Layout

- `convert.py` — parquet → JPEG images + YOLO labels under `data/` (git-ignored)
- `train.py`, `evaluate.py`, `article_rule.py` — training, per-class evaluation, article-cut metric
- `NOTES.md` — decisions, numbers, dead ends
- `deliver/` — metrics, curves, rendered test pages, `classes.txt`

## Results (phase 1, FINLAM test split, 48 pages)

yolo26m at imgsz 1280, trained on the 623 FINLAM train pages only. Full tables, the article-cut
metric and the runs that did not make it are in [`deliver/metrics.md`](deliver/metrics.md)
and [`NOTES.md`](NOTES.md).

| class | AP50 | P | R |
|---|---|---|---|
| ARTICLE-TITLE | 0.795 | 0.75 | 0.80 |
| ARTICLE-SUBTITLE | 0.658 | 0.62 | 0.73 |
| ARTICLE-INSIDEHEADING | 0.557 | 0.47 | 0.72 |
| all classes (mAP50) | 0.579 | | |

Article rule (new article at every title run, ground-truth zones and order): pairwise F1
0.584 with predicted classes vs 0.633 with ground-truth classes. SECTION-TITLE has no
instances in either dataset and is never predicted.

## Weights

Published on the Hugging Face Hub: <https://huggingface.co/storytracer/cuttlefisher>
(not committed to git).

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("storytracer/cuttlefisher", "best.pt"))
results = model.predict("page.jpg", imgsz=1280, conf=0.35, max_det=600)
```

## Reference

Mocaër et al., *Towards Hierarchical Structure Understanding of Newspaper Images*,
ICDAR-HIP 2026, [arXiv:2607.15082](https://arxiv.org/abs/2607.15082).

## Licence

MIT, matching the datasets.
