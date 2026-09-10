---
license: mit
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - ultralytics
  - yolo
  - yolo26
  - object-detection
  - historical-newspapers
  - layout-analysis
datasets:
  - Teklia/Newspapers-finlam
  - Teklia/Newspapers-finlam-La-Liberte
language:
  - fr
  - en
base_model: Ultralytics/YOLO26
---

# cuttlefisher — newspaper block and title detector (YOLO26)

**Task:** object detection of the layout blocks of a historical newspaper page, with the
block classes needed to cut a page into articles — above all `ARTICLE-TITLE`.

The model is meant for the bottom-up article-segmentation pipeline of Mocaër et al. (ICDAR-HIP
2026): detect blocks and classes, order them, and start a new article at every title in
reading order. It was trained for [SquiddleOCR](https://github.com/storytracer/squiddleocr), which already
has regions and reading order from eynollah and only lacks a reliable title class.
Code, notes and full metrics: <https://github.com/storytracer/cuttlefisher>.

## Files

| file | what |
|---|---|
| `best.pt` | **phase 1**: yolo26m, imgsz 1280, trained on `Teklia/Newspapers-finlam` only (623 pages, 149 newspapers) |
| `args.yaml` | the ultralytics training arguments of `best.pt` |
| `classes.txt` | class names in id order |
| `results_test.md` / `.json` | per-class P/R/AP50/AP50-95 on the FINLAM test split |
| `article_rule_test.md` | article-cut metric (see below) |

A phase-2 model trained on FINLAM + La Liberté will be added under `phase2/` if it improves
the diverse test split; this card will say so.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("storytracer/cuttlefisher", "best.pt"))
results = model.predict("page.jpg", imgsz=1280, conf=0.35, max_det=600)

for box in results[0].boxes:
    print(model.names[int(box.cls)], box.conf.item(), box.xyxy[0].tolist())
```

- `imgsz=1280`: the model was trained at 1280; pages are ~2000 px high in the training data
  and title boxes are one **line** each (median 14 px tall at 2000 px), so do not go smaller.
- `conf=0.35` balances title precision and recall (P 0.88 / R 0.77 on val). Use 0.15–0.25
  if a missed headline line costs more than a spurious title.
- `max_det=600`: dense pages carry up to 500 blocks; the ultralytics default of 300 caps recall.
- `iou` has no effect: YOLO26 is end-to-end (NMS-free).
- Plain `.pt`, no exotic export; runs on any ultralytics ≥ 8.4 with a CUDA or CPU torch.

## Classes (13, id order)

```
0  HEADER-TITLE            8  ARTICLE-INSIDEHEADING
1  HEADER-TEXT             9  CAPTION
2  ARTICLE-ILLUSTRATION   10  AUTHOR
3  ADVERTISEMENT          11  ARTICLE-TABLE
4  ANNOUNCEMENT           12  SECTION-TITLE
5  ARTICLE-TITLE
6  ARTICLE-TEXT
7  ARTICLE-SUBTITLE
```

Annotation granularity of the training data: text is one box per paragraph block; titles,
subtitles and inside-headings are **one box per line**. `SECTION-TITLE` has no instances
in the phase-1 training data (it only occurs in La Liberté), so the phase-1 model never
predicts it. `ADVERTISEMENT` has 35 training instances and is unreliable.

## Training data

Both datasets are from the Teklia / LITIS *FINLAM* project, MIT licence:

- [Teklia/Newspapers-finlam](https://huggingface.co/datasets/Teklia/Newspapers-finlam) —
  721 pages (623 / 50 / 48) from 149 French and English newspapers, 19th–20th century,
  images 2000 px high, 13 classes. **Phase 1 trains on this only.**
- [Teklia/Newspapers-finlam-La-Liberte](https://huggingface.co/datasets/Teklia/Newspapers-finlam-La-Liberte) —
  8,836 pages of *La Liberté* (1925–1928), 16 classes mapped onto the 13 above. Phase 2.

## Training recipe (phase 1, `best.pt`)

- `yolo26m.pt` (COCO pretrained), ultralytics 8.4.146, torch 2.14 + cu126
- imgsz 1280, batch 32 on 4 × RTX A6000 (8 per GPU, ~20 steps per epoch), epochs 150 with
  cos_lr, patience 30 (best at epoch 78, stopped at 108), cache=ram, max_det 600,
  default augmentation. 24 minutes.
- Also tried: yolo26s (−0.04 title mAP50), imgsz 1024 (tie on titles, worse overall),
  the full 150-epoch cosine schedule without early stopping and yolo26l (both slightly
  better mAP50-95, worse on the title classes). Exact numbers in the repository's `NOTES.md`.

## Results on the diverse FINLAM test split (48 pages, 11 035 boxes)

conf 0.001, max_det 600, single GPU

| class | instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| ARTICLE-TITLE | 1065 | 0.747 | 0.798 | **0.795** | 0.675 |
| ARTICLE-SUBTITLE | 350 | 0.620 | 0.726 | **0.658** | 0.517 |
| ARTICLE-INSIDEHEADING | 205 | 0.469 | 0.717 | **0.557** | 0.454 |
| SECTION-TITLE | 0 | – | – | – | – |
| HEADER-TITLE | 10 | 0.907 | 0.982 | 0.977 | 0.570 |
| HEADER-TEXT | 111 | 0.555 | 0.540 | 0.461 | 0.318 |
| ARTICLE-ILLUSTRATION | 148 | 0.784 | 0.859 | 0.854 | 0.709 |
| ADVERTISEMENT | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| ANNOUNCEMENT | 56 | 0.531 | 0.357 | 0.345 | 0.246 |
| ARTICLE-TEXT | 8504 | 0.814 | 0.776 | 0.811 | 0.654 |
| CAPTION | 70 | 0.760 | 0.635 | 0.664 | 0.460 |
| AUTHOR | 49 | 0.736 | 0.468 | 0.519 | 0.405 |
| ARTICLE-TABLE | 466 | 0.510 | 0.255 | 0.303 | 0.144 |
| **all (12 classes)** | 11035 | 0.703 | 0.593 | **0.579** | 0.429 |
| **title classes (3)** | | | | **0.670** | |

**Article rule.** Taking the ground-truth zones in ground-truth reading order, giving each
zone the class of its best-overlapping prediction (IoU ≥ 0.5) and starting a new article at
every run of title zones, pairwise same-article F1 is **0.584** with predicted classes
against **0.633** with ground-truth classes — the detector costs 0.05 F1; the ceiling is low
because many FINLAM articles have no title or several. Zone-level title F1 is 0.805
(P 0.85, R 0.76).

The paper reports 72.3 mAP50 on *La Liberté* (one newspaper, 8k training pages); the numbers
above are on a 149-newspaper test set and are not comparable.

## Citation

```bibtex
@inproceedings{mocaer2026hierarchical,
  title     = {Towards Hierarchical Structure Understanding of Newspaper Images},
  author    = {Moca{\"e}r, William and Tarride, Sol{\`e}ne and Constum, Thomas and Agbeti-Messan, Merveilles and Simon, Tom and Chatelain, Cl{\'e}ment and Nicolas, St{\'e}phane and Tranouez, Pierrick and Cretin, S{\'e}bastien},
  booktitle = {ICDAR Workshop on Historical Document Imaging and Processing (HIP)},
  year      = {2026},
  note      = {arXiv:2607.15082}
}
```

Datasets: Teklia / LITIS, FINLAM project, <https://finlam.projets.litislab.fr/>, MIT licence.
