# NOTES

Running log of decisions, numbers and dead ends. Newest at the bottom of each section.

## Machine (2026-09-10)

- 8 × RTX A6000 48 GB, driver 550.90.07 → CUDA 12.4 at most. GPU 0 was occupied by a
  transcriptions API at the start; it was freed shortly after and **all 8 GPUs are available**
  (`device=0,1,2,3,4,5,6,7`). The 1-GPU smoke run was done on GPU 1 before that.
- 126 cores, 679 GB RAM, /dev/shm 340 GB, ~198 GB free on /dev/vda1.
- No sudo. `uv` 0.10.9, `gh` and `hf` CLIs already authenticated.
- Torch: driver 550 cannot run cu130 wheels (needs ≥ 580), so torch is pinned to the cu126 index.

## Environment

- Python 3.12 via uv; deps in `pyproject.toml`: torch 2.14.0+cu126, ultralytics 8.4.146
  (has YOLO26; `yolo26s.pt` 10.0 M params, `yolo26m.pt` 21.9 M), datasets 5.0.1.
- `gh repo create` pushed over SSH and failed on host-key verification; remote switched to
  HTTPS with `gh auth setup-git`. Repo: https://github.com/storytracer/cuttlefisher
- PCIe link width at idle: 16x on all 8 GPUs.

## Phase 1 data: Teklia/Newspapers-finlam (2026-09-10)

- 4 parquet files, 991 MB. Images are JPEG bytes, height 2000, width 1280–1650. Written as-is.
- `zone_polygons` are percentages 0–100 (max seen 94.4), rectangles; bounding box used.
- Per-class instance counts after conversion (train / val / test):

  | class | train | val | test |
  |---|---|---|---|
  | HEADER-TITLE | 129 | 11 | 10 |
  | HEADER-TEXT | 1771 | 148 | 111 |
  | ARTICLE-ILLUSTRATION | 1272 | 89 | 148 |
  | ADVERTISEMENT | 35 | 0 | 1 |
  | ANNOUNCEMENT | 529 | 40 | 56 |
  | ARTICLE-TITLE | 11859 | 1198 | 1065 |
  | ARTICLE-TEXT | 108637 | 8837 | 8504 |
  | ARTICLE-SUBTITLE | 3014 | 297 | 350 |
  | ARTICLE-INSIDEHEADING | 2942 | 300 | 205 |
  | CAPTION | 527 | 24 | 70 |
  | AUTHOR | 519 | 27 | 49 |
  | ARTICLE-TABLE | 4524 | 339 | 466 |
  | SECTION-TITLE | **0** | **0** | **0** |
  | total | 135758 | 11310 | 11035 |

- **SECTION-TITLE does not occur at all in the small set**; it exists only in La Liberté. The
  class id is kept (12) so phase 2 can fill it. ADVERTISEMENT is 35/0/1: it cannot be
  evaluated on this set, val AP for it is undefined.
- ARTICLE-TEXT is 80 % of all boxes. Annotation granularity: text = paragraph blocks,
  titles = **one box per title line** (a 3-line headline is 3 ARTICLE-TITLE boxes).
- Boxes are small: median height 25 px at 2000 px page height, 5th percentile 7 px; title-ish
  boxes (TITLE/SUBTITLE/INSIDEHEADING) median 14 px. At `imgsz=1024` (÷1.95) 32 % of boxes are
  under 8 px tall — `imgsz=1280` or more will matter more here than usual.
- 218 boxes per page on average, max 524; 104/623 train pages exceed 300 →
  **`max_det=600`** for validation and inference, otherwise recall is capped.
- Sanity renders of 5 train pages in `checks/` looked right (boxes tight on the print).

## Smoke runs

- 100-image subset: `data/finlam_smoke.yaml` (txt lists of 100 train / 20 val images).
- 1 GPU (GPU 1), yolo26s, imgsz 1024, batch 8: 1 epoch in 0.006 h, mAP ≈ 0 as expected.
- 8 GPUs DDP, batch 64: 1 epoch in 0.004 h; all 8 GPUs at 8–10 GB, PCIe 16x under load.
  DDP works in this VM. Quirk: the final validation printed 31 402 instances = 4486 × 7 —
  the DDP end-of-training val double counts across ranks, so **all reported numbers come from a
  separate single-GPU `yolo val`**, never from the training log's last table.

## Article rule ceiling (step 6, ground-truth classes)

- `article_rule.py` with **ground-truth** classes on the 48 test pages: pairwise P 0.696,
  R 0.580, **F1 0.633** (mean page F1 0.609). That is the ceiling of the rule itself.
- Why so low: articles are contiguous in reading order (0 re-entries), but of 641 test
  articles **246 have no title zone at all** (203 start with ARTICLE-TEXT: continuations,
  briefs, fillers → merged into the preceding article, precision loss) and ~85 have more than
  one title run (13 have ≥ 10: listings and briefs columns where each item has its own
  headline but one `article_id` → split, recall loss). The detector gap is measured against
  this ceiling, not against 1.0.

## Phase 1 training

- Plan: two runs at a time, 4 GPUs each, batch 32 (8 per GPU, ~20 steps per epoch on
  623 pages), epochs 150, patience 30, cos_lr, cache=ram, max_det 600. `train.py`.
- Ultralytics 8.4 nests a relative `project=runs` under `runs/detect/runs/<name>`; the first
  four runs live there. `train.py` now passes an absolute project path.
- Ultralytics' `best.pt` is chosen by fitness = 0.1·mAP50 + 0.9·mAP50-95, not by mAP50.
- Val numbers below are from `evaluate.py` (single GPU, conf 0.001, iou 0.6, max_det 600) on
  the 50 val pages; "title mAP50" = mean AP50 of ARTICLE-TITLE, ARTICLE-SUBTITLE,
  ARTICLE-INSIDEHEADING (SECTION-TITLE has no instances).

  | run | GPUs | best ep / stopped | val mAP50 | val mAP50-95 | title mAP50 | TITLE AP50 | SUBTITLE | INSIDEHEADING | time |
  |---|---|---|---|---|---|---|---|---|---|
  | s1024 | 0–3 | 101 / 131 | 0.577 | 0.432 | 0.604 | 0.776 | 0.576 | 0.459 | 14 min |
  | s1280 | 4–7 | 75 / 105 | 0.589 | 0.454 | 0.626 | 0.789 | 0.560 | 0.527 | 16 min |
  | m1024 | 0–3 | 68 / 98 | 0.605 | 0.465 | 0.672 | 0.822 | 0.655 | 0.539 | 15 min |
  | m1280 | 4–7 | 78 / 108 | 0.631 | 0.491 | 0.663 | 0.818 | 0.607 | 0.563 | 24 min |

- yolo26m is clearly better than yolo26s (+0.04–0.07 title mAP50). 1024 vs 1280 on m is
  within val noise for titles (50 pages, 1198 TITLE boxes); 1280 wins overall mAP50 and
  mAP50-95, and the box sizes argue for it → **imgsz 1280**.
- Next: `m1280f` = same but patience 150 (full cosine schedule), and `l1280f` = yolo26l.

## Inference settings (sweep on val with m1280, `runs/eval/sweep_m1280_val.log`)

- `iou` has **no effect at all** (identical numbers for 0.5/0.6/0.7): YOLO26 is end-to-end,
  NMS-free, so there is no NMS threshold to tune. Only `conf` matters.
- Zone-level title F1 (GT zone gets the class of the best prediction with IoU ≥ 0.5):

  | conf | P | R | F1 | rule pairwise F1 |
  |---|---|---|---|---|
  | 0.15 | 0.860 | 0.792 | 0.825 | 0.426 |
  | 0.25 | 0.867 | 0.782 | 0.822 | 0.433 |
  | 0.35 | 0.884 | 0.773 | 0.825 | 0.434 |
  | 0.50 | 0.909 | 0.751 | 0.823 | 0.442 |

  Flat plateau; conf only trades precision for recall. **conf 0.35** is the compromise for
  the card; use 0.15–0.25 when recall matters more (a missed headline line splits a title run).
- Article rule on val with **ground-truth** classes: P 0.719, R 0.309, F1 0.432 — the val
  pages are full of listings with one headline per item under a single article id, so the
  rule's ceiling is low there. With m1280 predictions: 0.434, i.e. no measurable detector cost
  on val.

- Both stopped early (patience 30) around epoch 100–130, i.e. **before the cosine schedule
  reached its low-LR tail**. Worth one run with early stopping off once the size/imgsz choice
  is made. 1280 > 1024 as the box statistics predicted, mostly on INSIDEHEADING.
