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

  | m1280f | 0–3 | 150 / 150 | 0.619 | 0.491 | 0.628 | 0.792 | 0.554 | 0.538 | 33 min |
  | l1280f | 4–7 | 150 / 150 | 0.632 | 0.513 | 0.614 | 0.788 | 0.559 | 0.493 | 39 min |

- Running the full cosine schedule (patience 150) did **not** help the title classes
  (0.628 vs 0.663 for the early-stopped m1280); mAP50-95 crept up. yolo26l likewise: best
  overall mAP50-95 (0.513) but the lowest title mAP50. Neither is worth the size. Dead end.
- **Selection (phase 1): `m1280`.** On the stated criterion (val title mAP50) m1024 and m1280
  tie within noise (0.672 vs 0.663 on 50 pages); the tie-break was decided before looking at
  test, on val overall mAP50 (0.631 vs 0.605), mAP50-95 (0.491 vs 0.465) and the box-size
  statistics. Test numbers, computed afterwards for both, confirm it (title mAP50 0.670 vs
  0.645). Both test tables are in `deliver/`.

## Phase 1 test results (48 pages, `deliver/test_m1280.md`)

- m1280: mAP50 0.579 all classes, title mAP50 0.670; ARTICLE-TITLE AP50 0.795 (P 0.75 /
  R 0.80), SUBTITLE 0.658, INSIDEHEADING 0.557. Weak classes: ARTICLE-TABLE 0.30 (recall
  0.26), ANNOUNCEMENT 0.35, HEADER-TEXT 0.46; ADVERTISEMENT has 1 test instance, not
  meaningful. The paper's 72.3 mAP50 is on La Liberté (single title, 8k pages); not comparable.
- Article rule on test (conf 0.35): ground-truth classes F1 0.633, predicted 0.584 (m1024:
  0.563). **The detector costs 0.05 pairwise F1**; zone-level title F1 0.805 (P 0.85, R 0.76).
  The loss is all recall (0.58 → 0.50): a missed headline line inside a multi-line headline
  splits the title run into two articles.

## Phase 1 published

- https://huggingface.co/storytracer/cuttlefisher — `best.pt` = m1280, with `args.yaml`,
  `classes.txt`, `results_test.md/.json`, `article_rule_test.md`, model card (`hf/README.md`).

## Phase 2 data: Teklia/Newspapers-finlam-La-Liberte (2026-09-10)

- 22 parquet files, 9.4 GB, 7957 / 446 / 433 pages, JPEG 2500 px high (~1775 wide).
- Schema differs from the small set: coordinates are **fractions 0–1** (not percentages),
  article column is `zone_article_ids` (33 399 of 1.81 M zones have `None`), extra
  `zone_section_ids` (mostly None), no `newspaper_name`. `convert.py` handles all three.
- **Only 9 of the 16 classes have instances.** Full train/val/test counts of the source
  classes: HEADER-TITLE 1340/75/74, HEADER-TEXT 10452/561/634, ILLUSTRATION 21221/1207/1085,
  TITLE 133971/7498/7382, TEXT 1376516/77979/75032, SUBTITLE 34953/1929/2021,
  INSIDEHEADING 41155/2394/2232, TABLE 2818/167/152, ILLUSTRATEDTEXT 5622/328/339.
  **SECTION-TITLE, ADVERTISEMENT, ANNOUNCEMENT, CAPTION, AUTHOR, TABLECONTENT, ASIDE: 0.**
  So SECTION-TITLE cannot be learned from either dataset; class id 12 stays empty.
- Mapping verified on 3 rendered train pages (`checks/laliberte/`): boxes tight, titles are
  one box per line like FINLAM, a section head ("LA VIE FINANCIÈRE") is TITLE as in FINLAM.
  **Label conflict:** La Liberté annotates advertisements as TEXT + ILLUSTRATION, never
  ADVERTISEMENT, whereas FINLAM has an ADVERTISEMENT class (35 train boxes). Phase 2 will
  push ads toward ARTICLE-TEXT.
- Class dilution: 7957 single-newspaper pages vs 623 diverse ones; the FINLAM-only classes
  (ANNOUNCEMENT, CAPTION, AUTHOR, ADVERTISEMENT) would be 13× rarer per epoch. Phase 2 runs
  both **plain concatenation** and **FINLAM oversampled ×4** (the train txt lists the FINLAM
  images four times) to separate "more data" from "dilution".

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
