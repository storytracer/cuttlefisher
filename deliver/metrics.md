# cuttlefisher — metrics

Phase 1: YOLO26 trained on `Teklia/Newspapers-finlam` only (623 train pages, 149 newspapers).
All numbers from a separate single-GPU `evaluate.py` run (conf 0.001, max_det 600); the
training log's own final table double counts instances under DDP and is not used.

## Model selection (val split, 50 pages)

"Title mAP50" = mean AP50 of ARTICLE-TITLE, ARTICLE-SUBTITLE, ARTICLE-INSIDEHEADING
(SECTION-TITLE has no instances in this dataset). All runs: batch 32 on 4 GPUs (8 per GPU),
cos_lr, cache=ram, pretrained COCO weights, patience 30 unless noted.

| run | model | imgsz | best ep / stopped | val mAP50 | val mAP50-95 | title mAP50 | TITLE | SUBTITLE | INSIDEHEADING |
|---|---|---|---|---|---|---|---|---|---|
| s1024 | yolo26s | 1024 | 101 / 131 | 0.577 | 0.432 | 0.604 | 0.776 | 0.576 | 0.459 |
| s1280 | yolo26s | 1280 | 75 / 105 | 0.589 | 0.454 | 0.626 | 0.789 | 0.560 | 0.527 |
| m1024 | yolo26m | 1024 | 68 / 98 | 0.605 | 0.465 | **0.672** | **0.822** | **0.655** | 0.539 |
| **m1280** | yolo26m | 1280 | 78 / 108 | **0.631** | 0.491 | 0.663 | 0.818 | 0.607 | **0.563** |
| m1280f | yolo26m | 1280 | 150 / 150 (patience off) | 0.619 | 0.491 | 0.628 | 0.792 | 0.554 | 0.538 |
| l1280f | yolo26l | 1280 | 150 / 150 (patience off) | 0.632 | **0.513** | 0.614 | 0.788 | 0.559 | 0.493 |

m1024 and m1280 tie on title mAP50 within the noise of 50 pages; m1280 was chosen on the
secondary val criteria (overall mAP50, mAP50-95) and the box-size statistics (median title
box is 14 px tall at the native 2000 px height) before the test split was looked at.
Running the full cosine schedule or the larger yolo26l did not help the title classes.

## Test split (48 pages), per class — `m1280` (delivered as `best.pt`)

weights `runs/m1280/weights/best.pt`, imgsz 1280, conf 0.001, max_det 600

| class | instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| ARTICLE-TITLE | 1065 | 0.747 | 0.798 | **0.795** | 0.675 |
| SECTION-TITLE | 0 | – | – | – | – |
| ARTICLE-SUBTITLE | 350 | 0.620 | 0.726 | **0.658** | 0.517 |
| ARTICLE-INSIDEHEADING | 205 | 0.469 | 0.717 | **0.557** | 0.454 |
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

Same protocol for `m1024` (imgsz 1024): mAP50 0.557, title mAP50 0.645 (TITLE 0.791,
SUBTITLE 0.610, INSIDEHEADING 0.533); full table in `test_m1024.md`.

Reading: ARTICLE-TITLE is at 0.80 AP50 with balanced P/R; SUBTITLE and INSIDEHEADING are
recalled well (0.72) but confused with each other and with TITLE (precision 0.47–0.62).
ARTICLE-TABLE (0.30) and ANNOUNCEMENT (0.35) are the weak layout classes; ADVERTISEMENT has
35 train and 1 test instance and cannot be judged here. The paper's 72.3 mAP50 is on the
single-newspaper La Liberté set and is not comparable.

## Article rule (test split, ground-truth zones and reading order)

A new article starts at every run of ARTICLE-TITLE / SECTION-TITLE zones in reading order;
each ground-truth zone takes the class of the best-overlapping prediction (IoU ≥ 0.5, else
ARTICLE-TEXT). Score: pairwise F1 over same-article zone pairs against the true `article_id`.

| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |
|---|---|---|---|---|
| ground truth (ceiling of the rule) | 0.696 | 0.580 | **0.633** | 0.609 |
| predicted, `m1280`, imgsz 1280, conf 0.35 | 0.700 | 0.501 | **0.584** | 0.610 |
| predicted, `m1024`, imgsz 1024, conf 0.35 | 0.700 | 0.471 | 0.563 | 0.615 |

Zone-level title assignment with m1280 (is the zone a title after matching?): P 0.851,
R 0.763, F1 0.805 (813 tp, 142 fp, 252 fn over 11 035 zones).

The detector costs **0.05 pairwise F1** against the rule's own ceiling. The loss is entirely
recall: a headline line that is missed or matched to a non-title prediction splits a
multi-line title run into two articles. The ceiling itself is low because 246 of the 641
test articles have no title zone at all and ~85 contain several title runs (listings).

## Phase 2: adding La Liberté (7 957 pages of one newspaper, 1925–28)

Same recipe (yolo26m, imgsz 1280, batch 32 on 4 GPUs, cos_lr), 40 epochs with patience 10,
La Liberté's 16 classes mapped onto the 13 (its released annotations populate only 9 of
them — no SECTION-TITLE, ADVERTISEMENT, ANNOUNCEMENT, CAPTION, AUTHOR — and label
advertisements as text). Two variants: plain concatenation, and FINLAM oversampled ×4 to
counter the 13:1 page imbalance. Evaluated on the FINLAM splits only.

| model | val mAP50 | val title mAP50 | test mAP50 | test mAP50-95 | test title mAP50 | TITLE | SUBTITLE | INSIDEHEADING | article rule F1 | zone title F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| phase 1 `m1280` | 0.631 | 0.663 | 0.579 | 0.429 | **0.670** | 0.795 | **0.658** | 0.557 | **0.584** | 0.805 |
| `p2_combined` | 0.596 | 0.637 | 0.552 | 0.441 | 0.668 | **0.812** | 0.567 | **0.624** | 0.542 | 0.796 |
| `p2_combined_x4` | **0.648** | 0.661 | **0.585** | **0.456** | 0.642 | 0.783 | 0.553 | 0.589 | 0.527 | 0.807 |

Other test classes, phase 1 → plain → ×4: ARTICLE-TEXT 0.811 → 0.831 → 0.835,
ILLUSTRATION 0.854 → 0.869 → 0.876, ARTICLE-TABLE 0.303 → 0.338 → 0.372,
CAPTION 0.664 → 0.599 → 0.699, ANNOUNCEMENT 0.345 → 0.291 → 0.382, AUTHOR 0.519 → 0.508 → 0.525.

**Verdict: the extra 8 000 single-title pages do not help the diverse test split where it
matters.** Title mAP50 is flat or slightly down, ARTICLE-SUBTITLE loses 0.09 (the two
datasets' subtitle conventions differ), and the article rule gets worse (0.584 → 0.53–0.54)
because the phase-2 models emit more spurious title zones at equal recall. They do improve the
generic layout classes (text, illustration, table) and mAP50-95, and oversampling FINLAM ×4 is
necessary to keep the FINLAM-only classes (caption, announcement) from being diluted away.

Phase 1 `m1280` remains the delivered `best.pt`; `p2_combined_x4` is on the Hub under
`phase2/` for users who want the better layout classes. Tables: `test_p2_*.md`,
`article_rule_p2_*.md`, curves in `curves_phase2.png`.

## Inference settings

`iou` has no effect (YOLO26 is end-to-end, NMS-free). `conf` only trades precision for
recall at a flat zone-level title F1 of 0.82–0.83 on val (conf 0.15 → P 0.86 / R 0.79;
conf 0.5 → P 0.91 / R 0.75). Recommended: `imgsz=1280, conf=0.35, max_det=600`; lower conf
to 0.15–0.25 when missed headline lines hurt more than spurious titles.

## Files

- `test_m1280.md/.json`, `test_m1024.md/.json` — per-class tables
- `article_rule_m1280.md`, `article_rule_m1024.md` — article-rule tables
- `curves.png` — val mAP50, mAP50-95 and losses of the six runs
- `pages/` — six test pages, `_pred.jpg` (m1280, conf 0.35) next to `_gt.jpg`
- `classes.txt` — class names in id order
