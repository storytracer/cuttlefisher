# Integrating cuttlefisher (phase 1) into SquiddleOCR

Report for the SquiddleOCR Claude Code session. Source: https://github.com/storytracer/cuttlefisher

## What it is

A YOLO26-m block detector for historical newspaper pages, trained on
`Teklia/Newspapers-finlam` (623 pages, 149 French/English newspapers, 19th–20th c.). It
classifies layout blocks into 13 classes; the ones SquiddleOCR needs are the title classes,
so it can cut articles with "a new article starts at every title in reading order"
(Mocaër et al., ICDAR-HIP 2026, arXiv:2607.15082).

Weights: https://huggingface.co/storytracer/cuttlefisher — file `best.pt` (44 MB, plain
ultralytics checkpoint, no export needed). Phase-1 commit on the Hub:
`4fa084d297cacf3c548adc2700c6350fef437229` — pin it with `revision=` so a later phase-2
upload cannot change behaviour under you.

## Environment (DGX Spark, aarch64, torch cu130)

- `ultralytics >= 8.4.146` (YOLO26 needs an 8.4 release; the checkpoint was saved with
  8.4.146 and torch 2.14). `huggingface_hub` for the download.
- The `.pt` is architecture-independent; aarch64 + cu130 is fine. CPU also works, just slow.

## Inference

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

REV = "4fa084d297cacf3c548adc2700c6350fef437229"  # phase 1
model = YOLO(hf_hub_download("storytracer/cuttlefisher", "best.pt", revision=REV))

res = model.predict(page_image, imgsz=1280, conf=0.35, max_det=600, verbose=False)[0]
boxes = res.boxes.xyxy.cpu().numpy()      # (n, 4) pixels in the input image
classes = res.boxes.cls.cpu().numpy().astype(int)
scores = res.boxes.conf.cpu().numpy()
names = model.names                        # {0: 'HEADER-TITLE', ...}
```

Settings, and why:

| arg | value | why |
|---|---|---|
| `imgsz` | **1280** | trained at 1280; title boxes are ~14 px tall at 2000 px page height, 1024 loses them |
| `conf` | **0.35** (0.15–0.25 for more recall) | flat zone-level title F1 0.82–0.83 across 0.15–0.5; conf only trades P for R (0.35 → P 0.88 / R 0.77 on val) |
| `max_det` | **600** | dense pages carry up to 500 blocks; the default 300 silently caps recall |
| `iou` | leave default | no effect: YOLO26 is end-to-end, NMS-free |
| `batch` | as you like | predictions are per image; batching only speeds things up |

Pass the full-resolution page (any size; ultralytics letterboxes to 1280 on the long side).
The training pages were 2000 px high scans of full broadsheet pages; a full page, not a column
crop, is the expected input.

## Classes (ids are fixed, `classes.txt` on the Hub)

```
0 HEADER-TITLE   1 HEADER-TEXT   2 ARTICLE-ILLUSTRATION   3 ADVERTISEMENT   4 ANNOUNCEMENT
5 ARTICLE-TITLE  6 ARTICLE-TEXT  7 ARTICLE-SUBTITLE  8 ARTICLE-INSIDEHEADING
9 CAPTION  10 AUTHOR  11 ARTICLE-TABLE  12 SECTION-TITLE
```

- **Title classes for the article cut: `5 ARTICLE-TITLE` (and `12 SECTION-TITLE`).**
  `12` is never predicted by this model — the class has zero instances in the training
  data — but keep it in the rule so a phase-2 model can fill it in.
- `7 ARTICLE-SUBTITLE` and `8 ARTICLE-INSIDEHEADING` are *not* article starts: a subtitle
  belongs to the headline above it, an inside-heading is a crosshead inside an article.
- `3 ADVERTISEMENT` is unreliable (35 training boxes). `11 ARTICLE-TABLE` is weak (AP50 0.30).

## Two things about the boxes that matter for matching against eynollah regions

1. **Titles are one box per line.** A three-line headline comes back as three
   `ARTICLE-TITLE` boxes stacked. Text comes back as paragraph blocks. So an eynollah
   heading region that spans a whole headline will not have IoU ≥ 0.5 with any single
   predicted box.
2. Consequently, do **not** assign classes to eynollah regions by best-IoU alone.
   Recommended assignment, per eynollah region R:
   - for each class c, `cover[c]` = area of R covered by the union of predicted boxes of
     class c (clip boxes to R; sum is fine, overlaps between same-class boxes are rare);
   - if `cover[TITLE] / area(R) >= 0.5` → R is a title; else if some other class covers
     ≥ 0.5 use it; else ARTICLE-TEXT;
   - alternatively, keep eynollah's text-line segmentation and assign each *line* the class
     of the predicted box with the largest overlap — this matches the model's granularity
     exactly and is what the cuttlefisher evaluation does (IoU ≥ 0.5, else ARTICLE-TEXT).

## The article rule

Given zones in reading order with a class each:

```python
TITLE = {5, 12}
def cut_articles(classes):
    art, cur, prev_title = [], -1, False
    for c in classes:
        is_title = c in TITLE
        if is_title and not prev_title:   # a *run* of title zones starts one article
            cur += 1
        if cur < 0:                       # page starts without a title
            cur = 0
        art.append(cur)
        prev_title = is_title
    return art
```

The "run" matters: consecutive title zones (the lines of one headline, or headline +
kicker) are one article start, not several. A subtitle between two title lines breaks the
run in the reference implementation — if your reading order interleaves them, treat
`{5, 7, 12}` as run-continuing but only `{5, 12}` as run-starting.

## What to expect (FINLAM test split, 48 diverse pages)

- ARTICLE-TITLE AP50 0.795 (P 0.75 / R 0.80 at conf 0.001; P 0.85 / R 0.76 zone-level at
  conf 0.35); SUBTITLE 0.658; INSIDEHEADING 0.557; all-class mAP50 0.579.
- Article rule with ground-truth zones and order: pairwise same-article F1 **0.584** with
  predicted classes vs **0.633** with ground-truth classes — the detector costs ~0.05.
  The rest of the gap to 1.0 is the rule itself: 38 % of FINLAM articles have no title zone
  (continuations, briefs) and some have many (listings).
- Failure modes seen: a missed middle line of a headline splits it into two articles (hence
  the "run" logic and the lower-conf option); subtitles and inside-headings are confused with
  each other and with titles; banner headlines spanning the page get low confidence (~0.35).

## Later

Phase 2 (FINLAM + 8k La Liberté pages) is training. If it improves the diverse test split it
will be uploaded to the same Hub repo under a second file name, and the model card will say
which is which; `best.pt` at the pinned revision above stays the phase-1 model.
