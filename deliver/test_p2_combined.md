weights `runs/p2_combined/weights/best.pt`, split `test`, imgsz 1280, conf 0.001, iou 0.6, max_det 600

| class | instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| ARTICLE-TITLE | 1065 | 0.742 | 0.795 | 0.812 | 0.713 |
| SECTION-TITLE | 0 | – | – | – | – |
| ARTICLE-SUBTITLE | 350 | 0.626 | 0.574 | 0.567 | 0.479 |
| ARTICLE-INSIDEHEADING | 205 | 0.629 | 0.556 | 0.624 | 0.536 |
| HEADER-TITLE | 10 | 0.645 | 0.700 | 0.700 | 0.572 |
| HEADER-TEXT | 111 | 0.571 | 0.441 | 0.457 | 0.308 |
| ARTICLE-ILLUSTRATION | 148 | 0.832 | 0.831 | 0.893 | 0.750 |
| ADVERTISEMENT | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| ANNOUNCEMENT | 56 | 0.741 | 0.250 | 0.291 | 0.198 |
| ARTICLE-TEXT | 8504 | 0.873 | 0.746 | 0.834 | 0.694 |
| CAPTION | 70 | 0.705 | 0.586 | 0.599 | 0.457 |
| AUTHOR | 49 | 0.736 | 0.404 | 0.508 | 0.419 |
| ARTICLE-TABLE | 466 | 0.569 | 0.216 | 0.338 | 0.163 |
| **all (12 classes)** | 11035 | 0.639 | 0.508 | **0.552** | 0.441 |
| **title classes (3)** | | | | **0.668** | |
