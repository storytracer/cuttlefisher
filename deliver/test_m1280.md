weights `runs/m1280/weights/best.pt`, split `test`, imgsz 1280, conf 0.001, iou 0.6, max_det 600

| class | instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| ARTICLE-TITLE | 1065 | 0.747 | 0.798 | 0.795 | 0.675 |
| SECTION-TITLE | 0 | – | – | – | – |
| ARTICLE-SUBTITLE | 350 | 0.620 | 0.726 | 0.658 | 0.517 |
| ARTICLE-INSIDEHEADING | 205 | 0.469 | 0.717 | 0.557 | 0.454 |
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
