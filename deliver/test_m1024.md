weights `runs/m1024/weights/best.pt`, split `test`, imgsz 1024, conf 0.001, iou 0.6, max_det 600

| class | instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| ARTICLE-TITLE | 1065 | 0.743 | 0.764 | 0.791 | 0.652 |
| SECTION-TITLE | 0 | – | – | – | – |
| ARTICLE-SUBTITLE | 350 | 0.618 | 0.649 | 0.610 | 0.459 |
| ARTICLE-INSIDEHEADING | 205 | 0.503 | 0.673 | 0.533 | 0.415 |
| HEADER-TITLE | 10 | 0.797 | 0.900 | 0.825 | 0.537 |
| HEADER-TEXT | 111 | 0.553 | 0.667 | 0.533 | 0.346 |
| ARTICLE-ILLUSTRATION | 148 | 0.739 | 0.811 | 0.809 | 0.685 |
| ADVERTISEMENT | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| ANNOUNCEMENT | 56 | 0.537 | 0.304 | 0.265 | 0.167 |
| ARTICLE-TEXT | 8504 | 0.859 | 0.736 | 0.792 | 0.626 |
| CAPTION | 70 | 0.741 | 0.643 | 0.663 | 0.412 |
| AUTHOR | 49 | 0.720 | 0.437 | 0.542 | 0.412 |
| ARTICLE-TABLE | 466 | 0.539 | 0.247 | 0.317 | 0.167 |
| **all (12 classes)** | 11035 | 0.696 | 0.569 | **0.557** | 0.407 |
| **title classes (3)** | | | | **0.645** | |
