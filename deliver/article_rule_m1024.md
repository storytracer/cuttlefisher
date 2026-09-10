| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |
|---|---|---|---|---|
| ground truth | 0.696 | 0.580 | **0.633** | 0.609 |
| predicted (`m1024`, imgsz 1024, conf 0.35) | 0.700 | 0.471 | **0.563** | 0.615 |

Zone-level title assignment (ARTICLE-TITLE or SECTION-TITLE, after IoU ≥ 0.5 matching): P 0.842, R 0.739, F1 0.787 (787 tp, 148 fp, 278 fn over 11035 zones on 48 pages).
