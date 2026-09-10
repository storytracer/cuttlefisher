| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |
|---|---|---|---|---|
| ground truth | 0.696 | 0.580 | **0.633** | 0.609 |
| predicted (`m1280`, imgsz 1280, conf 0.35) | 0.700 | 0.501 | **0.584** | 0.610 |

Zone-level title assignment (ARTICLE-TITLE or SECTION-TITLE, after IoU ≥ 0.5 matching): P 0.851, R 0.763, F1 0.805 (813 tp, 142 fp, 252 fn over 11035 zones on 48 pages).
