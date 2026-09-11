| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |
|---|---|---|---|---|
| ground truth | 0.696 | 0.580 | **0.633** | 0.609 |
| predicted (`p2_combined_x4`, imgsz 1280, conf 0.35) | 0.670 | 0.434 | **0.527** | 0.588 |

Zone-level title assignment (ARTICLE-TITLE or SECTION-TITLE, after IoU ≥ 0.5 matching): P 0.833, R 0.782, F1 0.807 (833 tp, 167 fp, 232 fn over 11035 zones on 48 pages).
