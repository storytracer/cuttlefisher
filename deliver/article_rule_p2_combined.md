| classes used by the rule | pairwise P | pairwise R | pairwise F1 (micro) | mean page F1 |
|---|---|---|---|---|
| ground truth | 0.696 | 0.580 | **0.633** | 0.609 |
| predicted (`p2_combined`, imgsz 1280, conf 0.35) | 0.784 | 0.414 | **0.542** | 0.600 |

Zone-level title assignment (ARTICLE-TITLE or SECTION-TITLE, after IoU ≥ 0.5 matching): P 0.809, R 0.784, F1 0.796 (835 tp, 197 fp, 230 fn over 11035 zones on 48 pages).
