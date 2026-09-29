# What each part of the 624-term formula does (8 particles)

*tuned on the network's predictions (from 100 if-statements per neuron, pruned)*. Validation accuracy 65.52%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a clipped sum of if-statements on jet-shape quantities), and each class score adds some scales and subtracts others; the formula makes the same decision as the original network on about 90% of test jets. Tops are recognised by size: the t score subtracts compactness (neuron 13, on which tops sit far below every other type, -46%) and adds overall size (neuron 10, +20%), so it closely follows girth, width and e2. Quarks and gluons both sit high on the narrow-and-light scale (neuron 9), which feeds both of their scores; the g score also adds the light-jet scale (neuron 2) and the moderate-mass scale (neuron 1, on which quarks sit lowest) and subtracts quark-likeness (neuron 5: pT concentrated in the few hardest particles), while the q score subtracts the two-prong scale (neuron 4) and overall size (neuron 10). W and Z jets sit high on scales for a compact, elongated two-prong jet (neurons 7 and 13 feed both boson scores, neuron 0 and the compact-centred neuron 11 only the W score), and both boson scores subtract the broad-spread and wide-angle-radiation scales (neurons 6 and 3), on which tops and gluons sit much higher. W and Z are split mainly by width: Z jets sit highest on an intermediate-width scale (neuron 14), which the Z score adds (+5%) and the W score subtracts (-8%), and the 'wider than a W' scale (neuron 15) takes more out of the W score (-7%) than out of the Z score (-2%).

## The 5 class scores

### score g: gluon: light jet, many hard particles

High for light jets whose pT is spread over many hard particles (it rises with z_7 and planar flow and falls with the pT share of the 5 hardest particles); it averages 2.04 for g jets, 1.15 for q and 1.05 for t, and is near zero for W and Z (AUC 0.82 for gluons against the rest).

Adds the light-jet scale (neuron 2, +32%), narrow-light-ness (neuron 9, +22%) and the moderate-mass scale (neuron 1, +15%), plus a little broad spread (neuron 6, +5%); subtracts quark-likeness (neuron 5, -16%) and, less, the compact two-prong scale (neuron 0, -5%) and the two-prong substructure scale (neuron 4, -4%).

*computed:* largest for g (2.04), then q (1.15), then t (1.05), then W (0.09), then Z (-0.03); it separates g jets from the rest best (AUC 0.82: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets (it falls with girth, LHA and lam1, rank correlations about -0.67); it averages 1.98 for q jets and 1.36 for g, with tops lower (0.18) and W and Z near or below zero (AUC 0.85 for quarks). Gluons are its main confusion.

Mostly adds narrow-light-ness (neuron 9, +49%); subtracts the two-prong substructure scale (neuron 4, -18%) and overall size (neuron 10, -14%); adds a little of broad spread (neuron 6, +10%), quark-likeness (neuron 5, +6%) and the narrow centred core (neuron 8, +2%).

*computed:* largest for q (1.98), then g (1.36), then t (0.18), then W (0.06), then Z (-0.18); it separates q jets from the rest best (AUC 0.85: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets that are not wider than a W; it averages 2.40 for W jets and 0.82 for Z, is slightly negative for q and g, and strongly negative for t (-2.64) (AUC 0.89).

Adds the compact-centred scale (neuron 11, +25%, its largest input), the two-prong scales (neurons 0 and 7, +9% each) and compactness (neuron 13, +8%); subtracts broad off-centre spread (neuron 6, -12%), wide-angle radiation (neuron 3, -12%), the 'wider than a W' scales (neurons 14 and 15) and a little of neurons 8 and 9.

*computed:* largest for W (2.40), then Z (0.82), then q (-0.15), then g (-0.49), then t (-2.64); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: two-prong jet, wider than a W

High for compact two-prong jets of W/Z size with an intermediate, Z-sized width; it averages 2.22 for Z jets and 1.36 for W (W is its main confusion), is slightly negative for q and g, and strongly negative for t (-2.18) (AUC 0.86).

Adds the two-prong W/Z scale (neuron 7, +26%, its largest input), two-prong substructure (neuron 4, +10%), compactness (neuron 13, +9%), the intermediate-width scale (neuron 14, +5%) and the moderate-mass scale (neuron 1, +5%); subtracts broad off-centre spread (neuron 6, -20%), wide-angle radiation (neuron 3, -18%) and a little of neurons 9 and 15.

*computed:* largest for Z (2.22), then W (1.36), then q (-0.06), then g (-0.32), then t (-2.18); it separates Z jets from the rest best (AUC 0.86: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and e2 with rank correlations of about 0.89-0.91. It averages 2.86 for t jets, 0.42 for Z and 0.16 for W, is near zero for g and negative for q (-1.40) (AUC 0.91).

Subtracts compactness (neuron 13, -46%) and adds overall size (neuron 10, +20%); also subtracts quark-likeness (neuron 5, -16%) and adds two-prong substructure (neuron 4, +12%) and a little of the narrow-core scale (neuron 8, +3%).

*computed:* largest for t (2.86), then Z (0.42), then W (0.16), then g (0.06), then q (-1.40); it separates t jets from the rest best (AUC 0.91: large for t)

## The 16 neurons (most important first)

### neuron 1: moderate mass and spread, hard jet (major)

- **What it measures:** Pushed up for small e2_sq (< 0.00817, its strongest term) and for high total pT (log of total pT > 6.38), and pushed down for narrow (width < 0.00868), light (mass < 53.3 GeV) and small jets (max ΔR < 0.251); it rises with mass, lam1 and width (rank correlations 0.527, 0.448, 0.447). Tops (1.37) and Z jets (1.32) sit highest, gluons (1.00) and W (0.90) in the middle, quarks lowest (0.53, zero for 58.5% of them).
- *computed — its value:* largest for t (1.37), then Z (1.32), then g (1.00), then W (0.90), then q (0.53); it separates q jets from the rest best (AUC 0.33: small for q)
- **How the class scores use it:** It raises the g score (+15%), where it marks the difference between gluons and the quarks that sit lowest on it, and it raises the Z score a little (+5%). It does not enter the q, W or t scores; the tops that also sit high on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+15%), Z (+5%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.974
if e2_sq < 0.0082: z += 338 × (0.0082 − e2_sq)
if width < 0.0087: z += -227 × (0.0087 − width)
if mass < 53.33: z += -0.051 × (53.33 − mass)
if log_sum_pt > 6.38: z += 4.63 × (log_sum_pt − 6.38)
if max_dr < 0.251: z += -7.17 × (0.251 − max_dr)
if lam1 < 0.0084 and lam2 < 0.00054: z += -433672 × (0.0084 − lam1) × (0.00054 − lam2)
if z_7 < 0.056: z += -74.21 × (0.056 − z_7)
if pt_7 > 34.53: z += 0.154 × (pt_7 − 34.53)
if log_sum_pt > 6.38 and max_dr < 0.198: z += 20.74 × (log_sum_pt − 6.38) × (0.198 − max_dr)
if mass < 49.67: z += 0.025 × (49.67 − mass)
if pt_7 > 34.53 and mass < 91.19: z += -0.0017 × (pt_7 − 34.53) × (91.19 − mass)
if log_sum_pt > 6.64: z += 6.78 × (log_sum_pt − 6.64)
if log_sum_pt > 6.57 and lam2 < 0.0011: z += 4123 × (log_sum_pt − 6.57) × (0.0011 − lam2)
if z_7 < 0.056 and girth2_top2 < 0.014: z += 2791 × (0.056 − z_7) × (0.014 − girth2_top2)
if sum_pt_top5 > 580: z += -0.0037 × (sum_pt_top5 − 580)
if log_sum_pt > 6.57: z += 3.42 × (log_sum_pt − 6.57)
if z_7 < 0.043: z += -58.19 × (0.043 − z_7)
if e2_sq < 0.0082 and planar_flow < 0.084: z += -5280 × (0.0082 − e2_sq) × (0.084 − planar_flow)
if log_sum_pt > 6.38 and centroid_offset < 0.024: z += -90.91 × (log_sum_pt − 6.38) × (0.024 − centroid_offset)
if LHA > 0.267: z += 8.11 × (LHA − 0.267)
if e2 < 0.036 and eccentricity > 0.978: z += 7155 × (0.036 − e2) × (eccentricity − 0.978)
if z_7 < 0.056 and lam2 < 0.0034: z += 6993 × (0.056 − z_7) × (0.0034 − lam2)
if log_sum_pt > 6.64 and girth2_top3 < 0.0079: z += -581 × (log_sum_pt − 6.64) × (0.0079 − girth2_top3)
if lam1 < 0.0084 and n_pt_above_50 > 6.00: z += -116 × (0.0084 − lam1) × (n_pt_above_50 − 6.00)
if e2 > 0.032: z += -15.97 × (e2 − 0.032)
if e2_sq < 0.0082 and n_pt_above_50 > 6.00: z += 92.10 × (0.0082 − e2_sq) × (n_pt_above_50 − 6.00)
if LHA < 0.347 and planar_flow < 0.084: z += -87.80 × (0.347 − LHA) × (0.084 − planar_flow)
if lam1 < 0.0084 and centroid_offset > 0.021: z += 13737 × (0.0084 − lam1) × (centroid_offset − 0.021)
if pt_7 > 34.53 and max_dr < 0.081: z += 0.941 × (pt_7 − 34.53) × (0.081 − max_dr)
if pt_7 > 53.44: z += -0.120 × (pt_7 − 53.44)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (70 GeV), very wide, pT spread over several particles, low pT** — 20.2% of jets, neuron 1.34. Mostly t (68%) with some g and Z; mass 69.6 GeV, width 0.0186, much of the pT 0.1-0.2 from the axis, and a softer total pT (555 GeV). Only this group passes LHA > 0.267 (98%, +0.895), with e2 > 0.0323 (-0.493) against it; e2_sq < 0.00817 passes just 17%, so the main positive term is mostly missing, but so are the light and narrow penalties (mass < 53.3 at 22%, width < 0.00868 at 16%). The result is 1.336 (on for 94%), which raises the g score (+0.522) and Z score (+0.167); the formula calls them t.
- **very light (12 GeV), very narrow, pT spread over several particles** — 14.0% of jets, neuron 0.15. Mostly g (50%) with quarks (25%); mass 11.7 GeV, width 0.0009, 60% of the pT within 0.025, and a softer leading particle (168.92 GeV). e2_sq < 0.00817 (+2.590) and mass < 49.7 (+0.959) are cancelled by mass < 53.3 (-2.137), width < 0.00868, lam1 < 0.00838 and lam2 < 0.000537 and max_dr < 0.251, which all pass. With log_sum_pt > 6.38 passing only 63% and pt_7 > 34.5 largely offset by pt_7 > 34.5 and mass < 91.2, the neuron stays low (0.146, on for 16%); the formula calls them g.
- **light (39 GeV), average width, pT spread over several particles, low pT** — 12.5% of jets, neuron 0.46. Mostly W (37%) with Z (23%), g and t mixed in; mass 39.4 GeV, width 0.0051, with 52% of the pT 0.05-0.1 from the axis and below-average total pT (596 GeV). e2_sq < 0.00817 always passes (+1.246) but max_dr < 0.251, width < 0.00868 and mass < 53.3 (which passes 93%) take about the same back. What decides whether it turns on are the pT terms log_sum_pt > 6.38 and pt_7 > 34.5, each passing under 60%, so the value is 0.459 and on for half; it mildly raises the g and Z scores and the formula calls them W.
- **medium-mass (65 GeV), average width, pT spread over several particles** — 11.0% of jets, neuron 1.31. Mostly Z (43%) with W (34%) and t (14%); mass 64.6 GeV, width 0.0069, total pT 808 GeV with a hard leading particle (305.89 GeV). log_sum_pt > 6.38 passes almost always (+1.449) while z_7 < 0.0556 always passes (-1.402); mass < 53.3 passes only 18%, so the light-jet penalty is mostly avoided, and e2_sq < 0.00817 (87%) adds +0.717. The neuron reaches 1.312 (on for 83%), raising the g (+0.513) and Z (+0.164) scores; the formula splits them between W and Z.
- **very light (10 GeV), very narrow, pT spread over several particles, high pT** — 9.6% of jets, neuron 0.47. Mostly q (51%) with gluons (24%); mass 10.2 GeV, width 0.0004, 85% of the pT within 0.025, and a hard leading particle (310.54 GeV). The positive terms e2_sq < 0.00817 (+2.683), log_sum_pt > 6.38 (+1.606) and log_sum_pt > 6.38 and max_dr < 0.198 (+1.146) are balanced by mass < 53.3 (-2.213), width < 0.00868, lam1 < 0.00838 and lam2 < 0.000537, max_dr < 0.251 and z_7 < 0.0556, which all pass. The near cancellation leaves 0.473 and on for 43%; the formula calls them q.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 9.1% of jets, neuron 2.24. A W/Z mixture (37% each) with t (14%); mass 57.8 GeV, width 0.0075, pT shared more evenly than usual (leading particle 172.97 GeV, 4th 81.1 GeV). pt_7 > 34.5 passes for all of them (+2.430), partly returned by pt_7 > 34.5 and mass < 91.2 (-0.871), while z_7 < 0.0556 passes only 7.7% and mass < 53.3 42%, so the usual penalties are small. This gives the neuron's highest group value, 2.242 (on for 92%), raising the g (+0.876) and Z (+0.28) scores; the formula calls them W.
- **very light (10 GeV), very narrow, leading particle 48% of pT, high pT** — 7.6% of jets, neuron 0.70. Mostly q (69%); mass 9.9 GeV, width 0.0002, 96% of the pT within 0.025, total pT 990 GeV with a leading particle of 472.48 GeV. The high-pT terms log_sum_pt > 6.38 (+2.383), log_sum_pt > 6.64 (+1.725) and log_sum_pt > 6.38 and max_dr < 0.198 (+1.577) plus e2_sq < 0.00817 (+2.715) fight z_7 < 0.0556 (-2.748), mass < 53.3, width < 0.00868 and lam1 < 0.00838 and lam2 < 0.000537. The result is 0.703 and on for 47%; the formula calls them q.
- **medium-mass (58 GeV), narrow, leading particle 48% of pT, high pT** — 6.4% of jets, neuron 1.65. Mostly W (44%) with Z (35%); mass 57.8 GeV, narrow (width 0.0042), total pT 931 GeV with a leading particle of 447.64 GeV and 54% of the pT 0.025-0.05 from the axis. The pT terms log_sum_pt > 6.38 (+2.103), log_sum_pt > 6.64 and log_sum_pt > 6.57 and lam2 < 0.00113 pass almost always, and e2 < 0.0356 and eccentricity > 0.978 adds +1.133 (76%), outweighing z_7 < 0.0556 (-2.387) and z_7 < 0.043. mass < 53.3 passes only 39%, so the neuron reaches 1.646 (on for 92.5%), raising the g and Z scores; the formula calls them W.
- **very light (11 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 1.45. Mostly g (54%) with quarks (27%); mass 10.9 GeV, width 0.0005, with pT spread evenly over the particles (leading 202.32 GeV, 4th 92.66 GeV) and 78% of it within 0.025. pt_7 > 34.5 always passes (+3.294), mostly cancelled by pt_7 > 34.5 and mass < 91.2 (-2.817); e2_sq < 0.00817 (+2.655) and e2_sq < 0.00817 and n_pt_above_50 > 6 (+1.285) against mass < 53.3, width < 0.00868, lam1 < 0.00838 and lam2 < 0.000537 and lam1 < 0.00838 and n_pt_above_50 > 6. The balance is 1.451 (on for 63.5%), raising the g score (+0.567) and Z score; the formula calls them g.
- **light (29 GeV), narrow, pT spread over several particles** — 4.4% of jets, neuron 1.20. A mixture: W 34%, Z 27%, g 16%, q 15%; mass 29.2 GeV, width 0.0029, with 63% of the pT 0.025-0.05 from the axis. The group is set by e2 < 0.0356 and eccentricity > 0.978, which passes for all of them (+2.325), plus e2_sq < 0.00817 (+2.089); against it e2_sq < 0.00817 and planar_flow < 0.0837 (-1.812) and LHA < 0.347 and planar_flow < 0.0837 also always pass, together with the light and narrow penalties. The net is 1.197 (on for 72%), raising the g and Z scores; the formula calls them W.

### neuron 2: light jet with hard trailing particles (major)

- **What it measures:** Pushed down for large angularity (LHA > 0.112, its strongest term), for a soft 8th-hardest particle (pT_7 < 53.4 GeV) and for very light jets (mass < 36.2 GeV), and pushed up for mass < 69.6 GeV and total pT < 788 GeV; overall it falls steeply with jet mass (rank correlation -0.86). Gluons (3.22) and quarks (3.02) sit highest, well above W (1.43), Z (1.16) and top (1.05) jets.
- *computed — its value:* largest for g (3.22), then q (3.02), then W (1.43), then Z (1.16), then t (1.05); it separates g jets from the rest best (AUC 0.76: large for g)
- **How the class scores use it:** Since gluons sit highest on it, the g score reads it as evidence for a gluon: it raises the g score, where it is the largest input (+32%). It does not enter the q, W, Z or t scores, even though quarks sit almost as high as gluons; quarks are pulled out of the g score by neuron 5 instead.
- *computed — used by:* raises the score of g (+32%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.32
if LHA > 0.112: z += -11.57 × (LHA − 0.112)
if pt_7 < 53.44: z += -0.063 × (53.44 − pt_7)
if mass < 36.23: z += -0.088 × (36.23 − mass)
if mass < 69.61: z += 0.023 × (69.61 − mass)
if sum_pt < 788: z += 0.0059 × (788 − sum_pt)
if pt_7 < 43.50: z += 0.063 × (43.50 − pt_7)
if mass < 36.23 and lam2 < 0.0011: z += 39.26 × (36.23 − mass) × (0.0011 − lam2)
if pt_7 > 30.48: z += 0.062 × (pt_7 − 30.48)
if log_sum_pt < 6.64: z += -2.63 × (6.64 − log_sum_pt)
if log_sum_pt < 6.57: z += 3.29 × (6.57 − log_sum_pt)
if lam1 < 0.006: z += 124 × (0.006 − lam1)
if lam1 < 0.0034: z += 268 × (0.0034 − lam1)
if pt_7 > 30.48 and C2 < 0.051: z += -1.04 × (pt_7 − 30.48) × (0.051 − C2)
if log_sum_pt < 6.46: z += 2.73 × (6.46 − log_sum_pt)
if width < 0.00017 and dr_7 < 0.223: z += -48855 × (0.00017 − width) × (0.223 − dr_7)
if width < 0.00017: z += 9165 × (0.00017 − width)
if pt_7 > 30.48 and max_dr > 0.093: z += -0.531 × (pt_7 − 30.48) × (max_dr − 0.093)
if lam1 < 0.006 and max_dr > 0.081: z += -2249 × (0.006 − lam1) × (max_dr − 0.081)
if log_sum_pt > 6.90: z += -8.56 × (log_sum_pt − 6.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (64 GeV), average width, pT spread over several particles** — 17.0% of jets, neuron 0.27. Mostly Z (38%) with W (33%) and t (20%); mass 63.6 GeV, width 0.0074, a hard leading particle (295.86 GeV) and half of the pT 0.05-0.1 from the axis. LHA > 0.112 (-2.132) and pt_7 < 53.4 (-1.513) always pass, and pt_7 < 43.5 (+0.905) returns only part; mass < 69.6 (72%) and sum_pt < 788 (55%) add little and mass < 36.2 almost never passes. The neuron stays low (0.272, on for 49%) and adds only +0.117 to the g score; the formula calls them W.
- **medium-mass (57 GeV), average width, pT spread over several particles** — 14.2% of jets, neuron 0.88. A W/Z mixture (W 37%, Z 36%) with some t and g; mass 56.7 GeV, width 0.0065, with pT shared more evenly than usual (4th particle 83.39 GeV against 71.7 GeV for all jets). LHA > 0.112 (-2.029) is partly offset by pt_7 > 30.5, which passes for all of them (+1.038), while pt_7 > 30.5 and C2 < 0.0512, pt_7 < 53.4 (82%) and pt_7 > 30.5 and max_dr > 0.0931 subtract smaller amounts; pt_7 < 43.5 passes for only 36%. The value 0.877 (on for 75%) raises the g score (+0.377); the formula calls them W.
- **light (42 GeV), average width, pT spread over several particles, low pT** — 10.7% of jets, neuron 2.19. An even mixture of Z (26%), W (26%) and t (24%) with g (16%); mass 41.7 GeV, width 0.0065, and a low total pT (560 GeV). sum_pt < 788 passes for all (+1.345), and log_sum_pt < 6.57 (+0.817) and mass < 69.6 (+0.652) add, against LHA > 0.112 (-2.022), pt_7 < 53.4 and log_sum_pt < 6.64; mass < 36.2 passes for 31%. The value 2.19 (on for 99.9%) raises the g score (+0.941); the formula calls them W.
- **very light (8 GeV), very narrow, pT spread over several particles** — 10.5% of jets, neuron 3.87. Mostly g (50%) with quarks (31%); mass 8.2 GeV, width 0.0004, 83% of the pT within 0.025, with the pT shared unusually evenly (4th particle 87.11 GeV). mass < 36.2 (-2.468) is outweighed by mass < 69.6 (+1.431), mass < 36.2 and lam2 < 0.00113 (+1.213), pt_7 > 30.5 (+1.119) and lam1 < 0.00338, and LHA > 0.112 passes for only 57% (-0.286). The value 3.867 (on for 99.9%) raises the g score (+1.662); the formula calls them g.
- **very light (11 GeV), very narrow, pT spread over several particles** — 10.3% of jets, neuron 3.11. Mostly q (41%) with gluons (26%) and some W and Z; mass 11.3 GeV, width 0.0006, 75% of the pT within 0.025 and a hard leading particle (313.32 GeV). mass < 36.2 (-2.188) and pt_7 < 53.4 (-1.620) are outweighed by mass < 69.6 (+1.357), mass < 36.2 and lam2 < 0.00113, pt_7 < 43.5 (+1.014) and lam1 < 0.00338, and LHA > 0.112 costs only -0.435. The value 3.11 (on for 99.9%) raises the g score (+1.336); the formula calls them q.
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 8.4% of jets, neuron 0.23. Mostly t (84%); mass 83.1 GeV, width 0.0231, with 38% of the pT 0.1-0.15 and 22% 0.15-0.2 from the axis. LHA > 0.112 subtracts -3.460, its largest amount in this neuron, and pt_7 < 53.4 and log_sum_pt < 6.64 more, against sum_pt < 788 (+1.256) and log_sum_pt < 6.57; mass < 69.6 passes for only 22% and mass < 36.2 never. The neuron stays low (0.234, on for 39%) and adds little to the g score (+0.101); the formula calls them t.
- **very light (6 GeV), very narrow, leading particle 42% of pT, high pT** — 8.2% of jets, neuron 3.31. Mostly q (71%) with gluons (15%); mass 6.0 GeV, width 0.0001, 99% of the pT within 0.025 and a 392.16 GeV leading particle. LHA > 0.112 almost never passes (1.3%), so the neuron keeps mass < 69.6 (+1.482), mass < 36.2 and lam2 < 0.00113 (+1.328), pt_7 < 43.5 (+1.326) and width < 0.000172 (+1.044) against mass < 36.2 (-2.659), pt_7 < 53.4 (-1.928) and width < 0.000172 and dr_7 < 0.223 (-1.145). The value 3.305 raises the g score (+1.42); the formula calls them q.
- **light (43 GeV), narrow, leading particle 46% of pT, high pT** — 7.9% of jets, neuron 1.12. Mostly W (36%) with Z (27%) and q (22%); mass 43.1 GeV, width 0.003, a very hard leading particle (397.82 GeV) and pT close to the axis. pt_7 < 53.4 (-1.953) is mostly returned by pt_7 < 43.5 (+1.351); LHA > 0.112 (96%, -1.003) and lam1 < 0.00595 and max_dr > 0.0805 (93%, -0.774) subtract, while mass < 69.6 and lam1 < 0.00595 add, and mass < 36.2 passes for 36%. The value 1.124 (on for 96%) raises the g score (+0.483); the formula calls them W.
- **very light (13 GeV), very narrow, pT spread over several particles, low pT** — 6.6% of jets, neuron 4.52. Mostly g (57%) with quarks (18%); mass 12.7 GeV, width 0.0015, and a low total pT (516 GeV, leader 128.12 GeV). The low-pT tests add: sum_pt < 788 (+1.606), log_sum_pt < 6.57 (+1.108) with mass < 69.6 (+1.326), against mass < 36.2 (-2.072), pt_7 < 53.4, log_sum_pt < 6.64 and LHA > 0.112 (-0.947). This gives the neuron's highest group value, 4.516 (on for all), raising the g score (+1.94); the formula calls them g.
- **medium-mass (51 GeV), very wide, pT spread over several particles, low pT** — 6.2% of jets, neuron 2.63. Mostly t (61%) with gluons (26%); mass 51.1 GeV, width 0.0187, and the lowest total pT here (414 GeV, leader 98.23 GeV), with pT spread 0.05-0.2 from the axis. LHA > 0.112 (-3.041) is more than made up by the low-pT tests sum_pt < 788 (+2.206), log_sum_pt < 6.57 (+1.827) and log_sum_pt < 6.46 (+1.217), with log_sum_pt < 6.64 and pt_7 < 53.4 subtracting. The value 2.627 (on for 99.9%) raises the g score (+1.129) even for these top-like jets; the formula calls them t.

### neuron 5: quark-likeness: pT in few particles (major)

- **What it measures:** Pushed up for light jets with little pT at wide angle (mass < 60.6 GeV with fewer than 2 particles at 0.2 ≤ ΔR < 0.4), small angularity (LHA < 0.216), small e2 (< 0.0356) and a soft 8th-hardest particle (z_7 < 0.0715); it rises with the summed pT of the 3 hardest particles and falls with z_7 (rank correlations 0.794 and -0.787). Quarks sit far highest (5.69); W (2.04) and Z (2.03) are in the middle, gluons (1.42) and tops (0.42) lowest.
- *computed — its value:* largest for q (5.69), then W (2.04), then Z (2.03), then g (1.42), then t (0.42); it separates q jets from the rest best (AUC 0.76: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops well below them, it raises the q score (+6%) and lowers the g score (-16%) and the t score (-16%). It does not enter the W or Z scores.
- *computed — used by:* raises the score of q (+6%); lowers the score of g (-16%), t (-16%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.095
if mass < 60.63 and n_dr_0p2_0p4 < 2.00: z += 0.021 × (60.63 − mass) × (2.00 − n_dr_0p2_0p4)
if LHA < 0.216: z += 28.78 × (0.216 − LHA)
if e2 < 0.036: z += 64.57 × (0.036 − e2)
if sum_pt_top2 < 548: z += -0.0041 × (548 − sum_pt_top2)
if z_7 < 0.071: z += 34.72 × (0.071 − z_7)
if mass_over_sum_pt < 0.108: z += -13.35 × (0.108 − mass_over_sum_pt)
if z_7 < 0.071 and lam2 < 0.0011: z += 31968 × (0.071 − z_7) × (0.0011 − lam2)
if pt_7 < 53.44: z += 0.034 × (53.44 − pt_7)
if log_sum_pt > 6.57 and lam1 < 0.012: z += -782 × (log_sum_pt − 6.57) × (0.012 − lam1)
if dr_0 < 0.022: z += -159 × (0.022 − dr_0)
if log_sum_pt > 6.57: z += 6.39 × (log_sum_pt − 6.57)
if girth2_top2 < 0.004: z += -246 × (0.004 − girth2_top2)
if log_sum_pt > 6.57 and dr_0 < 0.022: z += 582 × (log_sum_pt − 6.57) × (0.022 − dr_0)
if width < 0.0026: z += 473 × (0.0026 − width)
if LHA < 0.216 and log_sum_pt < 6.80: z += -86.02 × (0.216 − LHA) × (6.80 − log_sum_pt)
if LHA < 0.216 and centroid_offset > 0.0023: z += -2481 × (0.216 − LHA) × (centroid_offset − 0.0023)
if z_6 < 0.051: z += 60.17 × (0.051 − z_6)
if mass_over_sum_pt < 0.085: z += -9.93 × (0.085 − mass_over_sum_pt)
if sum_pt > 869: z += -0.017 × (sum_pt − 869)
if log_sum_pt > 6.84: z += -34.96 × (log_sum_pt − 6.84)
if e2_sq < 0.0053 and centroid_offset < 0.014: z += -20419 × (0.0053 − e2_sq) × (0.014 − centroid_offset)
if sum_pt > 869 and centroid_offset < 0.013: z += 1.90 × (sum_pt − 869) × (0.013 − centroid_offset)
if log_sum_pt > 6.57 and girth2_top2 < 0.0063: z += 595 × (log_sum_pt − 6.57) × (0.0063 − girth2_top2)
if mass_over_sum_pt < 0.085 and max_pair_mass < 18.10: z += -0.507 × (0.085 − mass_over_sum_pt) × (18.10 − max_pair_mass)
if LHA < 0.155: z += -19.48 × (0.155 − LHA)
if z_7 < 0.071 and sum_pt < 788: z += -0.263 × (0.071 − z_7) × (788 − sum_pt)
if log_sum_pt > 6.57 and n_dr_0p2_0p4 < 1.00: z += -3.00 × (log_sum_pt − 6.57) × (1.00 − n_dr_0p2_0p4)
if width < 0.0026 and centroid_offset < 0.024: z += 17761 × (0.0026 − width) × (0.024 − centroid_offset)
if sum_pt_top5 > 752: z += 0.010 × (sum_pt_top5 − 752)
if z_7 < 0.049 and centroid_offset < 0.011: z += -6466 × (0.049 − z_7) × (0.011 − centroid_offset)
if e2 < 0.036 and lam2 < 7.3e-05: z += 350012 × (0.036 − e2) × (7.3e-05 − lam2)
if z_7 < 0.049 and pt_5 < 73.75: z += -0.658 × (0.049 − z_7) × (73.75 − pt_5)
if z_7 < 0.023: z += 230 × (0.023 − z_7)
if z_7 < 0.049 and mass_top5 < 62.55: z += 0.534 × (0.049 − z_7) × (62.55 − mass_top5)
if log_sum_pt > 6.90 and centroid_offset < 0.018: z += -2715 × (log_sum_pt − 6.90) × (0.018 − centroid_offset)
if log_sum_pt > 6.57 and mean_phi2 < 0.00015: z += 32841 × (log_sum_pt − 6.57) × (0.00015 − mean_phi2)
if log_sum_pt > 6.90: z += 32.80 × (log_sum_pt − 6.90)
if z_6 < 0.029 and n_dr_0p2_0p4 < 2.00: z += 95.19 × (0.029 − z_6) × (2.00 − n_dr_0p2_0p4)
if sum_pt_top2 < 548 and dr_0 < 0.022: z += 0.310 × (548 − sum_pt_top2) × (0.022 − dr_0)
if z_7 < 0.032 and pt_5 < 29.88: z += -9.33 × (0.032 − z_7) × (29.88 − pt_5)
if log_sum_pt > 6.57 and mean_eta2 < 9e-05: z += 47180 × (log_sum_pt − 6.57) × (9e-05 − mean_eta2)
if mean_phi2 < 0.014 and pt_5 < 24.58: z += 23.24 × (0.014 − mean_phi2) × (24.58 − pt_5)
if z_6 < 0.029: z += -94.19 × (0.029 − z_6)
if log_sum_pt > 6.90 and dr_0 < 0.041: z += -573 × (log_sum_pt − 6.90) × (0.041 − dr_0)
if z_6 < 0.051 and e2_sq > 0.0039: z += -9242 × (0.051 − z_6) × (e2_sq − 0.0039)
if sum_pt_top5 > 752 and D2 < 1.68: z += -0.0076 × (sum_pt_top5 − 752) × (1.68 − D2)
if log_sum_pt > 6.90 and planar_flow < 0.045: z += -1570 × (log_sum_pt − 6.90) × (0.045 − planar_flow)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (56 GeV), wide, pT spread over several particles, low pT** — 44.4% of jets, neuron 0.20. The largest group (44% of jets) and a mixture: t 39%, Z 21%, W 20%, g 14%; mass 56.4 GeV, width 0.0116, below-average total pT (600 GeV) and pT spread 0.05-0.15 from the axis. The positive quark-like tests mostly fail here: LHA < 0.216 passes only 2.1% and e2 < 0.0356 33%, and mass < 60.6 and n_dr_0p2_0p4 < 2 passes 63% (+0.419); sum_pt_top2 < 548 always passes (-1.158) and mass_over_sum_pt < 0.108 takes a bit more. The neuron stays low (0.203, on for 28%) and barely moves any score; the formula calls them t.
- **medium-mass (59 GeV), average width, pT spread over several particles, high pT** — 15.6% of jets, neuron 2.89. A W/Z mixture (about 38% each) with some t; mass 59.1 GeV, narrow (width 0.0059), with a hard leading particle (324.6 GeV) and soft trailing particles. The soft-tail tests carry it: z_7 < 0.0715 (+1.245), z_7 < 0.0715 and lam2 < 0.00113 (+1.164), pt_7 < 53.4 and z_6 < 0.051, plus log_sum_pt > 6.57 (+0.902) partly returned by log_sum_pt > 6.57 and lam1 < 0.012 (-0.729); LHA < 0.216 passes only 15%. The value 2.89 (on for 97%) lowers the t (-0.723) and g (-0.542) scores and slightly raises q; the formula calls them W.
- **very light (13 GeV), very narrow, pT spread over several particles** — 12.8% of jets, neuron 1.65. Mostly g (38%) in a mixture with q (23%), W (17%) and Z (15%); mass 12.7 GeV, width 0.0012, with pT split between within 0.025 (47%) and 0.025-0.05 (45%). mass < 60.6 and n_dr_0p2_0p4 < 2 (+1.968), e2 < 0.0356 (+1.711) and LHA < 0.216 (87%) add, but mass_over_sum_pt < 0.108, LHA < 0.216 and centroid_offset > 0.00232 (-1.127), sum_pt_top2 < 548 and LHA < 0.216 and log_sum_pt < 6.8 take much of it back. The value 1.653 (on for 69%) lowers the t and g scores a little; the formula calls them g.
- **very light (8 GeV), very narrow, pT spread over several particles, high pT** — 8.2% of jets, neuron 6.28. Mostly q (57%) with gluons (26%); mass 8.2 GeV, width 0.0002, 97% of the pT within 0.025 and total pT 862 GeV. All the quark-like tests pass: LHA < 0.216 (+3.306), mass < 60.6 and n_dr_0p2_0p4 < 2, e2 < 0.0356 and log_sum_pt > 6.57 and dr_0 < 0.0216 (+1.507), against dr_0 < 0.0216 (-2.247) and log_sum_pt > 6.57 and lam1 < 0.012 (-1.713). The value 6.281 (on for 99.8%) lowers the t (-1.57) and g (-1.178) scores and raises q (+0.294); the formula calls them q.
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.6% of jets, neuron 0.58. Mostly g (57%) with quarks (32%); mass 7.9 GeV, width 0.0003, 93% of the pT within 0.025, but lower total pT (648 GeV) and a softer leader (173.33 GeV). LHA < 0.216 (+2.810) is cancelled by LHA < 0.216 and log_sum_pt < 6.8 (-2.722), which only bites for these lower-pT jets; mass < 60.6 and n_dr_0p2_0p4 < 2 and e2 < 0.0356 are balanced by dr_0 < 0.0216 (-1.889) and mass_over_sum_pt < 0.108. The neuron is small (0.577, on for 37%); the formula calls them g.
- **very light (8 GeV), very narrow, leading particle 43% of pT, high pT** — 4.3% of jets, neuron 11.20. Mostly q (72%); mass 7.9 GeV, width 0.0001, 98% of the pT within 0.025, total pT 966 GeV with a 417.11 GeV leading particle. LHA < 0.216 (+3.867), log_sum_pt > 6.57 and dr_0 < 0.0216 (+2.966) and sum_pt > 869 and centroid_offset < 0.0126 (+1.818) add to the usual light-jet terms, while log_sum_pt > 6.57 and lam1 < 0.012 (-2.783) and dr_0 < 0.0216 (-2.719) take back less. The value 11.199 (on for 99.7%) strongly lowers the t (-2.8) and g (-2.1) scores and raises q (+0.525); the formula calls them q.
- **medium-mass (55 GeV), narrow, leading particle 50% of pT, high pT** — 3.8% of jets, neuron 5.14. Mostly W (39%) with Z (33%) and some quarks; mass 55.0 GeV, width 0.0039, total pT 963 GeV with a 476.25 GeV leading particle and pT close to the axis. log_sum_pt > 6.57 (+1.889) and log_sum_pt > 6.57 and lam1 < 0.012 (-1.880) cancel; the soft-tail tests z_7 < 0.0715, z_7 < 0.0715 and lam2 < 0.00113 and z_6 < 0.051 add about 4.7, against sum_pt > 869 (-1.633) and log_sum_pt > 6.84 (69%). The value 5.139 (on for 85%) lowers the t and g scores and raises q; the formula calls them W.
- **very light (15 GeV), very narrow, leading particle 57% of pT, high pT** — 1.6% of jets, neuron 15.04. Mostly q (79%), a small group (1.6%); mass 14.6 GeV, width 0.0004, with the leading particle carrying 558.47 GeV of 978 GeV and the 8th only 8.21 GeV. Tests on the very soft tail pass only here: z_7 < 0.0232 (+3.398), mean_phi2 < 0.0143 and pt_5 < 24.6 (+3.069) and z_6 < 0.0289 and n_dr_0p2_0p4 < 2 (+2.985), with z_7 < 0.0324 and pt_5 < 29.9 (-3.313) the one strong brake, on top of LHA < 0.216 and log_sum_pt > 6.57 and dr_0 < 0.0216. The neuron's highest value, 15.042, strongly lowers the t (-3.761) and g (-2.82) scores and raises q (+0.705); the formula calls them q.
- **light (23 GeV), very narrow, leading particle 43% of pT, high pT** — 1.4% of jets, neuron 4.11. A gluon-quark mixture (g 43%, q 38%); mass 22.7 GeV, width 0.001, total pT 1136 GeV with a 491.12 GeV leader and 82% of the pT within 0.025. The very-high-pT tests take over: log_sum_pt > 6.84 (-6.670), log_sum_pt > 6.9 and centroid_offset < 0.0184 (-4.799), sum_pt > 869 and log_sum_pt > 6.57 and lam1 < 0.012 against log_sum_pt > 6.9 (+4.507) and sum_pt > 869 and centroid_offset < 0.0126 (+3.933). The value 4.114 (on for 64%) lowers the t and g scores and raises q; the formula splits them between q and g.
- **light (24 GeV), very narrow, leading particle 48% of pT, high pT** — 0.3% of jets, neuron 1.23. Mostly g (55%) with quarks (27%), a rare group (0.32%); mass 23.9 GeV, total pT 1386 GeV with a 663.06 GeV leading particle. The same high-pT tests grow larger: log_sum_pt > 6.84 (-13.492), log_sum_pt > 6.9 and centroid_offset < 0.0184 (-13.379), sum_pt > 869 and log_sum_pt > 6.9 and dr_0 < 0.0412 (-5.973) outweigh log_sum_pt > 6.9 (+10.907) and sum_pt > 869 and centroid_offset < 0.0126 (+8.930). The neuron ends at 1.231, on for only 31%; the formula calls them g.

### neuron 6: broad, off-centre spread (major)

- **What it measures:** Switched off for narrow jets (width < 0.0132 and girth2 < 0.00868 both push it down) and pushed up by small e2 (< 0.0503) and a large largest particle distance (max ΔR); it follows the pT share at 0.2 ≤ ΔR < 0.4, lam1 and the offset of the pT centroid from the axis (rank correlations 0.485, 0.461, 0.44). Tops sit far highest (3.81), then gluons (1.58) and quarks (0.72), with Z (0.40) and W (0.08) jets lowest.
- *computed — its value:* largest for t (3.81), then g (1.58), then q (0.72), then Z (0.40), then W (0.08); it separates t jets from the rest best (AUC 0.76: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so it lowers the W score (-12%) and the Z score (-20%). It raises the g score (+5%) and the q score (+10%), since among non-top jets a broad spread is QCD-like rather than boson-like; it does not enter the t score, which gets its width information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+5%), q (+10%); lowers the score of W (-12%), Z (-20%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = 6.27
if width < 0.013: z += -712 × (0.013 − width)
if girth2 < 0.0087: z += -1234 × (0.0087 − girth2)
if e2 < 0.050: z += 133 × (0.050 − e2)
if mass_over_sum_pt > 0.0084: z += -61.57 × (mass_over_sum_pt − 0.0084)
if girth2_top5 < 0.024: z += -90.86 × (0.024 − girth2_top5)
if mass_over_sum_pt < 0.154: z += 17.66 × (0.154 − mass_over_sum_pt)
z += 12.71 × max_dr
if centroid_offset > 0.0081 and lam2 < 0.0034: z += 48096 × (centroid_offset − 0.0081) × (0.0034 − lam2)
if centroid_offset > 0.018: z += -212 × (centroid_offset − 0.018)
if log_sum_pt < 6.70: z += -5.36 × (6.70 − log_sum_pt)
if lam1 < 0.0084: z += 225 × (0.0084 − lam1)
if girth2_top2 < 0.0095: z += 142 × (0.0095 − girth2_top2)
if log_sum_pt < 6.57: z += 6.33 × (6.57 − log_sum_pt)
if mass < 49.67 and z_dr_0p05_0p1 < 0.751: z += 0.060 × (49.67 − mass) × (0.751 − z_dr_0p05_0p1)
if D2 < 1.68: z += 1.09 × (1.68 − D2)
if lam2 < 0.00054 and z_dr_0p2_0p4 < 0.206: z += -7379 × (0.00054 − lam2) × (0.206 − z_dr_0p2_0p4)
if log_sum_pt < 6.70 and pt_7 < 45.75: z += 0.261 × (6.70 − log_sum_pt) × (45.75 − pt_7)
if girth > 0.087: z += 63.84 × (girth − 0.087)
if girth2_top5 > 0.0023: z += -105 × (girth2_top5 − 0.0023)
if lam2 > 0.0034: z += -2819 × (lam2 − 0.0034)
if LHA > 0.313 and eccentricity > 0.873: z += 425 × (LHA − 0.313) × (eccentricity − 0.873)
if pt1_dr01 < 5.35: z += -0.114 × (5.35 − pt1_dr01)
if girth2_top5 > 0.011: z += 191 × (girth2_top5 − 0.011)
if z_dr_0_0p05 > 0.768: z += -2.95 × (z_dr_0_0p05 − 0.768)
if centroid_offset > 0.0081 and planar_flow > 0.008: z += 88.98 × (centroid_offset − 0.0081) × (planar_flow − 0.008)
if sum_pt_top5 > 840: z += -0.028 × (sum_pt_top5 − 840)
if D2 < 1.68 and pt_4 < 90.62: z += -0.013 × (1.68 − D2) × (90.62 − pt_4)
if sum_pt > 988: z += 0.048 × (sum_pt − 988)
if log_sum_pt < 6.70 and mean_eta2 > 0.0042: z += 386 × (6.70 − log_sum_pt) × (mean_eta2 − 0.0042)
if centroid_offset > 0.018 and mean_phi2 < 0.0089: z += 9298 × (centroid_offset − 0.018) × (0.0089 − mean_phi2)
if mass_over_sum_pt > 0.0084 and pt_7 < 41.66: z += -0.409 × (mass_over_sum_pt − 0.0084) × (41.66 − pt_7)
if mass_over_sum_pt > 0.0084 and tau32 < 0.519: z += -27.44 × (mass_over_sum_pt − 0.0084) × (0.519 − tau32)
if pt_6 < 24.42: z += 0.230 × (24.42 − pt_6)
if girth2_top5 > 0.0023 and z_dr_0p05_0p1 > 0.292: z += 184 × (girth2_top5 − 0.0023) × (z_dr_0p05_0p1 − 0.292)
if girth2_top5 > 0.011 and pt_7 > 15.55: z += -4.02 × (girth2_top5 − 0.011) × (pt_7 − 15.55)
if lam2 > 0.00054: z += 305 × (lam2 − 0.00054)
if log_sum_pt < 6.70 and pt_6 < 36.81: z += 0.258 × (6.70 − log_sum_pt) × (36.81 − pt_6)
if girth2_top5 > 0.011 and mean_eta > 0.013: z += -9543 × (girth2_top5 − 0.011) × (mean_eta − 0.013)
if centroid_offset > 0.050: z += 114 × (centroid_offset − 0.050)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.159: z += 420 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.159)
if C2 > 0.011 and pt_7 > 31.86: z += 1.09 × (C2 − 0.011) × (pt_7 − 31.86)
if z_4 < 0.037: z += -181 × (0.037 − z_4)
if pt_4 < 31.12: z += 0.253 × (31.12 − pt_4)
if LHA > 0.313: z += -5.16 × (LHA − 0.313)
if log_sum_pt < 6.70 and z_7 < 0.049: z += 307 × (6.70 − log_sum_pt) × (0.049 − z_7)
if sum_pt > 988 and pt_6 > 29.91: z += -0.00033 × (sum_pt − 988) × (pt_6 − 29.91)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 32.0% of jets, neuron 0.08. Mostly q (45%) with many gluons (35%), about a third of all jets; mass 9.8 GeV, width 0.0004, 84% of the pT within 0.025. The narrow-jet switches girth2 < 0.00868 (-10.248) and width < 0.0132 (-9.159) always pass and outweigh e2 < 0.0503 (+5.868), mass_over_sum_pt < 0.154, lam1 < 0.00838 and mass < 49.7 and z_dr_0p05_0p1 < 0.751 (+1.783, which passes for all of them). The neuron is essentially off (0.079, on for 5.6%) and hardly moves any score; the formula calls them q.
- **medium-mass (55 GeV), average width, pT spread over several particles** — 20.7% of jets, neuron 0.24. Mostly W (49%) with Z (34%); mass 55.2 GeV, width 0.0058, 55% of the pT 0.05-0.1 from the axis. width < 0.0132 (-5.271), mass_over_sum_pt > 0.00837 (-4.096), girth2 < 0.00868 (-3.510) and girth2_top5 < 0.0244 all pass and outweigh e2 < 0.0503 (+2.093) and the max_dr term (+1.729); centroid_offset > 0.0184 passes for only 19%. The neuron stays low (0.239, on for 18.6%); the formula calls them W.
- **light (40 GeV), narrow, pT spread over several particles** — 13.8% of jets, neuron 0.20. Mostly W (40%) mixed with Z (23%), g (17%) and q (13%); mass 39.8 GeV, width 0.0032, 59% of the pT 0.025-0.05 from the axis. width < 0.0132 (-7.120) and girth2 < 0.00868 (-6.715), with mass_over_sum_pt > 0.00837 and girth2_top5 < 0.0244, outweigh e2 < 0.0503 (+3.989) and the max_dr term (+2.017); girth > 0.0872 never passes. The neuron stays low (0.204, on for 14.6%); the formula calls them W.
- **medium-mass (61 GeV), wide, pT spread over several particles** — 11.5% of jets, neuron 2.22. Mostly Z (49%) with t (31%); mass 61.2 GeV, width 0.0091, 60% of the pT 0.05-0.1 and 19% 0.1-0.15 from the axis. girth2 < 0.00868 passes only 62% (-0.680 here against about -6 elsewhere) and width < 0.0132 adds -3.054, so the narrow-jet switch is weaker; mass_over_sum_pt > 0.00837 (-5.185) and log_sum_pt < 6.7 subtract, while the max_dr term (+2.048), centroid_offset > 0.00809 and lam2 < 0.00341 (67%) and e2 < 0.0503 (86%) add. The value 2.218 (on for 72%) lowers the Z (-0.832) and W (-0.693) scores and raises q and g; the formula calls them Z.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 8.0% of jets, neuron 6.79. Mostly t (77%) with gluons (16%); mass 81.4 GeV, width 0.0219, with 33% of the pT 0.1-0.15 and 19% 0.15-0.2 from the axis. width < 0.0132, girth2 < 0.00868 and e2 < 0.0503 almost never pass, so the narrow-jet switches are gone; girth > 0.0872 (+2.970), the max_dr term (+2.942) and LHA > 0.313 and eccentricity > 0.873 (80%, +2.806) add, while mass_over_sum_pt > 0.00837 (-8.332), log_sum_pt < 6.7 and centroid_offset > 0.0184 subtract. The value 6.789 (on for 97%) lowers the Z (-2.546) and W (-2.121) scores and raises q (+0.849) and g (+0.743); the formula calls them t.
- **very light (19 GeV), narrow, pT spread over several particles, low pT** — 5.2% of jets, neuron 2.14. A mixture: g 32%, Z 26%, W 15%, t 15%, q 12%; mass 19.1 GeV, width 0.0031, with only 5.1% of the pT within 0.025 (against 32% for all jets) and the rest 0.025-0.1 from the axis. The off-centre tests pass for all of them: centroid_offset > 0.00809 and lam2 < 0.00341 (+4.790) and centroid_offset > 0.0184 (-4.386) nearly cancel, and e2 < 0.0503 (+4.855) and mass_over_sum_pt < 0.154 fight width < 0.0132 (-7.247) and girth2 < 0.00868 (-6.934). The value 2.137 (on for 83%) lowers the W and Z scores and raises q and g; the formula calls them g.
- **medium-mass (59 GeV), very wide, pT spread over several particles, low pT** — 4.1% of jets, neuron 7.16. Mostly t (67%) with gluons (23%); mass 59.3 GeV, width 0.0182, softer total pT (533 GeV), with 36% of the pT 0.1-0.15 from the axis. centroid_offset > 0.0184 (-9.871) is largely returned by centroid_offset > 0.00809 and lam2 < 0.00341 (92%, +6.235); mass_over_sum_pt > 0.00837 and log_sum_pt < 6.7 subtract while the max_dr term and girth > 0.0872 add, and the narrow-jet switches rarely pass (width < 0.0132 32%, girth2 < 0.00868 6.3%). The neuron's highest value, 7.16 (on for 93.5%), lowers the Z (-2.685) and W (-2.237) scores and raises q and g; the formula calls them t.
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 2.5% of jets, neuron 0.36. Mostly t (94%); mass 81.9 GeV, width 0.0252, with nearly no pT within 0.025 and most of it 0.05-0.2 from the axis. lam2 > 0.00341 passes for all of them (-8.051) and, with mass_over_sum_pt > 0.00837 (-8.847) and centroid_offset > 0.0184 (82%), outweighs girth > 0.0872 (+3.760) and the max_dr term (+3.304). The neuron stays low (0.364, on for 14%); the formula calls them t.
- **light (22 GeV), very narrow, leading particle 46% of pT, high pT** — 1.2% of jets, neuron 0.79. Mostly g (43%) with quarks (36%); mass 22.4 GeV, width 0.0008, total pT 1227 GeV with a 568.81 GeV leading particle and 83% of the pT within 0.025. sum_pt > 988 (+11.446) passes only here, partly cancelled by sum_pt_top5 > 840 (-7.211), while girth2 < 0.00868 (-9.800) and width < 0.0132 (-8.890) against e2 < 0.0503 (+5.755) behave as for other narrow jets. The value 0.794 is on for 37%; the formula calls them g.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.02. Mostly t (95%); mass 86.2 GeV, width 0.0305, with essentially all pT beyond 0.05 and most of it 0.1-0.2 from the axis. lam2 > 0.00341 subtracts -20.471 and mass_over_sum_pt > 0.00837 and centroid_offset > 0.0184 subtract more, far beyond what girth > 0.0872 (+4.990), girth2_top5 > 0.0115 and the max_dr term add. The neuron is off (on for 0.8%) and does nothing to the scores; the formula calls them t.

### neuron 7: two-prong shape of W/Z size (major)

- **What it measures:** Built from many small size steps: pushed down for very narrow jets (lam1 < 0.00838, girth < 0.0872, width < 0.00559) and for wider ones (girth2 > 0.00752), and pushed up for mass/pT above 0.0727 and 0.0848 but down again above 0.108, so it peaks in a W/Z-like mass/pT window; it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.537, -0.537, -0.424). Z jets sit highest (3.17), then W (1.63), with tops (0.86), gluons (0.76) and quarks (0.39) low.
- *computed — its value:* largest for Z (3.17), then W (1.63), then t (0.86), then g (0.76), then q (0.39); it separates Z jets from the rest best (AUC 0.83: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the W score (+9%) and the Z score (+26%), where it is the largest input. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+9%), Z (+26%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 10.09
if lam1 < 0.0084: z += -727 × (0.0084 − lam1)
if girth2 > 0.0075: z += -1134 × (girth2 − 0.0075)
if girth < 0.087: z += -75.96 × (0.087 − girth)
if girth2 > 0.0017: z += -528 × (girth2 − 0.0017)
if mass_over_sum_pt > 0.108: z += -504 × (mass_over_sum_pt − 0.108)
if width < 0.0056: z += -1027 × (0.0056 − width)
if mass_over_sum_pt > 0.085: z += 177 × (mass_over_sum_pt − 0.085)
if max_dr < 0.160: z += -29.20 × (0.160 − max_dr)
if mass_over_sum_pt > 0.073: z += 106 × (mass_over_sum_pt − 0.073)
if C2 < 0.067: z += -32.78 × (0.067 − C2)
if max_dr < 0.160 and z_dr_0p05_0p1 < 0.675: z += 43.40 × (0.160 − max_dr) × (0.675 − z_dr_0p05_0p1)
if mass_over_sum_pt > 0.090: z += -152 × (mass_over_sum_pt − 0.090)
if girth2 > 0.0044: z += -334 × (girth2 − 0.0044)
if centroid_offset < 0.038 and sum_pt > 560: z += 0.236 × (0.038 − centroid_offset) × (sum_pt − 560)
if girth2 > 0.0087: z += 423 × (girth2 − 0.0087)
if mass < 29.64: z += 0.125 × (29.64 − mass)
if e2 < 0.038: z += 56.18 × (0.038 − e2)
if e2 > 0.017: z += -53.11 × (e2 − 0.017)
if e2 < 0.025: z += 103 × (0.025 − e2)
if max_dr < 0.198 and z_dr_0p05_0p1 > 0.049: z += 37.85 × (0.198 − max_dr) × (z_dr_0p05_0p1 − 0.049)
if girth2 > 0.013: z += -447 × (girth2 − 0.013)
if lam1 < 0.0084 and D2 < 1.12: z += -1357 × (0.0084 − lam1) × (1.12 − D2)
if pt_7 < 48.72 and planar_flow < 0.695: z += -0.092 × (48.72 − pt_7) × (0.695 − planar_flow)
if LHA < 0.293: z += 7.53 × (0.293 − LHA)
if width < 0.0056 and n_dr_0p1_0p2 < 3.00: z += 75.88 × (0.0056 − width) × (3.00 − n_dr_0p1_0p2)
if mass_over_sum_pt < 0.131: z += 6.58 × (0.131 − mass_over_sum_pt)
if centroid_offset < 0.021: z += -56.28 × (0.021 − centroid_offset)
if girth2 > 0.0075 and log_sum_pt > 6.19: z += -1158 × (girth2 − 0.0075) × (log_sum_pt − 6.19)
if width < 0.00056: z += -4189 × (0.00056 − width)
if girth2 > 0.0044 and eccentricity > 0.946: z += 5105 × (girth2 − 0.0044) × (eccentricity − 0.946)
if girth2_top2 < 0.0011 and log_sum_pt > 6.33: z += -3387 × (0.0011 − girth2_top2) × (log_sum_pt − 6.33)
if girth < 0.041: z += -39.83 × (0.041 − girth)
if width < 0.0056 and n_dr_0p2_0p4 < 1.00: z += -159 × (0.0056 − width) × (1.00 − n_dr_0p2_0p4)
if lam1 < 0.0025: z += 448 × (0.0025 − lam1)
if mass > 80.40: z += -0.245 × (mass − 80.40)
if e2 < 0.050 and D2 < 1.12: z += 111 × (0.050 − e2) × (1.12 − D2)
if mass_over_sum_pt < 0.131 and z_dr_0p05_0p1 > 0.292: z += -34.79 × (0.131 − mass_over_sum_pt) × (z_dr_0p05_0p1 − 0.292)
if LHA < 0.293 and m012 < 16.90: z += -0.257 × (0.293 − LHA) × (16.90 − m012)
if girth2_top2 < 0.0011 and n_dr_0p2_0p4 < 1.00: z += 792 × (0.0011 − girth2_top2) × (1.00 − n_dr_0p2_0p4)
if centroid_offset > 0.031: z += 69.29 × (centroid_offset − 0.031)
if planar_flow < 0.195 and sum_pt > 616: z += 0.014 × (0.195 − planar_flow) × (sum_pt − 616)
if e2 < 0.038 and D2 < 1.12: z += 176 × (0.038 − e2) × (1.12 − D2)
if planar_flow < 0.195 and width > 0.0075: z += 832 × (0.195 − planar_flow) × (width − 0.0075)
if centroid_offset > 0.031 and pt_2 > 56.50: z += -1.53 × (centroid_offset − 0.031) × (pt_2 − 56.50)
if centroid_offset < 0.021 and C2 > 0.024: z += 1460 × (0.021 − centroid_offset) × (C2 − 0.024)
if mass_top5 > 53.61: z += 0.031 × (mass_top5 − 53.61)
if mass > 80.40 and eccentricity > 0.927: z += -1.14 × (mass − 80.40) × (eccentricity − 0.927)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (55 GeV), average width, pT spread over several particles** — 28.0% of jets, neuron 3.15. Mostly Z (42%) with W (34%) and some t, the largest group (28% of jets); mass 54.9 GeV, width 0.0065, 65% of the pT 0.05-0.1 from the axis. max_dr < 0.198 and z_dr_0p05_0p1 > 0.0488 passes for 87% (+1.881, much more than elsewhere), and the size penalties are mild: width < 0.00559 and girth2 > 0.00752 pass for only 24% and mass_over_sum_pt > 0.108 never, leaving girth2 > 0.00165 (-2.581), lam1 < 0.00838, C2 < 0.0673 and lam1 < 0.00838 and D2 < 1.12. The neuron's highest group value, 3.147 (on for 94.6%), raises the Z (+1.475) and W (+0.688) scores; the formula splits them between Z and W.
- **very light (7 GeV), very narrow, pT spread over several particles, high pT** — 20.6% of jets, neuron 0.02. Mostly q (54%) with gluons (32%); mass 7.1 GeV, width 0.0001, 98% of the pT within 0.025. All the too-narrow penalties pass: lam1 < 0.00838 (-6.009), girth < 0.0872 (-5.959), width < 0.00559 (-5.608) and max_dr < 0.16 (-3.838), which max_dr < 0.16 and z_dr_0p05_0p1 < 0.675 (+3.840) and mass < 29.6 (+2.815) cannot make up. The neuron is off (on for 2.3%) and adds nothing; the formula calls them q.
- **light (42 GeV), narrow, pT spread over several particles** — 17.7% of jets, neuron 1.76. Mostly W (39%) with Z (27%) and some g and q; mass 41.8 GeV, width 0.0036, 59% of the pT 0.025-0.05 from the axis. The narrow penalties lam1 < 0.00838 (-3.571), girth < 0.0872 (-3.091) and width < 0.00559 (97%) pass, and the mass/pT window bonuses do not (mass_over_sum_pt > 0.0727 3.8%, mass_over_sum_pt > 0.0848 never); centroid_offset < 0.0378 and sum_pt > 560 adds +1.105. The value 1.764 (on for 81%) raises the Z (+0.827) and W (+0.386) scores; the formula calls them W.
- **very light (14 GeV), very narrow, pT spread over several particles** — 15.4% of jets, neuron 0.76. Mostly g (41%) mixed with q (26%), W and Z (14% each); mass 14.0 GeV, width 0.001, with pT about evenly within 0.025 (51%) and 0.025-0.05 (43%). lam1 < 0.00838 (-5.450), width < 0.00559 (-4.749), girth < 0.0872 (-4.570) and max_dr < 0.16 pass for all of them, against max_dr < 0.16 and z_dr_0p05_0p1 < 0.675 (+2.750) and mass < 29.6 (+1.962); girth2 > 0.00752 never passes. The value 0.761 is on for 44% and mildly raises the Z and W scores; the formula calls them g.
- **medium-mass (60 GeV), wide, pT spread over several particles, low pT** — 5.1% of jets, neuron 1.13. Mostly t (65%) with gluons (21%); mass 59.7 GeV, width 0.0122, 47% of the pT 0.05-0.1 and 29% 0.1-0.15 from the axis. The narrow penalties are absent (lam1 < 0.00838 4.1%, width < 0.00559 never), but the wide ones pass: girth2 > 0.00165 (-5.595), girth2 > 0.00752 (-5.363), girth2 > 0.00437 and mass_over_sum_pt > 0.0904 (90%); the mass/pT window steps mass_over_sum_pt > 0.0727 and mass_over_sum_pt > 0.0848 add about +3.1 each. The balance is 1.132, on for 52%, raising Z and W a little; the formula calls them t.
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.8% of jets, neuron 0.00. Mostly t (78%); mass 72.7 GeV, width 0.0169, with pT spread 0.05-0.15 from the axis. girth2 > 0.00752 (-10.673), girth2 > 0.00165 (-8.068), mass_over_sum_pt > 0.108 (99%, -7.781) and mass_over_sum_pt > 0.0904 outweigh the window steps mass_over_sum_pt > 0.0848 (+6.853) and mass_over_sum_pt > 0.0727 (+5.389). The neuron is off for all of them and adds nothing; the formula calls them t.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 3.4% of jets, neuron 0.00. Mostly t (83%); mass 79.7 GeV, width 0.0212, with 37% of the pT 0.1-0.15 from the axis. The same ladder with larger steps: mass_over_sum_pt > 0.108 (-16.091) and girth2 > 0.00752 (-15.510) plus girth2 > 0.00165 and mass_over_sum_pt > 0.0904 beat mass_over_sum_pt > 0.0848 (+9.793) and mass_over_sum_pt > 0.0727. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 3.1% of jets, neuron 0.00. Mostly t (88%); mass 85.3 GeV, width 0.0259, with much of the pT 0.1-0.2 from the axis. mass_over_sum_pt > 0.108 (-24.344) and girth2 > 0.00752 (-20.893) dominate the positive mass/pT steps (mass_over_sum_pt > 0.0848 +12.701). The neuron is off and adds nothing; the formula calls them t.
- **heavy (92 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.00. Mostly t (83%); mass 91.7 GeV, width 0.0315, with 31% of the pT 0.15-0.2 from the axis. mass_over_sum_pt > 0.108 (-32.825), girth2 > 0.00752 (-27.205) and girth2 > 0.00165 outweigh mass_over_sum_pt > 0.0848 (+15.690) and mass_over_sum_pt > 0.0727. The neuron is off and adds nothing; the formula calls them t.
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.00. Mostly t (61%) with gluons (30%), a small group (0.61%); mass 94.6 GeV, width 0.0415, the lowest total pT (481 GeV), with most pT beyond 0.1 from the axis. The largest penalties of this neuron, mass_over_sum_pt > 0.108 (-44.775) and girth2 > 0.00752 (-38.536), exceed mass_over_sum_pt > 0.0848 (+19.901) and girth2 > 0.00868 (+13.885). The neuron is off and adds nothing; the formula calls them t.

### neuron 9: narrow, light single-core jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.0061, its strongest term, girth2 < 0.00752 and max ΔR < 0.222), and pushed down for mass < 53.3 GeV and for small mass/pT (< 0.0764); it falls with lam1, width and girth (rank correlations -0.668, -0.658, -0.635). Quarks sit highest (6.59), then gluons (5.24), far above W (1.99), top (1.88) and Z (1.08) jets.
- *computed — its value:* largest for q (6.59), then g (5.24), then W (1.99), then t (1.88), then Z (1.08); it separates q jets from the rest best (AUC 0.78: large for q)
- **How the class scores use it:** High values mean a light-parton jet, so it raises the q score (its largest input, +49%) and the g score (+22%), and lowers the W score (-3%) and the Z score (-4%) slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+22%), q (+49%); lowers the score of W (-3%), Z (-4%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.89
if width < 0.0061: z += 1797 × (0.0061 − width)
if mass < 53.33: z += -0.099 × (53.33 − mass)
if mass < 41.38 and centroid_offset < 0.027: z += -9.10 × (41.38 − mass) × (0.027 − centroid_offset)
if girth2 < 0.0075: z += 514 × (0.0075 − girth2)
if mass_over_sum_pt < 0.076: z += -59.66 × (0.076 − mass_over_sum_pt)
if mass < 41.38: z += 0.127 × (41.38 − mass)
if girth2 < 0.0044: z += 1012 × (0.0044 − girth2)
if max_dr < 0.222: z += 13.08 × (0.222 − max_dr)
if mass < 53.33 and centroid_offset < 0.027: z += 4.47 × (53.33 − mass) × (0.027 − centroid_offset)
if girth < 0.055: z += -84.92 × (0.055 − girth)
if e2 < 0.020: z += 204 × (0.020 − e2)
if mass < 53.33 and log_sum_pt < 6.84: z += 0.216 × (53.33 − mass) × (6.84 − log_sum_pt)
if width < 0.0061 and centroid_offset > 0.0033: z += -47078 × (0.0061 − width) × (centroid_offset − 0.0033)
if centroid_offset < 0.018: z += 137 × (0.018 − centroid_offset)
if mass < 29.64: z += -0.112 × (29.64 − mass)
if mass_over_sum_pt < 0.076 and girth2_top2 < 0.00076: z += -64618 × (0.076 − mass_over_sum_pt) × (0.00076 − girth2_top2)
if e2 < 0.017 and centroid_offset < 0.024: z += 11473 × (0.017 − e2) × (0.024 − centroid_offset)
if mass_over_sum_pt_sq < 0.003: z += 515 × (0.003 − mass_over_sum_pt_sq)
if e2 < 0.020 and pt_7 < 53.44: z += -4.63 × (0.020 − e2) × (53.44 − pt_7)
if lam1 < 0.006: z += -218 × (0.006 − lam1)
if girth2 < 0.00096: z += 2512 × (0.00096 − girth2)
if log_sum_pt > 6.38: z += -2.29 × (log_sum_pt − 6.38)
if e2 < 0.032 and dr01 < 0.056: z += 870 × (0.032 − e2) × (0.056 − dr01)
if mass < 53.33 and planar_flow < 0.322: z += 0.311 × (53.33 − mass) × (0.322 − planar_flow)
if lam1 < 0.006 and mean_phi2 < 0.0021: z += -95751 × (0.006 − lam1) × (0.0021 − mean_phi2)
if girth2_top2 < 0.00076: z += 1996 × (0.00076 − girth2_top2)
if centroid_offset < 0.018 and z_4 > 0.047: z += -1579 × (0.018 − centroid_offset) × (z_4 − 0.047)
if e2 < 0.032: z += -30.41 × (0.032 − e2)
if max_dr < 0.222 and z_top5 < 0.909: z += -36.52 × (0.222 − max_dr) × (0.909 − z_top5)
if girth2 < 0.0044 and centroid_offset > 0.011: z += -59443 × (0.0044 − girth2) × (centroid_offset − 0.011)
if lam1 < 0.0015: z += -798 × (0.0015 − lam1)
if lam2 > 0.0011: z += 997 × (lam2 − 0.0011)
if girth < 0.034: z += -43.93 × (0.034 − girth)
if mass < 53.33 and lam1 < 0.00087: z += -27.17 × (53.33 − mass) × (0.00087 − lam1)
if girth2 < 0.0075 and pt_7 > 20.12: z += -4.50 × (0.0075 − girth2) × (pt_7 − 20.12)
if centroid_offset < 0.018 and pt_4 > 47.34: z += 2.03 × (0.018 − centroid_offset) × (pt_4 − 47.34)
if girth2_top2 < 0.00076 and z_7 > 0.023: z += -53739 × (0.00076 − girth2_top2) × (z_7 − 0.023)
if mass < 41.38 and planar_flow < 0.322: z += -0.243 × (41.38 − mass) × (0.322 − planar_flow)
if lam1 < 0.006 and dr_2 < 0.028: z += -6549 × (0.006 − lam1) × (0.028 − dr_2)
if e2 < 0.032 and tau21 < 0.447: z += -222 × (0.032 − e2) × (0.447 − tau21)
if width < 0.00017: z += 11109 × (0.00017 − width)
if log_sum_pt > 6.38 and dr_2 < 0.028: z += 82.93 × (log_sum_pt − 6.38) × (0.028 − dr_2)
if log_sum_pt > 6.38 and dr_5 < 0.022: z += 164 × (log_sum_pt − 6.38) × (0.022 − dr_5)
if sum_pt > 840: z += -0.005 × (sum_pt − 840)
if e2 < 0.020 and eccentricity > 0.903: z += -952 × (0.020 − e2) × (eccentricity − 0.903)
if centroid_offset < 0.0095 and mean_phi2 < 0.0021: z += -39520 × (0.0095 − centroid_offset) × (0.0021 − mean_phi2)
if pt_5 < 35.50: z += 0.074 × (35.50 − pt_5)
if girth < 0.055 and dr_5 < 0.022: z += -961 × (0.055 − girth) × (0.022 − dr_5)
if girth2 > 0.019: z += 139 × (girth2 − 0.019)
if girth2_top2 < 0.00076 and pt_7 > 33.22: z += 109 × (0.00076 − girth2_top2) × (pt_7 − 33.22)
if sum_pt > 988 and dr_3 < 0.060: z += 0.436 × (sum_pt − 988) × (0.060 − dr_3)
if width < 0.0061 and C2 > 0.031: z += -11169 × (0.0061 − width) × (C2 − 0.031)
if C2 > 0.051: z += 8.53 × (C2 − 0.051)
if girth2 > 0.019 and pt_7 < 29.04: z += 54.92 × (girth2 − 0.019) × (29.04 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (64 GeV), wide, pT spread over several particles** — 37.6% of jets, neuron 0.42. The largest group (38% of jets), a mixture of t (34%), Z (31%) and W (20%); mass 64.4 GeV, width 0.0113, with half of the pT 0.05-0.1 and 20% 0.1-0.15 from the axis. Both the narrow-jet bonuses and the light-jet penalties are mostly missing: width < 0.0061 passes for only 21% and girth2 < 0.00752 for 47%, while mass < 53.3 passes for 27% and mass_over_sum_pt < 0.0764 for 20%. What is left, max_dr < 0.222 (77%, +0.829) and centroid_offset < 0.0184 (61%, +0.789) against small brakes, gives 0.424 (on for 37.5%) and barely moves the scores; the formula calls them t.
- **very light (7 GeV), very narrow, leading particle 41% of pT, high pT** — 12.2% of jets, neuron 8.59. Mostly q (67%) with gluons (20%); mass 7.1 GeV, width 0.0001, 98.5% of the pT within 0.025, total pT 934 GeV with a 385.12 GeV leading particle. width < 0.0061 (+10.806), mass < 53.3 and centroid_offset < 0.0269 (+4.957), girth2 < 0.00437 (+4.341) and mass < 41.4 (+4.339) outweigh mass < 41.4 and centroid_offset < 0.0269 (-7.488), mass < 53.3 (-4.585) and girth < 0.0546 (-4.104). The neuron's highest value, 8.59 (on for all), raises the q (+2.181) and g (+1.476) scores and lowers W and Z (-0.268 each); the formula calls them q.
- **light (49 GeV), narrow, pT spread over several particles** — 10.8% of jets, neuron 1.39. Mostly W (53%) with Z (28%); mass 49.1 GeV, width 0.0043, 53% of the pT 0.025-0.05 from the axis. width < 0.0061 (+3.184) and girth2 < 0.00752 (+1.642) always pass, helped by max_dr < 0.222 (85%) and centroid_offset < 0.0184 (68%), against width < 0.0061 and centroid_offset > 0.00334 (-0.973), mass_over_sum_pt < 0.0764 and mass < 53.3 (70%); mass < 41.4 passes for only 20.5%. The value 1.386 (on for 66%) mildly raises q and g; the formula calls them W.
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.8% of jets, neuron 8.52. Mostly g (53%) with quarks (37%); mass 8.2 GeV, width 0.0002, 94% of the pT within 0.025, with lower total pT (696 GeV) and a 197.35 GeV leader. The same balance as the other pencil-thin jets: width < 0.0061 (+10.568), girth2 < 0.00437, mass < 41.4 and mass < 53.3 and centroid_offset < 0.0269 (about +4.2 each) against mass < 41.4 and centroid_offset < 0.0269 (-6.249) and mass < 53.3 (-4.473). The value 8.523 (on for all) raises the q (+2.164) and g (+1.465) scores; the formula calls them g.
- **light (34 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 3.00. A mixture: W 34%, Z 21%, g 20%, q 18%; mass 34.3 GeV, width 0.0025, 57% of the pT 0.025-0.05 from the axis. width < 0.0061 (+6.406), girth2 < 0.00752 and girth2 < 0.00437 add about 10.8 in all, against width < 0.0061 and centroid_offset > 0.00334 (97%, -2.461), mass < 53.3 and mass_over_sum_pt < 0.0764; mass < 41.4 and centroid_offset < 0.0269 passes for 66%. The value 2.999 (on for 81%) raises q (+0.761) and g (+0.515); the formula calls them W.
- **very light (9 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 1.47. A mixture: g 34%, Z 24%, W 20%, q 13%; mass 9.1 GeV, width 0.0014, with 68% of the pT 0.025-0.05 from the axis and only 20% within 0.025. width < 0.0061 (+8.422), mass < 41.4 and mass < 53.3 and log_sum_pt < 6.84 (+3.664) add, but the off-centre tests width < 0.0061 and centroid_offset > 0.00334 (-6.045) and girth2 < 0.00437 and centroid_offset > 0.011 (-3.360) pass for all of them, along with mass < 53.3 and mass_over_sum_pt < 0.0764. The value 1.467 is on for 60%; the formula calls them g.
- **light (21 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 7.90. Mostly g (46%) with quarks (37%); mass 21.0 GeV, width 0.0011, 60% of the pT within 0.025 and 31% 0.025-0.05. width < 0.0061 (+9.049), girth2 < 0.00437, girth2 < 0.00752 and mass < 53.3 and centroid_offset < 0.0269 all pass, against mass < 41.4 and centroid_offset < 0.0269 (-3.401), mass < 53.3 (-3.206) and mass_over_sum_pt < 0.0764. The value 7.896 (on for all) raises the q (+2.005) and g (+1.357) scores; the formula calls them g.
- **very light (8 GeV), very narrow, pT spread over several particles** — 4.9% of jets, neuron 4.12. Mostly g (38%) in a mixture with q (25%), W (20%) and Z (14%); mass 8.0 GeV, width 0.0004, 86% of the pT within 0.025. width < 0.0061 (+10.200), mass < 41.4, girth2 < 0.00437 and girth2 < 0.00752 add a lot, but mass < 53.3 (-4.500), mass_over_sum_pt < 0.0764 and width < 0.0061 and centroid_offset > 0.00334 (-3.423, which passes for all of them) take more back than in the centred pencil groups. The value 4.118 (on for 96%) raises q and g; the formula calls them g.
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 5.68. Mostly t (94%); mass 82.3 GeV, width 0.0261, with nearly all pT beyond 0.05 and much of it 0.1-0.2 from the axis. None of the narrow bonuses pass (width < 0.0061 and girth2 < 0.00752 never do), but neither do the light penalties (mass < 53.3 5.8%, mass < 41.4 and centroid_offset < 0.0269 never); instead lam2 > 0.00113 passes only here (+5.920), with girth2 > 0.0188 (85%) and C2 > 0.0512. The neuron ends high, 5.683 (on for 99.8%), so these wide jets also raise the q (+1.443) and g (+0.977) scores; the formula calls them t.
- **light (29 GeV), average width, pT spread over several particles, low pT** — 3.9% of jets, neuron 3.01. A mixture: g 36%, t 23%, W 18%, Z 14%, q 10%; mass 29.4 GeV, width 0.0063, and the lowest total pT here (454 GeV, leader 108.29 GeV). mass < 53.3 and log_sum_pt < 6.84 passes for all (+3.866), with mass < 41.4, max_dr < 0.222, width < 0.0061 (65%) and girth2 < 0.00752 (80%), against mass < 53.3 (-2.371); girth2 < 0.00437 passes for 23% and mass < 41.4 and centroid_offset < 0.0269 for 51%. The value 3.012 (on for 95%) raises q (+0.765) and g (+0.518); the formula calls them g.

### neuron 10: overall jet size (e2, width, mass) (major)

- **What it measures:** Grows steadily with e2 and is pushed up for mass > 15.5 GeV and for mass/pT < 0.108, and down for a very thin jet (lam2 < 0.00341, lam1 < 0.00418); it follows width, e2 and mass/pT very closely (rank correlations 0.873, 0.872, 0.871). Tops sit highest (4.61), Z (1.73) and W (1.70) in the middle, gluons (0.97) and quarks (0.48) lowest.
- *computed — its value:* largest for t (4.61), then Z (1.73), then W (1.70), then g (0.97), then q (0.48); it separates t jets from the rest best (AUC 0.85: large for t)
- **How the class scores use it:** Large size marks a top, so it raises the t score (+20%); small size marks a quark, so it lowers the q score (-14%). It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+20%); lowers the score of q (-14%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.524
z += 63.80 × e2
if lam2 < 0.0034: z += -553 × (0.0034 − lam2)
if mass > 15.45: z += 0.056 × (mass − 15.45)
if mass_over_sum_pt < 0.108: z += 21.02 × (0.108 − mass_over_sum_pt)
if lam1 < 0.0042: z += -651 × (0.0042 − lam1)
if girth2 > 0.0087: z += 394 × (girth2 − 0.0087)
if log_sum_pt > 6.70: z += -22.04 × (log_sum_pt − 6.70)
if mass_over_sum_pt_sq < 0.0039: z += 353 × (0.0039 − mass_over_sum_pt_sq)
if girth2_top2 < 0.004: z += 235 × (0.004 − girth2_top2)
if girth2 < 0.0017: z += -1010 × (0.0017 − girth2)
if tau21 < 0.392: z += -3.01 × (0.392 − tau21)
if lam1 > 0.0084: z += -202 × (lam1 − 0.0084)
if sum_pt > 813: z += 0.011 × (sum_pt − 813)
if girth2_top5 < 0.0023: z += 456 × (0.0023 − girth2_top5)
if eccentricity > 0.903 and z_dr_0p2_0p4 < 0.056: z += 157 × (eccentricity − 0.903) × (0.056 − z_dr_0p2_0p4)
if centroid_offset > 0.0023: z += -22.88 × (centroid_offset − 0.0023)
if mass > 53.33: z += -0.049 × (mass − 53.33)
if n_dr_0p05_0p1 < 3.00: z += 0.180 × (3.00 − n_dr_0p05_0p1)
if lam2 > 0.00019: z += 558 × (lam2 − 0.00019)
if LHA > 0.347: z += -26.48 × (LHA − 0.347)
if LHA > 0.303: z += -10.31 × (LHA − 0.303)
if lam1 < 0.0042 and sum_pt > 988: z += -13.03 × (0.0042 − lam1) × (sum_pt − 988)
if lam2 > 0.00019 and n_pt_above_50 < 8.00: z += -60.61 × (lam2 − 0.00019) × (8.00 − n_pt_above_50)
if lam1 < 0.0042 and log_sum_pt > 6.70: z += 807 × (0.0042 − lam1) × (log_sum_pt − 6.70)
if lam2 > 0.00019 and D2 < 2.06: z += -200 × (lam2 − 0.00019) × (2.06 − D2)
if centroid_offset > 0.038: z += 40.72 × (centroid_offset − 0.038)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (54 GeV), average width, pT spread over several particles** — 31.5% of jets, neuron 2.39. Mostly Z (37%) with W (33%) and t (18%), the largest group (31% of jets); mass 53.7 GeV, width 0.0069, 57% of the pT 0.05-0.1 from the axis. The e2 term (+2.380) and mass > 15.5 (+2.152) carry it, with mass_over_sum_pt < 0.108 and eccentricity > 0.903 and z_dr_0p2_0p4 < 0.0564 (82%) adding a little, against lam2 < 0.00341 (-1.754) and tau21 < 0.392 (-0.734); lam1 < 0.00418 passes for only 5.2%. The value 2.393 (on for 99.9%) raises the t score (+0.897) and lowers q (-0.299); the formula calls them W.
- **very light (9 GeV), very narrow, pT spread over several particles** — 22.8% of jets, neuron 0.10. Mostly g (41%) with quarks (36%); mass 9.2 GeV, width 0.0005, 77% of the pT within 0.025. The e2 term adds only +0.424 and mass > 15.5 passes for 15%, while the thin-jet penalties lam1 < 0.00418 (-2.447), lam2 < 0.00341 (-1.858) and girth2 < 0.00165 pass; mass_over_sum_pt < 0.108 (+1.988), mass_over_sum_pt_sq < 0.0039 and girth2_top2 < 0.00401 nearly balance them. The neuron stays low (0.102, on for 25%); the formula calls them g.
- **light (30 GeV), narrow, pT spread over several particles** — 13.8% of jets, neuron 1.29. A mixture: g 27%, W 26%, Z 20%, q 15%, t 12%; mass 30.4 GeV, width 0.0032, 54% of the pT 0.025-0.05 from the axis. mass_over_sum_pt < 0.108 (+1.286), the e2 term (+1.205) and mass > 15.5 (91%) outweigh lam2 < 0.00341 (-1.776) and lam1 < 0.00418 (88%); girth2 > 0.00868 almost never passes. The value 1.288 (on for 95%) raises t (+0.483) and lowers q; the formula splits them between g and W.
- **very light (11 GeV), very narrow, leading particle 43% of pT, high pT** — 9.7% of jets, neuron 0.03. Mostly q (62%) with gluons (18%); mass 10.6 GeV, width 0.0003, 94% of the pT within 0.025, total pT 951 GeV with a 409.08 GeV leader. log_sum_pt > 6.7 (-3.424), lam1 < 0.00418 (-2.566), lam2 < 0.00341 and girth2 < 0.00165 outweigh mass_over_sum_pt < 0.108 (+2.035) and sum_pt > 813 (+1.552), while the e2 term adds only +0.279. The neuron is off (on for 4.3%) and hardly moves the scores; the formula calls them q.
- **medium-mass (74 GeV), very wide, pT spread over several particles, low pT** — 8.0% of jets, neuron 4.51. Mostly t (79%) with gluons (15%); mass 74.0 GeV, width 0.0182, 32% of the pT 0.05-0.1 and 36% 0.1-0.15 from the axis. The e2 term (+3.900), girth2 > 0.00868 (+3.771) and mass > 15.5 (+3.292) add, against lam1 > 0.00838 (-1.697), lam2 < 0.00341 (84%) and mass > 53.3 (88%); mass_over_sum_pt < 0.108 passes for only 6.6%. The value 4.513 (on for all) raises the t score (+1.692) and lowers q (-0.564); the formula calls them t.
- **medium-mass (67 GeV), average width, leading particle 44% of pT, high pT** — 6.2% of jets, neuron 1.08. Mostly Z (44%) with W (41%); mass 67.3 GeV, width 0.0054, total pT 942 GeV with a 418.79 GeV leading particle. mass > 15.5 (+2.916), the e2 term (+1.806) and sum_pt > 813 (+1.447) are largely cancelled by log_sum_pt > 6.7 (-3.188), lam2 < 0.00341 (-1.846) and tau21 < 0.392; girth2 > 0.00868 passes for only 3.6%. The value 1.076 (on for 88%) raises t (+0.403) and lowers q; the formula calls them W.
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 3.3% of jets, neuron 6.62. Mostly t (74%) with gluons (18%); mass 90.9 GeV, width 0.0311, with 30% of the pT 0.15-0.2 from the axis. girth2 > 0.00868 (+8.850), the e2 term (+5.583) and mass > 15.5 (+4.242) are partly taken back by lam1 > 0.00838 (-4.260), LHA > 0.347 (-2.524) and mass > 53.3. The value 6.624 (on for all) raises the t score (+2.484) and lowers q (-0.828); the formula calls them t.
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 3.1% of jets, neuron 9.84. Mostly t (96%); mass 84.9 GeV, width 0.0272, with almost no pT within 0.05 of the axis and most of it 0.1-0.2 out. girth2 > 0.00868 (+7.303), the e2 term (+5.472), lam2 > 0.000195 (+4.146, which passes for all of them) and mass > 15.5 (+3.900) outweigh lam1 > 0.00838 and LHA > 0.347 (about -2.3 each). This is the neuron's highest group, 9.837 (on for all), raising the t score (+3.689) and lowering q (-1.23); the formula calls them t.
- **very light (16 GeV), very narrow, leading particle 45% of pT, high pT** — 1.4% of jets, neuron 0.00. A quark-gluon mixture (q 44%, g 41%); mass 15.7 GeV, width 0.0004, total pT 1134 GeV with a 508.51 GeV leader and 90% of the pT within 0.025. log_sum_pt > 6.7 (-7.274) and lam1 < 0.00418 and sum_pt > 988 (-6.904) together with lam1 < 0.00418 and lam2 < 0.00341 outweigh sum_pt > 813 (+3.606) and mass_over_sum_pt < 0.108 (+1.986). The neuron is off for all of them; the formula calls them q.
- **very light (15 GeV), very narrow, leading particle 47% of pT, high pT** — 0.3% of jets, neuron 0.00. Mostly g (61%) with quarks (27%), a rare group (0.26%); mass 15.3 GeV, total pT 1408 GeV with a 668.54 GeV leading particle. lam1 < 0.00418 and sum_pt > 988 (-21.442) and log_sum_pt > 6.7 (-11.967) dwarf sum_pt > 813 (+6.696) and lam1 < 0.00418 and log_sum_pt > 6.7 (+1.727). The neuron is off; the formula calls them g.

### neuron 11: compact, centred, non-top jet (major)

- **What it measures:** Pushed up for compact jets (width < 0.00868, its strongest term, girth2 < 0.00668 and < 0.0132) with a pT centroid close to the axis (centroid offset < 0.0499), and pushed down for girth < 0.0872; it rises with eccentricity and total pT (rank correlations 0.393 and 0.373). W jets sit highest (4.34), then Z (2.75), gluons (1.65) and quarks (1.61); tops are lowest (0.65) and zero for 78.2% of them.
- *computed — its value:* largest for W (4.34), then Z (2.75), then g (1.65), then q (1.61), then t (0.65); it separates W jets from the rest best (AUC 0.84: large for W)
- **How the class scores use it:** It is the largest input of the W score, which it raises (+25%), since W jets sit highest on it. It does not enter the g, q, Z or t scores; in particular the Z score does not use it although Z jets sit second on it.
- *computed — used by:* raises the score of W (+25%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.166
if width < 0.0087: z += 967 × (0.0087 − width)
if girth2 < 0.0067: z += 987 × (0.0067 − girth2)
if girth < 0.087: z += -73.40 × (0.087 − girth)
if girth2 < 0.013: z += 320 × (0.013 − girth2)
if e2_sq < 0.0064: z += -785 × (0.0064 − e2_sq)
if centroid_offset < 0.050: z += 68.60 × (0.050 − centroid_offset)
if girth > 0.076: z += -184 × (girth − 0.076)
if lam1 < 0.0084: z += -374 × (0.0084 − lam1)
if girth > 0.076 and n_pt_above_50 < 7.00: z += 53.34 × (girth − 0.076) × (7.00 − n_pt_above_50)
if centroid_offset < 0.050 and log_sum_pt < 6.80: z += -172 × (0.050 − centroid_offset) × (6.80 − log_sum_pt)
if planar_flow < 0.253 and width > 0.0061: z += -4133 × (0.253 − planar_flow) × (width − 0.0061)
if girth > 0.102 and n_pt_above_50 < 7.00: z += -65.79 × (girth − 0.102) × (7.00 − n_pt_above_50)
if planar_flow < 0.253: z += 8.15 × (0.253 − planar_flow)
z += -48.35 × centroid_offset
if centroid_offset > 0.014: z += -79.41 × (centroid_offset − 0.014)
if centroid_offset < 0.050 and pt_7 < 48.72: z += -1.08 × (0.050 − centroid_offset) × (48.72 − pt_7)
if width < 0.0036: z += -455 × (0.0036 − width)
if girth > 0.102: z += -89.30 × (girth − 0.102)
if girth > 0.076 and pt_7 < 40.04: z += -7.17 × (girth − 0.076) × (40.04 − pt_7)
if log_sum_pt < 6.70: z += 2.42 × (6.70 − log_sum_pt)
if log_sum_pt < 6.70 and z_dr_0p2_0p4 < 0.056: z += 56.35 × (6.70 − log_sum_pt) × (0.056 − z_dr_0p2_0p4)
if LHA < 0.197: z += -17.63 × (0.197 − LHA)
if C2 < 0.036: z += -28.11 × (0.036 − C2)
if max_dr < 0.222: z += 3.62 × (0.222 − max_dr)
if mass < 21.78: z += 0.076 × (21.78 − mass)
if lam1 < 0.0048: z += -158 × (0.0048 − lam1)
if max_dr < 0.112: z += -9.31 × (0.112 − max_dr)
if sum_pt_top5 < 687: z += 0.002 × (687 − sum_pt_top5)
if sum_pt_top5 < 687 and n_dr_0p2_0p4 < 2.00: z += -0.0012 × (687 − sum_pt_top5) × (2.00 − n_dr_0p2_0p4)
if planar_flow < 0.253 and mass < 69.61: z += -0.116 × (0.253 − planar_flow) × (69.61 − mass)
if girth > 0.076 and z_dr_0p1_0p2 < 0.205: z += -810 × (girth − 0.076) × (0.205 − z_dr_0p1_0p2)
if centroid_offset < 0.050 and mean_phi2 < 0.0016: z += -8657 × (0.050 − centroid_offset) × (0.0016 − mean_phi2)
if girth2_top3 < 0.0022: z += -284 × (0.0022 − girth2_top3)
if log_sum_pt < 6.70 and dr_0 < 0.112: z += -22.60 × (6.70 − log_sum_pt) × (0.112 − dr_0)
if girth < 0.021: z += -67.90 × (0.021 − girth)
if planar_flow < 0.253 and D2 < 1.68: z += 1.72 × (0.253 − planar_flow) × (1.68 − D2)
if planar_flow < 0.253 and girth2 > 0.013: z += 1374 × (0.253 − planar_flow) × (girth2 − 0.013)
if LHA < 0.155 and z_7 < 0.028: z += 881 × (0.155 − LHA) × (0.028 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles, high pT** — 25.5% of jets, neuron 1.36. Mostly q (51%) with gluons (33%), a quarter of all jets; mass 8.5 GeV, width 0.0002, 95% of the pT within 0.025 and a harder leader (313.41 GeV). width < 0.00868 (+8.191), girth2 < 0.00668 (+6.392) and girth2 < 0.0132 (+4.170) are largely returned by girth < 0.0872 (-5.630), e2_sq < 0.00639 (-4.903) and lam1 < 0.00838 (-3.070), all of which pass; girth > 0.0761 never does. The value 1.359 (on for 94%) raises the W score (+0.509); the formula calls them q.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 21.5% of jets, neuron 2.92. Mostly Z (48%) with W (26%) and t (16%); mass 58.2 GeV, width 0.0072, 62% of the pT 0.05-0.1 from the axis. centroid_offset < 0.0499 (+2.528), girth2 < 0.0132, width < 0.00868 (90%) and planar_flow < 0.253 (88%, +1.483) add, against centroid_offset < 0.0499 and log_sum_pt < 6.8 (-1.600), girth < 0.0872 and girth > 0.0761 (57%); girth2 < 0.00668 passes for only 42%, so this group gets less of the compact bonus. The value 2.925 (on for 81%) raises the W score (+1.097); the formula calls them Z.
- **light (47 GeV), average width, pT spread over several particles** — 19.2% of jets, neuron 4.24. Mostly W (47%) with Z (27%); mass 46.6 GeV, width 0.0045, 48% of the pT 0.025-0.05 and 35% 0.05-0.1 from the axis. All the compact tests pass, width < 0.00868 (+4.031), girth2 < 0.0132, centroid_offset < 0.0499 and girth2 < 0.00668 (about +2.1 to +2.8 each), while the brakes girth < 0.0872 (-2.251) and e2_sq < 0.00639 (-1.824) are smaller than for the thinner jets, and the wide-jet tests (girth > 0.0761, planar_flow < 0.253 and width > 0.0061) almost never pass. This gives the neuron's highest group value, 4.243 (on for 96%), raising the W score (+1.591); the formula calls them W.
- **light (20 GeV), very narrow, pT spread over several particles** — 16.1% of jets, neuron 2.59. Mostly g (35%) in a mixture with q and W (21% each) and Z (16%); mass 20.2 GeV, width 0.0016, with 54% of the pT 0.025-0.05 and 34% within 0.025. width < 0.00868 (+6.859), girth2 < 0.00668 (+5.032) and girth2 < 0.0132 (+3.729) outweigh e2_sq < 0.00639 (-4.184), girth < 0.0872 (-3.938) and lam1 < 0.00838 (-2.588); girth > 0.0761 never passes. The value 2.591 (on for 97%) raises the W score (+0.972); the formula calls them g.
- **medium-mass (65 GeV), very wide, pT spread over several particles, low pT** — 6.3% of jets, neuron 0.02. Mostly t (68%) with gluons (19%); mass 65.0 GeV, width 0.0135, 52% of the pT 0.05-0.1 and 25% 0.1-0.15 from the axis. The compact bonuses are gone (width < 0.00868 1.8%, girth2 < 0.00668 never), and girth > 0.0761 (-4.924), planar_flow < 0.253 and width > 0.0061 (74%, -3.921) and the centroid_offset terms pull down, with girth > 0.0761 and n_pt_above_50 < 7 (+2.844) the only real help. The neuron is off (on for 1.9%) and adds nothing; the formula calls them t.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 4.0% of jets, neuron 0.00. Mostly t (90%); mass 80.0 GeV, width 0.0219, with 38% of the pT 0.1-0.15 from the axis. girth > 0.0761 (-10.972), girth > 0.102 and n_pt_above_50 < 7, girth > 0.102 and girth > 0.0761 and pt_7 < 40 outweigh girth > 0.0761 and n_pt_above_50 < 7 (+5.855), and none of the compact tests pass. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 3.0% of jets, neuron 0.00. Mostly t (71%) with gluons (20%); mass 85.8 GeV, width 0.0225, with 42% of the pT 0.1-0.15 from the axis. planar_flow < 0.253 and width > 0.0061 passes for all of them (-13.819) and girth > 0.0761 (-11.310) adds to it, far more than girth > 0.0761 and n_pt_above_50 < 7 (+6.215) and planar_flow < 0.253 and girth2 > 0.0132 (+2.546) return. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (78 GeV), very wide, pT spread over several particles, low pT** — 2.8% of jets, neuron 0.00. Mostly t (89%); mass 78.2 GeV, width 0.0268, low total pT (499 GeV), with almost no pT within 0.05 of the axis. girth > 0.0761 (-14.241) and girth > 0.0761 and n_pt_above_50 < 7 (+13.877) cancel, leaving girth > 0.102 and n_pt_above_50 < 7 (-11.131), girth > 0.102 and girth > 0.0761 and pt_7 < 40 to push it down. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 0.9% of jets, neuron 0.00. Mostly t (60%) with gluons (25%); mass 85.3 GeV, width 0.0316, with 34% of the pT 0.15-0.2 from the axis. planar_flow < 0.253 and width > 0.0061 (-21.818), girth > 0.0761 (-16.987) and girth > 0.102 and n_pt_above_50 < 7 (-15.237) far outweigh girth > 0.0761 and n_pt_above_50 < 7 (+17.319). The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.00. Mostly t (78%) with gluons (18%); mass 72.7 GeV, width 0.0347, the lowest total pT here (413 GeV), with 39% of the pT 0.15-0.2 from the axis. girth > 0.0761 and n_pt_above_50 < 7 (+25.585) is outweighed by girth > 0.102 and n_pt_above_50 < 7 (-23.415), girth > 0.0761 (-18.718) and girth > 0.0761 and pt_7 < 40 (-7.551). The neuron is off and adds nothing; the formula calls them t.

### neuron 13: compactness (narrower than a top) (major)

- **What it measures:** Driven mostly by girth < 0.148 (its strongest term), lam1 < 0.0164 and width < 0.0132, all pushing it up, i.e. by the jet being narrower than a typical top; small e2 (< 0.0503) pulls it down. It falls with max ΔR, lam1 and width (rank correlations -0.625, -0.602, -0.6); W (4.84), quarks (4.70), Z (4.41) and gluons (4.40) all sit high, tops far lower (1.42, zero for 51.1% of them).
- *computed — its value:* largest for W (4.84), then q (4.70), then Z (4.41), then g (4.40), then t (1.42); it separates t jets from the rest best (AUC 0.14: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-46%); it also raises the W score (+8%) and the Z score (+9%). It does not enter the g or q scores.
- *computed — used by:* raises the score of W (+8%), Z (+9%); lowers the score of t (-46%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.40
if girth < 0.148: z += 42.36 × (0.148 − girth)
if e2 < 0.050: z += -123 × (0.050 − e2)
if lam1 < 0.016: z += 263 × (0.016 − lam1)
if width < 0.013: z += 218 × (0.013 − width)
if centroid_offset < 0.038: z += 61.02 × (0.038 − centroid_offset)
if lam2 < 0.00031 and centroid_offset < 0.050: z += -152474 × (0.00031 − lam2) × (0.050 − centroid_offset)
if width < 0.0075: z += -295 × (0.0075 − width)
if sum_pt_top5 > 902: z += 0.251 × (sum_pt_top5 − 902)
if lam1 < 0.016 and centroid_offset < 0.038: z += -3626 × (0.016 − lam1) × (0.038 − centroid_offset)
if girth < 0.148 and log_sum_pt < 6.80: z += -47.56 × (0.148 − girth) × (6.80 − log_sum_pt)
if sum_pt > 988: z += 0.198 × (sum_pt − 988)
if sum_pt_top5 > 658 and pt_7 < 43.50: z += 0.00082 × (sum_pt_top5 − 658) × (43.50 − pt_7)
if LHA > 0.093: z += -4.99 × (LHA − 0.093)
if lam2 < 0.00031: z += 3421 × (0.00031 − lam2)
if girth < 0.148 and pt_7 < 38.53: z += -0.902 × (0.148 − girth) × (38.53 − pt_7)
if z_top5_slots > 0.931: z += -610 × (z_top5_slots − 0.931)
if lam1 < 0.0065: z += 181 × (0.0065 − lam1)
if sum_pt_top5 > 902 and D2 < 3.89: z += 0.092 × (sum_pt_top5 − 902) × (3.89 − D2)
if tau21 < 0.501 and max_dr > 0.016: z += -15.39 × (0.501 − tau21) × (max_dr − 0.016)
if lam1 < 0.016 and pt_6 < 56.53: z += -1.87 × (0.016 − lam1) × (56.53 − pt_6)
if sum_pt_top5 > 658: z += -0.0076 × (sum_pt_top5 − 658)
if mass > 53.33: z += -0.051 × (mass − 53.33)
if sum_pt > 988 and n_pt_above_50 > 6.00: z += -0.133 × (sum_pt − 988) × (n_pt_above_50 − 6.00)
if z_7 < 0.028: z += -208 × (0.028 − z_7)
if sum_pt_top5 > 840 and n_pt_above_50 > 2.00: z += 0.011 × (sum_pt_top5 − 840) × (n_pt_above_50 − 2.00)
if sum_pt_top5 > 902 and n_pt_above_50 > 6.00: z += -0.248 × (sum_pt_top5 − 902) × (n_pt_above_50 − 6.00)
if sum_pt > 988 and D2 < 3.89: z += 0.029 × (sum_pt − 988) × (3.89 − D2)
if e2 < 0.050 and pt_dispersion > 0.397: z += 131 × (0.050 − e2) × (pt_dispersion − 0.397)
if pt_6 < 31.91: z += -0.094 × (31.91 − pt_6)
if sum_pt_top5 > 658 and z_7 > 0.023: z += -0.547 × (sum_pt_top5 − 658) × (z_7 − 0.023)
if C2 > 0.067: z += -55.24 × (C2 − 0.067)
if pt_7 < 25.58: z += -0.096 × (25.58 − pt_7)
if girth > 0.102 and pt_4 > 81.38: z += -25.00 × (girth − 0.102) × (pt_4 − 81.38)
if sum_pt_top5 > 658 and tau32 < 0.363: z += -0.140 × (sum_pt_top5 − 658) × (0.363 − tau32)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (15 GeV), very narrow, pT spread over several particles** — 40.0% of jets, neuron 4.98. A quark-gluon mixture (q 36%, g 33%) with some W and Z, 40% of all jets; mass 15.1 GeV, width 0.0009, 68% of the pT within 0.025. The compactness terms girth < 0.148 (+5.386), lam1 < 0.0164 (+4.084), width < 0.0132 (+2.677) and centroid_offset < 0.0378 all pass, and only e2 < 0.0503 (-5.188) and width < 0.00752 (-1.942) push back. The value 4.979 (on for 96%) lowers the t score (-2.023) and raises W and Z a little; the formula calls them g.
- **medium-mass (52 GeV), average width, pT spread over several particles** — 38.9% of jets, neuron 4.27. Mostly Z (37%) with W (34%) and t (14%), 39% of all jets; mass 51.7 GeV, width 0.0061, 54% of the pT 0.05-0.1 from the axis. girth < 0.148 (+3.344), lam1 < 0.0164 (+2.772), width < 0.0132 and centroid_offset < 0.0378 pass, and the brakes are smaller than for thin jets (e2 < 0.0503 -2.000, lam2 < 0.000306 and centroid_offset < 0.0499 -1.095, width < 0.00752 80%). The value 4.267 (on for 98%) lowers the t score (-1.734) and raises W (+0.3) and Z; the formula calls them W.
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 16.9% of jets, neuron 0.42. Mostly t (79%) with gluons (15%); mass 77.0 GeV, width 0.0215, with pT spread 0.05-0.2 from the axis. The compactness bonuses mostly fail (girth < 0.148 71%, lam1 < 0.0164 42%, width < 0.0132 13%), and LHA > 0.0932 (-1.506), mass > 53.3 (88%, -1.255), C2 > 0.0673 and tau21 < 0.501 and max_dr > 0.0156 pull down. The neuron stays low (0.425, on for 29%), so the t score loses only -0.173 here; the formula calls them t.
- **light (24 GeV), very narrow, leading particle 57% of pT, high pT** — 2.5% of jets, neuron 4.01. Mostly q (64%) with W and Z; mass 24.5 GeV, width 0.0013, total pT 988 GeV of which the leader carries 557.72 GeV, while the softest particles are only about 10-20 GeV. z_top5_slots > 0.931 passes for 92% and subtracts -16.192, set against sum_pt_top5 > 902 (85%, +12.080), sum_pt_top5 > 658 and pt_7 < 43.5 (+7.708) and sum_pt > 988 (53%), with z_7 < 0.0281 (-3.628) as a further brake. The balance leaves 4.007, on for 53%, lowering the t score (-1.628); the formula calls them q.
- **light (29 GeV), very narrow, leading particle 51% of pT, high pT** — 0.9% of jets, neuron 8.15. Mostly q (45%) with gluons (28%), a small group (0.92%); mass 28.7 GeV, width 0.0014, total pT 1139 GeV with a 584.64 GeV leading particle. sum_pt_top5 > 902 (+38.015), sum_pt > 988 (+29.776) and sum_pt_top5 > 902 and D2 < 3.89 (74%, +17.220) add far more than z_top5_slots > 0.931 (43%, -6.462) and e2 < 0.0503 take away, so their sum (about 95 on average) is far above the largest value the neuron can hold (just under 16); on the network's number grid it wraps around, so the value is effectively scrambled (average 8.148, on for 99.8%). This lowers the t score strongly (-3.31) and raises W and Z; the formula calls them q.
- **light (26 GeV), very narrow, pT spread over several particles, high pT** — 0.4% of jets, neuron 6.53. Mostly g (66%), a rare group (0.4%); mass 26.4 GeV, total pT 1123 GeV shared among many hard particles (4th hardest 120.1 GeV against 71.7 GeV for all jets). sum_pt > 988 (+26.578) is cancelled by sum_pt > 988 and n_pt_above_50 > 6 (-28.818); sum_pt > 988 and D2 < 3.89 (+9.969) and, for about half, sum_pt_top5 > 902 against sum_pt_top5 > 902 and n_pt_above_50 > 6 decide the rest. The value 6.53 (on for 97%) lowers the t score (-2.653); the formula calls them g. The sum (about 17 on average) exceeds the largest value the neuron can hold (just under 16), so for many of these jets the value wraps around on the network's number grid.
- **light (31 GeV), very narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 7.28. Mostly g (60%), only 0.15% of jets; mass 30.8 GeV, total pT 1267 GeV with many hard particles (4th 125.48 GeV). Four large bonuses, sum_pt > 988 (+55.135), sum_pt_top5 > 902 (+40.321), sum_pt_top5 > 902 and D2 < 3.89 and sum_pt > 988 and D2 < 3.89, face two large brakes for many-particle jets, sum_pt_top5 > 902 and n_pt_above_50 > 6 (-60.855) and sum_pt > 988 and n_pt_above_50 > 6 (-60.292). The value 7.283 (on for 92%) lowers the t score (-2.959); the formula calls them g. The sum (about 43 on average) exceeds the largest value the neuron can hold (just under 16), so for many of these jets the value wraps around on the network's number grid.
- **light (29 GeV), very narrow, leading particle 56% of pT, high pT** — 0.1% of jets, neuron 8.60. Mostly g (43%) with quarks (38%), only 0.15% of jets; mass 29.1 GeV, total pT 1405 GeV with a 789.66 GeV leading particle. The pT bonuses are huge, sum_pt_top5 > 902 (+101.790), sum_pt > 988 (+82.321) and sum_pt_top5 > 902 and D2 < 3.89 (80%), and with few hard particles the many-particle brakes do not apply, so their sum (about 273 on average) is far above the largest value the neuron can hold (just under 16) and wraps around on the network's number grid; the average value is 8.6. This lowers the t score (-3.494) and raises W and Z; the formula calls them g.
- **light (37 GeV), very narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 6.30. Mostly g (71%), the rarest group (0.06%); mass 36.5 GeV, total pT 1473 GeV with pT spread over many hard particles (4th 132.72 GeV). The largest terms of this formula almost cancel: sum_pt_top5 > 902 and n_pt_above_50 > 6 (-147.464) and sum_pt > 988 and n_pt_above_50 > 6 (-111.525) against sum_pt > 988 (+95.830), sum_pt_top5 > 902 (+88.090) and sum_pt_top5 > 902 and D2 < 3.89 (+76.503). The value 6.296 (on for 83%) lowers the t score (-2.558); the formula calls them g. The sum (about 57 on average) exceeds the largest value the neuron can hold (just under 16), so for many of these jets the value wraps around on the network's number grid.

### neuron 14: intermediate width (Z-sized spread) (major)

- **What it measures:** Pushed down for the narrowest jets (girth < 0.0872, width < 0.0061 and < 0.00752) but up for girth2 < 0.0132, width < 0.00868 and e2 < 0.0445, so it peaks for jets of intermediate width, a bit wider than a typical W; it falls with τ21 and rises with eccentricity and the number of particles above 10 GeV (rank correlations -0.376, 0.372, 0.366). Z jets sit highest (1.06), well above tops (0.29), W (0.15), gluons (0.13) and quarks (0.10).
- *computed — its value:* largest for Z (1.06), then t (0.29), then W (0.15), then g (0.13), then q (0.10); it separates Z jets from the rest best (AUC 0.78: large for Z)
- **How the class scores use it:** It is the W/Z separator: Z jets sit high and W jets low on it, so it raises the Z score (+5%) and lowers the W score (-8%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+5%); lowers the score of W (-8%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.932
if girth < 0.087: z += -125 × (0.087 − girth)
if girth2 < 0.013: z += 547 × (0.013 − girth2)
if e2 < 0.044: z += 210 × (0.044 − e2)
if width < 0.0061: z += -1308 × (0.0061 − width)
if lam1 > 0.0042: z += 852 × (lam1 − 0.0042)
if width < 0.0075: z += -756 × (0.0075 − width)
if width < 0.0087: z += 571 × (0.0087 − width)
if C2 < 0.067: z += -40.09 × (0.067 − C2)
if width < 0.0075 and n_dr_0p1_0p2 < 3.00: z += 161 × (0.0075 − width) × (3.00 − n_dr_0p1_0p2)
if mass < 69.61: z += -0.048 × (69.61 − mass)
if mass_over_sum_pt_sq < 0.0082: z += -315 × (0.0082 − mass_over_sum_pt_sq)
if e2_sq < 0.017: z += 111 × (0.017 − e2_sq)
if lam1 > 0.006: z += -529 × (lam1 − 0.006)
if max_dr < 0.081: z += -79.64 × (0.081 − max_dr)
if e2 < 0.041: z += -69.58 × (0.041 − e2)
if lam1 > 0.0042 and D2 > 0.415: z += -792 × (lam1 − 0.0042) × (D2 − 0.415)
if mass < 86.40: z += 0.026 × (86.40 − mass)
if e2_sq < 0.0058: z += 444 × (0.0058 − e2_sq)
if e2 < 0.038: z += 68.79 × (0.038 − e2)
if lam1 > 0.0025: z += -236 × (lam1 − 0.0025)
if mass_over_sum_pt_sq < 0.012: z += -123 × (0.012 − mass_over_sum_pt_sq)
if max_dr < 0.177: z += -12.26 × (0.177 − max_dr)
if girth2 < 0.013 and n_dr_0p1_0p2 < 3.00: z += -41.30 × (0.013 − girth2) × (3.00 − n_dr_0p1_0p2)
if LHA < 0.303: z += -9.54 × (0.303 − LHA)
if z_dr_0p05_0p1 < 0.588: z += 1.87 × (0.588 − z_dr_0p05_0p1)
if lam1 > 0.0025 and D2 > 0.415: z += 332 × (lam1 − 0.0025) × (D2 − 0.415)
if n_dr_0p05_0p1 < 5.00: z += -0.153 × (5.00 − n_dr_0p05_0p1)
if lam1 > 0.0054: z += 147 × (lam1 − 0.0054)
if z_dr_0p05_0p1 < 0.588 and C2 < 0.067: z += -23.25 × (0.588 − z_dr_0p05_0p1) × (0.067 − C2)
if lam1 > 0.012: z += -270 × (lam1 − 0.012)
if girth < 0.034: z += -40.58 × (0.034 − girth)
if lam1 > 0.006 and D2 > 1.68: z += -3530 × (lam1 − 0.006) × (D2 − 1.68)
if z_dr_0_0p05 > 0.608: z += -1.36 × (z_dr_0_0p05 − 0.608)
if z_dr_0p05_0p1 < 0.588 and n_dr_0p1_0p2 < 3.00: z += 0.269 × (0.588 − z_dr_0p05_0p1) × (3.00 − n_dr_0p1_0p2)
if centroid_offset > 0.013: z += -26.85 × (centroid_offset − 0.013)
if girth2 < 0.013 and eccentricity > 0.970: z += 3484 × (0.013 − girth2) × (eccentricity − 0.970)
if width < 0.0087 and D2 < 1.00: z += -590 × (0.0087 − width) × (1.00 − D2)
if centroid_offset > 0.050: z += -224 × (centroid_offset − 0.050)
if tau21 < 0.136: z += 10.30 × (0.136 − tau21)
if e2 < 0.041 and D2 < 1.00: z += 223 × (0.041 − e2) × (1.00 − D2)
if planar_flow < 0.112 and max_dr < 0.160: z += -143 × (0.112 − planar_flow) × (0.160 − max_dr)
if D2 < 1.12: z += -0.617 × (1.12 − D2)
if girth2 < 0.0044 and D2 < 0.876: z += -9430 × (0.0044 − girth2) × (0.876 − D2)
if planar_flow < 0.112 and centroid_offset > 0.0095: z += 431 × (0.112 − planar_flow) × (centroid_offset − 0.0095)
if lam1 > 0.0054 and max_dr < 0.160: z += 8634 × (lam1 − 0.0054) × (0.160 − max_dr)
if e2 < 0.038 and D2 < 1.00: z += -202 × (0.038 − e2) × (1.00 − D2)
if planar_flow < 0.112 and centroid_offset > 0.018: z += -656 × (0.112 − planar_flow) × (centroid_offset − 0.018)
if lam1 > 0.0073: z += 44.83 × (lam1 − 0.0073)
if lam1 > 0.0025 and D2 > 1.68: z += -331 × (lam1 − 0.0025) × (D2 − 1.68)
if girth2 < 0.013 and centroid_offset > 0.031: z += -6567 × (0.013 − girth2) × (centroid_offset − 0.031)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles** — 28.3% of jets, neuron 0.00. Mostly q (46%) with gluons (34%), 28% of all jets; mass 7.8 GeV, width 0.0003, 88% of the pT within 0.025. The too-narrow penalties girth < 0.0872 (-9.358), width < 0.0061 (-7.641) and width < 0.00752 (-5.491) outweigh e2 < 0.0445 (+8.322), girth2 < 0.0132 (+7.100) and width < 0.00868 (+4.808), and lam1 > 0.00418 never passes. The neuron is off for all of them and adds nothing; the formula calls them q.
- **medium-mass (57 GeV), average width, pT spread over several particles** — 22.4% of jets, neuron 1.05. Mostly Z (46%) with W (30%) and t (15%); mass 57.1 GeV, width 0.0071, 67% of the pT 0.05-0.1 from the axis. Being a little wider than the narrow groups, these jets mostly escape width < 0.0061 (23%) and width < 0.00752 (66%) and girth < 0.0872 costs only -1.219, while girth2 < 0.0132 (+3.350), lam1 > 0.00418 (+2.294), e2_sq < 0.0172 and e2 < 0.0445 (80%) add; C2 < 0.0673 takes -1.786. The neuron's highest group value, 1.047 (on for 73%), raises the Z score (+0.393) and lowers W (-0.785); the formula calls them Z.
- **light (47 GeV), average width, pT spread over several particles** — 17.3% of jets, neuron 0.47. Mostly W (43%) with Z (29%); mass 46.7 GeV, width 0.0046, 51% of the pT 0.025-0.05 from the axis. girth2 < 0.0132 (+4.713), e2 < 0.0445 (+3.892) and width < 0.00868 (+2.316) are largely cancelled by girth < 0.0872 (-3.821), width < 0.00752 and width < 0.0061 (96%), and lam1 > 0.00418 passes for 61% only. The value 0.475 (on for 39%) lowers the W score (-0.356) and raises Z; the formula calls them W.
- **light (26 GeV), very narrow, pT spread over several particles** — 14.6% of jets, neuron 0.07. A mixture: g 31%, W 23%, q 23%, Z 17%; mass 26.2 GeV, width 0.0019, with 51% of the pT 0.025-0.05 and 36% within 0.025. girth < 0.0872 (-6.559), width < 0.0061 (-5.483) and width < 0.00752 (-4.244) roughly cancel e2 < 0.0445 (+6.463), girth2 < 0.0132 (+6.198) and width < 0.00868 (+3.866), and lam1 > 0.00418 never passes. The neuron is nearly off (0.074, on for 9%); the formula calls them g.
- **medium-mass (68 GeV), very wide, pT spread over several particles, low pT** — 6.9% of jets, neuron 0.28. Mostly t (73%) with gluons (18%); mass 68.5 GeV, width 0.0152, with pT spread 0.05-0.15 from the axis. The narrow-jet tests mostly fail (girth < 0.0872 6.6%, e2 < 0.0445 17%), so the value comes from a ladder of lam1 steps: lam1 > 0.00418 (+8.312) and lam1 > 0.00543 add, while lam1 > 0.00595 (-4.224), lam1 > 0.00246, lam1 > 0.00418 and D2 > 0.415 (64%) and C2 < 0.0673 subtract. The value 0.277 is on for 32%, lowering W slightly and raising Z; the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 4.8% of jets, neuron 0.00. Mostly t (90%); mass 80.8 GeV, width 0.0237, with 32% of the pT 0.1-0.15 from the axis. lam1 > 0.00418 (+12.644), lam1 > 0.00246 and D2 > 0.415 and lam1 > 0.00543 are outweighed by lam1 > 0.00418 and D2 > 0.415 (-9.560), lam1 > 0.00595 (-6.913) and lam1 > 0.00246. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (89 GeV), very wide, pT spread over several particles, low pT** — 2.6% of jets, neuron 0.02. Mostly t (68%) with gluons (23%); mass 89.2 GeV, width 0.0277, with almost no pT within 0.05 and most of it 0.1-0.2 from the axis. lam1 > 0.00418 (+19.278) and lam1 > 0.00543 are cancelled by lam1 > 0.00595 (-11.031), lam1 > 0.00246, lam1 > 0.012 (-3.998) and, for half, lam1 > 0.00418 and D2 > 0.415. The neuron is nearly off (0.018, on for 7.3%); the formula calls them t.
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 1.7% of jets, neuron 0.00. Mostly t (82%); mass 86.8 GeV, width 0.0334, softer total pT (500 GeV), with pT spread broadly out to beyond 0.2 from the axis. lam1 > 0.00418 (+21.969) and lam1 > 0.00246 and D2 > 0.415 are outweighed by lam1 > 0.00418 and D2 > 0.415 (-13.054), lam1 > 0.00595 (-12.702), lam1 > 0.00246 and lam1 > 0.012. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (66 GeV), very wide, pT spread over several particles, low pT** — 1.1% of jets, neuron 0.00. Mostly t (72%) with gluons (17%); mass 65.7 GeV, width 0.0147, 48% of the pT 0.05-0.1 from the axis. lam1 > 0.00595 and D2 > 1.68 passes for all of them (-12.275) and adds to lam1 > 0.00418 and D2 > 0.415 (-12.028), more than lam1 > 0.00418 (+7.008) and lam1 > 0.00246 and D2 > 0.415 (+6.176) return. The neuron is off and adds nothing; the formula calls them t.
- **medium-mass (71 GeV), very wide, pT spread over several particles** — 0.3% of jets, neuron 0.00. Mostly t (83%), a rare group (0.28%); mass 71.3 GeV, width 0.0163, with a hard leading particle (224.04 GeV) and more pT near the axis than other top-like groups. lam1 > 0.00595 and D2 > 1.68 subtracts -34.844 and lam1 > 0.00418 and D2 > 0.415 -20.337, far more than lam1 > 0.00246 and D2 > 0.415 (+10.011) and lam1 > 0.00418 (+9.020) add. The neuron is off and adds nothing; the formula calls them t.

### neuron 0: compact, massive, elongated two-prong jet (moderate)

- **What it measures:** Pushed up for compact jets (width < 0.00868 and girth2 < 0.0132 are its two strongest terms) and pushed down for light jets (mass < 29.6 GeV), for the very narrowest jets (girth < 0.0761, lam1 < 0.00838) and for small mass/pT (< 0.131); it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.566, -0.566, -0.553). Z (1.87) and W (1.78) jets sit highest, tops (0.42), gluons (0.21) and quarks (0.18) low; it is zero for 83.8% of gluons and 87.6% of quarks.
- *computed — its value:* largest for Z (1.87), then W (1.78), then t (0.42), then g (0.21), then q (0.18); it separates Z jets from the rest best (AUC 0.76: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, it raises the W score (+9%) and lowers the g score (-5%). It does not enter the q, Z or t scores; the Z score, although Z jets sit highest on it, relies on the related two-prong neuron 7 instead.
- *computed — used by:* raises the score of W (+9%); lowers the score of g (-5%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.906
if width < 0.0087: z += 886 × (0.0087 − width)
if girth2 < 0.013: z += 469 × (0.013 − girth2)
if mass < 29.64: z += -0.395 × (29.64 − mass)
if mass < 29.64 and dr_0 < 0.112: z += 2.98 × (29.64 − mass) × (0.112 − dr_0)
if girth < 0.076: z += -63.41 × (0.076 − girth)
if lam1 < 0.0084: z += -300 × (0.0084 − lam1)
if girth < 0.087: z += -35.34 × (0.087 − girth)
if mass_over_sum_pt < 0.131: z += -16.67 × (0.131 − mass_over_sum_pt)
if mass < 64.62: z += -0.034 × (64.62 − mass)
if width < 0.0044: z += -588 × (0.0044 − width)
if mass < 56.92: z += -0.036 × (56.92 − mass)
if girth2 < 0.019: z += 57.91 × (0.019 − girth2)
if e2 < 0.020: z += 135 × (0.020 − e2)
if girth2_top3 < 0.004: z += 386 × (0.004 − girth2_top3)
if LHA < 0.293: z += -9.25 × (0.293 − LHA)
if lam1 < 0.0005: z += 6615 × (0.0005 − lam1)
if mass_over_sum_pt_sq < 0.0082: z += -129 × (0.0082 − mass_over_sum_pt_sq)
if girth2_top3 < 0.0079: z += -114 × (0.0079 − girth2_top3)
if mass < 64.62 and pt_7 < 40.04: z += -0.0022 × (64.62 − mass) × (40.04 − pt_7)
if centroid_offset < 0.031: z += 29.81 × (0.031 − centroid_offset)
if sum_pt > 813: z += -0.014 × (sum_pt − 813)
if girth2_top5 < 0.0083: z += -81.62 × (0.0083 − girth2_top5)
if lam1 < 0.0054 and n_dr_0p2_0p4 < 2.00: z += -86.45 × (0.0054 − lam1) × (2.00 − n_dr_0p2_0p4)
if lam1 < 0.0015: z += -826 × (0.0015 − lam1)
if mass < 64.62 and centroid_offset > 0.013: z += 1.99 × (64.62 − mass) × (centroid_offset − 0.013)
if sum_pt_top5 > 687: z += 0.0086 × (sum_pt_top5 − 687)
if mass < 29.64 and phi_1 > -0.059: z += 0.735 × (29.64 − mass) × (phi_1 − -0.059)
if log_sum_pt > 6.64 and dr_4 < 0.072: z += 114 × (log_sum_pt − 6.64) × (0.072 − dr_4)
if z_dr_0_0p05 > 0.848: z += 4.90 × (z_dr_0_0p05 − 0.848)
if girth2 < 0.019 and eccentricity > 0.960: z += 1546 × (0.019 − girth2) × (eccentricity − 0.960)
if lam1 < 0.0054: z += 112 × (0.0054 − lam1)
if mass_over_sum_pt_sq < 0.0082 and n_pt_above_50 < 8.00: z += 19.46 × (0.0082 − mass_over_sum_pt_sq) × (8.00 − n_pt_above_50)
if girth2_top5 < 0.00066: z += -1658 × (0.00066 − girth2_top5)
if mass < 21.78: z += -0.041 × (21.78 − mass)
if sum_pt > 902 and pt_7 > 29.04: z += -0.0018 × (sum_pt − 902) × (pt_7 − 29.04)
if sum_pt > 902: z += -0.011 × (sum_pt − 902)
if log_sum_pt > 6.64 and dr_3 < 0.078: z += 45.50 × (log_sum_pt − 6.64) × (0.078 − dr_3)
if lam1 < 0.0065 and D2 < 0.876: z += -1528 × (0.0065 − lam1) × (0.876 − D2)
if girth2_top5 < 0.00066 and dr_7 < 0.223: z += 4316 × (0.00066 − girth2_top5) × (0.223 − dr_7)
if sum_pt_top5 > 687 and dr_4 < 0.063: z += -0.079 × (sum_pt_top5 − 687) × (0.063 − dr_4)
if mass < 56.92 and C2 > 0.024: z += 1.17 × (56.92 − mass) × (C2 − 0.024)
if sum_pt > 902 and pt_7 < 25.58: z += 0.0013 × (sum_pt − 902) × (25.58 − pt_7)
if sum_pt_top5 > 687 and z_7 > 0.023: z += -0.320 × (sum_pt_top5 − 687) × (z_7 − 0.023)
if mass < 29.64 and D2 < 0.876: z += -2.85 × (29.64 − mass) × (0.876 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (58 GeV), average width, pT spread over several particles** — 18.9% of jets, neuron 2.49. Mostly Z (52%) with many W (26%) and some t; mass 57.9 GeV on average, width near average, and most pT sits 0.05-0.1 from the axis (66% of it, against 27% for all jets), as expected for two hard prongs. girth2 < 0.0132 always passes and adds +2.886, width < 0.00868 passes 95% of the time (+1.421), and girth2 < 0.0188 and centroid_offset < 0.0312 add a little more, while mass < 29.6 almost never passes, so the big light-jet penalty is absent. The narrowness penalties (lam1 < 0.00838, girth < 0.0872, mass_over_sum_pt < 0.131) take back only about 1.6, leaving the neuron at 2.49 and on for 96%; this raises the W score (+0.856) and lowers the g score (-0.389).
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 18.8% of jets, neuron 0.11. Mostly t (76%) with some gluons (15%); mass 75.0 GeV and width 0.0204, about three times the average, with much of the pT 0.1-0.2 from the axis and a softer total pT (566 GeV). width < 0.00868 never passes and girth2 < 0.0132 passes only 22% of the time, so the two terms that build this neuron are missing; small pieces from mass < 64.6 and centroid_offset > 0.0126 and centroid_offset < 0.0312 are cancelled by mass_over_sum_pt < 0.131 and mass < 64.6. The neuron stays near zero (0.11, on for 14%) and adds almost nothing to any class score; the formula calls them t.
- **light (48 GeV), average width, pT spread over several particles** — 17.7% of jets, neuron 1.84. Mostly W (49%) mixed with Z (28%) and a little g and t; mass 48.4 GeV, narrower than average (width 0.0048), with the pT 0.025-0.1 from the axis and a slightly harder leading particle. The compactness terms girth2 < 0.0132 (+3.959) and width < 0.00868 (+3.437) always pass; the narrowness penalties lam1 < 0.00838, mass_over_sum_pt < 0.131, girth < 0.0761 and girth < 0.0872 also always pass and take about 1 each, and mass < 29.6 passes only 5.3%. The balance gives 1.841 (on for 92%), which raises the W score (+0.633) and lowers the g score (-0.288); the formula calls them W.
- **very light (6 GeV), very narrow, pT spread over several particles** — 14.4% of jets, neuron 0.00. A gluon-quark mixture (q 44%, g 41%); mass only 6.3 GeV and width 0.0002, with 97% of the pT within 0.025 of the axis. Every compactness test passes, so width < 0.00868 (+7.549), mass < 29.6 and dr_0 < 0.112 (+7.165) and girth2 < 0.0132 (+6.137) add a lot, but mass < 29.6 (-9.215), girth < 0.0761 (-4.165), girth < 0.0872, width < 0.00437 and lam1 < 0.00838 (about -2.5 each) take more. The sum falls below zero, the neuron is off for all of them and adds nothing to any score; the formula splits them almost evenly between g and q.
- **light (35 GeV), narrow, pT spread over several particles** — 9.8% of jets, neuron 0.83. A mixture with no majority (W 32%, q 22%, g 21%, Z 19%); mass 35.3 GeV, narrow (width 0.0024), with a hard leading particle (307.7 GeV against 240.22 for all jets) and pT mostly 0.025-0.05 from the axis. All the compactness and narrowness tests pass (width < 0.00868 +5.551, girth2 < 0.0132 +5.079 against girth < 0.0761 -2.473, lam1 < 0.00838, girth < 0.0872, mass_over_sum_pt < 0.131, width < 0.00437), so the value hinges on mass < 29.6, which passes for 36% of them, and sum_pt > 813, which passes for 45%. The neuron ends at 0.827 and is on for about half, mildly raising W (+0.284) and lowering g (-0.129); the formula calls them W.
- **very light (17 GeV), very narrow, pT spread over several particles** — 8.2% of jets, neuron 0.01. Mostly g (42%) with many quarks (34%); mass 16.7 GeV, width 0.001, with 58% of the pT within 0.025 and most of the rest 0.025-0.05 from the axis. The positive terms width < 0.00868 (+6.825), girth2 < 0.0132 (+5.753) and mass < 29.6 and dr_0 < 0.112 (+3.447) are outweighed by mass < 29.6 (-5.129) together with the narrowness penalties girth < 0.0761, lam1 < 0.00838, girth < 0.0872 and width < 0.00437, which all pass. The neuron is essentially off (0.006) and adds nothing to the scores; the formula calls them g.
- **very light (8 GeV), very narrow, leading particle 46% of pT, high pT** — 6.6% of jets, neuron 0.00. Mostly q (69%) with some gluons; mass 7.6 GeV, width 0.0001, 98% of the pT within 0.025, and a very hard leading particle (459.9 GeV, total 993 GeV). As in the other light groups, mass < 29.6 (-8.701) and girth < 0.0761 (-4.409) plus the other narrowness penalties outweigh width < 0.00868 (+7.608), mass < 29.6 and dr_0 < 0.112 (+7.027) and girth2 < 0.0132; lam1 < 0.000505 passes almost always here (+2.837) but cannot lift the sum. The neuron is off (on for 0.2%) and adds nothing; the formula calls them q.
- **very light (7 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 0.03. A mixture: g 36%, Z 23%, W 20%, q 13%; very light (6.7 GeV) but not pencil-thin (width 0.0013), with 58% of the pT 0.025-0.05 from the axis. mass < 29.6 always passes and subtracts -9.052, more than the compactness terms (width < 0.00868, girth2 < 0.0132, mass < 29.6 and dr_0 < 0.112) can return. What sets this group apart is mass < 64.6 and centroid_offset > 0.0126, which passes 98% of the time (+2.199), but the neuron still stays off (on for 3.7%); the formula calls them g.
- **light (25 GeV), very narrow, pT spread over several particles, high pT** — 0.4% of jets, neuron 0.00. Mostly g (70%), a rare group (0.39% of jets); total pT 1222 GeV with all eight particles hard, mass 25.2 GeV and 75% of the pT within 0.025. sum_pt > 902 and pt_7 > 29 always passes and subtracts -16.976, and sum_pt > 813 (-5.689) and sum_pt > 902 (-3.440) add more penalty, burying the compactness terms. The neuron is off and adds nothing; the formula calls them g.
- **light (43 GeV), narrow, pT spread over several particles, high pT** — 0.0% of jets, neuron 0.00. Mostly g (65%), only 0.03% of jets; total pT 1539 GeV, leading particle 531.66 GeV, mass 42.6 GeV. sum_pt > 902 and pt_7 > 29 passes for all of them and subtracts -58.450 on average, with sum_pt > 813 (-10.101) on top; sum_pt_top5 > 687 (+4.931) is cancelled by sum_pt_top5 > 687 and z_7 > 0.0232 (-4.807). The neuron is off and adds nothing; the formula calls them g.

### neuron 3: radiation at wide angle (ΔR 0.2-0.4) (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and girth2 < 0.0132 are its two strongest terms, both pushing down) and pushed up for e2 < 0.0445; it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.634 and 0.626). Tops sit clearly highest (2.74), then gluons (0.75) and quarks (0.36); Z jets are rarely on it (0.16) and W jets almost never (0.03, zero for 97.1% of them).
- *computed — its value:* largest for t (2.74), then g (0.75), then q (0.36), then Z (0.16), then W (0.03); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so it lowers the W score (-12%) and the Z score (-18%). It does not enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-12%), Z (-18%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 3.65
if girth2 < 0.0087: z += -1807 × (0.0087 − girth2)
if girth2 < 0.013: z += -466 × (0.013 − girth2)
if e2 < 0.044: z += 147 × (0.044 − e2)
if girth2 < 0.0044: z += 1157 × (0.0044 − girth2)
if mass_over_sum_pt > 0.068 and n_dr_0_0p05 < 5.00: z += 25.18 × (mass_over_sum_pt − 0.068) × (5.00 − n_dr_0_0p05)
if lam1 < 0.0073: z += 441 × (0.0073 − lam1)
if lam1 > 0.0042: z += -430 × (lam1 − 0.0042)
if e2 > 0.029: z += -97.05 × (e2 − 0.029)
if z_dr_0p1_0p2 < 0.468: z += 2.62 × (0.468 − z_dr_0p1_0p2)
if e2 < 0.038: z += 49.72 × (0.038 − e2)
if mass > 36.23 and eccentricity > 0.621: z += 0.179 × (mass − 36.23) × (eccentricity − 0.621)
if mass_over_sum_pt > 0.068: z += -37.14 × (mass_over_sum_pt − 0.068)
if width > 0.019: z += 590 × (width − 0.019)
if centroid_offset > 0.014: z += 62.02 × (centroid_offset − 0.014)
if mass > 36.23: z += -0.029 × (mass − 36.23)
if lam1 > 0.012: z += -291 × (lam1 − 0.012)
if LHA > 0.326: z += 28.59 × (LHA − 0.326)
if width < 0.0067: z += -106 × (0.0067 − width)
if mass_over_sum_pt > 0.090: z += 31.60 × (mass_over_sum_pt − 0.090)
if e2 > 0.050: z += 66.04 × (e2 − 0.050)
if mass_over_sum_pt_sq > 0.0064: z += -88.92 × (mass_over_sum_pt_sq − 0.0064)
if max_dr > 0.145: z += 7.91 × (max_dr − 0.145)
if lam1 > 0.0084: z += -103 × (lam1 − 0.0084)
if mass_over_sum_pt > 0.090 and pt_6 < 56.53: z += 1.27 × (mass_over_sum_pt − 0.090) × (56.53 − pt_6)
if mass_over_sum_pt > 0.108: z += -33.19 × (mass_over_sum_pt − 0.108)
if mass > 69.61: z += -0.071 × (mass − 69.61)
if lam1 > 0.0042 and z_7 < 0.075: z += -3123 × (lam1 − 0.0042) × (0.075 − z_7)
if lam1 > 0.016: z += 219 × (lam1 − 0.016)
if C2 > 0.095: z += -172 × (C2 − 0.095)
if max_dr > 0.145 and dr_1 < 0.048: z += -692 × (max_dr − 0.145) × (0.048 − dr_1)
if LHA > 0.313 and max_dr < 0.145: z += -2200 × (LHA − 0.313) × (0.145 − max_dr)
if girth2 < 0.0087 and pt1_dr01 > 5.35: z += 18.41 × (0.0087 − girth2) × (pt1_dr01 − 5.35)
if centroid_offset > 0.014 and pt_7 > 25.58: z += 1.32 × (centroid_offset − 0.014) × (pt_7 − 25.58)
if LHA > 0.313 and mass_top3 < 50.35: z += -0.296 × (LHA − 0.313) × (50.35 − mass_top3)
if LHA > 0.424: z += -56.34 × (LHA − 0.424)
if mass_over_sum_pt > 0.068 and dr_7 < 0.042: z += -5048 × (mass_over_sum_pt − 0.068) × (0.042 − dr_7)
if mass_over_sum_pt > 0.108 and z_7 < 0.068: z += 1003 × (mass_over_sum_pt − 0.108) × (0.068 − z_7)
if mass_over_sum_pt > 0.108 and dr_7 < 0.049: z += 12013 × (mass_over_sum_pt − 0.108) × (0.049 − dr_7)
if mass_over_sum_pt > 0.090 and max_dr < 0.145: z += 11014 × (mass_over_sum_pt − 0.090) × (0.145 − max_dr)
if width > 0.019 and z_3 > 0.090: z += -1860 × (width − 0.019) × (z_3 − 0.090)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (9 GeV), very narrow, pT spread over several particles** — 32.2% of jets, neuron 0.02. Mostly q (45%) with many gluons (35%), the largest group (32% of jets); mass 9.2 GeV, width 0.0003, 85% of the pT within 0.025 and almost nothing beyond 0.1. girth2 < 0.00868 (-15.065) and girth2 < 0.0132 (-6.012) always pass and outweigh the positive e2 < 0.0445 (+5.703), girth2 < 0.00437 (+4.665), lam1 < 0.00733 and e2 < 0.0385; mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 never passes. The neuron is essentially off (0.019, on for 3.2%) and changes no score; the formula calls them q.
- **medium-mass (55 GeV), average width, pT spread over several particles** — 16.1% of jets, neuron 0.24. Mostly W (46%) with Z (35%); mass 54.9 GeV, width 0.0061, 60% of the pT 0.05-0.1 from the axis and very little beyond 0.2. Both girth2 penalties still pass (girth2 < 0.00868 -4.591, girth2 < 0.0132 -3.310), and girth2 < 0.00437 never does; e2 < 0.0445, mass > 36.2 and eccentricity > 0.621 (+1.226), z_dr_0p1_0p2 < 0.468 and mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (82%) do not quite make up for them. The neuron stays low (0.237, on for 13.8%), only slightly lowering the W and Z scores; the formula calls them W.
- **light (45 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 0.20. Mostly W (46%) with Z (27%); mass 45.5 GeV, narrow (width 0.0044), with half of the pT 0.025-0.05 from the axis. girth2 < 0.00868 (-7.758) and girth2 < 0.0132 (-4.127) outweigh e2 < 0.0445 (+2.859), lam1 < 0.00733 and z_dr_0p1_0p2 < 0.468, while mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 passes only 9.1%. The neuron stays low (0.195, on for 11.8%) with a small drop in the W and Z scores; the formula calls them W.
- **medium-mass (59 GeV), wide, pT spread over several particles** — 11.2% of jets, neuron 1.26. Mostly Z (53%) with t (27%); mass 59.4 GeV, width 0.0087, pT mostly 0.05-0.1 from the axis with 20% at 0.1-0.15. girth2 < 0.00868 passes only 65% (-1.031 against about -9 elsewhere), so the main off-switch is weak; mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+2.020) and mass > 36.2 and eccentricity > 0.621 (+1.467) are partly cancelled by girth2 < 0.0132, lam1 > 0.00418 and e2 > 0.0285. The neuron sits at 1.26 and is on for 40%, which lowers the W (-0.63) and Z (-0.709) scores and slightly raises t; the formula calls them Z.
- **light (28 GeV), narrow, pT spread over several particles** — 11.0% of jets, neuron 0.21. A mixture: g 29%, W 26%, q 19%, Z 18%; mass 28.4 GeV, width 0.0023, 58% of the pT 0.025-0.05 from the axis. girth2 < 0.00868 (-11.612) and girth2 < 0.0132 (-5.121) outweigh e2 < 0.0445 (+4.241), girth2 < 0.00437, lam1 < 0.00733 and z_dr_0p1_0p2 < 0.468; neither mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 nor lam1 > 0.00418 ever passes. The neuron stays low (0.206, on for 14%); the formula calls them g.
- **medium-mass (68 GeV), very wide, pT spread over several particles, low pT** — 6.0% of jets, neuron 3.59. Mostly t (74%) with some gluons; mass 67.9 GeV, width 0.0153, pT spread 0.05-0.15 from the axis (43% and 33%). girth2 < 0.00868 never passes and girth2 < 0.0132 passes 25%, so the off-switch is gone; mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+5.269), mass > 36.2 and eccentricity > 0.621 and centroid_offset > 0.0144 outweigh lam1 > 0.00418 (-4.157), e2 > 0.0285 and mass_over_sum_pt > 0.0681. The neuron reaches 3.588 (on for 88%), which lowers the W (-1.794) and Z (-2.018) scores and raises t (+0.224); the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 4.9% of jets, neuron 3.19. Mostly t (84%); mass 80.7 GeV, width 0.0217, with 41% of the pT 0.1-0.15 and 19% 0.15-0.2 from the axis. Both girth2 penalties never pass; mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+8.834) and LHA > 0.326 (+2.205) beat lam1 > 0.00418 (-6.330), e2 > 0.0285 (-4.108), mass_over_sum_pt > 0.0681 and lam1 > 0.012. The value 3.189 (on for 90%) lowers the W and Z scores and raises t; the formula calls them t.
- **medium-mass (89 GeV), very wide, pT spread over several particles, low pT** — 3.7% of jets, neuron 3.92. Mostly t (85%); mass 88.8 GeV, width 0.0289, with much of the pT 0.1-0.2 from the axis. mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+11.754) and width > 0.0188 (+5.942) always pass and beat lam1 > 0.00418 (-8.853), e2 > 0.0285 (-5.672), lam1 > 0.012 and mass_over_sum_pt > 0.0681. The value 3.924 (on for 92%) lowers the W (-1.962) and Z (-2.207) scores and raises t (+0.245); the formula calls them t.
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 1.0% of jets, neuron 4.78. Mostly t (63%) with gluons (27%); mass 93.8 GeV, width 0.0386, the softest total pT (497 GeV), with pT reaching well beyond 0.15 from the axis. The same pair of large terms grows further: mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+14.905) and width > 0.0188 (+11.683) against lam1 > 0.00418 (-13.282), e2 > 0.0285 and lam1 > 0.012. The neuron's highest value, 4.783 (on for 93%), strongly lowers the W (-2.391) and Z (-2.69) scores and raises t; the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 0.2% of jets, neuron 2.28. Mostly t (93%), a rare group (0.24%); mass 81.4 GeV, width 0.0248, with a quarter of the pT within 0.025 but much of the rest 0.15-0.3 from the axis. Only here does mass_over_sum_pt > 0.108 and dr_7 < 0.0488 pass (+13.718; the softest particle sits on the axis), fighting mass_over_sum_pt > 0.0681 and dr_7 < 0.0422 (-8.409), lam1 > 0.00418 (-8.359) and e2 > 0.0285, with width > 0.0188 (83%) adding +3.644. The value 2.281 is on for 41% and then lowers the W and Z scores; the formula calls them t.

### neuron 4: two-prong substructure (low D2, τ21) (moderate)

- **What it measures:** Pushed up for small e2_sq (< 0.0172, its strongest term) but pushed down for small mass/pT squared (< 0.0171), so it needs a jet with real mass packed into a compact shape; it falls with D2 and τ21 and rises with the pT share at 0.05 ≤ ΔR < 0.1 (rank correlations -0.64, -0.618, 0.569). Z (5.15) and W (4.32) jets sit highest, tops in the middle (3.02), gluons (2.34) and quarks (1.22) lowest.
- *computed — its value:* largest for Z (5.15), then W (4.32), then t (3.02), then g (2.34), then q (1.22); it separates q jets from the rest best (AUC 0.26: small for q)
- **How the class scores use it:** It raises the Z score (+10%) and the t score (+12%), and lowers the q score (-18%) and the g score (-4%), since quarks and gluons sit lowest on it while bosons and tops have hard substructure. It does not enter the W score, although W jets sit second on it.
- *computed — used by:* raises the score of Z (+10%), t (+12%); lowers the score of g (-4%), q (-18%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

```
z = -12.38
if e2_sq < 0.017: z += 1998 × (0.017 − e2_sq)
if mass_over_sum_pt_sq < 0.017: z += -1504 × (0.017 − mass_over_sum_pt_sq)
if girth2 > 0.0036: z += 739 × (girth2 − 0.0036)
if e2 > 0.020: z += 180 × (e2 − 0.020)
if width > 0.00032: z += -364 × (width − 0.00032)
if girth2 > 0.0075: z += -902 × (girth2 − 0.0075)
if mass_over_sum_pt > 0.108: z += 387 × (mass_over_sum_pt − 0.108)
if e2_sq < 0.003: z += -1814 × (0.003 − e2_sq)
if width > 0.0017: z += 369 × (width − 0.0017)
if tau21 < 0.238: z += 34.10 × (0.238 − tau21)
if e2_sq < 0.024: z += 104 × (0.024 − e2_sq)
if width > 0.0026: z += 404 × (width − 0.0026)
if girth < 0.076: z += 61.73 × (0.076 − girth)
if girth < 0.125: z += 24.07 × (0.125 − girth)
if mass_over_sum_pt > 0.090: z += -189 × (mass_over_sum_pt − 0.090)
if mass_over_sum_pt > 0.108 and D2 < 3.89: z += -71.88 × (mass_over_sum_pt − 0.108) × (3.89 − D2)
if e2 > 0.0071: z += 50.27 × (e2 − 0.0071)
if C2 > 0.015: z += -61.79 × (C2 − 0.015)
if lam2 < 0.00031 and mass_top3 < 50.35: z += -109 × (0.00031 − lam2) × (50.35 − mass_top3)
if centroid_offset < 0.014 and z_dr_0p05_0p1 < 0.675: z += -387 × (0.014 − centroid_offset) × (0.675 − z_dr_0p05_0p1)
if mass < 80.40: z += 0.019 × (80.40 − mass)
if girth2_top2 < 0.0095: z += -121 × (0.0095 − girth2_top2)
if lam2 < 0.0011: z += 808 × (0.0011 − lam2)
if e2_sq < 0.0072: z += 203 × (0.0072 − e2_sq)
if max_dr > 0.103: z += 15.15 × (max_dr − 0.103)
if tau21 < 0.238 and lam2 < 0.0011: z += -11392 × (0.238 − tau21) × (0.0011 − lam2)
if girth2 > 0.013: z += -425 × (girth2 − 0.013)
if mass < 56.92: z += 0.028 × (56.92 − mass)
if girth < 0.048: z += -50.00 × (0.048 − girth)
if girth2_top3 < 0.005: z += 227 × (0.005 − girth2_top3)
if girth2_top2 < 0.0076: z += -114 × (0.0076 − girth2_top2)
if sum_pt < 764: z += 0.0047 × (764 − sum_pt)
if e2 > 0.041: z += -87.89 × (e2 − 0.041)
if C2 > 0.067: z += -118 × (C2 − 0.067)
if mass < 41.38: z += 0.027 × (41.38 − mass)
if centroid_offset < 0.014: z += 70.69 × (0.014 − centroid_offset)
if max_dr > 0.198: z += -23.75 × (max_dr − 0.198)
if mass < 69.61 and dr_3 < 0.104: z += 0.113 × (69.61 − mass) × (0.104 − dr_3)
if girth2_top2 < 0.0095 and centroid_offset > 0.016: z += 13073 × (0.0095 − girth2_top2) × (centroid_offset − 0.016)
if girth2_top2 < 0.0031: z += 189 × (0.0031 − girth2_top2)
if girth > 0.102: z += 44.68 × (girth − 0.102)
if mass_over_sum_pt_sq < 0.0011: z += -781 × (0.0011 − mass_over_sum_pt_sq)
if tau21 < 0.238 and mass < 62.55: z += -0.446 × (0.238 − tau21) × (62.55 − mass)
if e2 > 0.063: z += -105 × (e2 − 0.063)
if sum_pt < 764 and n_dr_0p2_0p4 < 1.00: z += -0.0022 × (764 − sum_pt) × (1.00 − n_dr_0p2_0p4)
if max_dr > 0.198 and z_7 > 0.028: z += 437 × (max_dr − 0.198) × (z_7 − 0.028)
if tau21 < 0.238 and dr_7 < 0.175: z += -26.98 × (0.238 − tau21) × (0.175 − dr_7)
if max_dr > 0.103 and eccentricity > 0.984: z += -679 × (max_dr − 0.103) × (eccentricity − 0.984)
if max_dr > 0.103 and pt_7 > 37.16: z += -1.37 × (max_dr − 0.103) × (pt_7 − 37.16)
if tau21 < 0.238 and pt_7 > 33.22: z += 0.240 × (0.238 − tau21) × (pt_7 − 33.22)
if C2 > 0.015 and pt_7 > 38.53: z += 2.90 × (C2 − 0.015) × (pt_7 − 38.53)
if tau21 < 0.238 and e2_sq > 0.012: z += -785 × (0.238 − tau21) × (e2_sq − 0.012)
if tau21 < 0.238 and planar_flow > 0.045: z += 23.32 × (0.238 − tau21) × (planar_flow − 0.045)
if girth2 > 0.019 and pt_6 > 62.25: z += -157 × (girth2 − 0.019) × (pt_6 − 62.25)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 36.4% of jets, neuron 1.26. Mostly q (42%) with many gluons (35%), the largest group (36% of jets); mass 10.1 GeV, width 0.0005, 78% of the pT within 0.025. The two giant terms nearly cancel: e2_sq < 0.0172 (+33.760) against mass_over_sum_pt_sq < 0.0171 (-25.381), with e2_sq < 0.00301 (-4.985) passing only in this group and girth < 0.0761, girth < 0.125 and e2_sq < 0.0238 adding a few units back; girth2 > 0.00356 and e2 > 0.0205 almost never pass. The value 1.261 (on for 76%) slightly lowers the q and g scores and raises Z and t; the formula calls them q.
- **medium-mass (56 GeV), average width, pT spread over several particles** — 14.7% of jets, neuron 6.17. Mostly W (55%) with Z (30%); mass 56.3 GeV, width 0.0057, 55% of the pT 0.05-0.1 from the axis. After e2_sq < 0.0172 (+23.451) and mass_over_sum_pt_sq < 0.0171 (-17.622) cancel down to about 6, tau21 < 0.238 always passes (+5.152) and e2 > 0.0205 (97%) adds more, while girth2 > 0.00752, mass_over_sum_pt > 0.108 and e2_sq < 0.00301 never pass. The value 6.175 (on for 99.6%) raises the t (+0.772) and Z (+0.482) scores and lowers q (-0.579) and g; the formula calls them W.
- **light (38 GeV), narrow, pT spread over several particles** — 13.9% of jets, neuron 2.81. Mostly W (36%) in a broad mixture with Z (23%), g (19%) and q (13%); mass 37.9 GeV, width 0.0033, 55% of the pT 0.025-0.05 from the axis. e2_sq < 0.0172 (+28.717) minus mass_over_sum_pt_sq < 0.0171 (-21.585) leaves a few units, girth and e2_sq < 0.0238 terms add a little, and tau21 < 0.238 passes for 58%; the bonuses girth2 > 0.00356 (39%) and e2 > 0.0205 (45%) are often missing. The value 2.805 (on for 89%) raises Z and t and lowers q and g; the formula calls them W.
- **medium-mass (61 GeV), average width, pT spread over several particles** — 10.7% of jets, neuron 8.04. Mostly Z (61%) with t (22%); mass 60.8 GeV, width 0.0083, 62% of the pT 0.05-0.1 from the axis. e2_sq < 0.0172 (+18.646) and mass_over_sum_pt_sq < 0.0171 (-14.005) net about 4.6, and tau21 < 0.238 (89%, +4.420), e2 > 0.0205 (+3.944) and girth2 > 0.00356 (+3.519) all add, with width > 0.000319 taking -2.913. This is the neuron's highest group (8.041, on for 98.7%): it raises the t (+1.005) and Z (+0.628) scores and lowers q (-0.754) and g; the formula calls them Z.
- **light (47 GeV), average width, pT spread over several particles** — 7.7% of jets, neuron 4.50. A W/Z mixture (W 35%, Z 32%) with t (14%); mass 46.7 GeV, width 0.006, lower total pT (633 GeV) and pT mostly 0.025-0.1 from the axis. e2_sq < 0.0172 (+23.335) and mass_over_sum_pt_sq < 0.0171 (-17.534) net about 6, e2 > 0.0205 and girth2 > 0.00356 add and width > 0.000319 subtracts, but tau21 < 0.238 passes only 43% and the top-like tests never pass. The value 4.5 (on for 95%) raises Z and t and lowers q and g; the formula calls them W.
- **medium-mass (76 GeV), very wide, pT spread over several particles, low pT** — 4.3% of jets, neuron 1.65. Mostly t (79%); mass 75.8 GeV, width 0.018, pT spread 0.05-0.15 from the axis. e2_sq < 0.0172 passes only 64%, so the usual big pair adds little; instead three pairs of large terms nearly cancel: girth2 > 0.00356 (+10.644) with girth2 > 0.00752 (-9.421), mass_over_sum_pt > 0.108 (+7.899) with mass_over_sum_pt > 0.0904 (-7.168), and e2 > 0.0205 (+7.319) with width > 0.000319 (-6.421). The result, 1.65, is on for half of them and nudges t and Z up; the formula calls them t.
- **medium-mass (64 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 4.11. Mostly t (69%) with gluons (20%); mass 64.0 GeV, width 0.0135, 44% of the pT 0.05-0.1 and 31% 0.1-0.15 from the axis. All the main tests pass: e2_sq < 0.0172 (+10.679) with mass_over_sum_pt_sq < 0.0171 (-8.008), girth2 > 0.00356 (+7.332) with girth2 > 0.00752 (-5.379), and e2 > 0.0205 (+5.243) with width > 0.000319 (-4.790), each pair leaving a positive rest. The value 4.105 (on for 78%) raises the t (+0.513) and Z scores and lowers q and g; the formula calls them t.
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 4.0% of jets, neuron 0.51. Mostly t (85%); mass 82.2 GeV, width 0.0235, with 35% of the pT 0.1-0.15 and 22% 0.15-0.2 from the axis. e2_sq < 0.0172 almost never passes, so the neuron is built only from cancelling pairs: mass_over_sum_pt > 0.108 (+15.476) and girth2 > 0.00356 (+14.748) against girth2 > 0.00752 (-14.431) and mass_over_sum_pt > 0.0904 (-10.864), and e2 > 0.0205 against width > 0.000319. The net is small (0.512, on for 25%); the formula calls them t.
- **medium-mass (90 GeV), very wide, pT spread over several particles, low pT** — 3.3% of jets, neuron 0.11. Mostly t (86%); mass 89.5 GeV, width 0.0298, with pT pushed far out (31% of it 0.15-0.2 from the axis). As in the previous group, e2_sq < 0.0172 never passes, and the large terms mass_over_sum_pt > 0.108 (+23.355), girth2 > 0.00356 and e2 > 0.0205 are cancelled by girth2 > 0.00752 (-20.126), mass_over_sum_pt > 0.0904 and mass_over_sum_pt > 0.108 and D2 < 3.89 (-12.940). The neuron stays near zero (0.11, on for 7.7%); the formula calls them t.
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.01. Mostly t (65%) with gluons (26%), a small group (0.83%); mass 94.3 GeV, width 0.0399, the widest jets, with most pT beyond 0.1 from the axis. The cancelling terms are at their largest: mass_over_sum_pt > 0.108 (+33.054), girth2 > 0.00356 and width > 0.00264 against girth2 > 0.00752 (-29.237), mass_over_sum_pt > 0.0904 and mass_over_sum_pt > 0.108 and D2 < 3.89 (-19.033). They leave the neuron off (0.014, on for 1.2%); the formula calls them t.

### neuron 8: narrow, centred single core (moderate)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00502, its strongest term) and narrow, centred ones (girth2 < 0.00668 with centroid offset < 0.0236), but pulled back for the very narrowest (girth < 0.0611); it falls with centroid offset, width and lam1 (rank correlations -0.534, -0.515, -0.51). Quarks (1.31) and gluons (1.01) sit highest, W (0.50) in the middle, Z (0.21) and tops (0.13) low.
- *computed — its value:* largest for q (1.31), then g (1.01), then W (0.50), then Z (0.21), then t (0.13); it separates q jets from the rest best (AUC 0.73: large for q)
- **How the class scores use it:** A single narrow core is evidence against a two-prong W, so it lowers the W score (-5%); it raises the q score a little (+2%), where quarks sit highest, and raises the t score slightly (+3%), a small correction since tops sit lowest on it. It does not enter the g or Z scores.
- *computed — used by:* raises the score of q (+2%), t (+3%); lowers the score of W (-5%); does not (or hardly) enter the score of g, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.617
if width < 0.005: z += 1877 × (0.005 − width)
if girth < 0.061 and width < 0.005: z += -26878 × (0.061 − girth) × (0.005 − width)
if girth2 < 0.0067 and centroid_offset < 0.024: z += 38550 × (0.0067 − girth2) × (0.024 − centroid_offset)
if width < 0.005 and z_dr_0p2_0p4 < 0.101: z += 6739 × (0.005 − width) × (0.101 − z_dr_0p2_0p4)
if girth2 < 0.0067: z += -310 × (0.0067 − girth2)
if mass < 29.64 and centroid_offset < 0.024: z += -7.16 × (29.64 − mass) × (0.024 − centroid_offset)
if girth < 0.061 and centroid_offset > 0.0068: z += -7483 × (0.061 − girth) × (centroid_offset − 0.0068)
if girth < 0.061: z += -31.96 × (0.061 − girth)
if girth < 0.061 and lam2 < 0.00019: z += 196083 × (0.061 − girth) × (0.00019 − lam2)
if girth < 0.061 and log_sum_pt > 6.67: z += 337 × (0.061 − girth) × (log_sum_pt − 6.67)
if mass < 21.78: z += -0.112 × (21.78 − mass)
if mass < 21.78 and centroid_offset > 0.031: z += -110 × (21.78 − mass) × (centroid_offset − 0.031)
if LHA < 0.197 and lam2 < 0.00031: z += -66155 × (0.197 − LHA) × (0.00031 − lam2)
if LHA < 0.197: z += -17.91 × (0.197 − LHA)
if LHA < 0.197 and width < 0.00056: z += 46513 × (0.197 − LHA) × (0.00056 − width)
if C2 < 0.027: z += -50.05 × (0.027 − C2)
if width < 0.005 and centroid_offset > 0.0068: z += 39202 × (0.005 − width) × (centroid_offset − 0.0068)
if lam2 < 0.00019: z += 3533 × (0.00019 − lam2)
if C2 < 0.027 and width < 0.0044: z += -17302 × (0.027 − C2) × (0.0044 − width)
if max_dr < 0.177: z += 5.61 × (0.177 − max_dr)
if girth2 < 0.00096: z += 1865 × (0.00096 − girth2)
if max_dr < 0.177 and lam2 < 0.00019: z += 36010 × (0.177 − max_dr) × (0.00019 − lam2)
if mass < 21.78 and lam2 < 0.00019: z += -464 × (21.78 − mass) × (0.00019 − lam2)
if log_sum_pt > 6.70 and z_dr_0p2_0p4 < 0.206: z += -43.22 × (log_sum_pt − 6.70) × (0.206 − z_dr_0p2_0p4)
if girth2 < 0.0067 and centroid_offset > 0.018: z += -49875 × (0.0067 − girth2) × (centroid_offset − 0.018)
if sum_pt_top5 > 658 and girth2 < 0.0017: z += -7.20 × (sum_pt_top5 − 658) × (0.0017 − girth2)
if e2 < 0.025 and centroid_offset < 0.031: z += 1904 × (0.025 − e2) × (0.031 − centroid_offset)
if girth2_top3 < 0.00082: z += -1364 × (0.00082 − girth2_top3)
if C2 < 0.027 and width < 0.00096: z += 76634 × (0.027 − C2) × (0.00096 − width)
if width < 0.005 and mass_over_sum_pt_sq > 0.00012: z += -294634 × (0.005 − width) × (mass_over_sum_pt_sq − 0.00012)
if sum_pt_top5 > 658: z += 0.0058 × (sum_pt_top5 − 658)
if girth2 < 0.0067 and planar_flow < 0.401: z += 619 × (0.0067 − girth2) × (0.401 − planar_flow)
if C2 < 0.027 and width < 0.0061: z += -7113 × (0.027 − C2) × (0.0061 − width)
if mass < 15.45 and width < 0.00096: z += -141 × (15.45 − mass) × (0.00096 − width)
if width < 0.005 and pt_7 < 48.72: z += -7.48 × (0.005 − width) × (48.72 − pt_7)
if girth < 0.061 and z_dr_0p2_0p4 < 0.056: z += -226 × (0.061 − girth) × (0.056 − z_dr_0p2_0p4)
if mass < 21.78 and centroid_offset < 0.027: z += 2.96 × (21.78 − mass) × (0.027 − centroid_offset)
if girth < 0.061 and lam1 > 0.00028: z += -21717 × (0.061 − girth) × (lam1 − 0.00028)
if girth2_top5 < 0.00022: z += -6695 × (0.00022 − girth2_top5)
if log_sum_pt > 6.70 and girth2 < 0.019: z += -332 × (log_sum_pt − 6.70) × (0.019 − girth2)
if e2 < 0.017 and mass_top5 < 9.26: z += 9.78 × (0.017 − e2) × (9.26 − mass_top5)
if girth2 < 0.0067 and log_sum_pt > 6.67: z += -894 × (0.0067 − girth2) × (log_sum_pt − 6.67)
if z_dr_0_0p05 > 0.848 and lam2 < 0.00054: z += -6534 × (z_dr_0_0p05 − 0.848) × (0.00054 − lam2)
if mass < 21.78 and centroid_offset > 0.016: z += 12.13 × (21.78 − mass) × (centroid_offset − 0.016)
if C2 < 0.027 and centroid_offset > 0.016: z += -6010 × (0.027 − C2) × (centroid_offset − 0.016)
if log_sum_pt > 6.70 and pt_7 < 48.72: z += 0.185 × (log_sum_pt − 6.70) × (48.72 − pt_7)
if width < 0.00032: z += 3517 × (0.00032 − width)
if log_sum_pt > 6.70 and width < 0.0067: z += -828 × (log_sum_pt − 6.70) × (0.0067 − width)
if e2 < 0.025 and eccentricity > 0.873: z += -429 × (0.025 − e2) × (eccentricity − 0.873)
if mass < 15.45 and z_7 < 0.059: z += 2.58 × (15.45 − mass) × (0.059 − z_7)
if log_sum_pt > 6.70 and mass_over_sum_pt_sq < 0.00024: z += 28792 × (log_sum_pt − 6.70) × (0.00024 − mass_over_sum_pt_sq)
if LHA < 0.197 and girth2 > 0.0067: z += 480093 × (0.197 − LHA) × (girth2 − 0.0067)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (62 GeV), wide, pT spread over several particles** — 52.7% of jets, neuron 0.14. Half of all jets (53%), a mixture of t (35%), Z (27%) and W (23%) with some g; mass 61.7 GeV, width 0.0112, with pT spread 0.05-0.15 from the axis. The narrow-core tests mostly fail: width < 0.00502 passes for only 14% and girth2 < 0.00668 and centroid_offset < 0.0236 for 36%, so the neuron's main positive terms are nearly absent; small pieces such as lam2 < 0.000195 and max_dr < 0.177 (61% each) and C2 < 0.027 roughly cancel. The neuron stays low (0.139, on for 24%) and barely moves the scores; the formula calls them t.
- **light (37 GeV), narrow, pT spread over several particles** — 11.4% of jets, neuron 1.06. Mostly W (37%) mixed with g (20%), Z (21%) and q (14%); mass 36.7 GeV, width 0.0029, 59% of the pT 0.025-0.05 from the axis. width < 0.00502 passes for all (+4.038), with width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+1.367) and girth2 < 0.00668 and centroid_offset < 0.0236 (78%), but width < 0.00502 and mass_over_sum_pt_sq > 0.000123 (-1.368), girth < 0.0611 and centroid_offset > 0.00679 (89%), girth2 < 0.00668, girth < 0.0611 and width < 0.00502 and girth < 0.0611 and lam1 > 0.000276 take most of it back. The value 1.062 (on for 50%) lowers the W score (-0.265) and raises t (+0.199) and q; the formula calls them W.
- **very light (7 GeV), very narrow, pT spread over several particles** — 11.0% of jets, neuron 1.29. Mostly q (49%) with gluons (39%); mass 6.5 GeV, width 0.0001, 98% of the pT within 0.025. width < 0.00502 (+9.202), girth2 < 0.00668 and centroid_offset < 0.0236 (+4.763), width < 0.00502 and z_dr_0p2_0p4 < 0.101 and LHA < 0.197 and width < 0.000561 (+2.054) outweigh the very-narrow brakes girth < 0.0611 and width < 0.00502 (-6.887) and mass < 29.6 and centroid_offset < 0.0236 (-3.106). The value 1.295 (on for 90%) lowers the W score (-0.324) and raises t (+0.243) and q; the formula calls them q.
- **very light (16 GeV), very narrow, pT spread over several particles** — 9.9% of jets, neuron 2.15. Mostly g (44%) with quarks (36%); mass 15.8 GeV, width 0.0007, 77% of the pT within 0.025. width < 0.00502 (+8.157), girth2 < 0.00668 and centroid_offset < 0.0236 (+3.185) and width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+2.946) all pass, while the brakes for the very narrowest are weaker than in the pencil-thin groups (girth < 0.0611 and width < 0.00502 -4.741, girth2 < 0.00668, girth < 0.0611). This gives the neuron's highest group value, 2.148 (on for 75%), which lowers the W score (-0.537) and raises t (+0.403) and q; the formula calls them g.
- **very light (8 GeV), very narrow, leading particle 45% of pT, high pT** — 7.5% of jets, neuron 0.95. Mostly q (67%) with gluons (19%); mass 8.3 GeV, width 0.0001, total pT 1004 GeV with a 452.44 GeV leading particle and 98% of the pT within 0.025. As for the other pencil-thin jets, width < 0.00502 (+9.231) and girth2 < 0.00668 and centroid_offset < 0.0236 (+5.260) fight girth < 0.0611 and width < 0.00502 (-7.232) and mass < 29.6 and centroid_offset < 0.0236; the high pT adds girth < 0.0611 and log_sum_pt > 6.67 (+4.349) but also sum_pt_top5 > 658 and girth2 < 0.00165 (-2.754). The value 0.954 (on for 89%) lowers W and raises t and q; the formula calls them q.
- **very light (9 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 0.10. A mixture: g 31%, W 26%, Z 23%, q 15%; mass 9.3 GeV, width 0.0009, with 55% of the pT 0.025-0.05 from the axis and 42% within 0.025. width < 0.00502 (+7.697), width < 0.00502 and centroid_offset > 0.00679 (+2.889) and width < 0.00502 and z_dr_0p2_0p4 < 0.101 are cancelled by the off-centre brakes girth < 0.0611 and centroid_offset > 0.00679 (-4.333), girth < 0.0611 and width < 0.00502 and girth2 < 0.00668 and centroid_offset > 0.0184 (96%). The neuron stays low (0.098, on for 8.8%); the formula calls them g.
- **very light (8 GeV), narrow, pT spread over several particles** — 1.1% of jets, neuron 0.00. A mixture: g 37%, Z 29%, t 13%, W 12%; mass 8.2 GeV, width 0.002, with 76% of the pT 0.025-0.05 from the axis and only 4.3% within 0.025. mass < 21.8 and centroid_offset > 0.0312 passes only in these off-centre groups and subtracts -12.126, with girth2 < 0.00668 and centroid_offset > 0.0184 and girth < 0.0611 and centroid_offset > 0.00679 adding about -9.4, more than width < 0.00502, width < 0.00502 and centroid_offset > 0.00679 and mass < 21.8 and centroid_offset > 0.0163 (+3.797) return. The neuron is off for all of them; the formula calls them g.
- **very light (7 GeV), narrow, pT spread over several particles, low pT** — 0.5% of jets, neuron 0.00. Mostly g (48%) with t (22%), q and Z; mass 6.8 GeV, width 0.0031, with almost no pT within 0.025 and nearly all of it 0.025-0.1 from the axis. mass < 21.8 and centroid_offset > 0.0312 subtracts -32.337, far more than mass < 21.8 and centroid_offset > 0.0163 (+6.271) and width < 0.00502 add, with girth2 < 0.00668 and centroid_offset > 0.0184 and C2 < 0.027 and centroid_offset > 0.0163 subtracting more. The neuron is off; the formula calls them g.
- **very light (6 GeV), average width, pT spread over several particles, low pT** — 0.2% of jets, neuron 0.00. Mostly g (55%) with t (28%), a rare group (0.18%); mass 6.2 GeV, width 0.0051, with 93% of the pT 0.05-0.1 from the axis and none within 0.025. mass < 21.8 and centroid_offset > 0.0312 subtracts -63.597, against only mass < 21.8 and centroid_offset > 0.0163 (+9.817); width < 0.00502 passes for just 55%. The neuron is off; the formula calls them g.
- **very light (5 GeV), wide, pT spread over several particles, low pT** — 0.0% of jets, neuron 0.00. Mostly t (62%), only 0.04% of jets; mass 5.1 GeV but width 0.0106, with all the pT 0.05-0.15 from the axis, as if one tight prong sits away from the jet axis. mass < 21.8 and centroid_offset > 0.0312 subtracts -124.046 on average, dwarfing mass < 21.8 and centroid_offset > 0.0163 (+16.676), and width < 0.00502 never passes. The neuron is off; the formula calls them t.

### neuron 15: slightly wider than a W (moderate)

- **What it measures:** Pushed up for width < 0.0132 but down for narrow jets (width < 0.00668) and for light jets (mass < 36.2 GeV, its strongest term), so it responds to massive jets just wider than the typical W but not broad; it follows the number of particles at 0.2 ≤ ΔR < 0.4 and above 10 GeV (rank correlations 0.471 and 0.43). Z jets sit highest (0.88), then tops (0.41), with gluons (0.17), quarks (0.12) and W (0.06) low.
- *computed — its value:* largest for Z (0.88), then t (0.41), then g (0.17), then q (0.12), then W (0.06); it separates Z jets from the rest best (AUC 0.72: large for Z)
- **How the class scores use it:** W jets sit lowest on it, so the W score reads it as evidence against a W: it lowers the W score (-7%). It also lowers the Z score, but only slightly (-2%), even though Z jets sit highest; it does not enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-7%), Z (-2%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -2.58
if mass < 36.23: z += -0.469 × (36.23 − mass)
if width < 0.013: z += 557 × (0.013 − width)
if width < 0.0067: z += -1011 × (0.0067 − width)
if e2_sq < 0.012: z += -378 × (0.012 − e2_sq)
if e2 < 0.025: z += 343 × (0.025 − e2)
if mass_over_sum_pt < 0.068: z += -86.37 × (0.068 − mass_over_sum_pt)
if e2_sq < 0.0082: z += -436 × (0.0082 − e2_sq)
if lam1 < 0.012: z += -226 × (0.012 − lam1)
if lam1 < 0.0065: z += 566 × (0.0065 − lam1)
if e2 < 0.041: z += 89.72 × (0.041 − e2)
if mass_over_sum_pt_sq < 0.0072: z += 425 × (0.0072 − mass_over_sum_pt_sq)
if width < 0.0075: z += -381 × (0.0075 − width)
if girth > 0.034: z += 42.51 × (girth − 0.034)
if lam1 < 0.0084: z += -258 × (0.0084 − lam1)
if girth2 < 0.019: z += 80.10 × (0.019 − girth2)
if z_dr_0p1_0p2 < 0.328: z += 3.62 × (0.328 − z_dr_0p1_0p2)
if e2 < 0.063: z += 21.82 × (0.063 − e2)
if mass > 80.40: z += -0.498 × (mass − 80.40)
if e2 < 0.038: z += -38.28 × (0.038 − e2)
if z_dr_0p05_0p1 > 0.751: z += -21.09 × (z_dr_0p05_0p1 − 0.751)
if girth2_top3 < 0.0022: z += -530 × (0.0022 − girth2_top3)
if tau21 < 0.238 and z_dr_0p2_0p4 < 0.206: z += 36.85 × (0.238 − tau21) × (0.206 − z_dr_0p2_0p4)
if LHA > 0.347: z += -42.45 × (LHA − 0.347)
if girth2_top2 < 0.004: z += -193 × (0.004 − girth2_top2)
if e2 < 0.032: z += 27.80 × (0.032 − e2)
if z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 2.00: z += 7.62 × (z_dr_0p05_0p1 − 0.751) × (2.00 − n_dr_0p2_0p4)
if girth2_top2 < 0.0076: z += 66.38 × (0.0076 − girth2_top2)
if LHA > 0.177 and sum_pt_top3 > 353: z += 0.045 × (LHA − 0.177) × (sum_pt_top3 − 353)
if width < 0.0075 and e2 > 0.025: z += -68520 × (0.0075 − width) × (e2 − 0.025)
if D2 < 0.746: z += -2.33 × (0.746 − D2)
if tau21 < 0.238: z += 3.90 × (0.238 − tau21)
if tau21 < 0.238 and girth2_top5 > 0.0083: z += -1687 × (0.238 − tau21) × (girth2_top5 − 0.0083)
if planar_flow < 0.084: z += -9.23 × (0.084 − planar_flow)
if mass > 80.40 and z_dr_0p2_0p4 < 0.206: z += 2.07 × (mass − 80.40) × (0.206 − z_dr_0p2_0p4)
if girth2_top2 < 0.0076 and mean_phi < 0.0016: z += 7423 × (0.0076 − girth2_top2) × (0.0016 − mean_phi)
if tau21 < 0.238 and z_dr_0p05_0p1 < 0.588: z += -12.53 × (0.238 − tau21) × (0.588 − z_dr_0p05_0p1)
if width < 0.0061 and log_sum_pt > 6.90: z += -7484 × (0.0061 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 32.08 × (log_sum_pt − 6.90)
if LHA > 0.177 and z_top5 > 0.865: z += -212 × (LHA − 0.177) × (z_top5 − 0.865)
if log_sum_pt > 6.90 and mean_phi > 0.017: z += 48039 × (log_sum_pt − 6.90) × (mean_phi − 0.017)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (7 GeV), very narrow, pT spread over several particles** — 26.6% of jets, neuron 0.00. Mostly q (44%) with gluons (34%), 27% of all jets; mass 6.8 GeV, width 0.0004, 84% of the pT within 0.025. mass < 36.2 subtracts -13.798 and width < 0.00668, mass_over_sum_pt < 0.0681 and e2_sq < 0.0117 add further penalties, far more than width < 0.0132 (+7.179) and e2 < 0.0245 (+6.856) return. The neuron is off for all of them and adds nothing; the formula calls them q.
- **medium-mass (56 GeV), average width, pT spread over several particles** — 23.0% of jets, neuron 0.89. Mostly Z (45%) with W (29%) and t (14%); mass 56.0 GeV, width 0.0066, with pT spread 0.025-0.15 from the axis. mass < 36.2 passes for only 5.4% and width < 0.00668 for 56%, so the main penalties are mostly avoided; width < 0.0132 (+3.674), girth > 0.0336 (+1.607) and girth2 < 0.0188 add, against e2_sq < 0.0117 (-2.012), lam1 < 0.012 and e2_sq < 0.00817 (90%). The neuron's highest group value, 0.892 (on for 56%), lowers the W score (-0.614) and slightly the Z score (-0.139); the formula calls them Z.
- **light (41 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 0.22. Mostly W (39%) with Z (25%), g and q; mass 41.0 GeV, width 0.0033, 58% of the pT 0.025-0.05 from the axis. width < 0.0132 (+5.532) and e2 < 0.0245 (78%) are outweighed by the narrowness penalties width < 0.00668 (-3.401), e2_sq < 0.0117 (-3.290), e2_sq < 0.00817, lam1 < 0.012 and mass_over_sum_pt < 0.0681, with mass < 36.2 passing for 37%. The neuron stays low (0.216, on for 21%); the formula calls them W.
- **medium-mass (67 GeV), very wide, pT spread over several particles, low pT** — 12.3% of jets, neuron 0.39. Mostly t (78%) with gluons (16%); mass 66.5 GeV, width 0.0194, low total pT (520 GeV), with pT spread 0.05-0.2 from the axis. width < 0.0132 passes for only 21%, so the compact bonus is mostly missing, but so are the light and narrow penalties; girth > 0.0336 (+3.891) is set against LHA > 0.347 (80.5%, -1.800) and tau21 < 0.238 and girth2_top5 > 0.00833 (50%). The value 0.39 (on for 34%) lowers the W score a little; the formula calls them t.
- **very light (19 GeV), very narrow, pT spread over several particles** — 9.9% of jets, neuron 0.00. Mostly g (41%) with quarks (33%); mass 19.0 GeV, width 0.0013, half of the pT within 0.025 and 37% 0.025-0.05. mass < 36.2 (-8.076), width < 0.00668 (-5.451), e2_sq < 0.0117 and mass_over_sum_pt < 0.0681 outweigh width < 0.0132 (+6.656) and e2 < 0.0245 (+3.959). The neuron is off (0.002) and adds nothing; the formula calls them g.
- **medium-mass (53 GeV), average width, pT spread over several particles** — 8.6% of jets, neuron 0.48. Mostly W (49%) with Z (30%); mass 53.1 GeV, width 0.0062, with 96% of the pT 0.05-0.1 from the axis (against 27% for all jets). Only this group always passes z_dr_0p05_0p1 > 0.751 (-4.481), mostly returned by z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 2 (+3.183); width < 0.0132 (+3.929) and girth > 0.0336 then face e2_sq < 0.0117, lam1 < 0.012 and width < 0.00668 (71%). The value 0.479 (on for 30%) lowers the W score (-0.33); the formula calls them W.
- **heavy (94 GeV), very wide, pT spread over several particles** — 3.3% of jets, neuron 0.10. Mostly t (82%); mass 93.6 GeV, width 0.0241, with pT spread 0.05-0.2 from the axis. mass > 80.4 (-6.580), LHA > 0.347 (88%) and tau21 < 0.238 and girth2_top5 > 0.00833 (44%) outweigh girth > 0.0336 (+4.556) and mass > 80.4 and z_dr_0p2_0p4 < 0.206 (61%, +1.981); width < 0.0132 passes for only 6.6%. The neuron is nearly off (0.104, on for 5.6%); the formula calls them t.
- **heavy (110 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 0.02. Mostly t (79%) with some gluons; mass 110.2 GeV, width 0.0271, with pT spread 0.05-0.2 from the axis. mass > 80.4 subtracts -14.837 and LHA > 0.347 -3.264, more than girth > 0.0336 (+4.950) and mass > 80.4 and z_dr_0p2_0p4 < 0.206 (48%) add. The neuron is essentially off (on for 0.8%); the formula calls them t.
- **very light (12 GeV), very narrow, leading particle 45% of pT, high pT** — 0.9% of jets, neuron 0.05. Mostly g (51%) with quarks (38%); mass 12.5 GeV, width 0.0002, total pT 1236 GeV with a 557.82 GeV leading particle and 94% of the pT within 0.025. mass < 36.2 (95%, -11.555), width < 0.0061 and log_sum_pt > 6.9 (-9.527) and width < 0.00668 (-6.537) outweigh width < 0.0132 (+7.261), e2 < 0.0245 (+7.016) and log_sum_pt > 6.9 (+6.994). The neuron is essentially off (on for 1.5%); the formula calls them g.
- **heavy (137 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.18. A gluon-top mixture (g 51%, t 34%), a rare group (0.31%); the heaviest jets here (136.7 GeV), width 0.0263, total pT 880 GeV, with pT spread 0.1-0.2 from the axis. mass > 80.4 subtracts -28.029, only partly returned by mass > 80.4 and z_dr_0p2_0p4 < 0.206 (65%, +10.195), girth > 0.0336 and LHA > 0.177 and sum_pt_top3 > 353, while LHA > 0.347 and tau21 < 0.238 and girth2_top5 > 0.00833 subtract more. The neuron is nearly off (0.183, on for 5.5%); the formula calls them t.

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs girth2 > 0.0188 (pushes up) and is pushed down strongly for e2 > 0.0634; it follows the number of particles above 10 GeV (rank correlation 0.829). Only tops reach it with any frequency (non-zero for 18.8% of them), so tops sit highest (0.22), then gluons (0.07) and quarks (0.02), with W and Z at 0.00.
- *computed — its value:* largest for t (0.22), then g (0.07), then q (0.02), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.59: large for t)
- **How the class scores use it:** It does not enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.924
if girth2 > 0.019: z += 259 × (girth2 − 0.019)
if e2 > 0.063: z += -82.97 × (e2 − 0.063)
if girth2 > 0.019 and pt_7 > 15.55: z += 4.82 × (girth2 − 0.019) × (pt_7 − 15.55)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **light (36 GeV), narrow, pT spread over several particles** — 91.1% of jets, neuron 0.00. Nearly all jets (91%), an even mixture of every class (W 22%, Z 22%, g 21%, q 21%, t 14%); mass 35.9 GeV and width 0.0044, close to the averages. Neither girth2 > 0.0188 (1.1%) nor e2 > 0.0634 (1.4%) nor girth2 > 0.0188 and pt_7 > 15.6 passes for these jets, so nothing is added. The neuron is off for all of them and changes no score; the formula calls them W.
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 1.7% of jets, neuron 0.08. Mostly t (89%); mass 83.3 GeV, width 0.0242, with 34% of the pT 0.1-0.15 and 24% 0.15-0.2 from the axis. All three tests pass: girth2 > 0.0188 (+1.389) and girth2 > 0.0188 and pt_7 > 15.6 (+0.493) against e2 > 0.0634 (-1.321), which nearly cancel. The neuron is small (0.082, on for 21%) and lowers the t score only slightly (-0.041); the formula calls them t.
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 1.5% of jets, neuron 0.26. Mostly t (91%); mass 87.2 GeV, width 0.0273, with about 30% of the pT at each of 0.1-0.15 and 0.15-0.2 from the axis. girth2 > 0.0188 (+2.196) and girth2 > 0.0188 and pt_7 > 15.6 (+0.790) slightly outweigh e2 > 0.0634 (-1.965). The value 0.259 is on for about half and lowers the t score (-0.13); the formula calls them t.
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.00. Mostly t (81%) with gluons (14%); mass 77.4 GeV, width 0.0202, with 41% of the pT 0.1-0.15 from the axis. e2 > 0.0634 passes for all of them (-0.778) while girth2 > 0.0188 passes for 85% and adds only +0.402, so the penalty wins. The neuron is off and changes no score; the formula calls them t.
- **heavy (90 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.94. Mostly t (86%); mass 90.4 GeV, width 0.0308, with 34% of the pT 0.15-0.2 from the axis. girth2 > 0.0188 (+3.102) and girth2 > 0.0188 and pt_7 > 15.6 (+1.156) clearly beat e2 > 0.0634 (-2.411). The value 0.942 (on for 89%) lowers the t score (-0.471); the formula calls them t.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 1.2% of jets, neuron 0.33. Mostly t (80%) with gluons (14%); mass 80.0 GeV, width 0.023, with 43% of the pT 0.1-0.15 from the axis. girth2 > 0.0188 (+1.087) and girth2 > 0.0188 and pt_7 > 15.6 always pass, while e2 > 0.0634 passes for only 63% (-0.277), which sets this group apart. The value 0.333 (on for 57%) lowers the t score a little (-0.167); the formula calls them t.
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 1.92. Mostly t (78%) with gluons (17%); mass 90.6 GeV, width 0.0352, softer total pT (497 GeV) and much pT beyond 0.15 from the axis. girth2 > 0.0188 (+4.235) and girth2 > 0.0188 and pt_7 > 15.6 (+1.463) beat e2 > 0.0634 (-2.853). The value 1.92 (on for 99.6%) lowers the t score (-0.96); the formula calls them t.
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 0.5% of jets, neuron 1.75. Mostly t (66%) with gluons (25%); mass 84.3 GeV, width 0.0288, with about a third of the pT at each of 0.1-0.15 and 0.15-0.2 from the axis. girth2 > 0.0188 (+2.595) and girth2 > 0.0188 and pt_7 > 15.6 (+0.965) pass for all, while e2 > 0.0634 passes for 91% and takes only -0.886. The value 1.747 (on for all) lowers the t score (-0.874); the formula calls them t.
- **heavy (96 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 3.64. Mostly t (59%) with many gluons (30%), a rare group (0.31%); mass 95.6 GeV, width 0.0417, soft total pT (487 GeV). girth2 > 0.0188 (+5.922) and girth2 > 0.0188 and pt_7 > 15.6 (+2.036) outweigh e2 > 0.0634 (-3.397). The value 3.638 (on for all) lowers the t score (-1.819), a brake on the top score for these broadest jets; the formula calls them t.
- **heavy (93 GeV), very wide, pT spread over several particles, low pT** — 0.1% of jets, neuron 5.80. Mostly t (60%) with gluons (28%), only 0.09% of jets; mass 93.2 GeV, the widest jets (width 0.0519), the lowest total pT (426 GeV) and pT shared unusually evenly (leader 91.3 GeV, 4th 52.06 GeV). girth2 > 0.0188 (+8.575) and girth2 > 0.0188 and pt_7 > 15.6 (+2.863) far outweigh e2 > 0.0634 (-4.718). The neuron's highest value, 5.797, lowers the t score by -2.899; the formula still calls them t.
