# What each part of the 641-term formula does (8 particles)

*the tuned formula (start)*. Validation accuracy 65.56%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a clipped sum of if-statements on jet-shape quantities), and each class score adds some scales and subtracts others. Tops are recognised by size: the t score subtracts compactness (neuron 13, on which tops sit far below every other type) and adds overall size (neuron 10: e2, width, mass), so it closely follows jet width and girth. Quarks and gluons both sit high on the narrow-and-light scale (neuron 9), which feeds both of their scores; they are told apart by how the pT is shared among the particles: the g score adds gluon-likeness (neuron 2: even the 8th-hardest particle is still hard) and subtracts quark-likeness (neuron 5: pT concentrated in the few hardest particles), while the q score subtracts overall size (neuron 10) and the low-C2 scale (neuron 4). W and Z jets sit high on scales for a compact, centred, elongated two-prong jet (neurons 7 and 13 feed both boson scores, neurons 0 and 11 only the W score), and both boson scores subtract the broad-off-centre and wide-angle-radiation scales (neurons 6 and 3), on which tops and gluons sit much higher. W and Z are split mainly by width: Z jets sit highest on an intermediate-width scale (neuron 14), which the Z score adds and the W score subtracts, and the 'slightly wider than a W' scale (neuron 15) takes a larger share out of the W score (-11%) than out of the Z score (-4%), consistent with the heavier Z opening wider at the same pT.

## The 5 class scores

### score g: gluon: many hard particles, light jet

High for light, fairly narrow jets whose pT is spread over many hard particles (it rises with z_7 and τ21 and falls with eccentricity); it averages 2.24 for g jets, 1.39 for q and 1.09 for t, and is near zero for W and Z (AUC 0.81 for gluons against the rest).

Adds gluon-likeness (neuron 2, +35%) and narrow-light-ness (neuron 9, +26%), plus smaller amounts of neurons 1 and 6; subtracts quark-likeness (neuron 5, -16%) and, less, the compact two-prong scale (neuron 0, -6%) and neuron 4 (-4%).

*computed:* largest for g (2.24), then q (1.39), then t (1.09), then W (0.12), then Z (-0.03); it separates g jets from the rest best (AUC 0.81: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets (it falls with mass, lam1 and girth); it averages 2.26 for q jets and 1.51 for g, with tops lower (0.26) and W and Z near or below zero (AUC 0.84 for quarks). Gluons are its main confusion.

Mostly adds narrow-light-ness (neuron 9, +49%); subtracts overall size (neuron 10, -18%) and the low-C2 scale (neuron 4, -16%); adds a little of broad spread (neuron 6, +9%) and of quark-likeness (neuron 5, +5%).

*computed:* largest for q (2.26), then g (1.51), then t (0.26), then W (0.00), then Z (-0.16); it separates q jets from the rest best (AUC 0.84: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets that are not wider than a W; it averages 2.56 for W jets and 0.47 for Z, is near zero for q, negative for g and strongly negative for t (-2.63) (AUC 0.90).

Adds the compact-centred scale (neuron 11, +24%, its largest input), compactness (neuron 13) and the two-prong scales (neurons 0 and 7); subtracts broad off-centre spread (neuron 6, -13%), the 'wider than a W' scales (neurons 15 and 14), wide-angle radiation (neuron 3) and a little of neurons 8 and 9.

*computed:* largest for W (2.56), then Z (0.47), then q (-0.03), then g (-0.46), then t (-2.63); it separates W jets from the rest best (AUC 0.90: large for W)

### score Z: Z: two-prong jet, wider than a W

High for compact two-prong jets of W/Z size with an intermediate, Z-sized width; it averages 2.34 for Z jets and 1.57 for W (W is its main confusion), is near zero for q and g, and strongly negative for t (-1.99) (AUC 0.85).

Adds the two-prong W/Z scale (neuron 7, +27%, its largest input), neuron 4, compactness (neuron 13), the intermediate-width scale (neuron 14) and a little of neuron 1; subtracts broad off-centre spread (neuron 6, -21%), wide-angle radiation (neuron 3, -12%) and a little of neurons 9 and 15.

*computed:* largest for Z (2.34), then W (1.57), then q (0.10), then g (-0.22), then t (-1.99); it separates Z jets from the rest best (AUC 0.85: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and e2 with rank correlations of about 0.89-0.90. It averages 2.99 for t jets, 0.32 for Z and 0.15 for W, is near zero for g and negative for q (-1.45) (AUC 0.91).

Subtracts compactness (neuron 13, -46%) and adds overall size (neuron 10, +26%); also subtracts quark-likeness (neuron 5, -13%) and adds smaller amounts of neurons 4 (+10%) and 8 (+3%).

*computed:* largest for t (2.99), then Z (0.32), then W (0.15), then g (-0.01), then q (-1.45); it separates t jets from the rest best (AUC 0.91: large for t)

## The 16 neurons (most important first)

### neuron 0: compact, massive, elongated two-prong jet (major)

- **What it measures:** Pushed up for compact jets (girth2 < 0.0132 and width < 0.00868 are its two strongest terms) and pushed down for light jets (mass < 21.8 GeV and < 29.6 GeV), for small mass/pT (< 0.131) and for the very narrowest jets (girth < 0.0761); it rises with eccentricity and mass and falls with planar flow and τ21 (rank correlations 0.615, 0.476, -0.615, -0.61), so it measures a compact, elongated two-prong jet with real mass. Z (2.31) and W (1.96) jets sit highest, tops in the middle (0.66), gluons (0.27) and quarks (0.20) lowest.
- *computed — its value:* largest for Z (2.31), then W (1.96), then t (0.66), then g (0.27), then q (0.20); it separates Z jets from the rest best (AUC 0.76: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, it raises the W score (+9%) and lowers the g score (-6%). It does not enter the q, Z or t scores; the Z score, although Z jets sit highest on it, relies on the related neuron 7 instead.
- *computed — used by:* raises the score of W (+9%); lowers the score of g (-6%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.718):

- `` — 20.0% of jets, value 3.37 (2.00…4.75), formula right 78%
- `` — 8.3% of jets, value 1.45 (0.00…3.00), formula right 55%
- `` — 16.0% of jets, value 1.43 (0.00…2.62), formula right 58%
- `` — 2.0% of jets, value 1.17 (0.00…2.50), formula right 72%
- `` — 3.4% of jets, value 0.40 (0.00…1.38), formula right 68%
- `` — 2.4% of jets, value 0.34 (0.00…1.25), formula right 48%
- `` — 11.7% of jets, value 0.15 (0.00…0.38), formula right 84%
- `` — 36.1% of jets, value 0.00 (0.00…0.00), formula right 59%

```
z = -0.686
if girth2 < 0.013: z += 411 × (0.013 − girth2)
if width < 0.0087: z += 755 × (0.0087 − width)
if mass_over_sum_pt < 0.131: z += -23.12 × (0.131 − mass_over_sum_pt)
if mass_over_sum_pt_sq < 0.0082: z += -332 × (0.0082 − mass_over_sum_pt_sq)
if mass < 29.64 and phi_1 > -0.059: z += -3.11 × (29.64 − mass) × (phi_1 − -0.059)
if girth < 0.076: z += -45.83 × (0.076 − girth)
if mass < 21.78: z += -0.282 × (21.78 − mass)
if lam1 < 0.0084: z += -247 × (0.0084 − lam1)
if width < 0.0044: z += -649 × (0.0044 − width)
if girth < 0.087: z += -26.95 × (0.087 − girth)
if z_dr_0_0p05 > 0.848: z += 16.13 × (z_dr_0_0p05 − 0.848)
if mass < 29.64: z += -0.117 × (29.64 − mass)
if girth2 < 0.019: z += 61.08 × (0.019 − girth2)
if centroid_offset < 0.031: z += 43.41 × (0.031 − centroid_offset)
if lam1 < 0.0065: z += 243 × (0.0065 − lam1)
if girth2_top3 < 0.0079: z += -131 × (0.0079 − girth2_top3)
if mass < 64.62: z += -0.020 × (64.62 − mass)
if girth2_top5 < 0.0083: z += -121 × (0.0083 − girth2_top5)
if mass < 64.62 and centroid_offset > 0.013: z += 3.32 × (64.62 − mass) × (centroid_offset − 0.013)
if mass < 64.62 and pt_7 < 40.04: z += -0.0022 × (64.62 − mass) × (40.04 − pt_7)
if girth2_top3 < 0.004: z += 243 × (0.004 − girth2_top3)
if lam1 < 0.0054: z += -191 × (0.0054 − lam1)
if girth2 < 0.019 and eccentricity > 0.960: z += 2494 × (0.019 − girth2) × (eccentricity − 0.960)
if sum_pt > 902: z += -0.031 × (sum_pt − 902)
if sum_pt_top5 > 687: z += 0.0098 × (sum_pt_top5 − 687)
if mass < 56.92: z += -0.014 × (56.92 − mass)
if e2 < 0.020: z += -56.11 × (0.020 − e2)
if log_sum_pt > 6.64: z += -4.80 × (log_sum_pt − 6.64)
if planar_flow < 0.148 and centroid_offset < 0.050: z += 108 × (0.148 − planar_flow) × (0.050 − centroid_offset)
if planar_flow < 0.148: z += 3.31 × (0.148 − planar_flow)
if planar_flow < 0.148 and sum_pt_top2 < 381: z += -0.051 × (0.148 − planar_flow) × (381 − sum_pt_top2)
if lam1 < 0.0065 and D2 < 0.876: z += -1941 × (0.0065 − lam1) × (0.876 − D2)
if sum_pt > 902 and pt_7 < 25.58: z += 0.0022 × (sum_pt − 902) × (25.58 − pt_7)
if sum_pt_top5 > 687 and z_7 > 0.023: z += -0.738 × (sum_pt_top5 − 687) × (z_7 − 0.023)
if log_sum_pt > 6.38: z += 0.685 × (log_sum_pt − 6.38)
if mass < 56.92 and C2 > 0.024: z += 0.948 × (56.92 − mass) × (C2 − 0.024)
if planar_flow < 0.148 and max_dr < 0.112: z += -149 × (0.148 − planar_flow) × (0.112 − max_dr)
if mass < 29.64 and D2 < 0.876: z += -2.99 × (29.64 − mass) × (0.876 − D2)
if girth2 < 0.013 and centroid_offset > 0.014: z += -1929 × (0.013 − girth2) × (centroid_offset − 0.014)
if girth2_top3 < 0.0079 and pt_6 < 35.28: z += -3.41 × (0.0079 − girth2_top3) × (35.28 − pt_6)
if planar_flow < 0.013: z += 109 × (0.013 − planar_flow)
if sum_pt > 902 and dr_2 < 0.038: z += -0.154 × (sum_pt − 902) × (0.038 − dr_2)
if log_sum_pt > 6.64 and dr_7 < 0.094: z += 14.79 × (log_sum_pt − 6.64) × (0.094 − dr_7)
if width < 0.0044 and n_dr_0p1_0p2 > 1.00: z += -571 × (0.0044 − width) × (n_dr_0p1_0p2 − 1.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 18.7% of jets, neuron 0.00, formula right for 59%.  
- **group 2** — 18.5% of jets, neuron 0.38, formula right for 79%.  
- **group 3** — 17.5% of jets, neuron 3.05, formula right for 73%.  
- **group 4** — 17.3% of jets, neuron 2.18, formula right for 66%.  
- **group 5** — 10.2% of jets, neuron 0.75, formula right for 52%.  
- **group 6** — 7.6% of jets, neuron 0.00, formula right for 55%.  
- **group 7** — 4.3% of jets, neuron 0.00, formula right for 75%.  
- **group 8** — 3.0% of jets, neuron 0.06, formula right for 46%.  
- **group 9** — 2.0% of jets, neuron 1.32, formula right for 63%.  
- **group 10** — 0.8% of jets, neuron 0.05, formula right for 65%.  

### neuron 2: gluon-likeness: many hard particles (major)

- **What it measures:** Moved up when the 8th-hardest particle is still hard (pT_7 > 30.5 GeV, and further when pT_7 is not below 53.4 GeV) and for total pT below 788 GeV, and down for very light jets (mass < 36.2 GeV) and large angularity (LHA > 0.112); overall it falls with jet mass (rank correlation -0.665). Gluons sit highest (3.80), then quarks (2.70), with W (1.98), Z (1.70) and top (1.68) jets lower.
- *computed — its value:* largest for g (3.80), then q (2.70), then W (1.98), then Z (1.70), then t (1.68); it separates g jets from the rest best (AUC 0.79: large for g)
- **How the class scores use it:** Since gluons sit highest on it, the g score reads it as direct evidence for a gluon: it raises the g score, where it is the largest input (+35%). It does not enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+35%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.741):

- `` — 21.5% of jets, value 4.55 (3.31…5.81), formula right 55%
- `` — 12.8% of jets, value 3.23 (2.00…4.50), formula right 49%
- `` — 12.8% of jets, value 2.53 (1.38…3.56), formula right 60%
- `` — 15.0% of jets, value 2.12 (1.19…3.12), formula right 71%
- `` — 5.7% of jets, value 1.29 (0.38…2.12), formula right 84%
- `` — 17.3% of jets, value 1.13 (0.38…1.94), formula right 71%
- `` — 11.3% of jets, value 0.48 (0.00…1.19), formula right 82%
- `` — 3.5% of jets, value 0.37 (0.00…1.38), formula right 77%

```
z = 2.81
if pt_7 < 53.44: z += -0.073 × (53.44 − pt_7)
if sum_pt < 788: z += 0.011 × (788 − sum_pt)
if pt_7 > 30.48: z += 0.156 × (pt_7 − 30.48)
if mass < 36.23: z += -0.083 × (36.23 − mass)
if sum_pt_top5 < 687: z += -0.0061 × (687 − sum_pt_top5)
if LHA > 0.112: z += -5.84 × (LHA − 0.112)
if lam1 < 0.006: z += 279 × (0.006 − lam1)
if mass < 69.61: z += 0.021 × (69.61 − mass)
if mass < 36.23 and lam2 < 0.0011: z += 48.39 × (36.23 − mass) × (0.0011 − lam2)
if pt_7 < 43.50: z += 0.046 × (43.50 − pt_7)
if pt_7 > 30.48 and C2 < 0.051: z += -1.94 × (pt_7 − 30.48) × (0.051 − C2)
if z_7 > 0.046: z += -32.77 × (z_7 − 0.046)
if planar_flow < 0.695: z += -0.884 × (0.695 − planar_flow)
if log_sum_pt < 6.46: z += 3.90 × (6.46 − log_sum_pt)
if log_sum_pt > 6.27: z += 0.786 × (log_sum_pt − 6.27)
if mass < 69.61 and z_7 < 0.068: z += -0.301 × (69.61 − mass) × (0.068 − z_7)
if lam1 < 0.0034: z += 130 × (0.0034 − lam1)
if z_7 < 0.028 and width < 0.0075: z += -17837 × (0.028 − z_7) × (0.0075 − width)
if pt_7 > 30.48 and max_dr > 0.093: z += -0.523 × (pt_7 − 30.48) × (max_dr − 0.093)
if mass < 8.38 and centroid_offset < 0.011: z += 45.30 × (8.38 − mass) × (0.011 − centroid_offset)
if max_dr < 0.160: z += -1.88 × (0.160 − max_dr)
if lam1 < 0.006 and max_dr > 0.081: z += -2028 × (0.006 − lam1) × (max_dr − 0.081)
if z_7 < 0.028: z += 68.88 × (0.028 − z_7)
if z_7 < 0.017: z += -322 × (0.017 − z_7)
if log_sum_pt > 6.90: z += -19.00 × (log_sum_pt − 6.90)
if lam1 < 0.0034 and centroid_offset < 0.0068: z += -32518 × (0.0034 − lam1) × (0.0068 − centroid_offset)
if width < 0.00017 and dr_7 < 0.223: z += -24039 × (0.00017 − width) × (0.223 − dr_7)
if log_sum_pt > 6.84: z += 8.54 × (log_sum_pt − 6.84)
if girth2 < 4.8e-05: z += -75530 × (4.8e-05 − girth2)
if mass < 69.61 and max_pair_mass > 13.05: z += -0.002 × (69.61 − mass) × (max_pair_mass − 13.05)
if log_sum_pt > 6.80: z += 3.65 × (log_sum_pt − 6.80)
if width < 9.1e-05: z += -9679 × (9.1e-05 − width)
if pt_6 < 56.53 and m012 > 32.62: z += -0.0007 × (56.53 − pt_6) × (m012 − 32.62)
if sum_pt_top5 > 840 and dr_7 < 0.049: z += 0.169 × (sum_pt_top5 − 840) × (0.049 − dr_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 18.1% of jets, neuron 0.84, formula right for 70%.  
- **group 2** — 15.8% of jets, neuron 3.00, formula right for 58%.  
- **group 3** — 12.1% of jets, neuron 1.46, formula right for 69%.  
- **group 4** — 11.2% of jets, neuron 1.55, formula right for 73%.  
- **group 5** — 9.4% of jets, neuron 2.24, formula right for 68%.  
- **group 6** — 8.7% of jets, neuron 3.06, formula right for 65%.  
- **group 7** — 8.0% of jets, neuron 4.59, formula right for 55%.  
- **group 8** — 8.0% of jets, neuron 5.00, formula right for 57%.  
- **group 9** — 5.1% of jets, neuron 2.63, formula right for 70%.  
- **group 10** — 3.4% of jets, neuron 0.35, formula right for 76%.  

### neuron 6: broad, off-centre spread (major)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and width < 0.0132 both push it down) and pushed up by a large largest particle distance (max ΔR) and small e2 (< 0.0503); it follows the offset of the pT centroid from the jet axis and the jet width (rank correlations 0.556 and 0.518). Tops sit far highest (4.54), then gluons (1.85) and quarks (0.86), with Z (0.59) and W (0.20) jets lowest.
- *computed — its value:* largest for t (4.54), then g (1.85), then q (0.86), then Z (0.59), then W (0.20); it separates t jets from the rest best (AUC 0.82: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so it lowers the W score (-13%) and the Z score (-21%). It raises the g score (+6%) and the q score (+9%), since among non-top jets a broad spread is QCD-like rather than boson-like; it does not enter the t score, which gets its width information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+6%), q (+9%); lowers the score of W (-13%), Z (-21%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.689):

- `` — 8.7% of jets, value 8.06 (4.12…12.25), formula right 77%
- `` — 6.9% of jets, value 5.19 (1.75…8.88), formula right 70%
- `` — 2.0% of jets, value 4.44 (1.50…8.00), formula right 48%
- `` — 3.8% of jets, value 2.29 (0.00…5.75), formula right 94%
- `` — 3.5% of jets, value 2.24 (0.00…4.62), formula right 47%
- `` — 3.3% of jets, value 2.11 (0.00…5.62), formula right 56%
- `` — 71.7% of jets, value 0.31 (0.00…1.12), formula right 64%

```
z = 8.52
if girth2 < 0.0087: z += -1359 × (0.0087 − girth2)
if width < 0.013: z += -657 × (0.013 − width)
if mass_over_sum_pt > 0.0084: z += -79.51 × (mass_over_sum_pt − 0.0084)
z += 16.25 × max_dr
if e2 < 0.050: z += 71.61 × (0.050 − e2)
if lam1 < 0.0084: z += 297 × (0.0084 − lam1)
if lam1 < 0.0073: z += 253 × (0.0073 − lam1)
if lam2 < 0.00054: z += -2034 × (0.00054 − lam2)
if girth2 < 0.0036: z += 622 × (0.0036 − girth2)
if mass < 49.67 and z_dr_0p05_0p1 < 0.751: z += 0.062 × (49.67 − mass) × (0.751 − z_dr_0p05_0p1)
if girth > 0.087: z += 89.03 × (girth − 0.087)
if centroid_offset > 0.018: z += -105 × (centroid_offset − 0.018)
if mass_over_sum_pt < 0.154: z += 5.22 × (0.154 − mass_over_sum_pt)
if log_sum_pt < 6.70 and pt_7 < 45.75: z += 0.236 × (6.70 − log_sum_pt) × (45.75 − pt_7)
if log_sum_pt < 6.70: z += -2.31 × (6.70 − log_sum_pt)
if C2 > 0.011: z += -21.69 × (C2 − 0.011)
if centroid_offset > 0.0081 and lam2 < 0.0034: z += 14174 × (centroid_offset − 0.0081) × (0.0034 − lam2)
if log_sum_pt < 6.57: z += 2.99 × (6.57 − log_sum_pt)
if mass_over_sum_pt > 0.0084 and tau32 < 0.519: z += -58.42 × (mass_over_sum_pt − 0.0084) × (0.519 − tau32)
if D2 < 1.68 and pt_4 < 90.62: z += -0.017 × (1.68 − D2) × (90.62 − pt_4)
if girth2_top5 > 0.011: z += 184 × (girth2_top5 − 0.011)
if centroid_offset > 0.0081: z += 21.56 × (centroid_offset − 0.0081)
if C2 > 0.011 and pt_7 > 31.86: z += 2.75 × (C2 − 0.011) × (pt_7 − 31.86)
if mass < 49.67: z += -0.013 × (49.67 − mass)
if D2 < 1.68: z += 0.361 × (1.68 − D2)
if e2 < 0.050 and n_dr_0p05_0p1 < 4.00: z += -2.41 × (0.050 − e2) × (4.00 − n_dr_0p05_0p1)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.159: z += 792 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.159)
if lam2 > 0.0034: z += -1052 × (lam2 − 0.0034)
if LHA > 0.313 and eccentricity > 0.873: z += 180 × (LHA − 0.313) × (eccentricity − 0.873)
if log_sum_pt < 6.70 and z_7 < 0.049: z += 690 × (6.70 − log_sum_pt) × (0.049 − z_7)
if girth2_top5 > 0.0023: z += -32.55 × (girth2_top5 − 0.0023)
if centroid_offset > 0.0081 and C2 < 0.095: z += 211 × (centroid_offset − 0.0081) × (0.095 − C2)
if centroid_offset > 0.0081 and planar_flow > 0.008: z += 33.43 × (centroid_offset − 0.0081) × (planar_flow − 0.008)
if log_sum_pt < 6.33 and z_7 < 0.071: z += 871 × (6.33 − log_sum_pt) × (0.071 − z_7)
if centroid_offset > 0.018 and mean_phi2 < 0.0089: z += 3782 × (centroid_offset − 0.018) × (0.0089 − mean_phi2)
if centroid_offset > 0.018 and pt_2 > 56.50: z += 0.523 × (centroid_offset − 0.018) × (pt_2 − 56.50)
if log_sum_pt < 6.33: z += 1.52 × (6.33 − log_sum_pt)
if lam1 < 0.0073 and D2 < 1.23: z += 134 × (0.0073 − lam1) × (1.23 − D2)
if log_sum_pt < 6.70 and pt_6 < 36.81: z += 0.117 × (6.70 − log_sum_pt) × (36.81 − pt_6)
if log_sum_pt < 6.70 and mean_phi > 0.0091: z += -46.33 × (6.70 − log_sum_pt) × (mean_phi − 0.0091)
if z_7 > 0.062: z += -4.52 × (z_7 − 0.062)
if log_sum_pt < 6.33 and pt_6 > 27.58: z += 0.065 × (6.33 − log_sum_pt) × (pt_6 − 27.58)
if lam2 < 0.00054 and mean_eta > 0.026: z += -131384 × (0.00054 − lam2) × (mean_eta − 0.026)
if centroid_offset > 0.018 and pt_5 < 24.58: z += 61.00 × (centroid_offset − 0.018) × (24.58 − pt_5)
if width < 0.013 and mean_eta > 0.026: z += 9072 × (0.013 − width) × (mean_eta − 0.026)
if centroid_offset > 0.050: z += -13.47 × (centroid_offset − 0.050)
if width < 0.013 and mean_phi > 0.026: z += 6150 × (0.013 − width) × (mean_phi − 0.026)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 29.6% of jets, neuron 0.32, formula right for 61%.  
- **group 2** — 18.1% of jets, neuron 0.44, formula right for 70%.  
- **group 3** — 13.1% of jets, neuron 0.50, formula right for 58%.  
- **group 4** — 11.4% of jets, neuron 1.63, formula right for 73%.  
- **group 5** — 10.4% of jets, neuron 0.94, formula right for 50%.  
- **group 6** — 7.1% of jets, neuron 5.79, formula right for 78%.  
- **group 7** — 4.5% of jets, neuron 8.23, formula right for 80%.  
- **group 8** — 3.1% of jets, neuron 7.82, formula right for 72%.  
- **group 9** — 2.4% of jets, neuron 1.44, formula right for 95%.  
- **group 10** — 0.3% of jets, neuron 7.91, formula right for 65%.  

### neuron 7: two-prong shape of W/Z size (major)

- **What it measures:** Mostly set by girth2 steps (pushed down above 0.00165 and above 0.00752, partly back up above 0.00868) and by e2 steps; it is also pushed down for very narrow jets (width < 0.00559, girth < 0.0872) and for mass/pT > 0.0904; it rises with eccentricity and the pT share at 0.05 ≤ ΔR < 0.1 and falls with planar flow and τ21 (rank correlations 0.564, 0.507, -0.564, -0.52). Z jets sit highest (3.50), then W (2.26), with top (1.02), gluon (0.80) and quark (0.41) jets low.
- *computed — its value:* largest for Z (3.50), then W (2.26), then t (1.02), then g (0.80), then q (0.41); it separates Z jets from the rest best (AUC 0.83: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the W score (+9%) and the Z score (+27%), where it is the largest input. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+9%), Z (+27%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.798):

- `` — 23.8% of jets, value 4.10 (2.62…5.88), formula right 71%
- `` — 2.9% of jets, value 2.39 (1.38…3.38), formula right 43%
- `` — 19.5% of jets, value 2.34 (1.00…3.50), formula right 62%
- `` — 3.9% of jets, value 1.34 (0.50…2.00), formula right 45%
- `` — 2.0% of jets, value 0.85 (0.00…2.00), formula right 71%
- `` — 2.0% of jets, value 0.80 (0.00…1.88), formula right 52%
- `` — 31.6% of jets, value 0.06 (0.00…0.12), formula right 61%
- `` — 14.3% of jets, value 0.00 (0.00…0.00), formula right 82%

```
z = 9.88
if girth2 > 0.0075: z += -1416 × (girth2 − 0.0075)
if girth2 > 0.0087: z += 1477 × (girth2 − 0.0087)
if girth < 0.087: z += -80.24 × (0.087 − girth)
if girth2 > 0.0017: z += -541 × (girth2 − 0.0017)
if mass_over_sum_pt > 0.090: z += -319 × (mass_over_sum_pt − 0.090)
if width < 0.0056: z += -1015 × (0.0056 − width)
if e2 < 0.038: z += 141 × (0.038 − e2)
if e2 < 0.050: z += -88.24 × (0.050 − e2)
if lam1 < 0.0084: z += -448 × (0.0084 − lam1)
if mass_over_sum_pt > 0.085: z += 201 × (mass_over_sum_pt − 0.085)
if girth2 > 0.0044: z += -491 × (girth2 − 0.0044)
if mass_over_sum_pt > 0.073: z += 109 × (mass_over_sum_pt − 0.073)
if LHA < 0.293: z += 13.33 × (0.293 − LHA)
if girth2 > 0.0044 and eccentricity > 0.946: z += 8771 × (girth2 − 0.0044) × (eccentricity − 0.946)
if pt_7 < 48.72 and planar_flow < 0.695: z += -0.083 × (48.72 − pt_7) × (0.695 − planar_flow)
if planar_flow < 0.195: z += -6.70 × (0.195 − planar_flow)
if e2_sq < 0.0011: z += 1563 × (0.0011 − e2_sq)
if e2 < 0.025: z += 54.71 × (0.025 − e2)
if e2 < 0.050 and D2 < 1.12: z += 156 × (0.050 − e2) × (1.12 − D2)
if centroid_offset < 0.021: z += -43.83 × (0.021 − centroid_offset)
if planar_flow < 0.195 and sum_pt > 616: z += 0.035 × (0.195 − planar_flow) × (sum_pt − 616)
if max_dr < 0.198: z += -3.97 × (0.198 − max_dr)
if mass < 29.64: z += 0.044 × (29.64 − mass)
if lam1 < 0.0084 and D2 < 1.12: z += -733 × (0.0084 − lam1) × (1.12 − D2)
if planar_flow < 0.195 and width > 0.0075: z += -2189 × (0.195 − planar_flow) × (width − 0.0075)
if girth2_top3 < 0.0059: z += 78.27 × (0.0059 − girth2_top3)
if mass > 80.40: z += -0.184 × (mass − 80.40)
if girth2_top2 < 0.0011: z += -740 × (0.0011 − girth2_top2)
if girth < 0.041: z += -19.66 × (0.041 − girth)
if pt_7 < 48.72: z += 0.012 × (48.72 − pt_7)
if width < 0.00056: z += -1487 × (0.00056 − width)
if mass > 80.40 and eccentricity > 0.927: z += -2.79 × (mass − 80.40) × (eccentricity − 0.927)
if centroid_offset < 0.021 and C2 > 0.024: z += 1701 × (0.021 − centroid_offset) × (C2 − 0.024)
if planar_flow < 0.195 and mass_top3 > 23.66: z += 0.120 × (0.195 − planar_flow) × (mass_top3 − 23.66)
if e2 < 0.025 and tau21 < 0.447: z += 161 × (0.025 − e2) × (0.447 − tau21)
if centroid_offset > 0.031 and pt_2 > 56.50: z += -1.09 × (centroid_offset − 0.031) × (pt_2 − 56.50)
if girth2_top2 < 0.0011 and centroid_offset > 0.0068: z += 46611 × (0.0011 − girth2_top2) × (centroid_offset − 0.0068)
if planar_flow < 0.195 and pt_6 < 35.28: z += -0.201 × (0.195 − planar_flow) × (35.28 − pt_6)
if centroid_offset > 0.031 and pt_0 > 376: z += -9.27 × (centroid_offset − 0.031) × (pt_0 − 376)
if centroid_offset > 0.031: z += 4.45 × (centroid_offset − 0.031)
if e2_sq < 0.0011 and phi_1 < -0.0095: z += -6215 × (0.0011 − e2_sq) × (-0.0095 − phi_1)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 33.7% of jets, neuron 0.30, formula right for 60%.  
- **group 2** — 29.0% of jets, neuron 3.81, formula right for 71%.  
- **group 3** — 19.4% of jets, neuron 1.77, formula right for 55%.  
- **group 4** — 4.1% of jets, neuron 1.39, formula right for 68%.  
- **group 5** — 3.9% of jets, neuron 0.01, formula right for 78%.  
- **group 6** — 3.5% of jets, neuron 0.00, formula right for 82%.  
- **group 7** — 3.3% of jets, neuron 0.00, formula right for 89%.  
- **group 8** — 1.8% of jets, neuron 0.00, formula right for 90%.  
- **group 9** — 0.9% of jets, neuron 0.00, formula right for 70%.  
- **group 10** — 0.4% of jets, neuron 0.00, formula right for 64%.  

### neuron 9: narrow, light single-core jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.0061, its strongest term, and girth2 < 0.00752), and for a pT centroid close to the axis; low mass (< 53.3 GeV) counts in its favour only when the jet is also centred and not very hard (log of total pT < 6.84); it falls with lam1, width and mass (rank correlations -0.659, -0.648, -0.623). Quarks sit highest (9.20), then gluons (6.72), far above top (2.35), W (2.33) and Z (1.40) jets.
- *computed — its value:* largest for q (9.20), then g (6.72), then t (2.35), then W (2.33), then Z (1.40); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** High values mean a light-parton (QCD) jet, so it raises the q score (its largest input, +49%) and the g score (+26%), and lowers the W score (-3%) and the Z score (-5%) slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+26%), q (+49%); lowers the score of W (-3%), Z (-5%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.784):

- `` — 19.9% of jets, value 11.97 (9.00…14.50), formula right 65%
- `` — 2.4% of jets, value 8.95 (5.25…12.25), formula right 58%
- `` — 6.1% of jets, value 8.47 (4.25…12.75), formula right 51%
- `` — 2.1% of jets, value 8.34 (0.50…15.25), formula right 66%
- `` — 3.3% of jets, value 7.52 (2.75…13.00), formula right 94%
- `` — 3.4% of jets, value 4.67 (1.00…8.00), formula right 46%
- `` — 10.6% of jets, value 2.24 (0.00…5.50), formula right 45%
- `` — 52.3% of jets, value 0.83 (0.00…2.75), formula right 71%

```
z = -1.64
if width < 0.0061: z += 1635 × (0.0061 − width)
if girth2 < 0.0075: z += 487 × (0.0075 − girth2)
if mass < 53.33: z += -0.088 × (53.33 − mass)
if girth2 < 0.0044: z += 896 × (0.0044 − girth2)
if girth < 0.055: z += -86.57 × (0.055 − girth)
if centroid_offset < 0.018: z += 194 × (0.018 − centroid_offset)
if mass < 53.33 and centroid_offset < 0.027: z += 4.12 × (53.33 − mass) × (0.027 − centroid_offset)
if mass < 53.33 and log_sum_pt < 6.84: z += 0.225 × (53.33 − mass) × (6.84 − log_sum_pt)
if log_sum_pt > 6.38: z += -5.51 × (log_sum_pt − 6.38)
if width < 0.0061 and centroid_offset > 0.0033: z += -52030 × (0.0061 − width) × (centroid_offset − 0.0033)
if mass_over_sum_pt < 0.076: z += -38.71 × (0.076 − mass_over_sum_pt)
if max_dr < 0.222: z += 8.44 × (0.222 − max_dr)
if girth2 < 0.00096: z += 4246 × (0.00096 − girth2)
if mass < 29.64: z += -0.092 × (29.64 − mass)
if e2 < 0.020: z += 116 × (0.020 − e2)
if mass < 41.38: z += 0.050 × (41.38 − mass)
if mass < 53.33 and lam1 < 0.00087: z += -63.83 × (53.33 − mass) × (0.00087 − lam1)
if centroid_offset < 0.018 and z_4 > 0.047: z += -2389 × (0.018 − centroid_offset) × (z_4 − 0.047)
if mass < 53.33 and planar_flow < 0.322: z += 0.332 × (53.33 − mass) × (0.322 − planar_flow)
if e2 < 0.032: z += -41.19 × (0.032 − e2)
if mass < 41.38 and planar_flow < 0.322: z += -0.405 × (41.38 − mass) × (0.322 − planar_flow)
if lam1 < 0.0015: z += -822 × (0.0015 − lam1)
if centroid_offset < 0.018 and pt_4 > 47.34: z += 2.58 × (0.018 − centroid_offset) × (pt_4 − 47.34)
if lam2 > 0.0011: z += 849 × (lam2 − 0.0011)
if e2 < 0.032 and dr01 < 0.056: z += 403 × (0.032 − e2) × (0.056 − dr01)
if mass < 29.64 and mean_phi2 < 0.0021: z += -15.36 × (29.64 − mass) × (0.0021 − mean_phi2)
if centroid_offset < 0.018 and z_5 > 0.037: z += -991 × (0.018 − centroid_offset) × (z_5 − 0.037)
if girth2 > 0.019: z += 232 × (girth2 − 0.019)
if n_dr_0p2_0p4 > 1.00: z += 1.04 × (n_dr_0p2_0p4 − 1.00)
if e2 < 0.032 and tau21 < 0.447: z += -198 × (0.032 − e2) × (0.447 − tau21)
if C2 > 0.051: z += 23.14 × (C2 − 0.051)
if pt_5 < 35.50: z += 0.063 × (35.50 − pt_5)
if width < 0.00017: z += 6423 × (0.00017 − width)
if e2 < 0.020 and eccentricity > 0.903: z += -666 × (0.020 − e2) × (eccentricity − 0.903)
if max_dr > 0.160: z += -3.37 × (max_dr − 0.160)
if e2 < 0.020 and pt_7 < 53.44: z += 0.593 × (0.020 − e2) × (53.44 − pt_7)
if width < 0.0061 and C2 > 0.031: z += -10698 × (0.0061 − width) × (C2 − 0.031)
if girth2 > 0.019 and planar_flow < 0.401: z += -493 × (girth2 − 0.019) × (0.401 − planar_flow)
if C2 > 0.051 and n_dr_0p05_0p1 > 1.00: z += -9.17 × (C2 − 0.051) × (n_dr_0p05_0p1 − 1.00)
if girth2 > 0.019 and mean_eta < 0.026: z += -1862 × (girth2 − 0.019) × (0.026 − mean_eta)
if C2 > 0.051 and eccentricity < 0.621: z += 203 × (C2 − 0.051) × (0.621 − eccentricity)
if girth2 > 0.019 and pt_7 < 29.04: z += 63.06 × (girth2 − 0.019) × (29.04 − pt_7)
if pt_5 < 35.50 and min_pair_mass > 0.173: z += -0.026 × (35.50 − pt_5) × (min_pair_mass − 0.173)
if lam2 > 0.0011 and mass_top2 > 16.31: z += 12.81 × (lam2 − 0.0011) × (mass_top2 − 16.31)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 21.8% of jets, neuron 0.46, formula right for 71%.  
- **group 2** — 19.2% of jets, neuron 0.61, formula right for 77%.  
- **group 3** — 15.5% of jets, neuron 12.20, formula right for 67%.  
- **group 4** — 10.7% of jets, neuron 1.73, formula right for 58%.  
- **group 5** — 7.9% of jets, neuron 9.77, formula right for 59%.  
- **group 6** — 6.1% of jets, neuron 8.48, formula right for 54%.  
- **group 7** — 5.2% of jets, neuron 4.68, formula right for 47%.  
- **group 8** — 4.9% of jets, neuron 3.25, formula right for 48%.  
- **group 9** — 4.4% of jets, neuron 1.79, formula right for 43%.  
- **group 10** — 4.1% of jets, neuron 7.03, formula right for 93%.  

### neuron 10: overall jet size (e2, width, mass) (major)

- **What it measures:** Grows steadily with e2 (its strongest term) and is pushed up for mass > 15.5 GeV and for mass/pT < 0.0904; it follows e2, width and girth2 very closely (rank correlations 0.79, 0.785, 0.785). Tops sit highest (6.03), W (3.27) and Z (3.14) in the middle, gluons (1.99) and quarks (1.63) lowest.
- *computed — its value:* largest for t (6.03), then W (3.27), then Z (3.14), then g (1.99), then q (1.63); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** Large size marks a top, so it raises the t score (+26%); small size marks a quark, so it lowers the q score (-18%). It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+26%); lowers the score of q (-18%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.821):

- `` — 2.3% of jets, value 13.75 (10.38…17.10), formula right 95%
- `` — 3.6% of jets, value 9.37 (5.75…12.74), formula right 88%
- `` — 2.2% of jets, value 7.30 (3.44…10.75), formula right 81%
- `` — 59.5% of jets, value 3.42 (2.31…4.31), formula right 65%
- `` — 7.5% of jets, value 1.88 (1.12…2.69), formula right 53%
- `` — 24.8% of jets, value 0.91 (0.31…1.69), formula right 62%

```
z = -1.73
z += 67.39 × e2
if mass_over_sum_pt < 0.090: z += 37.67 × (0.090 − mass_over_sum_pt)
if mass > 15.45: z += 0.039 × (mass − 15.45)
if lam1 < 0.0042: z += -465 × (0.0042 − lam1)
if lam2 > 0.00019: z += 1562 × (lam2 − 0.00019)
if girth2 < 0.0017: z += -1616 × (0.0017 − girth2)
if girth2 < 0.0067: z += 213 × (0.0067 − girth2)
if log_sum_pt > 6.70: z += -17.60 × (log_sum_pt − 6.70)
if girth2_top2 < 0.0024: z += 538 × (0.0024 − girth2_top2)
if eccentricity > 0.903 and z_dr_0p2_0p4 < 0.056: z += 226 × (eccentricity − 0.903) × (0.056 − z_dr_0p2_0p4)
if lam1 < 0.0065: z += -165 × (0.0065 − lam1)
if n_dr_0p05_0p1 < 3.00: z += 0.248 × (3.00 − n_dr_0p05_0p1)
if lam1 < 0.0042 and log_sum_pt > 6.70: z += 3909 × (0.0042 − lam1) × (log_sum_pt − 6.70)
if sum_pt > 813: z += 0.011 × (sum_pt − 813)
if lam1 < 0.0042 and dr_7 < 0.175: z += 1221 × (0.0042 − lam1) × (0.175 − dr_7)
if tau21 < 0.392 and pt_4 > 39.81: z += -0.080 × (0.392 − tau21) × (pt_4 − 39.81)
if LHA > 0.303 and tau21 < 0.553: z += -39.40 × (LHA − 0.303) × (0.553 − tau21)
if LHA > 0.303: z += -11.27 × (LHA − 0.303)
if tau21 < 0.392: z += 1.41 × (0.392 − tau21)
if lam2 > 0.00019 and planar_flow > 0.013: z += -587 × (lam2 − 0.00019) × (planar_flow − 0.013)
if max_dr > 0.122: z += -4.82 × (max_dr − 0.122)
if tau21 < 0.392 and z_4 > 0.075: z += 79.07 × (0.392 − tau21) × (z_4 − 0.075)
if C2 > 0.051: z += 28.14 × (C2 − 0.051)
if lam1 < 0.0042 and sum_pt > 988: z += -9.05 × (0.0042 − lam1) × (sum_pt − 988)
if tau32 < 0.269: z += 10.95 × (0.269 − tau32)
if mass > 15.45 and n_dr_0p2_0p4 < 2.00: z += -0.0028 × (mass − 15.45) × (2.00 − n_dr_0p2_0p4)
if lam2 > 0.00019 and D2 < 2.06: z += -266 × (lam2 − 0.00019) × (2.06 − D2)
if lam2 > 0.00019 and tau21 < 0.501: z += 1977 × (lam2 − 0.00019) × (0.501 − tau21)
if lam2 > 0.00019 and n_pt_above_50 < 8.00: z += -48.85 × (lam2 − 0.00019) × (8.00 − n_pt_above_50)
if log_sum_pt < 6.27: z += -2.55 × (6.27 − log_sum_pt)
if centroid_offset > 0.0023 and pt_7 < 34.53: z += 0.658 × (centroid_offset − 0.0023) × (34.53 − pt_7)
if tau32 < 0.269 and n_dr_0p2_0p4 < 2.00: z += -3.88 × (0.269 − tau32) × (2.00 − n_dr_0p2_0p4)
if n_dr_0p2_0p4 > 2.00: z += 0.435 × (n_dr_0p2_0p4 − 2.00)
if C2 > 0.051 and dr_7 < 0.223: z += -27.20 × (C2 − 0.051) × (0.223 − dr_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.5% of jets, neuron 3.57, formula right for 69%.  
- **group 2** — 20.9% of jets, neuron 1.03, formula right for 55%.  
- **group 3** — 15.9% of jets, neuron 2.89, formula right for 51%.  
- **group 4** — 9.2% of jets, neuron 1.58, formula right for 70%.  
- **group 5** — 8.9% of jets, neuron 5.10, formula right for 77%.  
- **group 6** — 6.0% of jets, neuron 3.10, formula right for 73%.  
- **group 7** — 4.0% of jets, neuron 10.12, formula right for 90%.  
- **group 8** — 1.7% of jets, neuron 0.20, formula right for 68%.  
- **group 9** — 1.6% of jets, neuron 14.45, formula right for 95%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 71%.  

### neuron 11: compact, centred, flat (non-top) jet (major)

- **What it measures:** Pushed up for width < 0.00868 (its strongest term) and a pT centroid close to the axis (centroid offset < 0.0499), and down for girth < 0.0872; it falls with centroid offset and planar flow (rank correlations -0.379 and -0.37). W jets sit highest (4.30), then Z (3.17), gluons (2.23) and quarks (2.21); tops are lowest (0.82) and at zero for 70.8% of them.
- *computed — its value:* largest for W (4.30), then Z (3.17), then g (2.23), then q (2.21), then t (0.82); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** It is the largest input of the W score, which it raises (+24%), since W jets sit highest on it. It does not enter the g, q, Z or t scores; in particular the Z score does not use it although Z jets sit second on it.
- *computed — used by:* raises the score of W (+24%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.713):

- `` — 24.4% of jets, value 4.80 (3.19…6.44), formula right 72%
- `` — 28.1% of jets, value 2.92 (1.25…4.38), formula right 57%
- `` — 5.1% of jets, value 2.77 (0.69…4.44), formula right 58%
- `` — 22.6% of jets, value 1.72 (0.81…2.56), formula right 60%
- `` — 2.1% of jets, value 1.03 (0.00…2.56), formula right 60%
- `` — 2.0% of jets, value 0.23 (0.00…0.94), formula right 65%
- `` — 15.6% of jets, value 0.01 (0.00…0.00), formula right 82%

```
z = -1.62
if width < 0.0087: z += 1085 × (0.0087 − width)
if centroid_offset < 0.050: z += 86.90 × (0.050 − centroid_offset)
if girth < 0.087: z += -67.29 × (0.087 − girth)
if girth2 < 0.0067: z += 715 × (0.0067 − girth2)
if girth2 < 0.013: z += 225 × (0.013 − girth2)
if lam1 < 0.0084: z += -334 × (0.0084 − lam1)
if e2_sq < 0.0064: z += -461 × (0.0064 − e2_sq)
if planar_flow < 0.253: z += 8.61 × (0.253 − planar_flow)
if sum_pt_top5 < 687: z += 0.0069 × (687 − sum_pt_top5)
if centroid_offset < 0.050 and log_sum_pt < 6.80: z += -115 × (0.050 − centroid_offset) × (6.80 − log_sum_pt)
if e2 < 0.044: z += -36.67 × (0.044 − e2)
if girth > 0.076: z += -54.47 × (girth − 0.076)
if width < 0.0036: z += -450 × (0.0036 − width)
if n_dr_0p1_0p2 < 3.00: z += -0.262 × (3.00 − n_dr_0p1_0p2)
if LHA < 0.155: z += -32.08 × (0.155 − LHA)
if C2 < 0.036: z += -24.04 × (0.036 − C2)
if centroid_offset > 0.014: z += -46.96 × (centroid_offset − 0.014)
if max_dr < 0.222: z += 3.06 × (0.222 − max_dr)
if lam1 < 0.0048: z += -170 × (0.0048 − lam1)
if planar_flow < 0.253 and width > 0.0061: z += -1129 × (0.253 − planar_flow) × (width − 0.0061)
if girth > 0.076 and n_pt_above_50 < 7.00: z += 10.64 × (girth − 0.076) × (7.00 − n_pt_above_50)
if planar_flow < 0.253 and mass < 69.61: z += -0.123 × (0.253 − planar_flow) × (69.61 − mass)
if planar_flow < 0.253 and max_dr > 0.103: z += -47.77 × (0.253 − planar_flow) × (max_dr − 0.103)
if n_dr_0_0p05 < 7.00: z += -0.066 × (7.00 − n_dr_0_0p05)
if max_dr < 0.112: z += -6.62 × (0.112 − max_dr)
if centroid_offset < 0.050 and pt_7 < 48.72: z += 0.361 × (0.050 − centroid_offset) × (48.72 − pt_7)
if centroid_offset < 0.050 and n_dr_0p05_0p1 > 2.00: z += 3.43 × (0.050 − centroid_offset) × (n_dr_0p05_0p1 − 2.00)
if pt_7 < 29.04: z += -0.053 × (29.04 − pt_7)
if mass < 15.45: z += 0.048 × (15.45 − mass)
if pt_7 < 29.04 and n_dr_0p2_0p4 < 2.00: z += -0.025 × (29.04 − pt_7) × (2.00 − n_dr_0p2_0p4)
if pt_7 < 29.04 and mass_top2 < 22.84: z += 0.0019 × (29.04 − pt_7) × (22.84 − mass_top2)
if mass_top5 < 9.26: z += 0.038 × (9.26 − mass_top5)
z += 3.93 × centroid_offset
if LHA < 0.155 and z_7 < 0.028: z += 1026 × (0.155 − LHA) × (0.028 − z_7)
if e2 < 0.044 and pt_5 > 43.06: z += -0.236 × (0.044 − e2) × (pt_5 − 43.06)
if e2_sq < 0.0064 and D2 < 1.33: z += 133 × (0.0064 − e2_sq) × (1.33 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 17.2% of jets, neuron 1.72, formula right for 66%.  
- **group 2** — 15.7% of jets, neuron 4.32, formula right for 73%.  
- **group 3** — 13.1% of jets, neuron 3.94, formula right for 64%.  
- **group 4** — 10.5% of jets, neuron 2.87, formula right for 46%.  
- **group 5** — 10.4% of jets, neuron 1.84, formula right for 74%.  
- **group 6** — 9.8% of jets, neuron 3.34, formula right for 50%.  
- **group 7** — 8.0% of jets, neuron 3.06, formula right for 61%.  
- **group 8** — 6.1% of jets, neuron 0.03, formula right for 73%.  
- **group 9** — 5.5% of jets, neuron 0.02, formula right for 90%.  
- **group 10** — 3.8% of jets, neuron 0.00, formula right for 72%.  

### neuron 13: compactness (girth below top size) (major)

- **What it measures:** Driven mostly by girth < 0.148 (its dominant term) and lam1 < 0.0164, both pushing it up, i.e. by the jet being narrower than a typical top; small e2 (< 0.0503) and very small width (< 0.00752) pull it down a little. It falls with width, girth2 and girth (rank correlations -0.82, -0.82, -0.814); quarks (7.15), W (6.16), gluons (5.89) and Z (5.73) all sit high, tops far lower (1.98).
- *computed — its value:* largest for q (7.15), then W (6.16), then g (5.89), then Z (5.73), then t (1.98); it separates t jets from the rest best (AUC 0.09: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-46%); it also raises the W score (+9%) and the Z score (+10%). It does not enter the g or q scores.
- *computed — used by:* raises the score of W (+9%), Z (+10%); lowers the score of t (-46%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.806):

- `` — 22.6% of jets, value 8.40 (6.62…9.88), formula right 62%
- `` — 6.8% of jets, value 6.41 (4.75…7.75), formula right 59%
- `` — 23.0% of jets, value 6.10 (4.75…7.62), formula right 71%
- `` — 28.8% of jets, value 4.98 (3.50…6.38), formula right 57%
- `` — 2.8% of jets, value 3.05 (1.73…4.50), formula right 66%
- `` — 3.4% of jets, value 1.78 (0.12…3.25), formula right 70%
- `` — 3.6% of jets, value 0.93 (0.00…2.00), formula right 80%
- `` — 9.0% of jets, value 0.20 (0.00…0.75), formula right 84%

```
z = 2.55
if girth < 0.148: z += 46.42 × (0.148 − girth)
if lam1 < 0.016: z += 177 × (0.016 − lam1)
if e2 < 0.050: z += -78.48 × (0.050 − e2)
if width < 0.0075: z += -378 × (0.0075 − width)
if width < 0.013: z += 146 × (0.013 − width)
if centroid_offset < 0.038: z += 43.42 × (0.038 − centroid_offset)
if lam1 < 0.0065: z += 321 × (0.0065 − lam1)
if girth < 0.148 and log_sum_pt < 6.80: z += -37.57 × (0.148 − girth) × (6.80 − log_sum_pt)
if girth < 0.148 and pt_7 < 38.53: z += -1.09 × (0.148 − girth) × (38.53 − pt_7)
if lam2 < 0.00031 and centroid_offset < 0.050: z += -91305 × (0.00031 − lam2) × (0.050 − centroid_offset)
if LHA > 0.093: z += -4.32 × (LHA − 0.093)
if lam2 < 0.00031: z += 2996 × (0.00031 − lam2)
if sum_pt_top5 > 658 and pt_7 < 43.50: z += 0.00056 × (sum_pt_top5 − 658) × (43.50 − pt_7)
if tau21 < 0.501 and max_dr > 0.016: z += -16.14 × (0.501 − tau21) × (max_dr − 0.016)
if lam1 < 0.016 and centroid_offset < 0.038: z += -1509 × (0.016 − lam1) × (0.038 − centroid_offset)
if lam1 < 0.016 and pt_6 < 56.53: z += -1.67 × (0.016 − lam1) × (56.53 − pt_6)
if z_6 < 0.067: z += 19.93 × (0.067 − z_6)
if pt_7 < 25.58: z += -0.190 × (25.58 − pt_7)
if sum_pt < 764: z += -0.002 × (764 − sum_pt)
if sum_pt > 988: z += -0.043 × (sum_pt − 988)
if e2 < 0.050 and pt_dispersion > 0.397: z += 101 × (0.050 − e2) × (pt_dispersion − 0.397)
if C2 > 0.067: z += -56.29 × (C2 − 0.067)
if sum_pt_top5 > 840: z += 0.017 × (sum_pt_top5 − 840)
if tau21 < 0.501: z += -0.594 × (0.501 − tau21)
if z_7 < 0.028: z += -93.79 × (0.028 − z_7)
if sum_pt_top5 < 531: z += 0.0026 × (531 − sum_pt_top5)
if sum_pt_top5 > 658 and z_7 > 0.023: z += -0.311 × (sum_pt_top5 − 658) × (z_7 − 0.023)
if sum_pt_top5 > 658: z += 0.0011 × (sum_pt_top5 − 658)
if sum_pt_top5 > 902 and D2 < 3.89: z += 0.008 × (sum_pt_top5 − 902) × (3.89 − D2)
if sum_pt > 988 and n_pt_above_50 > 6.00: z += 0.018 × (sum_pt − 988) × (n_pt_above_50 − 6.00)
if sum_pt_top5 > 840 and z_dr_0p1_0p2 < 0.159: z += 0.032 × (sum_pt_top5 − 840) × (0.159 − z_dr_0p1_0p2)
if sum_pt > 988 and D2 < 3.89: z += -0.004 × (sum_pt − 988) × (3.89 − D2)
if sum_pt_top5 > 902: z += -0.006 × (sum_pt_top5 − 902)
if sum_pt < 764 and z_4 < 0.037: z += 6.59 × (764 − sum_pt) × (0.037 − z_4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.9% of jets, neuron 4.99, formula right for 70%.  
- **group 2** — 22.9% of jets, neuron 7.62, formula right for 55%.  
- **group 3** — 16.7% of jets, neuron 5.37, formula right for 53%.  
- **group 4** — 10.3% of jets, neuron 0.99, formula right for 75%.  
- **group 5** — 8.0% of jets, neuron 7.75, formula right for 65%.  
- **group 6** — 5.9% of jets, neuron 0.32, formula right for 92%.  
- **group 7** — 4.6% of jets, neuron 5.91, formula right for 76%.  
- **group 8** — 3.6% of jets, neuron 8.10, formula right for 75%.  
- **group 9** — 1.0% of jets, neuron 7.30, formula right for 61%.  
- **group 10** — 0.2% of jets, neuron 6.97, formula right for 70%.  

### neuron 14: intermediate width (Z-sized spread) (major)

- **What it measures:** Pushed down for the narrowest jets (width < 0.00752, its strongest term; also girth < 0.0872) but up for width < 0.00868 and girth2 < 0.0132, so it peaks for jets of intermediate width, a bit wider than a typical W; it rises with the number of particles above 10 GeV and at 0.2 ≤ ΔR < 0.4 and with eccentricity (rank correlations 0.355, 0.343, 0.341). Z jets sit highest (1.51), well above tops (0.41), gluons (0.22), W (0.18) and quarks (0.16).
- *computed — its value:* largest for Z (1.51), then t (0.41), then g (0.22), then W (0.18), then q (0.16); it separates Z jets from the rest best (AUC 0.78: large for Z)
- **How the class scores use it:** It is the W/Z separator: Z jets sit high and W jets low on it, so it raises the Z score (+7%) and lowers the W score (-9%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+7%); lowers the score of W (-9%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.568):

- `` — 13.0% of jets, value 2.37 (0.50…4.06), formula right 72%
- `` — 2.1% of jets, value 0.84 (0.00…1.88), formula right 64%
- `` — 21.9% of jets, value 0.70 (0.00…2.19), formula right 69%
- `` — 2.0% of jets, value 0.25 (0.00…0.88), formula right 74%
- `` — 2.7% of jets, value 0.19 (0.00…0.75), formula right 57%
- `` — 2.3% of jets, value 0.15 (0.00…0.62), formula right 55%
- `` — 13.6% of jets, value 0.02 (0.00…0.00), formula right 83%
- `` — 42.3% of jets, value 0.00 (0.00…0.00), formula right 57%

```
z = -0.959
if width < 0.0075: z += -1804 × (0.0075 − width)
if girth2 < 0.013: z += 656 × (0.013 − girth2)
if width < 0.0087: z += 1054 × (0.0087 − width)
if girth < 0.087: z += -89.24 × (0.087 − girth)
if mass_over_sum_pt_sq < 0.012: z += -394 × (0.012 − mass_over_sum_pt_sq)
if e2 < 0.038: z += 139 × (0.038 − e2)
if max_dr < 0.177: z += -21.65 × (0.177 − max_dr)
if e2_sq < 0.017: z += 111 × (0.017 − e2_sq)
if z_dr_0p05_0p1 < 0.588 and C2 < 0.067: z += -82.45 × (0.588 − z_dr_0p05_0p1) × (0.067 − C2)
if width < 0.0075 and n_dr_0p1_0p2 < 3.00: z += 135 × (0.0075 − width) × (3.00 − n_dr_0p1_0p2)
if lam1 > 0.0025: z += 298 × (lam1 − 0.0025)
if girth2 < 0.013 and n_dr_0p1_0p2 < 3.00: z += -49.10 × (0.013 − girth2) × (3.00 − n_dr_0p1_0p2)
if z_dr_0p05_0p1 < 0.588: z += 2.10 × (0.588 − z_dr_0p05_0p1)
if lam1 > 0.0042: z += -210 × (lam1 − 0.0042)
if mass < 76.66: z += -0.015 × (76.66 − mass)
if width < 0.0087 and D2 < 1.00: z += -1466 × (0.0087 − width) × (1.00 − D2)
if lam1 > 0.006: z += 205 × (lam1 − 0.006)
if e2 < 0.036: z += -35.91 × (0.036 − e2)
if n_dr_0p05_0p1 < 5.00: z += -0.140 × (5.00 − n_dr_0p05_0p1)
if lam1 > 0.0084: z += -224 × (lam1 − 0.0084)
if girth2 < 0.0044 and n_dr_0p2_0p4 < 1.00: z += -263 × (0.0044 − girth2) × (1.00 − n_dr_0p2_0p4)
if width < 0.0075 and planar_flow < 0.112: z += -6232 × (0.0075 − width) × (0.112 − planar_flow)
if planar_flow < 0.112 and centroid_offset > 0.0095: z += 1365 × (0.112 − planar_flow) × (centroid_offset − 0.0095)
if D2 < 1.12: z += -1.62 × (1.12 − D2)
if lam1 > 0.0073: z += -173 × (lam1 − 0.0073)
if girth2 < 0.013 and eccentricity > 0.970: z += 5885 × (0.013 − girth2) × (eccentricity − 0.970)
if z_dr_0p05_0p1 < 0.588 and n_dr_0p1_0p2 < 3.00: z += 0.307 × (0.588 − z_dr_0p05_0p1) × (3.00 − n_dr_0p1_0p2)
if width < 0.0075 and log_sum_pt < 6.80: z += 334 × (0.0075 − width) × (6.80 − log_sum_pt)
if planar_flow < 0.112 and max_dr < 0.160: z += -209 × (0.112 − planar_flow) × (0.160 − max_dr)
if planar_flow < 0.112: z += 6.08 × (0.112 − planar_flow)
if planar_flow < 0.112 and centroid_offset > 0.018: z += -1332 × (0.112 − planar_flow) × (centroid_offset − 0.018)
if D2 < 1.12 and centroid_offset < 0.031: z += 39.43 × (1.12 − D2) × (0.031 − centroid_offset)
if lam1 > 0.0054: z += -58.42 × (lam1 − 0.0054)
if girth2 < 0.013 and D2 < 1.00: z += 155 × (0.013 − girth2) × (1.00 − D2)
if D2 < 0.746: z += 1.50 × (0.746 − D2)
if girth2 < 0.0044: z += -78.68 × (0.0044 − girth2)
if z_dr_0p05_0p1 > 0.751: z += -5.30 × (z_dr_0p05_0p1 − 0.751)
if z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 1.00: z += 5.60 × (z_dr_0p05_0p1 − 0.751) × (1.00 − n_dr_0p2_0p4)
if lam1 > 0.0054 and max_dr < 0.160: z += 8853 × (lam1 − 0.0054) × (0.160 − max_dr)
if width < 0.0075 and D2 < 1.00: z += 529 × (0.0075 − width) × (1.00 − D2)
if e2 < 0.038 and D2 < 1.00: z += 199 × (0.038 − e2) × (1.00 − D2)
if mass < 76.66 and D2 < 0.746: z += -0.068 × (76.66 − mass) × (0.746 − D2)
if planar_flow < 0.112 and sum_pt < 740: z += -0.048 × (0.112 − planar_flow) × (740 − sum_pt)
if z_dr_0p05_0p1 < 0.588 and eccentricity > 0.970: z += 39.19 × (0.588 − z_dr_0p05_0p1) × (eccentricity − 0.970)
if lam1 > 0.0073 and max_dr < 0.133: z += 43515 × (lam1 − 0.0073) × (0.133 − max_dr)
if z_dr_0p05_0p1 > 0.751 and eccentricity > 0.984: z += 161 × (z_dr_0p05_0p1 − 0.751) × (eccentricity − 0.984)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.8% of jets, neuron 0.00, formula right for 62%.  
- **group 2** — 13.9% of jets, neuron 2.28, formula right for 71%.  
- **group 3** — 10.8% of jets, neuron 0.00, formula right for 50%.  
- **group 4** — 10.4% of jets, neuron 0.94, formula right for 63%.  
- **group 5** — 10.3% of jets, neuron 0.27, formula right for 54%.  
- **group 6** — 9.8% of jets, neuron 0.15, formula right for 74%.  
- **group 7** — 7.5% of jets, neuron 0.34, formula right for 77%.  
- **group 8** — 6.5% of jets, neuron 0.01, formula right for 86%.  
- **group 9** — 2.4% of jets, neuron 0.00, formula right for 77%.  
- **group 10** — 1.4% of jets, neuron 0.77, formula right for 64%.  

### neuron 1: moderate width at high pT (moderate)

- **What it measures:** Pushed down for narrow jets (width < 0.00868, its strongest term) and up for small e2_sq (< 0.00817) and for high total pT (log of total pT > 6.38); it follows lam1, width, girth2 and mass (rank correlations about +0.40 to +0.42). Z (0.75) and top (0.69) jets sit highest, gluons in the middle (0.53), W (0.30) and quarks (0.21) lowest; its clearest separation is that it is small for quarks.
- *computed — its value:* largest for Z (0.75), then t (0.69), then g (0.53), then W (0.30), then q (0.21); it separates q jets from the rest best (AUC 0.35: small for q)
- **How the class scores use it:** It raises the g score (+7%), where it marks the difference between gluons and the quarks that sit lowest on it, and it raises the Z score slightly (+2%). It does not (or hardly) enter the q, W or t scores; the Z and top jets that also sit high on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+7%), Z (+2%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.292):

- `` — 3.0% of jets, value 2.40 (0.38…4.75), formula right 72%
- `` — 14.6% of jets, value 1.06 (0.00…2.50), formula right 68%
- `` — 8.0% of jets, value 0.96 (0.00…2.25), formula right 79%
- `` — 2.0% of jets, value 0.50 (0.00…1.62), formula right 67%
- `` — 3.7% of jets, value 0.45 (0.00…1.50), formula right 43%
- `` — 36.4% of jets, value 0.42 (0.00…1.38), formula right 68%
- `` — 32.4% of jets, value 0.05 (0.00…0.00), formula right 60%

```
z = 0.820
if width < 0.0087: z += -472 × (0.0087 − width)
if e2_sq < 0.0082: z += 469 × (0.0082 − e2_sq)
if z_7 < 0.056: z += -82.42 × (0.056 − z_7)
if log_sum_pt > 6.38: z += 3.38 × (log_sum_pt − 6.38)
if log_sum_pt > 6.38 and centroid_offset < 0.024: z += -226 × (log_sum_pt − 6.38) × (0.024 − centroid_offset)
if log_sum_pt > 6.57 and lam2 < 0.0011: z += 6612 × (log_sum_pt − 6.57) × (0.0011 − lam2)
if C2 < 0.042: z += -30.05 × (0.042 − C2)
if lam1 < 0.0084: z += -140 × (0.0084 − lam1)
if log_sum_pt > 6.57: z += 6.56 × (log_sum_pt − 6.57)
if log_sum_pt > 6.57 and girth2_top3 < 0.0068: z += -1270 × (log_sum_pt − 6.57) × (0.0068 − girth2_top3)
if LHA < 0.347: z += -4.49 × (0.347 − LHA)
if mass_over_sum_pt > 0.055: z += 22.74 × (mass_over_sum_pt − 0.055)
if pt_7 > 34.53: z += 0.112 × (pt_7 − 34.53)
if z_7 < 0.056 and girth2_top3 < 0.0079: z += 7159 × (0.056 − z_7) × (0.0079 − girth2_top3)
if pt_7 > 34.53 and mass < 91.19: z += -0.0016 × (pt_7 − 34.53) × (91.19 − mass)
if girth < 0.076: z += 13.02 × (0.076 − girth)
if log_sum_pt > 6.38 and max_dr < 0.198: z += 16.94 × (log_sum_pt − 6.38) × (0.198 − max_dr)
if e2_sq < 0.0082 and planar_flow < 0.084: z += -6447 × (0.0082 − e2_sq) × (0.084 − planar_flow)
if mass < 53.33: z += -0.016 × (53.33 − mass)
if max_dr < 0.047: z += -57.71 × (0.047 − max_dr)
if e2 > 0.032: z += -33.08 × (e2 − 0.032)
if e2 < 0.036 and eccentricity > 0.978: z += 6639 × (0.036 − e2) × (eccentricity − 0.978)
if z_7 < 0.056 and C2 < 0.042: z += 817 × (0.056 − z_7) × (0.042 − C2)
if e2 > 0.032 and tau32 < 0.641: z += -76.18 × (e2 − 0.032) × (0.641 − tau32)
if tau21 < 0.198 and lam2 < 0.00019: z += 25721 × (0.198 − tau21) × (0.00019 − lam2)
if width < 0.0087 and planar_flow < 0.084: z += -2105 × (0.0087 − width) × (0.084 − planar_flow)
if lam1 < 0.0084 and centroid_offset > 0.021: z += 15642 × (0.0084 − lam1) × (centroid_offset − 0.021)
if tau21 < 0.198: z += -2.79 × (0.198 − tau21)
if lam1 < 0.0084 and n_pt_above_50 > 6.00: z += -74.70 × (0.0084 − lam1) × (n_pt_above_50 − 6.00)
if LHA < 0.347 and planar_flow < 0.084: z += -77.50 × (0.347 − LHA) × (0.084 − planar_flow)
if e2_sq < 0.0082 and n_pt_above_50 > 6.00: z += 74.84 × (0.0082 − e2_sq) × (n_pt_above_50 − 6.00)
if planar_flow < 0.084: z += 3.68 × (0.084 − planar_flow)
if tau21 < 0.198 and pt_6 < 50.25: z += -0.204 × (0.198 − tau21) × (50.25 − pt_6)
if e2 < 0.036: z += 4.56 × (0.036 − e2)
if log_sum_pt > 6.57 and n_pt_above_50 > 7.00: z += -6.98 × (log_sum_pt − 6.57) × (n_pt_above_50 − 7.00)
if pt_7 > 53.44 and mass_top3 < 45.59: z += 0.0043 × (pt_7 − 53.44) × (45.59 − mass_top3)
if log_sum_pt > 6.57 and planar_flow < 0.033: z += 101 × (log_sum_pt − 6.57) × (0.033 − planar_flow)
if pt_7 > 34.53 and centroid_offset > 0.014: z += 1.32 × (pt_7 − 34.53) × (centroid_offset − 0.014)
if pt_7 > 53.44 and mass_top3 < 32.62: z += -0.0041 × (pt_7 − 53.44) × (32.62 − mass_top3)
if pt_7 > 34.53 and max_dr < 0.081: z += 0.328 × (pt_7 − 34.53) × (0.081 − max_dr)
if planar_flow < 0.084 and tau32 < 0.269: z += -366 × (0.084 − planar_flow) × (0.269 − tau32)
if pt_7 > 53.44 and tau21 < 0.553: z += -0.141 × (pt_7 − 53.44) × (0.553 − tau21)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 17.1% of jets, neuron 0.45, formula right for 62%.  
- **group 2** — 17.0% of jets, neuron 0.16, formula right for 52%.  
- **group 3** — 16.8% of jets, neuron 0.98, formula right for 71%.  
- **group 4** — 9.7% of jets, neuron 0.18, formula right for 56%.  
- **group 5** — 8.7% of jets, neuron 1.01, formula right for 80%.  
- **group 6** — 8.5% of jets, neuron 0.47, formula right for 87%.  
- **group 7** — 7.9% of jets, neuron 0.16, formula right for 73%.  
- **group 8** — 5.3% of jets, neuron 0.67, formula right for 67%.  
- **group 9** — 4.9% of jets, neuron 0.59, formula right for 56%.  
- **group 10** — 4.1% of jets, neuron 0.18, formula right for 60%.  

### neuron 3: radiation at wide angle (ΔR 0.2-0.4) (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and girth2 < 0.0132 are its two strongest terms, both pushing down) and pushed up for e2 < 0.0445; it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.613 and 0.611). Tops sit clearly highest (1.88), then gluons (0.66) and quarks (0.36); Z jets are rarely on it (0.14) and W jets almost never (0.01, zero for 98.8% of them).
- *computed — its value:* largest for t (1.88), then g (0.66), then q (0.36), then Z (0.14), then W (0.01); it separates t jets from the rest best (AUC 0.69: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so it lowers the W score (-8%) and the Z score (-12%). It does not (or hardly) enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-8%), Z (-12%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.576):

- `` — 2.0% of jets, value 6.58 (1.75…11.00), formula right 63%
- `` — 9.2% of jets, value 3.29 (0.00…6.62), formula right 72%
- `` — 2.1% of jets, value 2.49 (0.12…6.00), formula right 47%
- `` — 2.5% of jets, value 2.30 (0.00…5.88), formula right 76%
- `` — 6.8% of jets, value 0.47 (0.00…1.62), formula right 62%
- `` — 6.3% of jets, value 0.28 (0.00…0.75), formula right 91%
- `` — 71.1% of jets, value 0.02 (0.00…0.00), formula right 63%

```
z = 3.06
if girth2 < 0.0087: z += -1247 × (0.0087 − girth2)
if girth2 < 0.013: z += -468 × (0.013 − girth2)
if e2 < 0.044: z += 156 × (0.044 − e2)
if lam1 < 0.0073: z += 347 × (0.0073 − lam1)
if girth2 < 0.0044: z += 652 × (0.0044 − girth2)
if mass_over_sum_pt > 0.068 and n_dr_0_0p05 < 5.00: z += 13.98 × (mass_over_sum_pt − 0.068) × (5.00 − n_dr_0_0p05)
if mass_over_sum_pt > 0.068: z += -57.22 × (mass_over_sum_pt − 0.068)
if e2 > 0.029: z += -58.27 × (e2 − 0.029)
if mass > 36.23 and eccentricity > 0.621: z += 0.126 × (mass − 36.23) × (eccentricity − 0.621)
if e2 < 0.044 and z_dr_0p05_0p1 < 0.964: z += -30.43 × (0.044 − e2) × (0.964 − z_dr_0p05_0p1)
if centroid_offset > 0.014: z += 69.27 × (centroid_offset − 0.014)
if max_dr > 0.145: z += 15.92 × (max_dr − 0.145)
if lam1 > 0.012: z += -329 × (lam1 − 0.012)
if mass > 36.23: z += -0.027 × (mass − 36.23)
if mass_over_sum_pt > 0.090: z += 41.24 × (mass_over_sum_pt − 0.090)
if mass_over_sum_pt > 0.068 and tau32 < 0.519: z += -120 × (mass_over_sum_pt − 0.068) × (0.519 − tau32)
if mass > 36.23 and lam2 < 0.0011: z += 27.90 × (mass − 36.23) × (0.0011 − lam2)
if width < 0.0067: z += 83.44 × (0.0067 − width)
if LHA > 0.313 and eccentricity > 0.960: z += 1277 × (LHA − 0.313) × (eccentricity − 0.960)
if mass_over_sum_pt > 0.090 and pt_6 < 56.53: z += -1.59 × (mass_over_sum_pt − 0.090) × (56.53 − pt_6)
if LHA > 0.313: z += 14.39 × (LHA − 0.313)
if mass_over_sum_pt > 0.108: z += -40.62 × (mass_over_sum_pt − 0.108)
if e2 > 0.029 and eccentricity > 0.960: z += -1373 × (e2 − 0.029) × (eccentricity − 0.960)
if mass_top5 > 53.61: z += -0.085 × (mass_top5 − 53.61)
if LHA > 0.313 and mass_top3 < 50.35: z += -0.562 × (LHA − 0.313) × (50.35 − mass_top3)
if mass > 69.61: z += -0.066 × (mass − 69.61)
if e2 < 0.044 and z_dr_0p1_0p2 > 0: z += -240 × (0.044 − e2) × (z_dr_0p1_0p2 − 0)
if lam1 > 0.012 and pt_6 < 38.25: z += 28.53 × (lam1 − 0.012) × (38.25 − pt_6)
if LHA > 0.313 and max_dr < 0.145: z += -2554 × (LHA − 0.313) × (0.145 − max_dr)
if mass_over_sum_pt > 0.090 and pt_7 < 48.72: z += 0.898 × (mass_over_sum_pt − 0.090) × (48.72 − pt_7)
if LHA > 0.326: z += 8.11 × (LHA − 0.326)
if mass_over_sum_pt > 0.068 and dr_7 < 0.042: z += -9052 × (mass_over_sum_pt − 0.068) × (0.042 − dr_7)
if LHA > 0.313 and planar_flow > 0.008: z += -17.71 × (LHA − 0.313) × (planar_flow − 0.008)
if width > 0.019: z += 115 × (width − 0.019)
if lam1 > 0.016 and eccentricity > 0.960: z += -9652 × (lam1 − 0.016) × (eccentricity − 0.960)
if max_dr > 0.145 and dr_3 < 0.046: z += -536 × (max_dr − 0.145) × (0.046 − dr_3)
if mean_eta < -0.013 and pt_4 < 68.12: z += -2.08 × (-0.013 − mean_eta) × (68.12 − pt_4)
if width > 0.019 and pt_dispersion < 0.492: z += 996 × (width − 0.019) × (0.492 − pt_dispersion)
if e2 > 0.029 and n_pt_above_50 > 3.00: z += -3.76 × (e2 − 0.029) × (n_pt_above_50 − 3.00)
if mean_eta < -0.013 and z_4 < 0.120: z += 1026 × (-0.013 − mean_eta) × (0.120 − z_4)
if mass > 69.61 and max_dr < 0.177: z += 3.38 × (mass − 69.61) × (0.177 − max_dr)
if centroid_offset > 0.014 and eta_0 < -0.040: z += -409 × (centroid_offset − 0.014) × (-0.040 − eta_0)
if mass > 36.23 and dr_6 < 0.046: z += -1.68 × (mass − 36.23) × (0.046 − dr_6)
if mean_eta < -0.013: z += 17.78 × (-0.013 − mean_eta)
if mass_over_sum_pt > 0.090 and max_dr < 0.145: z += 11114 × (mass_over_sum_pt − 0.090) × (0.145 − max_dr)
if LHA > 0.424: z += -25.17 × (LHA − 0.424)
if centroid_offset > 0.014 and phi_7 > -0.042: z += 62.41 × (centroid_offset − 0.014) × (phi_7 − -0.042)
if mass > 36.23 and pt_7 < 20.12: z += -0.0058 × (mass − 36.23) × (20.12 − pt_7)
if LHA > 0.313 and pt_5 < 29.88: z += 4.88 × (LHA − 0.313) × (29.88 − pt_5)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.7% of jets, neuron 0.00, formula right for 60%.  
- **group 2** — 16.4% of jets, neuron 0.18, formula right for 70%.  
- **group 3** — 13.7% of jets, neuron 0.25, formula right for 61%.  
- **group 4** — 11.3% of jets, neuron 0.11, formula right for 50%.  
- **group 5** — 9.3% of jets, neuron 0.65, formula right for 74%.  
- **group 6** — 6.7% of jets, neuron 2.98, formula right for 80%.  
- **group 7** — 4.6% of jets, neuron 0.45, formula right for 91%.  
- **group 8** — 4.5% of jets, neuron 5.01, formula right for 70%.  
- **group 9** — 1.6% of jets, neuron 1.75, formula right for 68%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 88%.  

### neuron 4: low C2, not too spread out (moderate)

- **What it measures:** Pushed up for width > 0.00165 and small e2_sq (< 0.0172), and down for light jets (mass < 69.6 GeV), with a mass/pT window (up above 0.108, down above 0.0904); it falls with C2, D2 and the largest particle distance (rank correlations -0.502, -0.379, -0.376). It is on for almost all g, q, W and Z jets (Z highest at 4.68, then g 4.19, W 4.00, q 3.73) but zero for 36% of tops, which sit lowest (3.05).
- *computed — its value:* largest for Z (4.68), then g (4.19), then W (4.00), then q (3.73), then t (3.05); it separates t jets from the rest best (AUC 0.35: small for t)
- **How the class scores use it:** It raises the Z score (+11%) and the t score (+10%), lowers the q score (-16%) and the g score (-4%), and does not enter the W score. The Z score reads it as mildly boson-like, since Z jets sit highest on it; the t score adds it although tops sit lowest, so that use is a correction next to the much larger size inputs of the t score (neurons 13 and 10) rather than a sign of a top.
- *computed — used by:* raises the score of Z (+11%), t (+10%); lowers the score of g (-4%), q (-16%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.486):

- `` — 18.1% of jets, value 6.22 (3.75…8.50), formula right 76%
- `` — 8.2% of jets, value 5.69 (3.50…8.00), formula right 57%
- `` — 56.0% of jets, value 3.75 (1.75…5.00), formula right 60%
- `` — 3.0% of jets, value 3.09 (0.00…6.25), formula right 60%
- `` — 3.1% of jets, value 3.06 (0.00…7.25), formula right 74%
- `` — 2.2% of jets, value 1.04 (0.00…2.75), formula right 61%
- `` — 2.0% of jets, value 0.87 (0.00…3.00), formula right 69%
- `` — 7.4% of jets, value 0.17 (0.00…0.00), formula right 87%

```
z = -7.01
if width > 0.0017: z += 1001 × (width − 0.0017)
if e2_sq < 0.017: z += 402 × (0.017 − e2_sq)
if width > 0.00032: z += -494 × (width − 0.00032)
if mass < 69.61: z += -0.074 × (69.61 − mass)
if mass_over_sum_pt > 0.108: z += 371 × (mass_over_sum_pt − 0.108)
if e2_sq < 0.024: z += 106 × (0.024 − e2_sq)
if mass_over_sum_pt > 0.090: z += -231 × (mass_over_sum_pt − 0.090)
if e2 > 0.0071: z += 74.28 × (e2 − 0.0071)
if mass < 56.92: z += 0.076 × (56.92 − mass)
if tau21 < 0.238: z += 27.54 × (0.238 − tau21)
if mass_over_sum_pt > 0.108 and D2 < 3.89: z += -63.08 × (mass_over_sum_pt − 0.108) × (3.89 − D2)
if girth2_top3 < 0.005: z += 404 × (0.005 − girth2_top3)
if lam2 < 0.00031: z += -4855 × (0.00031 − lam2)
if C2 > 0.015: z += -49.02 × (C2 − 0.015)
if sum_pt < 764: z += 0.008 × (764 − sum_pt)
if tau21 < 0.238 and lam2 < 0.0011: z += -11762 × (0.238 − tau21) × (0.0011 − lam2)
if width > 0.00032 and C2 < 0.067: z += 3999 × (width − 0.00032) × (0.067 − C2)
if max_dr > 0.103: z += 14.39 × (max_dr − 0.103)
if lam2 < 0.00031 and mass_top3 < 50.35: z += -70.33 × (0.00031 − lam2) × (50.35 − mass_top3)
if centroid_offset < 0.014: z += 104 × (0.014 − centroid_offset)
if C2 > 0.067: z += -148 × (C2 − 0.067)
if girth < 0.048: z += 33.46 × (0.048 − girth)
if mass < 69.61 and dr_3 < 0.104: z += 0.165 × (69.61 − mass) × (0.104 − dr_3)
if lam2 < 0.0011: z += -328 × (0.0011 − lam2)
if C2 > 0.015 and pt_7 > 38.53: z += 12.52 × (C2 − 0.015) × (pt_7 − 38.53)
if centroid_offset < 0.014 and z_dr_0p05_0p1 < 0.675: z += -140 × (0.014 − centroid_offset) × (0.675 − z_dr_0p05_0p1)
if mass_over_sum_pt_sq < 0.0011: z += 966 × (0.0011 − mass_over_sum_pt_sq)
if tau21 < 0.238 and mass < 62.55: z += -0.651 × (0.238 − tau21) × (62.55 − mass)
if max_dr > 0.198: z += -22.67 × (max_dr − 0.198)
if girth2 > 0.013: z += -190 × (girth2 − 0.013)
if tau21 < 0.238 and pt_7 > 33.22: z += 0.838 × (0.238 − tau21) × (pt_7 − 33.22)
if max_dr > 0.103 and pt_7 > 37.16: z += -2.66 × (max_dr − 0.103) × (pt_7 − 37.16)
if girth2_top2 < 0.0095 and centroid_offset > 0.016: z += 10970 × (0.0095 − girth2_top2) × (centroid_offset − 0.016)
if e2 > 0.063: z += -120 × (e2 − 0.063)
if sum_pt < 764 and n_dr_0p2_0p4 < 1.00: z += -0.0028 × (764 − sum_pt) × (1.00 − n_dr_0p2_0p4)
if sum_pt_top5 < 431: z += 0.012 × (431 − sum_pt_top5)
if tau21 < 0.238 and e2_sq > 0.012: z += -1813 × (0.238 − tau21) × (e2_sq − 0.012)
if width > 0.00032 and z_dr_0p2_0p4 < 0.056: z += -732 × (width − 0.00032) × (0.056 − z_dr_0p2_0p4)
if tau21 < 0.238 and dr_7 < 0.175: z += 22.81 × (0.238 − tau21) × (0.175 − dr_7)
if lam2 < 0.0011 and n_dr_0p1_0p2 > 2.00: z += -252 × (0.0011 − lam2) × (n_dr_0p1_0p2 − 2.00)
if tau21 < 0.238 and pt_2 > 73.69: z += 0.037 × (0.238 − tau21) × (pt_2 − 73.69)
if tau21 < 0.238 and planar_flow > 0.045: z += 26.04 × (0.238 − tau21) × (planar_flow − 0.045)
if girth > 0.102: z += 7.62 × (girth − 0.102)
if tau21 < 0.238 and mean_phi < -0.026: z += -452 × (0.238 − tau21) × (-0.026 − mean_phi)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 35.9% of jets, neuron 4.19, formula right for 59%.  
- **group 2** — 22.0% of jets, neuron 5.71, formula right for 76%.  
- **group 3** — 15.3% of jets, neuron 2.52, formula right for 53%.  
- **group 4** — 9.2% of jets, neuron 4.19, formula right for 60%.  
- **group 5** — 4.8% of jets, neuron 4.84, formula right for 70%.  
- **group 6** — 4.1% of jets, neuron 1.91, formula right for 80%.  
- **group 7** — 3.1% of jets, neuron 0.71, formula right for 87%.  
- **group 8** — 2.9% of jets, neuron 0.23, formula right for 94%.  
- **group 9** — 1.8% of jets, neuron 2.94, formula right for 73%.  
- **group 10** — 0.8% of jets, neuron 0.81, formula right for 71%.  

### neuron 5: quark-likeness: pT in few particles (moderate)

- **What it measures:** Pushed up when the 8th-hardest particle carries little of the pT (z_7 < 0.0715, its strongest term, partly cancelled when the pT centroid is close to the axis) and for small e2 (< 0.0356); it falls with e2 and z_7 and rises with the summed pT of the 3 hardest particles (rank correlations -0.628, -0.576, 0.589). Quarks sit far highest (5.03); Z (2.18), W (2.13) and gluons (1.84) are similar, and tops lowest (0.90).
- *computed — its value:* largest for q (5.03), then Z (2.18), then W (2.13), then g (1.84), then t (0.90); it separates q jets from the rest best (AUC 0.77: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops well below them, it raises the q score (+5%) and lowers the g score (-16%) and the t score (-13%). It does not (or hardly) enter the W or Z scores.
- *computed — used by:* raises the score of q (+5%); lowers the score of g (-16%), t (-13%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.701):

- `` — 8.8% of jets, value 8.62 (5.50…11.88), formula right 68%
- `` — 2.0% of jets, value 5.16 (0.00…10.75), formula right 69%
- `` — 4.2% of jets, value 4.83 (2.12…7.38), formula right 65%
- `` — 17.2% of jets, value 3.84 (1.12…6.38), formula right 56%
- `` — 3.1% of jets, value 1.97 (0.00…4.00), formula right 80%
- `` — 16.6% of jets, value 1.54 (0.00…3.50), formula right 54%
- `` — 11.3% of jets, value 1.33 (0.00…2.88), formula right 60%
- `` — 36.7% of jets, value 0.54 (0.00…1.25), formula right 75%

```
z = 0.388
if z_7 < 0.071: z += 72.42 × (0.071 − z_7)
if z_7 < 0.071 and centroid_offset < 0.031: z += -3192 × (0.071 − z_7) × (0.031 − centroid_offset)
if width < 0.0026 and centroid_offset < 0.024: z += 82594 × (0.0026 − width) × (0.024 − centroid_offset)
if log_sum_pt > 6.57 and lam1 < 0.012: z += -1121 × (log_sum_pt − 6.57) × (0.012 − lam1)
if e2 < 0.036: z += 51.87 × (0.036 − e2)
if dr_0 < 0.022: z += -175 × (0.022 − dr_0)
if mean_phi2 < 0.014 and max_pair_mass < 40.05: z += 1.79 × (0.014 − mean_phi2) × (40.05 − max_pair_mass)
if sum_pt_top2 < 548 and girth2_top3 < 0.004: z += -2.15 × (548 − sum_pt_top2) × (0.004 − girth2_top3)
if sum_pt > 869: z += -0.029 × (sum_pt − 869)
if LHA < 0.216: z += 15.33 × (0.216 − LHA)
if dr_0 < 0.026: z += -94.60 × (0.026 − dr_0)
if mean_phi2 < 0.014: z += 41.03 × (0.014 − mean_phi2)
if girth2_top5 < 0.0023: z += -575 × (0.0023 − girth2_top5)
if LHA < 0.216 and log_sum_pt < 6.80: z += -107 × (0.216 − LHA) × (6.80 − log_sum_pt)
if e2_sq < 0.0053: z += 192 × (0.0053 − e2_sq)
if mass_over_sum_pt < 0.085 and max_pair_mass < 18.10: z += 0.776 × (0.085 − mass_over_sum_pt) × (18.10 − max_pair_mass)
if sum_pt_top2 < 548: z += -0.0019 × (548 − sum_pt_top2)
if log_sum_pt > 6.57 and dr_0 < 0.022: z += 535 × (log_sum_pt − 6.57) × (0.022 − dr_0)
if z_7 < 0.049 and mass_top5 < 62.55: z += 1.08 × (0.049 − z_7) × (62.55 − mass_top5)
if width < 0.0026: z += 415 × (0.0026 − width)
if z_7 < 0.071 and lam1 < 0.0015: z += -26297 × (0.071 − z_7) × (0.0015 − lam1)
if sum_pt_top5 > 752: z += 0.014 × (sum_pt_top5 − 752)
if log_sum_pt > 6.57 and girth2_top2 < 0.0063: z += 674 × (log_sum_pt − 6.57) × (0.0063 − girth2_top2)
if mass_over_sum_pt < 0.085: z += -8.10 × (0.085 − mass_over_sum_pt)
if z_7 < 0.071 and lam2 < 0.0011: z += 11840 × (0.071 − z_7) × (0.0011 − lam2)
if sum_pt > 869 and centroid_offset < 0.013: z += 1.74 × (sum_pt − 869) × (0.013 − centroid_offset)
if e2 < 0.036 and lam2 < 7.3e-05: z += 417996 × (0.036 − e2) × (7.3e-05 − lam2)
if z_7 < 0.071 and sum_pt < 788: z += -0.242 × (0.071 − z_7) × (788 − sum_pt)
if z_7 < 0.032: z += 90.22 × (0.032 − z_7)
if sum_pt_top2 < 548 and dr_0 < 0.022: z += 0.409 × (548 − sum_pt_top2) × (0.022 − dr_0)
if z_7 < 0.071 and mean_phi2 < 0.0043: z += -2586 × (0.071 − z_7) × (0.0043 − mean_phi2)
if z_7 < 0.049: z += 17.35 × (0.049 − z_7)
if z_7 < 0.071 and n_dr_0p2_0p4 < 1.00: z += 6.93 × (0.071 − z_7) × (1.00 − n_dr_0p2_0p4)
if log_sum_pt > 6.90 and dr_0 < 0.041: z += -1044 × (log_sum_pt − 6.90) × (0.041 − dr_0)
if log_sum_pt > 6.90: z += -27.83 × (log_sum_pt − 6.90)
if z_6 < 0.029: z += 111 × (0.029 − z_6)
if log_sum_pt > 6.57 and mean_phi2 < 0.00015: z += 16793 × (log_sum_pt − 6.57) × (0.00015 − mean_phi2)
if e2_sq < 0.0053 and centroid_offset < 0.014: z += -5279 × (0.0053 − e2_sq) × (0.014 − centroid_offset)
if log_sum_pt > 6.57 and mean_eta2 < 9e-05: z += 31034 × (log_sum_pt − 6.57) × (9e-05 − mean_eta2)
if log_sum_pt > 6.90 and centroid_offset < 0.018: z += -968 × (log_sum_pt − 6.90) × (0.018 − centroid_offset)
if e2_sq < 0.0053 and n_pt_above_50 > 6.00: z += -57.11 × (0.0053 − e2_sq) × (n_pt_above_50 − 6.00)
if mass_over_sum_pt < 0.085 and mass_top3 > 28.35: z += -3.77 × (0.085 − mass_over_sum_pt) × (mass_top3 − 28.35)
if LHA < 0.216 and planar_flow < 0.084: z += 450 × (0.216 − LHA) × (0.084 − planar_flow)
if z_7 < 0.032 and pt_5 > 33.03: z += 2.62 × (0.032 − z_7) × (pt_5 − 33.03)
if LHA < 0.216 and mass_top3 > 3.56: z += -1.64 × (0.216 − LHA) × (mass_top3 − 3.56)
if girth2 < 4.8e-05: z += 29633 × (4.8e-05 − girth2)
if pt_5 < 24.58: z += -0.071 × (24.58 − pt_5)
if log_sum_pt > 6.57 and centroid_offset > 0.018: z += -192 × (log_sum_pt − 6.57) × (centroid_offset − 0.018)
if LHA < 0.216 and n_dr_0p2_0p4 > 0: z += -7.86 × (0.216 − LHA) × (n_dr_0p2_0p4 − 0)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.2% of jets, neuron 0.61, formula right for 69%.  
- **group 2** — 20.7% of jets, neuron 1.57, formula right for 70%.  
- **group 3** — 11.9% of jets, neuron 3.20, formula right for 69%.  
- **group 4** — 11.3% of jets, neuron 2.21, formula right for 48%.  
- **group 5** — 8.6% of jets, neuron 5.52, formula right for 61%.  
- **group 6** — 8.1% of jets, neuron 1.30, formula right for 60%.  
- **group 7** — 5.6% of jets, neuron 9.54, formula right for 76%.  
- **group 8** — 3.9% of jets, neuron 3.30, formula right for 67%.  
- **group 9** — 1.4% of jets, neuron 1.56, formula right for 66%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 71%.  

### neuron 8: narrow core, nothing at wide angle (moderate)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00502, its strongest term), more so with little pT at 0.2 ≤ ΔR < 0.4 (below 0.101), and down for the very narrowest jets (girth < 0.0611) and for light, centred jets (mass < 29.6 GeV with centroid offset < 0.0236); it falls with width, girth and max ΔR (rank correlations -0.519, -0.497, -0.491). Quarks (1.51) and gluons (1.22) sit highest; W (0.44), Z (0.24) and tops (0.14) low.
- *computed — its value:* largest for q (1.51), then g (1.22), then W (0.44), then Z (0.24), then t (0.14); it separates q jets from the rest best (AUC 0.73: large for q)
- **How the class scores use it:** A single narrow core is evidence against a two-prong W, so it lowers the W score (-4%); it also raises the t score slightly (+3%), a small correction since tops sit lowest on it. Although quarks and gluons sit highest on it, it does not (or hardly) enter the g, q or Z scores.
- *computed — used by:* raises the score of t (+3%); lowers the score of W (-4%); does not (or hardly) enter the score of g, q, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.623):

- `` — 10.4% of jets, value 3.53 (1.12…5.88), formula right 59%
- `` — 18.4% of jets, value 1.08 (0.00…2.38), formula right 65%
- `` — 3.7% of jets, value 1.03 (0.00…2.88), formula right 49%
- `` — 6.0% of jets, value 0.72 (0.00…2.50), formula right 47%
- `` — 16.4% of jets, value 0.24 (0.00…0.75), formula right 73%
- `` — 7.2% of jets, value 0.16 (0.00…0.62), formula right 61%
- `` — 5.2% of jets, value 0.06 (0.00…0.00), formula right 45%
- `` — 32.8% of jets, value 0.02 (0.00…0.00), formula right 74%

```
z = -0.433
if width < 0.005: z += 1307 × (0.005 − width)
if girth < 0.061 and width < 0.005: z += -18206 × (0.061 − girth) × (0.005 − width)
if width < 0.005 and z_dr_0p2_0p4 < 0.101: z += 6301 × (0.005 − width) × (0.101 − z_dr_0p2_0p4)
if girth2 < 0.0067 and centroid_offset < 0.024: z += 26071 × (0.0067 − girth2) × (0.024 − centroid_offset)
if mass < 29.64 and centroid_offset < 0.024: z += -8.38 × (29.64 − mass) × (0.024 − centroid_offset)
if girth < 0.061: z += -42.32 × (0.061 − girth)
if e2 < 0.025 and centroid_offset < 0.031: z += 4741 × (0.025 − e2) × (0.031 − centroid_offset)
if LHA < 0.197 and lam2 < 0.00031: z += -86292 × (0.197 − LHA) × (0.00031 − lam2)
if mass < 21.78: z += -0.129 × (21.78 − mass)
if girth < 0.061 and lam2 < 0.00019: z += 181989 × (0.061 − girth) × (0.00019 − lam2)
if z_dr_0_0p05 > 0.848 and lam2 < 0.00054: z += -18789 × (z_dr_0_0p05 − 0.848) × (0.00054 − lam2)
if max_dr < 0.177: z += 6.72 × (0.177 − max_dr)
if e2 < 0.017: z += 117 × (0.017 − e2)
if max_dr < 0.177 and lam2 < 0.00019: z += 44377 × (0.177 − max_dr) × (0.00019 − lam2)
if width < 0.005 and centroid_offset > 0.0068: z += -39510 × (0.005 − width) × (centroid_offset − 0.0068)
if LHA < 0.197: z += -16.74 × (0.197 − LHA)
if log_sum_pt > 6.70 and z_dr_0p2_0p4 < 0.206: z += -46.00 × (log_sum_pt − 6.70) × (0.206 − z_dr_0p2_0p4)
if LHA < 0.197 and width < 0.00056: z += 31132 × (0.197 − LHA) × (0.00056 − width)
if log_sum_pt > 6.70: z += -8.31 × (log_sum_pt − 6.70)
if e2 < 0.025: z += 38.39 × (0.025 − e2)
if girth2_top5 < 0.00022: z += -9548 × (0.00022 − girth2_top5)
if pt_7 > 34.53: z += -0.061 × (pt_7 − 34.53)
if girth2 < 0.0067 and planar_flow < 0.401: z += 675 × (0.0067 − girth2) × (0.401 − planar_flow)
if girth2 < 0.00096: z += 1081 × (0.00096 − girth2)
if girth < 0.061 and centroid_offset > 0.0068: z += -2525 × (0.061 − girth) × (centroid_offset − 0.0068)
if log_sum_pt > 6.70 and centroid_offset < 0.024: z += 343 × (log_sum_pt − 6.70) × (0.024 − centroid_offset)
if mass < 15.45 and lam2 < 0.00031: z += -300 × (15.45 − mass) × (0.00031 − lam2)
if mass < 21.78 and centroid_offset > 0.016: z += -13.79 × (21.78 − mass) × (centroid_offset − 0.016)
if z_dr_0_0p05 > 0.848: z += -3.61 × (z_dr_0_0p05 − 0.848)
if girth < 0.061 and z_dr_0p2_0p4 < 0.056: z += -175 × (0.061 − girth) × (0.056 − z_dr_0p2_0p4)
if mass < 21.78 and centroid_offset < 0.027: z += -2.27 × (21.78 − mass) × (0.027 − centroid_offset)
if girth < 0.061 and lam1 > 0.00028: z += -15654 × (0.061 − girth) × (lam1 − 0.00028)
if log_sum_pt > 6.70 and pt_7 < 48.72: z += 0.203 × (log_sum_pt − 6.70) × (48.72 − pt_7)
if LHA < 0.197 and z_7 > 0.017: z += 259 × (0.197 − LHA) × (z_7 − 0.017)
if girth2 < 0.0067 and tau21 < 0.501: z += -503 × (0.0067 − girth2) × (0.501 − tau21)
if e2 < 0.025 and sum_pt > 788: z += 0.258 × (0.025 − e2) × (sum_pt − 788)
if girth2 < 0.0067 and centroid_offset > 0.018: z += -20907 × (0.0067 − girth2) × (centroid_offset − 0.018)
if log_sum_pt > 6.70 and girth2 < 0.019: z += -203 × (log_sum_pt − 6.70) × (0.019 − girth2)
if girth2 < 0.0067 and n_dr_0p05_0p1 > 0: z += -50.39 × (0.0067 − girth2) × (n_dr_0p05_0p1 − 0)
if mass < 8.38: z += 0.200 × (8.38 − mass)
if width < 0.005 and mass_over_sum_pt_sq > 0.00012: z += -100211 × (0.005 − width) × (mass_over_sum_pt_sq − 0.00012)
if LHA < 0.197 and z_6 > 0.022: z += -132 × (0.197 − LHA) × (z_6 − 0.022)
if mass < 15.45 and z_7 < 0.059: z += -2.22 × (15.45 − mass) × (0.059 − z_7)
if centroid_offset < 0.0033: z += 373 × (0.0033 − centroid_offset)
if log_sum_pt > 6.70 and mass_over_sum_pt_sq < 0.00024: z += 23925 × (log_sum_pt − 6.70) × (0.00024 − mass_over_sum_pt_sq)
if sum_pt_top5 > 658: z += -0.0016 × (sum_pt_top5 − 658)
if width < 0.005 and sum_pt_top2 < 248: z += 4.55 × (0.005 − width) × (248 − sum_pt_top2)
if LHA < 0.197 and mean_phi > -0.00086: z += -783 × (0.197 − LHA) × (mean_phi − -0.00086)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 48.7% of jets, neuron 0.11, formula right for 72%.  
- **group 2** — 9.5% of jets, neuron 0.80, formula right for 52%.  
- **group 3** — 8.6% of jets, neuron 1.12, formula right for 61%.  
- **group 4** — 8.3% of jets, neuron 2.56, formula right for 59%.  
- **group 5** — 6.3% of jets, neuron 0.91, formula right for 76%.  
- **group 6** — 5.9% of jets, neuron 3.32, formula right for 55%.  
- **group 7** — 4.8% of jets, neuron 0.05, formula right for 76%.  
- **group 8** — 4.5% of jets, neuron 0.23, formula right for 45%.  
- **group 9** — 2.8% of jets, neuron 0.00, formula right for 44%.  
- **group 10** — 0.5% of jets, neuron 0.00, formula right for 52%.  

### neuron 15: slightly wider than a W (moderate)

- **What it measures:** Pushed up for width < 0.0132 but switched down for narrow jets (width < 0.00668), and pushed up for small e2 (< 0.0411), so it responds to jets just wider than the typical W but not broad; it follows lam1 and width (rank correlations 0.417 and 0.415) and falls with τ21 (-0.401). Z jets sit highest (1.50), then tops (0.88), with gluons (0.36), quarks (0.23) and W (0.20) low.
- *computed — its value:* largest for Z (1.50), then t (0.88), then g (0.36), then q (0.23), then W (0.20); it separates Z jets from the rest best (AUC 0.73: large for Z)
- **How the class scores use it:** W jets sit lowest on it, so the W score reads it as evidence against a W: it lowers the W score (-11%). It also lowers the Z score, but with a smaller share (-4%), even though Z jets sit highest; it does not (or hardly) enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-11%), Z (-4%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.627):

- `` — 13.9% of jets, value 2.79 (0.88…4.50), formula right 74%
- `` — 6.7% of jets, value 1.52 (0.00…2.88), formula right 65%
- `` — 5.3% of jets, value 1.02 (0.00…2.73), formula right 61%
- `` — 2.1% of jets, value 0.81 (0.00…1.88), formula right 83%
- `` — 19.2% of jets, value 0.32 (0.00…1.25), formula right 68%
- `` — 5.5% of jets, value 0.18 (0.00…0.75), formula right 48%
- `` — 10.5% of jets, value 0.05 (0.00…0.00), formula right 82%
- `` — 36.8% of jets, value 0.00 (0.00…0.00), formula right 59%

```
z = -1.96
if width < 0.013: z += 518 × (0.013 − width)
if width < 0.0067: z += -1319 × (0.0067 − width)
if e2 < 0.041: z += 126 × (0.041 − e2)
if lam1 < 0.0084: z += -513 × (0.0084 − lam1)
if lam1 < 0.0065: z += 709 × (0.0065 − lam1)
if mass_over_sum_pt_sq < 0.0072: z += 556 × (0.0072 − mass_over_sum_pt_sq)
if e2_sq < 0.0082: z += -450 × (0.0082 − e2_sq)
if e2 < 0.025: z += 251 × (0.025 − e2)
if width < 0.0075: z += -516 × (0.0075 − width)
if girth2 < 0.019: z += 133 × (0.019 − girth2)
if e2_sq < 0.012: z += -226 × (0.012 − e2_sq)
if lam1 < 0.012: z += -199 × (0.012 − lam1)
if girth > 0.034: z += 31.87 × (girth − 0.034)
if z_dr_0p1_0p2 < 0.328: z += 3.98 × (0.328 − z_dr_0p1_0p2)
if lam2 < 0.0011: z += -606 × (0.0011 − lam2)
if LHA > 0.347: z += -61.87 × (LHA − 0.347)
if mass_over_sum_pt < 0.068: z += -23.05 × (0.068 − mass_over_sum_pt)
if tau21 < 0.238 and z_dr_0p2_0p4 < 0.206: z += 48.39 × (0.238 − tau21) × (0.206 − z_dr_0p2_0p4)
if girth2_top3 < 0.0022: z += -596 × (0.0022 − girth2_top3)
if lam2 < 0.00031: z += -2277 × (0.00031 − lam2)
if width < 0.0061: z += 135 × (0.0061 − width)
if tau21 < 0.238: z += 5.74 × (0.238 − tau21)
if mass < 36.23: z += -0.029 × (36.23 − mass)
if LHA > 0.177 and sum_pt_top3 > 353: z += 0.046 × (LHA − 0.177) × (sum_pt_top3 − 353)
if z_dr_0p05_0p1 > 0.751: z += -8.98 × (z_dr_0p05_0p1 − 0.751)
if n_dr_0_0p05 < 2.00: z += 0.302 × (2.00 − n_dr_0_0p05)
if girth2_top2 < 0.0076: z += 41.00 × (0.0076 − girth2_top2)
if e2 < 0.063: z += -4.99 × (0.063 − e2)
if width < 0.0075 and e2 > 0.025: z += -45347 × (0.0075 − width) × (e2 − 0.025)
if log_sum_pt > 6.90: z += 38.43 × (log_sum_pt − 6.90)
if width < 0.0061 and log_sum_pt > 6.90: z += -7695 × (0.0061 − width) × (log_sum_pt − 6.90)
if D2 < 0.746: z += -1.54 × (0.746 − D2)
if tau21 < 0.238 and z_dr_0p05_0p1 < 0.588: z += -11.34 × (0.238 − tau21) × (0.588 − z_dr_0p05_0p1)
if width < 0.0075 and planar_flow < 0.061: z += -5290 × (0.0075 − width) × (0.061 − planar_flow)
if lam1 < 0.0065 and pt1_dr01 > 1.22: z += 34.11 × (0.0065 − lam1) × (pt1_dr01 − 1.22)
if lam1 < 0.0084 and m01 > 16.31: z += -27.53 × (0.0084 − lam1) × (m01 − 16.31)
if tau21 < 0.238 and max_dr < 0.122: z += -149 × (0.238 − tau21) × (0.122 − max_dr)
if n_dr_0p05_0p1 > 5.00: z += 0.243 × (n_dr_0p05_0p1 − 5.00)
if mass > 80.40 and z_dr_0p2_0p4 < 0.206: z += -0.741 × (mass − 80.40) × (0.206 − z_dr_0p2_0p4)
if z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 2.00: z += 1.03 × (z_dr_0p05_0p1 − 0.751) × (2.00 − n_dr_0p2_0p4)
if girth2 < 0.019 and mass_top2 > 22.84: z += 2.21 × (0.019 − girth2) × (mass_top2 − 22.84)
if LHA > 0.177 and z_top5 > 0.865: z += -76.42 × (LHA − 0.177) × (z_top5 − 0.865)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.1% of jets, neuron 0.00, formula right for 60%.  
- **group 2** — 14.0% of jets, neuron 2.66, formula right for 72%.  
- **group 3** — 12.6% of jets, neuron 0.65, formula right for 71%.  
- **group 4** — 10.4% of jets, neuron 0.46, formula right for 62%.  
- **group 5** — 9.8% of jets, neuron 0.07, formula right for 52%.  
- **group 6** — 9.4% of jets, neuron 1.06, formula right for 78%.  
- **group 7** — 8.6% of jets, neuron 0.31, formula right for 53%.  
- **group 8** — 7.5% of jets, neuron 0.01, formula right for 83%.  
- **group 9** — 1.3% of jets, neuron 0.10, formula right for 67%.  
- **group 10** — 0.3% of jets, neuron 0.13, formula right for 70%.  

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs girth2 > 0.0188 (pushes up) and is pushed down strongly for e2 > 0.0634; it follows the number of particles above 10 GeV (rank correlation 0.847). Only tops reach it with any frequency (non-zero for 15.8% of them), so tops sit highest (0.23), then gluons (0.09) and quarks (0.03), with W and Z at 0.00.
- *computed — its value:* largest for t (0.23), then g (0.09), then q (0.03), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.57: large for t)
- **How the class scores use it:** It does not (or hardly) enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.419):

- `` — 2.1% of jets, value 2.20 (0.00…5.00), formula right 79%
- `` — 2.0% of jets, value 0.90 (0.00…2.25), formula right 85%
- `` — 2.4% of jets, value 0.32 (0.00…1.12), formula right 88%
- `` — 93.5% of jets, value 0.00 (0.00…0.00), formula right 64%

```
z = -1.36
if e2 > 0.063: z += -116 × (e2 − 0.063)
if girth2 > 0.019: z += 195 × (girth2 − 0.019)
if girth2 > 0.019 and pt_7 > 15.55: z += 6.70 × (girth2 − 0.019) × (pt_7 − 15.55)
if mass > 91.19: z += 0.082 × (mass − 91.19)
if girth2 > 0.019 and lam2 > 0.00054: z += 15108 × (girth2 − 0.019) × (lam2 − 0.00054)
if n_dr_0p2_0p4 > 1.00: z += 0.127 × (n_dr_0p2_0p4 − 1.00)
if mean_phi > 0.026: z += 24.44 × (mean_phi − 0.026)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 91.4% of jets, neuron 0.00, formula right for 64%.  
- **group 2** — 2.6% of jets, neuron 0.22, formula right for 83%.  
- **group 3** — 2.3% of jets, neuron 0.26, formula right for 88%.  
- **group 4** — 1.7% of jets, neuron 0.62, formula right for 88%.  
- **group 5** — 0.7% of jets, neuron 1.95, formula right for 84%.  
- **group 6** — 0.6% of jets, neuron 1.05, formula right for 76%.  
- **group 7** — 0.4% of jets, neuron 3.02, formula right for 91%.  
- **group 8** — 0.3% of jets, neuron 3.01, formula right for 62%.  
- **group 9** — 0.1% of jets, neuron 7.91, formula right for 74%.  
- **group 10** — 0.0% of jets, neuron 10.63, formula right for 75%.  
