# What each part of the 686-term formula does (64 particles)

*the tuned formula (start)*. Validation accuracy 81.42%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales, and each class score adds some of these scales and subtracts others. Gluon and quark jets are split mainly by the particle-count scale (neuron 1: many particles sharing the pT thinly), which the g score adds and the q score subtracts, while both light-jet scores subtract heavy two-prong-ness (neuron 4) and add the light, one-prong scales (9, 6 and 12). W and Z jets are recognised as compact jets with few particles (neurons 5 and 3) and above all by the absence of busy, wide radiation (neuron 8), which both boson scores subtract heavily. The two bosons are then told apart by mass: the light-side W scale (0) and the 80-93 GeV two-prong scale (11) are added to the W score, the 91-101 GeV window (7) is added to the Z score, and the above-the-Z scale (14) is subtracted from the W score. Top jets are picked out by hard particles spread far from the axis in a heavy jet (neuron 10) plus busy wide radiation (8), and the t score is pulled down strongly by the high-pT scale (13), on which top jets sit lowest.

## The 5 class scores

### score g: Many particles, round one-prong, not heavy

High for jets with many particles, a round one-prong pT pattern (large τ21 and D2) and no heavy two-prong mass: gluon jets score highest (3.06; AUC 0.93), quark jets next (0.77), top jets near zero (-0.05) and Z (-2.28) and W (-2.53) jets far below.

Adds the particle-count scale (neuron 1, +43%) and the light one-prong scale (9, +10%), with small additions from 6 and 12; subtracts heavy two-prong-ness (4, -21%), the sparse-jet scale (3, -10%) and the compact pT-window scale (5, -8%).

*computed:* largest for g (3.06), then q (0.77), then t (-0.05), then Z (-2.28), then W (-2.53); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light one-prong jet with few particles

High for light, one-prong jets that are not particle-rich (small mass of the hardest particles, small e2, large D2): quark jets score highest (2.70; AUC 0.89), gluon jets next (1.15), top jets near zero (0.06), W (-0.71) and Z (-0.75) jets below.

Subtracts heavy two-prong-ness (neuron 4, -40%), the particle-count scale (1, -20%) and the 80-93 GeV two-prong scale (11, -7%); adds the light one-prong scale (9, +17%), the one-prong scale away from 93-120 GeV (6, +7%) and lightness (12, +7%).

*computed:* largest for q (2.70), then g (1.15), then t (0.06), then W (-0.71), then Z (-0.75); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact two-prong jet at the W mass

High for compact, narrow two-prong jets with mass up to about the W: W jets score highest (3.53; AUC 0.97), quark jets slightly below zero (-0.29), Z (-1.43) and gluon (-2.27) jets further below and top jets far below (-5.31).

Subtracts busy wide radiation (neuron 8, -34%) and mass just above the Z (14, -14%), plus small amounts of 7, 12 and 9; adds the compact pT-window scale (5, +11%), the 80-93 GeV two-prong scale (11, +8%), light-side W mass (0, +8%), heavy two-prong-ness (4, +6%) and the sparse-jet scale (3, +4%).

*computed:* largest for W (3.53), then q (-0.29), then Z (-1.43), then g (-2.27), then t (-5.31); it separates W jets from the rest best (AUC 0.97: large for W)

### score Z: Compact two-prong jet at the Z mass

High for compact jets with few particles, an empty outer ring and mass at or just above 91 GeV: Z jets score highest (3.53; AUC 0.96), quark (-0.17) and W (-0.30) jets slightly below zero, gluon (-2.17) and top (-5.67) jets far below.

Subtracts busy wide radiation (neuron 8, -37%), the one-prong scale away from 93-120 GeV (6, -15%), light-side W mass (0, -14%) and a little of 2; adds the compact pT-window scale (5, +13%), the 91-101 GeV window (7, +9%) and small amounts of 3, 12 and 15.

*computed:* largest for Z (3.53), then q (-0.17), then W (-0.30), then g (-2.17), then t (-5.67); it separates Z jets from the rest best (AUC 0.96: large for Z)

### score t: Heavy, wide jet with spread-out hard prongs

High for heavy, wide jets whose hard particles are spread far from the axis (large girth, LHA and e2): top jets score highest (3.38; AUC 0.95), W (-0.22) and gluon (-0.36) jets slightly below zero, Z (-1.09) and quark (-1.28) jets further below.

Subtracts the high-pT, below-top-mass scale (neuron 13, -38%), the compact pT-window scale (5, -11%) and small amounts of 12 and 7; adds widely spread hard particles (10, +27%), busy wide radiation (8, +10%) and a little heavy two-prong-ness (4, +4%).

*computed:* largest for t (3.38), then W (-0.22), then g (-0.36), then Z (-1.09), then q (-1.28); it separates t jets from the rest best (AUC 0.95: large for t)

## The 16 neurons (most important first)

### neuron 1: Many particles, thinly shared pT (major)

- **What it measures:** Grows with the number of particles and falls when a few hardest particles hold most of the pT (for example when the 50 hardest carry more than 0.959 of it); a broad, soft spread and a small mass of the 30 hardest (below 80.4 GeV) also push it up. Gluon jets sit far highest (mean 5.89 for g, AUC 0.93), then top (2.75) and quark (2.22) jets, with Z (1.54) and W (1.39) jets lowest.
- *computed — its value:* largest for g (5.89), then t (2.75), then q (2.22), then Z (1.54), then W (1.39); it separates g jets from the rest best (AUC 0.93: large for g)
- **How the class scores use it:** It raises the g score (+43%) and lowers the q score (-20%), so it is the main handle for gluon versus quark; freezing it costs 12.096 points of accuracy. The W, Z and t scores hardly use it, even though top jets sit fairly high on it.
- *computed — used by:* raises the score of g (+43%); lowers the score of q (-20%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.706):

- `` — 5.1% of jets, value 7.89 (6.00…9.72), formula right 93%
- `` — 6.2% of jets, value 5.95 (3.61…8.09), formula right 79%
- `` — 2.0% of jets, value 5.28 (3.09…7.28), formula right 75%
- `` — 6.7% of jets, value 5.09 (3.25…6.94), formula right 69%
- `` — 4.9% of jets, value 3.29 (2.00…4.84), formula right 79%
- `` — 17.7% of jets, value 3.15 (1.41…5.00), formula right 79%
- `` — 8.4% of jets, value 2.92 (1.03…5.00), formula right 70%
- `` — 49.0% of jets, value 1.19 (0.00…2.53), formula right 85%

```
z = 1.37
if log_sum_pt > 6.91: z += 29.56 × (log_sum_pt − 6.91)
if z_top50_slots > 0.959: z += -37.94 × (z_top50_slots − 0.959)
if log_sum_pt > 6.89: z += 19.50 × (log_sum_pt − 6.89)
if sum_pt_top2 < 689: z += 0.0028 × (689 − sum_pt_top2)
if LHA < 0.404: z += 6.00 × (0.404 − LHA)
if sum_pt_top50 > 959: z += -0.0096 × (sum_pt_top50 − 959)
if tau32 > 0.329: z += 2.09 × (tau32 − 0.329)
if mass_top30 < 80.40: z += 0.056 × (80.40 − mass_top30)
if log_sum_pt > 6.81: z += -5.87 × (log_sum_pt − 6.81)
if z_top20_slots < 0.958: z += -9.54 × (0.958 − z_top20_slots)
if mass_top50 < 117: z += -0.018 × (117 − mass_top50)
if max_dr < 0.436: z += -7.37 × (0.436 − max_dr)
if n_pt_above_10 < 31.00: z += -0.055 × (31.00 − n_pt_above_10)
if mass_top30 < 80.40 and mass_top5 < 68.43: z += -0.00064 × (80.40 − mass_top30) × (68.43 − mass_top5)
if z_top30_slots > 0.934 and mass_top10 < 91.19: z += -0.312 × (z_top30_slots − 0.934) × (91.19 − mass_top10)
if sum_pt_top5 < 902: z += 0.0015 × (902 − sum_pt_top5)
if z_top30_slots > 0.934: z += 14.11 × (z_top30_slots − 0.934)
if sum_pt_top30 < 1073: z += 0.0045 × (1073 − sum_pt_top30)
if max_dr < 0.436 and z_dr_0p2_0p4 < 0.194: z += 28.10 × (0.436 − max_dr) × (0.194 − z_dr_0p2_0p4)
if log_sum_pt > 6.99: z += -17.78 × (log_sum_pt − 6.99)
if mass_top20 < 40.20: z += -0.090 × (40.20 − mass_top20)
if n_dr_0p2_0p4 < 13.00: z += -0.047 × (13.00 − n_dr_0p2_0p4)
if n_particles > 38.00: z += 0.022 × (n_particles − 38.00)
if n_particles > 38.00 and dr_0 < 0.144: z += 0.266 × (n_particles − 38.00) × (0.144 − dr_0)
if girth2_top10 < 0.0077: z += 60.02 × (0.0077 − girth2_top10)
if lam2 < 0.00083: z += -701 × (0.00083 − lam2)
if pt_9 < 34.06: z += -0.020 × (34.06 − pt_9)
if mass_top20 < 40.20 and mean_phi2 < 0.0059: z += 8.83 × (40.20 − mass_top20) × (0.0059 − mean_phi2)
if mass_over_sum_pt < 0.074: z += 16.89 × (0.074 − mass_over_sum_pt)
if n_particles > 38.00 and z_top50_slots < 0.985: z += 1.57 × (n_particles − 38.00) × (0.985 − z_top50_slots)
if sum_pt_top2 < 689 and n_dr_0p2_0p4 < 7.00: z += -0.00023 × (689 − sum_pt_top2) × (7.00 − n_dr_0p2_0p4)
if girth2_top15 < 0.0033: z += 155 × (0.0033 − girth2_top15)
if mass_top20 < 40.20 and n_real_top40 > 26.00: z += 0.0034 × (40.20 − mass_top20) × (n_real_top40 − 26.00)
if girth2_top3 < 0.00082: z += 549 × (0.00082 − girth2_top3)
if n_particles > 38.00 and dr_1 < 0.161: z += 0.111 × (n_particles − 38.00) × (0.161 − dr_1)
if girth2_top20 < 0.0012: z += 865 × (0.0012 − girth2_top20)
if girth2_top10 < 0.0013: z += 371 × (0.0013 − girth2_top10)
if log_sum_pt > 6.96: z += -3.52 × (log_sum_pt − 6.96)
if girth2_top3 < 0.00082 and n_dr_0p05_0p1 < 10.00: z += -92.85 × (0.00082 − girth2_top3) × (10.00 − n_dr_0p05_0p1)
if n_dr_0p1_0p2 < 7.00 and dr_6 < 0.079: z += -1.52 × (7.00 − n_dr_0p1_0p2) × (0.079 − dr_6)
if z_top30_slots > 0.934 and m012 > 16.90: z += 0.258 × (z_top30_slots − 0.934) × (m012 − 16.90)
if n_particles > 38.00 and mass_top15 < 57.87: z += -0.00052 × (n_particles − 38.00) × (57.87 − mass_top15)
if D2 < 1.23: z += 0.527 × (1.23 − D2)
if z_top30_slots > 0.934 and dr_5 < 0.058: z += -59.01 × (z_top30_slots − 0.934) × (0.058 − dr_5)
if n_pt_above_10 < 31.00 and mean_phi2 < 0.017: z += 0.152 × (31.00 − n_pt_above_10) × (0.017 − mean_phi2)
if girth2_top5 < 0.00066: z += 155 × (0.00066 − girth2_top5)
if log_sum_pt > 7.14: z += -2.90 × (log_sum_pt − 7.14)
if max_dr < 0.436 and n_real_top30 < 26.00: z += 0.267 × (0.436 − max_dr) × (26.00 − n_real_top30)
if z_dr_0_0p05 > 0.879: z += -0.869 × (z_dr_0_0p05 − 0.879)
if n_dr_0_0p05 > 30.00: z += -0.049 × (n_dr_0_0p05 − 30.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.4% of jets, neuron 0.90, formula right for 88%.  
- **group 2** — 18.6% of jets, neuron 2.06, formula right for 82%.  
- **group 3** — 11.5% of jets, neuron 2.06, formula right for 78%.  
- **group 4** — 11.5% of jets, neuron 3.35, formula right for 80%.  
- **group 5** — 9.4% of jets, neuron 3.00, formula right for 66%.  
- **group 6** — 9.0% of jets, neuron 4.30, formula right for 75%.  
- **group 7** — 5.1% of jets, neuron 5.64, formula right for 82%.  
- **group 8** — 4.1% of jets, neuron 5.91, formula right for 83%.  
- **group 9** — 3.3% of jets, neuron 7.29, formula right for 87%.  
- **group 10** — 1.0% of jets, neuron 8.24, formula right for 91%.  

### neuron 4: Heavy two-prong-ness (major)

- **What it measures:** Grows with the mass of the 10-20 hardest particles and with e2, and with a small D2 (a two-prong pattern); jets lighter than 80.8 GeV, very narrow jets and jets with a broad, soft spread (LHA above 0.115) are pushed down. Z (1.69) and W (1.60) jets sit highest, top jets next (1.15), gluon jets low (0.30) and quark jets lowest (0.15; AUC 0.16, small for q).
- *computed — its value:* largest for Z (1.69), then W (1.60), then t (1.15), then g (0.30), then q (0.15); it separates q jets from the rest best (AUC 0.16: small for q)
- **How the class scores use it:** It lowers the q score (-40%) and the g score (-21%) strongly, because a heavy, pronged jet is not a light jet, and it raises the W (+6%) and t (+4%) scores a little. The Z score hardly uses it.
- *computed — used by:* raises the score of W (+6%), t (+4%); lowers the score of g (-21%), q (-40%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.743):

- `` — 13.6% of jets, value 2.19 (1.72…2.69), formula right 96%
- `` — 8.7% of jets, value 1.72 (0.94…2.44), formula right 92%
- `` — 19.2% of jets, value 1.60 (0.75…2.25), formula right 85%
- `` — 24.2% of jets, value 0.84 (0.00…1.69), formula right 76%
- `` — 2.0% of jets, value 0.79 (0.06…1.44), formula right 78%
- `` — 3.4% of jets, value 0.25 (0.00…0.75), formula right 70%
- `` — 2.0% of jets, value 0.07 (0.00…0.25), formula right 61%
- `` — 26.9% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = 2.64
if girth < 0.121: z += -22.13 × (0.121 − girth)
if mass < 80.78: z += -0.110 × (80.78 − mass)
if LHA > 0.115: z += -7.42 × (LHA − 0.115)
if mass < 101: z += 0.038 × (101 − mass)
if D2 < 6.92: z += 0.173 × (6.92 − D2)
if girth2_top15 < 0.016: z += 72.51 × (0.016 − girth2_top15)
if width > 0.0096: z += 191 × (width − 0.0096)
if mass_top40 < 94.64: z += -0.026 × (94.64 − mass_top40)
if mass < 121: z += 0.013 × (121 − mass)
if mass_top30 < 153: z += -0.0059 × (153 − mass_top30)
if z_dr_0p2_0p4 < 0.068: z += -10.56 × (0.068 − z_dr_0p2_0p4)
if girth2_top40 < 0.013: z += 66.65 × (0.013 − girth2_top40)
if lam1 > 0.0077: z += -148 × (lam1 − 0.0077)
if mass_top40 < 83.33 and D2 < 6.92: z += -0.012 × (83.33 − mass_top40) × (6.92 − D2)
if mass_over_sum_pt > 0.060: z += -11.23 × (mass_over_sum_pt − 0.060)
if girth < 0.062: z += -27.35 × (0.062 − girth)
if mass < 121 and max_dr < 0.402: z += 0.145 × (121 − mass) × (0.402 − max_dr)
if girth2_top15 < 0.0073: z += -122 × (0.0073 − girth2_top15)
if n_dr_0p2_0p4 < 15.00: z += 0.041 × (15.00 − n_dr_0p2_0p4)
if sum_pt < 1066: z += -0.0057 × (1066 − sum_pt)
if mass < 86.40: z += -0.022 × (86.40 − mass)
if mass < 101 and girth2_top2 < 0.004: z += 4.09 × (101 − mass) × (0.004 − girth2_top2)
if mass < 74.25 and max_dr < 0.394: z += -0.679 × (74.25 − mass) × (0.394 − max_dr)
if n_particles > 29.00: z += -0.011 × (n_particles − 29.00)
if e2 > 0.037: z += 40.14 × (e2 − 0.037)
if z_dr_0p2_0p4 < 0.194: z += 1.21 × (0.194 − z_dr_0p2_0p4)
if n_dr_0p2_0p4 < 15.00 and max_dr < 0.436: z += -0.196 × (15.00 − n_dr_0p2_0p4) × (0.436 − max_dr)
if tau21 < 0.470: z += -1.26 × (0.470 − tau21)
if girth2_top10 < 0.0048: z += -80.89 × (0.0048 − girth2_top10)
if sum_pt_top20 < 994: z += 0.0011 × (994 − sum_pt_top20)
if eccentricity > 0.868: z += 2.70 × (eccentricity − 0.868)
if lam2 > 0.0024: z += -165 × (lam2 − 0.0024)
if tau32 < 0.577: z += 2.21 × (0.577 − tau32)
if mass_top10 < 45.59: z += 0.0058 × (45.59 − mass_top10)
if mass_over_sum_pt > 0.098: z += -5.52 × (mass_over_sum_pt − 0.098)
if mass_top40 < 83.33: z += 0.0047 × (83.33 − mass_top40)
if C2 > 0.109: z += 15.33 × (C2 − 0.109)
if mass_top20 > 104: z += -0.016 × (mass_top20 − 104)
if mass > 144: z += 0.015 × (mass − 144)
if pt_7 < 38.53: z += -0.0085 × (38.53 − pt_7)
if planar_flow < 0.304 and n_dr_0p05_0p1 > 6.00: z += -0.092 × (0.304 − planar_flow) × (n_dr_0p05_0p1 − 6.00)
if n_dr_0p2_0p4 < 15.00 and n_dr_0p1_0p2 > 10.00: z += -0.0012 × (15.00 − n_dr_0p2_0p4) × (n_dr_0p1_0p2 − 10.00)
if mass < 74.25: z += 0.0031 × (74.25 − mass)
if mass_top15 < 69.03: z += 0.001 × (69.03 − mass_top15)
if n_particles > 29.00 and n_dr_0_0p05 < 25.00: z += -7.8e-05 × (n_particles − 29.00) × (25.00 − n_dr_0_0p05)
if mass_top30 < 60.44: z += 0.0019 × (60.44 − mass_top30)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 34.0% of jets, neuron 1.75, formula right for 89%.  
- **group 2** — 17.0% of jets, neuron 0.91, formula right for 75%.  
- **group 3** — 11.1% of jets, neuron 1.39, formula right for 90%.  
- **group 4** — 9.6% of jets, neuron 0.00, formula right for 69%.  
- **group 5** — 8.4% of jets, neuron 0.55, formula right for 71%.  
- **group 6** — 7.3% of jets, neuron 0.00, formula right for 77%.  
- **group 7** — 6.5% of jets, neuron 0.00, formula right for 75%.  
- **group 8** — 3.0% of jets, neuron 0.00, formula right for 80%.  
- **group 9** — 2.3% of jets, neuron 1.45, formula right for 79%.  
- **group 10** — 0.8% of jets, neuron 0.00, formula right for 87%.  

### neuron 5: Compact jet in a pT window (major)

- **What it measures:** Runs opposite to the particle count: it rises when the pT of the 40 hardest particles passes 906.6 GeV, when the 50 hardest hold more than 0.959 of the pT and when the 30 hardest have mass above 68.3 GeV, and is cut back when the log of the total pT exceeds 6.91 and when m/pT is above 0.0905. Quark (1.43) and Z (1.42) jets sit highest, W jets next (1.12), top (0.57) and gluon (0.49) jets lowest (AUC 0.26, small for g).
- *computed — its value:* largest for q (1.43), then Z (1.42), then W (1.12), then t (0.57), then g (0.49); it separates g jets from the rest best (AUC 0.26: small for g)
- **How the class scores use it:** It raises the W (+11%) and Z (+13%) scores and lowers the g (-8%) and t (-11%) scores: a compact, few-particle jet with some mass is boson-like, not gluon- or top-like. The q score hardly uses it, although quark jets sit high on it.
- *computed — used by:* raises the score of W (+11%), Z (+13%); lowers the score of g (-8%), t (-11%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.563):

- `` — 58.4% of jets, value 1.50 (0.81…2.25), formula right 82%
- `` — 6.6% of jets, value 0.78 (0.00…1.50), formula right 76%
- `` — 2.0% of jets, value 0.71 (0.00…1.44), formula right 76%
- `` — 4.3% of jets, value 0.37 (0.00…0.94), formula right 80%
- `` — 10.8% of jets, value 0.27 (0.00…0.81), formula right 73%
- `` — 2.0% of jets, value 0.25 (0.00…0.76), formula right 82%
- `` — 5.1% of jets, value 0.13 (0.00…0.50), formula right 94%
- `` — 10.9% of jets, value 0.03 (0.00…0.00), formula right 84%

```
z = 0.504
if log_sum_pt > 6.91: z += -26.40 × (log_sum_pt − 6.91)
if sum_pt_top40 > 907: z += 0.0094 × (sum_pt_top40 − 907)
if log_sum_pt > 6.86: z += 11.69 × (log_sum_pt − 6.86)
if z_top40_slots > 0.930: z += -17.08 × (z_top40_slots − 0.930)
if sum_pt > 908 and girth2_top2 < 0.014: z += 0.465 × (sum_pt − 908) × (0.014 − girth2_top2)
if mass_top30 > 68.29: z += 0.028 × (mass_top30 − 68.29)
if mass_over_sum_pt > 0.090: z += -32.78 × (mass_over_sum_pt − 0.090)
if z_top50_slots > 0.959: z += 13.41 × (z_top50_slots − 0.959)
if log_sum_pt > 6.92 and girth2_top2 < 0.014: z += -880 × (log_sum_pt − 6.92) × (0.014 − girth2_top2)
if n_particles < 64.00 and e2 < 0.039: z += 1.52 × (64.00 − n_particles) × (0.039 − e2)
if girth2_top2 < 0.014: z += -35.84 × (0.014 − girth2_top2)
if mass > 64.49: z += 0.011 × (mass − 64.49)
if girth < 0.086: z += -11.16 × (0.086 − girth)
if sum_pt_top50 > 1061: z += -0.010 × (sum_pt_top50 − 1061)
if mass_top50 > 158: z += -0.158 × (mass_top50 − 158)
if mass_over_sum_pt > 0.098: z += 20.02 × (mass_over_sum_pt − 0.098)
if log_sum_pt > 6.94: z += -5.41 × (log_sum_pt − 6.94)
if mass_top50 > 91.19: z += -0.014 × (mass_top50 − 91.19)
if girth2_top30 < 0.018: z += -18.24 × (0.018 − girth2_top30)
if z_top30_slots > 0.934: z += -5.80 × (z_top30_slots − 0.934)
if sum_pt_top10 > 846: z += 0.0045 × (sum_pt_top10 − 846)
if max_dr > 0.436: z += 20.11 × (max_dr − 0.436)
if girth2_top10 < 0.0048: z += 91.69 × (0.0048 − girth2_top10)
if mass_top30 > 102: z += -0.019 × (mass_top30 − 102)
if mass_over_sum_pt_sq > 0.029: z += -692 × (mass_over_sum_pt_sq − 0.029)
if girth2_top30 < 0.018 and tau32 < 0.864: z += 93.62 × (0.018 − girth2_top30) × (0.864 − tau32)
if sum_pt_top3 > 331 and n_dr_0p05_0p1 < 27.00: z += -3.1e-05 × (sum_pt_top3 − 331) × (27.00 − n_dr_0p05_0p1)
if log_sum_pt > 6.99: z += 3.91 × (log_sum_pt − 6.99)
if mass > 173: z += -0.107 × (mass − 173)
if sum_pt_top15 > 951: z += -0.0031 × (sum_pt_top15 − 951)
if n_particles < 64.00 and mass_top5 > 3.07: z += -0.00018 × (64.00 − n_particles) × (mass_top5 − 3.07)
if sum_pt_top3 > 331: z += 0.0005 × (sum_pt_top3 − 331)
if max_dr > 0.240 and tau21 < 0.642: z += -2.76 × (max_dr − 0.240) × (0.642 − tau21)
if mass_top40 > 137: z += -0.020 × (mass_top40 − 137)
if log_sum_pt > 6.92: z += 1.42 × (log_sum_pt − 6.92)
if max_dr > 0.240: z += 0.445 × (max_dr − 0.240)
if sum_pt > 908 and pt1_dr01 > 0.253: z += -6.2e-05 × (sum_pt − 908) × (pt1_dr01 − 0.253)
if log_sum_pt > 6.96: z += 1.66 × (log_sum_pt − 6.96)
if sum_pt_top20 > 1005: z += 0.0021 × (sum_pt_top20 − 1005)
if log_sum_pt > 7.14: z += 7.53 × (log_sum_pt − 7.14)
if mass_over_sum_pt > 0.171: z += 72.31 × (mass_over_sum_pt − 0.171)
if z_dr_0p2_0p4 < 0.052: z += -1.35 × (0.052 − z_dr_0p2_0p4)
if sum_pt_top3 > 331 and n_dr_0p1_0p2 > 8.00: z += 7.5e-05 × (sum_pt_top3 − 331) × (n_dr_0p1_0p2 − 8.00)
if sum_pt_top5 > 902: z += -0.0061 × (sum_pt_top5 − 902)
if n_dr_0p2_0p4 < 13.00: z += -0.0028 × (13.00 − n_dr_0p2_0p4)
if mass_top50 > 158 and z_dr_0p05_0p1 > 0.591: z += 1.58 × (mass_top50 − 158) × (z_dr_0p05_0p1 − 0.591)
if z_dr_0_0p05 > 0.879: z += 0.681 × (z_dr_0_0p05 − 0.879)
if sum_pt_top5 > 631: z += -1.8e-05 × (sum_pt_top5 − 631)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 47.6% of jets, neuron 1.35, formula right for 79%.  
- **group 2** — 20.3% of jets, neuron 1.10, formula right for 82%.  
- **group 3** — 10.4% of jets, neuron 0.90, formula right for 80%.  
- **group 4** — 8.2% of jets, neuron 0.10, formula right for 82%.  
- **group 5** — 5.9% of jets, neuron 0.05, formula right for 93%.  
- **group 6** — 3.8% of jets, neuron 0.03, formula right for 86%.  
- **group 7** — 1.4% of jets, neuron 1.98, formula right for 81%.  
- **group 8** — 1.1% of jets, neuron 0.02, formula right for 90%.  
- **group 9** — 1.0% of jets, neuron 0.00, formula right for 73%.  
- **group 10** — 0.4% of jets, neuron 0.02, formula right for 80%.  

### neuron 8: Busy, wide radiation (major)

- **What it measures:** Grows with the number and pT share of particles at 0.2 <= ΔR < 0.4, with the minor-axis width lam2 and with the total particle count, and falls when the 40 hardest particles hold most of the pT. Top jets sit far highest (mean 6.60 for t, AUC 0.91), gluon jets next (2.26), quark jets lower (0.99), Z (0.27) and W (0.08) jets near zero.
- *computed — its value:* largest for t (6.60), then g (2.26), then q (0.99), then Z (0.27), then W (0.08); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W (-34%) and Z (-37%) scores, since a two-prong boson is compact with an empty outer ring, and raises the t score (+10%). The g and q scores hardly use it; freezing it costs 2.176 points.
- *computed — used by:* raises the score of t (+10%); lowers the score of W (-34%), Z (-37%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.802):

- `` — 11.2% of jets, value 9.17 (4.69…13.62), formula right 84%
- `` — 3.5% of jets, value 6.11 (3.50…9.44), formula right 70%
- `` — 5.2% of jets, value 5.67 (3.21…7.62), formula right 82%
- `` — 2.0% of jets, value 4.50 (0.00…9.87), formula right 66%
- `` — 4.7% of jets, value 3.23 (1.94…4.62), formula right 76%
- `` — 13.9% of jets, value 1.17 (0.00…2.75), formula right 74%
- `` — 59.5% of jets, value 0.17 (0.00…0.56), formula right 84%

```
z = -0.531
if mass_over_sum_pt > 0.098: z += -91.96 × (mass_over_sum_pt − 0.098)
if mass > 101: z += -0.087 × (mass − 101)
if mass > 80.78: z += 0.049 × (mass − 80.78)
if mass > 64.49: z += 0.031 × (mass − 64.49)
if girth < 0.097: z += 25.49 × (0.097 − girth)
if mass_over_sum_pt_sq < 0.014: z += -126 × (0.014 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.077 and sum_pt < 1116: z += 0.343 × (mass_over_sum_pt − 0.077) × (1116 − sum_pt)
if mass_over_sum_pt > 0.089: z += 48.05 × (mass_over_sum_pt − 0.089)
if mass > 87.36: z += 0.042 × (mass − 87.36)
if mass_top50 > 82.04: z += -0.035 × (mass_top50 − 82.04)
if girth2_top20 > 0.008: z += 210 × (girth2_top20 − 0.008)
if n_dr_0p2_0p4 < 21.00: z += -0.047 × (21.00 − n_dr_0p2_0p4)
if girth2_top40 > 0.0052: z += 116 × (girth2_top40 − 0.0052)
if lam1 < 0.020: z += -39.54 × (0.020 − lam1)
if mass_over_sum_pt > 0.077: z += 22.40 × (mass_over_sum_pt − 0.077)
if lam1 < 0.0082: z += 144 × (0.0082 − lam1)
if mass_top50 > 97.93: z += 0.035 × (mass_top50 − 97.93)
if mass_top40 > 79.55: z += -0.023 × (mass_top40 − 79.55)
if sum_pt < 1013: z += 0.018 × (1013 − sum_pt)
if z_dr_0p2_0p4 < 0.091: z += 6.03 × (0.091 − z_dr_0p2_0p4)
if lam2 < 0.0014: z += -495 × (0.0014 − lam2)
if girth2_top20 > 0.0057: z += -85.38 × (girth2_top20 − 0.0057)
if mass_over_sum_pt_sq < 0.0062: z += -224 × (0.0062 − mass_over_sum_pt_sq)
if girth2_top30 < 0.0075: z += 116 × (0.0075 − girth2_top30)
if n_particles > 51.00: z += 0.068 × (n_particles − 51.00)
if girth2_top40 < 0.0077: z += 114 × (0.0077 − girth2_top40)
if e2 > 0.025: z += -17.10 × (e2 − 0.025)
if sum_pt_top40 < 1002: z += 0.0061 × (1002 − sum_pt_top40)
if n_dr_0p2_0p4 < 15.00: z += -0.019 × (15.00 − n_dr_0p2_0p4)
if girth2_top40 > 0.0052 and sum_pt_top30 < 912: z += 0.658 × (girth2_top40 − 0.0052) × (912 − sum_pt_top30)
if mass > 144: z += -0.032 × (mass − 144)
if mass > 125: z += -0.017 × (mass − 125)
if n_particles > 51.00 and z_top50_slots > 0.970: z += -2.89 × (n_particles − 51.00) × (z_top50_slots − 0.970)
if sum_pt_top40 < 1002 and log_sum_pt < 6.81: z += 0.065 × (1002 − sum_pt_top40) × (6.81 − log_sum_pt)
if z_dr_0p1_0p2 < 0.155: z += 1.36 × (0.155 − z_dr_0p1_0p2)
if D2 < 1.79: z += 0.294 × (1.79 − D2)
if sum_pt_top40 < 1002 and max_dr < 0.394: z += -0.095 × (1002 − sum_pt_top40) × (0.394 − max_dr)
if lam2 < 0.0014 and n_dr_0p05_0p1 > 4.00: z += 17.86 × (0.0014 − lam2) × (n_dr_0p05_0p1 − 4.00)
if sum_pt < 1013 and z_0 > 0.089: z += -0.030 × (1013 − sum_pt) × (z_0 − 0.089)
if C2 > 0.073: z += 4.76 × (C2 − 0.073)
if sum_pt < 1013 and n_real_top50 < 43.00: z += -0.00084 × (1013 − sum_pt) × (43.00 − n_real_top50)
if mass_top15 > 30.36: z += 0.0017 × (mass_top15 − 30.36)
if sum_pt_top30 < 1012: z += -0.00095 × (1012 − sum_pt_top30)
if z_dr_0p1_0p2 > 0.219: z += 0.686 × (z_dr_0p1_0p2 − 0.219)
if mass > 87.36 and log_sum_pt < 6.90: z += 0.141 × (mass − 87.36) × (6.90 − log_sum_pt)
if z_top40_slots < 0.983: z += -2.72 × (0.983 − z_top40_slots)
if z_top15_slots < 0.814: z += 0.856 × (0.814 − z_top15_slots)
if n_dr_0p2_0p4 < 21.00 and mean_eta2 > 0.0068: z += -8.48 × (21.00 − n_dr_0p2_0p4) × (mean_eta2 − 0.0068)
if sum_pt < 1013 and mean_eta2 < 0.0012: z += -11.26 × (1013 − sum_pt) × (0.0012 − mean_eta2)
if n_particles > 51.00 and tau32 < 0.899: z += -0.018 × (n_particles − 51.00) × (0.899 − tau32)
if z_dr_0p1_0p2 > 0.219 and mean_phi > 0.00044: z += -2010 × (z_dr_0p1_0p2 − 0.219) × (mean_phi − 0.00044)
if n_dr_0p2_0p4 < 15.00 and z_dr_0p1_0p2 > 0.334: z += 0.164 × (15.00 − n_dr_0p2_0p4) × (z_dr_0p1_0p2 − 0.334)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 46.4% of jets, neuron 0.50, formula right for 85%.  
- **group 2** — 28.1% of jets, neuron 0.28, formula right for 75%.  
- **group 3** — 8.1% of jets, neuron 9.17, formula right for 94%.  
- **group 4** — 5.7% of jets, neuron 4.66, formula right for 71%.  
- **group 5** — 3.8% of jets, neuron 8.69, formula right for 75%.  
- **group 6** — 3.4% of jets, neuron 3.85, formula right for 79%.  
- **group 7** — 1.8% of jets, neuron 5.79, formula right for 66%.  
- **group 8** — 1.4% of jets, neuron 5.92, formula right for 81%.  
- **group 9** — 0.9% of jets, neuron 5.34, formula right for 73%.  
- **group 10** — 0.3% of jets, neuron 7.31, formula right for 56%.  

### neuron 10: Hard particles spread wide, heavy (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (large spread of the 5 hardest, little pT within ΔR < 0.05, large LHA and e2) and is pushed down for mass below 120.6 GeV. Top jets sit highest (2.31; AUC 0.85), Z, gluon and W jets in the middle (1.01-1.09), quark jets lowest (0.65).
- *computed — its value:* largest for t (2.31), then Z (1.09), then g (1.02), then W (1.01), then q (0.65); it separates t jets from the rest best (AUC 0.85: large for t)
- **How the class scores use it:** Only the t score uses it, raising it (+27%): widely spread hard prongs in a heavy jet are the main positive sign of a top; freezing it costs 1.62 points.
- *computed — used by:* raises the score of t (+27%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.641):

- `` — 16.5% of jets, value 2.58 (1.56…3.56), formula right 89%
- `` — 24.7% of jets, value 1.67 (0.75…2.50), formula right 76%
- `` — 18.9% of jets, value 1.19 (0.44…1.88), formula right 90%
- `` — 8.3% of jets, value 1.00 (0.00…1.94), formula right 74%
- `` — 3.6% of jets, value 0.58 (0.00…1.50), formula right 81%
- `` — 2.0% of jets, value 0.58 (0.00…1.81), formula right 77%
- `` — 3.2% of jets, value 0.28 (0.00…0.88), formula right 76%
- `` — 22.8% of jets, value 0.15 (0.00…0.56), formula right 77%

```
z = 4.09
if girth2_top5 < 0.024: z += -77.51 × (0.024 − girth2_top5)
if mass < 121: z += -0.032 × (121 − mass)
if e2 < 0.065: z += -27.54 × (0.065 − e2)
if mass < 86.40: z += 0.053 × (86.40 − mass)
if mass < 80.40: z += 0.052 × (80.40 − mass)
if girth < 0.121: z += -9.86 × (0.121 − girth)
if z_dr_0p2_0p4 < 0.091: z += 9.01 × (0.091 − z_dr_0p2_0p4)
if z_dr_0_0p05 > 0.767: z += -10.05 × (z_dr_0_0p05 − 0.767)
if mass > 144: z += -0.107 × (mass − 144)
if girth2_top10 > 0.00024: z += 62.21 × (girth2_top10 − 0.00024)
if lam1 > 0.0019: z += -64.48 × (lam1 − 0.0019)
if girth2_top5 < 0.024 and sum_pt_top3 < 656: z += 0.109 × (0.024 − girth2_top5) × (656 − sum_pt_top3)
if n_dr_0p2_0p4 < 13.00: z += -0.063 × (13.00 − n_dr_0p2_0p4)
if mass_top50 > 137: z += 0.086 × (mass_top50 − 137)
if mass < 121 and tau21 < 0.470: z += 0.096 × (121 − mass) × (0.470 − tau21)
if mass < 62.55: z += -0.036 × (62.55 − mass)
if max_dr < 0.402: z += -3.21 × (0.402 − max_dr)
if e2 < 0.065 and z_dr_0p1_0p2 < 0.219: z += -33.36 × (0.065 − e2) × (0.219 − z_dr_0p1_0p2)
if dr_1 < 0.069: z += 5.15 × (0.069 − dr_1)
if girth2_top30 < 0.0054: z += 110 × (0.0054 − girth2_top30)
if dr_0 < 0.064 and n_dr_0p2_0p4 > 2.00: z += 0.852 × (0.064 − dr_0) × (n_dr_0p2_0p4 − 2.00)
if n_dr_0p2_0p4 < 6.00: z += -0.081 × (6.00 − n_dr_0p2_0p4)
if mass > 163: z += -0.072 × (mass − 163)
if sum_pt_top50 < 959: z += -0.0091 × (959 − sum_pt_top50)
if girth2_top5 < 0.024 and n_particles > 34.00: z += 0.359 × (0.024 − girth2_top5) × (n_particles − 34.00)
if girth2_top10 > 0.0077: z += -30.84 × (girth2_top10 − 0.0077)
if mass < 121 and D2 < 1.98: z += -0.0063 × (121 − mass) × (1.98 − D2)
if z_dr_0p2_0p4 < 0.091 and max_dr > 0.274: z += 10.63 × (0.091 − z_dr_0p2_0p4) × (max_dr − 0.274)
if mass < 121 and mass_top3 > 16.90: z += 0.0004 × (121 − mass) × (mass_top3 − 16.90)
if max_dr < 0.240: z += 8.26 × (0.240 − max_dr)
if n_dr_0p2_0p4 < 13.00 and dr_2 < 0.065: z += 0.247 × (13.00 − n_dr_0p2_0p4) × (0.065 − dr_2)
if mass_top50 > 161: z += 0.025 × (mass_top50 − 161)
if sum_pt_top15 > 935 and dr_4 < 0.063: z += -0.029 × (sum_pt_top15 − 935) × (0.063 − dr_4)
if tau21 < 0.428: z += 0.336 × (0.428 − tau21)
if lam1 > 0.0019 and pt_dispersion > 0.275: z += -147 × (lam1 − 0.0019) × (pt_dispersion − 0.275)
if mass_top50 > 173 and sum_pt < 1261: z += -0.00033 × (mass_top50 − 173) × (1261 − sum_pt)
if girth2_top30 < 0.0054 and sum_pt_top5 < 631: z += 0.270 × (0.0054 − girth2_top30) × (631 − sum_pt_top5)
if planar_flow < 0.356: z += -0.219 × (0.356 − planar_flow)
if mass_top50 > 161 and D2 > 1.23: z += 0.014 × (mass_top50 − 161) × (D2 − 1.23)
if dr_0 < 0.064: z += -0.559 × (0.064 − dr_0)
if mass_top50 > 173 and max_pair_mass < 33.38: z += -0.0022 × (mass_top50 − 173) × (33.38 − max_pair_mass)
if z_dr_0_0p05 > 0.767 and n_particles > 36.00: z += -0.034 × (z_dr_0_0p05 − 0.767) × (n_particles − 36.00)
if mass_top50 > 173: z += 0.017 × (mass_top50 − 173)
if mass_top50 > 161 and pt_9 < 41.44: z += 0.00042 × (mass_top50 − 161) × (41.44 − pt_9)
if lam1 > 0.0047: z += -1.21 × (lam1 − 0.0047)
if sum_pt_top50 < 959 and mass_top10 > 23.39: z += 9.1e-06 × (959 − sum_pt_top50) × (mass_top10 − 23.39)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.7% of jets, neuron 1.47, formula right for 93%.  
- **group 2** — 16.8% of jets, neuron 1.09, formula right for 77%.  
- **group 3** — 13.1% of jets, neuron 0.21, formula right for 78%.  
- **group 4** — 12.3% of jets, neuron 0.99, formula right for 72%.  
- **group 5** — 10.2% of jets, neuron 0.11, formula right for 76%.  
- **group 6** — 9.7% of jets, neuron 2.34, formula right for 71%.  
- **group 7** — 6.4% of jets, neuron 2.32, formula right for 93%.  
- **group 8** — 5.4% of jets, neuron 2.56, formula right for 87%.  
- **group 9** — 1.2% of jets, neuron 0.16, formula right for 76%.  
- **group 10** — 0.2% of jets, neuron 0.00, formula right for 78%.  

### neuron 0: Light-side W mass (moderate)

- **What it measures:** Falls as the jet gets heavier and wider: mass above 78.3 GeV pushes it down (partly given back above 91.0 GeV, then cut again above 92.9 GeV), while a small m/pT (below about 0.0905) pushes it up. W jets sit far highest (mean 1.79 for W, AUC 0.96), then quark (0.36) and gluon (0.26) jets, with Z (0.12) and top (0.09) jets lowest.
- *computed — its value:* largest for W (1.79), then q (0.36), then g (0.26), then Z (0.12), then t (0.09); it separates W jets from the rest best (AUC 0.96: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the Z score (-14%): a jet high on this scale is taken as a W rather than a Z. The g, q and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of Z (-14%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.844):

- `` — 14.5% of jets, value 2.09 (1.50…2.59), formula right 92%
- `` — 6.5% of jets, value 1.21 (0.34…1.94), formula right 69%
- `` — 10.5% of jets, value 0.81 (0.03…1.47), formula right 68%
- `` — 2.0% of jets, value 0.37 (0.00…0.94), formula right 69%
- `` — 20.7% of jets, value 0.25 (0.00…0.59), formula right 77%
- `` — 2.0% of jets, value 0.11 (0.00…0.38), formula right 83%
- `` — 43.8% of jets, value 0.00 (0.00…0.00), formula right 85%

```
z = 0.964
if mass > 78.26: z += -0.140 × (mass − 78.26)
if mass > 91.03: z += 0.192 × (mass − 91.03)
if mass > 92.86: z += -0.123 × (mass − 92.86)
if mass_over_sum_pt_sq < 0.0082: z += 471 × (0.0082 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.077: z += -46.78 × (mass_over_sum_pt − 0.077)
if mass_over_sum_pt > 0.083: z += 55.28 × (mass_over_sum_pt − 0.083)
if girth < 0.057: z += -55.08 × (0.057 − girth)
if mass_top50 < 82.04: z += -0.045 × (82.04 − mass_top50)
if girth2_top40 < 0.0063: z += -342 × (0.0063 − girth2_top40)
if girth2_top50 < 0.0074: z += 235 × (0.0074 − girth2_top50)
if width < 0.0096: z += -97.01 × (0.0096 − width)
if girth2_top20 < 0.006: z += -185 × (0.006 − girth2_top20)
if log_sum_pt < 7.02: z += 3.22 × (7.02 − log_sum_pt)
if lam1 < 0.0059: z += -192 × (0.0059 − lam1)
if girth2_top40 < 0.029: z += 12.63 × (0.029 − girth2_top40)
if LHA < 0.260: z += 5.30 × (0.260 − LHA)
if girth2_top40 < 0.0063 and girth2_top3 < 0.0029: z += 50659 × (0.0063 − girth2_top40) × (0.0029 − girth2_top3)
if n_dr_0p2_0p4 < 10.00: z += 0.045 × (10.00 − n_dr_0p2_0p4)
if z_top30_slots > 0.920: z += -3.67 × (z_top30_slots − 0.920)
if mass > 74.25: z += -0.0057 × (mass − 74.25)
if mass_top50 < 82.04 and z_dr_0p05_0p1 < 0.213: z += 0.061 × (82.04 − mass_top50) × (0.213 − z_dr_0p05_0p1)
if girth2_top40 < 0.0043: z += 153 × (0.0043 − girth2_top40)
if n_particles < 62.00: z += 0.0064 × (62.00 − n_particles)
if mass > 78.26 and n_dr_0p2_0p4 < 7.00: z += -0.009 × (mass − 78.26) × (7.00 − n_dr_0p2_0p4)
if girth2_top20 < 0.0053: z += 58.98 × (0.0053 − girth2_top20)
if log_sum_pt < 6.99 and sum_pt_top40 > 956: z += 0.044 × (6.99 − log_sum_pt) × (sum_pt_top40 − 956)
if sum_pt < 1013 and z_dr_0p2_0p4 < 0.194: z += -0.030 × (1013 − sum_pt) × (0.194 − z_dr_0p2_0p4)
if girth2_top20 < 0.0075 and girth2_top2 < 0.0031: z += -10651 × (0.0075 − girth2_top20) × (0.0031 − girth2_top2)
if sum_pt < 1013 and z_dr_0p1_0p2 > 0.089: z += -0.019 × (1013 − sum_pt) × (z_dr_0p1_0p2 − 0.089)
if z_dr_0_0p05 > 0.879: z += -3.71 × (z_dr_0_0p05 − 0.879)
if sum_pt_top20 > 942: z += 0.0013 × (sum_pt_top20 − 942)
if mass_top20 < 70.42: z += 0.0033 × (70.42 − mass_top20)
if sum_pt < 1013: z += -0.002 × (1013 − sum_pt)
if z_dr_0p2_0p4 < 0.037 and pt_2 < 127: z += 0.065 × (0.037 − z_dr_0p2_0p4) × (127 − pt_2)
if log_sum_pt < 6.99 and n_dr_0p1_0p2 > 11.00: z += -0.112 × (6.99 − log_sum_pt) × (n_dr_0p1_0p2 − 11.00)
if mass_top50 < 62.55: z += 0.0061 × (62.55 − mass_top50)
if z_top30_slots > 0.920 and C2 > 0.061: z += -75.85 × (z_top30_slots − 0.920) × (C2 − 0.061)
if sum_pt_top30 > 1073: z += -0.0012 × (sum_pt_top30 − 1073)
if n_particles < 62.00 and n_real_top30 < 30.00: z += 0.00044 × (62.00 − n_particles) × (30.00 − n_real_top30)
if n_dr_0p2_0p4 < 10.00 and n_real_top30 < 30.00: z += -0.0022 × (10.00 − n_dr_0p2_0p4) × (30.00 − n_real_top30)
if n_dr_0p2_0p4 < 10.00 and z_top50_slots < 0.985: z += -1.98 × (10.00 − n_dr_0p2_0p4) × (0.985 − z_top50_slots)
if sum_pt < 1013 and n_dr_0p1_0p2 > 11.00: z += 9e-05 × (1013 − sum_pt) × (n_dr_0p1_0p2 − 11.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.9% of jets, neuron 1.64, formula right for 82%.  
- **group 2** — 21.8% of jets, neuron 0.03, formula right for 88%.  
- **group 3** — 16.4% of jets, neuron 0.24, formula right for 79%.  
- **group 4** — 12.2% of jets, neuron 0.62, formula right for 71%.  
- **group 5** — 8.9% of jets, neuron 0.00, formula right for 92%.  
- **group 6** — 5.0% of jets, neuron 0.00, formula right for 70%.  
- **group 7** — 4.8% of jets, neuron 0.00, formula right for 80%.  
- **group 8** — 4.3% of jets, neuron 0.00, formula right for 72%.  
- **group 9** — 1.5% of jets, neuron 0.00, formula right for 77%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 78%.  

### neuron 3: Sparse jet, empty outer ring (moderate)

- **What it measures:** Rises for a small m/pT, few particles in total (under 46) and few particles at 0.2 <= ΔR < 0.4, and a thin minor axis (small lam2); a busy outer ring pulls it down. W jets sit highest (1.04), then quark (0.77) and Z (0.51) jets, gluon jets low (0.22) and top jets almost at zero (0.03; AUC 0.18, small for t).
- *computed — its value:* largest for W (1.04), then q (0.77), then Z (0.51), then g (0.22), then t (0.03); it separates t jets from the rest best (AUC 0.18: small for t)
- **How the class scores use it:** It raises the W and Z scores (+4% each) and lowers the g score (-10%): a sparse, clean jet looks like a boson or a quark, not a gluon. The q and t scores hardly use it.
- *computed — used by:* raises the score of W (+4%), Z (+4%); lowers the score of g (-10%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.739):

- `` — 6.2% of jets, value 2.06 (1.34…2.78), formula right 94%
- `` — 2.1% of jets, value 1.36 (0.81…1.91), formula right 99%
- `` — 15.4% of jets, value 1.21 (0.56…1.84), formula right 83%
- `` — 2.8% of jets, value 1.19 (0.41…1.97), formula right 96%
- `` — 2.1% of jets, value 0.65 (0.00…1.47), formula right 91%
- `` — 11.8% of jets, value 0.51 (0.00…1.00), formula right 76%
- `` — 4.6% of jets, value 0.42 (0.00…0.97), formula right 94%
- `` — 55.0% of jets, value 0.08 (0.00…0.28), formula right 78%

```
z = -0.179
if mass_over_sum_pt_sq < 0.0075: z += 204 × (0.0075 − mass_over_sum_pt_sq)
if n_dr_0p2_0p4 < 5.00 and z_top50_slots > 0.979: z += 11.19 × (5.00 − n_dr_0p2_0p4) × (z_top50_slots − 0.979)
if tau21 < 0.428 and lam1 < 0.016: z += -227 × (0.428 − tau21) × (0.016 − lam1)
if mass_over_sum_pt_sq < 0.0075 and mass_top20 < 137: z += -0.753 × (0.0075 − mass_over_sum_pt_sq) × (137 − mass_top20)
if n_particles < 46.00 and sum_pt_top40 > 859: z += 0.00012 × (46.00 − n_particles) × (sum_pt_top40 − 859)
if lam2 < 0.00062 and n_dr_0p4_up < 1.00: z += 702 × (0.00062 − lam2) × (1.00 − n_dr_0p4_up)
if n_dr_0p1_0p2 < 15.00: z += 0.018 × (15.00 − n_dr_0p1_0p2)
if lam2 < 0.00062: z += 610 × (0.00062 − lam2)
if n_particles < 46.00: z += 0.013 × (46.00 − n_particles)
if lam1 < 0.0077: z += -31.46 × (0.0077 − lam1)
if girth2_top5 < 0.00011: z += -7572 × (0.00011 − girth2_top5)
if mass_over_sum_pt_sq < 0.0075 and girth2_top15 > 0.0049: z += 422434 × (0.0075 − mass_over_sum_pt_sq) × (girth2_top15 − 0.0049)
if n_dr_0p2_0p4 < 7.00: z += -0.031 × (7.00 − n_dr_0p2_0p4)
if n_particles < 46.00 and mass_top30 > 73.33: z += -0.0014 × (46.00 − n_particles) × (mass_top30 − 73.33)
if n_particles < 46.00 and eta_0 > -0.029: z += -0.288 × (46.00 − n_particles) × (eta_0 − -0.029)
if n_dr_0p2_0p4 < 5.00 and dr_0 < 0.041: z += -5.18 × (5.00 − n_dr_0p2_0p4) × (0.041 − dr_0)
if n_dr_0p2_0p4 < 5.00 and z_dr_0p1_0p2 > 0.0034: z += 0.279 × (5.00 − n_dr_0p2_0p4) × (z_dr_0p1_0p2 − 0.0034)
if lam2 < 0.00035: z += 1046 × (0.00035 − lam2)
if z_dr_0p2_0p4 < 0.0014: z += 249 × (0.0014 − z_dr_0p2_0p4)
if mass_over_sum_pt < 0.051: z += 9.91 × (0.051 − mass_over_sum_pt)
if n_dr_0p2_0p4 < 5.00 and sum_pt_top30 < 988: z += -0.0015 × (5.00 − n_dr_0p2_0p4) × (988 − sum_pt_top30)
if z_top20_slots > 0.923: z += -1.27 × (z_top20_slots − 0.923)
if n_dr_0p2_0p4 < 7.00 and phi_0 > -0.040: z += -0.243 × (7.00 − n_dr_0p2_0p4) × (phi_0 − -0.040)
if max_dr < 0.298 and tau21 > 0.428: z += -15.93 × (0.298 − max_dr) × (tau21 − 0.428)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 32.7% of jets, neuron 0.01, formula right for 78%.  
- **group 2** — 10.8% of jets, neuron 0.49, formula right for 75%.  
- **group 3** — 10.3% of jets, neuron 0.19, formula right for 70%.  
- **group 4** — 9.2% of jets, neuron 0.18, formula right for 82%.  
- **group 5** — 8.3% of jets, neuron 1.87, formula right for 97%.  
- **group 6** — 8.3% of jets, neuron 0.98, formula right for 97%.  
- **group 7** — 6.7% of jets, neuron 0.87, formula right for 77%.  
- **group 8** — 6.3% of jets, neuron 0.36, formula right for 89%.  
- **group 9** — 3.8% of jets, neuron 1.50, formula right for 83%.  
- **group 10** — 3.4% of jets, neuron 1.28, formula right for 75%.  

### neuron 6: One-prong jet away from 93-120 GeV (moderate)

- **What it measures:** Follows one-prong-ness (large D2 and τ21, a round rather than elongated pT pattern) and is pushed down for masses between about 93 and 120 GeV and for m/pT above 0.0508; masses below 92.9 GeV push it up. Quark jets sit highest (1.57; AUC 0.81), gluon jets next (1.28), top jets in between (0.87), Z (0.20) and W (0.16) jets near zero.
- *computed — its value:* largest for q (1.57), then g (1.28), then t (0.87), then Z (0.20), then W (0.16); it separates q jets from the rest best (AUC 0.81: large for q)
- **How the class scores use it:** It raises the q (+7%) and g (+3%) scores and lowers the Z score (-15%): a one-prong jet outside the Z mass range is not a Z. The W and t scores hardly use it.
- *computed — used by:* raises the score of g (+3%), q (+7%); lowers the score of Z (-15%); does not (or hardly) enter the score of W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.64):

- `` — 4.3% of jets, value 2.41 (2.12…2.62), formula right 84%
- `` — 6.4% of jets, value 2.03 (1.69…2.25), formula right 77%
- `` — 3.6% of jets, value 1.77 (1.31…2.06), formula right 74%
- `` — 11.9% of jets, value 1.62 (0.56…2.75), formula right 75%
- `` — 6.2% of jets, value 1.40 (0.94…1.75), formula right 74%
- `` — 7.5% of jets, value 0.89 (0.38…1.38), formula right 68%
- `` — 13.9% of jets, value 0.77 (0.00…1.94), formula right 85%
- `` — 46.2% of jets, value 0.13 (0.00…0.44), formula right 85%

```
z = 0.257
if mass < 101: z += -0.070 × (101 − mass)
if mass_over_sum_pt > 0.051: z += -38.85 × (mass_over_sum_pt − 0.051)
if mass < 121: z += -0.035 × (121 − mass)
if mass < 92.86: z += 0.068 × (92.86 − mass)
if mass < 173: z += 0.011 × (173 − mass)
if width < 0.0096: z += -208 × (0.0096 − width)
if mass_over_sum_pt_sq < 0.020: z += 53.25 × (0.020 − mass_over_sum_pt_sq)
if n_dr_0p2_0p4 < 21.00: z += -0.048 × (21.00 − n_dr_0p2_0p4)
if mass < 86.40: z += 0.035 × (86.40 − mass)
if girth2_top20 > 0.0018: z += 63.77 × (girth2_top20 − 0.0018)
if mass_over_sum_pt > 0.051 and n_dr_0p2_0p4 < 21.00: z += 0.998 × (mass_over_sum_pt − 0.051) × (21.00 − n_dr_0p2_0p4)
if lam1 < 0.0073: z += 137 × (0.0073 − lam1)
if girth2_top20 < 0.008: z += 97.76 × (0.008 − girth2_top20)
if lam2 < 0.0064: z += 56.19 × (0.0064 − lam2)
if lam1 > 0.0082: z += 110 × (lam1 − 0.0082)
if mass_top10 < 85.41: z += 0.007 × (85.41 − mass_top10)
if mass_top50 < 71.80: z += 0.032 × (71.80 − mass_top50)
if e2 < 0.048: z += 13.46 × (0.048 − e2)
if girth2_top50 < 0.014: z += -31.20 × (0.014 − girth2_top50)
if mass_over_sum_pt > 0.051 and z_dr_0_0p05 < 0.908: z += 6.85 × (mass_over_sum_pt − 0.051) × (0.908 − z_dr_0_0p05)
if mass_over_sum_pt > 0.051 and lam2 < 0.0024: z += 4834 × (mass_over_sum_pt − 0.051) × (0.0024 − lam2)
if mass_over_sum_pt > 0.051 and z_dr_0p1_0p2 < 0.286: z += 44.57 × (mass_over_sum_pt − 0.051) × (0.286 − z_dr_0p1_0p2)
if mass_over_sum_pt > 0.051 and n_dr_0p1_0p2 < 17.00: z += -1.07 × (mass_over_sum_pt − 0.051) × (17.00 − n_dr_0p1_0p2)
if LHA > 0.228: z += 2.28 × (LHA − 0.228)
if girth2_top20 > 0.0018 and max_dr < 0.436: z += 274 × (girth2_top20 − 0.0018) × (0.436 − max_dr)
if mass_over_sum_pt > 0.051 and C2 < 0.142: z += 50.55 × (mass_over_sum_pt − 0.051) × (0.142 − C2)
if girth2_top3 > 0.010 and D2 < 4.45: z += 25.12 × (girth2_top3 − 0.010) × (4.45 − D2)
if mass_over_sum_pt > 0.051 and max_pair_mass > 13.05: z += -0.246 × (mass_over_sum_pt − 0.051) × (max_pair_mass − 13.05)
if lam1 > 0.0082 and pt_4 < 68.12: z += 1.89 × (lam1 − 0.0082) × (68.12 − pt_4)
if mass < 92.86 and sum_pt_top15 < 1083: z += -3.1e-05 × (92.86 − mass) × (1083 − sum_pt_top15)
if lam1 > 0.012: z += -41.69 × (lam1 − 0.012)
if e2 < 0.044: z += 4.73 × (0.044 − e2)
if e2 < 0.030: z += -10.53 × (0.030 − e2)
if e2 > 0.056: z += -63.34 × (e2 − 0.056)
if LHA > 0.228 and D2 < 3.35: z += -0.704 × (LHA − 0.228) × (3.35 − D2)
if girth2_top20 > 0.017: z += -39.70 × (girth2_top20 − 0.017)
if mass_over_sum_pt > 0.051 and eta_0 < 0.080: z += 9.14 × (mass_over_sum_pt − 0.051) × (0.080 − eta_0)
if mass < 121 and sum_pt_top50 < 1009: z += 3.8e-05 × (121 − mass) × (1009 − sum_pt_top50)
if log_sum_pt < 6.81: z += 5.64 × (6.81 − log_sum_pt)
if LHA > 0.404: z += 7.99 × (LHA − 0.404)
if e2 > 0.056 and max_dr < 0.369: z += -1546 × (e2 − 0.056) × (0.369 − max_dr)
if mass_top10 > 99.07: z += 0.020 × (mass_top10 − 99.07)
if lam1 > 0.025: z += 71.77 × (lam1 − 0.025)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 23.4% of jets, neuron 0.10, formula right for 85%.  
- **group 2** — 21.5% of jets, neuron 0.17, formula right for 89%.  
- **group 3** — 8.1% of jets, neuron 1.80, formula right for 74%.  
- **group 4** — 7.9% of jets, neuron 0.86, formula right for 90%.  
- **group 5** — 7.8% of jets, neuron 1.31, formula right for 72%.  
- **group 6** — 7.1% of jets, neuron 2.30, formula right for 82%.  
- **group 7** — 6.8% of jets, neuron 1.14, formula right for 69%.  
- **group 8** — 6.6% of jets, neuron 0.68, formula right for 66%.  
- **group 9** — 6.6% of jets, neuron 1.28, formula right for 74%.  
- **group 10** — 4.2% of jets, neuron 1.54, formula right for 87%.  

### neuron 7: Mass window 91-101 GeV (moderate)

- **What it measures:** Switches on mainly for mass between 91.19 and 101 GeV (below 91.19 GeV it is pushed down hard), helped by an elongated two-prong pattern (small τ21 and D2) and a not-too-narrow jet. Almost only Z jets sit high on it (mean 2.20 for Z, AUC 0.91); W, gluon, top and quark jets all stay near zero (0.10 or less).
- *computed — its value:* largest for Z (2.20), then W (0.10), then g (0.07), then t (0.06), then q (0.04); it separates Z jets from the rest best (AUC 0.91: large for Z)
- **How the class scores use it:** It raises the Z score (+9%) and lowers the W (-6%) and t (-3%) scores: a mass just above the Z peak points to a Z. The g and q scores hardly use it.
- *computed — used by:* raises the score of Z (+9%); lowers the score of W (-6%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.772):

- `` — 8.2% of jets, value 3.55 (2.06…5.06), formula right 97%
- `` — 11.5% of jets, value 1.24 (0.06…2.62), formula right 82%
- `` — 6.1% of jets, value 0.51 (0.00…2.00), formula right 78%
- `` — 2.0% of jets, value 0.48 (0.00…1.44), formula right 82%
- `` — 7.0% of jets, value 0.12 (0.00…0.44), formula right 83%
- `` — 9.2% of jets, value 0.05 (0.00…0.19), formula right 92%
- `` — 21.9% of jets, value 0.02 (0.00…0.00), formula right 81%
- `` — 34.1% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = -0.337
if mass < 91.19: z += -0.207 × (91.19 − mass)
if e2_sq < 0.0096: z += 460 × (0.0096 − e2_sq)
if mass < 101: z += 0.062 × (101 − mass)
if girth2_top40 < 0.008: z += -571 × (0.008 − girth2_top40)
if girth2_top20 < 0.008: z += -458 × (0.008 − girth2_top20)
if girth < 0.086: z += -42.52 × (0.086 − girth)
if mass < 101 and max_dr < 0.391: z += 0.920 × (101 − mass) × (0.391 − max_dr)
if mass < 91.19 and max_dr < 0.391: z += -1.37 × (91.19 − mass) × (0.391 − max_dr)
if girth2_top40 < 0.013: z += 157 × (0.013 − girth2_top40)
if mass < 121: z += 0.025 × (121 − mass)
if mass_top50 < 97.93 and D2 < 1.60: z += -0.418 × (97.93 − mass_top50) × (1.60 − D2)
if mass < 101 and D2 < 1.60: z += 0.308 × (101 − mass) × (1.60 − D2)
if girth2_top40 < 0.0088: z += 252 × (0.0088 − girth2_top40)
if girth2_top20 < 0.0064: z += 316 × (0.0064 − girth2_top20)
if mass < 86.40: z += -0.044 × (86.40 − mass)
if mass_over_sum_pt < 0.090: z += -32.73 × (0.090 − mass_over_sum_pt)
if LHA < 0.320: z += 7.74 × (0.320 − LHA)
if mass < 82.85: z += 0.049 × (82.85 − mass)
if z_dr_0p2_0p4 < 0.091: z += -9.77 × (0.091 − z_dr_0p2_0p4)
if girth2_top40 < 0.0077: z += 218 × (0.0077 − girth2_top40)
if girth2_top50 < 0.014: z += -50.10 × (0.014 − girth2_top50)
if mass_top30 < 86.25: z += -0.016 × (86.25 − mass_top30)
if e2 < 0.028: z += 48.66 × (0.028 − e2)
if mass_top30 < 76.42: z += 0.020 × (76.42 − mass_top30)
if mass_top20 < 66.84: z += 0.018 × (66.84 − mass_top20)
if mass_top50 < 97.93: z += -0.0095 × (97.93 − mass_top50)
if n_dr_0p2_0p4 < 13.00 and z_1st < 0.499: z += 0.129 × (13.00 − n_dr_0p2_0p4) × (0.499 − z_1st)
if D2 < 1.79 and n_dr_0p2_0p4 < 9.00: z += 0.130 × (1.79 − D2) × (9.00 − n_dr_0p2_0p4)
if z_top50_slots < 0.991: z += -36.73 × (0.991 − z_top50_slots)
if mass_top50 < 79.21: z += 0.016 × (79.21 − mass_top50)
if z_dr_0p2_0p4 < 0.0047: z += 178 × (0.0047 − z_dr_0p2_0p4)
if D2 < 1.79 and girth2_top50 < 0.0078: z += -471 × (1.79 − D2) × (0.0078 − girth2_top50)
if n_dr_0p2_0p4 < 13.00: z += 0.017 × (13.00 − n_dr_0p2_0p4)
if C2 < 0.056: z += -8.62 × (0.056 − C2)
if mass < 91.19 and pt_dispersion < 0.338: z += -0.168 × (91.19 − mass) × (0.338 − pt_dispersion)
if n_dr_0p2_0p4 < 13.00 and planar_flow < 0.536: z += 0.055 × (13.00 − n_dr_0p2_0p4) × (0.536 − planar_flow)
if n_dr_0p2_0p4 < 2.00: z += -0.286 × (2.00 − n_dr_0p2_0p4)
if mass < 91.19 and z_dr_0p05_0p1 > 0.403: z += -0.137 × (91.19 − mass) × (z_dr_0p05_0p1 − 0.403)
if D2 < 1.79 and girth2_top50 < 0.0061: z += 647 × (1.79 − D2) × (0.0061 − girth2_top50)
if max_dr < 0.194 and dr_8 > 0.0081: z += -307 × (0.194 − max_dr) × (dr_8 − 0.0081)
if mass_top30 < 76.42 and D2 < 1.60: z += 0.194 × (76.42 − mass_top30) × (1.60 − D2)
if LHA < 0.320 and pt_dispersion < 0.355: z += 7.93 × (0.320 − LHA) × (0.355 − pt_dispersion)
if max_dr < 0.194: z += 10.55 × (0.194 − max_dr)
if mass < 86.40 and z_dr_0p05_0p1 > 0.403: z += 0.036 × (86.40 − mass) × (z_dr_0p05_0p1 − 0.403)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 34.4% of jets, neuron 0.41, formula right for 82%.  
- **group 2** — 15.9% of jets, neuron 0.26, formula right for 78%.  
- **group 3** — 12.5% of jets, neuron 2.42, formula right for 93%.  
- **group 4** — 8.3% of jets, neuron 0.11, formula right for 92%.  
- **group 5** — 8.2% of jets, neuron 0.00, formula right for 72%.  
- **group 6** — 7.3% of jets, neuron 0.00, formula right for 68%.  
- **group 7** — 6.9% of jets, neuron 0.00, formula right for 78%.  
- **group 8** — 4.2% of jets, neuron 0.00, formula right for 78%.  
- **group 9** — 1.7% of jets, neuron 0.00, formula right for 82%.  
- **group 10** — 0.6% of jets, neuron 0.00, formula right for 88%.  

### neuron 9: Light one-prong jet off the W (moderate)

- **What it measures:** Large when the 50 hardest particles have mass below 136.8 GeV and falls as the mass and width of the jet grow; it is pushed down for mass between 62.55 and about 82.9 GeV, the W side. Quark jets sit highest (2.04; AUC 0.82), gluon jets next (1.25), top (0.33), W (0.20) and Z (0.16) jets low.
- *computed — its value:* largest for q (2.04), then g (1.25), then t (0.33), then W (0.20), then Z (0.16); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** It raises the q (+17%) and g (+10%) scores and lowers the W score slightly (-3%): a light jet away from the W mass is a light-quark or gluon jet. The Z and t scores hardly use it.
- *computed — used by:* raises the score of g (+10%), q (+17%); lowers the score of W (-3%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.87):

- `` — 2.2% of jets, value 4.21 (1.81…6.62), formula right 68%
- `` — 16.0% of jets, value 2.71 (2.19…3.25), formula right 79%
- `` — 8.2% of jets, value 1.85 (1.25…2.50), formula right 70%
- `` — 3.3% of jets, value 0.89 (0.25…1.50), formula right 61%
- `` — 17.7% of jets, value 0.33 (0.00…1.00), formula right 84%
- `` — 52.5% of jets, value 0.04 (0.00…0.00), formula right 84%

```
z = 0.600
if mass_top50 < 137: z += 0.053 × (137 − mass_top50)
if mass_top50 < 161: z += -0.034 × (161 − mass_top50)
if mass > 62.55: z += -0.060 × (mass − 62.55)
if girth2_top15 < 0.021: z += -72.82 × (0.021 − girth2_top15)
if mass > 82.85: z += 0.056 × (mass − 82.85)
if mass > 92.86: z += 0.039 × (mass − 92.86)
if mass < 137: z += 0.0097 × (137 − mass)
if girth < 0.044: z += -74.90 × (0.044 − girth)
if mass_top40 < 91.29: z += -0.023 × (91.29 − mass_top40)
if girth2_top40 < 0.0063: z += 225 × (0.0063 − girth2_top40)
if LHA < 0.209: z += 15.38 × (0.209 − LHA)
if mass > 144: z += -0.076 × (mass − 144)
if mass_top40 < 80.89: z += 0.024 × (80.89 − mass_top40)
if mass_top15 < 78.96: z += -0.012 × (78.96 − mass_top15)
if mass_top40 < 121: z += 0.0057 × (121 − mass_top40)
if sum_pt < 950: z += 0.028 × (950 − sum_pt)
if mass > 53.87 and eccentricity > 0.588: z += 0.019 × (mass − 53.87) × (eccentricity − 0.588)
if girth2_top15 < 0.01: z += 34.69 × (0.01 − girth2_top15)
if log_sum_pt < 6.86: z += -20.18 × (6.86 − log_sum_pt)
if girth2_top15 < 0.021 and n_particles > 41.00: z += 1.42 × (0.021 − girth2_top15) × (n_particles − 41.00)
if lam1 < 0.0052: z += 112 × (0.0052 − lam1)
if width > 0.026: z += -261 × (width − 0.026)
if sum_pt_top40 < 956: z += 0.007 × (956 − sum_pt_top40)
if mass_top15 < 52.71: z += 0.0084 × (52.71 − mass_top15)
if girth > 0.097: z += 9.64 × (girth − 0.097)
if z_top40_slots < 0.930: z += -28.53 × (0.930 − z_top40_slots)
if girth < 0.028: z += -29.10 × (0.028 − girth)
if girth2 < 0.0036: z += 137 × (0.0036 − girth2)
if girth2_top40 < 0.0063 and sum_pt < 1023: z += 1.57 × (0.0063 − girth2_top40) × (1023 − sum_pt)
if sum_pt < 950 and z_0 < 0.412: z += -0.028 × (950 − sum_pt) × (0.412 − z_0)
if log_sum_pt < 6.86 and z_top20_slots > 0.897: z += 203 × (6.86 − log_sum_pt) × (z_top20_slots − 0.897)
if mass_top40 < 121 and max_dr > 0.379: z += 0.053 × (121 − mass_top40) × (max_dr − 0.379)
if n_dr_0p1_0p2 > 21.00: z += -0.025 × (n_dr_0p1_0p2 − 21.00)
if log_sum_pt < 6.86 and dr_7 < 0.083: z += 144 × (6.86 − log_sum_pt) × (0.083 − dr_7)
if mass > 173: z += 0.031 × (mass − 173)
if sum_pt_top40 < 956 and z_top15_slots > 0.795: z += -0.051 × (956 − sum_pt_top40) × (z_top15_slots − 0.795)
if e2 > 0.065: z += -41.03 × (e2 − 0.065)
if sum_pt_top40 < 956 and z_top30_slots > 0.973: z += -0.310 × (956 − sum_pt_top40) × (z_top30_slots − 0.973)
if z_top40_slots < 0.930 and n_pt_above_10 > 28.00: z += 1.31 × (0.930 − z_top40_slots) × (n_pt_above_10 − 28.00)
if z_dr_0p05_0p1 > 0.711: z += -0.704 × (z_dr_0p05_0p1 − 0.711)
if mass_top40 < 80.89 and eta_1 > 0.058: z += -6.35 × (80.89 − mass_top40) × (eta_1 − 0.058)
if sum_pt_top50 > 1246: z += -0.00047 × (sum_pt_top50 − 1246)
if e2 < 0.048: z += 0.039 × (0.048 − e2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.2% of jets, neuron 0.12, formula right for 82%.  
- **group 2** — 22.1% of jets, neuron 0.01, formula right for 88%.  
- **group 3** — 13.6% of jets, neuron 2.74, formula right for 79%.  
- **group 4** — 12.4% of jets, neuron 2.09, formula right for 72%.  
- **group 5** — 10.2% of jets, neuron 0.20, formula right for 91%.  
- **group 6** — 6.1% of jets, neuron 0.51, formula right for 76%.  
- **group 7** — 6.0% of jets, neuron 0.10, formula right for 72%.  
- **group 8** — 1.8% of jets, neuron 2.82, formula right for 64%.  
- **group 9** — 1.2% of jets, neuron 0.03, formula right for 74%.  
- **group 10** — 0.5% of jets, neuron 3.12, formula right for 59%.  

### neuron 12: Lightness: mass below 83 GeV (moderate)

- **What it measures:** Essentially on for jets lighter than 82.9 GeV (more so below 74.3 GeV), but pushed down for very light jets (mass below 62.55 GeV, or the 50 hardest below 77.4 GeV), very narrow jets and large C2. Quark jets sit highest (1.28; AUC 0.80), gluon jets next (0.57), then W (0.40), top (0.17) and Z (0.12) jets.
- *computed — its value:* largest for q (1.28), then g (0.57), then W (0.40), then t (0.17), then Z (0.12); it separates q jets from the rest best (AUC 0.80: large for q)
- **How the class scores use it:** It raises the q (+7%), g (+3%) and Z (+3%) scores and lowers the W (-4%) and t (-4%) scores: small corrections that credit light jets to the light-jet scores and fine-tune the W/Z balance.
- *computed — used by:* raises the score of g (+3%), q (+7%), Z (+3%); lowers the score of W (-4%), t (-4%) (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.653):

- `` — 16.9% of jets, value 1.75 (1.12…2.38), formula right 75%
- `` — 2.0% of jets, value 1.29 (0.00…4.12), formula right 84%
- `` — 4.0% of jets, value 0.87 (0.00…1.50), formula right 81%
- `` — 3.9% of jets, value 0.76 (0.00…1.38), formula right 70%
- `` — 20.6% of jets, value 0.36 (0.00…0.75), formula right 82%
- `` — 52.6% of jets, value 0.07 (0.00…0.12), formula right 84%

```
z = -0.049
if mass < 82.85: z += 0.081 × (82.85 − mass)
if mass_top50 < 77.38: z += -0.057 × (77.38 − mass_top50)
if mass_top50 < 80.40: z += 0.042 × (80.40 − mass_top50)
if mass < 74.25: z += 0.053 × (74.25 − mass)
if girth < 0.050: z += -34.44 × (0.050 − girth)
if mass < 62.55: z += -0.050 × (62.55 − mass)
if C2 > 0.061: z += -13.22 × (C2 − 0.061)
if mass < 86.40: z += -0.017 × (86.40 − mass)
if mass_top50 < 77.38 and z_dr_0p05_0p1 < 0.851: z += 0.026 × (77.38 − mass_top50) × (0.851 − z_dr_0p05_0p1)
if girth2_top50 < 0.0074: z += 95.60 × (0.0074 − girth2_top50)
if mass < 86.40 and z_dr_0p1_0p2 < 0.065: z += -0.271 × (86.40 − mass) × (0.065 − z_dr_0p1_0p2)
if planar_flow > 0.259: z += -0.569 × (planar_flow − 0.259)
if mass_top40 < 89.68: z += 0.0077 × (89.68 − mass_top40)
if girth < 0.050 and n_particles < 64.00: z += -0.534 × (0.050 − girth) × (64.00 − n_particles)
if mass_top40 < 77.94: z += -0.011 × (77.94 − mass_top40)
if mass_top40 < 89.68 and sum_pt_top2 < 502: z += -4.4e-05 × (89.68 − mass_top40) × (502 − sum_pt_top2)
if girth2_top5 < 0.001: z += 354 × (0.001 − girth2_top5)
if mass_top20 > 125 and C2 > 0.056: z += -1.17 × (mass_top20 − 125) × (C2 − 0.056)
if LHA < 0.228: z += 2.83 × (0.228 − LHA)
if LHA > 0.404: z += 22.42 × (LHA − 0.404)
if z_dr_0_0p05 > 0.908: z += -6.78 × (z_dr_0_0p05 − 0.908)
if mass_top40 < 89.68 and n_particles > 22.00: z += -0.00019 × (89.68 − mass_top40) × (n_particles − 22.00)
if log_sum_pt < 6.86: z += -6.90 × (6.86 − log_sum_pt)
if mass_top20 > 125: z += 0.033 × (mass_top20 − 125)
if mass_top30 > 138: z += 0.023 × (mass_top30 − 138)
if D2 < 3.81: z += 0.026 × (3.81 − D2)
if girth2_top10 < 0.00074 and n_particles > 51.00: z += -164 × (0.00074 − girth2_top10) × (n_particles − 51.00)
if mass < 86.40 and z_dr_0p05_0p1 > 0.299: z += 0.082 × (86.40 − mass) × (z_dr_0p05_0p1 − 0.299)
if z_dr_0p1_0p2 > 0.688 and min_pair_mass > 0.740: z += -1.73 × (z_dr_0p1_0p2 − 0.688) × (min_pair_mass − 0.740)
if LHA > 0.404 and lam2 < 0.0037: z += 12401 × (LHA − 0.404) × (0.0037 − lam2)
if mass_top20 > 125 and pt_2 > 73.69: z += 0.00072 × (mass_top20 − 125) × (pt_2 − 73.69)
if mass_top30 > 138 and dr_3 < 0.187: z += 0.201 × (mass_top30 − 138) × (0.187 − dr_3)
if mass_top20 > 125 and pt_2 > 112: z += -0.0022 × (mass_top20 − 125) × (pt_2 − 112)
if girth2_top5 > 0.017: z += -12.52 × (girth2_top5 − 0.017)
if log_sum_pt < 6.86 and n_particles > 62.00: z += 1.23 × (6.86 − log_sum_pt) × (n_particles − 62.00)
if mass_top20 > 125 and pt_2 > 138: z += 0.0023 × (mass_top20 − 125) × (pt_2 − 138)
if girth2_top15 < 0.0058: z += -2.29 × (0.0058 − girth2_top15)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 52.5% of jets, neuron 0.16, formula right for 85%.  
- **group 2** — 12.5% of jets, neuron 0.04, formula right for 81%.  
- **group 3** — 6.6% of jets, neuron 1.60, formula right for 76%.  
- **group 4** — 6.5% of jets, neuron 1.34, formula right for 73%.  
- **group 5** — 6.1% of jets, neuron 0.47, formula right for 64%.  
- **group 6** — 6.0% of jets, neuron 1.04, formula right for 71%.  
- **group 7** — 5.2% of jets, neuron 1.90, formula right for 84%.  
- **group 8** — 4.1% of jets, neuron 0.65, formula right for 92%.  
- **group 9** — 0.2% of jets, neuron 0.84, formula right for 73%.  
- **group 10** — 0.2% of jets, neuron 0.00, formula right for 96%.  

### neuron 13: High total pT, below top mass (moderate)

- **What it measures:** Grows with the total jet pT and the particle count, is pushed up for mass above 74.3 GeV and pushed down for mass above 143.8 GeV. All non-top types sit at similar, high values (gluon 2.41, Z 2.14, W 1.95, quark 1.89); top jets sit lowest (1.00; AUC 0.19, small for t).
- *computed — its value:* largest for g (2.41), then Z (2.14), then W (1.95), then q (1.89), then t (1.00); it separates t jets from the rest best (AUC 0.19: small for t)
- **How the class scores use it:** Only the t score uses it, lowering it strongly (-38%): a jet high on this scale is short of the top mass and so not a top. It is the main negative top handle.
- *computed — used by:* lowers the score of t (-38%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.743):

- `` — 29.9% of jets, value 2.75 (2.12…3.31), formula right 82%
- `` — 39.7% of jets, value 2.04 (1.47…2.66), formula right 83%
- `` — 10.8% of jets, value 1.33 (0.59…2.09), formula right 73%
- `` — 3.7% of jets, value 1.16 (0.56…1.88), formula right 84%
- `` — 4.2% of jets, value 0.69 (0.19…1.25), formula right 94%
- `` — 3.7% of jets, value 0.48 (0.00…1.19), formula right 77%
- `` — 2.2% of jets, value 0.28 (0.00…0.72), formula right 93%
- `` — 5.9% of jets, value 0.11 (0.00…0.44), formula right 69%

```
z = 1.81
if log_sum_pt < 7.02: z += -7.34 × (7.02 − log_sum_pt)
if mass > 74.25: z += 0.027 × (mass − 74.25)
if mass > 144: z += -0.135 × (mass − 144)
z += 0.011 × n_particles
if e2_sq < 0.0096: z += 121 × (0.0096 − e2_sq)
if mass_top50 > 97.93: z += -0.034 × (mass_top50 − 97.93)
if mass_top50 > 137: z += 0.096 × (mass_top50 − 137)
if sum_pt_top40 < 1053: z += 0.0069 × (1053 − sum_pt_top40)
if sum_pt < 1013: z += -0.020 × (1013 − sum_pt)
if sum_pt < 1053: z += -0.0091 × (1053 − sum_pt)
if sum_pt_top50 < 1014: z += 0.015 × (1014 − sum_pt_top50)
if mass_over_sum_pt > 0.171: z += 292 × (mass_over_sum_pt − 0.171)
if mass > 137: z += -0.032 × (mass − 137)
if mass_over_sum_pt_sq > 0.029: z += -782 × (mass_over_sum_pt_sq − 0.029)
if mass > 173: z += 0.175 × (mass − 173)
if sum_pt_top40 < 1053 and D2 < 5.38: z += -0.00076 × (1053 − sum_pt_top40) × (5.38 − D2)
if mass_top10 > 56.92: z += 0.013 × (mass_top10 − 56.92)
if n_dr_0p2_0p4 < 4.00: z += -0.137 × (4.00 − n_dr_0p2_0p4)
if log_sum_pt < 6.81: z += 18.98 × (6.81 − log_sum_pt)
if sum_pt < 1085 and tau21 > 0.186: z += 0.0041 × (1085 − sum_pt) × (tau21 − 0.186)
if z_top10_slots > 0.830: z += -3.60 × (z_top10_slots − 0.830)
if sum_pt_top50 < 1014 and D2 < 4.45: z += -0.0018 × (1014 − sum_pt_top50) × (4.45 − D2)
if sum_pt < 986: z += -0.0053 × (986 − sum_pt)
if sum_pt < 1085 and pt_3 < 89.69: z += -3.5e-05 × (1085 − sum_pt) × (89.69 − pt_3)
if sum_pt_top50 < 959 and D2 < 4.45: z += 0.0035 × (959 − sum_pt_top50) × (4.45 − D2)
if mass_over_sum_pt > 0.079: z += 2.20 × (mass_over_sum_pt − 0.079)
if sum_pt_top40 < 956: z += -0.0028 × (956 − sum_pt_top40)
if z_top10_slots > 0.830 and D2 < 3.81: z += 1.87 × (z_top10_slots − 0.830) × (3.81 − D2)
if mass > 161: z += -0.021 × (mass − 161)
if sum_pt_top20 > 957: z += -0.00088 × (sum_pt_top20 − 957)
if mass_top50 > 169: z += -0.049 × (mass_top50 − 169)
if sum_pt < 1013 and dr_5 < 0.051: z += -0.110 × (1013 − sum_pt) × (0.051 − dr_5)
if C2 > 0.073: z += -2.03 × (C2 − 0.073)
if mass > 161 and D2 < 5.38: z += -0.0045 × (mass − 161) × (5.38 − D2)
if girth < 0.074: z += 1.28 × (0.074 − girth)
if log_sum_pt < 6.81 and dr_5 < 0.051: z += -257 × (6.81 − log_sum_pt) × (0.051 − dr_5)
if mass > 74.25 and sum_pt < 1017: z += -3e-05 × (mass − 74.25) × (1017 − sum_pt)
if mass > 144 and sum_pt < 1008: z += 0.00023 × (mass − 144) × (1008 − sum_pt)
if sum_pt > 1261: z += -0.0018 × (sum_pt − 1261)
if sum_pt > 1261 and dr_7 < 0.083: z += -0.039 × (sum_pt − 1261) × (0.083 − dr_7)
if sum_pt_top30 > 1111: z += 0.00096 × (sum_pt_top30 − 1111)
if sum_pt_top50 < 959: z += 0.00055 × (959 − sum_pt_top50)
if mass > 74.25 and D2 > 0.603: z += 0.00012 × (mass − 74.25) × (D2 − 0.603)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 70.7% of jets, neuron 2.30, formula right for 82%.  
- **group 2** — 11.2% of jets, neuron 1.00, formula right for 70%.  
- **group 3** — 7.0% of jets, neuron 0.66, formula right for 93%.  
- **group 4** — 5.1% of jets, neuron 1.36, formula right for 84%.  
- **group 5** — 2.7% of jets, neuron 0.11, formula right for 67%.  
- **group 6** — 1.3% of jets, neuron 0.35, formula right for 86%.  
- **group 7** — 0.8% of jets, neuron 0.07, formula right for 59%.  
- **group 8** — 0.6% of jets, neuron 1.49, formula right for 78%.  
- **group 9** — 0.4% of jets, neuron 0.77, formula right for 67%.  
- **group 10** — 0.1% of jets, neuron 2.32, formula right for 80%.  

### neuron 14: Mass just above the Z (moderate)

- **What it measures:** Large for m/pT between about 0.0905 and 0.118 and for mass above 91.0 GeV (masses below 89.7 GeV push it down); it rises with mass overall. Z jets sit highest (1.55; AUC 0.88), gluon (0.51) and top (0.38) jets well below, quark (0.16) and W (0.15) jets lowest.
- *computed — its value:* largest for Z (1.55), then g (0.51), then t (0.38), then q (0.16), then W (0.15); it separates Z jets from the rest best (AUC 0.88: large for Z)
- **How the class scores use it:** Only the W score uses it, lowering it (-14%): a jet heavier than the Z peak is not a W. The Z score hardly uses it even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-14%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.782):

- `` — 12.0% of jets, value 1.88 (1.19…2.44), formula right 93%
- `` — 2.0% of jets, value 1.85 (0.84…3.06), formula right 73%
- `` — 14.4% of jets, value 1.34 (0.59…2.00), formula right 79%
- `` — 2.0% of jets, value 0.51 (0.03…0.97), formula right 69%
- `` — 20.4% of jets, value 0.28 (0.00…0.84), formula right 82%
- `` — 6.5% of jets, value 0.22 (0.00…0.50), formula right 91%
- `` — 4.9% of jets, value 0.14 (0.00…0.41), formula right 84%
- `` — 37.6% of jets, value 0.01 (0.00…0.00), formula right 77%

```
z = -0.019
if mass_over_sum_pt < 0.090: z += -168 × (0.090 − mass_over_sum_pt)
if mass_over_sum_pt < 0.098: z += 93.63 × (0.098 − mass_over_sum_pt)
if mass_over_sum_pt < 0.118: z += 48.19 × (0.118 − mass_over_sum_pt)
if mass < 89.74: z += -0.086 × (89.74 − mass)
if mass < 91.03: z += -0.064 × (91.03 − mass)
if girth2_top50 < 0.0093: z += -270 × (0.0093 − girth2_top50)
if girth2_top30 < 0.012: z += -141 × (0.012 − girth2_top30)
if mass_over_sum_pt < 0.141: z += 13.98 × (0.141 − mass_over_sum_pt)
if mass_top50 < 117: z += 0.020 × (117 − mass_top50)
if mass < 137: z += 0.012 × (137 − mass)
if lam1 < 0.012: z += -101 × (0.012 − lam1)
if mass_top40 < 80.89: z += -0.044 × (80.89 − mass_top40)
if lam2 < 0.0024: z += 293 × (0.0024 − lam2)
if mass_top40 < 111: z += -0.012 × (111 − mass_top40)
if girth2_top20 < 0.017 and z_top50_slots > 0.970: z += -1459 × (0.017 − girth2_top20) × (z_top50_slots − 0.970)
if sum_pt_top50 > 976: z += -0.0049 × (sum_pt_top50 − 976)
if lam1 < 0.0062: z += 202 × (0.0062 − lam1)
if n_dr_0p2_0p4 < 21.00: z += 0.023 × (21.00 − n_dr_0p2_0p4)
if girth2 < 0.0079: z += 128 × (0.0079 − girth2)
z += 0.00048 × sum_pt_top5
if mass_top40 < 83.33: z += 0.021 × (83.33 − mass_top40)
if girth < 0.086: z += -9.72 × (0.086 − girth)
if max_dr > 0.240: z += -2.18 × (max_dr − 0.240)
if log_sum_pt > 6.94: z += 6.87 × (log_sum_pt − 6.94)
if girth2_top50 < 0.0063: z += -164 × (0.0063 − girth2_top50)
if mass_top30 < 91.19: z += 0.011 × (91.19 − mass_top30)
if mass < 80.40: z += 0.020 × (80.40 − mass)
if girth2_top20 < 0.017 and tau21 < 0.642: z += -102 × (0.017 − girth2_top20) × (0.642 − tau21)
if lam1 < 0.0062 and z_top50_slots > 0.985: z += 7942 × (0.0062 − lam1) × (z_top50_slots − 0.985)
if lam1 < 0.0073: z += 71.96 × (0.0073 − lam1)
if girth2_top30 < 0.0061: z += 93.39 × (0.0061 − girth2_top30)
if girth2_top10 < 0.0069: z += 35.86 × (0.0069 − girth2_top10)
if e2 < 0.033: z += 12.76 × (0.033 − e2)
if n_dr_0p2_0p4 < 21.00 and n_dr_0p1_0p2 < 21.00: z += -0.00068 × (21.00 − n_dr_0p2_0p4) × (21.00 − n_dr_0p1_0p2)
if max_dr > 0.240 and mass_top10 < 56.92: z += 0.034 × (max_dr − 0.240) × (56.92 − mass_top10)
if lam2 < 0.0012: z += 142 × (0.0012 − lam2)
if z_top50_slots < 0.959: z += 87.08 × (0.959 − z_top50_slots)
if girth2_top20 < 0.017: z += -4.98 × (0.017 − girth2_top20)
if mass < 91.03 and z_dr_0p1_0p2 < 0.219: z += 0.017 × (91.03 − mass) × (0.219 − z_dr_0p1_0p2)
if mass_top40 < 91.29 and z_dr_0p2_0p4 < 0.068: z += -0.044 × (91.29 − mass_top40) × (0.068 − z_dr_0p2_0p4)
if mass_top40 < 91.29: z += -0.0023 × (91.29 − mass_top40)
if n_dr_0p2_0p4 < 21.00 and n_real_top40 < 40.00: z += -0.00065 × (21.00 − n_dr_0p2_0p4) × (40.00 − n_real_top40)
if girth2 < 0.0079 and n_dr_0p05_0p1 > 5.00: z += -2.36 × (0.0079 − girth2) × (n_dr_0p05_0p1 − 5.00)
if mass_top50 < 117 and mass_top2 > 28.79: z += -0.00092 × (117 − mass_top50) × (mass_top2 − 28.79)
if max_dr > 0.402: z += -0.396 × (max_dr − 0.402)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 22.2% of jets, neuron 0.43, formula right for 80%.  
- **group 2** — 20.6% of jets, neuron 1.70, formula right for 87%.  
- **group 3** — 20.4% of jets, neuron 0.31, formula right for 86%.  
- **group 4** — 7.3% of jets, neuron 0.00, formula right for 76%.  
- **group 5** — 7.3% of jets, neuron 0.21, formula right for 74%.  
- **group 6** — 6.8% of jets, neuron 0.00, formula right for 73%.  
- **group 7** — 5.9% of jets, neuron 0.00, formula right for 83%.  
- **group 8** — 5.6% of jets, neuron 0.00, formula right for 66%.  
- **group 9** — 2.5% of jets, neuron 0.95, formula right for 82%.  
- **group 10** — 1.4% of jets, neuron 0.15, formula right for 91%.  

### neuron 2: pT held by the hardest particles (minor)

- **What it measures:** Follows the total pT of the 20-30 hardest particles and rises when the 30 hardest carry more than 0.905 of the jet pT; jets lighter than 92.9 GeV are pushed down. It differs little between types: gluon jets highest (0.59), then Z (0.47), quark and W (0.35), and top jets lowest (0.17; AUC 0.25, small for t).
- *computed — its value:* largest for g (0.59), then Z (0.47), then q (0.35), then W (0.35), then t (0.17); it separates t jets from the rest best (AUC 0.25: small for t)
- **How the class scores use it:** Only the Z score uses it, lowering it slightly (-3%); it is a small correction and freezing it changes almost nothing.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.524):

- `` — 2.0% of jets, value 2.16 (0.38…5.12), formula right 83%
- `` — 2.0% of jets, value 1.63 (1.00…2.25), formula right 91%
- `` — 2.0% of jets, value 0.94 (0.50…1.38), formula right 86%
- `` — 20.9% of jets, value 0.57 (0.25…0.75), formula right 82%
- `` — 26.4% of jets, value 0.39 (0.25…0.62), formula right 86%
- `` — 17.7% of jets, value 0.25 (0.00…0.50), formula right 76%
- `` — 29.0% of jets, value 0.09 (0.00…0.25), formula right 78%

```
z = -0.213
if mass < 92.86: z += -0.040 × (92.86 − mass)
if z_top30_slots > 0.905: z += 9.76 × (z_top30_slots − 0.905)
if sum_pt > 1017: z += 0.0085 × (sum_pt − 1017)
if girth2_top15 < 0.016: z += 38.73 × (0.016 − girth2_top15)
if mass_over_sum_pt < 0.098: z += 15.62 × (0.098 − mass_over_sum_pt)
if mass_over_sum_pt < 0.141: z += -5.80 × (0.141 − mass_over_sum_pt)
if sum_pt > 1066: z += -0.011 × (sum_pt − 1066)
if sum_pt_top50 > 997: z += -0.0051 × (sum_pt_top50 − 997)
if mass_top50 < 86.40: z += 0.015 × (86.40 − mass_top50)
if mass_over_sum_pt < 0.074: z += 20.64 × (0.074 − mass_over_sum_pt)
if lam2 < 0.00097: z += -491 × (0.00097 − lam2)
if log_sum_pt > 6.90 and girth2_top10 < 0.020: z += 185 × (log_sum_pt − 6.90) × (0.020 − girth2_top10)
if sum_pt_top50 > 1039: z += 0.0042 × (sum_pt_top50 − 1039)
if log_sum_pt > 7.06: z += 12.84 × (log_sum_pt − 7.06)
if sum_pt_top40 > 1013: z += -0.0034 × (sum_pt_top40 − 1013)
if sum_pt_top40 > 1070: z += 0.0056 × (sum_pt_top40 − 1070)
if girth2_top15 < 0.0061: z += -51.32 × (0.0061 − girth2_top15)
if sum_pt_top50 > 1107: z += -0.0054 × (sum_pt_top50 − 1107)
if mass_top30 < 91.70: z += 0.0038 × (91.70 − mass_top30)
if mass_top40 > 161: z += -0.106 × (mass_top40 − 161)
if lam1 < 0.0062: z += -43.99 × (0.0062 − lam1)
if sum_pt_top30 > 1019: z += -0.0022 × (sum_pt_top30 − 1019)
if sum_pt_top20 > 909: z += 0.0009 × (sum_pt_top20 − 909)
if log_sum_pt > 6.90: z += 0.916 × (log_sum_pt − 6.90)
if sum_pt > 996: z += -0.00081 × (sum_pt − 996)
if sum_pt_top50 > 1107 and C2 > 0.098: z += 0.829 × (sum_pt_top50 − 1107) × (C2 − 0.098)
if log_sum_pt > 7.06 and mass_top20 > 66.84: z += 0.192 × (log_sum_pt − 7.06) × (mass_top20 − 66.84)
if mass_top40 > 161 and pt_9 < 41.44: z += 0.0048 × (mass_top40 − 161) × (41.44 − pt_9)
if n_pt_above_10 > 18.00 and z_4 > 0.036: z += 0.686 × (n_pt_above_10 − 18.00) × (z_4 − 0.036)
if log_sum_pt > 6.90 and mass_top50 > 85.87: z += -0.033 × (log_sum_pt − 6.90) × (mass_top50 − 85.87)
if max_dr < 0.402: z += 0.556 × (0.402 − max_dr)
if sum_pt_top50 > 1039 and dr_3 > 0.0046: z += -0.021 × (sum_pt_top50 − 1039) × (dr_3 − 0.0046)
if mass < 92.86 and sum_pt_top40 < 1070: z += -2.5e-05 × (92.86 − mass) × (1070 − sum_pt_top40)
if log_sum_pt > 6.90 and n_dr_0_0p05 < 18.00: z += 0.093 × (log_sum_pt − 6.90) × (18.00 − n_dr_0_0p05)
if sum_pt_top20 > 1129: z += -0.0037 × (sum_pt_top20 − 1129)
if sum_pt_top50 > 997 and mean_phi > 0.00011: z += 3.56 × (sum_pt_top50 − 997) × (mean_phi − 0.00011)
if sum_pt_top20 > 1129 and girth2_top3 < 0.0079: z += 0.146 × (sum_pt_top20 − 1129) × (0.0079 − girth2_top3)
if sum_pt_top40 > 1013 and eta_5 < -0.114: z += -0.077 × (sum_pt_top40 − 1013) × (-0.114 − eta_5)
if sum_pt > 1066 and pt_6 < 36.81: z += 6.4e-05 × (sum_pt − 1066) × (36.81 − pt_6)
if sum_pt_top20 > 1129 and eta_1 > 0.041: z += -0.124 × (sum_pt_top20 − 1129) × (eta_1 − 0.041)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 37.7% of jets, neuron 0.42, formula right for 86%.  
- **group 2** — 26.1% of jets, neuron 0.11, formula right for 80%.  
- **group 3** — 13.3% of jets, neuron 0.36, formula right for 78%.  
- **group 4** — 10.7% of jets, neuron 0.23, formula right for 66%.  
- **group 5** — 7.6% of jets, neuron 0.61, formula right for 83%.  
- **group 6** — 3.1% of jets, neuron 1.41, formula right for 87%.  
- **group 7** — 0.8% of jets, neuron 2.14, formula right for 91%.  
- **group 8** — 0.4% of jets, neuron 4.26, formula right for 84%.  
- **group 9** — 0.2% of jets, neuron 0.65, formula right for 83%.  
- **group 10** — 0.0% of jets, neuron 3.61, formula right for 88%.  

### neuron 11: Clean two-prong jet, 80-93 GeV (minor)

- **What it measures:** Large for mass below 92.9 GeV but pushed down below 80.4 GeV, so it peaks around 80-93 GeV, especially with few particles at 0.2 <= ΔR < 0.4, a small e2 and an elongated two-prong pattern; very narrow jets are pushed down. W jets sit highest (1.68; AUC 0.82), Z jets next (1.01), top (0.37), quark (0.26) and gluon (0.24) jets low.
- *computed — its value:* largest for W (1.68), then Z (1.01), then t (0.37), then q (0.26), then g (0.24); it separates W jets from the rest best (AUC 0.82: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the q score (-7%): a clean two-prong jet at the W mass is a W, not a quark jet. The g, Z and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of q (-7%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.759):

- `` — 8.8% of jets, value 2.73 (1.75…3.75), formula right 95%
- `` — 6.4% of jets, value 2.01 (1.25…2.75), formula right 98%
- `` — 4.9% of jets, value 1.54 (0.62…2.50), formula right 91%
- `` — 2.0% of jets, value 0.70 (0.38…1.12), formula right 85%
- `` — 7.6% of jets, value 0.62 (0.00…1.44), formula right 81%
- `` — 7.9% of jets, value 0.57 (0.00…1.25), formula right 78%
- `` — 2.0% of jets, value 0.38 (0.00…1.19), formula right 79%
- `` — 60.4% of jets, value 0.27 (0.00…0.62), formula right 77%

```
z = -0.106
if mass < 92.86: z += 0.064 × (92.86 − mass)
if girth2_top30 < 0.0064: z += -337 × (0.0064 − girth2_top30)
if mass_top50 < 86.40: z += -0.034 × (86.40 − mass_top50)
if mass_over_sum_pt_sq < 0.0082: z += 159 × (0.0082 − mass_over_sum_pt_sq)
if mass < 80.40: z += -0.031 × (80.40 − mass)
if mass < 101: z += 0.014 × (101 − mass)
if n_dr_0p2_0p4 < 10.00 and n_dr_0p1_0p2 < 21.00: z += 0.0066 × (10.00 − n_dr_0p2_0p4) × (21.00 − n_dr_0p1_0p2)
if e2 < 0.028: z += 46.88 × (0.028 − e2)
if n_dr_0p2_0p4 < 10.00: z += -0.063 × (10.00 − n_dr_0p2_0p4)
if mass < 101 and z_dr_0p1_0p2 < 0.334: z += -0.034 × (101 − mass) × (0.334 − z_dr_0p1_0p2)
if lam1 < 0.0067: z += -107 × (0.0067 − lam1)
if girth2_top30 < 0.0058: z += 136 × (0.0058 − girth2_top30)
if n_dr_0p2_0p4 < 10.00 and z_dr_0p2_0p4 < 0.068: z += 0.803 × (10.00 − n_dr_0p2_0p4) × (0.068 − z_dr_0p2_0p4)
if width < 0.0062: z += -143 × (0.0062 − width)
if mass < 80.40 and z_dr_0p2_0p4 < 0.037: z += -0.585 × (80.40 − mass) × (0.037 − z_dr_0p2_0p4)
if z_dr_0p2_0p4 < 0.0064: z += 112 × (0.0064 − z_dr_0p2_0p4)
if n_dr_0p2_0p4 < 10.00 and girth2 < 0.0055: z += -31.18 × (10.00 − n_dr_0p2_0p4) × (0.0055 − girth2)
if mass_top50 < 86.40 and girth2_top15 < 0.0042: z += 3.59 × (86.40 − mass_top50) × (0.0042 − girth2_top15)
if z_top5_slots > 0.535: z += -1.73 × (z_top5_slots − 0.535)
if e2 < 0.025: z += 28.65 × (0.025 − e2)
if z_dr_0_0p05 < 0.096: z += 5.82 × (0.096 − z_dr_0_0p05)
if e2 < 0.039: z += -6.48 × (0.039 − e2)
if girth2_top5 < 0.0072: z += -20.32 × (0.0072 − girth2_top5)
if mass < 101 and planar_flow < 0.356: z += 0.084 × (101 − mass) × (0.356 − planar_flow)
if n_dr_0p2_0p4 < 10.00 and z_dr_0p05_0p1 > 0.591: z += -0.322 × (10.00 − n_dr_0p2_0p4) × (z_dr_0p05_0p1 − 0.591)
if n_dr_0p2_0p4 < 10.00 and tau21 > 0.130: z += 0.062 × (10.00 − n_dr_0p2_0p4) × (tau21 − 0.130)
if mass < 80.40 and sum_pt_top2 < 465: z += -7.9e-05 × (80.40 − mass) × (465 − sum_pt_top2)
if mass < 80.40 and D2 < 3.35: z += -0.017 × (80.40 − mass) × (3.35 − D2)
if mass_top30 < 60.44: z += 0.0082 × (60.44 − mass_top30)
if n_dr_0p2_0p4 < 5.00: z += 0.046 × (5.00 − n_dr_0p2_0p4)
if mass < 101 and mass_top10 > 44.19: z += 0.00053 × (101 − mass) × (mass_top10 − 44.19)
if max_dr < 0.298: z += 3.09 × (0.298 − max_dr)
if z_top30_slots > 0.980: z += 5.97 × (z_top30_slots − 0.980)
if girth2_top10 < 0.0013: z += 128 × (0.0013 − girth2_top10)
if mass < 101 and n_dr_0p2_0p4 > 10.00: z += -0.003 × (101 − mass) × (n_dr_0p2_0p4 − 10.00)
if n_dr_0p2_0p4 < 5.00 and n_real_top30 > 22.00: z += 0.0042 × (5.00 − n_dr_0p2_0p4) × (n_real_top30 − 22.00)
if mass < 92.86 and D2 < 1.23: z += 0.043 × (92.86 − mass) × (1.23 − D2)
if z_dr_0p05_0p1 > 0.851: z += 6.47 × (z_dr_0p05_0p1 − 0.851)
if n_dr_0p2_0p4 < 10.00 and n_dr_0p1_0p2 < 9.00: z += -0.0027 × (10.00 − n_dr_0p2_0p4) × (9.00 − n_dr_0p1_0p2)
if log_sum_pt < 6.89: z += 1.34 × (6.89 − log_sum_pt)
if z_dr_0_0p05 < 0.096 and max_dr > 0.194: z += 0.815 × (0.096 − z_dr_0_0p05) × (max_dr − 0.194)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 34.5% of jets, neuron 0.26, formula right for 80%.  
- **group 2** — 10.1% of jets, neuron 1.57, formula right for 95%.  
- **group 3** — 10.1% of jets, neuron 0.64, formula right for 73%.  
- **group 4** — 8.6% of jets, neuron 2.35, formula right for 94%.  
- **group 5** — 7.7% of jets, neuron 0.18, formula right for 75%.  
- **group 6** — 7.7% of jets, neuron 0.19, formula right for 73%.  
- **group 7** — 6.6% of jets, neuron 0.24, formula right for 68%.  
- **group 8** — 5.7% of jets, neuron 0.45, formula right for 83%.  
- **group 9** — 4.6% of jets, neuron 0.30, formula right for 74%.  
- **group 10** — 4.3% of jets, neuron 2.76, formula right for 96%.  

### neuron 15: Narrow hard core (weak) (minor)

- **What it measures:** Rises when the jet is narrow with little pT at 0.2 <= ΔR < 0.4 and the pT concentrated close to the axis (large share within ΔR < 0.05, small spread of the 5 hardest, small LHA). It is small for all types: quark jets slightly highest (0.31), top jets lowest (0.10).
- *computed — its value:* largest for q (0.31), then g (0.20), then W (0.18), then Z (0.18), then t (0.10); it separates q jets from the rest best (AUC 0.64: large for q)
- **How the class scores use it:** It raises the Z score slightly (+2%); the g, q, W and t scores hardly use it.
- *computed — used by:* raises the score of Z (+2%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.428):

- `` — 2.1% of jets, value 1.45 (0.56…2.50), formula right 65%
- `` — 30.8% of jets, value 0.37 (0.00…0.88), formula right 74%
- `` — 5.0% of jets, value 0.25 (0.00…0.69), formula right 84%
- `` — 20.7% of jets, value 0.12 (0.00…0.44), formula right 81%
- `` — 3.3% of jets, value 0.09 (0.00…0.31), formula right 84%
- `` — 4.0% of jets, value 0.04 (0.00…0.19), formula right 92%
- `` — 34.1% of jets, value 0.01 (0.00…0.00), formula right 87%

```
z = -0.661
if girth2_top50 < 0.014 and z_dr_0p2_0p4 < 0.129: z += 2575 × (0.014 − girth2_top50) × (0.129 − z_dr_0p2_0p4)
if girth2_top50 < 0.014: z += -207 × (0.014 − girth2_top50)
if e2_sq < 0.0069 and z_dr_0p2_0p4 < 0.129: z += 6786 × (0.0069 − e2_sq) × (0.129 − z_dr_0p2_0p4)
if girth < 0.121: z += 23.90 × (0.121 − girth)
if e2_sq < 0.0069: z += -627 × (0.0069 − e2_sq)
if girth < 0.121 and z_dr_0p2_0p4 < 0.129: z += -146 × (0.121 − girth) × (0.129 − z_dr_0p2_0p4)
if lam1 < 0.016: z += -79.61 × (0.016 − lam1)
if girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194: z += 6641 × (0.0015 − girth2_top5) × (0.194 − z_dr_0p2_0p4)
if log_sum_pt < 7.02: z += 5.10 × (7.02 − log_sum_pt)
if girth2_top5 < 0.0015: z += -976 × (0.0015 − girth2_top5)
if z_dr_0p2_0p4 < 0.020: z += -54.10 × (0.020 − z_dr_0p2_0p4)
if girth2_top10 < 0.0077: z += 114 × (0.0077 − girth2_top10)
if mass_top50 < 71.80: z += -0.030 × (71.80 − mass_top50)
if z_dr_0p1_0p2 < 0.120 and n_dr_0p2_0p4 < 26.00: z += 0.282 × (0.120 − z_dr_0p1_0p2) × (26.00 − n_dr_0p2_0p4)
if girth < 0.050: z += -27.70 × (0.050 − girth)
if e2_sq < 0.0069 and z_dr_0p1_0p2 < 0.219: z += -689 × (0.0069 − e2_sq) × (0.219 − z_dr_0p1_0p2)
if z_dr_0p1_0p2 < 0.187: z += 2.23 × (0.187 − z_dr_0p1_0p2)
if girth2_top5 < 0.0072: z += 51.44 × (0.0072 − girth2_top5)
if lam2 < 0.0018: z += 196 × (0.0018 − lam2)
if dr_0 < 0.052: z += -9.06 × (0.052 − dr_0)
if sum_pt < 1002: z += -0.0095 × (1002 − sum_pt)
if mass_top30 < 80.40: z += 0.0087 × (80.40 − mass_top30)
if dr_3 < 0.053: z += -7.18 × (0.053 − dr_3)
if z_dr_0p1_0p2 < 0.120 and sum_pt_top3 < 788: z += -0.007 × (0.120 − z_dr_0p1_0p2) × (788 − sum_pt_top3)
if z_dr_0p1_0p2 < 0.120 and z_dr_0p2_0p4 < 0.037: z += -96.13 × (0.120 − z_dr_0p1_0p2) × (0.037 − z_dr_0p2_0p4)
if girth2_top10 < 0.0077 and z_top30_slots < 0.986: z += -943 × (0.0077 − girth2_top10) × (0.986 − z_top30_slots)
if z_dr_0p1_0p2 < 0.120 and mass_top5 < 40.20: z += 0.062 × (0.120 − z_dr_0p1_0p2) × (40.20 − mass_top5)
if girth2_top5 < 0.0072 and max_dr < 0.332: z += -1396 × (0.0072 − girth2_top5) × (0.332 − max_dr)
if log_sum_pt < 7.02 and z_dr_0p2_0p4 < 0.020: z += 136 × (7.02 − log_sum_pt) × (0.020 − z_dr_0p2_0p4)
if girth2_top5 < 0.0072 and sum_pt_top40 < 985: z += 1.43 × (0.0072 − girth2_top5) × (985 − sum_pt_top40)
if girth2_top5 < 0.0015 and n_dr_0p2_0p4 > 0: z += -18.02 × (0.0015 − girth2_top5) × (n_dr_0p2_0p4 − 0)
if C2 < 0.088: z += -1.63 × (0.088 − C2)
if girth2_top10 < 0.0032: z += -46.27 × (0.0032 − girth2_top10)
if z_dr_0_0p05 < 0.846 and log_sum_pt > 6.91: z += 2.91 × (0.846 − z_dr_0_0p05) × (log_sum_pt − 6.91)
if z_dr_0p1_0p2 < 0.120 and planar_flow < 0.826: z += 3.05 × (0.120 − z_dr_0p1_0p2) × (0.826 − planar_flow)
if sum_pt < 1002 and tau32 < 0.662: z += -0.034 × (1002 − sum_pt) × (0.662 − tau32)
if z_dr_0p05_0p1 > 0.851: z += -5.85 × (z_dr_0p05_0p1 − 0.851)
if z_dr_0p1_0p2 < 0.120 and n_dr_0p4_up > 0: z += -3.02 × (0.120 − z_dr_0p1_0p2) × (n_dr_0p4_up − 0)
if sum_pt_top50 < 934: z += 0.0024 × (934 − sum_pt_top50)
if sum_pt < 1002 and z_dr_0p2_0p4 < 0.020: z += -0.232 × (1002 − sum_pt) × (0.020 − z_dr_0p2_0p4)
if z_dr_0_0p05 < 0.846 and sum_pt_top40 > 1096: z += -0.0036 × (0.846 − z_dr_0_0p05) × (sum_pt_top40 − 1096)
if girth < 0.050 and z_dr_0p2_0p4 > 0.068: z += -9732 × (0.050 − girth) × (z_dr_0p2_0p4 − 0.068)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 21.5% of jets, neuron 0.08, formula right for 82%.  
- **group 2** — 15.8% of jets, neuron 0.04, formula right for 88%.  
- **group 3** — 15.1% of jets, neuron 0.13, formula right for 91%.  
- **group 4** — 11.2% of jets, neuron 0.16, formula right for 80%.  
- **group 5** — 10.3% of jets, neuron 0.37, formula right for 79%.  
- **group 6** — 9.1% of jets, neuron 0.27, formula right for 75%.  
- **group 7** — 6.4% of jets, neuron 0.35, formula right for 72%.  
- **group 8** — 5.4% of jets, neuron 0.32, formula right for 70%.  
- **group 9** — 3.9% of jets, neuron 0.39, formula right for 72%.  
- **group 10** — 1.3% of jets, neuron 1.05, formula right for 60%.  
