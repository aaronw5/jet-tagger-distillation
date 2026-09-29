# What each part of the 150-term formula does (64 particles)

*the formula simplified by hand with the training data (main result)*. Validation accuracy 81.07%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Each jet is placed on 16 scales, and each class score adds some scales and subtracts others. Gluon and quark jets are told apart mainly by gluon-likeness (many particles sharing the pT thinly, neuron 1), which the g score adds and the q score subtracts, while both light-jet scores subtract heavy two-prong-ness (neuron 4) and add the light one-prong scales (9, 6, 12). W and Z jets are recognised by a compact, few-particle, sparse jet (neurons 5 and 3, added to both boson scores) and above all by the absence of busy, wide radiation (neuron 8), which both boson scores subtract strongly. W and Z are then split by mass: W-side mass (0) and the clean two-prong scale (11) raise the W score, the 91.2-101 GeV window (7) raises the Z score, mass just above the Z (14) lowers the W score, and W-side mass (0) and one-prong-ness outside the Z window (6) lower the Z score. Top jets are recognised by hard particles spread far from the axis in a heavy jet (neuron 10) and by busy, wide radiation (8), while the t score is pulled down strongly by the high-pT scale (13), on which top jets sit lowest.

## The 5 class scores

### score g: Many soft particles, no heavy prongs

High for particle-rich, round one-prong jets without a heavy two-prong mass: gluon jets score highest (mean 3.01 for g, AUC 0.93), quark jets next (0.75 for q), top jets near zero (0.11 for t), Z and W jets far below (-2.02 and -2.19).

Adds gluon-likeness (neuron 1, +45%) and light one-prong (9, +10%), with small additions from 6 and 12 (+3% each); subtracts heavy two-prong-ness (4, -20%), the sparse, thin scale (3, -9%) and the compact-core scale (5, -8%).

*computed:* largest for g (3.01), then q (0.75), then t (0.11), then Z (-2.02), then W (-2.19); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light one-prong jet, not particle-rich

High for light one-prong jets that are not particle-rich: quark jets score highest (mean 2.64 for q, AUC 0.89), gluon jets next (1.18 for g), top jets near zero (0.23 for t), Z and W jets below zero (-0.75 and -0.76).

Subtracts heavy two-prong-ness (neuron 4, -38%), gluon-likeness (1, -21%) and the clean two-prong scale (11, -7%); adds light one-prong (9, +17%), lightness (12, +7%) and one-prong away from the Z window (6, +7%).

*computed:* largest for q (2.64), then g (1.18), then t (0.23), then Z (-0.75), then W (-0.76); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact two-prong jet at the W mass

High for compact two-prong jets with a quiet outer ring and mass up to about the W: W jets score highest (mean 3.24 for W, AUC 0.97), quark jets just below zero (-0.19), Z jets lower (-0.87), gluon (-2.05) and especially top jets (-5.37) far below.

Subtracts busy wide radiation (neuron 8, -34%) and mass just above the Z (14, -13%), plus small amounts of 7 (-6%), 12 (-4%) and 9 (-3%); adds the compact-core scale (5, +11%), the clean two-prong scale (11, +9%), W-side mass (0, +8%), heavy two-prong-ness (4, +6%) and the sparse, thin scale (3, +4%).

*computed:* largest for W (3.24), then q (-0.19), then Z (-0.87), then g (-2.05), then t (-5.37); it separates W jets from the rest best (AUC 0.97: large for W)

### score Z: Compact two-prong jet at the Z mass

High for compact two-prong jets at or just above 91.2 GeV: Z jets score highest (mean 3.39 for Z, AUC 0.95), quark and W jets slightly below zero (-0.10 and -0.26), gluon (-1.86) and especially top jets (-5.72) far below.

Subtracts busy wide radiation (neuron 8, -36%), one-prong away from the Z window (6, -15%), W-side mass (0, -14%) and a little of 2 (-3%); adds the compact-core scale (5, +13%), the 91.2-101 GeV window (7, +8%) and small amounts of 3 (+4%), 12 (+3%) and 15 (+3%).

*computed:* largest for Z (3.39), then q (-0.10), then W (-0.26), then g (-1.86), then t (-5.72); it separates Z jets from the rest best (AUC 0.95: large for Z)

### score t: Heavy jet with widely spread hard prongs

High for heavy jets whose hard particles are spread far from the axis: top jets score highest (mean 3.26 for t, AUC 0.95), W and gluon jets slightly below zero (-0.30 and -0.40), Z and quark jets further below (-0.88 and -1.08).

Subtracts the high-pT scale (neuron 13, -37%, on which top jets sit lowest), the compact-core scale (5, -11%) and small amounts of 12 (-4%) and 7 (-3%); adds spread-out hard particles (10, +28%), busy wide radiation (8, +9%) and a little heavy two-prong-ness (4, +4%).

*computed:* largest for t (3.26), then W (-0.30), then g (-0.40), then Z (-0.88), then q (-1.08); it separates t jets from the rest best (AUC 0.95: large for t)

## The 16 neurons (most important first)

### neuron 1: Gluon-likeness: many soft particles (major)

- **What it measures:** Grows with the number of particles and with how thinly the pT is shared (a small pT share held by the 20-40 hardest particles); it is also pushed up when the natural log of the total pT is above 6.91, when the 2 hardest particles carry less than 662 GeV and when τ32 is above 0.276. Gluon jets sit far highest (mean 5.82 for g, AUC 0.92), top jets next (2.84 for t), then quark (2.26 for q), Z (1.81 for Z) and W jets (1.72 for W).
- *computed — its value:* largest for g (5.82), then t (2.84), then q (2.26), then Z (1.81), then W (1.72); it separates g jets from the rest best (AUC 0.92: large for g)
- **How the class scores use it:** It raises the g score (+45%) and lowers the q score (-21%), making it the main gluon-versus-quark handle; freezing it costs 11.968 points. The W, Z and t scores hardly use it, even though top jets are fairly high on it.
- *computed — used by:* raises the score of g (+45%); lowers the score of q (-21%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.69):

- `` — 3.9% of jets, value 7.93 (6.22…9.69), formula right 94%
- `` — 7.2% of jets, value 6.18 (4.03…8.28), formula right 80%
- `` — 4.8% of jets, value 5.40 (3.66…7.09), formula right 71%
- `` — 2.4% of jets, value 5.12 (3.44…7.12), formula right 77%
- `` — 5.1% of jets, value 3.53 (2.22…5.23), formula right 77%
- `` — 20.7% of jets, value 3.30 (1.62…5.09), formula right 77%
- `` — 29.2% of jets, value 2.05 (0.59…3.72), formula right 80%
- `` — 26.7% of jets, value 1.13 (0.00…2.31), formula right 86%

```
z = 1.15
if log_sum_pt > 6.91: z += 42.30 × (log_sum_pt − 6.91)
if sum_pt_top2 < 662: z += 0.0035 × (662 − sum_pt_top2)
if sum_pt_top50 > 930: z += -0.0091 × (sum_pt_top50 − 930)
if z_top50_slots > 0.960: z += -29.50 × (z_top50_slots − 0.960)
if tau32 > 0.276: z += 2.15 × (tau32 − 0.276)
if log_sum_pt > 6.99: z += -22.40 × (log_sum_pt − 6.99)
if n_particles > 38.60 and dr_0 < 0.136: z += 0.550 × (n_particles − 38.60) × (0.136 − dr_0)
if z_top20_slots < 0.931: z += -5.47 × (0.931 − z_top20_slots)
if sum_pt_top30 < 1070: z += 0.0031 × (1070 − sum_pt_top30)
if pt_9 < 33.60: z += -0.034 × (33.60 − pt_9)
if max_dr < 0.435: z += -3.12 × (0.435 − max_dr)
if sum_pt_top2 < 748 and n_dr_0p2_0p4 < 7.95: z += -0.00024 × (748 − sum_pt_top2) × (7.95 − n_dr_0p2_0p4)
if girth2_top3 < 0.00073: z += 1170 × (0.00073 − girth2_top3)
if mass_top20 < 48.50 and n_real_top40 > 27.00: z += 0.0032 × (48.50 − mass_top20) × (n_real_top40 − 27.00)
if n_particles > 37.90 and z_top50_slots < 0.985: z += 1.48 × (n_particles − 37.90) × (0.985 − z_top50_slots)
if girth2_top3 < 0.00099 and n_dr_0p05_0p1 < 9.47: z += -80.00 × (0.00099 − girth2_top3) × (9.47 − n_dr_0p05_0p1)
if n_dr_0p1_0p2 < 6.42 and dr_6 < 0.075: z += -1.39 × (6.42 − n_dr_0p1_0p2) × (0.075 − dr_6)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.2% of jets, neuron 2.09, formula right for 80%.  
- **group 2** — 20.6% of jets, neuron 2.23, formula right for 84%.  
- **group 3** — 18.3% of jets, neuron 1.01, formula right for 80%.  
- **group 4** — 10.5% of jets, neuron 4.08, formula right for 79%.  
- **group 5** — 9.7% of jets, neuron 4.28, formula right for 75%.  
- **group 6** — 6.1% of jets, neuron 5.28, formula right for 80%.  
- **group 7** — 4.0% of jets, neuron 6.21, formula right for 83%.  
- **group 8** — 2.3% of jets, neuron 7.19, formula right for 87%.  
- **group 9** — 1.1% of jets, neuron 8.05, formula right for 90%.  
- **group 10** — 0.3% of jets, neuron 8.66, formula right for 90%.  

### neuron 4: Heavy two-prong-ness (major)

- **What it measures:** Grows with the mass of the 10-20 hardest particles and with e2, and with small D2 (a two-prong pattern): it is pushed up for mass between 80.4 and 109 GeV and for D2 below 6.52, and pushed down for jets with more than 22.7 particles. Z and W jets sit highest (1.58 for Z, 1.57 for W), top jets next (1.06 for t), gluon jets (0.29 for g) and quark jets lowest (0.16 for q; AUC 0.17: small for q).
- *computed — its value:* largest for Z (1.58), then W (1.57), then t (1.06), then g (0.29), then q (0.16); it separates q jets from the rest best (AUC 0.17: small for q)
- **How the class scores use it:** It lowers the q (-38%) and g (-20%) scores strongly, since a heavy pronged jet is not a light quark or gluon jet, and it raises the W (+6%) and t (+4%) scores a little; freezing it costs 1.052 points. The Z score hardly uses it.
- *computed — used by:* raises the score of W (+6%), t (+4%); lowers the score of g (-20%), q (-38%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.763):

- `` — 12.8% of jets, value 2.23 (1.75…2.75), formula right 97%
- `` — 8.4% of jets, value 1.54 (0.88…2.12), formula right 91%
- `` — 25.1% of jets, value 1.43 (0.69…2.06), formula right 84%
- `` — 20.1% of jets, value 0.71 (0.00…1.44), formula right 74%
- `` — 3.4% of jets, value 0.49 (0.00…1.06), formula right 74%
- `` — 2.0% of jets, value 0.18 (0.00…0.62), formula right 65%
- `` — 28.2% of jets, value 0.00 (0.00…0.00), formula right 73%

```
z = 0.494
if mass < 80.40: z += -0.139 × (80.40 − mass)
if mass < 109: z += 0.046 × (109 − mass)
if D2 < 6.52: z += 0.258 × (6.52 − D2)
if mass_top40 < 84.90 and D2 < 6.27: z += -0.016 × (84.90 − mass_top40) × (6.27 − D2)
if n_particles > 22.70: z += -0.018 × (n_particles − 22.70)
if sum_pt < 1070: z += -0.006 × (1070 − sum_pt)
if girth < 0.063: z += -22.70 × (0.063 − girth)
if mass < 124 and max_dr < 0.401: z += 0.110 × (124 − mass) × (0.401 − max_dr)
if z_dr_0p2_0p4 < 0.070: z += -6.43 × (0.070 − z_dr_0p2_0p4)
if tau21 < 0.451: z += -1.65 × (0.451 − tau21)
if mass < 74.40 and max_dr < 0.350: z += -0.652 × (74.40 − mass) × (0.350 − max_dr)
if C2 > 0.106: z += 18.10 × (C2 − 0.106)
if tau32 < 0.578: z += 2.44 × (0.578 − tau32)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.4% of jets, neuron 1.84, formula right for 92%.  
- **group 2** — 17.7% of jets, neuron 0.70, formula right for 75%.  
- **group 3** — 13.2% of jets, neuron 0.78, formula right for 72%.  
- **group 4** — 9.4% of jets, neuron 1.39, formula right for 87%.  
- **group 5** — 6.6% of jets, neuron 0.01, formula right for 66%.  
- **group 6** — 6.4% of jets, neuron 0.00, formula right for 81%.  
- **group 7** — 5.0% of jets, neuron 0.00, formula right for 70%.  
- **group 8** — 5.0% of jets, neuron 0.00, formula right for 74%.  
- **group 9** — 4.0% of jets, neuron 0.00, formula right for 75%.  
- **group 10** — 1.2% of jets, neuron 0.00, formula right for 85%.  

### neuron 5: Compact core, few particles (major)

- **What it measures:** Pushed up when the natural log of the total pT is above 6.85 but mostly cut back again above 6.91, and pushed up for mass above 63.2 GeV; it grows as the particle count falls and the hardest particles carry more of the pT. Quark jets sit highest (1.45 for q), Z jets (1.31 for Z) and W jets (1.15 for W) next, top (0.59 for t) and gluon jets lowest (0.54 for g; AUC 0.28: small for g).
- *computed — its value:* largest for q (1.45), then Z (1.31), then W (1.15), then t (0.59), then g (0.54); it separates g jets from the rest best (AUC 0.28: small for g)
- **How the class scores use it:** It raises the W (+11%) and Z (+13%) scores and lowers the t (-11%) and g (-8%) scores: a compact, few-particle jet with some mass is boson-like rather than top- or gluon-like. The q score hardly uses it, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+11%), Z (+13%); lowers the score of g (-8%), t (-11%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.568):

- `` — 61.5% of jets, value 1.47 (0.81…2.19), formula right 81%
- `` — 2.9% of jets, value 0.66 (0.00…1.25), formula right 75%
- `` — 4.1% of jets, value 0.63 (0.00…1.25), formula right 77%
- `` — 2.0% of jets, value 0.35 (0.00…0.88), formula right 79%
- `` — 2.0% of jets, value 0.28 (0.00…0.94), formula right 81%
- `` — 10.8% of jets, value 0.26 (0.00…0.69), formula right 72%
- `` — 6.0% of jets, value 0.13 (0.00…0.56), formula right 93%
- `` — 10.7% of jets, value 0.05 (0.00…0.19), formula right 84%

```
z = -0.865
if log_sum_pt > 6.85: z += 30.20 × (log_sum_pt − 6.85)
if log_sum_pt > 6.91: z += -43.80 × (log_sum_pt − 6.91)
if mass > 63.20: z += 0.028 × (mass − 63.20)
if n_particles < 67.60 and e2 < 0.039: z += 1.47 × (67.60 − n_particles) × (0.039 − e2)
if mass_top50 > 102: z += -0.031 × (mass_top50 − 102)
if mass_top50 > 157: z += -0.209 × (mass_top50 − 157)
if z_top30_slots > 0.943: z += -5.92 × (z_top30_slots − 0.943)
if max_dr > 0.436: z += 19.70 × (max_dr − 0.436)
if girth2_top30 < 0.023 and tau32 < 0.858: z += 77.30 × (0.023 − girth2_top30) × (0.858 − tau32)
if mass_over_sum_pt_sq > 0.029: z += -448 × (mass_over_sum_pt_sq − 0.029)
if mass_top50 > 156 and z_dr_0p05_0p1 > 0.587: z += 1.32 × (mass_top50 − 156) × (z_dr_0p05_0p1 − 0.587)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 35.2% of jets, neuron 1.26, formula right for 76%.  
- **group 2** — 29.1% of jets, neuron 1.37, formula right for 83%.  
- **group 3** — 11.7% of jets, neuron 0.58, formula right for 79%.  
- **group 4** — 11.2% of jets, neuron 0.54, formula right for 90%.  
- **group 5** — 6.0% of jets, neuron 0.07, formula right for 82%.  
- **group 6** — 2.9% of jets, neuron 0.04, formula right for 87%.  
- **group 7** — 1.3% of jets, neuron 1.97, formula right for 82%.  
- **group 8** — 1.3% of jets, neuron 0.00, formula right for 74%.  
- **group 9** — 0.9% of jets, neuron 0.01, formula right for 90%.  
- **group 10** — 0.3% of jets, neuron 0.03, formula right for 80%.  

### neuron 8: Busy, wide radiation pattern (major)

- **What it measures:** Grows with the number and pT share of particles at 0.2 <= ΔR < 0.4, with the minor-axis width lam2 and with the total particle count; it is also pushed up for mass above 91.2 GeV (partly undone above 104 GeV) and for a mass-to-pT ratio above 0.0762 with total pT below 1130 GeV. Top jets sit far highest (mean 6.55 for t, AUC 0.91), gluon jets next (2.14 for g), quark jets lower (0.97 for q), Z (0.28 for Z) and W jets near zero (0.09 for W).
- *computed — its value:* largest for t (6.55), then g (2.14), then q (0.97), then Z (0.28), then W (0.09); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W (-34%) and Z (-36%) scores strongly, because a two-prong boson is compact with a quiet outer ring, and it raises the t score (+9%); freezing it costs 2.34 points. The g and q scores hardly use it.
- *computed — used by:* raises the score of t (+9%); lowers the score of W (-34%), Z (-36%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.806):

- `` — 10.4% of jets, value 9.32 (4.77…13.75), formula right 86%
- `` — 4.7% of jets, value 6.73 (3.81…10.44), formula right 71%
- `` — 4.1% of jets, value 5.42 (3.19…7.38), formula right 81%
- `` — 2.0% of jets, value 4.46 (0.00…9.56), formula right 66%
- `` — 4.4% of jets, value 2.75 (1.56…4.06), formula right 75%
- `` — 3.2% of jets, value 2.35 (0.88…3.81), formula right 67%
- `` — 71.2% of jets, value 0.29 (0.00…1.00), formula right 82%

```
z = 0.721
if mass > 91.20: z += 0.105 × (mass − 91.20)
if mass_over_sum_pt > 0.076 and sum_pt < 1130: z += 0.477 × (mass_over_sum_pt − 0.076) × (1130 − sum_pt)
if mass > 104: z += -0.109 × (mass − 104)
if n_dr_0p2_0p4 < 20.20: z += -0.068 × (20.20 − n_dr_0p2_0p4)
if n_particles > 49.30: z += 0.076 × (n_particles − 49.30)
if sum_pt < 1010: z += 0.014 × (1010 − sum_pt)
if lam2 < 0.0014: z += -343 × (0.0014 − lam2)
if girth2_top40 > 0.0014 and sum_pt_top30 < 932: z += 0.687 × (girth2_top40 − 0.0014) × (932 − sum_pt_top30)
if girth2_top30 < 0.0081: z += 74.30 × (0.0081 − girth2_top30)
if sum_pt_top40 < 1040 and max_dr < 0.415: z += -0.064 × (1040 − sum_pt_top40) × (0.415 − max_dr)
if n_particles > 50.60 and z_top50_slots > 0.972: z += -3.12 × (n_particles − 50.60) × (z_top50_slots − 0.972)
if C2 > 0.066: z += 7.66 × (C2 − 0.066)
if sum_pt_top40 < 958 and log_sum_pt < 6.82: z += 0.069 × (958 − sum_pt_top40) × (6.82 − log_sum_pt)
if sum_pt < 1010 and n_real_top50 < 43.20: z += -0.001 × (1010 − sum_pt) × (43.20 − n_real_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 75.2% of jets, neuron 0.42, formula right for 81%.  
- **group 2** — 6.9% of jets, neuron 9.02, formula right for 94%.  
- **group 3** — 4.2% of jets, neuron 3.53, formula right for 69%.  
- **group 4** — 3.6% of jets, neuron 3.50, formula right for 81%.  
- **group 5** — 3.4% of jets, neuron 8.60, formula right for 77%.  
- **group 6** — 3.0% of jets, neuron 7.51, formula right for 70%.  
- **group 7** — 2.3% of jets, neuron 8.71, formula right for 85%.  
- **group 8** — 0.8% of jets, neuron 3.64, formula right for 76%.  
- **group 9** — 0.4% of jets, neuron 7.62, formula right for 66%.  
- **group 10** — 0.3% of jets, neuron 8.18, formula right for 55%.  

### neuron 10: Spread-out hard particles, heavy mass (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (wide spread of the 5 hardest, little pT within ΔR < 0.05, large LHA and e2); it is cut back most for mass between 80.4 and 123 GeV, so heavier jets escape that penalty. Top jets sit highest (mean 2.25 for t, AUC 0.84), Z, gluon and W jets in the middle (1.23 for Z, 1.00 for g and W), quark jets lowest (0.76 for q).
- *computed — its value:* largest for t (2.25), then Z (1.23), then g (1.00), then W (1.00), then q (0.76); it separates t jets from the rest best (AUC 0.84: large for t)
- **How the class scores use it:** Only the t score uses it, raising it (+28%): widely spread hard prongs in a heavy jet are the main positive top signature; freezing it costs 1.904 points.
- *computed — used by:* raises the score of t (+28%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.636):

- `` — 20.5% of jets, value 2.41 (1.44…3.38), formula right 88%
- `` — 27.2% of jets, value 1.62 (0.81…2.38), formula right 76%
- `` — 12.4% of jets, value 1.14 (0.44…1.75), formula right 92%
- `` — 10.0% of jets, value 0.93 (0.12…1.75), formula right 75%
- `` — 2.2% of jets, value 0.65 (0.00…1.44), formula right 80%
- `` — 2.0% of jets, value 0.60 (0.00…1.75), formula right 76%
- `` — 4.8% of jets, value 0.31 (0.00…0.94), formula right 74%
- `` — 21.0% of jets, value 0.19 (0.00…0.62), formula right 77%

```
z = 4.71
if girth2_top5 < 0.025: z += -99.20 × (0.025 − girth2_top5)
if mass < 123: z += -0.034 × (123 − mass)
if mass < 80.40: z += 0.098 × (80.40 − mass)
if e2 < 0.061: z += -33.40 × (0.061 − e2)
if z_dr_0p2_0p4 < 0.088: z += 12.60 × (0.088 − z_dr_0p2_0p4)
if z_dr_0_0p05 > 0.769: z += -12.30 × (z_dr_0_0p05 − 0.769)
if girth2_top5 < 0.026 and sum_pt_top3 < 668: z += 0.128 × (0.026 − girth2_top5) × (668 − sum_pt_top3)
if n_dr_0p2_0p4 < 11.30: z += -0.107 × (11.30 − n_dr_0p2_0p4)
if mass > 144: z += -0.093 × (mass − 144)
if mass_top50 > 137: z += 0.080 × (mass_top50 − 137)
if lam1 > 0.0025: z += -53.60 × (lam1 − 0.0025)
if mass < 122 and tau21 < 0.470: z += 0.079 × (122 − mass) × (0.470 − tau21)
if max_dr < 0.436: z += -2.55 × (0.436 − max_dr)
if dr_0 < 0.061 and n_dr_0p2_0p4 > -0.940: z += 0.679 × (0.061 − dr_0) × (n_dr_0p2_0p4 − -0.940)
if mass > 162: z += -0.056 × (mass − 162)
if sum_pt_top50 < 954: z += -0.0083 × (954 − sum_pt_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 29.1% of jets, neuron 1.46, formula right for 91%.  
- **group 2** — 14.2% of jets, neuron 1.23, formula right for 72%.  
- **group 3** — 9.7% of jets, neuron 2.37, formula right for 92%.  
- **group 4** — 9.6% of jets, neuron 0.38, formula right for 80%.  
- **group 5** — 9.5% of jets, neuron 0.61, formula right for 73%.  
- **group 6** — 9.4% of jets, neuron 0.06, formula right for 75%.  
- **group 7** — 9.0% of jets, neuron 2.49, formula right for 73%.  
- **group 8** — 7.7% of jets, neuron 1.16, formula right for 69%.  
- **group 9** — 1.6% of jets, neuron 0.50, formula right for 77%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 74%.  

### neuron 0: W-side mass, below the Z (moderate)

- **What it measures:** Set by jet mass: it is cut back for jets heavier than 80.4 GeV (most strongly between 80.4 and 91.2 GeV, the Z side of the W peak) and pushed up for a small mass-to-pT ratio (squared below 0.00822); overall it falls as mass and width grow. W jets sit far highest (mean 1.71 for W, AUC 0.95), quark and gluon jets low (0.37 for q, 0.30 for g), Z and top jets lowest (0.12 for Z, 0.09 for t).
- *computed — its value:* largest for W (1.71), then q (0.37), then g (0.30), then Z (0.12), then t (0.09); it separates W jets from the rest best (AUC 0.95: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the Z score (-14%): sitting high here is evidence for a W rather than a Z. The g, q and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of Z (-14%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.842):

- `` — 21.3% of jets, value 1.79 (1.06…2.34), formula right 85%
- `` — 9.5% of jets, value 0.73 (0.06…1.25), formula right 70%
- `` — 2.6% of jets, value 0.52 (0.00…1.38), formula right 58%
- `` — 2.2% of jets, value 0.29 (0.00…0.81), formula right 70%
- `` — 19.1% of jets, value 0.26 (0.00…0.59), formula right 77%
- `` — 2.0% of jets, value 0.07 (0.00…0.25), formula right 85%
- `` — 43.2% of jets, value 0.00 (0.00…0.00), formula right 84%

```
z = 1.15
if mass > 80.40: z += -0.199 × (mass − 80.40)
if mass > 91.20: z += 0.203 × (mass − 91.20)
if mass_over_sum_pt_sq < 0.0082: z += 465 × (0.0082 − mass_over_sum_pt_sq)
if lam1 < 0.0058: z += -305 × (0.0058 − lam1)
if girth < 0.056: z += -36.50 × (0.056 − girth)
if girth2_top20 < 0.0062: z += -167 × (0.0062 − girth2_top20)
if mass_top50 < 81.80: z += -0.012 × (81.80 − mass_top50)
if n_dr_0p2_0p4 < 11.00: z += 0.030 × (11.00 − n_dr_0p2_0p4)
if mass_top50 < 72.00 and z_dr_0p05_0p1 < 0.259: z += 0.062 × (72.00 − mass_top50) × (0.259 − z_dr_0p05_0p1)
if sum_pt < 1010 and z_dr_0p2_0p4 < 0.212: z += -0.037 × (1010 − sum_pt) × (0.212 − z_dr_0p2_0p4)
if sum_pt < 1020 and z_dr_0p1_0p2 > 0.077: z += -0.018 × (1020 − sum_pt) × (z_dr_0p1_0p2 − 0.077)
if z_dr_0_0p05 > 0.883: z += -4.39 × (z_dr_0_0p05 − 0.883)
if mass > 80.40 and n_dr_0p2_0p4 < 6.91: z += -0.007 × (mass − 80.40) × (6.91 − n_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.5% of jets, neuron 1.49, formula right for 79%.  
- **group 2** — 25.0% of jets, neuron 0.39, formula right for 76%.  
- **group 3** — 21.3% of jets, neuron 0.02, formula right for 88%.  
- **group 4** — 6.8% of jets, neuron 0.00, formula right for 92%.  
- **group 5** — 5.0% of jets, neuron 0.00, formula right for 87%.  
- **group 6** — 4.3% of jets, neuron 0.00, formula right for 69%.  
- **group 7** — 3.8% of jets, neuron 0.00, formula right for 72%.  
- **group 8** — 3.8% of jets, neuron 0.00, formula right for 73%.  
- **group 9** — 1.3% of jets, neuron 0.00, formula right for 75%.  
- **group 10** — 0.3% of jets, neuron 0.01, formula right for 75%.  

### neuron 3: Sparse, thin jet (moderate)

- **What it measures:** Rises for sparse, thin jets: few particles at 0.2 <= ΔR < 0.4 with the 50 hardest carrying more than 0.979 of the pT, a thin minor axis (lam2 below 0.000569), a low mass-to-pT ratio and fewer than 46 particles; a narrow two-prong shape (τ21 below 0.419) pushes it down. W jets (0.83 for W) and quark jets (0.77 for q) sit highest, Z jets in the middle (0.58 for Z), gluon jets low (0.23 for g) and top jets near zero (0.04 for t; AUC 0.17: small for t).
- *computed — its value:* largest for W (0.83), then q (0.77), then Z (0.58), then g (0.23), then t (0.04); it separates t jets from the rest best (AUC 0.17: small for t)
- **How the class scores use it:** It raises the W and Z scores (+4% each) and lowers the g score (-9%): a sparse, thin jet looks like a boson or a quark, not a gluon. The q and t scores hardly use it.
- *computed — used by:* raises the score of W (+4%), Z (+4%); lowers the score of g (-9%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.799):

- `` — 5.8% of jets, value 1.76 (1.34…2.22), formula right 94%
- `` — 4.2% of jets, value 1.38 (1.00…1.81), formula right 95%
- `` — 10.5% of jets, value 1.22 (0.72…1.72), formula right 77%
- `` — 5.1% of jets, value 0.84 (0.41…1.34), formula right 93%
- `` — 10.8% of jets, value 0.68 (0.12…1.22), formula right 92%
- `` — 13.5% of jets, value 0.42 (0.03…0.81), formula right 74%
- `` — 2.0% of jets, value 0.14 (0.00…0.53), formula right 83%
- `` — 47.9% of jets, value 0.04 (0.00…0.16), formula right 77%

```
z = -0.217
if n_dr_0p2_0p4 < 4.84 and z_top50_slots > 0.979: z += 13.30 × (4.84 − n_dr_0p2_0p4) × (z_top50_slots − 0.979)
if lam2 < 0.00057: z += 2040 × (0.00057 − lam2)
if mass_over_sum_pt_sq < 0.0081: z += 105 × (0.0081 − mass_over_sum_pt_sq)
if tau21 < 0.419 and lam1 < 0.021: z += -186 × (0.419 − tau21) × (0.021 − lam1)
if n_particles < 46.00 and sum_pt_top40 > 822: z += 0.00013 × (46.00 − n_particles) × (sum_pt_top40 − 822)
if n_dr_0p2_0p4 < 5.20 and dr_0 < 0.049: z += -6.07 × (5.20 − n_dr_0p2_0p4) × (0.049 − dr_0)
if girth2_top5 < 0.00012: z += -6960 × (0.00012 − girth2_top5)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 34.8% of jets, neuron 0.02, formula right for 78%.  
- **group 2** — 15.8% of jets, neuron 0.36, formula right for 73%.  
- **group 3** — 10.3% of jets, neuron 0.04, formula right for 76%.  
- **group 4** — 7.5% of jets, neuron 1.65, formula right for 97%.  
- **group 5** — 6.5% of jets, neuron 0.47, formula right for 86%.  
- **group 6** — 6.3% of jets, neuron 1.04, formula right for 76%.  
- **group 7** — 5.8% of jets, neuron 0.90, formula right for 94%.  
- **group 8** — 5.7% of jets, neuron 0.88, formula right for 95%.  
- **group 9** — 4.0% of jets, neuron 1.51, formula right for 83%.  
- **group 10** — 3.4% of jets, neuron 1.08, formula right for 76%.  

### neuron 6: One-prong, away from the Z window (moderate)

- **What it measures:** Follows one-prong-ness (large τ21 and D2, a round rather than elongated pT pattern); it is pushed down for mass below 101 GeV, most for 91.2-101 GeV where the lift for mass below 91.2 GeV is missing, and for a mass-to-pT ratio above 0.0534. Quark jets sit highest (1.48 for q), gluon jets next (1.13 for g), top jets close behind (0.98 for t), Z (0.22 for Z) and W jets near zero (0.16 for W; AUC 0.16: small for W).
- *computed — its value:* largest for q (1.48), then g (1.13), then t (0.98), then Z (0.22), then W (0.16); it separates W jets from the rest best (AUC 0.16: small for W)
- **How the class scores use it:** It raises the q (+7%) and g (+3%) scores and lowers the Z score (-15%): a round, one-prong jet outside the Z mass window is not a Z. The W and t scores hardly use it.
- *computed — used by:* raises the score of g (+3%), q (+7%); lowers the score of Z (-15%); does not (or hardly) enter the score of W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.793):

- `` — 3.9% of jets, value 2.32 (2.12…2.50), formula right 85%
- `` — 6.2% of jets, value 1.98 (1.75…2.19), formula right 77%
- `` — 4.7% of jets, value 1.66 (0.75…2.62), formula right 84%
- `` — 5.8% of jets, value 1.65 (1.38…1.94), formula right 74%
- `` — 4.7% of jets, value 1.33 (1.00…1.69), formula right 73%
- `` — 21.1% of jets, value 1.06 (0.25…1.81), formula right 79%
- `` — 7.0% of jets, value 0.86 (0.44…1.31), formula right 68%
- `` — 46.7% of jets, value 0.12 (0.00…0.38), formula right 85%

```
z = 1.22
if mass < 101: z += -0.075 × (101 − mass)
if mass < 91.20: z += 0.065 × (91.20 − mass)
if mass_over_sum_pt > 0.053: z += -25.30 × (mass_over_sum_pt − 0.053)
if n_dr_0p2_0p4 < 17.70: z += -0.059 × (17.70 − n_dr_0p2_0p4)
if lam1 > 0.0072: z += 186 × (lam1 − 0.0072)
if mass_top50 < 73.10: z += 0.054 × (73.10 − mass_top50)
if mass_over_sum_pt > 0.054 and n_dr_0p2_0p4 < 19.40: z += 1.76 × (mass_over_sum_pt − 0.054) × (19.40 − n_dr_0p2_0p4)
if mass_top10 < 92.50: z += 0.0082 × (92.50 − mass_top10)
if e2 > 0.052: z += -49.40 × (e2 − 0.052)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.3% of jets, neuron 0.08, formula right for 83%.  
- **group 2** — 21.9% of jets, neuron 0.20, formula right for 89%.  
- **group 3** — 8.4% of jets, neuron 0.95, formula right for 86%.  
- **group 4** — 7.9% of jets, neuron 1.76, formula right for 73%.  
- **group 5** — 7.6% of jets, neuron 1.30, formula right for 72%.  
- **group 6** — 7.3% of jets, neuron 0.85, formula right for 68%.  
- **group 7** — 6.8% of jets, neuron 2.22, formula right for 82%.  
- **group 8** — 6.6% of jets, neuron 0.70, formula right for 66%.  
- **group 9** — 5.4% of jets, neuron 1.52, formula right for 86%.  
- **group 10** — 3.9% of jets, neuron 1.70, formula right for 78%.  

### neuron 7: Mass in the 91.2-101 GeV window (moderate)

- **What it measures:** Switches on mostly for mass between 91.2 and 101 GeV, lifted further by a compact jet (small spread of the 40 hardest particles) and an elongated two-prong pattern (small τ21 and D2). Almost only Z jets sit high on it (mean 2.04 for Z, AUC 0.92); all other types stay near zero (at most 0.11).
- *computed — its value:* largest for Z (2.04), then W (0.11), then g (0.07), then t (0.07), then q (0.04); it separates Z jets from the rest best (AUC 0.92: large for Z)
- **How the class scores use it:** It raises the Z score (+8%) and lowers the W (-6%) and t (-3%) scores: a mass just above the Z peak points to a Z. The g and q scores hardly use it.
- *computed — used by:* raises the score of Z (+8%); lowers the score of W (-6%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.775):

- `` — 8.4% of jets, value 3.33 (1.88…4.94), formula right 97%
- `` — 11.7% of jets, value 1.12 (0.19…2.25), formula right 81%
- `` — 6.1% of jets, value 0.51 (0.00…1.88), formula right 77%
- `` — 2.1% of jets, value 0.44 (0.00…1.38), formula right 84%
- `` — 7.0% of jets, value 0.12 (0.00…0.44), formula right 80%
- `` — 8.9% of jets, value 0.07 (0.00…0.31), formula right 92%
- `` — 22.0% of jets, value 0.02 (0.00…0.00), formula right 81%
- `` — 33.9% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = -0.726
if mass < 91.20: z += -0.227 × (91.20 − mass)
if mass < 101: z += 0.089 × (101 − mass)
if girth2_top40 < 0.013: z += 269 × (0.013 − girth2_top40)
if mass < 102 and max_dr < 0.393: z += 0.819 × (102 − mass) × (0.393 − max_dr)
if girth2_top20 < 0.0081: z += -323 × (0.0081 − girth2_top20)
if mass < 91.20 and max_dr < 0.391: z += -1.31 × (91.20 − mass) × (0.391 − max_dr)
if mass_top50 < 97.20 and D2 < 1.61: z += -0.439 × (97.20 − mass_top50) × (1.61 − D2)
if mass < 100 and D2 < 1.61: z += 0.307 × (100 − mass) × (1.61 − D2)
if mass_top20 < 66.00: z += 0.040 × (66.00 − mass_top20)
if D2 < 2.06 and n_dr_0p2_0p4 < 8.39: z += 0.185 × (2.06 − D2) × (8.39 − n_dr_0p2_0p4)
if D2 < 1.99 and girth2_top50 < 0.0079: z += -564 × (1.99 − D2) × (0.0079 − girth2_top50)
if z_top50_slots < 0.998: z += -27.60 × (0.998 − z_top50_slots)
if D2 < 2.00 and girth2_top50 < 0.0062: z += 715 × (2.00 − D2) × (0.0062 − girth2_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.5% of jets, neuron 0.03, formula right for 79%.  
- **group 2** — 16.8% of jets, neuron 1.14, formula right for 82%.  
- **group 3** — 13.9% of jets, neuron 0.06, formula right for 75%.  
- **group 4** — 11.8% of jets, neuron 2.16, formula right for 93%.  
- **group 5** — 10.0% of jets, neuron 0.00, formula right for 71%.  
- **group 6** — 8.5% of jets, neuron 0.00, formula right for 76%.  
- **group 7** — 8.0% of jets, neuron 0.14, formula right for 91%.  
- **group 8** — 4.3% of jets, neuron 0.00, formula right for 77%.  
- **group 9** — 1.7% of jets, neuron 0.00, formula right for 82%.  
- **group 10** — 0.5% of jets, neuron 0.00, formula right for 88%.  

### neuron 9: Light one-prong jet, off the W mass (moderate)

- **What it measures:** Large when the 50 hardest particles have mass below 128 GeV, and it follows one-prong-ness (large τ21, a round pT pattern); it is pushed down for mass between 61.8 and 80.4 GeV, the W side, and when the 15 hardest particles sit close to the axis (spread below 0.0191). Quark jets sit highest (mean 2.02 for q, AUC 0.83), gluon jets next (1.23 for g), top, W and Z jets low (0.35 for t, 0.20 for W, 0.17 for Z).
- *computed — its value:* largest for q (2.02), then g (1.23), then t (0.35), then W (0.20), then Z (0.17); it separates q jets from the rest best (AUC 0.83: large for q)
- **How the class scores use it:** It raises the q (+17%) and g (+10%) scores and lowers the W score slightly (-3%): a light, one-prong jet off the W mass is a light quark or gluon jet. The Z and t scores hardly use it.
- *computed — used by:* raises the score of g (+10%), q (+17%); lowers the score of W (-3%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.886):

- `` — 2.5% of jets, value 4.12 (2.00…6.56), formula right 68%
- `` — 13.4% of jets, value 2.73 (2.25…3.25), formula right 80%
- `` — 10.3% of jets, value 1.93 (1.38…2.44), formula right 71%
- `` — 3.3% of jets, value 0.87 (0.44…1.31), formula right 60%
- `` — 16.6% of jets, value 0.36 (0.00…0.75), formula right 85%
- `` — 53.8% of jets, value 0.05 (0.00…0.19), formula right 83%

```
z = 0.013
if mass > 61.80: z += -0.074 × (mass − 61.80)
if mass > 80.40: z += 0.111 × (mass − 80.40)
if mass_top50 < 128: z += 0.045 × (128 − mass_top50)
if girth2_top15 < 0.019: z += -104 × (0.019 − girth2_top15)
if mass > 142: z += -0.048 × (mass − 142)
if girth2_top15 < 0.015 and n_particles > 35.10: z += 2.20 × (0.015 − girth2_top15) × (n_particles − 35.10)
if width > 0.026: z += -134 × (width − 0.026)
if girth2_top40 < 0.0058 and sum_pt < 997: z += 3.67 × (0.0058 − girth2_top40) × (997 − sum_pt)
if log_sum_pt < 6.85 and dr_7 < 0.085: z += 160 × (6.85 − log_sum_pt) × (0.085 − dr_7)
if mass_top40 < 113 and max_dr > 0.386: z += 0.059 × (113 − mass_top40) × (max_dr − 0.386)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.2% of jets, neuron 0.17, formula right for 81%.  
- **group 2** — 25.0% of jets, neuron 2.47, formula right for 75%.  
- **group 3** — 21.9% of jets, neuron 0.03, formula right for 88%.  
- **group 4** — 8.8% of jets, neuron 0.27, formula right for 92%.  
- **group 5** — 5.0% of jets, neuron 0.11, formula right for 69%.  
- **group 6** — 4.9% of jets, neuron 0.54, formula right for 80%.  
- **group 7** — 4.5% of jets, neuron 0.25, formula right for 72%.  
- **group 8** — 1.5% of jets, neuron 0.03, formula right for 76%.  
- **group 9** — 1.0% of jets, neuron 4.85, formula right for 70%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 76%.  

### neuron 11: Clean two-prong, quiet outer ring (moderate)

- **What it measures:** Large when there are few particles at 0.1-0.4 in ΔR and almost no pT at 0.2 <= ΔR < 0.4, and for mass below 103 GeV with an elongated (low planar flow) pattern; extremely narrow jets and jets lighter than 80.4 GeV with small D2 are pushed down. W jets sit highest (mean 1.58 for W, AUC 0.82), Z jets next (1.31 for Z), quark, top and gluon jets low (0.28 for q and t, 0.23 for g).
- *computed — its value:* largest for W (1.58), then Z (1.31), then q (0.28), then t (0.28), then g (0.23); it separates W jets from the rest best (AUC 0.82: large for W)
- **How the class scores use it:** It raises the W score (+9%) and lowers the q score (-7%): a clean two-prong jet with a quiet outer ring is a W, not a quark jet. The g, Z and t scores hardly use it, although Z jets also sit high on it.
- *computed — used by:* raises the score of W (+9%); lowers the score of q (-7%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.863):

- `` — 9.8% of jets, value 2.69 (2.06…3.31), formula right 97%
- `` — 7.6% of jets, value 2.00 (1.38…2.56), formula right 96%
- `` — 8.7% of jets, value 1.24 (0.62…1.88), formula right 88%
- `` — 3.1% of jets, value 0.66 (0.44…0.81), formula right 83%
- `` — 5.1% of jets, value 0.53 (0.00…1.31), formula right 81%
- `` — 9.4% of jets, value 0.37 (0.19…0.81), formula right 79%
- `` — 54.3% of jets, value 0.24 (0.00…0.50), formula right 75%
- `` — 2.0% of jets, value 0.20 (0.00…0.75), formula right 74%

```
z = 0.196
if n_dr_0p2_0p4 < 10.60 and n_dr_0p1_0p2 < 26.90: z += 0.0071 × (10.60 − n_dr_0p2_0p4) × (26.90 − n_dr_0p1_0p2)
if n_dr_0p2_0p4 < 10.00 and girth2 < 0.0065: z += -38.90 × (10.00 − n_dr_0p2_0p4) × (0.0065 − girth2)
if z_dr_0p2_0p4 < 0.0063: z += 178 × (0.0063 − z_dr_0p2_0p4)
if mass < 103 and planar_flow < 0.362: z += 0.103 × (103 − mass) × (0.362 − planar_flow)
if mass < 80.40 and D2 < 3.34: z += -0.028 × (80.40 − mass) × (3.34 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 42.4% of jets, neuron 0.22, formula right for 76%.  
- **group 2** — 12.9% of jets, neuron 0.56, formula right for 77%.  
- **group 3** — 9.7% of jets, neuron 1.92, formula right for 95%.  
- **group 4** — 9.1% of jets, neuron 2.79, formula right for 97%.  
- **group 5** — 6.0% of jets, neuron 1.36, formula right for 89%.  
- **group 6** — 6.0% of jets, neuron 0.21, formula right for 74%.  
- **group 7** — 5.4% of jets, neuron 0.29, formula right for 78%.  
- **group 8** — 3.5% of jets, neuron 0.64, formula right for 84%.  
- **group 9** — 2.9% of jets, neuron 0.01, formula right for 68%.  
- **group 10** — 2.0% of jets, neuron 0.02, formula right for 74%.  

### neuron 12: Lightness: mass below 80.4 GeV (moderate)

- **What it measures:** Essentially on for jets lighter than 80.4 GeV, but pushed down for large C2 (above 0.0528) and for very light jets (mass of the 50 hardest below 63 GeV); it falls as mass and width grow. Quark jets sit highest (mean 1.24 for q, AUC 0.79), gluon jets next (0.75 for g), W, top and Z jets low (0.24 for W, 0.19 for t, 0.18 for Z).
- *computed — its value:* largest for q (1.24), then g (0.75), then W (0.24), then t (0.19), then Z (0.18); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** It raises the q (+7%), g (+3%) and Z (+3%) scores and lowers the W (-4%) and t (-4%) scores: small corrections that credit light jets to the quark and gluon scores and fine-tune the boson and top balance.
- *computed — used by:* raises the score of g (+3%), q (+7%), Z (+3%); lowers the score of W (-4%), t (-4%) (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.742):

- `` — 11.2% of jets, value 1.88 (1.62…2.12), formula right 78%
- `` — 9.1% of jets, value 1.41 (1.00…1.75), formula right 74%
- `` — 2.0% of jets, value 1.31 (0.00…4.38), formula right 82%
- `` — 3.1% of jets, value 1.09 (0.75…1.38), formula right 71%
- `` — 2.0% of jets, value 0.57 (0.12…0.88), formula right 69%
- `` — 4.3% of jets, value 0.57 (0.00…1.00), formula right 62%
- `` — 68.2% of jets, value 0.11 (0.00…0.25), formula right 84%

```
z = 0.085
if mass < 80.40: z += 0.081 × (80.40 − mass)
if C2 > 0.053: z += -19.00 × (C2 − 0.053)
if mass_top50 < 63.00: z += -0.066 × (63.00 − mass_top50)
if mass_top30 > 136: z += 0.069 × (mass_top30 − 136)
if mass_top20 > 131 and C2 > 0.065: z += -1.26 × (mass_top20 − 131) × (C2 − 0.065)
if LHA > 0.405 and lam2 < 0.0047: z += 12300 × (LHA − 0.405) × (0.0047 − lam2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 45.3% of jets, neuron 0.14, formula right for 86%.  
- **group 2** — 13.8% of jets, neuron 0.03, formula right for 71%.  
- **group 3** — 8.0% of jets, neuron 0.00, formula right for 84%.  
- **group 4** — 7.4% of jets, neuron 1.59, formula right for 74%.  
- **group 5** — 7.4% of jets, neuron 1.37, formula right for 72%.  
- **group 6** — 6.9% of jets, neuron 1.00, formula right for 67%.  
- **group 7** — 5.9% of jets, neuron 1.90, formula right for 83%.  
- **group 8** — 4.1% of jets, neuron 0.20, formula right for 94%.  
- **group 9** — 1.0% of jets, neuron 3.31, formula right for 74%.  
- **group 10** — 0.2% of jets, neuron 0.55, formula right for 67%.  

### neuron 13: High pT, not top-heavy (moderate)

- **What it measures:** Grows with the total jet pT (pushed down when the total pT is below 1020 GeV or its natural log below 6.99), and is pushed down for mass above 144 GeV and for jets with very few particles at 0.2 <= ΔR < 0.4. All non-top types sit at similar, high values (1.79 for q up to 2.30 for g); top jets sit lowest (0.98 for t; AUC 0.19: small for t).
- *computed — its value:* largest for g (2.30), then Z (2.11), then W (2.01), then q (1.79), then t (0.98); it separates t jets from the rest best (AUC 0.19: small for t)
- **How the class scores use it:** Only the t score uses it, pulling it down strongly (-37%): a high-pT jet without a top-like mass is not a top. It is the main negative top handle.
- *computed — used by:* lowers the score of t (-37%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.806):

- `` — 28.9% of jets, value 2.77 (2.31…3.03), formula right 82%
- `` — 36.0% of jets, value 2.05 (1.50…2.53), formula right 83%
- `` — 13.8% of jets, value 1.39 (0.81…2.00), formula right 74%
- `` — 3.1% of jets, value 1.23 (0.59…1.94), formula right 80%
- `` — 4.2% of jets, value 0.75 (0.19…1.38), formula right 93%
- `` — 4.2% of jets, value 0.55 (0.00…1.25), formula right 77%
- `` — 2.8% of jets, value 0.28 (0.00…0.81), formula right 94%
- `` — 7.1% of jets, value 0.14 (0.00…0.53), formula right 70%

```
z = 3.03
if log_sum_pt < 6.99: z += -13.80 × (6.99 − log_sum_pt)
if sum_pt < 1020: z += -0.023 × (1020 − sum_pt)
if sum_pt_top40 < 1020: z += 0.013 × (1020 − sum_pt_top40)
if mass > 144: z += -0.092 × (mass − 144)
if n_dr_0p2_0p4 < 4.70: z += -0.154 × (4.70 − n_dr_0p2_0p4)
if log_sum_pt < 6.82: z += 20.40 × (6.82 − log_sum_pt)
if mass_top10 > 66.70: z += 0.016 × (mass_top10 − 66.70)
if mass > 173: z += 0.103 × (mass − 173)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 37.0% of jets, neuron 2.65, formula right for 82%.  
- **group 2** — 36.7% of jets, neuron 1.86, formula right for 81%.  
- **group 3** — 10.2% of jets, neuron 0.87, formula right for 71%.  
- **group 4** — 6.8% of jets, neuron 0.80, formula right for 90%.  
- **group 5** — 3.6% of jets, neuron 0.30, formula right for 92%.  
- **group 6** — 3.0% of jets, neuron 0.15, formula right for 69%.  
- **group 7** — 1.2% of jets, neuron 1.11, formula right for 76%.  
- **group 8** — 1.0% of jets, neuron 0.06, formula right for 61%.  
- **group 9** — 0.3% of jets, neuron 2.20, formula right for 75%.  
- **group 10** — 0.2% of jets, neuron 0.05, formula right for 61%.  

### neuron 14: Mass just above the Z (moderate)

- **What it measures:** Large for mass between 91.2 and 131 GeV; pushed down for a mass-to-pT ratio below 0.089 and when a particle lies beyond ΔR 0.237. Z jets sit highest (mean 1.26 for Z, AUC 0.88), gluon and top jets well below (0.41 for g, 0.37 for t), W and quark jets lowest (0.23 for W, 0.14 for q).
- *computed — its value:* largest for Z (1.26), then g (0.41), then t (0.37), then W (0.23), then q (0.14); it separates Z jets from the rest best (AUC 0.88: large for Z)
- **How the class scores use it:** Only the W score uses it, pulling it down (-13%): a jet heavier than the Z peak is not a W. The Z score hardly uses it even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-13%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.825):

- `` — 2.1% of jets, value 1.88 (1.06…3.09), formula right 75%
- `` — 25.0% of jets, value 1.36 (0.84…1.75), formula right 85%
- `` — 4.7% of jets, value 0.72 (0.38…1.06), formula right 70%
- `` — 5.9% of jets, value 0.39 (0.19…0.56), formula right 95%
- `` — 6.9% of jets, value 0.20 (0.00…0.41), formula right 84%
- `` — 18.6% of jets, value 0.14 (0.00…0.44), formula right 83%
- `` — 4.9% of jets, value 0.08 (0.00…0.25), formula right 80%
- `` — 31.9% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = 0.349
if mass < 91.20: z += -0.140 × (91.20 − mass)
if mass < 131: z += 0.038 × (131 − mass)
if mass_over_sum_pt < 0.089: z += -26.70 × (0.089 − mass_over_sum_pt)
if max_dr > 0.237: z += -2.12 × (max_dr − 0.237)
if z_top50_slots < 0.961: z += 88.40 × (0.961 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.7% of jets, neuron 1.35, formula right for 85%.  
- **group 2** — 21.1% of jets, neuron 0.28, formula right for 86%.  
- **group 3** — 20.4% of jets, neuron 0.20, formula right for 82%.  
- **group 4** — 5.9% of jets, neuron 0.00, formula right for 77%.  
- **group 5** — 5.6% of jets, neuron 0.00, formula right for 71%.  
- **group 6** — 5.6% of jets, neuron 0.00, formula right for 73%.  
- **group 7** — 5.5% of jets, neuron 0.02, formula right for 65%.  
- **group 8** — 5.0% of jets, neuron 0.00, formula right for 69%.  
- **group 9** — 4.0% of jets, neuron 0.00, formula right for 84%.  
- **group 10** — 2.2% of jets, neuron 2.15, formula right for 76%.  

### neuron 2: Near-constant: very hard jet, large C2 (minor)

- **What it measures:** Almost constant: it is on for every jet and steps up only when the 50 hardest particles carry more than 1090 GeV and C2 is above 0.0936. It barely differs between jet types: slightly highest for gluons (0.50 for g), with the other types at 0.38-0.42 (AUC 0.53).
- *computed — its value:* largest for g (0.50), then t (0.42), then q (0.39), then W (0.38), then Z (0.38); it separates g jets from the rest best (AUC 0.53: large for g)
- **How the class scores use it:** Only the Z score uses it, and only slightly, pulling it down (-3%); it is a small correction.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.492):

- `` — 2.0% of jets, value 2.19 (0.38…4.88), formula right 81%
- `` — 3.9% of jets, value 0.47 (0.38…0.75), formula right 80%
- `` — 2.1% of jets, value 0.40 (0.38…0.50), formula right 76%
- `` — 83.0% of jets, value 0.38 (0.38…0.38), formula right 80%
- `` — 9.0% of jets, value 0.38 (0.38…0.38), formula right 84%

```
z = 0.380
if sum_pt_top50 > 1090 and C2 > 0.094: z += 0.637 × (sum_pt_top50 − 1090) × (C2 − 0.094)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 98.2% of jets, neuron 0.38, formula right for 81%.  
- **group 2** — 0.9% of jets, neuron 1.35, formula right for 82%.  
- **group 3** — 0.4% of jets, neuron 2.67, formula right for 76%.  
- **group 4** — 0.2% of jets, neuron 4.21, formula right for 81%.  
- **group 5** — 0.1% of jets, neuron 6.10, formula right for 87%.  
- **group 6** — 0.1% of jets, neuron 2.94, formula right for 67%.  
- **group 7** — 0.0% of jets, neuron 4.61, formula right for 86%.  

### neuron 15: Softer jets, quiet middle ring (weak) (minor)

- **What it measures:** On when the natural log of the total pT is below 7.04 and when little pT sits at 0.1 <= ΔR < 0.2 (share below 0.231, planar flow below 1.09); it falls as the total pT grows. It is small for all types: highest for quark and W jets (0.28 for q and W), lowest for gluon jets (0.17 for g; AUC 0.38: small for g).
- *computed — its value:* largest for q (0.28), then W (0.28), then Z (0.22), then t (0.21), then g (0.17); it separates g jets from the rest best (AUC 0.38: small for g)
- **How the class scores use it:** It raises the Z score slightly (+3%); the g, q, W and t scores hardly use it.
- *computed — used by:* raises the score of Z (+3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.682):

- `` — 2.5% of jets, value 0.62 (0.38…1.00), formula right 69%
- `` — 17.4% of jets, value 0.56 (0.38…0.75), formula right 78%
- `` — 2.9% of jets, value 0.40 (0.25…0.69), formula right 73%
- `` — 8.7% of jets, value 0.35 (0.12…0.56), formula right 81%
- `` — 29.9% of jets, value 0.17 (0.00…0.38), formula right 76%
- `` — 16.9% of jets, value 0.12 (0.06…0.19), formula right 86%
- `` — 21.6% of jets, value 0.03 (0.00…0.12), formula right 87%

```
z = -0.204
if log_sum_pt < 7.04: z += 2.02 × (7.04 − log_sum_pt)
if z_dr_0p1_0p2 < 0.231 and planar_flow < 1.09: z += 3.35 × (0.231 − z_dr_0p1_0p2) × (1.09 − planar_flow)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 25.7% of jets, neuron 0.05, formula right for 86%.  
- **group 2** — 17.7% of jets, neuron 0.23, formula right for 77%.  
- **group 3** — 11.8% of jets, neuron 0.44, formula right for 76%.  
- **group 4** — 11.8% of jets, neuron 0.00, formula right for 82%.  
- **group 5** — 9.2% of jets, neuron 0.22, formula right for 80%.  
- **group 6** — 8.4% of jets, neuron 0.67, formula right for 84%.  
- **group 7** — 7.8% of jets, neuron 0.11, formula right for 81%.  
- **group 8** — 4.2% of jets, neuron 0.42, formula right for 80%.  
- **group 9** — 2.7% of jets, neuron 0.61, formula right for 66%.  
- **group 10** — 0.7% of jets, neuron 1.14, formula right for 60%.  
