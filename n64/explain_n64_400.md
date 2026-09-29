# What each part of the 400-term formula does (64 particles)

*tuned on the network's predictions (from 60 if-statements per neuron, pruned)*. Validation accuracy 80.96%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales, and each class score adds some scales and subtracts others. Gluon and quark jets are split mainly by the particle-count scale (neuron 1), which the g score adds and the q score subtracts, while both light-jet scores add the lightness scale (9) and subtract the boson-mass two-prong scale (4). W and Z jets are recognised as compact jets in a middle pT range (neuron 5) with a quiet outer ring, since both boson scores subtract the busy, wide-jet scale (8) heavily. The two bosons are then told apart by mass: the up-to-the-W-mass scale (0) and the two-prong W-range scale (11) are added to the W score while the Z score subtracts neuron 0, and the 91-101 GeV window (7) is added to the Z score while the heavier-than-the-W scale (14) is subtracted from the W score. Top jets are picked out by hard particles spread far from the axis (neuron 10), and the t score is pulled down by the high-pT scale (13) and the compact-jet scale (5), on which top jets sit lowest.

## The 5 class scores

### score g: Many particles, round one-prong jet

High for jets with many particles, a round one-prong pT pattern (large planar flow, τ21 and D2) and pT not held by a few hard particles: gluon jets score highest (3.29; AUC 0.93), quark jets next (1.11), top jets slightly above zero (0.54) and Z (-1.74) and W (-1.76) jets far below.

Adds the particle-count scale (neuron 1, +47%) and lightness (9, +16%); subtracts the two-prong boson-mass scale (4, -14%), the compact pT-window scale (5, -11%) and the few-particle scale (3, -8%).

*computed:* largest for g (3.29), then q (1.11), then t (0.54), then Z (-1.74), then W (-1.76); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light one-prong jet, few particles

High for light, one-prong jets that are not particle-rich (small mass of the 15-20 hardest particles, large τ21 and D2): quark jets score highest (2.84; AUC 0.89), gluon jets next (1.35), top jets slightly above zero (0.50), W (-0.24) and Z (-0.48) jets below.

Subtracts the two-prong boson-mass scale (neuron 4, -29%), the particle-count scale (1, -23%) and the two-prong W-range scale (11, -9%); adds lightness (9, +28%), the below-83-GeV scale (12, +5%) and the not-two-prong scale (6, +3%).

*computed:* largest for q (2.84), then g (1.35), then t (0.50), then W (-0.24), then Z (-0.48); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact two-prong jet at the W mass

High for compact jets with a quiet outer ring, a small minor-axis width and a mass not above the W: W jets score highest (3.48; AUC 0.97), quark (0.43) and Z (0.28) jets slightly above zero, gluon (-1.12) and top (-2.56) jets below.

Subtracts the busy, wide-jet scale (neuron 8, -23%) and heavier-than-the-W (14, -11%), plus small amounts of 9, 7 and 12; adds the compact pT-window scale (5, +17%), mass up to the W peak (0, +12%), the two-prong W-range scale (11, +12%), the two-prong boson-mass scale (4, +5%) and the few-particle scale (3, +4%).

*computed:* largest for W (3.48), then q (0.43), then Z (0.28), then g (-1.12), then t (-2.56); it separates W jets from the rest best (AUC 0.97: large for W)

### score Z: Compact jet with few particles, Z mass

High for compact jets with few particles, a quiet outer ring and pT held by the hardest particles, at or just above 91 GeV: Z jets score highest (3.47; AUC 0.96), quark (0.56) and W (0.40) jets slightly above zero, gluon (-0.92) and top (-2.50) jets below.

Subtracts the busy, wide-jet scale (neuron 8, -26%), mass up to the W peak (0, -23%), the not-two-prong scale (6, -8%) and a little of 2; adds the compact pT-window scale (5, +22%), the 91-101 GeV window (7, +8%) and small amounts of 15, 3 and 12.

*computed:* largest for Z (3.47), then q (0.56), then W (0.40), then g (-0.92), then t (-2.50); it separates Z jets from the rest best (AUC 0.96: large for Z)

### score t: Wide jet with spread-out hard prongs

High for wide jets whose hard particles are spread far from the axis (large girth and LHA, little pT within ΔR < 0.05) and with a large m/pT: top jets score highest (3.21; AUC 0.95), gluon jets near zero (-0.09), W (-0.44), Z (-0.91) and quark (-1.06) jets below.

Subtracts the high-pT, below-top-mass scale (neuron 13, -33%), the compact pT-window scale (5, -16%) and small amounts of 15, 12 and 7; adds widely spread hard particles (10, +31%), the busy, wide-jet scale (8, +6%) and a little of 4 and 0.

*computed:* largest for t (3.21), then g (-0.09), then W (-0.44), then Z (-0.91), then q (-1.06); it separates t jets from the rest best (AUC 0.95: large for t)

## The 16 neurons (most important first)

### neuron 0: Mass up to the W peak (major)

- **What it measures:** Falls as the jet gets heavier: mass above 78.3 GeV pushes it down, and again above 91.0 and 92.9 GeV, a small m/pT (below about 0.0905) pushes it up, and very narrow jets are pushed down. W jets sit far highest (mean 1.61 for W, AUC 0.92), then quark (0.93) and gluon (0.62) jets, with Z (0.23) and top (0.11) jets lowest.
- *computed — its value:* largest for W (1.61), then q (0.93), then g (0.62), then Z (0.23), then t (0.11); it separates W jets from the rest best (AUC 0.92: large for W)
- **How the class scores use it:** It raises the W score (+12%) and lowers the Z score (-23%), so a jet at or below the W mass is taken as a W rather than a Z; it also raises the t score a little (+2%). The g and q scores do not use it; freezing it costs 1.06 points of accuracy.
- *computed — used by:* raises the score of W (+12%), t (+2%); lowers the score of Z (-23%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.39
if mass > 78.26: z += -0.131 × (mass − 78.26)
if mass > 92.86: z += -0.126 × (mass − 92.86)
if mass_over_sum_pt > 0.083: z += 83.10 × (mass_over_sum_pt − 0.083)
if mass > 91.03: z += -0.085 × (mass − 91.03)
if mass_over_sum_pt_sq < 0.0082: z += 482 × (0.0082 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.077: z += -43.13 × (mass_over_sum_pt − 0.077)
if girth < 0.057: z += -62.39 × (0.057 − girth)
if girth2_top20 < 0.0075: z += -241 × (0.0075 − girth2_top20)
if girth2_top20 < 0.0053: z += 230 × (0.0053 − girth2_top20)
if lam1 < 0.0059: z += -216 × (0.0059 − lam1)
if girth2_top50 < 0.0074: z += 147 × (0.0074 − girth2_top50)
if log_sum_pt < 7.02: z += 3.29 × (7.02 − log_sum_pt)
if girth2_top40 < 0.0063: z += -187 × (0.0063 − girth2_top40)
if girth2_top40 < 0.0063 and girth2_top3 < 0.0029: z += 66290 × (0.0063 − girth2_top40) × (0.0029 − girth2_top3)
if LHA < 0.260: z += 4.55 × (0.260 − LHA)
if sum_pt < 1013: z += -0.0089 × (1013 − sum_pt)
if girth2_top20 < 0.006: z += -77.67 × (0.006 − girth2_top20)
if mass_top50 < 82.04: z += -0.0097 × (82.04 − mass_top50)
if width < 0.0096: z += -32.06 × (0.0096 − width)
if mass > 92.86 and n_dr_0p2_0p4 < 7.00: z += 0.044 × (mass − 92.86) × (7.00 − n_dr_0p2_0p4)
if sum_pt < 1013 and z_dr_0p1_0p2 > 0.089: z += -0.012 × (1013 − sum_pt) × (z_dr_0p1_0p2 − 0.089)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (78 GeV), narrow, pT spread over several particles** — 28.0% of jets, neuron 1.45.  
- **light (42 GeV), very narrow, pT spread over several particles** — 25.4% of jets, neuron 1.08.  
- **heavy (91 GeV), average width, pT spread over several particles** — 22.0% of jets, neuron 0.09.  
- **very heavy (171 GeV), very wide, pT spread over several particles** — 8.6% of jets, neuron 0.00.  
- **heavy (111 GeV), average width, pT spread over several particles** — 5.0% of jets, neuron 0.00.  
- **very heavy (153 GeV), very wide, pT spread over several particles** — 4.8% of jets, neuron 0.00.  
- **heavy (132 GeV), wide, pT spread over several particles** — 4.2% of jets, neuron 0.00.  
- **very heavy (200 GeV), very wide, pT spread over several particles** — 1.4% of jets, neuron 0.00.  
- **very heavy (156 GeV), wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00.  
- **very heavy (257 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00.  

### neuron 1: Particle count, pT spread thin (major)

- **What it measures:** Grows with the number of particles and falls when the 50 hardest particles hold nearly all the pT (above 0.959 of it); a large total pT (log of the total pT above 6.89), a not-too-broad spread (LHA below 0.404) and a small pT in the 2 hardest (below 689.25 GeV) push it up. Gluon jets sit far highest (mean 5.82 for g, AUC 0.93), then top (2.98) and quark (2.63) jets, with Z (1.92) and W (1.77) jets lowest.
- *computed — its value:* largest for g (5.82), then t (2.98), then q (2.63), then Z (1.92), then W (1.77); it separates g jets from the rest best (AUC 0.93: large for g)
- **How the class scores use it:** It raises the g score (+47%) and lowers the q score (-23%), so it is the main handle for gluon versus quark; freezing it costs 11.036 points of accuracy. The W, Z and t scores hardly use it, even though top jets sit fairly high on it.
- *computed — used by:* raises the score of g (+47%); lowers the score of q (-23%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.75
if z_top50_slots > 0.959: z += -52.00 × (z_top50_slots − 0.959)
if log_sum_pt > 6.89: z += 24.27 × (log_sum_pt − 6.89)
if LHA < 0.404: z += 8.83 × (0.404 − LHA)
if log_sum_pt > 6.91: z += 24.10 × (log_sum_pt − 6.91)
if sum_pt_top2 < 689: z += 0.0036 × (689 − sum_pt_top2)
if z_top20_slots < 0.958: z += -11.66 × (0.958 − z_top20_slots)
if mass_top30 < 80.40: z += 0.055 × (80.40 − mass_top30)
if sum_pt_top50 > 959: z += -0.0086 × (sum_pt_top50 − 959)
if log_sum_pt > 6.81: z += -4.94 × (log_sum_pt − 6.81)
if max_dr < 0.436: z += -7.09 × (0.436 − max_dr)
if mass_top50 < 117: z += -0.017 × (117 − mass_top50)
if sum_pt_top5 < 902: z += 0.002 × (902 − sum_pt_top5)
if n_pt_above_10 < 31.00: z += -0.052 × (31.00 − n_pt_above_10)
if mass_top30 < 80.40 and mass_top5 < 68.43: z += -0.00063 × (80.40 − mass_top30) × (68.43 − mass_top5)
if n_particles > 38.00: z += 0.040 × (n_particles − 38.00)
if max_dr < 0.436 and z_dr_0p2_0p4 < 0.194: z += 30.03 × (0.436 − max_dr) × (0.194 − z_dr_0p2_0p4)
if log_sum_pt > 6.96: z += -14.10 × (log_sum_pt − 6.96)
if sum_pt_top30 < 1073: z += 0.0033 × (1073 − sum_pt_top30)
if z_top30_slots > 0.934 and mass_top10 < 91.19: z += -0.181 × (z_top30_slots − 0.934) × (91.19 − mass_top10)
if mass_top20 < 40.20: z += -0.067 × (40.20 − mass_top20)
if z_top30_slots > 0.934: z += 7.18 × (z_top30_slots − 0.934)
if n_particles > 38.00 and dr_0 < 0.144: z += 0.247 × (n_particles − 38.00) × (0.144 − dr_0)
if lam2 < 0.00083: z += -789 × (0.00083 − lam2)
if sum_pt_top2 < 689 and n_dr_0p2_0p4 < 7.00: z += -0.00034 × (689 − sum_pt_top2) × (7.00 − n_dr_0p2_0p4)
if n_pt_above_10 < 31.00 and mean_phi2 < 0.017: z += 1.31 × (31.00 − n_pt_above_10) × (0.017 − mean_phi2)
if mass_top20 < 40.20 and mean_phi2 < 0.0059: z += 9.63 × (40.20 − mass_top20) × (0.0059 − mean_phi2)
if z_top30_slots > 0.934 and mass_top5 > 22.18: z += 0.443 × (z_top30_slots − 0.934) × (mass_top5 − 22.18)
if log_sum_pt > 6.99: z += -8.16 × (log_sum_pt − 6.99)
if girth2_top5 < 0.00066: z += 886 × (0.00066 − girth2_top5)
if girth2_top10 < 0.0013: z += 411 × (0.0013 − girth2_top10)
if girth2_top15 < 0.0033: z += 133 × (0.0033 − girth2_top15)
if n_particles > 38.00 and z_top50_slots < 0.985: z += 1.14 × (n_particles − 38.00) × (0.985 − z_top50_slots)
if mass_top20 < 40.20 and n_real_top40 > 26.00: z += 0.0027 × (40.20 − mass_top20) × (n_real_top40 − 26.00)
if z_dr_0_0p05 > 0.879: z += -4.73 × (z_dr_0_0p05 − 0.879)
if log_sum_pt > 7.14: z += -5.03 × (log_sum_pt − 7.14)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (86 GeV), average width, pT spread over several particles** — 27.3% of jets, neuron 1.32.  
- **heavy (119 GeV), wide, pT spread over several particles** — 17.6% of jets, neuron 2.24.  
- **light (34 GeV), very narrow, pT spread over several particles** — 14.4% of jets, neuron 2.35.  
- **heavy (97 GeV), average width, pT spread over several particles** — 10.8% of jets, neuron 3.31.  
- **heavy (140 GeV), very wide, pT spread over several particles** — 10.1% of jets, neuron 4.28.  
- **medium-mass (59 GeV), narrow, pT spread over several particles** — 6.0% of jets, neuron 4.76.  
- **heavy (117 GeV), average width, pT spread over several particles, high pT** — 5.1% of jets, neuron 5.77.  
- **light (47 GeV), very narrow, pT spread over several particles** — 4.2% of jets, neuron 5.81.  
- **heavy (97 GeV), narrow, pT spread over several particles, high pT** — 3.5% of jets, neuron 6.99.  
- **heavy (111 GeV), narrow, pT spread over several particles, high pT** — 1.1% of jets, neuron 7.56.  

### neuron 4: Two-prong jet at boson mass (major)

- **What it measures:** Follows two-prong-ness (small D2 and τ21) and the mass of the 10-15 hardest particles; mass below 101 and 120.6 GeV pushes it up, while mass below 86.4 or 80.8 GeV, a light 40 hardest (below 94.6 GeV), a broad soft spread (LHA above 0.115) and very narrow jets push it down. Z (1.27) and W (1.24) jets sit highest, top jets next (0.57), gluon (0.10) and quark (0.08; AUC 0.19, small for q) jets near zero.
- *computed — its value:* largest for Z (1.27), then W (1.24), then t (0.57), then g (0.10), then q (0.08); it separates q jets from the rest best (AUC 0.19: small for q)
- **How the class scores use it:** It lowers the q score (-29%) and the g score (-14%), because a pronged jet near the boson mass is not a light jet, and raises the W (+5%) and t (+3%) scores a little. The Z score hardly uses it, although Z jets sit highest on it.
- *computed — used by:* raises the score of W (+5%), t (+3%); lowers the score of g (-14%), q (-29%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.60
if LHA > 0.115: z += -10.63 × (LHA − 0.115)
if mass < 101: z += 0.067 × (101 − mass)
if girth < 0.121: z += -24.63 × (0.121 − girth)
if mass < 121: z += 0.028 × (121 − mass)
if mass_top40 < 94.64: z += -0.043 × (94.64 − mass_top40)
if mass < 86.40: z += -0.058 × (86.40 − mass)
if mass_top30 < 153: z += -0.0097 × (153 − mass_top30)
if mass_top40 < 83.33: z += 0.049 × (83.33 − mass_top40)
if girth2_top15 < 0.016: z += 63.88 × (0.016 − girth2_top15)
if width > 0.0096: z += 167 × (width − 0.0096)
if D2 < 6.92: z += 0.102 × (6.92 − D2)
if girth2_top15 < 0.0073: z += -146 × (0.0073 − girth2_top15)
if mass_top40 < 83.33 and D2 < 6.92: z += -0.011 × (83.33 − mass_top40) × (6.92 − D2)
if mass < 80.78: z += -0.033 × (80.78 − mass)
if n_dr_0p2_0p4 < 15.00: z += 0.043 × (15.00 − n_dr_0p2_0p4)
if z_dr_0p2_0p4 < 0.068: z += -7.71 × (0.068 − z_dr_0p2_0p4)
if lam1 > 0.0077: z += -102 × (lam1 − 0.0077)
if mass < 121 and max_dr < 0.402: z += 0.100 × (121 − mass) × (0.402 − max_dr)
if girth < 0.062: z += -17.87 × (0.062 − girth)
if mass < 74.25 and max_dr < 0.394: z += -0.450 × (74.25 − mass) × (0.394 − max_dr)
if e2 > 0.037: z += 37.61 × (e2 − 0.037)
if n_particles > 29.00 and n_dr_0_0p05 < 25.00: z += -0.00075 × (n_particles − 29.00) × (25.00 − n_dr_0_0p05)
if sum_pt < 1066: z += -0.0026 × (1066 − sum_pt)
if mass_top20 > 104: z += -0.031 × (mass_top20 − 104)
if mass_over_sum_pt > 0.098: z += -9.22 × (mass_over_sum_pt − 0.098)
if mass > 144: z += 0.021 × (mass − 144)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (96 GeV), average width, pT spread over several particles** — 22.1% of jets, neuron 1.07.  
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 17.2% of jets, neuron 1.33.  
- **very heavy (173 GeV), very wide, pT spread over several particles** — 11.3% of jets, neuron 0.75.  
- **medium-mass (85 GeV), narrow, pT spread over several particles** — 10.7% of jets, neuron 0.74.  
- **very heavy (141 GeV), very wide, pT spread over several particles** — 8.8% of jets, neuron 0.23.  
- **medium-mass (51 GeV), very narrow, pT spread over several particles** — 8.1% of jets, neuron 0.01.  
- **medium-mass (68 GeV), narrow, pT spread over several particles** — 7.4% of jets, neuron 0.08.  
- **light (29 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.00.  
- **light (37 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 0.00.  
- **light (22 GeV), very narrow, pT spread over several particles** — 1.8% of jets, neuron 0.00.  

### neuron 5: Compact jet in a pT window (major)

- **What it measures:** Runs opposite to the particle count: it rises when the 50 hardest particles hold more than 0.959 of the pT, when the 40 hardest carry more than 906.6 GeV and when the 30 hardest have mass above 68.3 GeV, and is cut back when the log of the total pT exceeds 6.91, when the 40 hardest hold more than 0.930 of the pT and when m/pT is above 0.0905. Z jets sit highest (mean 2.10 for Z, AUC 0.82), then W (1.73) and quark (1.65) jets, with gluon (0.65) and top (0.64) jets lowest.
- *computed — its value:* largest for Z (2.10), then W (1.73), then q (1.65), then g (0.65), then t (0.64); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** It raises the W (+17%) and Z (+22%) scores and lowers the g (-11%) and t (-16%) scores: a compact, few-particle jet with some mass is boson-like, not gluon- or top-like. The q score hardly uses it.
- *computed — used by:* raises the score of W (+17%), Z (+22%); lowers the score of g (-11%), t (-16%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.673
if log_sum_pt > 6.91: z += -34.62 × (log_sum_pt − 6.91)
if z_top40_slots > 0.930: z += -25.70 × (z_top40_slots − 0.930)
if log_sum_pt > 6.86: z += 11.62 × (log_sum_pt − 6.86)
if z_top50_slots > 0.959: z += 31.62 × (z_top50_slots − 0.959)
if sum_pt_top40 > 907: z += 0.0061 × (sum_pt_top40 − 907)
if mass_top30 > 68.29: z += 0.036 × (mass_top30 − 68.29)
if mass_over_sum_pt > 0.090: z += -38.30 × (mass_over_sum_pt − 0.090)
if mass > 64.49: z += 0.016 × (mass − 64.49)
if sum_pt > 908 and girth2_top2 < 0.014: z += 0.253 × (sum_pt − 908) × (0.014 − girth2_top2)
if n_particles < 64.00 and e2 < 0.039: z += 1.40 × (64.00 − n_particles) × (0.039 − e2)
if mass_top50 > 91.19: z += -0.023 × (mass_top50 − 91.19)
if log_sum_pt > 6.92 and girth2_top2 < 0.014: z += -468 × (log_sum_pt − 6.92) × (0.014 − girth2_top2)
if girth < 0.086: z += -8.27 × (0.086 − girth)
if log_sum_pt > 6.99: z += 9.97 × (log_sum_pt − 6.99)
if mass_over_sum_pt > 0.080: z += -10.67 × (mass_over_sum_pt − 0.080)
if mass_over_sum_pt > 0.171: z += 340 × (mass_over_sum_pt − 0.171)
if mass_over_sum_pt_sq > 0.029: z += -668 × (mass_over_sum_pt_sq − 0.029)
if sum_pt_top15 > 951: z += -0.005 × (sum_pt_top15 − 951)
if log_sum_pt > 6.92: z += 2.62 × (log_sum_pt − 6.92)
if sum_pt_top50 > 1061: z += -0.0041 × (sum_pt_top50 − 1061)
if sum_pt_top20 > 1005: z += 0.0046 × (sum_pt_top20 − 1005)
if log_sum_pt > 6.96: z += 2.50 × (log_sum_pt − 6.96)
if log_sum_pt > 6.99 and z_dr_0p05_0p1 < 0.299: z += 16.10 × (log_sum_pt − 6.99) × (0.299 − z_dr_0p05_0p1)
if n_particles < 64.00 and z_dr_0p2_0p4 > 0.0047: z += 0.133 × (64.00 − n_particles) × (z_dr_0p2_0p4 − 0.0047)
if mass_top50 > 158: z += -0.031 × (mass_top50 − 158)
if mass > 173: z += -0.024 × (mass − 173)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (68 GeV), narrow, pT spread over several particles** — 43.3% of jets, neuron 2.02.  
- **medium-mass (75 GeV), narrow, pT spread over several particles** — 18.4% of jets, neuron 1.38.  
- **very heavy (159 GeV), very wide, pT spread over several particles** — 12.2% of jets, neuron 0.43.  
- **heavy (95 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 1.18.  
- **medium-mass (73 GeV), narrow, pT spread over several particles** — 7.1% of jets, neuron 0.57.  
- **heavy (96 GeV), narrow, pT spread over several particles, high pT** — 3.9% of jets, neuron 0.24.  
- **very heavy (166 GeV), very wide, pT spread over several particles** — 2.5% of jets, neuron 0.28.  
- **very heavy (182 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 0.23.  
- **heavy (109 GeV), narrow, pT spread over several particles, high pT** — 1.2% of jets, neuron 0.13.  
- **very heavy (209 GeV), very wide, pT spread over several particles** — 0.5% of jets, neuron 0.19.  

### neuron 7: Z mass window above 91 GeV (major)

- **What it measures:** Switches on mainly for mass between 91.19 and 101 GeV (up to 120.6 GeV more weakly); mass below 91.19 and 82.85 GeV pushes it down hard, as do narrow jets, while a small e2 and a two-prong pattern (small τ21 and D2) help. Almost only Z jets sit high on it (mean 1.57 for Z, AUC 0.92); W, top, gluon and quark jets all stay near zero (0.06 or less).
- *computed — its value:* largest for Z (1.57), then W (0.06), then t (0.05), then g (0.04), then q (0.03); it separates Z jets from the rest best (AUC 0.92: large for Z)
- **How the class scores use it:** It raises the Z score (+8%) and lowers the W (-5%) and t (-2%) scores: a mass just above the Z peak points to a Z. The g and q scores hardly use it; freezing it costs 1.616 points.
- *computed — used by:* raises the score of Z (+8%); lowers the score of W (-5%), t (-2%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.509
if mass < 91.19: z += -0.138 × (91.19 − mass)
if mass < 82.85: z += -0.170 × (82.85 − mass)
if e2_sq < 0.0096: z += 460 × (0.0096 − e2_sq)
if mass_top50 < 71.80: z += -0.173 × (71.80 − mass_top50)
if girth < 0.086: z += -50.04 × (0.086 − girth)
if girth2_top40 < 0.008: z += -540 × (0.008 − girth2_top40)
if mass < 121: z += 0.033 × (121 − mass)
if girth2_top20 < 0.008: z += -356 × (0.008 − girth2_top20)
if mass < 101: z += 0.042 × (101 − mass)
if mass < 101 and max_dr < 0.391: z += 0.808 × (101 − mass) × (0.391 − max_dr)
if mass_top50 < 97.93 and D2 < 1.60: z += -0.398 × (97.93 − mass_top50) × (1.60 − D2)
if mass < 101 and D2 < 1.60: z += 0.279 × (101 − mass) × (1.60 − D2)
if mass < 91.19 and max_dr < 0.391: z += -1.00 × (91.19 − mass) × (0.391 − max_dr)
if LHA < 0.320: z += 9.80 × (0.320 − LHA)
if girth2_top40 < 0.0077: z += 261 × (0.0077 − girth2_top40)
if girth2_top40 < 0.013: z += 89.85 × (0.013 − girth2_top40)
if mass_top50 < 97.93: z += -0.024 × (97.93 − mass_top50)
if girth2_top20 < 0.0064: z += 247 × (0.0064 − girth2_top20)
if z_dr_0p2_0p4 < 0.091: z += -5.38 × (0.091 − z_dr_0p2_0p4)
if mass_top20 < 66.84: z += 0.021 × (66.84 − mass_top20)
if n_dr_0p2_0p4 < 13.00 and z_1st < 0.499: z += 0.167 × (13.00 − n_dr_0p2_0p4) × (0.499 − z_1st)
if girth2_top40 < 0.0088: z += 82.99 × (0.0088 − girth2_top40)
if D2 < 1.79 and n_dr_0p2_0p4 < 9.00: z += 0.159 × (1.79 − D2) × (9.00 − n_dr_0p2_0p4)
if mass_top30 < 76.42: z += 0.018 × (76.42 − mass_top30)
if mass_over_sum_pt < 0.090: z += -12.03 × (0.090 − mass_over_sum_pt)
if mass_top50 < 71.80 and max_dr < 0.394: z += 0.513 × (71.80 − mass_top50) × (0.394 − max_dr)
if C2 < 0.056: z += -17.79 × (0.056 − C2)
if max_dr < 0.383: z += -3.93 × (0.383 − max_dr)
if mass < 91.19 and pt_dispersion < 0.338: z += -0.312 × (91.19 − mass) × (0.338 − pt_dispersion)
if mass < 86.40: z += -0.0089 × (86.40 − mass)
if z_top50_slots < 0.991: z += -22.96 × (0.991 − z_top50_slots)
if mass < 91.19 and z_dr_0p05_0p1 > 0.403: z += -0.206 × (91.19 − mass) × (z_dr_0p05_0p1 − 0.403)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (134 GeV), wide, pT spread over several particles** — 34.2% of jets, neuron 0.30.  
- **medium-mass (82 GeV), narrow, pT spread over several particles** — 17.3% of jets, neuron 0.15.  
- **medium-mass (87 GeV), average width, pT spread over several particles** — 13.1% of jets, neuron 1.66.  
- **light (47 GeV), very narrow, pT spread over several particles** — 8.1% of jets, neuron 0.00.  
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 7.8% of jets, neuron 0.06.  
- **medium-mass (63 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.00.  
- **light (30 GeV), very narrow, pT spread over several particles** — 6.4% of jets, neuron 0.00.  
- **light (32 GeV), very narrow, pT spread over several particles** — 3.6% of jets, neuron 0.00.  
- **light (26 GeV), very narrow, pT spread over several particles** — 1.6% of jets, neuron 0.00.  
- **very light (18 GeV), very narrow, pT spread over several particles** — 0.5% of jets, neuron 0.00.  

### neuron 8: Busy outer ring, wide heavy jet (major)

- **What it measures:** Grows with the number and pT share of particles at 0.2 <= ΔR < 0.4 and with m/pT and the jet width; mass above 64.5, 80.8 and 87.4 GeV pushes it up, mass above 101 GeV and very narrow jets partly push it back down. Top jets sit far highest (mean 3.55 for t, AUC 0.91), gluon jets next (1.40), quark jets lower (0.64), Z (0.31) and W (0.07) jets near zero.
- *computed — its value:* largest for t (3.55), then g (1.40), then q (0.64), then Z (0.31), then W (0.07); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W (-23%) and Z (-26%) scores, since a two-prong boson is compact with a quiet outer ring, and raises the t score (+6%). The g and q scores hardly use it.
- *computed — used by:* raises the score of t (+6%); lowers the score of W (-23%), Z (-26%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.439
if mass > 64.49: z += 0.053 × (mass − 64.49)
if mass > 101: z += -0.107 × (mass − 101)
if lam1 < 0.020: z += -89.39 × (0.020 − lam1)
if mass > 80.78: z += 0.053 × (mass − 80.78)
if girth < 0.097: z += 27.84 × (0.097 − girth)
if mass_over_sum_pt > 0.098: z += -70.98 × (mass_over_sum_pt − 0.098)
if n_dr_0p2_0p4 < 21.00: z += -0.059 × (21.00 − n_dr_0p2_0p4)
if girth2_top20 > 0.008: z += 227 × (girth2_top20 − 0.008)
if mass > 87.36: z += 0.034 × (mass − 87.36)
if z_dr_0p2_0p4 < 0.091: z += 9.71 × (0.091 − z_dr_0p2_0p4)
if mass_top50 > 82.04: z += -0.026 × (mass_top50 − 82.04)
if girth2_top20 > 0.0057: z += -122 × (girth2_top20 − 0.0057)
if mass_over_sum_pt_sq < 0.014: z += -65.19 × (0.014 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.077 and sum_pt < 1116: z += 0.186 × (mass_over_sum_pt − 0.077) × (1116 − sum_pt)
if mass_over_sum_pt > 0.077: z += 17.84 × (mass_over_sum_pt − 0.077)
if mass_over_sum_pt_sq < 0.0062: z += -253 × (0.0062 − mass_over_sum_pt_sq)
if lam1 < 0.0082: z += 111 × (0.0082 − lam1)
if lam2 < 0.0014: z += -412 × (0.0014 − lam2)
if girth2_top40 > 0.0052: z += 54.73 × (girth2_top40 − 0.0052)
if z_dr_0p1_0p2 < 0.155: z += 3.84 × (0.155 − z_dr_0p1_0p2)
if sum_pt < 1013: z += 0.013 × (1013 − sum_pt)
if mass_top50 > 97.93: z += 0.019 × (mass_top50 − 97.93)
if e2 > 0.025: z += -20.27 × (e2 − 0.025)
if sum_pt_top40 < 1002: z += 0.0064 × (1002 − sum_pt_top40)
if girth2_top30 < 0.0075: z += 69.24 × (0.0075 − girth2_top30)
if sum_pt_top30 < 1012: z += -0.003 × (1012 − sum_pt_top30)
if mass_over_sum_pt > 0.089: z += 9.00 × (mass_over_sum_pt − 0.089)
if mass_top40 > 79.55: z += -0.0077 × (mass_top40 − 79.55)
if mass > 87.36 and log_sum_pt < 6.90: z += -0.402 × (mass − 87.36) × (6.90 − log_sum_pt)
if mass > 144: z += -0.022 × (mass − 144)
if sum_pt_top40 < 1002 and log_sum_pt < 6.81: z += -0.020 × (1002 − sum_pt_top40) × (6.81 − log_sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 28.1% of jets, neuron 0.25.  
- **light (43 GeV), very narrow, pT spread over several particles** — 26.1% of jets, neuron 0.11.  
- **heavy (93 GeV), average width, pT spread over several particles** — 21.6% of jets, neuron 0.80.  
- **very heavy (171 GeV), very wide, pT spread over several particles** — 8.0% of jets, neuron 4.27.  
- **heavy (122 GeV), wide, pT spread over several particles** — 5.4% of jets, neuron 3.00.  
- **very heavy (143 GeV), very wide, pT spread over several particles** — 4.0% of jets, neuron 4.12.  
- **very heavy (167 GeV), very wide, pT spread over several particles, high pT** — 2.6% of jets, neuron 2.53.  
- **medium-mass (64 GeV), average width, pT spread over several particles, low pT** — 1.7% of jets, neuron 3.95.  
- **very heavy (161 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 5.83.  
- **very heavy (221 GeV), very wide, pT spread over several particles** — 1.0% of jets, neuron 3.67.  

### neuron 9: Lightness of the jet (major)

- **What it measures:** Large when the 50 hardest particles have mass below 136.8 GeV and falls as the jet mass and width grow; mass above 62.55 GeV and very narrow jets push it down, partly given back above 82.85 and 92.9 GeV. Quark jets sit highest (2.88; AUC 0.82), gluon jets next (1.77), W (0.57) and top (0.55) jets low and Z jets lowest (0.30).
- *computed — its value:* largest for q (2.88), then g (1.77), then W (0.57), then t (0.55), then Z (0.30); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** It raises the q (+28%) and g (+16%) scores and lowers the W score (-6%): a light jet is a light-quark or gluon jet. The Z and t scores hardly use it; freezing it costs 1.136 points.
- *computed — used by:* raises the score of g (+16%), q (+28%); lowers the score of W (-6%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.908
if mass_top50 < 137: z += 0.065 × (137 − mass_top50)
if mass_top50 < 161: z += -0.025 × (161 − mass_top50)
if mass > 62.55: z += -0.051 × (mass − 62.55)
if girth2_top15 < 0.021: z += -107 × (0.021 − girth2_top15)
if mass > 82.85: z += 0.049 × (mass − 82.85)
if e2 < 0.048: z += -35.28 × (0.048 − e2)
if LHA < 0.209: z += 26.12 × (0.209 − LHA)
if girth < 0.044: z += -79.19 × (0.044 − girth)
if mass > 92.86: z += 0.034 × (mass − 92.86)
if mass_top40 < 91.29: z += -0.024 × (91.29 − mass_top40)
if mass > 144: z += -0.102 × (mass − 144)
if mass < 137: z += 0.0074 × (137 − mass)
if girth2_top15 < 0.01: z += 62.09 × (0.01 − girth2_top15)
if mass_top40 < 121: z += 0.0059 × (121 − mass_top40)
if sum_pt < 950: z += 0.032 × (950 − sum_pt)
if girth2 < 0.0036: z += 413 × (0.0036 − girth2)
if mass_top40 < 80.89: z += 0.015 × (80.89 − mass_top40)
if girth2_top40 < 0.0063: z += 130 × (0.0063 − girth2_top40)
if girth2_top15 < 0.021 and n_particles > 41.00: z += 1.81 × (0.021 − girth2_top15) × (n_particles − 41.00)
if lam1 < 0.0052: z += 153 × (0.0052 − lam1)
if girth < 0.028: z += -66.20 × (0.028 − girth)
if log_sum_pt < 6.86: z += -19.79 × (6.86 − log_sum_pt)
if girth > 0.097: z += 11.99 × (girth − 0.097)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 26.9% of jets, neuron 0.53.  
- **heavy (92 GeV), average width, pT spread over several particles** — 22.1% of jets, neuron 0.08.  
- **light (32 GeV), very narrow, pT spread over several particles** — 13.2% of jets, neuron 3.93.  
- **medium-mass (54 GeV), very narrow, pT spread over several particles** — 12.1% of jets, neuron 2.94.  
- **very heavy (172 GeV), very wide, pT spread over several particles** — 10.2% of jets, neuron 0.24.  
- **very heavy (146 GeV), very wide, pT spread over several particles** — 6.2% of jets, neuron 0.72.  
- **heavy (118 GeV), wide, pT spread over several particles** — 5.9% of jets, neuron 0.28.  
- **medium-mass (73 GeV), average width, pT spread over several particles, low pT** — 1.8% of jets, neuron 3.15.  
- **very heavy (221 GeV), very wide, pT spread over several particles, high pT** — 1.1% of jets, neuron 0.00.  
- **medium-mass (55 GeV), average width, pT spread over several particles, low pT** — 0.6% of jets, neuron 4.86.  

### neuron 10: Hard particles spread wide (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (large e2, LHA, girth and spread of the 5 hardest, little pT within ΔR < 0.05); e2 below 0.0652, a tight spread of the 5 hardest and mass below 120.6 GeV push it down, while the 50 hardest above 136.8 GeV push it up. Top jets sit highest (2.58; AUC 0.88), then Z (1.35), W (1.05) and gluon (0.82) jets, with quark jets lowest (0.53).
- *computed — its value:* largest for t (2.58), then Z (1.35), then W (1.05), then g (0.82), then q (0.53); it separates t jets from the rest best (AUC 0.88: large for t)
- **How the class scores use it:** Only the t score uses it, raising it (+31%): widely spread hard prongs are the main positive sign of a top; freezing it costs 2.44 points.
- *computed — used by:* raises the score of t (+31%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 3.78
if e2 < 0.065: z += -45.20 × (0.065 − e2)
if girth2_top5 < 0.024: z += -64.43 × (0.024 − girth2_top5)
if girth2_top10 > 0.00024: z += 126 × (girth2_top10 − 0.00024)
if mass < 121: z += -0.017 × (121 − mass)
if girth2_top5 < 0.024 and sum_pt_top3 < 656: z += 0.165 × (0.024 − girth2_top5) × (656 − sum_pt_top3)
if mass < 80.40: z += 0.053 × (80.40 − mass)
if z_dr_0p2_0p4 < 0.091: z += 7.86 × (0.091 − z_dr_0p2_0p4)
if lam1 > 0.0019: z += -59.45 × (lam1 − 0.0019)
if mass_top50 > 137: z += 0.081 × (mass_top50 − 137)
if mass > 144: z += -0.085 × (mass − 144)
if n_dr_0p2_0p4 < 13.00: z += -0.050 × (13.00 − n_dr_0p2_0p4)
if max_dr < 0.402: z += -4.31 × (0.402 − max_dr)
if girth2_top10 > 0.0077: z += -99.15 × (girth2_top10 − 0.0077)
if z_dr_0_0p05 > 0.767: z += -4.34 × (z_dr_0_0p05 − 0.767)
if mass < 86.40: z += 0.014 × (86.40 − mass)
if e2 < 0.065 and z_dr_0p1_0p2 < 0.219: z += -37.99 × (0.065 − e2) × (0.219 − z_dr_0p1_0p2)
if mass < 121 and tau21 < 0.470: z += 0.046 × (121 − mass) × (0.470 − tau21)
if mass > 163: z += -0.096 × (mass − 163)
if girth2_top15 < 0.0022: z += -296 × (0.0022 − girth2_top15)
if n_dr_0p2_0p4 < 6.00: z += -0.078 × (6.00 − n_dr_0p2_0p4)
if dr_0 < 0.064 and n_dr_0p2_0p4 > 2.00: z += 0.705 × (0.064 − dr_0) × (n_dr_0p2_0p4 − 2.00)
if mass < 62.55: z += -0.017 × (62.55 − mass)
if sum_pt_top50 < 959: z += -0.007 × (959 − sum_pt_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (87 GeV), average width, pT spread over several particles** — 30.5% of jets, neuron 1.61.  
- **medium-mass (89 GeV), average width, pT spread over several particles** — 23.0% of jets, neuron 0.79.  
- **light (30 GeV), very narrow, pT spread over several particles** — 13.2% of jets, neuron 0.06.  
- **medium-mass (54 GeV), narrow, pT spread over several particles** — 13.0% of jets, neuron 0.53.  
- **heavy (130 GeV), wide, pT spread over several particles** — 7.0% of jets, neuron 2.76.  
- **very heavy (174 GeV), very wide, pT spread over several particles** — 5.2% of jets, neuron 2.65.  
- **very heavy (156 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 3.18.  
- **very heavy (169 GeV), very wide, pT spread over several particles** — 3.3% of jets, neuron 2.37.  
- **very heavy (204 GeV), very wide, pT spread over several particles** — 1.1% of jets, neuron 0.19.  
- **very heavy (259 GeV), very wide, pT spread over several particles, high pT** — 0.2% of jets, neuron 0.00.  

### neuron 3: Few particles, small m/pT (moderate)

- **What it measures:** Rises for a small m/pT (squared value below 0.00751), fewer than 46 particles and few particles in the rings 0.1 <= ΔR < 0.2 (under 15) and 0.2 <= ΔR < 0.4 (under 7); very narrow jets are pushed down. Quark jets sit highest (0.88), then W (0.65) and Z (0.36) jets, gluon jets low (0.13) and top jets almost at zero (0.03; AUC 0.23, small for t).
- *computed — its value:* largest for q (0.88), then W (0.65), then Z (0.36), then g (0.13), then t (0.03); it separates t jets from the rest best (AUC 0.23: small for t)
- **How the class scores use it:** It raises the W and Z scores (+4% each) and lowers the g score (-8%): a sparse jet is not a gluon. The q and t scores hardly use it, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+4%), Z (+4%); lowers the score of g (-8%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.306
if mass_over_sum_pt_sq < 0.0075: z += 229 × (0.0075 − mass_over_sum_pt_sq)
if n_particles < 46.00: z += 0.057 × (46.00 − n_particles)
if mass_over_sum_pt_sq < 0.0075 and mass_top20 < 137: z += -1.19 × (0.0075 − mass_over_sum_pt_sq) × (137 − mass_top20)
if n_dr_0p1_0p2 < 15.00: z += 0.042 × (15.00 − n_dr_0p1_0p2)
if lam1 < 0.0077: z += -84.11 × (0.0077 − lam1)
if tau21 < 0.428 and lam1 < 0.016: z += -248 × (0.428 − tau21) × (0.016 − lam1)
if n_dr_0p2_0p4 < 7.00: z += 0.084 × (7.00 − n_dr_0p2_0p4)
if n_dr_0p2_0p4 < 5.00 and z_top50_slots > 0.979: z += 7.24 × (5.00 − n_dr_0p2_0p4) × (z_top50_slots − 0.979)
if n_particles < 46.00 and mass_top30 > 73.33: z += -0.0017 × (46.00 − n_particles) × (mass_top30 − 73.33)
if n_dr_0p2_0p4 < 5.00 and sum_pt_top30 < 988: z += -0.0032 × (5.00 − n_dr_0p2_0p4) × (988 − sum_pt_top30)
if n_dr_0p2_0p4 < 5.00 and dr_0 < 0.041: z += -4.64 × (5.00 − n_dr_0p2_0p4) × (0.041 − dr_0)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (133 GeV), wide, pT spread over several particles** — 33.9% of jets, neuron 0.01.  
- **light (50 GeV), very narrow, pT spread over several particles** — 11.1% of jets, neuron 0.26.  
- **medium-mass (75 GeV), narrow, pT spread over several particles** — 10.4% of jets, neuron 0.11.  
- **heavy (90 GeV), average width, pT spread over several particles** — 8.6% of jets, neuron 0.25.  
- **medium-mass (84 GeV), narrow, pT spread over several particles** — 8.3% of jets, neuron 0.77.  
- **light (38 GeV), very narrow, pT spread over several particles** — 7.6% of jets, neuron 1.29.  
- **medium-mass (86 GeV), average width, pT spread over several particles** — 7.5% of jets, neuron 0.29.  
- **medium-mass (86 GeV), average width, pT spread over several particles** — 6.0% of jets, neuron 1.06.  
- **light (25 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 1.84.  
- **medium-mass (77 GeV), average width, pT spread over several particles** — 1.4% of jets, neuron 0.00.  

### neuron 11: Two-prong split near W mass (moderate)

- **What it measures:** Follows two-prong-ness (small D2, large e2, the hardest particle away from the axis, little pT within ΔR < 0.05); mass below 92.9 GeV pushes it up, but a jet lighter than 80.4 GeV with an almost empty outer ring, a 50 hardest below 86.4 GeV and very narrow jets are pushed down. W (1.55) and Z (1.48) jets sit highest, top jets next (0.85), gluon (0.32) and quark (0.14; AUC 0.17, small for q) jets low.
- *computed — its value:* largest for W (1.55), then Z (1.48), then t (0.85), then g (0.32), then q (0.14); it separates q jets from the rest best (AUC 0.17: small for q)
- **How the class scores use it:** It raises the W score (+12%) and lowers the q score (-9%): a split jet near the W mass is a W, not a quark jet. The g, Z and t scores hardly use it.
- *computed — used by:* raises the score of W (+12%); lowers the score of q (-9%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.878
if mass < 80.40 and z_dr_0p2_0p4 < 0.037: z += -4.48 × (80.40 − mass) × (0.037 − z_dr_0p2_0p4)
if mass_top50 < 86.40: z += -0.061 × (86.40 − mass_top50)
if mass < 92.86: z += 0.037 × (92.86 − mass)
if girth2_top30 < 0.0064: z += -293 × (0.0064 − girth2_top30)
if n_dr_0p2_0p4 < 10.00 and z_dr_0p2_0p4 < 0.068: z += 1.94 × (10.00 − n_dr_0p2_0p4) × (0.068 − z_dr_0p2_0p4)
if mass_over_sum_pt_sq < 0.0082: z += 182 × (0.0082 − mass_over_sum_pt_sq)
if z_top5_slots > 0.535: z += -4.59 × (z_top5_slots − 0.535)
if e2 < 0.039: z += -32.40 × (0.039 − e2)
if mass < 101: z += 0.015 × (101 − mass)
if lam1 < 0.0067: z += -176 × (0.0067 − lam1)
if e2 < 0.028: z += 56.12 × (0.028 − e2)
if girth2_top5 < 0.0072: z += -78.63 × (0.0072 − girth2_top5)
if mass < 62.55: z += -0.039 × (62.55 − mass)
if girth2_top5 < 0.0072 and eta_0 < 0.055: z += 911 × (0.0072 − girth2_top5) × (0.055 − eta_0)
if sum_pt_top3 > 439: z += 0.0021 × (sum_pt_top3 − 439)
if z_dr_0p2_0p4 < 0.0064: z += 84.45 × (0.0064 − z_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (137 GeV), very wide, pT spread over several particles** — 30.7% of jets, neuron 0.85.  
- **medium-mass (86 GeV), average width, pT spread over several particles** — 21.9% of jets, neuron 2.32.  
- **medium-mass (89 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 0.66.  
- **medium-mass (83 GeV), narrow, pT spread over several particles** — 8.9% of jets, neuron 0.46.  
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 5.5% of jets, neuron 0.03.  
- **medium-mass (56 GeV), narrow, pT spread over several particles** — 5.4% of jets, neuron 0.00.  
- **light (47 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 0.00.  
- **light (37 GeV), very narrow, pT spread over several particles** — 5.0% of jets, neuron 0.00.  
- **light (29 GeV), very narrow, pT spread over several particles** — 4.8% of jets, neuron 0.00.  
- **light (20 GeV), very narrow, pT spread over several particles** — 3.0% of jets, neuron 0.00.  

### neuron 13: High total pT, below top mass (moderate)

- **What it measures:** Grows with the total jet pT (sums below about 1012.7 and 1052.9 GeV push it down); mass above 74.3 GeV pushes it up and mass above 143.8 GeV pushes it down. The non-top types sit at similar, high values (Z 1.93, W 1.74, gluon 1.64, quark 1.48); top jets sit lowest (0.64; AUC 0.16, small for t).
- *computed — its value:* largest for Z (1.93), then W (1.74), then g (1.64), then q (1.48), then t (0.64); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** Only the t score uses it, lowering it strongly (-33%): a jet high on this scale is short of the top mass and so not a top. It is the main negative top handle.
- *computed — used by:* lowers the score of t (-33%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.07
if mass > 74.25: z += 0.024 × (mass − 74.25)
if mass_top50 > 137: z += 0.119 × (mass_top50 − 137)
if sum_pt < 1013: z += -0.025 × (1013 − sum_pt)
if mass > 144: z += -0.120 × (mass − 144)
if sum_pt_top40 < 1053: z += 0.0082 × (1053 − sum_pt_top40)
if sum_pt < 1053: z += -0.011 × (1053 − sum_pt)
if sum_pt_top50 < 1014: z += 0.017 × (1014 − sum_pt_top50)
if mass_top50 > 97.93: z += -0.030 × (mass_top50 − 97.93)
if sum_pt_top40 < 1053 and D2 < 5.38: z += -0.0019 × (1053 − sum_pt_top40) × (5.38 − D2)
if log_sum_pt < 7.02: z += -2.61 × (7.02 − log_sum_pt)
if mass_over_sum_pt_sq > 0.029: z += -804 × (mass_over_sum_pt_sq − 0.029)
if mass_over_sum_pt > 0.171: z += 277 × (mass_over_sum_pt − 0.171)
if mass > 74.25 and D2 > 0.603: z += -0.004 × (mass − 74.25) × (D2 − 0.603)
if mass > 137: z += -0.030 × (mass − 137)
if mass > 161: z += -0.081 × (mass − 161)
if sum_pt_top40 < 956: z += -0.0079 × (956 − sum_pt_top40)
if mass > 173: z += 0.136 × (mass − 173)
if sum_pt < 986: z += -0.0067 × (986 − sum_pt)
if sum_pt_top50 < 959 and D2 < 4.45: z += 0.0039 × (959 − sum_pt_top50) × (4.45 − D2)
if mass > 74.25 and sum_pt < 1017: z += 7.7e-05 × (mass − 74.25) × (1017 − sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (75 GeV), narrow, pT spread over several particles** — 69.4% of jets, neuron 1.90.  
- **medium-mass (83 GeV), average width, pT spread over several particles** — 11.6% of jets, neuron 0.92.  
- **very heavy (173 GeV), very wide, pT spread over several particles** — 6.4% of jets, neuron 0.21.  
- **very heavy (159 GeV), very wide, pT spread over several particles** — 5.1% of jets, neuron 0.79.  
- **medium-mass (90 GeV), wide, pT spread over several particles, low pT** — 4.1% of jets, neuron 0.05.  
- **very heavy (179 GeV), very wide, pT spread over several particles** — 1.3% of jets, neuron 0.02.  
- **medium-mass (64 GeV), average width, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00.  
- **very heavy (218 GeV), very wide, pT spread over several particles, high pT** — 0.6% of jets, neuron 0.44.  
- **very heavy (200 GeV), very wide, pT spread over several particles** — 0.5% of jets, neuron 0.01.  
- **very heavy (271 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.43.  

### neuron 2: Sparse jet, quiet outer ring (weak) (minor)

- **What it measures:** Falls with the number of particles at 0.2 <= ΔR < 0.4, with the minor-axis width lam2 and with the particle count; a high total pT combined with a heavy jet (50 hardest above 85.9 GeV) or a busy outer ring pushes it down, as does mass below 92.9 GeV. It is small for all types: W jets highest (0.56; AUC 0.73), then Z (0.36), quark (0.31), gluon (0.19) and top (0.10) jets.
- *computed — its value:* largest for W (0.56), then Z (0.36), then q (0.31), then g (0.19), then t (0.10); it separates W jets from the rest best (AUC 0.73: large for W)
- **How the class scores use it:** Only the Z score uses it, lowering it slightly (-3%); freezing it changes almost nothing.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.664
if log_sum_pt > 6.90 and mass_top50 > 85.87: z += -1.61 × (log_sum_pt − 6.90) × (mass_top50 − 85.87)
if log_sum_pt > 6.90 and n_dr_0p2_0p4 > 4.00: z += -3.13 × (log_sum_pt − 6.90) × (n_dr_0p2_0p4 − 4.00)
if mass_over_sum_pt < 0.098: z += 37.26 × (0.098 − mass_over_sum_pt)
if mass < 92.86: z += -0.047 × (92.86 − mass)
if sum_pt > 1066: z += -0.027 × (sum_pt − 1066)
if log_sum_pt > 6.90 and n_real_top30 > 22.00: z += -1.76 × (log_sum_pt − 6.90) × (n_real_top30 − 22.00)
if sum_pt_top50 > 1107 and z_top20_slots > 0.807: z += -0.400 × (sum_pt_top50 − 1107) × (z_top20_slots − 0.807)
if z_top30_slots > 0.905: z += 10.88 × (z_top30_slots − 0.905)
if sum_pt > 1017: z += 0.013 × (sum_pt − 1017)
if sum_pt_top50 > 1039: z += -0.017 × (sum_pt_top50 − 1039)
if sum_pt_top50 > 1157: z += 0.037 × (sum_pt_top50 − 1157)
if sum_pt > 996: z += 0.008 × (sum_pt − 996)
if sum_pt_top20 > 1129: z += 0.066 × (sum_pt_top20 − 1129)
if n_pt_above_10 > 18.00: z += 0.090 × (n_pt_above_10 − 18.00)
if sum_pt_top40 > 1013 and eta_5 < -0.114: z += -3.75 × (sum_pt_top40 − 1013) × (-0.114 − eta_5)
if sum_pt_top50 > 1039 and dr_3 > 0.0046: z += 0.137 × (sum_pt_top50 − 1039) × (dr_3 − 0.0046)
if mass_top40 > 161 and pt_9 < 41.44: z += -0.024 × (mass_top40 − 161) × (41.44 − pt_9)
if log_sum_pt > 6.90 and girth2_top10 < 0.020: z += 192 × (log_sum_pt − 6.90) × (0.020 − girth2_top10)
if sum_pt_top15 > 1003 and girth2_top3 < 0.004: z += -4.20 × (sum_pt_top15 − 1003) × (0.004 − girth2_top3)
if log_sum_pt > 6.90 and n_dr_0p05_0p1 > 9.00: z += 0.488 × (log_sum_pt − 6.90) × (n_dr_0p05_0p1 − 9.00)
if mass < 92.86 and sum_pt_top40 < 1070: z += 0.00011 × (92.86 − mass) × (1070 − sum_pt_top40)
if sum_pt_top50 > 1157 and n_pt_above_50 < 5.00: z += -0.051 × (sum_pt_top50 − 1157) × (5.00 − n_pt_above_50)
if mass_top20 > 125: z += -0.044 × (mass_top20 − 125)
if sum_pt_top50 > 1107: z += -0.0027 × (sum_pt_top50 − 1107)
if sum_pt_top50 > 997 and mean_phi > 0.00011: z += 7.40 × (sum_pt_top50 − 997) × (mean_phi − 0.00011)
if sum_pt_top20 > 1129 and pt_7 > 45.75: z += 0.00028 × (sum_pt_top20 − 1129) × (pt_7 − 45.75)
if sum_pt_top20 > 1129 and girth2_top3 < 0.0079: z += -0.357 × (sum_pt_top20 − 1129) × (0.0079 − girth2_top3)
if sum_pt_top20 > 1129 and eta_1 > 0.041: z += 0.725 × (sum_pt_top20 − 1129) × (eta_1 − 0.041)
if sum_pt_top40 > 1013: z += -0.00014 × (sum_pt_top40 − 1013)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (85 GeV), average width, pT spread over several particles** — 87.4% of jets, neuron 0.33.  
- **very heavy (156 GeV), very wide, pT spread over several particles** — 4.9% of jets, neuron 0.00.  
- **medium-mass (69 GeV), narrow, pT spread over several particles, high pT** — 4.4% of jets, neuron 0.19.  
- **very heavy (169 GeV), very wide, pT spread over several particles, high pT** — 1.7% of jets, neuron 0.01.  
- **medium-mass (67 GeV), very narrow, pT spread over several particles, high pT** — 0.9% of jets, neuron 0.52.  
- **very heavy (197 GeV), very wide, pT spread over several particles, high pT** — 0.5% of jets, neuron 0.00.  
- **very heavy (249 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  
- **very heavy (162 GeV), wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  

### neuron 6: Not two-prong, outside boson masses (minor)

- **What it measures:** Weakly follows one-prong-ness (large τ21 and D2) and a broad spread (LHA above 0.228); m/pT above 0.0508 and mass below 120.6 and 101 GeV push it down, only partly given back below 92.9 GeV, so jets in the boson mass range sit near zero. Top (0.58), gluon (0.53) and quark (0.50) jets sit at similar values, while Z (0.05) and W (0.04; AUC 0.29, small for W) jets are almost always at zero.
- *computed — its value:* largest for t (0.58), then g (0.53), then q (0.50), then Z (0.05), then W (0.04); it separates W jets from the rest best (AUC 0.29: small for W)
- **How the class scores use it:** It raises the q score (+3%) and lowers the Z score (-8%): a jet that is not a two-prong boson-mass jet is not a Z. The g, W and t scores hardly use it.
- *computed — used by:* raises the score of q (+3%); lowers the score of Z (-8%); does not (or hardly) enter the score of g, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.018
if mass_over_sum_pt > 0.051: z += -49.35 × (mass_over_sum_pt − 0.051)
if mass < 121: z += -0.042 × (121 − mass)
if mass < 101: z += -0.067 × (101 − mass)
if mass < 92.86: z += 0.070 × (92.86 − mass)
if LHA > 0.228: z += 18.60 × (LHA − 0.228)
if girth2_top20 > 0.0018: z += 118 × (girth2_top20 − 0.0018)
if e2 < 0.048: z += 32.07 × (0.048 − e2)
if mass < 173: z += 0.0066 × (173 − mass)
if mass < 86.40: z += 0.037 × (86.40 − mass)
if lam1 < 0.0073: z += 190 × (0.0073 − lam1)
if mass_over_sum_pt > 0.051 and n_dr_0p2_0p4 < 21.00: z += 1.07 × (mass_over_sum_pt − 0.051) × (21.00 − n_dr_0p2_0p4)
if e2 < 0.044: z += 21.74 × (0.044 − e2)
if girth2_top50 < 0.014: z += -48.84 × (0.014 − girth2_top50)
if LHA > 0.228 and D2 < 3.35: z += -3.07 × (LHA − 0.228) × (3.35 − D2)
if girth2_top3 > 0.010 and D2 < 4.45: z += 74.95 × (girth2_top3 − 0.010) × (4.45 − D2)
if mass_over_sum_pt > 0.051 and lam2 < 0.0024: z += 8490 × (mass_over_sum_pt − 0.051) × (0.0024 − lam2)
if girth2_top3 > 0.010: z += -181 × (girth2_top3 − 0.010)
if n_dr_0p2_0p4 < 21.00: z += -0.024 × (21.00 − n_dr_0p2_0p4)
if width < 0.0096: z += -72.92 × (0.0096 − width)
if girth2_top20 > 0.017: z += -163 × (girth2_top20 − 0.017)
if lam1 > 0.012: z += -107 × (lam1 − 0.012)
if mass_over_sum_pt_sq < 0.020: z += 15.76 × (0.020 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.051 and z_dr_0_0p05 < 0.908: z += 5.74 × (mass_over_sum_pt − 0.051) × (0.908 − z_dr_0_0p05)
if mass_over_sum_pt > 0.051 and z_dr_0p1_0p2 < 0.286: z += 37.74 × (mass_over_sum_pt − 0.051) × (0.286 − z_dr_0p1_0p2)
if e2 > 0.056: z += 99.19 × (e2 − 0.056)
if mass < 92.86 and sum_pt_top15 < 1083: z += -3.4e-05 × (92.86 − mass) × (1083 − sum_pt_top15)
if mass < 121 and n_dr_0p05_0p1 > 12.00: z += -0.00097 × (121 − mass) × (n_dr_0p05_0p1 − 12.00)
if lam1 > 0.0082: z += 34.70 × (lam1 − 0.0082)
if lam1 > 0.0082 and pt_4 < 68.12: z += 1.42 × (lam1 − 0.0082) × (68.12 − pt_4)
if mass_over_sum_pt > 0.051 and max_pair_mass > 13.05: z += -0.175 × (mass_over_sum_pt − 0.051) × (max_pair_mass − 13.05)
if LHA > 0.404 and z_2 > 0.050: z += 790 × (LHA − 0.404) × (z_2 − 0.050)
if LHA > 0.404: z += -13.39 × (LHA − 0.404)
if mass_top50 < 71.80: z += 0.0025 × (71.80 − mass_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (95 GeV), average width, pT spread over several particles** — 19.4% of jets, neuron 0.17.  
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 14.9% of jets, neuron 0.00.  
- **medium-mass (85 GeV), narrow, pT spread over several particles** — 14.2% of jets, neuron 0.05.  
- **light (28 GeV), very narrow, pT spread over several particles** — 10.8% of jets, neuron 0.69.  
- **light (47 GeV), very narrow, pT spread over several particles** — 10.6% of jets, neuron 0.36.  
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 8.6% of jets, neuron 0.11.  
- **heavy (134 GeV), wide, pT spread over several particles** — 8.0% of jets, neuron 1.22.  
- **very heavy (166 GeV), very wide, pT spread over several particles** — 6.2% of jets, neuron 0.32.  
- **very heavy (169 GeV), very wide, pT spread over several particles** — 5.2% of jets, neuron 0.74.  
- **very heavy (184 GeV), very wide, pT spread over several particles** — 2.1% of jets, neuron 0.94.  

### neuron 12: Light jet below 83 GeV (minor)

- **What it measures:** Mainly on for jets lighter than 82.85 GeV (more so below 74.3 GeV, and when the 50 hardest are below 80.4 GeV), but pushed down for very light jets (below 62.55 GeV), very narrow jets and a low total pT. Quark jets sit highest (0.93; AUC 0.82), gluon jets next (0.37), W (0.14), top (0.13) and Z (0.08) jets near zero.
- *computed — its value:* largest for q (0.93), then g (0.37), then W (0.14), then t (0.13), then Z (0.08); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** It raises the q (+5%) and Z (+2%) scores and lowers the W (-3%) and t (-3%) scores: small corrections that credit light jets to the quark side. The g score hardly uses it.
- *computed — used by:* raises the score of q (+5%), Z (+2%); lowers the score of W (-3%), t (-3%); does not (or hardly) enter the score of g (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.228
if mass < 82.85: z += 0.080 × (82.85 − mass)
if mass_top50 < 80.40: z += 0.050 × (80.40 − mass_top50)
if mass < 74.25: z += 0.054 × (74.25 − mass)
if girth < 0.050: z += -50.35 × (0.050 − girth)
if mass_top50 < 77.38: z += -0.042 × (77.38 − mass_top50)
if mass < 86.40: z += -0.028 × (86.40 − mass)
if mass < 62.55: z += -0.063 × (62.55 − mass)
if girth2_top50 < 0.0074: z += 120 × (0.0074 − girth2_top50)
if mass_top40 < 77.94: z += -0.016 × (77.94 − mass_top40)
if girth2_top50 < 0.0074 and pt_dispersion < 0.429: z += -915 × (0.0074 − girth2_top50) × (0.429 − pt_dispersion)
if log_sum_pt < 6.86: z += -20.38 × (6.86 − log_sum_pt)
if LHA < 0.228: z += 5.07 × (0.228 − LHA)
if mass < 86.40 and z_dr_0p1_0p2 < 0.065: z += -0.195 × (86.40 − mass) × (0.065 − z_dr_0p1_0p2)
if mass_top40 < 89.68 and sum_pt_top2 < 502: z += 5.4e-05 × (89.68 − mass_top40) × (502 − sum_pt_top2)
if LHA > 0.404: z += 17.51 × (LHA − 0.404)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (110 GeV), wide, pT spread over several particles** — 65.1% of jets, neuron 0.06.  
- **light (35 GeV), very narrow, pT spread over several particles** — 6.6% of jets, neuron 1.19.  
- **light (47 GeV), very narrow, pT spread over several particles** — 6.5% of jets, neuron 1.00.  
- **medium-mass (73 GeV), narrow, pT spread over several particles** — 6.4% of jets, neuron 0.28.  
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.85.  
- **light (23 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 1.39.  
- **heavy (115 GeV), very wide, pT spread over several particles, low pT** — 2.4% of jets, neuron 0.00.  
- **light (37 GeV), very narrow, pT spread over several particles, low pT** — 0.7% of jets, neuron 0.00.  
- **heavy (91 GeV), wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.00.  
- **light (45 GeV), average width, pT spread over several particles, low pT** — 0.2% of jets, neuron 0.00.  

### neuron 14: Heavier than the W (minor)

- **What it measures:** Rises with mass: it is large for m/pT between about 0.0905 and 0.141 and for mass between 91.03 and 136.8 GeV, while m/pT below 0.0905, mass below 91.03 and 89.7 GeV and narrow jets push it down. Z jets sit highest (0.91; AUC 0.82), top (0.38) and gluon (0.37) jets well below, quark (0.13) and W (0.04) jets lowest.
- *computed — its value:* largest for Z (0.91), then t (0.38), then g (0.37), then q (0.13), then W (0.04); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** Only the W score uses it, lowering it (-11%): a jet heavier than the W is not a W. The Z score hardly uses it even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-11%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.737
if mass_over_sum_pt < 0.090: z += -161 × (0.090 − mass_over_sum_pt)
if mass_over_sum_pt < 0.098: z += 82.88 × (0.098 − mass_over_sum_pt)
if mass < 91.03: z += -0.101 × (91.03 − mass)
if mass_over_sum_pt < 0.118: z += 40.32 × (0.118 − mass_over_sum_pt)
if girth2_top30 < 0.012: z += -239 × (0.012 − girth2_top30)
if mass_over_sum_pt < 0.141: z += 19.78 × (0.141 − mass_over_sum_pt)
if girth2_top50 < 0.0093: z += -339 × (0.0093 − girth2_top50)
if mass < 137: z += 0.020 × (137 − mass)
if sum_pt_top50 > 976: z += -0.013 × (sum_pt_top50 − 976)
if mass < 89.74: z += -0.059 × (89.74 − mass)
if lam1 < 0.0073: z += 366 × (0.0073 − lam1)
if mass_top40 < 67.73: z += 0.105 × (67.73 − mass_top40)
if mass_top50 < 89.08: z += 0.048 × (89.08 − mass_top50)
if lam2 < 0.0024: z += 502 × (0.0024 − lam2)
if girth2_top20 < 0.017 and z_top50_slots > 0.970: z += -2758 × (0.017 − girth2_top20) × (z_top50_slots − 0.970)
if log_sum_pt > 6.94: z += 17.32 × (log_sum_pt − 6.94)
if mass < 80.40: z += -0.059 × (80.40 − mass)
if girth2 < 0.0079: z += 273 × (0.0079 − girth2)
if mass_top40 < 111: z += -0.016 × (111 − mass_top40)
if lam1 < 0.012: z += -94.27 × (0.012 − lam1)
if lam1 < 0.0062: z += -299 × (0.0062 − lam1)
if max_dr > 0.240: z += -3.86 × (max_dr − 0.240)
if lam1 < 0.0062 and z_top50_slots > 0.985: z += 21879 × (0.0062 − lam1) × (z_top50_slots − 0.985)
if n_dr_0p2_0p4 < 21.00 and n_dr_0p1_0p2 < 21.00: z += -0.0029 × (21.00 − n_dr_0p2_0p4) × (21.00 − n_dr_0p1_0p2)
if mass_top50 < 117: z += 0.010 × (117 − mass_top50)
if mass < 91.03 and z_dr_0p1_0p2 < 0.219: z += 0.112 × (91.03 − mass) × (0.219 − z_dr_0p1_0p2)
if mass_top30 < 91.19: z += 0.011 × (91.19 − mass_top30)
if mass_top40 < 80.89: z += -0.018 × (80.89 − mass_top40)
if mass_top40 < 91.29 and z_dr_0p2_0p4 < 0.068: z += -0.207 × (91.29 − mass_top40) × (0.068 − z_dr_0p2_0p4)
if e2 < 0.033: z += 22.03 × (0.033 − e2)
if girth < 0.086: z += -6.24 × (0.086 − girth)
if mass_top40 < 91.29: z += 0.007 × (91.29 − mass_top40)
if z_dr_0p2_0p4 < 0.020 and z_0 < 0.359: z += 125 × (0.020 − z_dr_0p2_0p4) × (0.359 − z_0)
if n_dr_0p2_0p4 < 21.00 and n_real_top40 < 40.00: z += -0.0016 × (21.00 − n_dr_0p2_0p4) × (40.00 − n_real_top40)
if lam2 < 0.0024 and sum_pt_top3 < 462: z += -1.46 × (0.0024 − lam2) × (462 − sum_pt_top3)
if max_dr < 0.332: z += -3.13 × (0.332 − max_dr)
if max_dr > 0.402: z += 2.97 × (max_dr − 0.402)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 22.1% of jets, neuron 0.07.  
- **heavy (91 GeV), average width, pT spread over several particles** — 21.6% of jets, neuron 1.02.  
- **very heavy (150 GeV), very wide, pT spread over several particles** — 21.4% of jets, neuron 0.41.  
- **light (26 GeV), very narrow, pT spread over several particles** — 8.5% of jets, neuron 0.00.  
- **light (42 GeV), very narrow, pT spread over several particles** — 8.5% of jets, neuron 0.00.  
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.00.  
- **medium-mass (83 GeV), narrow, pT spread over several particles, high pT** — 4.2% of jets, neuron 0.30.  
- **very heavy (159 GeV), wide, pT spread over several particles, high pT** — 3.1% of jets, neuron 0.91.  
- **light (50 GeV), very narrow, pT spread over several particles, high pT** — 2.4% of jets, neuron 0.00.  
- **heavy (94 GeV), narrow, pT spread over several particles, high pT** — 0.9% of jets, neuron 0.19.  

### neuron 15: Narrow jet, quiet rings (weak) (minor)

- **What it measures:** Rises for a narrow jet (girth below 0.121) with a small pT share at 0.2 <= ΔR < 0.4, and falls with the particle count and pT share at 0.1 <= ΔR < 0.2 and with lam2. It is small for all types: W jets highest (0.65), then quark (0.43), Z (0.38) and gluon (0.33) jets, top jets lowest (0.10; AUC 0.21, small for t).
- *computed — its value:* largest for W (0.65), then q (0.43), then Z (0.38), then g (0.33), then t (0.10); it separates t jets from the rest best (AUC 0.21: small for t)
- **How the class scores use it:** It raises the Z score slightly (+5%) and lowers the t score slightly (-4%); the g, q and W scores hardly use it.
- *computed — used by:* raises the score of Z (+5%); lowers the score of t (-4%); does not (or hardly) enter the score of g, q, W (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.733
if girth2_top50 < 0.014 and z_dr_0p2_0p4 < 0.129: z += 2072 × (0.014 − girth2_top50) × (0.129 − z_dr_0p2_0p4)
if girth2_top50 < 0.014: z += -202 × (0.014 − girth2_top50)
if e2_sq < 0.0069 and z_dr_0p2_0p4 < 0.129: z += 6540 × (0.0069 − e2_sq) × (0.129 − z_dr_0p2_0p4)
if girth < 0.121: z += 23.11 × (0.121 − girth)
if z_dr_0p2_0p4 < 0.020 and z_dr_0p1_0p2 > 0.286: z += -15373 × (0.020 − z_dr_0p2_0p4) × (z_dr_0p1_0p2 − 0.286)
if e2_sq < 0.0069: z += -561 × (0.0069 − e2_sq)
if girth2_top5 < 0.0015: z += -1855 × (0.0015 − girth2_top5)
if girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194: z += 8896 × (0.0015 − girth2_top5) × (0.194 − z_dr_0p2_0p4)
if girth < 0.121 and z_dr_0p2_0p4 < 0.129: z += -70.74 × (0.121 − girth) × (0.129 − z_dr_0p2_0p4)
if lam2 < 0.0018: z += 392 × (0.0018 − lam2)
if mass_top50 < 71.80: z += -0.041 × (71.80 − mass_top50)
if girth < 0.050: z += -39.06 × (0.050 − girth)
if lam1 < 0.016: z += -33.35 × (0.016 − lam1)
if z_dr_0p1_0p2 < 0.120 and mass_top5 < 40.20: z += 0.239 × (0.120 − z_dr_0p1_0p2) × (40.20 − mass_top5)
if z_dr_0p1_0p2 < 0.187: z += 2.90 × (0.187 − z_dr_0p1_0p2)
if log_sum_pt < 7.02: z += 2.58 × (7.02 − log_sum_pt)
if sum_pt < 1002: z += -0.014 × (1002 − sum_pt)
if z_dr_0p2_0p4 < 0.020: z += -24.47 × (0.020 − z_dr_0p2_0p4)
if sum_pt < 986: z += 0.012 × (986 − sum_pt)
if z_dr_0p1_0p2 < 0.120 and z_dr_0p2_0p4 < 0.037: z += -149 × (0.120 − z_dr_0p1_0p2) × (0.037 − z_dr_0p2_0p4)
if girth2_top5 < 0.0072 and sum_pt_top40 < 985: z += -0.802 × (0.0072 − girth2_top5) × (985 − sum_pt_top40)
if girth2_top20 < 0.0029 and sum_pt > 1116: z += -1.43 × (0.0029 − girth2_top20) × (sum_pt − 1116)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (86 GeV), average width, pT spread over several particles** — 40.0% of jets, neuron 0.52.  
- **very heavy (150 GeV), very wide, pT spread over several particles** — 23.5% of jets, neuron 0.12.  
- **light (36 GeV), very narrow, pT spread over several particles** — 18.9% of jets, neuron 0.36.  
- **medium-mass (66 GeV), narrow, pT spread over several particles** — 12.4% of jets, neuron 0.61.  
- **heavy (92 GeV), average width, pT spread over several particles** — 2.4% of jets, neuron 0.00.  
- **heavy (96 GeV), average width, pT spread over several particles** — 1.6% of jets, neuron 0.00.  
- **heavy (106 GeV), average width, pT spread over several particles** — 0.6% of jets, neuron 0.00.  
- **heavy (128 GeV), wide, pT spread over several particles** — 0.3% of jets, neuron 0.00.  
- **heavy (131 GeV), wide, pT spread over several particles** — 0.2% of jets, neuron 0.00.  
- **very heavy (142 GeV), wide, pT spread over several particles** — 0.1% of jets, neuron 0.00.  
