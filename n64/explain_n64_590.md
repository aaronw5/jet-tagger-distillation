# What each part of the 590-term formula does (64 particles)

*tuned on the network's predictions (from 100 if-statements per neuron, pruned)*. Validation accuracy 81.08%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales, and each class score adds some of these scales and subtracts others. Gluon and quark jets are split mainly by the particle-count scale (neuron 1: many particles sharing the pT thinly), which the g score adds and the q score subtracts, while both light-jet scores subtract the elongated two-prong scale (neuron 4) and add the light, off-W-mass scale (neuron 9). W and Z jets are recognised as compact jets with few particles (neuron 5) and above all by the absence of a wide, busy spread for their pT (neuron 8), which both boson scores subtract heavily. The two bosons are then told apart by mass: the light-side W scale (0) and the clean 80-93 GeV two-prong scale (11) are added to the W score while the 93-101 GeV window (7) and the just-above-the-Z scale (14) are subtracted from it, and the Z score adds the window (7) and subtracts the W-side scale (0) and the one-prong scale (6). Top jets are picked out by hard particles spread far from the axis (neuron 10), and the t score is pulled down strongly by the high-pT, lighter-than-top scale (13) and the few-particle scale (5), on both of which top jets sit lowest.

## The 5 class scores

### score g: Many particles, round one-prong, not heavy

High for jets with many particles, a round one-prong pT pattern (large τ21, D2 and planar flow) and light hardest particles: gluon jets score highest (3.32; AUC 0.93), quark jets next (1.21), top jets near zero (0.39), W (-1.66) and Z (-1.93) jets far below.

Adds the particle-count scale (neuron 1, +47%) and the light, off-W-mass scale (9, +14%); subtracts the elongated two-prong scale (4, -15%), the few-particle compact scale (5, -11%) and the narrow one-prong scale (3, -9%).

*computed:* largest for g (3.32), then q (1.21), then t (0.39), then W (-1.66), then Z (-1.93); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light one-prong jet with few particles

High for light, one-prong jets that are not particle-rich (large D2 and τ21, small e2, light 15 hardest particles): quark jets score highest (2.91; AUC 0.89), gluon jets next (1.40), top jets near zero (0.37), W (-0.21) and Z (-0.63) jets below.

Subtracts the elongated two-prong scale (neuron 4, -31%), the particle-count scale (1, -24%) and the 80-93 GeV two-prong scale (11, -6%); adds the light, off-W-mass scale (9, +25%) and, in small amounts, the high-pT scale (2, +5%), the one-prong scale away from the Z mass (6, +5%) and the light no-dense-core scale (12, +3%).

*computed:* largest for q (2.91), then g (1.40), then t (0.37), then W (-0.21), then Z (-0.63); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact, clean two-prong jet at the W mass

High for compact, narrow jets with few particles in the outer ring and mass up to about the W: W jets score highest (3.53; AUC 0.96), quark (0.46) and Z (0.16) jets slightly above zero, gluon (-1.13) and top (-2.97) jets far below.

Subtracts the wide, busy scale (neuron 8, -25%), mass just above the Z peak (14, -10%), the 93-101 GeV window (7, -6%) and the light off-W scale (9, -6%); adds the few-particle compact scale (5, +20%), the light-side W scale (0, +9%), the 80-93 GeV two-prong scale (11, +8%), the elongated two-prong scale (4, +6%) and the narrow one-prong scale (3, +5%).

*computed:* largest for W (3.53), then q (0.46), then Z (0.16), then g (-1.13), then t (-2.97); it separates W jets from the rest best (AUC 0.96: large for W)

### score Z: Compact two-prong jet at the Z mass

High for compact jets with few particles, a thin minor axis and mass at or just above 91 GeV: Z jets score highest (3.40; AUC 0.95), quark (0.50) and W (0.35) jets slightly above zero, gluon (-1.11) and top (-2.88) jets far below.

Subtracts the wide, busy scale (neuron 8, -25%), the light-side W scale (0, -15%), the one-prong scale away from the Z mass (6, -11%) and the high-pT scale (2, -8%); adds the few-particle compact scale (5, +22%), the 93-101 GeV window (7, +8%), and small amounts of the narrow one-prong scale (3, +5%) and the dense-core scale (15, +4%).

*computed:* largest for Z (3.40), then q (0.50), then W (0.35), then g (-1.11), then t (-2.88); it separates Z jets from the rest best (AUC 0.95: large for Z)

### score t: Heavy, wide jet with spread-out hard prongs

High for wide jets whose hard particles are spread far from the axis (large girth, LHA and e2): top jets score highest (3.10; AUC 0.95), gluon (-0.18) and W (-0.43) jets slightly below zero, quark (-1.08) and Z (-1.11) jets further below.

Subtracts the high-pT, lighter-than-top scale (neuron 13, -34%), the few-particle compact scale (5, -17%) and small amounts of the dense-core scale (15, -3%) and the 93-101 GeV window (7, -3%); adds widely spread hard particles (10, +30%), the wide, busy scale (8, +6%) and a little of the elongated two-prong scale (4, +3%).

*computed:* largest for t (3.10), then g (-0.18), then W (-0.43), then q (-1.08), then Z (-1.11); it separates t jets from the rest best (AUC 0.95: large for t)

## The 16 neurons (most important first)

### neuron 1: Many particles, thinly shared pT (major)

- **What it measures:** Grows with the number of particles and falls when a few hardest particles hold most of the pT (for example when the 50 hardest carry more than 0.959 of it, or when fewer than 31 particles have pT above 10 GeV); a broad spread (LHA below 0.404 still counts) and a light set of the 30 hardest (below 80.4 GeV) also push it up. Gluon jets sit far highest (6.40; AUC 0.92), then top (3.11) and quark (2.80) jets, with Z (2.34) and W (2.26) jets lowest.
- *computed — its value:* largest for g (6.40), then t (3.11), then q (2.80), then Z (2.34), then W (2.26); it separates g jets from the rest best (AUC 0.92: large for g)
- **How the class scores use it:** It raises the g score (+47%) and lowers the q score (-24%), so it is the main handle for gluon versus quark; freezing it costs 11.692 points of accuracy. The W, Z and t scores hardly use it, even though top jets sit fairly high on it.
- *computed — used by:* raises the score of g (+47%); lowers the score of q (-24%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.32
if log_sum_pt > 6.91: z += 29.09 × (log_sum_pt − 6.91)
if z_top50_slots > 0.959: z += -40.30 × (z_top50_slots − 0.959)
if log_sum_pt > 6.89: z += 21.44 × (log_sum_pt − 6.89)
if log_sum_pt > 6.81: z += -9.33 × (log_sum_pt − 6.81)
if sum_pt_top50 > 959: z += -0.014 × (sum_pt_top50 − 959)
if sum_pt > 950: z += 0.011 × (sum_pt − 950)
if LHA < 0.404: z += 7.39 × (0.404 − LHA)
if n_pt_above_10 < 31.00: z += -0.091 × (31.00 − n_pt_above_10)
if sum_pt_top2 < 689: z += 0.0027 × (689 − sum_pt_top2)
if z_top30_slots > 0.934 and girth2_top5 < 0.017: z += 1590 × (z_top30_slots − 0.934) × (0.017 − girth2_top5)
if mass < 121: z += -0.018 × (121 − mass)
if mass_top30 < 80.40: z += 0.039 × (80.40 − mass_top30)
if mass_top30 < 80.40 and mass_top5 < 68.43: z += -0.00062 × (80.40 − mass_top30) × (68.43 − mass_top5)
if sum_pt_top30 < 1073: z += 0.0053 × (1073 − sum_pt_top30)
if z_top30_slots > 0.934 and mass_top10 < 91.19: z += -0.309 × (z_top30_slots − 0.934) × (91.19 − mass_top10)
if max_dr < 0.436 and z_dr_0p2_0p4 < 0.194: z += 30.95 × (0.436 − max_dr) × (0.194 − z_dr_0p2_0p4)
if n_particles > 26.00: z += 0.020 × (n_particles − 26.00)
if log_sum_pt > 6.96: z += -13.76 × (log_sum_pt − 6.96)
if max_dr < 0.436: z += -4.09 × (0.436 − max_dr)
if sum_pt_top5 < 902: z += 0.0011 × (902 − sum_pt_top5)
if z_top50_slots > 0.959 and girth2_top3 < 0.0012: z += 25898 × (z_top50_slots − 0.959) × (0.0012 − girth2_top3)
if n_particles > 38.00 and girth2_top2 < 0.023: z += 1.90 × (n_particles − 38.00) × (0.023 − girth2_top2)
if z_top20_slots < 0.958: z += -4.06 × (0.958 − z_top20_slots)
if girth2_top10 < 0.0077: z += 84.15 × (0.0077 − girth2_top10)
if n_pt_above_10 > 16.00: z += -0.052 × (n_pt_above_10 − 16.00)
if n_pt_above_10 < 31.00 and lam1 < 0.0047: z += 17.63 × (31.00 − n_pt_above_10) × (0.0047 − lam1)
if girth2_top15 < 0.0033 and n_real_top40 > 29.00: z += 39.94 × (0.0033 − girth2_top15) × (n_real_top40 − 29.00)
if z_top30_slots > 0.934 and mass_top5 > 22.18: z += 0.470 × (z_top30_slots − 0.934) × (mass_top5 − 22.18)
if girth2_top15 < 0.0033: z += -236 × (0.0033 − girth2_top15)
if n_pt_above_10 < 31.00 and mean_phi2 < 0.017: z += 1.18 × (31.00 − n_pt_above_10) × (0.017 − mean_phi2)
if n_dr_0p2_0p4 < 13.00: z += -0.030 × (13.00 − n_dr_0p2_0p4)
if girth2_top3 < 0.00082: z += -856 × (0.00082 − girth2_top3)
if n_particles > 38.00 and z_top50_slots < 0.985: z += 1.88 × (n_particles − 38.00) × (0.985 − z_top50_slots)
if z_top30_slots > 0.934 and girth2_top3 < 0.0012: z += -10002 × (z_top30_slots − 0.934) × (0.0012 − girth2_top3)
if log_sum_pt > 6.99: z += -7.30 × (log_sum_pt − 6.99)
if z_top40_slots < 0.967: z += -17.27 × (0.967 − z_top40_slots)
if LHA < 0.404 and lam2 < 0.0018: z += -860 × (0.404 − LHA) × (0.0018 − lam2)
if pt_9 < 34.06 and pt1_dr01 < 28.39: z += -0.00064 × (34.06 − pt_9) × (28.39 − pt1_dr01)
if girth2_top5 < 0.00066: z += 870 × (0.00066 − girth2_top5)
if sum_pt_top2 < 689 and mass_top5 > 2.16: z += 1.4e-05 × (689 − sum_pt_top2) × (mass_top5 − 2.16)
if n_particles > 38.00 and dr_4 < 0.200: z += 0.082 × (n_particles − 38.00) × (0.200 − dr_4)
if sum_pt_top2 < 689 and n_dr_0p2_0p4 < 7.00: z += -0.00017 × (689 − sum_pt_top2) × (7.00 − n_dr_0p2_0p4)
if z_top30_slots > 0.934 and D2 < 1.98: z += 5.67 × (z_top30_slots − 0.934) × (1.98 − D2)
if mass_top20 < 40.20 and mean_phi2 < 0.0059: z += 4.73 × (40.20 − mass_top20) × (0.0059 − mean_phi2)
if n_dr_0p2_0p4 < 13.00 and girth2_top3 < 0.00082: z += 71.16 × (13.00 − n_dr_0p2_0p4) × (0.00082 − girth2_top3)
if girth2_top10 < 0.00041: z += 1585 × (0.00041 − girth2_top10)
if log_sum_pt > 7.06: z += -6.72 × (log_sum_pt − 7.06)
if z_dr_0_0p05 > 0.879: z += -4.34 × (z_dr_0_0p05 − 0.879)
if sum_pt_top40 > 1032: z += 0.0018 × (sum_pt_top40 − 1032)
if log_sum_pt > 7.14: z += -6.42 × (log_sum_pt − 7.14)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (85 GeV), average width, pT spread over several particles** — 26.5% of jets, neuron 1.71.  
- **heavy (120 GeV), wide, pT spread over several particles** — 19.9% of jets, neuron 2.36.  
- **light (34 GeV), very narrow, pT spread over several particles** — 12.7% of jets, neuron 2.10.  
- **heavy (106 GeV), average width, pT spread over several particles** — 11.1% of jets, neuron 4.47.  
- **heavy (139 GeV), very wide, pT spread over several particles** — 8.2% of jets, neuron 4.63.  
- **medium-mass (57 GeV), narrow, pT spread over several particles** — 6.9% of jets, neuron 4.62.  
- **heavy (97 GeV), average width, pT spread over several particles, high pT** — 6.0% of jets, neuron 7.02.  
- **light (45 GeV), very narrow, pT spread over several particles** — 4.7% of jets, neuron 5.38.  
- **heavy (100 GeV), average width, pT spread over several particles, high pT** — 3.0% of jets, neuron 8.12.  
- **heavy (112 GeV), narrow, pT spread over several particles, high pT** — 1.0% of jets, neuron 9.03.  

### neuron 4: Elongated two-prong jet, Z-like mass (major)

- **What it measures:** Grows with an elongated two-prong pT pattern (small D2 and τ21, large eccentricity) and heavy hardest particles (the 10-15 hardest); mass below 91.19 GeV, and more so below 86.4 GeV, light 40 hardest particles (below 62.55 GeV) and a broad, soft spread (LHA above 0.115) push it down, and mass above 101.05 GeV loses a boost. Z jets sit highest (1.56; AUC 0.85), then W (1.22) and top (0.69) jets, with gluon (0.21) and quark (0.12) jets lowest.
- *computed — its value:* largest for Z (1.56), then W (1.22), then t (0.69), then g (0.21), then q (0.12); it separates Z jets from the rest best (AUC 0.85: large for Z)
- **How the class scores use it:** It lowers the q score (-31%) and the g score (-15%) strongly, because a heavy, two-prong jet is not a light jet, and it raises the W (+6%) and t (+3%) scores a little. The Z score hardly uses it, even though Z jets sit highest on it.
- *computed — used by:* raises the score of W (+6%), t (+3%); lowers the score of g (-15%), q (-31%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.06
if LHA > 0.115: z += -8.93 × (LHA − 0.115)
if mass < 101: z += 0.049 × (101 − mass)
if mass_top40 < 62.55: z += -0.152 × (62.55 − mass_top40)
if girth2_top40 < 0.013: z += 119 × (0.013 − girth2_top40)
if girth < 0.121: z += -12.71 × (0.121 − girth)
if mass < 86.40: z += -0.049 × (86.40 − mass)
if lam1 < 0.020: z += -50.35 × (0.020 − lam1)
if width > 0.0096: z += 199 × (width − 0.0096)
if girth2_top15 < 0.016: z += 65.36 × (0.016 − girth2_top15)
if mass < 91.19: z += -0.037 × (91.19 − mass)
if D2 < 6.92: z += 0.148 × (6.92 − D2)
if mass_top30 < 153: z += -0.0074 × (153 − mass_top30)
if mass_top40 < 94.64: z += -0.026 × (94.64 − mass_top40)
if girth2_top30 > 0.00086: z += 68.00 × (girth2_top30 − 0.00086)
if z_top50_slots > 0.959: z += 13.83 × (z_top50_slots − 0.959)
if n_dr_0p2_0p4 < 26.00: z += 0.027 × (26.00 − n_dr_0p2_0p4)
if e2 < 0.056: z += -17.82 × (0.056 − e2)
if mass_top40 < 83.33 and D2 < 6.92: z += -0.014 × (83.33 − mass_top40) × (6.92 − D2)
if mass_top40 < 83.33: z += 0.032 × (83.33 − mass_top40)
if mass_over_sum_pt > 0.060: z += -12.32 × (mass_over_sum_pt − 0.060)
if girth2_top30 > 0.0064: z += -94.46 × (girth2_top30 − 0.0064)
if girth < 0.062: z += -25.68 × (0.062 − girth)
if girth2_top50 > 0.0081: z += 85.68 × (girth2_top50 − 0.0081)
if lam1 > 0.0077: z += -101 × (lam1 − 0.0077)
if girth2_top50 > 0.0093: z += -76.62 × (girth2_top50 − 0.0093)
if girth2_top15 < 0.0073: z += -85.79 × (0.0073 − girth2_top15)
if mass_top15 < 69.03 and girth2_top2 < 0.004: z += -3.06 × (69.03 − mass_top15) × (0.004 − girth2_top2)
if z_dr_0p2_0p4 < 0.068: z += -4.54 × (0.068 − z_dr_0p2_0p4)
if mass < 101 and girth2_top2 < 0.004: z += 2.47 × (101 − mass) × (0.004 − girth2_top2)
if sum_pt_top20 < 994: z += 0.002 × (994 − sum_pt_top20)
if mass < 74.25: z += -0.021 × (74.25 − mass)
if mass_top15 < 69.03: z += 0.0096 × (69.03 − mass_top15)
if mass < 121 and max_dr < 0.402: z += 0.071 × (121 − mass) × (0.402 − max_dr)
if mass < 80.78: z += -0.016 × (80.78 − mass)
if sum_pt < 1066: z += -0.0031 × (1066 − sum_pt)
if mass > 144: z += 0.039 × (mass − 144)
if mass_over_sum_pt > 0.098: z += -11.63 × (mass_over_sum_pt − 0.098)
if n_particles > 29.00 and n_dr_0_0p05 < 25.00: z += -0.00051 × (n_particles − 29.00) × (25.00 − n_dr_0_0p05)
if e2 < 0.025: z += 25.00 × (0.025 − e2)
if mass < 78.26: z += -0.011 × (78.26 − mass)
if e2 > 0.037: z += 20.94 × (e2 − 0.037)
if mass_top20 > 104: z += -0.024 × (mass_top20 − 104)
if mass_top30 < 68.29: z += 0.0073 × (68.29 − mass_top30)
if mass_top40 > 137: z += -0.013 × (mass_top40 − 137)
if LHA > 0.404: z += -9.43 × (LHA − 0.404)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (96 GeV), average width, pT spread over several particles** — 25.3% of jets, neuron 1.41.  
- **medium-mass (81 GeV), narrow, pT spread over several particles** — 25.1% of jets, neuron 1.12.  
- **very heavy (168 GeV), very wide, pT spread over several particles** — 10.6% of jets, neuron 0.76.  
- **heavy (137 GeV), wide, pT spread over several particles** — 7.8% of jets, neuron 0.37.  
- **medium-mass (65 GeV), narrow, pT spread over several particles** — 7.6% of jets, neuron 0.01.  
- **medium-mass (52 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 0.00.  
- **light (24 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 0.00.  
- **light (34 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 0.00.  
- **light (42 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 0.00.  
- **very heavy (197 GeV), very wide, pT spread over several particles** — 1.8% of jets, neuron 0.75.  

### neuron 5: Few particles, compact jet (major)

- **What it measures:** Runs opposite to the particle count and the minor-axis width: it rises when the 50 hardest hold more than 0.959 of the pT and the 30 hardest have mass above 68.3 GeV, and falls when more than 22 of the 50 slots are filled, when some particle lies beyond ΔR 0.240 and when the mass exceeds 89.74 GeV. Z (2.45) and W (2.19) jets sit highest, quark jets next (1.81), gluon (0.67) and top (0.47; AUC 0.15, small for t) jets lowest.
- *computed — its value:* largest for Z (2.45), then W (2.19), then q (1.81), then g (0.67), then t (0.47); it separates t jets from the rest best (AUC 0.15: small for t)
- **How the class scores use it:** It raises the W (+20%) and Z (+22%) scores and lowers the t (-17%) and g (-11%) scores: a compact jet with few particles is boson-like, not top- or gluon-like. The q score hardly uses it; freezing it alone costs only 0.072 points, as other scales carry similar information.
- *computed — used by:* raises the score of W (+20%), Z (+22%); lowers the score of g (-11%), t (-17%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.701
if log_sum_pt > 6.86: z += 12.81 × (log_sum_pt − 6.86)
if log_sum_pt > 6.91: z += -22.28 × (log_sum_pt − 6.91)
if z_top40_slots > 0.930: z += -18.78 × (z_top40_slots − 0.930)
if mass_over_sum_pt_sq < 0.026: z += 56.14 × (0.026 − mass_over_sum_pt_sq)
if z_top50_slots > 0.959: z += 24.97 × (z_top50_slots − 0.959)
if max_dr > 0.240: z += -5.94 × (max_dr − 0.240)
if mass_top30 > 68.29: z += 0.030 × (mass_top30 − 68.29)
if n_real_top50 > 22.00: z += -0.030 × (n_real_top50 − 22.00)
if mass_over_sum_pt > 0.080: z += 30.33 × (mass_over_sum_pt − 0.080)
if mass > 89.74: z += -0.034 × (mass − 89.74)
if mass_over_sum_pt > 0.079: z += -26.11 × (mass_over_sum_pt − 0.079)
if n_particles < 64.00 and e2 < 0.039: z += 1.97 × (64.00 − n_particles) × (0.039 − e2)
if sum_pt > 908 and girth2_top2 < 0.014: z += 0.311 × (sum_pt − 908) × (0.014 − girth2_top2)
if sum_pt_top50 > 934: z += 0.0041 × (sum_pt_top50 − 934)
if sum_pt_top40 > 907: z += 0.0036 × (sum_pt_top40 − 907)
if mass > 64.49: z += 0.014 × (mass − 64.49)
if log_sum_pt > 6.89 and z_dr_0p1_0p2 < 0.187: z += -65.66 × (log_sum_pt − 6.89) × (0.187 − z_dr_0p1_0p2)
if log_sum_pt > 6.89: z += -6.64 × (log_sum_pt − 6.89)
if mass > 173: z += -0.528 × (mass − 173)
if mass_over_sum_pt_sq > 0.029: z += -1721 × (mass_over_sum_pt_sq − 0.029)
if girth < 0.086: z += -12.14 × (0.086 − girth)
if log_sum_pt > 6.92 and girth2_top2 < 0.014: z += -644 × (log_sum_pt − 6.92) × (0.014 − girth2_top2)
if n_dr_0p2_0p4 < 13.00: z += 0.052 × (13.00 − n_dr_0p2_0p4)
if log_sum_pt > 6.94 and z_dr_0p1_0p2 < 0.187: z += 70.18 × (log_sum_pt − 6.94) × (0.187 − z_dr_0p1_0p2)
if log_sum_pt > 6.96: z += 9.79 × (log_sum_pt − 6.96)
if girth2_top10 < 0.009: z += -55.07 × (0.009 − girth2_top10)
if z_dr_0p2_0p4 < 0.052: z += -8.46 × (0.052 − z_dr_0p2_0p4)
if mass > 163: z += -0.154 × (mass − 163)
if sum_pt_top40 > 936: z += 0.0021 × (sum_pt_top40 − 936)
if girth < 0.070: z += -11.53 × (0.070 − girth)
if mass_over_sum_pt > 0.171: z += -357 × (mass_over_sum_pt − 0.171)
if mass_over_sum_pt_sq > 0.029 and phi_6 > -0.117: z += -7187 × (mass_over_sum_pt_sq − 0.029) × (phi_6 − -0.117)
if sum_pt_top50 > 1061: z += -0.0067 × (sum_pt_top50 − 1061)
if log_sum_pt > 6.99: z += 9.04 × (log_sum_pt − 6.99)
if mass_over_sum_pt > 0.090: z += -12.17 × (mass_over_sum_pt − 0.090)
if girth2_top30 < 0.018 and z_dr_0p1_0p2 < 0.286: z += 73.42 × (0.018 − girth2_top30) × (0.286 − z_dr_0p1_0p2)
if girth2_top10 < 0.0048: z += 87.84 × (0.0048 − girth2_top10)
if mass_top30 > 102: z += -0.019 × (mass_top30 − 102)
if sum_pt_top15 > 951: z += -0.0057 × (sum_pt_top15 − 951)
if z_top20_slots > 0.923: z += -6.68 × (z_top20_slots − 0.923)
if girth < 0.086 and n_pt_above_50 > 2.00: z += -1.38 × (0.086 − girth) × (n_pt_above_50 − 2.00)
if mass_top40 > 137: z += -0.037 × (mass_top40 − 137)
if n_particles < 64.00 and z_dr_0p2_0p4 > 0.0047: z += 0.290 × (64.00 − n_particles) × (z_dr_0p2_0p4 − 0.0047)
if log_sum_pt > 6.94: z += -3.18 × (log_sum_pt − 6.94)
if sum_pt_top10 > 846: z += 0.0031 × (sum_pt_top10 − 846)
if mass_top50 > 91.19: z += 0.0054 × (mass_top50 − 91.19)
if girth < 0.070 and n_pt_above_50 > 2.00: z += 1.27 × (0.070 − girth) × (n_pt_above_50 − 2.00)
if sum_pt_top20 > 1005: z += 0.0033 × (sum_pt_top20 − 1005)
if max_dr > 0.436: z += 5.81 × (max_dr − 0.436)
if mass_top50 > 158 and z_dr_0p05_0p1 > 0.591: z += 1.22 × (mass_top50 − 158) × (z_dr_0p05_0p1 − 0.591)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (72 GeV), narrow, pT spread over several particles** — 67.3% of jets, neuron 1.94.  
- **very heavy (155 GeV), very wide, pT spread over several particles** — 14.9% of jets, neuron 0.29.  
- **medium-mass (81 GeV), narrow, pT spread over several particles** — 11.6% of jets, neuron 1.14.  
- **medium-mass (85 GeV), narrow, pT spread over several particles, high pT** — 2.9% of jets, neuron 1.08.  
- **very heavy (174 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 0.00.  
- **very heavy (187 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.00.  
- **very heavy (206 GeV), very wide, pT spread over several particles, high pT** — 0.6% of jets, neuron 0.00.  
- **very heavy (225 GeV), very wide, pT spread over several particles** — 0.2% of jets, neuron 0.00.  
- **very heavy (271 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  
- **very heavy (218 GeV), very wide, pT spread over several particles** — 0.1% of jets, neuron 0.00.  

### neuron 7: Mass window about 93-101 GeV (major)

- **What it measures:** Switches on mainly for jet mass between 92.9 and 101.05 GeV (every mass step below 92.9 GeV pushes it down, and it loses its boosts above 101.05 and 120.6 GeV), helped by an elongated two-prong pattern (large eccentricity, small τ21 and D2), few particles at 0.2 <= ΔR < 0.4 and a not-too-narrow jet. Almost only Z jets sit high on it (1.79; AUC 0.91); W (0.14), gluon (0.07), top (0.05) and quark (0.03) jets stay near zero.
- *computed — its value:* largest for Z (1.79), then W (0.14), then g (0.07), then t (0.05), then q (0.03); it separates Z jets from the rest best (AUC 0.91: large for Z)
- **How the class scores use it:** It raises the Z score (+8%) and lowers the W (-6%) and t (-3%) scores: a mass just above the Z peak points to a Z; freezing it costs 3.004 points. The g and q scores hardly use it.
- *computed — used by:* raises the score of Z (+8%); lowers the score of W (-6%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.302
if LHA < 0.320: z += 21.77 × (0.320 − LHA)
if mass < 121: z += 0.040 × (121 − mass)
if mass < 82.85: z += -0.131 × (82.85 − mass)
if girth < 0.086: z += -52.05 × (0.086 − girth)
if mass < 87.36: z += -0.100 × (87.36 − mass)
if e2_sq < 0.0096: z += 396 × (0.0096 − e2_sq)
if girth2_top20 < 0.008: z += -406 × (0.008 − girth2_top20)
if mass < 92.86: z += -0.067 × (92.86 − mass)
if sum_pt < 1261: z += -0.0049 × (1261 − sum_pt)
if mass_top50 < 92.17: z += -0.060 × (92.17 − mass_top50)
if mass < 101 and max_dr < 0.391: z += 0.880 × (101 − mass) × (0.391 − max_dr)
if mass < 101: z += 0.044 × (101 − mass)
if girth2_top40 < 0.008: z += -395 × (0.008 − girth2_top40)
if mass_top50 < 71.80: z += -0.120 × (71.80 − mass_top50)
if LHA < 0.333: z += -11.52 × (0.333 − LHA)
if mass_top50 < 97.93 and D2 < 1.60: z += -0.402 × (97.93 − mass_top50) × (1.60 − D2)
if mass < 91.19 and max_dr < 0.391: z += -1.18 × (91.19 − mass) × (0.391 − max_dr)
if girth2_top40 < 0.0088: z += 291 × (0.0088 − girth2_top40)
if mass_top50 < 86.40: z += 0.059 × (86.40 − mass_top50)
if mass_over_sum_pt < 0.090: z += -43.78 × (0.090 − mass_over_sum_pt)
if mass < 101 and D2 < 1.60: z += 0.266 × (101 − mass) × (1.60 − D2)
if mass < 91.19: z += -0.038 × (91.19 − mass)
if girth2_top30 < 0.0084: z += -201 × (0.0084 − girth2_top30)
if mass_top50 < 79.21 and pt_2 < 138: z += 0.0014 × (79.21 − mass_top50) × (138 − pt_2)
if girth2_top30 < 0.012: z += 87.70 × (0.012 − girth2_top30)
if girth2_top50 < 0.0078: z += 220 × (0.0078 − girth2_top50)
if mass < 86.40: z += 0.035 × (86.40 − mass)
if mass_top50 < 71.80 and max_dr < 0.394: z += 1.12 × (71.80 − mass_top50) × (0.394 − max_dr)
if mass_top30 < 76.42: z += 0.027 × (76.42 − mass_top30)
if n_dr_0p2_0p4 < 13.00 and z_1st < 0.499: z += 0.209 × (13.00 − n_dr_0p2_0p4) × (0.499 − z_1st)
if girth2_top40 < 0.0077: z += 139 × (0.0077 − girth2_top40)
if z_dr_0_0p05 > 0.181: z += 0.644 × (z_dr_0_0p05 − 0.181)
if girth2_top40 < 0.0077 and n_pt_above_10 > 10.00: z += -13.24 × (0.0077 − girth2_top40) × (n_pt_above_10 − 10.00)
if sum_pt_top30 < 1038: z += 0.0032 × (1038 − sum_pt_top30)
if mass_top20 < 66.84: z += 0.016 × (66.84 − mass_top20)
if mass_top30 < 86.25: z += 0.011 × (86.25 − mass_top30)
if D2 < 1.79 and sum_pt_top50 < 1246: z += 0.0033 × (1.79 − D2) × (1246 − sum_pt_top50)
if girth2_top20 < 0.0064: z += 101 × (0.0064 − girth2_top20)
if C2 < 0.056: z += -18.80 × (0.056 − C2)
if D2 < 1.79 and n_dr_0p2_0p4 < 9.00: z += 0.115 × (1.79 − D2) × (9.00 − n_dr_0p2_0p4)
if lam2 < 0.00062: z += 1130 × (0.00062 − lam2)
if mass < 91.19 and pt_dispersion < 0.338: z += -0.354 × (91.19 − mass) × (0.338 − pt_dispersion)
if girth2_top20 < 0.008 and z_dr_0p2_0p4 > 0.00066: z += 2872 × (0.008 − girth2_top20) × (z_dr_0p2_0p4 − 0.00066)
if D2 < 1.79: z += -0.429 × (1.79 − D2)
if z_top50_slots < 0.991: z += -20.30 × (0.991 − z_top50_slots)
if D2 < 1.79 and girth2_top50 < 0.0078: z += 299 × (1.79 − D2) × (0.0078 − girth2_top50)
if max_dr < 0.383 and eccentricity > 0.842: z += -19.04 × (0.383 − max_dr) × (eccentricity − 0.842)
if mass < 91.19 and z_dr_0p05_0p1 > 0.403: z += -0.090 × (91.19 − mass) × (z_dr_0p05_0p1 − 0.403)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very heavy (141 GeV), very wide, pT spread over several particles** — 29.8% of jets, neuron 0.27.  
- **medium-mass (87 GeV), average width, pT spread over several particles** — 16.6% of jets, neuron 0.40.  
- **medium-mass (88 GeV), average width, pT spread over several particles** — 12.5% of jets, neuron 1.97.  
- **light (49 GeV), very narrow, pT spread over several particles** — 9.1% of jets, neuron 0.00.  
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 8.0% of jets, neuron 0.20.  
- **light (30 GeV), very narrow, pT spread over several particles** — 7.8% of jets, neuron 0.00.  
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 7.4% of jets, neuron 0.00.  
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 4.3% of jets, neuron 0.21.  
- **light (31 GeV), very narrow, pT spread over several particles** — 3.7% of jets, neuron 0.00.  
- **very light (20 GeV), very narrow, pT spread over several particles** — 0.9% of jets, neuron 0.03.  

### neuron 8: Wide, busy jet for its pT (major)

- **What it measures:** Grows with the width of the jet relative to its pT (m/pT, girth², width, e2²) and with a busy outer ring (21 or more particles at 0.2 <= ΔR < 0.4); mass above 64.5, 80.8 and 87.4 GeV pushes it up, mass above 101.05 GeV pulls part of that back. Top jets sit far highest (3.95; AUC 0.92), then gluon (1.41) and quark (0.67) jets, with Z (0.32) and W (0.06) jets near zero.
- *computed — its value:* largest for t (3.95), then g (1.41), then q (0.67), then Z (0.32), then W (0.06); it separates t jets from the rest best (AUC 0.92: large for t)
- **How the class scores use it:** It lowers the W and Z scores (-25% each), since a boosted boson is a narrow, clean two-prong jet, and raises the t score a little (+6%). The g and q scores hardly use it; freezing it costs 1.008 points.
- *computed — used by:* raises the score of t (+6%); lowers the score of W (-25%), Z (-25%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.363
if mass > 64.49: z += 0.040 × (mass − 64.49)
if mass > 101: z += -0.100 × (mass − 101)
if mass_over_sum_pt_sq < 0.014: z += -154 × (0.014 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.098: z += -75.79 × (mass_over_sum_pt − 0.098)
if mass > 80.78: z += 0.045 × (mass − 80.78)
if girth < 0.097: z += 22.36 × (0.097 − girth)
if lam1 < 0.0082: z += 233 × (0.0082 − lam1)
if mass > 87.36: z += 0.041 × (mass − 87.36)
if girth2_top20 > 0.008: z += 216 × (girth2_top20 − 0.008)
if lam1 < 0.020: z += -46.48 × (0.020 − lam1)
if mass_over_sum_pt > 0.077 and sum_pt < 1116: z += 0.241 × (mass_over_sum_pt − 0.077) × (1116 − sum_pt)
if mass_top50 > 97.93: z += 0.042 × (mass_top50 − 97.93)
if mass_top50 > 82.04: z += -0.028 × (mass_top50 − 82.04)
if n_dr_0p2_0p4 < 21.00: z += -0.039 × (21.00 − n_dr_0p2_0p4)
if log_sum_pt < 6.94: z += 13.75 × (6.94 − log_sum_pt)
if mass_over_sum_pt > 0.089: z += 28.52 × (mass_over_sum_pt − 0.089)
if z_dr_0p2_0p4 < 0.091: z += 6.52 × (0.091 − z_dr_0p2_0p4)
if girth2_top20 > 0.0057: z += -94.56 × (girth2_top20 − 0.0057)
if girth < 0.140: z += -4.16 × (0.140 − girth)
if mass_over_sum_pt_sq < 0.0062: z += -221 × (0.0062 − mass_over_sum_pt_sq)
if mass > 125: z += -0.039 × (mass − 125)
if z_dr_0p1_0p2 < 0.155: z += 3.57 × (0.155 − z_dr_0p1_0p2)
if sum_pt_top30 < 1012: z += -0.004 × (1012 − sum_pt_top30)
if e2 > 0.025: z += -17.82 × (e2 − 0.025)
if sum_pt < 1013: z += 0.0092 × (1013 − sum_pt)
if mass_top15 > 30.36: z += 0.0052 × (mass_top15 − 30.36)
if mass_top40 > 79.55: z += -0.0093 × (mass_top40 − 79.55)
if sum_pt_top50 < 1031: z += -0.0047 × (1031 − sum_pt_top50)
if mass_over_sum_pt > 0.077: z += 6.95 × (mass_over_sum_pt − 0.077)
if z_top15_slots < 0.814: z += 3.46 × (0.814 − z_top15_slots)
if girth2_top40 > 0.0052: z += 21.61 × (girth2_top40 − 0.0052)
if sum_pt > 1261: z += -0.0087 × (sum_pt − 1261)
if mass > 87.36 and log_sum_pt < 6.90: z += -0.217 × (mass − 87.36) × (6.90 − log_sum_pt)
if sum_pt_top40 > 1226: z += 0.0086 × (sum_pt_top40 − 1226)
if z_top50_slots < 0.985: z += 16.77 × (0.985 − z_top50_slots)
if girth2_top40 > 0.0052 and sum_pt_top30 < 912: z += -0.277 × (girth2_top40 − 0.0052) × (912 − sum_pt_top30)
if sum_pt_top40 < 1002 and log_sum_pt < 6.81: z += -0.020 × (1002 − sum_pt_top40) × (6.81 − log_sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (86 GeV), average width, pT spread over several particles** — 46.6% of jets, neuron 0.42.  
- **light (45 GeV), very narrow, pT spread over several particles** — 27.8% of jets, neuron 0.07.  
- **very heavy (171 GeV), very wide, pT spread over several particles** — 7.7% of jets, neuron 4.44.  
- **heavy (120 GeV), wide, pT spread over several particles** — 5.8% of jets, neuron 3.38.  
- **very heavy (143 GeV), very wide, pT spread over several particles** — 4.0% of jets, neuron 4.72.  
- **very heavy (166 GeV), very wide, pT spread over several particles, high pT** — 2.5% of jets, neuron 2.46.  
- **medium-mass (59 GeV), narrow, pT spread over several particles, low pT** — 2.3% of jets, neuron 3.19.  
- **very heavy (169 GeV), very wide, pT spread over several particles** — 2.0% of jets, neuron 6.69.  
- **very heavy (224 GeV), very wide, pT spread over several particles, high pT** — 0.9% of jets, neuron 3.27.  
- **heavy (110 GeV), very wide, pT spread over several particles, low pT** — 0.5% of jets, neuron 8.94.  

### neuron 9: Light jet, mass off the W (major)

- **What it measures:** Falls as the mass of the hardest particles grows: mass between 62.55 and 82.85 GeV (the W side) pushes it down hard, and the 50 hardest weighing 136.8-160.8 GeV pull it down further; a small LHA (below 0.209) pushes it up. Quark jets sit highest (3.15; AUC 0.83), gluon jets next (1.80), top (0.41), W (0.38) and Z (0.27) jets low.
- *computed — its value:* largest for q (3.15), then g (1.80), then t (0.41), then W (0.38), then Z (0.27); it separates q jets from the rest best (AUC 0.83: large for q)
- **How the class scores use it:** It raises the q (+25%) and g (+14%) scores and lowers the W score (-6%): a light jet away from the W mass is a quark or gluon jet; freezing it costs 5.004 points. The Z and t scores hardly use it.
- *computed — used by:* raises the score of g (+14%), q (+25%); lowers the score of W (-6%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.589
if mass_top50 < 137: z += 0.056 × (137 − mass_top50)
if mass_top50 < 161: z += -0.033 × (161 − mass_top50)
if mass > 62.55: z += -0.057 × (mass − 62.55)
if mass > 82.85: z += 0.064 × (mass − 82.85)
if girth2_top15 < 0.021: z += -67.09 × (0.021 − girth2_top15)
if mass > 92.86: z += 0.035 × (mass − 92.86)
if mass_top40 < 91.29: z += -0.026 × (91.29 − mass_top40)
if LHA < 0.209: z += 22.75 × (0.209 − LHA)
if girth < 0.044: z += -72.51 × (0.044 − girth)
if mass > 144: z += -0.102 × (mass − 144)
if mass_top40 < 80.89: z += 0.028 × (80.89 − mass_top40)
if mass < 137: z += 0.0066 × (137 − mass)
if girth2_top15 < 0.01: z += 69.29 × (0.01 − girth2_top15)
if sum_pt < 950: z += 0.034 × (950 − sum_pt)
if mass_top30 < 76.42: z += 0.018 × (76.42 − mass_top30)
if mass_top40 < 121: z += 0.0053 × (121 − mass_top40)
if log_sum_pt < 6.86: z += -22.28 × (6.86 − log_sum_pt)
if mass_top15 > 40.20: z += -0.0065 × (mass_top15 − 40.20)
if girth2_top40 < 0.0063: z += 112 × (0.0063 − girth2_top40)
if girth2 < 0.0036: z += 281 × (0.0036 − girth2)
if girth > 0.097: z += 12.99 × (girth − 0.097)
if n_dr_0p2_0p4 < 5.00: z += 0.092 × (5.00 − n_dr_0p2_0p4)
if girth < 0.028: z += -38.94 × (0.028 − girth)
if sum_pt_top40 < 956: z += 0.0044 × (956 − sum_pt_top40)
if mass > 173: z += 0.067 × (mass − 173)
if width > 0.026: z += -88.63 × (width − 0.026)
if e2 > 0.065: z += -77.34 × (e2 − 0.065)
if sum_pt_top40 < 956 and z_top15_slots > 0.795: z += -0.069 × (956 − sum_pt_top40) × (z_top15_slots − 0.795)
if log_sum_pt < 6.86 and z_top20_slots > 0.897: z += 127 × (6.86 − log_sum_pt) × (z_top20_slots − 0.897)
if sum_pt_top40 < 956 and z_top30_slots > 0.973: z += -0.363 × (956 − sum_pt_top40) × (z_top30_slots − 0.973)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 26.6% of jets, neuron 0.27.  
- **heavy (92 GeV), average width, pT spread over several particles** — 22.2% of jets, neuron 0.01.  
- **light (32 GeV), very narrow, pT spread over several particles** — 13.3% of jets, neuron 4.49.  
- **medium-mass (55 GeV), very narrow, pT spread over several particles** — 12.3% of jets, neuron 2.95.  
- **very heavy (172 GeV), very wide, pT spread over several particles** — 10.1% of jets, neuron 0.17.  
- **very heavy (148 GeV), very wide, pT spread over several particles** — 6.0% of jets, neuron 0.66.  
- **heavy (119 GeV), wide, pT spread over several particles** — 6.0% of jets, neuron 0.20.  
- **medium-mass (72 GeV), average width, pT spread over several particles, low pT** — 1.9% of jets, neuron 3.01.  
- **very heavy (222 GeV), very wide, pT spread over several particles, high pT** — 1.0% of jets, neuron 0.04.  
- **medium-mass (54 GeV), average width, pT spread over several particles, low pT** — 0.5% of jets, neuron 4.61.  

### neuron 10: Hard particles spread far apart (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (large spread of the 2-5 hardest, little pT within ΔR < 0.05, large LHA and girth); e2 below 0.0652 and a pT share of the 5 hardest above 0.42 pull it down, and masses above 143.8 and 160.8 GeV trim it slightly. Top jets sit highest (2.28; AUC 0.83), Z (1.37) and W (1.34) jets in the middle, gluon (0.96) and quark (0.58) jets lowest.
- *computed — its value:* largest for t (2.28), then Z (1.37), then W (1.34), then g (0.96), then q (0.58); it separates t jets from the rest best (AUC 0.83: large for t)
- **How the class scores use it:** Only the t score uses it, raising it (+30%): widely spread hard prongs are the main positive sign of a top; freezing it costs 1.828 points.
- *computed — used by:* raises the score of t (+30%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.61
if mass < 87.36: z += 0.239 × (87.36 − mass)
if mass < 86.40: z += -0.205 × (86.40 − mass)
if e2 < 0.065: z += -31.09 × (0.065 − e2)
if z_top5 > 0.420: z += -3.43 × (z_top5 − 0.420)
if girth2_top10 > 0.00024: z += 80.14 × (girth2_top10 − 0.00024)
if mass < 80.40: z += 0.046 × (80.40 − mass)
if mass > 74.25: z += 0.020 × (mass − 74.25)
if lam1 > 0.0047: z += -90.44 × (lam1 − 0.0047)
if mass < 62.55: z += -0.061 × (62.55 − mass)
if mass > 144: z += -0.083 × (mass − 144)
if mass > 161: z += -0.187 × (mass − 161)
if D2 < 3.35: z += 0.238 × (3.35 − D2)
if mass_top50 > 137: z += 0.058 × (mass_top50 − 137)
if e2 < 0.065 and z_dr_0p1_0p2 < 0.219: z += -48.34 × (0.065 − e2) × (0.219 − z_dr_0p1_0p2)
if girth2_top5 < 0.024 and sum_pt_top3 < 656: z += 0.065 × (0.024 − girth2_top5) × (656 − sum_pt_top3)
if girth2 < 0.0059: z += 177 × (0.0059 − girth2)
if mass_top5 > 14.54: z += 0.012 × (mass_top5 − 14.54)
if girth < 0.121: z += -3.49 × (0.121 − girth)
if mass < 101: z += -0.0082 × (101 − mass)
if mass_top20 > 80.40: z += -0.020 × (mass_top20 − 80.40)
if LHA < 0.187: z += -10.78 × (0.187 − LHA)
if n_dr_0p2_0p4 < 13.00: z += -0.026 × (13.00 − n_dr_0p2_0p4)
if mass_top40 > 137: z += 0.044 × (mass_top40 − 137)
if z_top5 > 0.420 and dr_2 < 0.175: z += 6.61 × (z_top5 − 0.420) × (0.175 − dr_2)
if girth2_top10 > 0.0077: z += -55.30 × (girth2_top10 − 0.0077)
if mass > 163: z += 0.091 × (mass − 163)
if sum_pt_top30 < 1052: z += 0.0017 × (1052 − sum_pt_top30)
if dr_0 < 0.064 and n_dr_0p2_0p4 > 2.00: z += 0.783 × (0.064 − dr_0) × (n_dr_0p2_0p4 − 2.00)
if lam1 > 0.0019: z += 18.68 × (lam1 − 0.0019)
if z_dr_0_0p05 > 0.767: z += -2.19 × (z_dr_0_0p05 − 0.767)
if girth2_top15 < 0.0022: z += -247 × (0.0022 − girth2_top15)
if log_sum_pt < 6.90: z += -6.40 × (6.90 − log_sum_pt)
if sum_pt_top15 > 935: z += -0.0019 × (sum_pt_top15 − 935)
if mass < 121 and mass_top3 > 16.90: z += 0.00045 × (121 − mass) × (mass_top3 − 16.90)
if mass_top50 > 161: z += -0.044 × (mass_top50 − 161)
if mass_top50 > 158: z += 0.026 × (mass_top50 − 158)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (97 GeV), average width, pT spread over several particles** — 33.2% of jets, neuron 1.52.  
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 21.6% of jets, neuron 1.30.  
- **very heavy (151 GeV), very wide, pT spread over several particles** — 8.1% of jets, neuron 2.73.  
- **light (37 GeV), very narrow, pT spread over several particles** — 7.6% of jets, neuron 0.28.  
- **medium-mass (51 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 0.72.  
- **very heavy (173 GeV), very wide, pT spread over several particles** — 7.3% of jets, neuron 2.23.  
- **medium-mass (65 GeV), narrow, pT spread over several particles** — 6.7% of jets, neuron 0.99.  
- **light (24 GeV), very narrow, pT spread over several particles** — 6.6% of jets, neuron 0.04.  
- **very heavy (202 GeV), very wide, pT spread over several particles** — 1.3% of jets, neuron 0.13.  
- **very heavy (258 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00.  

### neuron 0: Light-side W mass, empty ring (moderate)

- **What it measures:** Falls as the jet gets heavier and as the ring at 0.2 <= ΔR < 0.4 fills up: mass above 74.3, 78.3 and 80.4 GeV pushes it down and mass above 92.9 GeV cuts it hard, while a small m/pT (below about 0.0905) pushes it up and very narrow jets are pushed down. W jets sit far highest (1.56; AUC 0.95), then quark (0.45), Z (0.26) and gluon (0.25) jets, with top jets lowest (0.08).
- *computed — its value:* largest for W (1.56), then q (0.45), then Z (0.26), then g (0.25), then t (0.08); it separates W jets from the rest best (AUC 0.95: large for W)
- **How the class scores use it:** It raises the W score (+9%) and lowers the Z score (-15%): a jet high on this scale is on the W side of the mass range rather than the Z side. The g, q and t scores hardly use it.
- *computed — used by:* raises the score of W (+9%); lowers the score of Z (-15%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.10
if mass > 92.86: z += -0.140 × (mass − 92.86)
if mass > 92.86 and max_dr < 0.391: z += -4.45 × (mass − 92.86) × (0.391 − max_dr)
if mass_over_sum_pt_sq < 0.0082: z += 475 × (0.0082 − mass_over_sum_pt_sq)
if mass > 78.26: z += -0.054 × (mass − 78.26)
if mass > 74.25: z += -0.039 × (mass − 74.25)
if mass_over_sum_pt > 0.083: z += 53.13 × (mass_over_sum_pt − 0.083)
if mass > 80.40: z += -0.036 × (mass − 80.40)
if girth < 0.057: z += -65.09 × (0.057 − girth)
if width < 0.0096: z += -185 × (0.0096 − width)
if mass_over_sum_pt > 0.077: z += -23.31 × (mass_over_sum_pt − 0.077)
if log_sum_pt < 7.02: z += 5.15 × (7.02 − log_sum_pt)
if mass > 92.86 and n_dr_0p2_0p4 < 7.00: z += -0.241 × (mass − 92.86) × (7.00 − n_dr_0p2_0p4)
if girth2_top20 < 0.006: z += -208 × (0.006 − girth2_top20)
if girth2_top40 < 0.0088: z += 109 × (0.0088 − girth2_top40)
if girth2_top20 < 0.006 and mass_top5 < 68.43: z += 3.07 × (0.006 − girth2_top20) × (68.43 − mass_top5)
if girth2_top40 < 0.0063: z += -217 × (0.0063 − girth2_top40)
if girth2_top50 < 0.0074: z += 142 × (0.0074 − girth2_top50)
if girth2_top20 < 0.0075: z += -96.27 × (0.0075 − girth2_top20)
if mass > 91.03: z += -0.017 × (mass − 91.03)
if mass_top30 < 89.17: z += -0.012 × (89.17 − mass_top30)
if mass_top50 < 82.04: z += -0.020 × (82.04 − mass_top50)
if log_sum_pt < 7.02 and max_dr < 0.391: z += 56.76 × (7.02 − log_sum_pt) × (0.391 − max_dr)
if sum_pt < 1013: z += -0.011 × (1013 − sum_pt)
if n_particles < 62.00: z += 0.012 × (62.00 − n_particles)
if LHA < 0.260: z += 4.96 × (0.260 − LHA)
if lam1 < 0.0067: z += -99.96 × (0.0067 − lam1)
if log_sum_pt < 6.99 and max_dr < 0.387: z += -70.00 × (6.99 − log_sum_pt) × (0.387 − max_dr)
if n_dr_0p2_0p4 < 10.00: z += 0.044 × (10.00 − n_dr_0p2_0p4)
if z_top30_slots > 0.920: z += -3.69 × (z_top30_slots − 0.920)
if girth2_top40 < 0.0063 and girth2_top3 < 0.0029: z += 39335 × (0.0063 − girth2_top40) × (0.0029 − girth2_top3)
if sum_pt_top40 < 859 and n_dr_0p1_0p2 < 33.00: z += -0.0023 × (859 − sum_pt_top40) × (33.00 − n_dr_0p1_0p2)
if mass_top20 < 70.42: z += 0.0089 × (70.42 − mass_top20)
if mass_top50 < 82.04 and dr_1 > 0.161: z += 1466 × (82.04 − mass_top50) × (dr_1 − 0.161)
if mass_top50 < 82.04 and z_dr_0p05_0p1 < 0.213: z += 0.030 × (82.04 − mass_top50) × (0.213 − z_dr_0p05_0p1)
if sum_pt < 1013 and z_dr_0p1_0p2 > 0.089: z += -0.013 × (1013 − sum_pt) × (z_dr_0p1_0p2 − 0.089)
if sum_pt < 1002: z += -0.0025 × (1002 − sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (87 GeV), average width, pT spread over several particles** — 53.3% of jets, neuron 0.75.  
- **light (44 GeV), very narrow, pT spread over several particles** — 27.5% of jets, neuron 0.46.  
- **very heavy (162 GeV), very wide, pT spread over several particles** — 12.4% of jets, neuron 0.00.  
- **very heavy (163 GeV), very wide, pT spread over several particles** — 4.4% of jets, neuron 0.00.  
- **very heavy (177 GeV), very wide, pT spread over several particles** — 1.5% of jets, neuron 0.00.  
- **heavy (131 GeV), wide, pT spread over several particles** — 0.8% of jets, neuron 0.00.  
- **very heavy (168 GeV), wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  

### neuron 3: Narrow, light, one-prong jet (moderate)

- **What it measures:** Rises as the jet gets narrower and lighter for its pT (small m/pT, width, girth and e2), with a thin minor axis (lam2 below 0.000615), few particles at 0.2 <= ΔR < 0.4 and light 20 hardest particles (below 45.6 GeV); a two-prong pattern (τ21 below 0.428) pushes it down. Quark jets sit highest (0.98), then W (0.65), gluon (0.49) and Z (0.46) jets, and top jets lowest (0.15; AUC 0.16, small for t).
- *computed — its value:* largest for q (0.98), then W (0.65), then g (0.49), then Z (0.46), then t (0.15); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** It raises the W and Z scores (+5% each) and lowers the g score (-9%): a narrow, clean jet is unlikely to be a gluon and fits a boson. The q and t scores hardly use it, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+5%), Z (+5%); lowers the score of g (-9%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.201
if mass_over_sum_pt_sq < 0.0075: z += 160 × (0.0075 − mass_over_sum_pt_sq)
if mass_over_sum_pt_sq < 0.0075 and mass_top20 < 137: z += -1.35 × (0.0075 − mass_over_sum_pt_sq) × (137 − mass_top20)
if tau21 < 0.428: z += -1.88 × (0.428 − tau21)
if lam2 < 0.00062: z += 1097 × (0.00062 − lam2)
if n_dr_0p2_0p4 < 5.00 and z_top50_slots > 0.979: z += 7.21 × (5.00 − n_dr_0p2_0p4) × (z_top50_slots − 0.979)
if mass_top20 < 45.59: z += 0.020 × (45.59 − mass_top20)
if z_top20_slots > 0.968 and girth2_top3 < 0.00082: z += 11425 × (z_top20_slots − 0.968) × (0.00082 − girth2_top3)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (133 GeV), very wide, pT spread over several particles** — 25.3% of jets, neuron 0.19.  
- **heavy (119 GeV), wide, pT spread over several particles** — 12.4% of jets, neuron 0.02.  
- **medium-mass (85 GeV), average width, pT spread over several particles** — 11.0% of jets, neuron 0.86.  
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 9.5% of jets, neuron 0.37.  
- **medium-mass (88 GeV), average width, pT spread over several particles** — 8.6% of jets, neuron 0.32.  
- **light (34 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 1.27.  
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.62.  
- **light (50 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.78.  
- **medium-mass (84 GeV), average width, pT spread over several particles** — 6.0% of jets, neuron 0.71.  
- **light (25 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 1.88.  

### neuron 6: One-prong jet away from Z mass (moderate)

- **What it measures:** Follows one-prong-ness (large τ21, small e2, light 10-30 hardest particles) and is pushed down for mass between about 86.4 and 101.05 GeV and for m/pT above 0.0508, while mass between 120.6 and 172.8 GeV pushes it up. Quark jets sit highest (1.28; AUC 0.81), gluon jets next (0.86), top jets in between (0.45), Z (0.14) and W (0.12) jets near zero.
- *computed — its value:* largest for q (1.28), then g (0.86), then t (0.45), then Z (0.14), then W (0.12); it separates q jets from the rest best (AUC 0.81: large for q)
- **How the class scores use it:** It raises the q score (+5%) and lowers the Z score (-11%): a one-prong jet outside the Z mass range is not a Z. The g, W and t scores hardly use it.
- *computed — used by:* raises the score of q (+5%); lowers the score of Z (-11%); does not (or hardly) enter the score of g, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.726
if mass < 101: z += -0.075 × (101 − mass)
if mass_over_sum_pt > 0.051: z += -40.80 × (mass_over_sum_pt − 0.051)
if mass < 173: z += 0.019 × (173 − mass)
if mass < 121: z += -0.039 × (121 − mass)
if mass_over_sum_pt_sq < 0.020: z += 121 × (0.020 − mass_over_sum_pt_sq)
if mass < 92.86: z += 0.065 × (92.86 − mass)
if mass_top40 < 161: z += -0.013 × (161 − mass_top40)
if girth2_top50 < 0.020: z += -77.10 × (0.020 − girth2_top50)
if e2 < 0.048: z += 35.48 × (0.048 − e2)
if sum_pt < 1085: z += 0.0087 × (1085 − sum_pt)
if girth2_top15 < 0.027: z += -29.14 × (0.027 − girth2_top15)
if mass < 86.40: z += 0.042 × (86.40 − mass)
if n_dr_0p2_0p4 < 21.00: z += -0.043 × (21.00 − n_dr_0p2_0p4)
if LHA > 0.260: z += 11.76 × (LHA − 0.260)
if log_sum_pt < 6.99: z += -7.22 × (6.99 − log_sum_pt)
if girth2_top20 > 0.0018: z += 73.54 × (girth2_top20 − 0.0018)
if width < 0.0096: z += -119 × (0.0096 − width)
if mass_over_sum_pt > 0.051 and n_dr_0p2_0p4 < 21.00: z += 0.978 × (mass_over_sum_pt − 0.051) × (21.00 − n_dr_0p2_0p4)
if mass_over_sum_pt > 0.051 and lam2 < 0.0024: z += 8150 × (mass_over_sum_pt − 0.051) × (0.0024 − lam2)
if girth2_top3 > 0.010: z += -177 × (girth2_top3 − 0.010)
if girth2_top3 > 0.010 and D2 < 4.45: z += 68.58 × (girth2_top3 − 0.010) × (4.45 − D2)
if girth2_top20 > 0.017: z += -209 × (girth2_top20 − 0.017)
if LHA > 0.228: z += 4.41 × (LHA − 0.228)
if lam1 < 0.0073: z += 113 × (0.0073 − lam1)
if LHA > 0.228 and D2 < 3.35: z += -2.44 × (LHA − 0.228) × (3.35 − D2)
if mass_top40 < 94.64: z += 0.0097 × (94.64 − mass_top40)
if e2 < 0.044: z += 12.15 × (0.044 − e2)
if e2 < 0.030: z += -26.97 × (0.030 − e2)
if girth2_top20 > 0.0018 and max_dr < 0.436: z += 359 × (girth2_top20 − 0.0018) × (0.436 − max_dr)
if mass_top50 < 71.80: z += 0.018 × (71.80 − mass_top50)
if lam1 > 0.0082 and sum_pt < 1167: z += 0.298 × (lam1 − 0.0082) × (1167 − sum_pt)
if e2 > 0.056: z += 112 × (e2 − 0.056)
if mass_over_sum_pt > 0.051 and max_pair_mass > 13.05: z += -0.279 × (mass_over_sum_pt − 0.051) × (max_pair_mass − 13.05)
if lam1 < 0.0047: z += 109 × (0.0047 − lam1)
if girth2_top20 < 0.008: z += 32.36 × (0.008 − girth2_top20)
if lam1 > 0.012: z += -48.69 × (lam1 − 0.012)
if e2 > 0.056 and n_real_top40 > 34.00: z += -12.19 × (e2 − 0.056) × (n_real_top40 − 34.00)
if lam1 > 0.0082: z += 28.53 × (lam1 − 0.0082)
if LHA > 0.404 and z_2 > 0.050: z += 495 × (LHA − 0.404) × (z_2 − 0.050)
if sum_pt_top40 > 1140: z += 0.0019 × (sum_pt_top40 − 1140)
if mass_top10 > 99.07: z += 0.033 × (mass_top10 − 99.07)
if e2 > 0.056 and max_dr < 0.369: z += -1233 × (e2 − 0.056) × (0.369 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 25.2% of jets, neuron 0.03.  
- **heavy (94 GeV), average width, pT spread over several particles** — 23.8% of jets, neuron 0.16.  
- **light (45 GeV), very narrow, pT spread over several particles** — 10.2% of jets, neuron 1.36.  
- **light (27 GeV), very narrow, pT spread over several particles** — 9.9% of jets, neuron 1.91.  
- **heavy (131 GeV), wide, pT spread over several particles** — 8.5% of jets, neuron 0.97.  
- **medium-mass (64 GeV), narrow, pT spread over several particles** — 8.4% of jets, neuron 0.60.  
- **very heavy (165 GeV), very wide, pT spread over several particles** — 5.7% of jets, neuron 0.25.  
- **very heavy (161 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 0.62.  
- **very heavy (175 GeV), very wide, pT spread over several particles** — 2.7% of jets, neuron 0.36.  
- **very heavy (183 GeV), very wide, pT spread over several particles** — 2.1% of jets, neuron 0.52.  

### neuron 12: Light jet without a dense core (moderate)

- **What it measures:** On for jets lighter than 82.85 GeV (more so below 78.3 and 74.3 GeV), but that is cancelled when more than 0.329 of the pT sits within ΔR < 0.05, and very light (below 62.55 GeV) or very narrow jets are pushed down. Quark jets sit highest (0.47; AUC 0.72), then gluon (0.31) and W (0.22) jets, with top (0.05) and Z (0.04) jets near zero.
- *computed — its value:* largest for q (0.47), then g (0.31), then W (0.22), then t (0.05), then Z (0.04); it separates q jets from the rest best (AUC 0.72: large for q)
- **How the class scores use it:** Only the q score uses it, raising it slightly (+3%): a light jet whose pT is not packed into a tiny core looks like a quark jet. The g, W, Z and t scores hardly use it.
- *computed — used by:* raises the score of q (+3%); does not (or hardly) enter the score of g, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.0004
if mass < 82.85 and z_dr_0_0p05 > 0.329: z += -0.130 × (82.85 − mass) × (z_dr_0_0p05 − 0.329)
if mass < 82.85: z += 0.074 × (82.85 − mass)
if mass_top50 < 77.38: z += -0.068 × (77.38 − mass_top50)
if mass < 78.26: z += 0.050 × (78.26 − mass)
if mass < 62.55: z += -0.090 × (62.55 − mass)
if girth < 0.050: z += -54.92 × (0.050 − girth)
if mass_top50 < 80.40 and z_dr_0p05_0p1 < 0.647: z += 0.059 × (80.40 − mass_top50) × (0.647 − z_dr_0p05_0p1)
if mass_top50 < 80.40: z += 0.030 × (80.40 − mass_top50)
if mass < 82.85 and z_dr_0p05_0p1 < 0.647: z += 0.046 × (82.85 − mass) × (0.647 − z_dr_0p05_0p1)
if mass < 74.25: z += 0.029 × (74.25 − mass)
if mass_top50 < 77.38 and z_dr_0p05_0p1 < 0.851: z += 0.029 × (77.38 − mass_top50) × (0.851 − z_dr_0p05_0p1)
if mass < 86.40: z += -0.015 × (86.40 − mass)
if girth < 0.050 and n_particles < 64.00: z += -0.848 × (0.050 − girth) × (64.00 − n_particles)
if LHA < 0.228: z += 6.31 × (0.228 − LHA)
if mass_top40 < 89.68 and n_particles > 22.00: z += -0.00053 × (89.68 − mass_top40) × (n_particles − 22.00)
if planar_flow > 0.259: z += -0.567 × (planar_flow − 0.259)
if e2_sq < 0.0069: z += 87.99 × (0.0069 − e2_sq)
if C2 > 0.061: z += -7.70 × (C2 − 0.061)
if log_sum_pt < 6.86: z += -15.65 × (6.86 − log_sum_pt)
if girth2_top5 < 0.001: z += 469 × (0.001 − girth2_top5)
if lam2 < 0.00062: z += -625 × (0.00062 − lam2)
if mass_top50 < 43.67: z += 0.050 × (43.67 − mass_top50)
if mass < 86.40 and max_dr < 0.391: z += 0.145 × (86.40 − mass) × (0.391 − max_dr)
if mass > 144: z += 0.022 × (mass − 144)
if mass_top50 < 43.67 and max_dr < 0.436: z += -0.422 × (43.67 − mass_top50) × (0.436 − max_dr)
if mass < 86.40 and z_dr_0p05_0p1 > 0.299: z += 0.104 × (86.40 − mass) × (z_dr_0p05_0p1 − 0.299)
if mass_top20 > 125 and C2 > 0.056: z += -0.364 × (mass_top20 − 125) × (C2 − 0.056)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **heavy (94 GeV), average width, pT spread over several particles** — 55.8% of jets, neuron 0.07.  
- **very heavy (173 GeV), very wide, pT spread over several particles** — 12.7% of jets, neuron 0.12.  
- **light (47 GeV), very narrow, pT spread over several particles** — 6.0% of jets, neuron 0.58.  
- **light (36 GeV), very narrow, pT spread over several particles** — 5.9% of jets, neuron 0.51.  
- **medium-mass (70 GeV), narrow, pT spread over several particles** — 5.7% of jets, neuron 0.45.  
- **medium-mass (58 GeV), narrow, pT spread over several particles** — 5.4% of jets, neuron 0.75.  
- **light (26 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 0.47.  
- **very light (18 GeV), very narrow, pT spread over several particles** — 1.4% of jets, neuron 0.43.  
- **heavy (100 GeV), wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.01.  
- **light (39 GeV), narrow, pT spread over several particles, low pT** — 0.5% of jets, neuron 0.00.  

### neuron 13: High pT, lighter than a top (moderate)

- **What it measures:** Grows with the total jet pT (below about 1053 GeV it is pushed down) and with the number of particles, especially near the axis, and is cut for heavy jets (mass above 143.8 and 160.8 GeV, or the 50 hardest above 97.9 GeV) and for a broad spread (LHA above 0.228). All non-top types sit at similar values (Z 2.06, W 1.87, gluon 1.81, quark 1.52); top jets sit lowest (0.68; AUC 0.18, small for t).
- *computed — its value:* largest for Z (2.06), then W (1.87), then g (1.81), then q (1.52), then t (0.68); it separates t jets from the rest best (AUC 0.18: small for t)
- **How the class scores use it:** Only the t score uses it, lowering it strongly (-34%): a jet high on this scale is not heavy and spread out like a top, so it is the main negative top handle. The g, q, W and Z scores hardly use it.
- *computed — used by:* lowers the score of t (-34%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 1.62
if girth2 < 0.014: z += 113 × (0.014 − girth2)
if mass > 161: z += -0.360 × (mass − 161)
if girth > 0.050: z += 21.78 × (girth − 0.050)
if z_dr_0p2_0p4 < 0.129: z += -6.58 × (0.129 − z_dr_0p2_0p4)
if LHA > 0.228: z += -9.56 × (LHA − 0.228)
z += 0.011 × n_particles
if sum_pt < 1053: z += -0.012 × (1053 − sum_pt)
if girth < 0.121: z += -7.61 × (0.121 − girth)
if mass > 144: z += -0.106 × (mass − 144)
if sum_pt_top40 < 1053: z += 0.0067 × (1053 − sum_pt_top40)
if e2_sq < 0.0096: z += 98.89 × (0.0096 − e2_sq)
if mass_top50 > 97.93: z += -0.029 × (mass_top50 − 97.93)
if sum_pt < 1013: z += -0.017 × (1013 − sum_pt)
if sum_pt_top50 < 1014: z += 0.012 × (1014 − sum_pt_top50)
if mass > 62.55: z += 0.0091 × (mass − 62.55)
if mass > 74.25: z += 0.012 × (mass − 74.25)
if mass_top40 > 137: z += 0.086 × (mass_top40 − 137)
if mass > 173: z += -0.385 × (mass − 173)
if z_top30_slots > 0.946: z += -10.38 × (z_top30_slots − 0.946)
if sum_pt_top50 < 988: z += 0.014 × (988 − sum_pt_top50)
if girth2_top40 > 0.0066: z += -53.92 × (girth2_top40 − 0.0066)
if log_sum_pt < 6.90: z += -13.62 × (6.90 − log_sum_pt)
if mass_over_sum_pt > 0.171: z += 375 × (mass_over_sum_pt − 0.171)
if mass_top50 > 137: z += 0.047 × (mass_top50 − 137)
if mass > 62.55 and n_dr_0p2_0p4 < 26.00: z += 0.00049 × (mass − 62.55) × (26.00 − n_dr_0p2_0p4)
if mass > 161 and D2 < 5.38: z += 0.032 × (mass − 161) × (5.38 − D2)
if sum_pt_top40 < 956: z += -0.012 × (956 − sum_pt_top40)
if mass > 121: z += -0.021 × (mass − 121)
if mass > 173 and pt_6 < 62.25: z += -0.012 × (mass − 173) × (62.25 − pt_6)
if mass_over_sum_pt > 0.079: z += 6.97 × (mass_over_sum_pt − 0.079)
if sum_pt_top50 < 959: z += 0.013 × (959 − sum_pt_top50)
if mass_over_sum_pt_sq > 0.029: z += -605 × (mass_over_sum_pt_sq − 0.029)
if sum_pt_top30 < 966: z += -0.0036 × (966 − sum_pt_top30)
if sum_pt_top20 > 957 and dr_0 < 0.072: z += -0.063 × (sum_pt_top20 − 957) × (0.072 − dr_0)
if mass > 74.25 and D2 > 0.603: z += -0.0027 × (mass − 74.25) × (D2 − 0.603)
if mass > 163: z += 0.061 × (mass − 163)
if mass_top5 > 37.76: z += 0.013 × (mass_top5 − 37.76)
if sum_pt_top50 < 1014 and D2 < 4.45: z += -0.0018 × (1014 − sum_pt_top50) × (4.45 − D2)
if sum_pt < 1013 and n_dr_0p1_0p2 < 26.00: z += -0.0003 × (1013 − sum_pt) × (26.00 − n_dr_0p1_0p2)
if sum_pt < 986: z += -0.006 × (986 − sum_pt)
if sum_pt_top30 > 1111 and dr_0 < 0.093: z += 0.081 × (sum_pt_top30 − 1111) × (0.093 − dr_0)
if mass_top15 > 86.40: z += -0.016 × (mass_top15 − 86.40)
if mass > 173 and z_6 < 0.042: z += 13.72 × (mass − 173) × (0.042 − z_6)
if log_sum_pt < 6.81: z += 12.20 × (6.81 − log_sum_pt)
if sum_pt_top30 > 1111: z += -0.0045 × (sum_pt_top30 − 1111)
if sum_pt < 1085 and sum_pt_top30 < 801: z += -4.6e-05 × (1085 − sum_pt) × (801 − sum_pt_top30)
if mass_top50 > 161: z += 0.029 × (mass_top50 − 161)
if mass_top50 > 169: z += -0.047 × (mass_top50 − 169)
if log_sum_pt < 6.81 and dr_5 < 0.051: z += -421 × (6.81 − log_sum_pt) × (0.051 − dr_5)
if sum_pt > 1261 and dr_0 < 0.112: z += -0.040 × (sum_pt − 1261) × (0.112 − dr_0)
if sum_pt > 1261: z += 0.0023 × (sum_pt − 1261)
if n_dr_0_0p05 > 30.00: z += -0.079 × (n_dr_0_0p05 − 30.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (73 GeV), narrow, pT spread over several particles** — 74.5% of jets, neuron 1.95.  
- **very heavy (145 GeV), very wide, pT spread over several particles** — 9.4% of jets, neuron 1.24.  
- **very heavy (172 GeV), very wide, pT spread over several particles** — 6.5% of jets, neuron 0.13.  
- **medium-mass (87 GeV), average width, pT spread over several particles** — 5.9% of jets, neuron 0.21.  
- **very heavy (192 GeV), very wide, pT spread over several particles** — 1.1% of jets, neuron 0.00.  
- **medium-mass (63 GeV), average width, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00.  
- **very heavy (173 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.06.  
- **very heavy (205 GeV), very wide, pT spread over several particles** — 0.4% of jets, neuron 0.00.  
- **very heavy (229 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00.  
- **very heavy (277 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  

### neuron 2: High total pT, tight hardest core (minor)

- **What it measures:** Follows the total jet pT (above about 1017 GeV pushes it up) and rises when the 15 hardest particles sit close to the axis and no particle is far out; mass below 92.9 GeV pushes it down, which is mostly given back below 125.1 and 91.19 GeV, so masses of about 93-125 GeV sit a little higher. Gluon jets sit highest (1.86), then W (1.32) and Z (1.23), with quark (0.59) and top (0.30; AUC 0.23, small for t) jets lowest.
- *computed — its value:* largest for g (1.86), then W (1.32), then Z (1.23), then q (0.59), then t (0.30); it separates t jets from the rest best (AUC 0.23: small for t)
- **How the class scores use it:** It raises the q score (+5%) and lowers the Z score (-8%) as a small balancing correction; the g, W and t scores hardly use it, and freezing it costs only 0.048 points.
- *computed — used by:* raises the score of q (+5%); lowers the score of Z (-8%); does not (or hardly) enter the score of g, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.457
if girth2_top15 < 0.016: z += 110 × (0.016 − girth2_top15)
if mass < 92.86: z += -0.060 × (92.86 − mass)
if mass_over_sum_pt < 0.141: z += -15.13 × (0.141 − mass_over_sum_pt)
if sum_pt > 1017: z += 0.015 × (sum_pt − 1017)
if mass < 125: z += 0.014 × (125 − mass)
if mass < 91.19: z += 0.032 × (91.19 − mass)
if max_dr < 0.402: z += 7.62 × (0.402 − max_dr)
if sum_pt > 1066: z += -0.012 × (sum_pt − 1066)
if lam1 < 0.0062: z += -187 × (0.0062 − lam1)
if log_sum_pt > 6.90 and girth2_top10 < 0.020: z += 344 × (log_sum_pt − 6.90) × (0.020 − girth2_top10)
if mass_top30 < 91.70: z += 0.012 × (91.70 − mass_top30)
if sum_pt_top50 > 997: z += -0.0044 × (sum_pt_top50 − 997)
if mass_over_sum_pt < 0.074: z += -24.78 × (0.074 − mass_over_sum_pt)
if sum_pt_top40 > 1070: z += 0.010 × (sum_pt_top40 − 1070)
if log_sum_pt > 6.90: z += -3.88 × (log_sum_pt − 6.90)
if girth2_top15 < 0.0061: z += 87.05 × (0.0061 − girth2_top15)
if sum_pt_top15 > 1003: z += 0.013 × (sum_pt_top15 − 1003)
if sum_pt > 996: z += 0.0026 × (sum_pt − 996)
if sum_pt_top20 > 909: z += -0.0023 × (sum_pt_top20 − 909)
if sum_pt_top50 > 1039: z += 0.0039 × (sum_pt_top50 − 1039)
if mass_over_sum_pt < 0.098: z += 5.45 × (0.098 − mass_over_sum_pt)
if mass_top40 > 161: z += -0.146 × (mass_top40 − 161)
if mass_top50 < 86.40: z += 0.0049 × (86.40 − mass_top50)
if log_sum_pt > 6.90 and pt_6 < 42.78: z += 0.274 × (log_sum_pt − 6.90) × (42.78 − pt_6)
if sum_pt_top50 > 1107: z += -0.0019 × (sum_pt_top50 − 1107)
if sum_pt > 1116: z += -0.0015 × (sum_pt − 1116)
if mass_top20 > 125: z += 0.010 × (mass_top20 − 125)
if sum_pt_top20 > 1129: z += 0.00079 × (sum_pt_top20 − 1129)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (86 GeV), average width, pT spread over several particles** — 41.6% of jets, neuron 1.09.  
- **very heavy (153 GeV), very wide, pT spread over several particles** — 18.8% of jets, neuron 0.12.  
- **light (31 GeV), very narrow, pT spread over several particles** — 12.7% of jets, neuron 0.16.  
- **medium-mass (57 GeV), narrow, pT spread over several particles** — 10.6% of jets, neuron 0.59.  
- **heavy (103 GeV), average width, pT spread over several particles** — 7.4% of jets, neuron 2.43.  
- **heavy (108 GeV), average width, pT spread over several particles, high pT** — 3.3% of jets, neuron 4.28.  
- **light (47 GeV), very narrow, pT spread over several particles** — 3.3% of jets, neuron 3.11.  
- **medium-mass (85 GeV), narrow, pT spread over several particles, high pT** — 1.7% of jets, neuron 3.99.  
- **very heavy (239 GeV), very wide, pT spread over several particles, high pT** — 0.4% of jets, neuron 0.03.  
- **heavy (104 GeV), narrow, pT spread over several particles, high pT** — 0.4% of jets, neuron 4.12.  

### neuron 11: Clean two-prong jet, 80-93 GeV (minor)

- **What it measures:** Large for mass below 92.9 GeV but pushed down below 80.4 GeV (and when the 50 hardest weigh below 86.4 GeV), so it peaks around 80-93 GeV, especially with few particles at 0.2 <= ΔR < 0.4, a thin minor axis and an elongated pattern; very narrow jets (small girth, e2 below 0.0388) are pushed down. W jets sit highest (1.22; AUC 0.81), Z jets next (0.89), gluon (0.34), quark (0.34) and top (0.33) jets low.
- *computed — its value:* largest for W (1.22), then Z (0.89), then g (0.34), then q (0.34), then t (0.33); it separates W jets from the rest best (AUC 0.81: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the q score (-6%): a clean two-prong jet at the W mass is a W, not a quark jet. The g, Z and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of q (-6%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.228
if mass < 92.86: z += 0.059 × (92.86 − mass)
if lam1 < 0.0067: z += -337 × (0.0067 − lam1)
if girth2_top20 < 0.0075: z += -206 × (0.0075 − girth2_top20)
if mass < 101: z += 0.023 × (101 − mass)
if e2 < 0.039: z += -40.63 × (0.039 − e2)
if girth < 0.077: z += -21.93 × (0.077 − girth)
if mass_top50 < 86.40: z += -0.032 × (86.40 − mass_top50)
if LHA < 0.372: z += 3.74 × (0.372 − LHA)
if e2 < 0.028: z += 70.08 × (0.028 − e2)
if mass_over_sum_pt_sq < 0.0082: z += 161 × (0.0082 − mass_over_sum_pt_sq)
if girth < 0.062: z += 27.43 × (0.062 − girth)
if girth < 0.086: z += 11.14 × (0.086 − girth)
if mass < 80.40: z += -0.028 × (80.40 − mass)
if girth2_top30 < 0.0064: z += -159 × (0.0064 − girth2_top30)
if girth2_top20 < 0.0057: z += 162 × (0.0057 − girth2_top20)
if n_dr_0p2_0p4 < 10.00 and n_dr_0p1_0p2 < 21.00: z += 0.006 × (10.00 − n_dr_0p2_0p4) × (21.00 − n_dr_0p1_0p2)
if mass < 62.55: z += -0.047 × (62.55 − mass)
if LHA < 0.310: z += -3.73 × (0.310 − LHA)
if mass < 80.40 and z_dr_0p2_0p4 < 0.037: z += -0.816 × (80.40 − mass) × (0.037 − z_dr_0p2_0p4)
if width < 0.0062: z += -170 × (0.0062 − width)
if mass_top30 < 60.44: z += 0.031 × (60.44 − mass_top30)
if mass < 101 and z_dr_0p1_0p2 < 0.334: z += 0.033 × (101 − mass) × (0.334 − z_dr_0p1_0p2)
if mass < 101 and z_dr_0p2_0p4 < 0.026: z += 0.513 × (101 − mass) × (0.026 − z_dr_0p2_0p4)
if n_dr_0p2_0p4 < 10.00 and z_dr_0p1_0p2 < 0.334: z += -0.175 × (10.00 − n_dr_0p2_0p4) × (0.334 − z_dr_0p1_0p2)
if z_top5_slots > 0.535: z += -1.41 × (z_top5_slots − 0.535)
if mass_top50 < 86.40 and girth2_top15 < 0.0042: z += 2.39 × (86.40 − mass_top50) × (0.0042 − girth2_top15)
if n_dr_0p2_0p4 < 5.00: z += 0.095 × (5.00 − n_dr_0p2_0p4)
if n_dr_0p2_0p4 < 10.00 and girth2 < 0.0055: z += -18.91 × (10.00 − n_dr_0p2_0p4) × (0.0055 − girth2)
if girth2_top30 < 0.0058: z += 60.62 × (0.0058 − girth2_top30)
if mass < 101 and sum_pt_top10 < 868: z += -3.9e-05 × (101 − mass) × (868 − sum_pt_top10)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very heavy (146 GeV), very wide, pT spread over several particles** — 26.3% of jets, neuron 0.26.  
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 13.0% of jets, neuron 1.64.  
- **heavy (91 GeV), average width, pT spread over several particles** — 12.1% of jets, neuron 1.28.  
- **heavy (95 GeV), average width, pT spread over several particles** — 8.7% of jets, neuron 0.24.  
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 8.1% of jets, neuron 0.52.  
- **light (35 GeV), very narrow, pT spread over several particles** — 7.3% of jets, neuron 0.32.  
- **light (48 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.46.  
- **medium-mass (62 GeV), narrow, pT spread over several particles** — 6.5% of jets, neuron 0.52.  
- **light (23 GeV), very narrow, pT spread over several particles** — 5.8% of jets, neuron 0.34.  
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 5.1% of jets, neuron 0.38.  

### neuron 14: Mass just above the Z peak (minor)

- **What it measures:** Large for m/pT between about 0.0905 and 0.118 together with mass above 91.03 GeV (masses below 89.74 and 91.03 GeV push it down) and below 136.8 GeV; it rises with mass overall. Z jets sit highest (0.94; AUC 0.86), gluon (0.34) and top (0.22) jets well below, quark (0.12) and W (0.04) jets lowest.
- *computed — its value:* largest for Z (0.94), then g (0.34), then t (0.22), then q (0.12), then W (0.04); it separates Z jets from the rest best (AUC 0.86: large for Z)
- **How the class scores use it:** Only the W score uses it, lowering it (-10%): a jet heavier than the W and at the Z peak or above is not a W. The Z score hardly uses it even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-10%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.424
if mass_over_sum_pt < 0.090: z += -128 × (0.090 − mass_over_sum_pt)
if mass_over_sum_pt < 0.118: z += 46.90 × (0.118 − mass_over_sum_pt)
if mass_over_sum_pt < 0.098: z += 78.16 × (0.098 − mass_over_sum_pt)
if mass < 89.74: z += -0.080 × (89.74 − mass)
if mass_over_sum_pt < 0.141: z += 21.05 × (0.141 − mass_over_sum_pt)
if mass_top50 < 89.08: z += 0.072 × (89.08 − mass_top50)
if mass_top50 < 92.17: z += -0.062 × (92.17 − mass_top50)
if mass < 91.03: z += -0.062 × (91.03 − mass)
if girth2_top50 < 0.0093: z += -299 × (0.0093 − girth2_top50)
if mass < 137: z += 0.016 × (137 − mass)
if girth2_top30 < 0.012: z += -141 × (0.012 − girth2_top30)
if mass_over_sum_pt < 0.083: z += -56.34 × (0.083 − mass_over_sum_pt)
if lam1 < 0.0073: z += 330 × (0.0073 − lam1)
if mass_top40 < 111: z += -0.021 × (111 − mass_top40)
if lam1 < 0.0082: z += -245 × (0.0082 − lam1)
if sum_pt_top50 > 976: z += -0.009 × (sum_pt_top50 − 976)
if mass < 80.40: z += -0.061 × (80.40 − mass)
if max_dr > 0.240: z += -4.78 × (max_dr − 0.240)
if lam2 < 0.0024: z += 376 × (0.0024 − lam2)
if mass < 91.03 and z_dr_0p1_0p2 < 0.219: z += 0.176 × (91.03 − mass) × (0.219 − z_dr_0p1_0p2)
if girth2 < 0.0079: z += 233 × (0.0079 − girth2)
if girth2_top20 < 0.011: z += -93.64 × (0.011 − girth2_top20)
if mass_top40 < 91.29: z += 0.026 × (91.29 − mass_top40)
if log_sum_pt > 6.94: z += 12.32 × (log_sum_pt − 6.94)
if n_dr_0p2_0p4 < 21.00 and n_dr_0p1_0p2 < 21.00: z += -0.003 × (21.00 − n_dr_0p2_0p4) × (21.00 − n_dr_0p1_0p2)
if girth2_top20 < 0.0057: z += 206 × (0.0057 − girth2_top20)
if mass_top40 < 80.89: z += -0.025 × (80.89 − mass_top40)
if mass_top50 < 117: z += 0.0083 × (117 − mass_top50)
if girth2_top30 < 0.0061: z += 170 × (0.0061 − girth2_top30)
if girth2_top10 < 0.0069: z += 89.98 × (0.0069 − girth2_top10)
if mass_top40 < 67.73: z += 0.033 × (67.73 − mass_top40)
if girth2_top20 < 0.017 and z_top50_slots > 0.970: z += -888 × (0.017 − girth2_top20) × (z_top50_slots − 0.970)
if mass_top10 < 45.59: z += -0.015 × (45.59 − mass_top10)
if lam2 < 0.0012 and dr_1 < 0.061: z += -13195 × (0.0012 − lam2) × (0.061 − dr_1)
if lam1 < 0.012: z += -29.58 × (0.012 − lam1)
if lam1 < 0.0062 and z_top50_slots > 0.985: z += 5904 × (0.0062 − lam1) × (z_top50_slots − 0.985)
if max_dr < 0.332: z += -4.11 × (0.332 − max_dr)
if max_dr > 0.240 and mass_top10 < 56.92: z += 0.037 × (max_dr − 0.240) × (56.92 − mass_top10)
if sum_pt_top50 > 976 and eccentricity > 0.949: z += -0.291 × (sum_pt_top50 − 976) × (eccentricity − 0.949)
if log_sum_pt > 6.94 and eccentricity > 0.949: z += 422 × (log_sum_pt − 6.94) × (eccentricity − 0.949)
if max_dr > 0.402: z += 3.49 × (max_dr − 0.402)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 22.2% of jets, neuron 0.12.  
- **very heavy (150 GeV), very wide, pT spread over several particles** — 21.7% of jets, neuron 0.19.  
- **heavy (91 GeV), average width, pT spread over several particles** — 21.3% of jets, neuron 1.06.  
- **light (38 GeV), very narrow, pT spread over several particles** — 7.5% of jets, neuron 0.00.  
- **medium-mass (52 GeV), very narrow, pT spread over several particles** — 6.8% of jets, neuron 0.00.  
- **light (24 GeV), very narrow, pT spread over several particles** — 6.7% of jets, neuron 0.00.  
- **medium-mass (66 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.00.  
- **medium-mass (87 GeV), narrow, pT spread over several particles, high pT** — 3.1% of jets, neuron 0.56.  
- **very heavy (161 GeV), wide, pT spread over several particles, high pT** — 2.8% of jets, neuron 0.79.  
- **medium-mass (55 GeV), very narrow, pT spread over several particles, high pT** — 1.9% of jets, neuron 0.00.  

### neuron 15: Dense core near the axis (minor)

- **What it measures:** Rises when the pT is packed close to the axis (large share within ΔR < 0.05, small LHA, girth, e2 and spread of the 5 hardest), with the strongest pushes for a girth below 0.1207 and e2 below 0.0408. Quark jets sit highest (0.67), then gluon (0.49), W (0.34) and Z (0.28) jets, and top jets lowest (0.08; AUC 0.24, small for t).
- *computed — its value:* largest for q (0.67), then g (0.49), then W (0.34), then Z (0.28), then t (0.08); it separates t jets from the rest best (AUC 0.24: small for t)
- **How the class scores use it:** It raises the Z score slightly (+4%) and lowers the t score slightly (-3%); the g, q and W scores hardly use it.
- *computed — used by:* raises the score of Z (+4%); lowers the score of t (-3%); does not (or hardly) enter the score of g, q, W (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.634
if girth < 0.121: z += 19.10 × (0.121 − girth)
if e2_sq < 0.0069 and z_dr_0p2_0p4 < 0.129: z += 4577 × (0.0069 − e2_sq) × (0.129 − z_dr_0p2_0p4)
if e2_sq < 0.0069: z += -520 × (0.0069 − e2_sq)
if girth < 0.121 and z_dr_0p2_0p4 < 0.129: z += -136 × (0.121 − girth) × (0.129 − z_dr_0p2_0p4)
if e2 < 0.041: z += 54.10 × (0.041 − e2)
if lam1 < 0.016: z += -62.85 × (0.016 − lam1)
if girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194: z += 7380 × (0.0015 − girth2_top5) × (0.194 − z_dr_0p2_0p4)
if girth2_top5 < 0.0015: z += -1080 × (0.0015 − girth2_top5)
if lam2 < 0.0018: z += 397 × (0.0018 − lam2)
if log_sum_pt < 7.02: z += 3.65 × (7.02 − log_sum_pt)
if girth < 0.057: z += -29.16 × (0.057 − girth)
if sum_pt < 1002: z += -0.020 × (1002 − sum_pt)
if lam1 < 0.0062: z += 163 × (0.0062 − lam1)
if sum_pt < 986: z += 0.020 × (986 − sum_pt)
if z_dr_0p1_0p2 < 0.187: z += 2.69 × (0.187 − z_dr_0p1_0p2)
if z_dr_0p1_0p2 < 0.120 and mass_top5 < 40.20: z += 0.168 × (0.120 − z_dr_0p1_0p2) × (40.20 − mass_top5)
if z_dr_0p1_0p2 < 0.120 and z_dr_0p2_0p4 < 0.052: z += -132 × (0.120 − z_dr_0p1_0p2) × (0.052 − z_dr_0p2_0p4)
if n_dr_0p1_0p2 < 13.00: z += 0.037 × (13.00 − n_dr_0p1_0p2)
if mass_top50 < 71.80: z += -0.017 × (71.80 − mass_top50)
if girth2_top50 < 0.014 and z_dr_0p2_0p4 < 0.129: z += 190 × (0.014 − girth2_top50) × (0.129 − z_dr_0p2_0p4)
if girth2_top30 < 0.012 and z_dr_0p2_0p4 > 0.026: z += -3094 × (0.012 − girth2_top30) × (z_dr_0p2_0p4 − 0.026)
if e2_sq < 0.0069 and z_dr_0p1_0p2 < 0.219: z += -386 × (0.0069 − e2_sq) × (0.219 − z_dr_0p1_0p2)
if dr_0 < 0.052: z += -6.37 × (0.052 − dr_0)
if mass_top30 < 62.55: z += -0.012 × (62.55 − mass_top30)
if sum_pt_top30 < 1038: z += -0.0013 × (1038 − sum_pt_top30)
if z_dr_0_0p05 < 0.846 and z_dr_0p2_0p4 < 0.0034: z += -323 × (0.846 − z_dr_0_0p05) × (0.0034 − z_dr_0p2_0p4)
if mass_top30 < 80.40: z += 0.0054 × (80.40 − mass_top30)
if girth2_top5 < 0.0015 and n_dr_0p2_0p4 > 0: z += -20.05 × (0.0015 − girth2_top5) × (n_dr_0p2_0p4 − 0)
if girth2_top5 < 0.0072 and sum_pt_top40 < 985: z += -0.666 × (0.0072 − girth2_top5) × (985 − sum_pt_top40)
if sum_pt_top50 < 934: z += -0.0029 × (934 − sum_pt_top50)
if sum_pt < 1002 and z_dr_0p2_0p4 < 0.020: z += -0.320 × (1002 − sum_pt) × (0.020 − z_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (87 GeV), average width, pT spread over several particles** — 30.4% of jets, neuron 0.12.  
- **very heavy (159 GeV), very wide, pT spread over several particles** — 18.3% of jets, neuron 0.10.  
- **light (32 GeV), very narrow, pT spread over several particles** — 13.8% of jets, neuron 0.83.  
- **medium-mass (56 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 0.77.  
- **heavy (95 GeV), average width, pT spread over several particles** — 8.3% of jets, neuron 0.60.  
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 6.6% of jets, neuron 0.62.  
- **medium-mass (74 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.57.  
- **heavy (117 GeV), very wide, pT spread over several particles** — 5.0% of jets, neuron 0.08.  
- **light (40 GeV), very narrow, pT spread over several particles, low pT** — 1.3% of jets, neuron 0.04.  
- **medium-mass (70 GeV), wide, pT spread over several particles, low pT** — 0.9% of jets, neuron 0.04.  
