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

- **medium-mass (85 GeV), average width, pT spread over several particles** — 26.5% of jets, neuron 1.71. A W/Z mixture (46% W, 42% Z); mass about 85 GeV and width 0.0071, with a harder leading particle than average (275 GeV against 240) and most pT at 0.025-0.1 from the axis (only 12% inside 0.025). With the intercept 1.3159, 'z_top30_slots > 0.934 and girth2_top5 < 0.0169' (+1.084), 'LHA < 0.404' (+0.95), 'log_sum_pt > 6.89' (+0.788) and 'sum_pt > 950' (+0.769) push up, while 'z_top50_slots > 0.959' (-1.663), 'n_pt_above_10 < 31' (-1.276), 'log_sum_pt > 6.81' (-1.098) and 'sum_pt_top50 > 959' (-0.905) pull down, all passing for nearly all of them: the pT is carried by few particles. The value, 1.708, is the lowest of this neuron and adds 1.068 to the g score and removes 0.32 from q. The formula calls them W (50%), with 42% called Z.
- **heavy (120 GeV), wide, pT spread over several particles** — 19.9% of jets, neuron 2.36. Mostly top (55%), mixed with Z (17%), W (13%), quark (8%) and gluon (7%); mass about 120 GeV and width 0.0163, soft leading particle (171 GeV) and pT spread over 0.05-0.2 from the axis. 'sum_pt_top2 < 689' (+1.099), 'sum_pt_top30 < 1.07e+03' (+0.842) and 'n_particles > 26' (+0.597) pass for almost all and lift it, against 'z_top50_slots > 0.959' (-1.383), 'n_pt_above_10 < 31' (-0.806) and 'log_sum_pt > 6.81' (91%, -0.782). The narrow-core tests such as 'girth2_top10 < 0.00768' pass for only 45%. The value 2.359 adds 1.474 to g and removes 0.442 from q.
- **light (34 GeV), very narrow, pT spread over several particles** — 12.7% of jets, neuron 2.10. Mostly quark (75%); light (about 34 GeV) and very narrow (width 0.0013), with a very hard leading particle (341 GeV against 240) and 90% of the pT within 0.025 of the axis. Large terms cancel: 'LHA < 0.404' (+2.161), 'mass_top30 < 80.4' (+1.891) and 'z_top30_slots > 0.934 and girth2_top5 < 0.0169' (+1.679) against 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.884), 'n_pt_above_10 < 31' (-1.685), 'z_top50_slots > 0.959' (-1.666) and 'mass < 121' (-1.591). 'n_pt_above_10 < 31 and lam1 < 0.00467' passes for all (+1.213), which marks them out. The value 2.103 adds 1.315 to g and removes 0.394 from q. The formula calls them q.
- **heavy (106 GeV), average width, pT spread over several particles** — 11.1% of jets, neuron 4.47. A four-way mixture (24% gluon, 26% W, 26% Z, 18% top); mass about 106 GeV and width 0.0105, total pT 1097 GeV (above the 1044 average), with ordinary pT sharing. The total-pT tests decide it: 'log_sum_pt > 6.91' (+2.617), 'log_sum_pt > 6.89' (+2.28) and 'sum_pt > 950' (+1.57) against 'sum_pt_top50 > 959' (-1.837), 'log_sum_pt > 6.81' (-1.763) and 'z_top50_slots > 0.959' (-1.323); 'log_sum_pt > 6.96' passes for almost all here but for only 21% elsewhere. The value 4.471 adds 2.794 to g and removes 0.838 from q. The formula's decisions are spread across g, W, Z and t much like the true mix.
- **heavy (139 GeV), very wide, pT spread over several particles** — 8.2% of jets, neuron 4.63. A top/gluon mixture (56% top, 30% gluon); mass about 139 GeV and width 0.0221, with the softest leading particle of all groups (112 GeV against 240) and pT spread wide, 15% at 0.15-0.2 from the axis. 'n_particles > 38 and z_top50_slots < 0.985' passes for all (only 13% elsewhere) and adds 1.318, with 'sum_pt_top30 < 1.07e+03' (+1.44), 'sum_pt_top2 < 689' (+1.337) and 'n_particles > 26' (+0.778), against 'z_top40_slots < 0.967' (-1.03) and 'z_top20_slots < 0.958' (-0.986). The value 4.627 adds 2.892 to g and removes 0.868 from q, a gluon-like push for jets that are mostly top.
- **medium-mass (57 GeV), narrow, pT spread over several particles** — 6.9% of jets, neuron 4.62. Mostly gluon (47%) mixed with 36% quark; mass about 57 GeV and width 0.0037, leading particle softer than average (177 GeV) and 55% of the pT within 0.025 of the axis. 'LHA < 0.404' (+1.574), 'mass_top30 < 80.4' (+1.509), 'sum_pt_top2 < 689' (+1.072) and 'girth2_top15 < 0.00327 and n_real_top40 > 29' (98% pass against 21% elsewhere, +0.916) lift it, against 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.437), 'z_top50_slots > 0.959' (-1.393) and 'mass < 121' (-1.163). The value 4.622 adds 2.889 to g and removes 0.867 from q. The formula calls them g (58%), with 36% called q.
- **heavy (97 GeV), average width, pT spread over several particles, high pT** — 6.0% of jets, neuron 7.02. Mostly gluon (61%) with about 10% each of W, Z, top and quark; mass about 97 GeV and width 0.0077, total pT 1221 GeV, well above the 1044 average. The total-pT tests grow with pT here: 'log_sum_pt > 6.91' (+5.732), 'log_sum_pt > 6.89' (+4.577) and 'sum_pt > 950' (+2.895) outweigh 'sum_pt_top50 > 959' (-3.543), 'log_sum_pt > 6.81' (-2.762) and 'log_sum_pt > 6.96' (-2.035); 'log_sum_pt > 7.06' passes for 94% (5% elsewhere). The value 7.024 adds 4.39 to g and removes 1.317 from q. The formula calls them g.
- **light (45 GeV), very narrow, pT spread over several particles** — 4.7% of jets, neuron 5.38. A gluon/quark mixture (50% gluon, 40% quark); light (about 45 GeV) and narrow (width 0.0018), total pT 1112 GeV, hard leading particle (318 GeV) and 83% of the pT within 0.025 of the axis. The total-pT tests ('log_sum_pt > 6.91' +3.013, 'log_sum_pt > 6.89' +2.572) and the light-narrow tests ('LHA < 0.404' +2.007, 'mass_top30 < 80.4' +1.722) push up, against 'sum_pt_top50 > 959' (-2.169), 'log_sum_pt > 6.81' (-1.89) and 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.689). The value 5.38 adds 3.363 to g and removes 1.009 from q. The formula calls them g (55%), with 43% called q.
- **heavy (100 GeV), average width, pT spread over several particles, high pT** — 3.0% of jets, neuron 8.12. Mostly gluon (76%); mass about 100 GeV and width 0.0067, total pT 1383 GeV and a hard leading particle (289 GeV). 'log_sum_pt > 6.91' (+9.336), 'log_sum_pt > 6.89' (+7.232) and 'sum_pt > 950' (+4.62) far outweigh 'sum_pt_top50 > 959' (-5.803), 'log_sum_pt > 6.81' (-3.917) and 'log_sum_pt > 6.96' (-3.739); 'log_sum_pt > 7.14' passes for all (2% elsewhere). The value 8.117 is well below the largest value (16) and adds 5.073 to g and removes 1.522 from q. The formula calls them g.
- **heavy (112 GeV), narrow, pT spread over several particles, high pT** — 1.0% of jets, neuron 9.03. Mostly gluon (82%); mass about 112 GeV and width 0.0059, the highest total pT (1681 GeV) and hardest leading particle (362 GeV) of this neuron. 'log_sum_pt > 6.91' (+14.922) and 'log_sum_pt > 6.89' (+11.349) outweigh 'sum_pt_top50 > 959' (-9.989), 'log_sum_pt > 6.96' (-6.382) and the other upper pT steps. The value 9.033, the highest of this neuron but still below 16, adds 5.646 to g and removes 1.694 from q. The formula calls them g.

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

- **heavy (96 GeV), average width, pT spread over several particles** — 25.3% of jets, neuron 1.41. Mostly Z (64%) with 14% gluon and 12% top; mass about 96 GeV and width 0.0084, ordinary pT sharing, and 43% of the pT at 0.05-0.1 from the axis (6% inside 0.025), the two-prong spacing. With the intercept 2.0597, 'D2 < 6.92' (+0.766), 'girth2_top15 < 0.0156' (+0.583), 'girth2_top40 < 0.0129' (+0.578) and 'n_dr_0p2_0p4 < 26' (+0.528) push up against 'LHA > 0.115' (-1.6), 'lam1 < 0.0205' (-0.653) and 'girth < 0.121' (-0.567). What sets them apart is that the low-mass penalties such as 'mass < 86.4' pass for only 6% (70% elsewhere), so the value, 1.411, is the highest of this neuron; it adds 0.485 to W and 0.265 to t and removes 1.235 from g and 1.499 from q. The formula calls them Z.
- **medium-mass (81 GeV), narrow, pT spread over several particles** — 25.1% of jets, neuron 1.12. Mostly W (68%) with 11% gluon and 9% Z; mass about 81 GeV and width 0.006, ordinary pT sharing, with the pT spread over 0.025-0.1 from the axis. The same shape terms as the Z-mass group ('mass < 101' +0.997, 'girth2_top40 < 0.0129' +0.851, 'D2 < 6.92' +0.709 against 'LHA > 0.115' -1.284, 'girth < 0.121' -0.762, 'lam1 < 0.0205' -0.762), but the mass is below the Z peak, so 'mass < 91.2' (97%, -0.393), 'mass_top40 < 83.3 and D2 < 6.92' (91%, -0.34) and 'mass < 86.4' (91%, -0.293) take some off. The value 1.123 adds 0.386 to W and 0.211 to t and removes 0.983 from g and 1.193 from q. The formula calls them W.
- **very heavy (168 GeV), very wide, pT spread over several particles** — 10.6% of jets, neuron 0.76. Mostly top (88%); mass about 168 GeV and width 0.0279, soft leading particle (157 GeV) and pT spread out, 27% at 0.1-0.15 from the axis against 12% for all jets. Large width terms nearly cancel: 'width > 0.00961' (+3.629), 'girth2_top30 > 0.000856' (+1.735), 'girth2_top50 > 0.00812' (+1.662) and 'mass > 144' (97%, +0.929) against 'LHA > 0.115' (-2.676), 'girth2_top30 > 0.00636' (-1.89), 'lam1 > 0.00767' (-1.531), 'girth2_top50 > 0.00926' (-1.399) and 'mass_over_sum_pt > 0.0603' (-1.31). The value 0.763 adds 0.262 to W and 0.143 to t and removes 0.667 from g and 0.81 from q. The formula calls them t.
- **heavy (137 GeV), wide, pT spread over several particles** — 7.8% of jets, neuron 0.37. A top/gluon mixture (61% top, 27% gluon); mass about 137 GeV and width 0.0181, soft leading particle (175 GeV), with 38% of the pT at 0.05-0.1 and 25% at 0.1-0.15 from the axis. 'LHA > 0.115' (-2.198), 'girth2_top30 > 0.00636' (-0.929), 'mass_over_sum_pt > 0.0603' (-0.908) and 'mass_over_sum_pt > 0.098' (all pass, -0.419) outweigh 'width > 0.00961' (+1.685), 'girth2_top30 > 0.000856' (+1.044) and 'D2 < 6.92' (+0.694); 'mass < 101' almost never passes. The value 0.37 (on for 74%) adds 0.127 to W and removes 0.393 from q. The formula calls them t (71%).
- **medium-mass (65 GeV), narrow, pT spread over several particles** — 7.6% of jets, neuron 0.01. Mostly gluon (45%) mixed with 32% quark and 12% W; mass about 65 GeV and width 0.0039, a hard leading particle (270 GeV) and 64% of the pT within 0.025 of the axis. 'mass < 101' (+1.745), 'girth2_top40 < 0.0129' (+1.129) and 'girth2_top15 < 0.0156' (+0.919) are outweighed by 'girth < 0.121' (-1.085), 'mass < 86.4' (-1.016), 'mass < 91.2' (-0.956), 'mass_top40 < 94.6' (-0.898) and 'lam1 < 0.0205' (-0.879), with 'mass < 78.3' also passing for 97%. The value is 0.011 and the neuron is on for only 5%, so it hardly touches the scores. The formula calls them g (53%), with 37% called q.
- **medium-mass (52 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 0.00. Mostly gluon (57%) mixed with 34% quark; mass about 52 GeV and width 0.0026, a softer leading particle (215 GeV) and 60% of the pT within 0.025 of the axis. 'mass_top40 < 62.5' passes for all (23% elsewhere) and removes 2.55, and with 'mass_top40 < 83.3 and D2 < 6.92' (-2.055), 'mass < 86.4' (-1.649) and 'mass < 91.2' (-1.441) it outweighs 'mass < 101' (+2.385), 'girth2_top40 < 0.0129' (+1.289) and 'mass_top40 < 83.3' (+1.204). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (61%).
- **light (24 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 0.00. Mostly quark (83%); the lightest (about 24 GeV) and narrowest (width 0.0006) jets here, with a very hard leading particle (397 GeV against 240) and 95% of the pT within 0.025 of the axis. 'mass_top40 < 62.5' removes 5.81 and 'mass < 86.4' (-3.01) and 'mass < 91.2' (-2.484) add to it, far more than 'mass < 101' (+3.761) and 'mass_top40 < 83.3' (+1.891) return; 'LHA > 0.115' passes for only 3%. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q.
- **light (34 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 0.00. Mostly quark (55%) mixed with 37% gluon; light (about 34 GeV) and narrow (width 0.0012), a leading particle of 231 GeV (near average) and 80% of the pT within 0.025 of the axis. 'mass_top40 < 62.5' (-4.769), 'mass_top40 < 83.3 and D2 < 6.92' (-2.876) and 'mass < 86.4' (-2.523) outweigh 'mass < 101' (+3.269) and the narrowness bonuses. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q (62%), with 38% called g.
- **light (42 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 0.00. Mostly quark (59%) with 26% gluon; light (about 42 GeV) and narrow (width 0.0017), a hard leading particle (348 GeV) and 91% of the pT within 0.025 of the axis. 'mass_top40 < 62.5' (-3.353), 'mass < 86.4' (-2.176) and 'mass < 91.2' (-1.845) outweigh 'mass < 101' (+2.919), 'mass_top40 < 83.3' (+1.373) and 'girth2_top40 < 0.0129' (+1.345). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q (76%).
- **very heavy (197 GeV), very wide, pT spread over several particles** — 1.8% of jets, neuron 0.75. A top/gluon mixture (62% top, 29% gluon); very heavy (about 197 GeV) and very wide (width 0.038), soft leading particle (143 GeV) and a quarter of the pT at 0.15-0.2 from the axis. The width terms are larger than for the top-mass group but still nearly cancel: 'width > 0.00961' (+5.65), 'girth2_top50 > 0.00812' (+2.521), 'girth2_top30 > 0.000856' (+2.412) and 'mass > 144' (+2.066) against 'LHA > 0.115' (-3.1), 'girth2_top30 > 0.00636' (-2.83), 'lam1 > 0.00767' (-2.381) and 'girth2_top50 > 0.00926' (-2.168); 'LHA > 0.404' passes for 97% (8% elsewhere). The value 0.747 adds 0.257 to W and 0.14 to t and removes 0.654 from g and 0.794 from q. The formula calls them t (76%).

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

- **medium-mass (72 GeV), narrow, pT spread over several particles** — 67.3% of jets, neuron 1.94. A broad mixture of two-thirds of all jets (27% W, 27% Z, 24% quark, 13% gluon, 8% top); mass about 72 GeV and width 0.0057, ordinary pT sharing and 32% of the pT within 0.025 of the axis. With the intercept 0.7007, 'mass_over_sum_pt_sq < 0.0258' (+1.128), 'z_top50_slots > 0.959' (98%, +0.946), 'log_sum_pt > 6.86' (91%, +0.874) and 'n_particles < 64 and e2 < 0.0388' (71%, +0.615) push up, against 'z_top40_slots > 0.93' (96%, -1.117), 'max_dr > 0.24' (87%, -0.67), 'log_sum_pt > 6.91' (70%, -0.519) and 'n_real_top50 > 22' (93%, -0.518). The value 1.944 (on for 96%) adds 1.124 to W and 1.337 to Z and removes 0.608 from g and 0.911 from t; the formula's decisions follow the mixed true content.
- **very heavy (155 GeV), very wide, pT spread over several particles** — 14.9% of jets, neuron 0.29. Mostly top (81%); mass about 155 GeV and width 0.0243, soft leading particle (163 GeV) and pT spread out over 0.05-0.2 from the axis. 'mass_over_sum_pt > 0.08' (+2.27), 'mass_top30 > 68.3' (+1.974) and 'mass > 64.5' (+1.277) are outweighed by 'mass > 89.7' (-2.245), 'mass_over_sum_pt > 0.0785' (-1.993), 'n_real_top50 > 22' (-0.821), 'max_dr > 0.24' (-0.812), 'mass_over_sum_pt > 0.0905' (-0.784) and 'mass_top30 > 102' (93%, -0.632). The value is 0.291 and the neuron is on for only 34%, so it adds little (0.2 to Z, -0.136 on t). The formula calls them t.
- **medium-mass (81 GeV), narrow, pT spread over several particles** — 11.6% of jets, neuron 1.14. Mostly gluon (55%) mixed with 15% quark, 14% W and 12% Z; mass about 81 GeV and width 0.0054, total pT 1187 GeV (above the 1044 average) and 43% of the pT within 0.025 of the axis. Here the total-pT steps fight each other: 'log_sum_pt > 6.91' (-3.738), 'log_sum_pt > 6.89 and z_dr_0p1_0p2 < 0.187' (86%, -1.383), 'log_sum_pt > 6.89' (-1.223) and 'log_sum_pt > 6.92 and girth2_top2 < 0.014' (-1.214) against 'log_sum_pt > 6.86' (+2.837), 'log_sum_pt > 6.96' (+1.161) and 'log_sum_pt > 6.94 and z_dr_0p1_0p2 < 0.187' (+1.13); 'log_sum_pt > 6.99' passes for all (9% elsewhere). The value 1.137 adds 0.657 to W and 0.782 to Z and removes 0.355 from g. The formula calls them g (62%).
- **medium-mass (85 GeV), narrow, pT spread over several particles, high pT** — 2.9% of jets, neuron 1.08. Mostly gluon (77%); mass about 85 GeV, narrow (width 0.004), total pT 1477 GeV, hard leading particle (333 GeV) and 55% of the pT within 0.025 of the axis. The same total-pT steps with larger amounts: 'log_sum_pt > 6.91' (-8.515), 'log_sum_pt > 6.89 and z_dr_0p1_0p2 < 0.187' (-3.704), 'log_sum_pt > 6.92 and girth2_top2 < 0.014' (-3.102) and 'sum_pt_top50 > 1.06e+03' (-2.688) against 'log_sum_pt > 6.86' (+5.583), 'log_sum_pt > 6.94 and z_dr_0p1_0p2 < 0.187' (+3.54), 'log_sum_pt > 6.96' (+3.26) and 'log_sum_pt > 6.99' (+2.737). They nearly cancel, leaving 1.077 (on for 84%), which adds 0.623 to W and 0.741 to Z and removes 0.505 from t. The formula calls them g.
- **very heavy (174 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 0.00. Mostly top (85%); mass about 174 GeV and width 0.033, soft leading particle (142 GeV) and a quarter of the pT at 0.1-0.15 from the axis. 'mass_over_sum_pt_sq > 0.0292' passes for all (3% elsewhere) and removes 6.461, with 'mass_over_sum_pt > 0.171' (-3.788), 'mass_over_sum_pt_sq > 0.0292 and phi_6 > -0.117' (79%, -3.01), 'mass > 89.7' (-2.913) and 'mass > 173' (64%, -2.53), against 'mass_over_sum_pt > 0.08' (+3.078) and 'mass_top30 > 68.3' (+2.455). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them t.
- **very heavy (187 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.00. Mostly top (67%) with 23% gluon; mass about 187 GeV and width 0.0393, a lower total pT (947 GeV) and a leading particle of only 129 GeV, with the pT far from the axis. 'mass_over_sum_pt_sq > 0.0292' (-17.113), 'mass_over_sum_pt_sq > 0.0292 and phi_6 > -0.117' (85%, -10.007), 'mass_over_sum_pt > 0.171' (-9.58) and 'mass > 173' (86%, -8.434) drive the sum far below zero. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them t (86%).
- **very heavy (206 GeV), very wide, pT spread over several particles, high pT** — 0.6% of jets, neuron 0.00. Mostly gluon (78%) with 13% top; very heavy (about 206 GeV), width 0.0265 and a high total pT (1299 GeV), with the pT spread over 0.05-0.2 from the axis. 'mass > 173' passes for all (4% elsewhere) and removes 17.515, with 'mass > 163' (-6.625), 'log_sum_pt > 6.91' (-5.556) and 'mass > 89.7' (-4.003); the m/pT cut 'mass_over_sum_pt_sq > 0.0292' passes for only 40% because the pT is high. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (88%).
- **very heavy (225 GeV), very wide, pT spread over several particles** — 0.2% of jets, neuron 0.00. A small, even gluon/top mixture (45% each, 8% quark); very heavy (about 225 GeV) and very wide (width 0.0435), with 27% of the pT at 0.15-0.2 from the axis. 'mass > 173' (-27.8), 'mass_over_sum_pt_sq > 0.0292' (-24.333), 'mass_over_sum_pt > 0.171' (-13.211) and 'mass > 163' (-9.617) all pass. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them t (57%), with 42% called g.
- **very heavy (271 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. Mostly gluon (89%); the heaviest jets of this neuron (about 271 GeV), width 0.0312, total pT 1554 GeV, with almost no pT near the axis. 'mass > 173' removes 51.579, with 'mass > 163' (-16.534), 'log_sum_pt > 6.91' (-9.61) and 'mass_over_sum_pt_sq > 0.0292' (63%, -7.03). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g.
- **very heavy (218 GeV), very wide, pT spread over several particles** — 0.1% of jets, neuron 0.00. Mostly top (60%) with 28% gluon; mass about 218 GeV and the largest width here (0.0482), total pT 998 GeV, a soft leading particle (119 GeV) and hardly any pT within 0.05 of the axis. 'mass_over_sum_pt_sq > 0.0292 and phi_6 > -0.117' (-36.567), 'mass_over_sum_pt_sq > 0.0292' (-32.2), 'mass > 173' (98.5%, -23.881) and 'mass_over_sum_pt > 0.171' (-17.033) give the deepest sum of this neuron. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them t (85%).

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

- **very heavy (141 GeV), very wide, pT spread over several particles** — 29.8% of jets, neuron 0.27. Mostly top (58%) mixed with 20% gluon and 13% Z; mass about 141 GeV and width 0.0201, soft leading particle (177 GeV) and pT spread over 0.05-0.2 from the axis. Too heavy for the mass window: 'mass < 92.9' and the narrowness tests pass for only about a tenth and 'mass < 121' for 34% (+0.281), so, with the intercept -0.3021, 'sum_pt < 1.26e+03' (95%, -1.182) outweighs 'sum_pt_top30 < 1.04e+03' (88%, +0.396). The value is 0.265, on for only 20%, adding 0.24 to Z and removing 0.166 from W. The formula calls them t (64%).
- **medium-mass (87 GeV), average width, pT spread over several particles** — 16.6% of jets, neuron 0.40. A Z/W/gluon mixture (34% Z, 30% W, 22% gluon); mass about 87 GeV and width 0.0067, total pT 1074 GeV and a hard leading particle (263 GeV), with 44% of the pT at 0.025-0.05 from the axis (22% for all jets). 'LHA < 0.32' (+1.802), 'mass < 121' (+1.371), 'e2_sq < 0.00961' (+1.179) and 'girth2_top40 < 0.00884' (+0.775) are balanced by 'girth < 0.0859' (-1.612), 'girth2_top20 < 0.00803' (-1.322), 'LHA < 0.333' (-1.102) and 'sum_pt < 1.26e+03' (-0.972); the D2 window tests that lift the Z group pass for at most 20% here. The value 0.403 (on for 47%) adds 0.365 to Z and removes 0.252 from W. The formula splits them between W (36%) and Z (35%).
- **medium-mass (88 GeV), average width, pT spread over several particles** — 12.5% of jets, neuron 1.97. Mostly Z (69%) with 22% W; mass about 88 GeV and width 0.0074, with 55% of the pT at 0.05-0.1 from the axis and almost none inside 0.025. 'mass < 101 and D2 < 1.6' (+2.28) and 'mass_top50 < 97.9 and D2 < 1.6' (-2.559) both pass for all (18% elsewhere) and nearly cancel; the value comes from 'mass < 121' (+1.307), 'e2_sq < 0.00961' (+0.889), 'mass < 101 and max_dr < 0.391' (83%, +0.826), 'D2 < 1.79 and n_dr_0p2_0p4 < 9' (96%, +0.669) and 'D2 < 1.79 and sum_pt_top50 < 1.25e+03' (+0.642) against 'sum_pt < 1.26e+03' (-1.122). The value 1.967, the highest of this neuron (on for 79%), adds 1.783 to Z and removes 1.229 from W and 0.553 from t. The formula calls them Z.
- **light (49 GeV), very narrow, pT spread over several particles** — 9.1% of jets, neuron 0.00. A gluon/quark mixture (47% gluon, 41% quark); light (about 49 GeV) and narrow (width 0.0022), with 76% of the pT within 0.025 of the axis. The mass steps below the window all pass: 'mass < 82.9' (-4.504), 'mass < 87.4' (-3.902), 'mass < 92.9' (-2.968) and 'mass_top50 < 71.8' (-2.956), plus 'girth < 0.0859' (-3.186), against 'LHA < 0.32' (+3.692), 'e2_sq < 0.00961' (+2.927) and 'mass < 121' (+2.904). The neuron is 0 for all of them and adds nothing to the scores. The formula splits them between g (52%) and q (48%).
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 8.0% of jets, neuron 0.20. Mostly W (86%); mass about 79 GeV and width 0.0058, with 61% of the pT at 0.05-0.1 from the axis and almost none inside 0.025. 'mass_top50 < 97.9 and D2 < 1.6' removes 6.335, which 'mass < 101 and D2 < 1.6' (+4.891), 'mass < 101 and max_dr < 0.391' (89%, +2.198), 'mass < 121' (+1.69) and 'e2_sq < 0.00961' (+1.491) more than repay, but being below the Z mass costs 'mass < 91.2 and max_dr < 0.391' (89%, -1.604) and 'sum_pt < 1.26e+03' (-1.103) takes more. The value is 0.203, on for 32%, adding 0.184 to Z and removing 0.127 from W. The formula calls them W.
- **light (30 GeV), very narrow, pT spread over several particles** — 7.8% of jets, neuron 0.00. Mostly quark (72%) with 16% gluon; light (about 30 GeV) and very narrow (width 0.0009), a hard leading particle (343 GeV) and 91% of the pT within 0.025 of the axis. 'mass < 82.9' (-6.874), 'mass < 87.4' (-5.716), 'mass_top50 < 71.8' (-4.971) and 'mass < 92.9' (-4.178) far outweigh 'LHA < 0.32' (+4.729) and 'mass < 121' (+3.632). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q.
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 7.4% of jets, neuron 0.00. Mostly gluon (52%) mixed with 26% quark and 12% W; mass about 67 GeV and width 0.004, with 61% of the pT within 0.025 of the axis. 'LHA < 0.32' (+2.912), 'e2_sq < 0.00961' (+2.233) and 'mass < 121' (+2.167) are outweighed by 'girth < 0.0859' (-2.57), 'girth2_top20 < 0.00803' (-2.446), 'mass < 82.9' (-2.107), 'mass < 87.4' (-2.064) and 'girth2_top40 < 0.00803' (-1.846). The value is 0.003, on for under 1%, so it adds essentially nothing. The formula calls them g (60%).
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 4.3% of jets, neuron 0.21. Mostly W (78%); mass about 78 GeV and width 0.0056, a softer leading particle (217 GeV) and 43% of the pT at 0.025-0.05 from the axis. 'mass < 101 and max_dr < 0.391' (+2.936), 'mass < 121' (+1.732), 'e2_sq < 0.00961' (+1.589) and 'LHA < 0.32' (+1.322) are cancelled by 'mass < 91.2 and max_dr < 0.391' (-2.247), 'girth < 0.0859' (-1.331), 'girth2_top20 < 0.00803' (-1.308) and 'sum_pt < 1.26e+03' (-1.063), with 'mass < 82.9' (96%) and 'mass < 87.4' taking a bit more. The value is 0.213, on for 37%, adding 0.193 to Z and removing 0.133 from W. The formula calls them W.
- **light (31 GeV), very narrow, pT spread over several particles** — 3.7% of jets, neuron 0.00. Mostly quark (72%) with 17% gluon; light (about 31 GeV) and very narrow (width 0.001), a hard leading particle (323 GeV) and 84% of the pT within 0.025 of the axis, all particles inside 0.391. Because the whole jet lies inside 0.391, the compact-jet terms are large and cancel: 'mass < 91.2 and max_dr < 0.391' (-7.061) against 'mass < 101 and max_dr < 0.391' (+6.16), and 'mass_top50 < 71.8 and max_dr < 0.394' (all, 20% elsewhere, +4.707) against 'mass_top50 < 71.8' (-4.972). The low-mass steps 'mass < 82.9' (-6.843) and 'mass < 87.4' (-5.693) then keep the neuron at 0, so it adds nothing to the scores. The formula calls them q.
- **very light (20 GeV), very narrow, pT spread over several particles** — 0.9% of jets, neuron 0.03. Mostly quark (86%); the lightest (about 20 GeV) and narrowest (width 0.0004) jets here, with a very hard leading particle (375 GeV) and 91% of the pT within 0.025 of the axis. The compact-jet terms are at their largest: 'mass < 91.2 and max_dr < 0.391' (-17.436) against 'mass < 101 and max_dr < 0.391' (+14.83) and 'mass_top50 < 71.8 and max_dr < 0.394' (+12.27), with 'mass < 82.9' (-8.248), 'mass < 87.4' (-6.769) and 'mass_top50 < 71.8' (-6.223) pulling down. The value is 0.025, on for 2.5%, so it hardly moves the scores. The formula calls them q.

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

- **medium-mass (86 GeV), average width, pT spread over several particles** — 46.6% of jets, neuron 0.42. A W/Z mixture (39% each, 10% gluon); mass about 86 GeV and width 0.0069, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. With the intercept -0.3625, 'mass > 64.5' (+0.851), 'girth < 0.0975' (+0.673), 'lam1 < 0.00824' (96%, +0.481) and 'z_dr_0p2_0p4 < 0.0912' (95%, +0.444) are balanced by 'mass_over_sum_pt_sq < 0.014' (-1.092), 'lam1 < 0.0205' (-0.664) and 'n_dr_0p2_0p4 < 21' (-0.595); 'mass > 101' passes for only 3%. The value 0.421 (on for 47%) removes 0.368 from W and 0.395 from Z and adds 0.089 to t. The formula splits them between W (43%) and Z (41%).
- **light (45 GeV), very narrow, pT spread over several particles** — 27.8% of jets, neuron 0.07. Mostly quark (51%) mixed with 34% gluon; light (about 45 GeV) and narrow (width 0.002), a hard leading particle (302 GeV) and 77% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.014' (-1.85), 'mass_over_sum_pt_sq < 0.00617' (-0.924) and 'lam1 < 0.0205' (-0.884) outweigh 'girth < 0.0975' (+1.678) and 'lam1 < 0.00824' (+1.577), and 'mass > 64.5' passes for only 15%. The value is 0.07, on for 14%, so it barely touches the scores. The formula calls them q (61%).
- **very heavy (171 GeV), very wide, pT spread over several particles** — 7.7% of jets, neuron 4.44. Mostly top (92%); mass about 171 GeV and width 0.0285, soft leading particle (160 GeV) and 27% of the pT at 0.1-0.15 and 20% at 0.15-0.2 from the axis. The rising mass steps 'mass > 64.5' (+4.241), 'mass > 80.8' (+4.084), 'mass > 87.4' (+3.42) and 'mass_top50 > 97.9' (+2.919), plus 'girth2_top20 > 0.00803' (+3.893), beat 'mass > 101' (-7.026), 'mass_over_sum_pt > 0.098' (-5.342), 'mass_top50 > 82' (-2.38) and 'mass > 125' (-1.813). The value 4.437 adds 0.936 to t and removes 3.882 from W and 4.16 from Z. The formula calls them t.
- **heavy (120 GeV), wide, pT spread over several particles** — 5.8% of jets, neuron 3.38. A top/gluon mixture (43% top, 40% gluon, 15% quark); mass about 120 GeV and width 0.0133, softer leading particle (189 GeV) and 39% of the pT at 0.05-0.1 from the axis. 'mass > 64.5' (+2.21), 'mass > 80.8' (+1.775), 'mass > 87.4' (+1.335), 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (76%, +1.004) and 'mass_over_sum_pt > 0.0887' (+0.752) outweigh 'mass > 101' (-1.905), 'mass_over_sum_pt > 0.098' (91%, -1.336) and 'mass_top50 > 82' (-0.941). The value 3.384 adds 0.714 to t and removes 2.961 from W and 3.173 from Z. The formula splits them between t (47%) and g (45%).
- **very heavy (143 GeV), very wide, pT spread over several particles** — 4.0% of jets, neuron 4.72. Mostly top (75%) with 14% gluon; mass about 143 GeV and width 0.0216, total pT 977 GeV (below average) and 34% of the pT at 0.05-0.1 and 27% at 0.1-0.15 from the axis. 'mass > 64.5' (+3.124), 'mass > 80.8' (+2.814), 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (+2.379), 'mass > 87.4' (+2.273) and 'girth2_top20 > 0.00803' (+2.202) outweigh 'mass > 101' (-4.21), 'mass_over_sum_pt > 0.098' (-3.684) and 'mass > 125' (98%, -0.708). The value 4.723 adds 0.996 to t and removes 4.133 from W and 4.428 from Z. The formula calls them t (90%).
- **very heavy (166 GeV), very wide, pT spread over several particles, high pT** — 2.5% of jets, neuron 2.46. A top/gluon mixture (50% top, 44% gluon); mass about 166 GeV and width 0.0191, total pT 1225 GeV (above average), with 37% of the pT at 0.05-0.1 from the axis. 'mass > 101' (-6.502), 'mass_over_sum_pt > 0.098' (98%, -2.987), 'mass_top50 > 82' (-2.174) and 'mass > 125' (-1.607) take back most of 'mass > 64.5' (+4.033), 'mass > 80.8' (+3.847), 'mass > 87.4' (+3.207) and 'mass_top50 > 97.9' (+2.609); the higher pT keeps m/pT and girth smaller than in the other top groups. The value 2.463 adds 0.519 to t and removes 2.155 from W and 2.309 from Z. The formula calls them t (55%), with 44% called g.
- **medium-mass (59 GeV), narrow, pT spread over several particles, low pT** — 2.3% of jets, neuron 3.19. Mostly gluon (43%) mixed with 32% quark and 23% top; mass about 59 GeV, width 0.0063, a low total pT (788 GeV) and soft leading particle (173 GeV). Here the low-pT tests decide it: 'log_sum_pt < 6.94' (+3.862) and 'sum_pt < 1.01e+03' (+2.067) with 'girth < 0.0975' (+1.042) and 'lam1 < 0.00824' (79%, +0.863), against 'mass_over_sum_pt_sq < 0.014' (-1.194), 'sum_pt_top50 < 1.03e+03' (-1.181), 'sum_pt_top30 < 1.01e+03' (-1.058) and 'sum_pt_top40 < 1e+03 and log_sum_pt < 6.81' (all, 3% elsewhere, -0.901). The value 3.187 adds 0.672 to t and removes 2.788 from W and 2.988 from Z. The formula calls them g (49%).
- **very heavy (169 GeV), very wide, pT spread over several particles** — 2.0% of jets, neuron 6.69. Mostly top (86%); mass about 169 GeV and width 0.0338, total pT 920 GeV, a soft leading particle (127 GeV) and nearly half the pT at 0.1-0.2 from the axis. 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (+5.033), 'girth2_top20 > 0.00803' (+5.017), 'mass > 64.5' (+4.14), 'mass > 80.8' (+3.969) and 'mass > 87.4' (+3.317) outweigh 'mass > 101' (-6.773), 'mass_over_sum_pt > 0.098' (-6.464), 'mass > 125' (-1.713) and 'mass > 87.4 and log_sum_pt < 6.9' (-1.344). The value 6.687, well below the largest value (16), adds 1.41 to t and removes 5.851 from W and 6.269 from Z. The formula calls them t.
- **very heavy (224 GeV), very wide, pT spread over several particles, high pT** — 0.9% of jets, neuron 3.27. Mostly gluon (60%) with 28% top; very heavy (about 224 GeV) and very wide (width 0.0358), total pT 1211 GeV and almost no pT within 0.05 of the axis. The biggest terms of this neuron nearly cancel: 'mass > 101' (-12.34), 'mass_over_sum_pt > 0.098' (-6.796) and 'mass > 125' (-3.901) against 'mass > 80.8' (+6.479), 'mass > 64.5' (+6.347), 'mass > 87.4' (+5.584), 'girth2_top20 > 0.00803' (+5.39) and 'mass_top50 > 97.9' (+4.942). The value 3.27 adds 0.69 to t and removes 2.861 from W and 3.065 from Z. The formula calls them g (64%).
- **heavy (110 GeV), very wide, pT spread over several particles, low pT** — 0.5% of jets, neuron 8.94. Mostly top (51%) with 35% gluon; mass about 110 GeV and width 0.0231, the lowest total pT here (730 GeV) and a soft leading particle (116 GeV). The low pT makes m/pT large: 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (+6.831), 'log_sum_pt < 6.94' (+4.945), 'sum_pt < 1.01e+03' (+2.593), 'girth2_top20 > 0.00803' (+2.467) and 'mass_over_sum_pt > 0.0887' (+1.762) outweigh 'mass_over_sum_pt > 0.098' (-3.983) and 'sum_pt_top40 < 1e+03 and log_sum_pt < 6.81' (-1.686). The value 8.942, the highest of this neuron but below 16, adds 1.886 to t and removes 7.824 from W and 8.383 from Z. The formula calls them t (61%).

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

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 26.6% of jets, neuron 0.27. Mostly W (68%) with 12% gluon; mass about 79 GeV and width 0.0059, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. With the intercept 0.5894, 'mass_top50 < 137' (+3.28) is taken back by 'mass_top50 < 161' (-2.68), 'girth2_top15 < 0.0215' (-1.138), 'mass > 62.5' (-0.952) and 'mass_top40 < 91.3' (-0.381), while 'mass > 82.9' passes for only 16%. The value is 0.268, on for 44%, adding 0.138 to g and 0.151 to q and removing 0.059 from W. The formula calls them W.
- **heavy (92 GeV), average width, pT spread over several particles** — 22.2% of jets, neuron 0.01. Mostly Z (75%) with 10% gluon; mass about 92 GeV and width 0.0079, ordinary pT sharing, with 41% of the pT at 0.05-0.1 from the axis. 'mass_top50 < 137' (+2.566) is cancelled by 'mass_top50 < 161' (-2.267), and 'mass > 62.5' (-1.671) and 'girth2_top15 < 0.0215' (-1.011) outweigh 'mass > 82.9' (all pass, +0.59). The value is 0.012, on for 4%, so it adds almost nothing. The formula calls them Z.
- **light (32 GeV), very narrow, pT spread over several particles** — 13.3% of jets, neuron 4.49. Mostly quark (69%) with 19% gluon; light (about 32 GeV) and very narrow (width 0.001), a hard leading particle (351 GeV) and 92% of the pT within 0.025 of the axis. 'mass_top50 < 137' (+5.917), 'LHA < 0.209' (all, 19% elsewhere, +2.439), 'mass_top40 < 80.9' (+1.427) and 'mass_top30 < 76.4' (+0.871) outweigh 'mass_top50 < 161' (-4.208), 'girth < 0.0436' (-2.272), 'mass_top40 < 91.3' (-1.567) and 'girth2_top15 < 0.0215' (-1.418), and 'mass > 62.5' never passes. The value 4.488 adds 2.314 to g and 2.524 to q and removes 0.982 from W. The formula calls them q.
- **medium-mass (55 GeV), very narrow, pT spread over several particles** — 12.3% of jets, neuron 2.95. Mostly gluon (49%) mixed with 38% quark; mass about 55 GeV and width 0.0027, ordinary pT sharing and 67% of the pT within 0.025 of the axis. The same pattern as the light group but weaker: 'mass_top50 < 137' (+4.708), 'LHA < 0.209' (87%, +0.932) and 'mass_top40 < 80.9' (+0.898) against 'mass_top50 < 161' (-3.507), 'girth2_top15 < 0.0215' (-1.373), 'mass_top40 < 91.3' (-1.086) and 'girth < 0.0436' (92%, -1.013). The value 2.954 adds 1.523 to g and 1.662 to q and removes 0.646 from W. The formula calls them g (54%), with 44% called q.
- **very heavy (172 GeV), very wide, pT spread over several particles** — 10.1% of jets, neuron 0.17. Mostly top (86%); mass about 172 GeV and width 0.0282, soft leading particle (162 GeV) and pT spread over 0.05-0.2 from the axis. The jet-mass steps nearly cancel: 'mass > 62.5' (-6.195) and 'mass > 144' (all, 6% elsewhere, -2.888) against 'mass > 82.9' (+5.698), 'mass > 92.9' (+2.785) and 'girth > 0.0975' (98%, +0.669), while 'mass_top50 < 137' never passes. The value is 0.169, on for 42%, so it adds little (0.095 to q). The formula calls them t.
- **very heavy (148 GeV), very wide, pT spread over several particles** — 6.0% of jets, neuron 0.66. Mostly top (67%) with 23% gluon; mass about 148 GeV and width 0.0216, softer leading particle (173 GeV) and pT spread over 0.05-0.15 from the axis. 'mass > 62.5' (-4.82) is mostly repaid by 'mass > 82.9' (+4.145) and 'mass > 92.9' (+1.93), and 'mass > 144' passes for 66% (-0.592). The value 0.661 (on for 80%) adds 0.341 to g and 0.372 to q and removes 0.145 from W. The formula calls them t (78%).
- **heavy (119 GeV), wide, pT spread over several particles** — 6.0% of jets, neuron 0.20. A top/gluon mixture (44% top, 39% gluon, 15% quark); mass about 119 GeV and width 0.0135, softer leading particle (189 GeV) and 38% of the pT at 0.05-0.1 from the axis. 'mass > 62.5' (-3.186), 'mass_top50 < 161' (-1.504) and 'girth2_top15 < 0.0215' (-0.795) roughly cancel 'mass > 82.9' (+2.3), 'mass_top50 < 137' (+1.249) and 'mass > 92.9' (+0.915); 'sum_pt < 950' passes for 20% (+0.276). The value is 0.198, on for 34%, so it adds little. The formula splits them between t (49%) and g (43%).
- **medium-mass (72 GeV), average width, pT spread over several particles, low pT** — 1.9% of jets, neuron 3.01. A three-way mixture (40% gluon, 32% top, 27% quark); mass about 72 GeV and width 0.0096, a low total pT (804 GeV) and soft leading particle (169 GeV). 'sum_pt < 950' passes for all (8% elsewhere) and adds 4.888, with 'mass_top50 < 137' (99%, +3.765) and 'sum_pt_top40 < 956' (+0.768), against 'log_sum_pt < 6.86' (-3.743), 'mass_top50 < 161' (-2.96), 'girth2_top15 < 0.0215' (95%, -1.033) and 'mass > 62.5' (60%, -0.98). The value 3.008 adds 1.551 to g and 1.692 to q and removes 0.658 from W. The formula calls them g (45%), with about a third called t.
- **very heavy (222 GeV), very wide, pT spread over several particles, high pT** — 1.0% of jets, neuron 0.04. Mostly gluon (59%) with 30% top; very heavy (about 222 GeV) and very wide (width 0.035), total pT 1227 GeV and almost no pT within 0.05 of the axis. Large mass steps cancel: 'mass > 62.5' (-9.002) and 'mass > 144' (-7.956) against 'mass > 82.9' (+8.866), 'mass > 92.9' (+4.53) and 'mass > 173' (all, 4% elsewhere, +3.275). The value is 0.044, on for 19%, so it adds almost nothing. The formula calls them g (64%).
- **medium-mass (54 GeV), average width, pT spread over several particles, low pT** — 0.5% of jets, neuron 4.61. Mostly gluon (52%) mixed with 35% quark and 13% top; mass about 54 GeV, width 0.0096, the lowest total pT here (622 GeV) and a soft leading particle (132 GeV). 'sum_pt < 950' (+11.012) and 'log_sum_pt < 6.86' (-9.638) nearly cancel, as do 'mass_top50 < 137' (+4.751) and 'mass_top50 < 161' (-3.532); 'sum_pt_top40 < 956' (+1.534) and 'log_sum_pt < 6.86 and z_top20_slots > 0.897' (54%, +1.893) tip it up. The value 4.608, the highest of this neuron and below 8, adds 2.376 to g and 2.592 to q and removes 1.008 from W. The formula calls them g (62%).

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

- **heavy (97 GeV), average width, pT spread over several particles** — 33.2% of jets, neuron 1.52. Mostly Z (53%) mixed with 19% gluon and 15% top; mass about 97 GeV and width 0.0089, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. With the intercept 1.6076, 'mass > 74.3' (+0.469), 'girth2_top10 > 0.000237' (+0.466), 'D2 < 3.35' (82%, +0.365) and 'girth2_top5 < 0.0244 and sum_pt_top3 < 656' (89%, +0.266) roughly balance 'e2 < 0.0652' (-0.969), 'z_top5 > 0.42' (80%, -0.498) and 'lam1 > 0.00467' (95%, -0.289); the low-mass pair 'mass < 87.4' / 'mass < 86.4' passes for only 12-15%. The value 1.52 adds 1.496 to the t score. The formula calls them Z (56%).
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 21.6% of jets, neuron 1.30. Mostly W (77%) with 10% gluon; mass about 79 GeV and width 0.0059, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. 'mass < 87.4' (+1.97) and 'mass < 86.4' (-1.491) both pass for all and leave a small gain; 'e2 < 0.0652' (-1.102) and 'z_top5 > 0.42' (87%, -0.581) pull down while 'D2 < 3.35' (+0.381) and 'girth2_top10 > 0.000237' (+0.314) push up. The value 1.303 adds 1.283 to t. The formula calls them W.
- **very heavy (151 GeV), very wide, pT spread over several particles** — 8.1% of jets, neuron 2.73. Mostly top (78%) with 15% gluon; mass about 151 GeV and width 0.0237, soft leading particle (162 GeV) and pT spread over 0.05-0.2 from the axis. 'mass > 74.3' (+1.567), 'girth2_top10 > 0.000237' (+1.46) and 'mass_top50 > 137' (79%, +0.651) outweigh 'lam1 > 0.00467' (-1.368), 'mass > 144' (74%, -0.818), 'mass_top20 > 80.4' (-0.616), 'girth2_top10 > 0.00768' (-0.6) and 'e2 < 0.0652' (87%, -0.4). The value 2.725, the highest of this neuron, adds 2.683 to t. The formula calls them t.
- **light (37 GeV), very narrow, pT spread over several particles** — 7.6% of jets, neuron 0.28. Mostly quark (61%) with 27% gluon; light (about 37 GeV) and very narrow (width 0.0014), a hard leading particle (301 GeV) and 85% of the pT within 0.025 of the axis. 'mass < 87.4' (+12.026) and 'mass < 86.4' (-10.11) are huge but nearly cancel; after 'mass < 80.4' (+2.004) and 'girth2 < 0.00593' (+0.808), the terms 'e2 < 0.0652' (-1.716), 'mass < 62.5' (-1.543), 'z_top5 > 0.42' (96%, -0.891) and 'LHA < 0.187' (96%, -0.674) take it below the intercept. The value 0.278 (on for 33%) adds 0.274 to t. The formula calls them q (75%).
- **medium-mass (51 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 0.72. A gluon/quark mixture (47% gluon, 41% quark); mass about 51 GeV and width 0.0024, ordinary pT sharing and 71% of the pT within 0.025 of the axis. 'mass < 87.4' (+8.801) and 'mass < 86.4' (-7.346) nearly cancel; 'mass < 80.4' (+1.38) and 'girth2 < 0.00593' (+0.621) against 'e2 < 0.0652' (-1.624), 'mass < 62.5' (-0.726) and 'z_top5 > 0.42' (87%, -0.645). The value 0.717 (on for 64%) adds 0.706 to t. The formula splits them between g (52%) and q (48%).
- **very heavy (173 GeV), very wide, pT spread over several particles** — 7.3% of jets, neuron 2.23. Mostly top (89%); mass about 173 GeV and width 0.0285, soft leading particle (164 GeV) and pT spread over 0.05-0.2 from the axis. 'mass > 161' passes for all (4% elsewhere) and removes 2.338, with 'mass > 144' (-2.449), 'lam1 > 0.00467' (-1.685) and 'mass_top20 > 80.4' (-1.025), against 'mass > 74.3' (+2.025), 'girth2_top10 > 0.000237' (+1.907), 'mass_top50 > 137' (+1.865), 'mass_top40 > 137' (+1.155) and 'mass > 163' (+0.953). The value 2.232 adds 2.197 to t. The formula calls them t.
- **medium-mass (65 GeV), narrow, pT spread over several particles** — 6.7% of jets, neuron 0.99. Mostly gluon (49%) mixed with 28% quark and 11% W; mass about 65 GeV and width 0.0039, with 55% of the pT within 0.025 of the axis. 'mass < 87.4' (+5.379) and 'mass < 86.4' (-4.413) leave about one unit, 'mass < 80.4' adds 0.718, and 'e2 < 0.0652' (-1.508) and 'z_top5 > 0.42' (80%, -0.536) take back part of it. The value 0.987 (on for 76%) adds 0.972 to t. The formula calls them g (57%).
- **light (24 GeV), very narrow, pT spread over several particles** — 6.6% of jets, neuron 0.04. Mostly quark (82%); the lightest (about 24 GeV) and narrowest (width 0.0006) jets here, with a very hard leading particle (367 GeV) and 93% of the pT within 0.025 of the axis. 'mass < 87.4' (+15.172) and 'mass < 86.4' (-12.807) nearly cancel, and 'mass < 62.5' (-2.34), 'e2 < 0.0652' (-1.816), 'z_top5 > 0.42' (-1.228) and 'LHA < 0.187' (-1.02) outweigh 'mass < 80.4' (+2.613) and the intercept. The value is 0.042, on for 8%, so it hardly adds to the scores. The formula calls them q.
- **very heavy (202 GeV), very wide, pT spread over several particles** — 1.3% of jets, neuron 0.13. A gluon/top mixture (48% gluon, 41% top); very heavy (about 202 GeV) and very wide (width 0.0339), with almost no pT within 0.05 of the axis. 'mass > 161' (-7.741), 'mass > 144' (-4.845), 'lam1 > 0.00467' (-2.147) and 'mass_top50 > 161' (-1.439) outweigh 'mass > 163' (+3.586), 'mass_top50 > 137' (+3.268), 'mass > 74.3' (+2.615), 'girth2_top10 > 0.000237' (+2.275) and 'mass_top40 > 137' (+2.065). The value is 0.131, on for 22%, adding just 0.129 to t. The formula splits them between t (52%) and g (48%).
- **very heavy (258 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00. Mostly gluon (73%) with 17% top; the heaviest jets here (about 258 GeV), width 0.036 and a high total pT (1401 GeV), with almost no pT within 0.05 of the axis. 'mass > 161' (-18.191), 'mass > 144' (-9.48) and 'mass_top50 > 161' (-3.79) outweigh 'mass > 163' (+8.677), 'mass_top50 > 137' (+6.347), 'mass_top40 > 137' (+4.318) and 'mass > 74.3' (+3.757). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (83%).

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

- **medium-mass (87 GeV), average width, pT spread over several particles** — 53.3% of jets, neuron 0.75. A W/Z mixture (34% W, 35% Z, with 13% gluon and 11% top); mass about 87 GeV and width 0.0074, below the 0.0093 average, with ordinary pT sharing (hardest particle 236 GeV against 240). Only 12% of the pT lies within 0.025 of the axis (29% for all jets); most sits at 0.025-0.1, as for two resolved prongs. With the intercept 1.1048, 'mass_over_sum_pt_sq < 0.00818' (79% pass, +0.635) and 'log_sum_pt < 7.02' (+0.465) push up, while the mass steps 'mass > 74.3' (94%, -0.524), 'mass > 78.3' (86%, -0.539) and 'mass > 80.4' (70%, -0.296), plus 'width < 0.00961' (-0.474), pull down; these mass steps are what set this group apart from the light group, where they almost never pass. The lighter, W-like jets stay positive and the heavier Z-like ones drop to zero (mean 0.747, on for 64%), which adds 0.561 to the W score and takes 1.028 from the Z score. The formula splits them between W (38%) and Z (36%).
- **light (44 GeV), very narrow, pT spread over several particles** — 27.5% of jets, neuron 0.46. Mostly quark (52%) with 35% gluon; light (about 44 GeV) and very narrow (width 0.0019), with a hard leading particle (300 GeV against 240) and 80% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00818' passes for all and adds a large 2.985, but the narrowness tests that also always pass take most of it back: 'girth < 0.0566' (-2.293), 'width < 0.00961' (-1.43), 'girth2_top20 < 0.00604' (-1.06) and 'girth2_top40 < 0.00626' (-0.996), partly offset by 'girth2_top20 < 0.00604 and mass_top5 < 68.4' (+0.975); the mass steps above 74.3 GeV almost never pass. The neuron ends at 0.455 (on for 86%), adding 0.342 to W and removing 0.626 from Z. The formula calls them q.
- **very heavy (162 GeV), very wide, pT spread over several particles** — 12.4% of jets, neuron 0.00. Mostly top (75%) with 18% gluon; heavy (about 162 GeV) and broad (width 0.0257), with a soft leading particle (174 GeV) and only 3% of the pT within 0.025 of the axis, much of it at 0.1-0.2. Every mass step passes: 'mass > 92.9' alone removes 9.62, with 'mass > 78.3' (-4.532), 'mass > 74.3' (-3.379) and 'mass > 80.4' (-2.919), which 'mass_over_sum_pt > 0.0833' (+3.995) cannot offset. 'mass > 92.9 and max_dr < 0.391' passes for only 52% here (-1.68), which separates them from the next two groups. The neuron stays at 0 for all of them and adds nothing to any score.
- **very heavy (163 GeV), very wide, pT spread over several particles** — 4.4% of jets, neuron 0.00. Mostly top (75%) with 20% gluon; mass about 163 GeV and width 0.0254, soft leading particle (154 GeV) and pT spread mainly over 0.05-0.2 from the axis, with every particle inside 0.391. Here 'mass > 92.9 and max_dr < 0.391' passes for all and removes 13.679 on top of 'mass > 92.9' (-9.82) and the other mass steps, against only 'mass_over_sum_pt > 0.0833' (+3.956). The sum is deeply negative, so the neuron is 0 for all of them and adds nothing to the scores.
- **very heavy (177 GeV), very wide, pT spread over several particles** — 1.5% of jets, neuron 0.00. Mostly top (70%) with 26% gluon; mass about 177 GeV and width 0.0272, the softest leading particle of this neuron (137 GeV), with 26% of the pT at 0.15-0.2 from the axis against 6% for all jets. 'mass > 92.9 and max_dr < 0.391' passes for all and removes 33.677 on average, far more than in the previous group, with 'mass > 92.9' (-11.772) and the other mass steps adding to it; 'mass_over_sum_pt > 0.0833' (+4.271) is the only real help. The neuron is 0 throughout and adds nothing to the scores.
- **heavy (131 GeV), wide, pT spread over several particles** — 0.8% of jets, neuron 0.00. A gluon/top mixture (53% gluon, 37% top); mass about 131 GeV and width 0.0139, total pT 1167 GeV, with little pT at 0.025 or less (3%) and most at 0.05-0.15. 'mass > 92.9 and n_dr_0p2_0p4 < 7' passes for all (it passes for only 4% of other jets) and removes 28.893, and 'mass > 92.9 and max_dr < 0.391' (88%, -10.651) and 'mass > 92.9' (-5.381) add to the cut. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (60%), with 36% called t.
- **very heavy (168 GeV), wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. A small gluon/top mixture (64% gluon, 30% top); mass about 168 GeV and width 0.0156, with a high total pT (1416 GeV against 1044) and most pT at 0.05-0.2 from the axis. 'mass > 92.9 and n_dr_0p2_0p4 < 7' removes 85.252 and 'mass > 92.9 and max_dr < 0.391' (96%) another 45.958, the largest cuts in this neuron, while 'log_sum_pt < 7.02' passes for only 15%. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (76%).

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

- **heavy (133 GeV), very wide, pT spread over several particles** — 25.3% of jets, neuron 0.19. Mostly top (53%) mixed with 23% gluon and 15% Z; mass about 133 GeV and width 0.0187, soft leading particle (172 GeV) and pT spread out (9% within 0.025 of the axis). Almost no if-statement passes ('mass_over_sum_pt_sq < 0.00751' 14%, 'tau21 < 0.428' 28%), so the neuron sits at its intercept 0.2008 (mean 0.191, on for 98%). That small value adds 0.083 each to W and Z and removes 0.143 from g. The formula calls them t (55%).
- **heavy (119 GeV), wide, pT spread over several particles** — 12.4% of jets, neuron 0.02. A top/Z mixture (41% top, 25% Z, 16% gluon, 12% quark); mass about 119 GeV and width 0.0147, with most pT at 0.025-0.15 from the axis. 'tau21 < 0.428' passes for all and removes 0.367, more than the intercept, while the narrow and light tests mostly fail ('mass_over_sum_pt_sq < 0.00751' 17%). The value is 0.02, on for only 16%, so it barely touches the scores. The formula calls them t (47%).
- **medium-mass (85 GeV), average width, pT spread over several particles** — 11.0% of jets, neuron 0.86. A W/Z mixture (55% W, 43% Z); mass about 85 GeV and width 0.0066, with 57% of the pT at 0.05-0.1 from the axis and almost none inside 0.025, the typical two-prong spacing. 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.621) and 'lam2 < 0.000615' (+0.455) pass for all, against 'tau21 < 0.428' (-0.497); 'mass_over_sum_pt_sq < 0.00751' passes for 69% (+0.164). The value 0.858 adds 0.376 each to W and Z and removes 0.644 from g. The formula calls them W (57%), with 42% called Z.
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 9.5% of jets, neuron 0.37. A W/gluon/quark mixture (43% W, 29% gluon, 15% quark); mass about 78 GeV and width 0.0054, with more of the pT near the axis (36% within 0.025, 34% at 0.025-0.05) than the two-prong W/Z groups. 'mass_over_sum_pt_sq < 0.00751' passes for all (+0.33) but its companion 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' takes back 0.232; 'tau21 < 0.428' (39%) and 'lam2 < 0.000615' (29%) matter little. The value 0.367 adds 0.161 to W and Z and removes 0.276 from g. The formula calls them W (52%).
- **medium-mass (88 GeV), average width, pT spread over several particles** — 8.6% of jets, neuron 0.32. A Z/W mixture (46% Z, 43% W); mass about 88 GeV and width 0.0072, with a hard leading particle (296 GeV against 240) and 42% of the pT at 0.025-0.05 from the axis. 'lam2 < 0.000615' passes for all (+0.379) but 'tau21 < 0.428' (-0.447) cancels it; 'mass_over_sum_pt_sq < 0.00751' (63%) and 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (46%) add a little. The value 0.321 adds 0.14 to W and Z and removes 0.241 from g. The formula splits them between W (47%) and Z (46%).
- **light (34 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 1.27. Mostly quark (65%) with 23% gluon; light (about 34 GeV) and very narrow (width 0.0011), hard leading particle (341 GeV) and 90% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00751' (+1.031) is nearly cancelled by 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' (-0.99); what remains comes from 'mass_top20 < 45.6' (+0.441), 'lam2 < 0.000615' (+0.359) and 'z_top20_slots > 0.968 and girth2_top3 < 0.000824' (69%, +0.148), with 'tau21 < 0.428' almost never passing. The value 1.266 adds 0.554 to W and Z and removes 0.95 from g. The formula calls them q (80%).
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.62. A quark/gluon mixture (41% quark, 40% gluon, 10% W); mass about 60 GeV and width 0.0032, with 62% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00751' (+0.694) against its companion '... and mass_top20 < 137' (-0.56) leaves a small net; 'mass_top20 < 45.6' (72%, +0.126) and 'lam2 < 0.000615' (46%, +0.104) add a little. The value 0.622 adds 0.272 to W and Z and removes 0.467 from g. The formula calls them q (53%), with 43% called g.
- **light (50 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.78. Mostly gluon (62%) with 31% quark; mass about 50 GeV and width 0.0022, a softer leading particle (225 GeV) than the quark-rich light groups, and 75% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00751' (+0.847) and '... and mass_top20 < 137' (-0.816) nearly cancel, so 'mass_top20 < 45.6', passing for all (+0.454), gives most of the value; 'lam2 < 0.000615' passes for only 43%. The value 0.778 adds 0.34 to W and Z and removes 0.583 from g. The formula calls them g (70%).
- **medium-mass (84 GeV), average width, pT spread over several particles** — 6.0% of jets, neuron 0.71. A W/Z mixture (55% W, 40% Z); mass about 84 GeV and width 0.0067, a softer leading particle (208 GeV) and 52% of the pT at 0.05-0.1 from the axis. 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' passes for all (+0.528), 'tau21 < 0.428' (82%) removes 0.202, and 'mass_over_sum_pt_sq < 0.00751' (72%, +0.163) and 'lam2 < 0.000615' (61%, +0.117) add a little; the weaker 'lam2' pass rate is what separates them from the other quiet-ring W/Z group. The value 0.712 adds 0.311 to W and Z and removes 0.534 from g. The formula calls them W (59%).
- **light (25 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 1.88. Mostly quark (79%); the lightest (about 25 GeV) and narrowest (width 0.0006) jets of this neuron, with a leading particle of 354 GeV and 90% of the pT within 0.025 of the axis. Every narrowness test passes: after 'mass_over_sum_pt_sq < 0.00751' (+1.096) and '... and mass_top20 < 137' (-1.083) cancel, 'mass_top20 < 45.6' (+0.504), 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.492), 'lam2 < 0.000615' (+0.49) and 'z_top20_slots > 0.968 and girth2_top3 < 0.000824' (79%, +0.19) add up. The value 1.884, the highest of this neuron, adds 0.824 to W and Z and removes 1.413 from g. The formula calls them q (95%).

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

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 25.2% of jets, neuron 0.03. Mostly W (71%) with 11% gluon; mass about 80 GeV and width 0.006, ordinary pT sharing, with the pT spread over 0.025-0.1 from the axis (17% inside 0.025). With the intercept 0.7257, the mass windows pull in both directions: 'mass < 173' (+1.734), 'mass_over_sum_pt_sq < 0.0199' (+1.686) and 'mass < 92.9' (+0.838) against 'mass < 101' (-1.573), 'mass < 121' (-1.571), 'mass_top40 < 161' (-1.086), 'mass_over_sum_pt > 0.0508' (-1.067) and 'girth2_top50 < 0.0195' (-1.052). The value is 0.029 and the neuron is on for only 10%, so it barely touches the scores. The formula calls them W.
- **heavy (94 GeV), average width, pT spread over several particles** — 23.8% of jets, neuron 0.16. Mostly Z (70%) with 15% gluon; mass about 94 GeV and width 0.008, ordinary pT sharing, with 41% of the pT at 0.05-0.1 from the axis. 'mass_over_sum_pt > 0.0508' (-1.566), 'mass < 121' (98%, -1.053), 'mass_top40 < 161' (-0.923), 'girth2_top50 < 0.0195' (-0.899), 'mass < 101' (89%, -0.651) and 'n_dr_0p2_0p4 < 21' (-0.63) outweigh 'mass < 173' (+1.479) and 'mass_over_sum_pt_sq < 0.0199' (+1.439); unlike the W-mass group, 'mass < 86.4' passes for only 3%. The value is 0.16 (on for 30%), which removes 0.155 from Z. The formula calls them Z.
- **light (45 GeV), very narrow, pT spread over several particles** — 10.2% of jets, neuron 1.36. A quark/gluon mixture (47% quark, 41% gluon); light (about 45 GeV) and narrow (width 0.002), a hard leading particle (276 GeV) and 77% of the pT within 0.025 of the axis. All the mass windows pass: 'mass < 101' (-4.181), 'mass < 121' (-2.918) and 'mass_top40 < 161' (-1.546) against 'mass < 92.9' (+3.116), 'mass < 173' (+2.384), 'mass_over_sum_pt_sq < 0.0199' (+2.171) and 'mass < 86.4' (+1.717); 'mass_top50 < 71.8' (all, 22% elsewhere, +0.504) helps and 'mass_over_sum_pt > 0.0508' passes for only 15%. The value 1.362 adds 0.298 to q and 0.213 to g and removes 1.319 from Z. The formula calls them q (57%), with 43% called g.
- **light (27 GeV), very narrow, pT spread over several particles** — 9.9% of jets, neuron 1.91. Mostly quark (77%); very light (about 27 GeV) and very narrow (width 0.0008), a very hard leading particle (348 GeV) and 91% of the pT within 0.025 of the axis. The same mass windows with larger amounts: 'mass < 101' (-5.543) and 'mass < 121' (-3.622) against 'mass < 92.9' (+4.305), 'mass < 173' (+2.723), 'mass < 86.4' (+2.475) and 'mass_over_sum_pt_sq < 0.0199' (+2.314), plus 'mass_top50 < 71.8' (+0.822); 'mass_over_sum_pt > 0.0508' almost never passes. The value 1.912, the highest of this neuron, adds 0.418 to q and 0.299 to g and removes 1.852 from Z. The formula calls them q.
- **heavy (131 GeV), wide, pT spread over several particles** — 8.5% of jets, neuron 0.97. A top/gluon mixture (52% top, 33% gluon, 13% quark); mass about 131 GeV and width 0.016, softer leading particle (183 GeV) and 41% of the pT at 0.05-0.1 from the axis. 'mass_over_sum_pt > 0.0508' (-3.067) is largely offset by the broad-jet bonuses 'LHA > 0.26' (97%, +0.983), 'girth2_top20 > 0.00182' (+0.788), 'LHA > 0.228' (+0.507) and 'mass < 173' (+0.791), with 'sum_pt < 1.09e+03' (73%, +0.726); the low-mass windows such as 'mass < 101' fail. The value 0.966 (on for 82%) adds 0.211 to q and 0.151 to g and removes 0.936 from Z. The formula calls them t (59%), with 35% called g.
- **medium-mass (64 GeV), narrow, pT spread over several particles** — 8.4% of jets, neuron 0.60. Mostly gluon (51%) mixed with 29% quark and 9% W; mass about 64 GeV and width 0.0037, ordinary pT sharing and 58% of the pT within 0.025 of the axis. 'mass < 101' (-2.807), 'mass < 121' (-2.208), 'mass_top40 < 161' (-1.353) and 'girth2_top50 < 0.0195' (-1.238) against 'mass < 173' (+2.041), 'mass_over_sum_pt_sq < 0.0199' (+1.963), 'mass < 92.9' (+1.915) and 'e2 < 0.0476' (+1.125), with 'e2 < 0.0303' (99%, -0.391) taking a little back. The value 0.599 (on for 93%) adds 0.131 to q and removes 0.58 from Z. The formula calls them g (59%), with 33% called q.
- **very heavy (165 GeV), very wide, pT spread over several particles** — 5.7% of jets, neuron 0.25. Mostly top (85%); mass about 165 GeV and width 0.027, soft leading particle (168 GeV) and 31% of the pT at 0.05-0.1 and 23% at 0.1-0.15 from the axis. 'mass_over_sum_pt > 0.0508' (-4.611) and 'girth2_top20 > 0.0166' (99%, 10% elsewhere, -1.373) outweigh 'LHA > 0.26' (+1.622), 'girth2_top20 > 0.00182' (+1.565), 'sum_pt < 1.09e+03' (88%, +0.761), 'LHA > 0.228' (+0.748) and 'lam1 > 0.00824 and sum_pt < 1.17e+03' (+0.695). The value is 0.246, on for 34%, so it adds little (-0.238 on Z). The formula calls them t.
- **very heavy (161 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 0.62. Mostly top (79%); mass about 161 GeV and width 0.0253, a soft leading particle (138 GeV) and 34% of the pT at 0.1-0.15 from the axis. 'mass_over_sum_pt > 0.0508' (-4.389) and 'girth2_top20 > 0.0166' (-1.432) pull down; 'girth2_top3 > 0.01' passes for all (12% elsewhere) and removes 2.506, but 'girth2_top3 > 0.01 and D2 < 4.45' gives back 2.252, and 'LHA > 0.26' (+1.83) and 'girth2_top20 > 0.00182' (+1.584) lift it. The value 0.62 (on for 45%) adds 0.136 to q and removes 0.601 from Z. The formula calls them t.
- **very heavy (175 GeV), very wide, pT spread over several particles** — 2.7% of jets, neuron 0.36. Mostly top (89%); mass about 175 GeV and width 0.0314, soft leading particle (167 GeV) and a quarter of the pT at 0.15-0.2 from the axis. 'mass_over_sum_pt > 0.0508' (-5.134) and 'girth2_top20 > 0.0166' (-2.704) against 'LHA > 0.26' (+2.126) and 'girth2_top20 > 0.00182' (+2.034), while 'girth2_top3 > 0.01' (-2.092) and its D2 companion (+2.118) cancel; 'e2 > 0.0556' passes for 96% (7% elsewhere, +1.803), partly taken back by 'e2 > 0.0556 and n_real_top40 > 34' (-1.146). The value is 0.365, on for 37%, which removes 0.353 from Z. The formula calls them t.
- **very heavy (183 GeV), very wide, pT spread over several particles** — 2.1% of jets, neuron 0.52. Mostly top (77%) with 16% gluon; mass about 183 GeV and width 0.0334, soft leading particle (138 GeV) and 30% of the pT at 0.15-0.2 from the axis. 'mass_over_sum_pt > 0.0508' (-5.346), 'girth2_top3 > 0.01' (-4.916) and 'girth2_top20 > 0.0166' (-3.355) against 'girth2_top3 > 0.01 and D2 < 4.45' (+5.023), 'girth2_top20 > 0.00182' (+2.262), 'LHA > 0.26' (+2.24) and 'e2 > 0.0556' (91%, +1.564). The value is 0.517, on for 36%, which removes 0.501 from Z. The formula calls them t.

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

- **heavy (94 GeV), average width, pT spread over several particles** — 55.8% of jets, neuron 0.07. Over half of all jets, a mixture of 32% W, 33% Z, 15% top and 14% gluon; mass about 94 GeV and width 0.0087, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. With the intercept 0.0004, the light-mass tests rarely pass ('mass < 82.9' 34%, 'mass < 78.3' 7%) and the small terms 'lam2 < 0.000615' (53%, -0.1), 'planar_flow > 0.259' (53%, -0.08) and 'C2 > 0.0612' (35%, -0.073) push down. The value is 0.068, on for 18%, so it barely touches the scores. The formula splits them between W (35%) and Z (34%), with 15% called t.
- **very heavy (173 GeV), very wide, pT spread over several particles** — 12.7% of jets, neuron 0.12. Mostly top (82%) with 15% gluon; mass about 173 GeV and width 0.0282, soft leading particle (160 GeV) and pT spread over 0.05-0.2 from the axis. 'mass > 144' passes for 98% (3% elsewhere) and adds 0.665, but 'C2 > 0.0612' (96%, -0.531), 'mass_top20 > 125 and C2 > 0.056' (54%, -0.202) and 'planar_flow > 0.259' (81%, -0.178) take it back. The value is 0.116, on for 21%, so it adds little. The formula calls them t.
- **light (47 GeV), very narrow, pT spread over several particles** — 6.0% of jets, neuron 0.58. An even gluon/quark mixture (45% gluon, 44% quark); light (about 47 GeV) and narrow (width 0.002), with 77% of the pT within 0.025 of the axis. 'mass < 82.9' (+2.652) is cancelled by 'mass < 82.9 and z_dr_0_0p05 > 0.329' (-2.698) because the core is dense; the rest comes from 'mass < 78.3' (+1.56), 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+1.201), 'mass_top50 < 80.4' (+1.05) and 'mass < 74.3' (+0.796) against 'mass_top50 < 77.4' (-2.123), 'girth < 0.0505' (-1.491) and 'mass < 62.5' (-1.399). The value 0.581 (on for 85%) adds 0.2 to q and 0.136 to g and removes 0.236 from W and 0.218 from t. The formula splits them between q (51%) and g (49%).
- **light (36 GeV), very narrow, pT spread over several particles** — 5.9% of jets, neuron 0.51. Mostly quark (63%) with 25% gluon; light (about 36 GeV) and very narrow (width 0.0012), a hard leading particle (310 GeV) and 87% of the pT within 0.025 of the axis. The same balance with larger terms: 'mass < 82.9 and z_dr_0_0p05 > 0.329' (-3.752), 'mass_top50 < 77.4' (-2.792), 'mass < 62.5' (-2.374) and 'girth < 0.0505' (-1.881) against 'mass < 82.9' (+3.454), 'mass < 78.3' (+2.1), 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+1.614) and 'mass_top50 < 80.4' (+1.352); 'mass_top50 < 43.7' passes for all (9% elsewhere). The value 0.513 (on for 83%) adds 0.176 to q and removes 0.208 from W. The formula calls them q (78%).
- **medium-mass (70 GeV), narrow, pT spread over several particles** — 5.7% of jets, neuron 0.45. Mostly gluon (44%) mixed with 24% quark and 19% W; mass about 70 GeV and width 0.0044, with 51% of the pT within 0.025 of the axis. 'mass < 82.9' (+0.937), 'mass < 78.3' (98%, +0.404), 'mass_top50 < 80.4' (+0.394) and 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+0.378) against 'mass < 82.9 and z_dr_0_0p05 > 0.329' (98%, -0.727), 'mass_top50 < 77.4' (-0.67), 'girth < 0.0505' (77%, -0.557) and 'mass_top40 < 89.7 and n_particles > 22' (99%, -0.436). The value 0.449 (on for 57%) adds 0.155 to q and removes 0.183 from W. The formula calls them g (52%), with about a quarter each called W and q.
- **medium-mass (58 GeV), narrow, pT spread over several particles** — 5.4% of jets, neuron 0.75. Mostly gluon (53%) mixed with 33% quark; mass about 58 GeV and width 0.003, with 65% of the pT within 0.025 of the axis. 'mass < 82.9' (+1.832), 'mass < 78.3' (+1.006), 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+0.801), 'mass_top50 < 80.4' (+0.747), 'mass < 82.9 and z_dr_0p05_0p1 < 0.647' (+0.629) and 'mass < 74.3' (+0.472) outweigh 'mass < 82.9 and z_dr_0_0p05 > 0.329' (-1.684), 'mass_top50 < 77.4' (-1.452) and 'girth < 0.0505' (99%, -1.04), while 'mass < 62.5' mostly fails. The value 0.746, the highest of this neuron, adds 0.256 to q, 0.233 to Z and 0.175 to g and removes 0.303 from W and 0.28 from t. The formula calls them g (59%).
- **light (26 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 0.47. Mostly quark (81%); very light (about 26 GeV) and very narrow (width 0.0007), a very hard leading particle (369 GeV) and 93% of the pT within 0.025 of the axis. 'mass < 82.9 and z_dr_0_0p05 > 0.329' (-4.761), 'mass_top50 < 77.4' (-3.475), 'mass < 62.5' (-3.297) and 'girth < 0.0505' (-2.216) against 'mass < 82.9' (+4.213), 'mass < 78.3' (+2.612), 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+2.032), 'mass_top50 < 80.4' (+1.66), 'mass < 74.3' (+1.412) and 'mass_top50 < 43.7' (+0.88). The value 0.47 (on for 88%) adds 0.162 to q and removes 0.191 from W. The formula calls them q.
- **very light (18 GeV), very narrow, pT spread over several particles** — 1.4% of jets, neuron 0.43. Mostly quark (87%); the lightest (about 18 GeV) and narrowest (width 0.0003) jets of all groups, with a leading particle of 387 GeV and 95% of the pT within 0.025 of the axis. 'mass < 82.9 and z_dr_0_0p05 > 0.329' (-5.495), 'mass < 62.5' (-4.018), 'mass_top50 < 77.4' (-4.015), 'girth < 0.0505' (-2.335) and 'mass_top50 < 43.7 and max_dr < 0.436' (all, 13% elsewhere, -2.126) against 'mass < 82.9' (+4.805), 'mass < 78.3' (+3.012), 'mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647' (+2.342), 'mass < 74.3' (+1.646) and 'mass_top50 < 43.7' (+1.276). The value 0.431 (on for 87%) adds 0.148 to q and removes 0.175 from W. The formula calls them q.
- **heavy (100 GeV), wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.01. A top/gluon/quark mixture (47% top, 37% gluon, 16% quark); mass about 100 GeV and width 0.018, a low total pT (773 GeV) and soft leading particle (134 GeV). 'log_sum_pt < 6.86' passes for all (8% elsewhere) and removes 3.304, far more than 'mass < 82.9' (34%, +0.305) gives, with 'C2 > 0.0612' (78%, -0.33) adding to the loss. The value is 0.005, on for under 1%, so it adds essentially nothing. The formula calls them t (55%).
- **light (39 GeV), narrow, pT spread over several particles, low pT** — 0.5% of jets, neuron 0.00. A gluon/quark mixture (55% gluon, 39% quark); light (about 39 GeV), width 0.0042, the lowest total pT here (651 GeV) and a soft leading particle (150 GeV), with 62% of the pT within 0.025 of the axis. 'log_sum_pt < 6.86' removes 6.115, and 'mass < 82.9 and z_dr_0_0p05 > 0.329' (95%, -2.992), 'mass_top50 < 77.4' (-2.658) and 'mass < 62.5' (98%, -2.152) outweigh 'mass < 82.9' (+3.269), 'mass < 78.3' (+1.975) and the other light-mass bonuses. The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (66%).

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

- **medium-mass (73 GeV), narrow, pT spread over several particles** — 74.5% of jets, neuron 1.95. Three quarters of all jets, a broad mixture (27% W, 27% Z, 22% quark, 20% gluon, only 4% top); mass about 73 GeV and width 0.0052, a harder leading particle than average (265 GeV) and 36% of the pT within 0.025 of the axis. With the intercept 1.6223, 'girth2 < 0.014' (+0.988), the 'n_particles' term (+0.479) and 'e2_sq < 0.00961' (96%, 14% elsewhere, +0.44) push up, against 'z_dr_0p2_0p4 < 0.129' (-0.713), 'girth < 0.121' (-0.527), 'z_top30_slots > 0.946' (78%, -0.331), 'sum_pt < 1.05e+03' (67%, -0.316) and 'LHA > 0.228' (57%, -0.311); the heavy-mass cuts almost never pass. The value 1.949, the highest of this neuron, removes 1.766 from the t score; the formula's decisions follow the mixed true content.
- **very heavy (145 GeV), very wide, pT spread over several particles** — 9.4% of jets, neuron 1.24. Mostly top (71%) with 20% gluon; mass about 145 GeV and width 0.0208, softer leading particle (170 GeV) and pT spread over 0.05-0.2 from the axis. 'girth > 0.0505' (+1.602), 'mass > 74.3' (+0.849), 'mass > 62.5' (+0.751) and the 'n_particles' term (+0.671) are partly cancelled by 'LHA > 0.228' (-1.423), 'mass_top50 > 97.9' (-1.234), 'mass > 144' (59%, -0.793), 'girth2_top40 > 0.00663' (-0.713) and 'mass > 121' (91%, -0.518). The value 1.241 (on for 86%) removes 1.125 from t. The formula calls them t (80%).
- **very heavy (172 GeV), very wide, pT spread over several particles** — 6.5% of jets, neuron 0.13. Mostly top (90%); mass about 172 GeV and width 0.0276, soft leading particle (166 GeV) and pT spread over 0.05-0.2 from the axis. 'mass > 161' passes for all (5% elsewhere) and removes 4.064, with 'mass > 144' (-2.989), 'mass_top50 > 97.9' (-2.031) and 'LHA > 0.228' (-1.775), against 'mass_top40 > 137' (+2.196), 'girth > 0.0505' (+2.114), 'mass_top50 > 137' (+1.471) and 'mass > 161 and D2 < 5.38' (+1.202). The value is 0.134, on for 26%, so it removes only 0.121 from t. The formula calls them t.
- **medium-mass (87 GeV), average width, pT spread over several particles** — 5.9% of jets, neuron 0.21. A top/gluon/quark mixture (46% top, 29% gluon, 22% quark); mass about 87 GeV and width 0.0111, a low total pT (888 GeV) and soft leading particle (174 GeV). The total-pT tests decide it: 'sum_pt < 1.01e+03' (-2.09), 'sum_pt < 1.05e+03' (-1.924), 'log_sum_pt < 6.9' (-1.577) and 'sum_pt_top40 < 956' (-1.215) outweigh 'sum_pt_top50 < 1.01e+03' (+1.738), 'sum_pt_top50 < 988' (+1.58), 'sum_pt_top40 < 1.05e+03' (+1.337) and 'sum_pt_top50 < 959' (+1.11). The value is 0.212, on for 27%, removing 0.192 from t. The formula calls them t (53%).
- **very heavy (192 GeV), very wide, pT spread over several particles** — 1.1% of jets, neuron 0.00. A top/gluon mixture (46% top, 44% gluon); very heavy (about 192 GeV), width 0.0293 and total pT 1154 GeV, with pT spread over 0.05-0.2 from the axis. 'mass > 161' (-11.191), 'mass > 173' (all, 4% elsewhere, -7.358), 'mass > 144' (-5.08) and 'mass > 173 and pt_6 < 62.2' (94%, -4.086) outweigh 'mass_top40 > 137' (+3.339), 'mass > 161 and D2 < 5.38' (+3.131) and 'mass_over_sum_pt > 0.171' (60%, +2.784). The neuron is 0 for all of them and adds nothing to the scores. The formula splits them between t (54%) and g (45%).
- **medium-mass (63 GeV), average width, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00. Mostly gluon (51%) mixed with 31% quark and 19% top; mass about 63 GeV, width 0.0106, a very low total pT (677 GeV) and soft leading particle (132 GeV). The low-pT tests take away far more than they give: 'sum_pt < 1.01e+03' (-5.618), 'log_sum_pt < 6.9' (-5.367), 'sum_pt < 1.05e+03' (-4.379), 'sum_pt_top40 < 956' (-3.593) and 'sum_pt < 1.09e+03 and sum_pt_top30 < 801' (all, 4% elsewhere, -3.432) against 'sum_pt_top50 < 988' (+4.419), 'sum_pt_top50 < 1.01e+03' (+4.293), 'sum_pt_top50 < 959' (+3.793) and 'log_sum_pt < 6.81' (+3.683). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (62%).
- **very heavy (173 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.06. Mostly top (82%); mass about 173 GeV and width 0.0367, total pT 909 GeV, a soft leading particle (125 GeV) and nearly half the pT at 0.1-0.2 from the axis. 'mass_over_sum_pt > 0.171' passes for all (4% elsewhere) and adds 7.576, with 'girth > 0.0505' (+2.708), but 'mass > 161' (92%, -4.709), 'mass_over_sum_pt_sq > 0.0292' (-4.474), 'mass > 144' (-3.118), 'LHA > 0.228' (-2.159) and 'mass_top50 > 97.9' (-1.984) outweigh them. The value is 0.056, on for 11%, so it hardly moves the scores. The formula calls them t.
- **very heavy (205 GeV), very wide, pT spread over several particles** — 0.4% of jets, neuron 0.00. Mostly top (62%) with 27% gluon; very heavy (about 205 GeV) and very wide (width 0.0427), total pT 996 GeV and almost no pT within 0.05 of the axis. 'mass > 161' (-15.806), 'mass > 173' (-12.3), 'mass > 173 and pt_6 < 62.2' (-9.696) and 'mass_over_sum_pt_sq > 0.0292' (-8.103) outweigh 'mass_over_sum_pt > 0.171' (+13.191). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them t (84%).
- **very heavy (229 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00. Mostly gluon (78%) with 13% top; very heavy (about 229 GeV), width 0.0317 and a high total pT (1321 GeV), with the pT spread far from the axis. 'mass > 161' (-24.599), 'mass > 173' (-21.716), 'mass > 173 and pt_6 < 62.2' (87%, -9.271) and 'mass > 144' (-9.013) far outweigh 'mass > 161 and D2 < 5.38' (+7.39) and 'mass_top40 > 137' (+6.191). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (89%).
- **very heavy (277 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. Mostly gluon (75%) with 17% top; the heaviest jets here (about 277 GeV), width 0.0395 and total pT 1430 GeV. 'mass > 161' (-41.922) and 'mass > 173' (-40.267) dwarf 'mass > 161 and D2 < 5.38' (+12.293), 'mass_over_sum_pt > 0.171' (83%, +10.634) and 'mass > 173 and z_6 < 0.0425' (85%, +10.145). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them g (82%).

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

- **medium-mass (86 GeV), average width, pT spread over several particles** — 41.6% of jets, neuron 1.09. A W/Z mixture (39% W, 40% Z); mass about 86 GeV and width 0.0072, ordinary total pT (1017 GeV) and pT sharing, with most pT at 0.025-0.1 from the axis (11% inside 0.025). With the intercept -0.4569, 'girth2_top15 < 0.0156' (+1.087), 'max_dr < 0.402' (92%, +0.609) and 'mass < 125' (+0.549) lift it, against 'mass_over_sum_pt < 0.141' (-0.853) and 'mass < 92.9' (87%, -0.465); 'sum_pt > 1.02e+03' passes for 55% (+0.209). The value 1.091 (on for 98%) adds 0.136 to the q score and removes 0.409 from Z. The formula splits them between W (43%) and Z (41%).
- **very heavy (153 GeV), very wide, pT spread over several particles** — 18.8% of jets, neuron 0.12. Mostly top (79%); mass about 153 GeV and width 0.0247, soft leading particle (160 GeV) and only 2% of the pT within 0.025 of the axis, much of it at 0.1-0.2. Too heavy for the mass tests: 'mass < 92.9', 'mass < 91.2' and 'mass < 125' (18%) hardly pass, and 'girth2_top15 < 0.0156' passes for only 28%; 'mass_top40 > 161' (27%, -0.335) pulls down while 'max_dr < 0.402' (+0.231) helps. The value is 0.122 and the neuron is on for only 29%, so it adds little (0.015 to q, -0.046 on Z). The formula calls them t.
- **light (31 GeV), very narrow, pT spread over several particles** — 12.7% of jets, neuron 0.16. Mostly quark (74%) with 13% gluon; light (about 31 GeV) and very narrow (width 0.001), a hard leading particle (326 GeV) and 89% of the pT within 0.025 of the axis. 'mass < 92.9' (-3.708), 'mass_over_sum_pt < 0.141' (-1.668), 'mass_over_sum_pt < 0.0744' (-1.081) and 'lam1 < 0.00619' (-1.025) all pass and outweigh 'mass < 91.2' (+1.927), 'girth2_top15 < 0.0156' (+1.675) and 'mass < 125' (+1.322). The value is 0.164, on for 22%, so it barely moves the scores. The formula calls them q.
- **medium-mass (57 GeV), narrow, pT spread over several particles** — 10.6% of jets, neuron 0.59. A quark/gluon mixture (42% quark, 39% gluon, 9% W); mass about 57 GeV and width 0.0034, ordinary pT sharing and 62% of the pT within 0.025 of the axis. The same tug of war as the light group but weaker: 'mass < 92.9' (-2.142) and 'mass_over_sum_pt < 0.141' (-1.267) against 'girth2_top15 < 0.0156' (+1.565), 'mass < 91.2' (+1.091) and 'mass < 125' (+0.954), with 'mass_over_sum_pt < 0.0744' passing for 97% (-0.431). The value 0.586 (on for 72%) adds 0.073 to q and removes 0.22 from Z. The formula calls them q (51%), with 44% called g.
- **heavy (103 GeV), average width, pT spread over several particles** — 7.4% of jets, neuron 2.43. A mixture led by gluon (38%) with 22% W, 22% Z and 12% top; mass about 103 GeV and width 0.0088, total pT 1146 GeV (above the 1044 average). 'sum_pt > 1.02e+03' passes for all and adds 1.92, with 'girth2_top15 < 0.0156' (+1.055) and 'log_sum_pt > 6.9 and girth2_top10 < 0.0198' (+0.67), against 'sum_pt > 1.07e+03' (-0.934), 'mass_over_sum_pt < 0.141' (-0.786) and 'sum_pt_top50 > 997' (-0.596). The value 2.428 adds 0.303 to q and removes 0.91 from Z. The formula calls them g (44%), with about a fifth each called W and Z.
- **heavy (108 GeV), average width, pT spread over several particles, high pT** — 3.3% of jets, neuron 4.28. Mostly gluon (67%) with 11% W and 9% Z; mass about 108 GeV and width 0.0078, total pT 1302 GeV. 'sum_pt > 1.02e+03' (+4.252), 'sum_pt_top40 > 1.07e+03' (+1.799) and 'log_sum_pt > 6.9 and girth2_top10 < 0.0198' (+1.408) outweigh 'sum_pt > 1.07e+03' (-2.771) and 'sum_pt_top50 > 997' (-1.248). The sum stays below the largest value (8), giving 4.281, which adds 0.535 to q and removes 1.606 from Z. The formula calls them g.
- **light (47 GeV), very narrow, pT spread over several particles** — 3.3% of jets, neuron 3.11. A gluon/quark mixture (68% gluon, 27% quark); light (about 47 GeV) and narrow (width 0.0016), total pT 1198 GeV, hard leading particle (312 GeV) and 80% of the pT within 0.025 of the axis. The pT tests ('sum_pt > 1.02e+03' +2.706, 'sum_pt_top40 > 1.07e+03' +1.157) and the narrow/light ones ('girth2_top15 < 0.0156' +1.664, 'mass < 91.2' +1.419) beat 'mass < 92.9' (-2.757), 'sum_pt > 1.07e+03' (-1.553) and 'mass_over_sum_pt < 0.141' (-1.544). The value 3.11 adds 0.389 to q and removes 1.166 from Z. The formula calls them g (73%).
- **medium-mass (85 GeV), narrow, pT spread over several particles, high pT** — 1.7% of jets, neuron 3.99. Mostly gluon (76%); mass about 85 GeV but narrow (width 0.004), total pT 1486 GeV, a very hard leading particle (347 GeV) and 55% of the pT within 0.025 of the axis. 'sum_pt > 1.02e+03' (+7.014), 'sum_pt_top40 > 1.07e+03' (+3.776), 'sum_pt_top15 > 1e+03' (93%, +2.747) and 'log_sum_pt > 6.9 and girth2_top10 < 0.0198' (+2.446) far outweigh 'sum_pt > 1.07e+03' (-4.946) and 'sum_pt_top50 > 997' (-2.089). Their sum is at or above the largest value (8), so the value wraps around and is effectively scrambled (the mean 3.988 is not a real level), and so is what it adds to the scores (0.498 to q, -1.495 on Z on average). The formula calls them g.
- **very heavy (239 GeV), very wide, pT spread over several particles, high pT** — 0.4% of jets, neuron 0.03. Mostly gluon (73%) with 15% top; very heavy (about 239 GeV) and very wide (width 0.0342), total pT 1319 GeV, with almost no pT near the axis and a quarter of it at 0.15-0.2. 'mass_top40 > 161' passes for all (6% elsewhere) and removes 8.657, more than 'sum_pt > 1.02e+03' (+4.503) and the other pT terms give back; the mass and girth tests that help other groups ('mass < 125', 'girth2_top15 < 0.0156') fail. The value is 0.031 and the neuron is on for under 2%, so it adds almost nothing. The formula calls them g (84%).
- **heavy (104 GeV), narrow, pT spread over several particles, high pT** — 0.4% of jets, neuron 4.12. Mostly gluon (79%); mass about 104 GeV, narrow (width 0.0041), the highest total pT here (1832 GeV) and a leading particle of 457 GeV, with 55% of the pT within 0.025 of the axis. 'sum_pt > 1.02e+03' (+12.177), 'sum_pt_top40 > 1.07e+03' (+7.113) and 'sum_pt_top15 > 1e+03' (+6.357) dwarf 'sum_pt > 1.07e+03' (-9.012) and 'sum_pt_top50 > 997' (-3.585), so the sum lies far above the largest value (8). The value wraps around and is effectively scrambled (mean 4.116), as is what it adds to the scores. The formula calls them g.

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

- **very heavy (146 GeV), very wide, pT spread over several particles** — 26.3% of jets, neuron 0.26. Mostly top (65%) with 19% gluon; mass about 146 GeV and width 0.0215, soft leading particle (174 GeV) and pT spread over 0.05-0.2 from the axis. Almost no if-statement passes ('mass < 92.9' 6%, 'mass < 101' 13%, 'LHA < 0.372' 44%), so the neuron sits near its intercept 0.2277 (mean 0.258, on for 93%). That adds 0.153 to W and removes 0.065 from q. The formula calls them t (72%).
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 13.0% of jets, neuron 1.64. Mostly W (89%); mass about 80 GeV and width 0.0061, with 54% of the pT at 0.05-0.1 from the axis and almost none inside 0.025, the clean two-prong spacing. 'mass < 92.9' (+0.739), 'n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21' (97%, +0.55), 'mass < 101' (+0.486), 'mass_over_sum_pt_sq < 0.00818' (+0.335), 'LHA < 0.372' (+0.309) and 'n_dr_0p2_0p4 < 5' (90%, 29% elsewhere, +0.294) add up, while 'girth2_top20 < 0.00754' (-0.364), 'lam1 < 0.00672' (-0.352) and 'mass_top50 < 86.4' (98%, -0.205) take a little. The value 1.639, the highest of this neuron, adds 0.973 to W and removes 0.41 from q. The formula calls them W.
- **heavy (91 GeV), average width, pT spread over several particles** — 12.1% of jets, neuron 1.28. Mostly Z (90%); mass about 91 GeV and width 0.0078, with 54% of the pT at 0.05-0.1 from the axis and almost none inside 0.025. 'n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21' (94%, +0.394), 'mass < 101' (+0.248), 'n_dr_0p2_0p4 < 5' (81%, +0.248), 'LHA < 0.372' (+0.246) and 'mass < 92.9' (86%, +0.156) add up, with small losses from 'n_dr_0p2_0p4 < 10 and z_dr_0p1_0p2 < 0.334' (-0.14) and 'z_top5_slots > 0.535' (-0.113); the narrowness penalties rarely pass ('mass_top50 < 86.4' 10%). The value 1.276 adds 0.758 to W and removes 0.319 from q. The formula calls them Z.
- **heavy (95 GeV), average width, pT spread over several particles** — 8.7% of jets, neuron 0.24. Mostly Z (50%) mixed with 28% gluon; mass about 95 GeV and width 0.008, with half the pT at 0.025-0.05 from the axis (22% for all jets). 'e2 < 0.0388' (-0.534), 'girth2_top20 < 0.00754' (95%, -0.422), 'girth < 0.0768' (95%, -0.363) and 'LHA < 0.31' (-0.225) roughly cancel 'LHA < 0.372' (+0.457), 'girth < 0.0859' (+0.284), 'e2 < 0.0279' (65%, +0.229) and 'mass < 101' (82%, +0.215). The value 0.239 (on for 58%) adds 0.142 to W. The formula calls them Z (56%).
- **medium-mass (78 GeV), narrow, pT spread over several particles** — 8.1% of jets, neuron 0.52. Mostly W (64%) with about 10% each of gluon, quark and Z; mass about 78 GeV and width 0.0057, a harder leading particle (267 GeV) and 47% of the pT at 0.025-0.05 from the axis. 'mass < 92.9' (+0.87), 'mass < 101' (+0.539), 'LHA < 0.372' (+0.496) and 'mass_over_sum_pt_sq < 0.00818' (+0.396) are partly cancelled by 'girth2_top20 < 0.00754' (-0.628), 'lam1 < 0.00672' (-0.602), 'e2 < 0.0388' (-0.536), 'girth < 0.0768' (-0.501) and 'mass_top50 < 86.4' (-0.302), which pass here but not for the clean W group. The value 0.524 adds 0.311 to W and removes 0.131 from q. The formula calls them W (75%).
- **light (35 GeV), very narrow, pT spread over several particles** — 7.3% of jets, neuron 0.32. Mostly quark (63%) with 25% gluon; light (about 35 GeV) and very narrow (width 0.0012), a hard leading particle (309 GeV) and 86% of the pT within 0.025 of the axis. 'mass < 92.9' (+3.365), 'mass < 101' (+1.534) and 'e2 < 0.0279' (+1.291) are cancelled by 'lam1 < 0.00672' (-1.978), 'mass_top50 < 86.4' (-1.639), 'girth2_top20 < 0.00754' (-1.416), 'girth < 0.0768' (-1.328), 'mass < 62.5' (-1.266) and 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' (-1.109). The value 0.317 adds 0.188 to W and removes 0.079 from q. The formula calls them q (77%).
- **light (48 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.46. A gluon/quark mixture (48% gluon, 41% quark); light (about 48 GeV) and narrow (width 0.0021), with 76% of the pT within 0.025 of the axis. 'mass < 92.9' (+2.608), 'mass < 101' (+1.232) and 'e2 < 0.0279' (+1.131) against 'lam1 < 0.00672' (-1.75), 'girth2_top20 < 0.00754' (-1.349), 'mass_top50 < 86.4' (-1.26), 'girth < 0.0768' (-1.153) and 'e2 < 0.0388' (-1.095). The value 0.461 adds 0.274 to W and removes 0.115 from q. The formula splits them between g (53%) and q (47%).
- **medium-mass (62 GeV), narrow, pT spread over several particles** — 6.5% of jets, neuron 0.52. Mostly gluon (52%) mixed with 32% quark; mass about 62 GeV and width 0.0035, with 60% of the pT within 0.025 of the axis. 'mass < 92.9' (+1.832), 'mass < 101' (+0.923), 'e2 < 0.0279' (99%, +0.886) and 'girth < 0.0617' (+0.73) are balanced by 'lam1 < 0.00672' (-1.385), 'girth2_top20 < 0.00754' (-1.19), 'e2 < 0.0388' (-0.952), 'girth < 0.0768' (-0.914) and 'mass_top50 < 86.4' (-0.876). The value 0.523 adds 0.311 to W and removes 0.131 from q. The formula calls them g (59%).
- **light (23 GeV), very narrow, pT spread over several particles** — 5.8% of jets, neuron 0.34. Mostly quark (82%); the lightest (about 23 GeV) and narrowest (width 0.0005) jets here, with a very hard leading particle (373 GeV) and 94% of the pT within 0.025 of the axis. 'mass < 92.9' (+4.085), 'mass < 101' (+1.821) and 'e2 < 0.0279' (+1.496) are cancelled by 'lam1 < 0.00672' (-2.136), 'mass_top50 < 86.4' (-2.025), 'mass < 62.5' (-1.839), 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' (-1.596) and 'mass < 80.4' (-1.591). The value 0.338 adds 0.201 to W and removes 0.084 from q. The formula calls them q.
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 5.1% of jets, neuron 0.38. A gluon-led mixture (54% gluon, 22% W, 15% quark); mass about 80 GeV but narrow (width 0.005), total pT 1156 GeV and 57% of the pT within 0.025 of the axis. 'e2 < 0.0279' passes for all (42% elsewhere, +0.821), with 'mass < 92.9' (91%, +0.801), 'LHA < 0.372' (+0.65) and 'girth < 0.0617' (+0.567), against 'girth2_top20 < 0.00754' (-1.056), 'lam1 < 0.00672' (-0.941), 'e2 < 0.0388' (-0.916) and 'girth < 0.0768' (-0.783). The value 0.38 (on for 67%) adds 0.226 to W and removes 0.095 from q. The formula calls them g (63%), with 26% called W.

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

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 22.2% of jets, neuron 0.12. Mostly W (75%); mass about 80 GeV and width 0.0061, ordinary pT sharing, with most pT at 0.025-0.1 from the axis. The m/pT windows almost cancel: 'mass_over_sum_pt < 0.118' (+1.896), 'mass_over_sum_pt < 0.098' (+1.576), 'mass_over_sum_pt < 0.141' (+1.329) and 'mass < 137' (+0.925) against 'mass_over_sum_pt < 0.0905' (-1.63), 'girth2_top50 < 0.00926' (-0.977), 'girth2_top30 < 0.0122' (-0.924), 'mass_top50 < 92.2' (-0.792) and 'mass < 89.7' (98%, -0.77), the last two marking them as below the Z mass. The value is 0.118, on for 25%, removing 0.162 from W. The formula calls them W.
- **very heavy (150 GeV), very wide, pT spread over several particles** — 21.7% of jets, neuron 0.19. Mostly top (74%) with 17% gluon; mass about 150 GeV and width 0.0235, soft leading particle (163 GeV) and pT spread over 0.05-0.2 from the axis. Too heavy for the m/pT windows ('mass_over_sum_pt < 0.118' 15%, 'mass < 137' 33%), so, with the intercept 0.4243, 'max_dr > 0.24' (-0.668) and 'sum_pt_top50 > 976' (52%, -0.246) leave little. The value is 0.194, on for 37%, removing 0.267 from W. The formula calls them t (83%).
- **heavy (91 GeV), average width, pT spread over several particles** — 21.3% of jets, neuron 1.06. Mostly Z (75%); mass about 91 GeV and width 0.008, ordinary pT sharing, with 42% of the pT at 0.05-0.1 from the axis. 'mass_over_sum_pt < 0.118' (+1.359), 'mass_over_sum_pt < 0.141' (+1.088), 'mass < 137' (+0.745), 'mass_over_sum_pt < 0.098' (95%, +0.693) and 'lam2 < 0.0024' (95%, +0.626) outweigh 'girth2_top30 < 0.0122' (-0.672), 'max_dr > 0.24' (84%, -0.501) and 'mass_top40 < 111' (-0.487); 'mass_over_sum_pt < 0.0833' passes for only 5%, so the low-m/pT penalties stay off. The value 1.056, the highest of this neuron, removes 1.451 from W. The formula calls them Z.
- **light (38 GeV), very narrow, pT spread over several particles** — 7.5% of jets, neuron 0.00. Mostly quark (60%) with 27% gluon; light (about 38 GeV) and very narrow (width 0.0014), a hard leading particle (294 GeV) and 84% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-6.89), 'mass < 89.7' (-4.188), 'mass_top50 < 92.2' (-3.432), 'mass < 91' (-3.312) and 'mass_over_sum_pt < 0.0833' (-2.618) outweigh 'mass_over_sum_pt < 0.098' (+4.778), 'mass_over_sum_pt < 0.118' (+3.817) and 'mass_top50 < 89.1' (+3.747). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q (75%).
- **medium-mass (52 GeV), very narrow, pT spread over several particles** — 6.8% of jets, neuron 0.00. An even gluon/quark mixture (44% gluon, 43% quark); mass about 52 GeV and width 0.0026, with 70% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-5.124), 'mass < 89.7' (-3.045), 'mass_top50 < 92.2' (-2.611), 'mass < 91' (-2.43) and 'girth2_top50 < 0.00926' (-2.032) outweigh 'mass_over_sum_pt < 0.098' (+3.703), 'mass_over_sum_pt < 0.118' (+3.173) and 'mass_top50 < 89.1' (+2.797). The neuron is 0 for all of them and adds nothing to the scores. The formula splits them between q (51%) and g (49%).
- **light (24 GeV), very narrow, pT spread over several particles** — 6.7% of jets, neuron 0.00. Mostly quark (82%); the lightest (about 24 GeV) and narrowest (width 0.0006) jets here, with a very hard leading particle (369 GeV) and 93% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-8.587), 'mass < 89.7' (-5.259), 'mass_top50 < 92.2' (-4.245), 'mass < 91' (-4.138) and 'mass < 80.4' (-3.403) outweigh 'mass_over_sum_pt < 0.098' (+5.81), 'mass_top50 < 89.1' (+4.687) and 'mass_over_sum_pt < 0.118' (+4.437). The neuron is 0 for all of them and adds nothing to the scores. The formula calls them q.
- **medium-mass (66 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.00. Mostly gluon (42%) mixed with 31% quark and 14% W; mass about 66 GeV and width 0.0043, with 51% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-3.238), 'mass < 89.7' (-1.878), 'mass_top50 < 92.2' (-1.765) and 'girth2_top50 < 0.00926' (-1.557) balance 'mass_over_sum_pt < 0.098' (+2.555), 'mass_over_sum_pt < 0.118' (+2.484), 'mass_top50 < 89.1' (+1.819) and 'mass_over_sum_pt < 0.141' (+1.593), leaving the sum just below zero. The value is 0.004, on for 1%, so it adds essentially nothing. The formula calls them g (51%).
- **medium-mass (87 GeV), narrow, pT spread over several particles, high pT** — 3.1% of jets, neuron 0.56. A gluon/W/Z mixture (51% gluon, 28% W, 16% Z); mass about 87 GeV but narrow (width 0.0044), total pT 1318 GeV and a hard leading particle (294 GeV). 'log_sum_pt > 6.94' passes for all (43% elsewhere) and adds 2.98, and with 'mass_over_sum_pt < 0.098' (+2.496), 'mass_over_sum_pt < 0.118' (+2.449), 'mass_over_sum_pt < 0.141' (+1.577) and 'lam1 < 0.00726' (+1.177) it outweighs 'mass_over_sum_pt < 0.0905' (-3.141), 'sum_pt_top50 > 976' (-2.97), 'girth2_top50 < 0.00926' (-1.52) and 'girth2_top30 < 0.0122' (-1.225). The value 0.56 (on for 62%) removes 0.771 from W. The formula calls them g (56%), with 29% called W.
- **very heavy (161 GeV), wide, pT spread over several particles, high pT** — 2.8% of jets, neuron 0.79. Mostly gluon (72%) with 19% top; mass about 161 GeV, width 0.0156 and total pT 1329 GeV, with 33% of the pT at 0.05-0.1 from the axis. 'log_sum_pt > 6.94' (+3.091) cancels 'sum_pt_top50 > 976' (-2.893); 'mass_over_sum_pt < 0.141' (77%, +0.518) and 'mass_over_sum_pt < 0.118' (51%, +0.477) pass because the pT is high, against 'max_dr > 0.24' (96%, -0.587), while the low-mass tests never pass. The value 0.794 (on for 87%) removes 1.092 from W. The formula calls them g (85%).
- **medium-mass (55 GeV), very narrow, pT spread over several particles, high pT** — 1.9% of jets, neuron 0.00. Mostly gluon (86%); mass about 55 GeV and narrow (width 0.0016), a high total pT (1405 GeV), a hard leading particle (330 GeV) and 79% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-6.58), 'sum_pt_top50 > 976' (-3.807), 'mass < 89.7' (-2.806) and 'mass_top50 < 92.2' (-2.519) outweigh 'mass_over_sum_pt < 0.098' (+4.589), 'log_sum_pt > 6.94' (+3.767), 'mass_over_sum_pt < 0.118' (+3.704) and 'mass_top50 < 89.1' (+2.69). The value is 0.001, on for 0.3%, so it adds essentially nothing. The formula calls them g (92%).

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

- **medium-mass (87 GeV), average width, pT spread over several particles** — 30.4% of jets, neuron 0.12. A Z/W mixture (44% Z, 42% W); mass about 87 GeV and width 0.0071, with half the pT at 0.05-0.1 from the axis and only 3% inside 0.025. With the intercept -0.6341, 'girth < 0.121' (+0.884), 'lam2 < 0.00178' (94%, +0.478), 'log_sum_pt < 7.02' (94%, +0.289) and 'e2 < 0.0408' (78%, +0.287) are cancelled by 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-0.761), 'lam1 < 0.0165' (-0.629) and 'z_dr_0_0p05 < 0.846 and z_dr_0p2_0p4 < 0.0034' (55%, -0.267); the dense-core tests such as 'girth2_top5 < 0.0015' pass for only 3%. The value is 0.116, on for 35%, so it adds little. The formula splits them between Z (46%) and W (45%).
- **very heavy (159 GeV), very wide, pT spread over several particles** — 18.3% of jets, neuron 0.10. Mostly top (73%) with 18% gluon; mass about 159 GeV and width 0.024, soft leading particle (169 GeV) and pT spread over 0.05-0.2 from the axis. The core tests mostly fail ('girth < 0.121' 31%, 'e2 < 0.0408' 17%), so only 'log_sum_pt < 7.02' (86%, +0.321) works against the intercept, with 'sum_pt < 1e+03' (42%, -0.232) and 'sum_pt_top30 < 1.04e+03' (-0.149) pulling down. The value is 0.098, on for 16%, so it adds little. The formula calls them t.
- **light (32 GeV), very narrow, pT spread over several particles** — 13.8% of jets, neuron 0.83. Mostly quark (67%) with 22% gluon; light (about 32 GeV) and very narrow (width 0.0009), a hard leading particle (343 GeV) and 90% of the pT within 0.025 of the axis. Every core test passes, in large pairs that partly cancel: 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (+3.402) against 'e2_sq < 0.00694' (-3.117), and 'girth < 0.121' (+2.052), 'girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194' (+1.966) and 'e2 < 0.0408' (+1.773) against 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-1.815), 'girth2_top5 < 0.0015' (-1.527), 'girth < 0.0566' (-1.262) and 'mass_top50 < 71.8' (-0.688). The value 0.834, the highest of this neuron, adds 0.469 to Z and removes 0.313 from t and 0.156 from W. The formula calls them q (80%).
- **medium-mass (56 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 0.77. Mostly gluon (49%) mixed with 37% quark; mass about 56 GeV and width 0.0026, with 73% of the pT within 0.025 of the axis. The same core pairs with smaller amounts: 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (+2.251) and 'e2_sq < 0.00694' (-2.238) cancel, and 'girth < 0.121' (+1.779), 'girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194' (+1.549) and 'e2 < 0.0408' (+1.517) outweigh 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-1.438), 'girth2_top5 < 0.0015' (-1.28) and 'lam1 < 0.0165' (-0.914). The value 0.765 adds 0.43 to Z and removes 0.287 from t. The formula calls them g (55%), with 44% called q.
- **heavy (95 GeV), average width, pT spread over several particles** — 8.3% of jets, neuron 0.60. Mostly Z (43%) mixed with 23% gluon and 18% W; mass about 95 GeV and width 0.0082, a hard leading particle (267 GeV) and 57% of the pT at 0.025-0.05 from the axis. 'girth < 0.121' (+1.166), 'e2 < 0.0408' (+0.783), 'z_dr_0p1_0p2 < 0.187' (+0.371) and 'lam2 < 0.00178' (83%, +0.363) outweigh 'girth2_top30 < 0.0122 and z_dr_0p2_0p4 > 0.0265' (99%, 14% elsewhere, -0.864), 'lam1 < 0.0165' (-0.588), 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (94%, -0.384) and 'girth2_top5 < 0.0015' (58%, -0.35). The value 0.6 (on for 81%) adds 0.337 to Z and removes 0.225 from t. The formula calls them Z (47%).
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 6.6% of jets, neuron 0.62. A W/gluon mixture (33% each) with about 15% each of quark and Z; mass about 80 GeV, width 0.0056, a hard leading particle (276 GeV) and 64% of the pT within 0.025 of the axis. 'girth < 0.121' (+1.522), 'e2 < 0.0408' (+1.199) and 'girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194' (+1.156) outweigh 'girth2_top5 < 0.0015' (-1.154), 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-0.896), 'lam1 < 0.0165' (-0.742) and 'e2_sq < 0.00694' (90%, -0.733, partly given back by its companion, +0.564). The value 0.624 (on for 87%) adds 0.351 to Z and removes 0.234 from t. The formula calls them W (41%), with 38% called g.
- **medium-mass (74 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.57. A W-led mixture (39% W, 28% gluon, 20% quark); mass about 74 GeV and width 0.0044, total pT 1120 GeV, with 45% of the pT at 0.025-0.05 and 30% at 0.05-0.1 from the axis. 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (+1.397) and 'e2_sq < 0.00694' (-1.32) cancel; 'girth < 0.121' (+1.284), 'e2 < 0.0408' (+0.896), 'lam2 < 0.00178' (96%, +0.423) and 'lam1 < 0.00619' (+0.41) outweigh 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-1.092) and 'lam1 < 0.0165' (-0.805). The value 0.57 (on for 88%) adds 0.321 to Z and removes 0.214 from t. The formula calls them W (42%).
- **heavy (117 GeV), very wide, pT spread over several particles** — 5.0% of jets, neuron 0.08. Mostly top (69%) with 19% gluon; mass about 117 GeV and width 0.0186, a low total pT (892 GeV) and soft leading particle (154 GeV). 'sum_pt < 1e+03' (-2.165) and 'sum_pt < 986' (+1.844) nearly cancel, and 'log_sum_pt < 7.02' (+0.819) and 'girth < 0.121' (60%, +0.419) do not lift it far above the intercept; 'sum_pt_top50 < 934' passes for 98% (5% elsewhere). The value is 0.078, on for 15%, so it adds little. The formula calls them t (83%).
- **light (40 GeV), very narrow, pT spread over several particles, low pT** — 1.3% of jets, neuron 0.04. A quark/gluon mixture (51% quark, 42% gluon); mass about 40 GeV and width 0.0025, a low total pT (843 GeV) and 75% of the pT within 0.025 of the axis. The core tests work as in the light groups ('girth < 0.121' +1.825, 'girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194' +1.59, 'e2 < 0.0408' +1.512 against 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' -1.497), but the low pT adds 'sum_pt < 1e+03' (-3.13, only partly repaid by 'sum_pt < 986' +2.811), 'girth2_top5 < 0.00716 and sum_pt_top40 < 985' (-0.673), 'mass_top50 < 71.8' (-0.556) and 'sum_pt_top50 < 934' (-0.268). The value is 0.039, on for 17%, so it adds little. The formula splits them between q (53%) and g (46%).
- **medium-mass (70 GeV), wide, pT spread over several particles, low pT** — 0.9% of jets, neuron 0.04. Mostly gluon (50%) mixed with 27% quark and 23% top; mass about 70 GeV, width 0.0128, the lowest total pT here (671 GeV) and a soft leading particle (125 GeV). 'sum_pt < 1e+03' (-6.508) and 'sum_pt < 986' (+6.19) nearly cancel; 'log_sum_pt < 7.02' (+1.887) and 'girth < 0.121' (79%, +0.897) are taken back by 'girth2_top5 < 0.00716 and sum_pt_top40 < 985' (71%, -0.856) and 'sum_pt_top50 < 934' (-0.789). The value is 0.044, on for 8%, so it adds little. The formula calls them g (62%).
