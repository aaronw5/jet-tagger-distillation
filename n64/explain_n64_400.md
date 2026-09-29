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

- **medium-mass (78 GeV), narrow, pT spread over several particles** — 28.0% of jets, neuron 1.45. Mostly W (65%), with 13% gluon and a few Z and top jets; mass about 78 GeV and width 0.0057, narrower than the 0.0093 average, with ordinary pT sharing (hardest particle 244 GeV). Only 19.3% of the pT lies within 0.025 of the axis (28.8% for all jets); it sits at 0.025-0.1, as for two resolved prongs. 'mass_over_sum_pt_sq < 0.00818' passes for 99.8% and adds 1.187, while 'girth2_top20 < 0.00754' (always passes, -0.687) and 'girth < 0.0566' (-0.277) take some back; 'mass > 78.3' passes for 66.6% but, sitting just above the cut, costs only 0.275. With the intercept 1.3858 the neuron ends at 1.449 and is on for 96.2%, which adds 1.087 to the W score and takes 1.992 from the Z score. The formula calls them W.
- **light (42 GeV), very narrow, pT spread over several particles** — 25.4% of jets, neuron 1.08. A quark-gluon mixture: 53.8% quark and 34.2% gluon; light (about 42 GeV) and very narrow (width 0.0017), with a hard leading particle (305 GeV against 240 for all jets) and 82% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00818' passes for all and adds a large 3.118, but the narrowness tests that also always pass pull it back: 'girth < 0.0566' (-2.296), 'girth2_top20 < 0.00754' (-1.614), 'lam1 < 0.00591' (-1.007) and 'girth2_top40 < 0.00626' (-0.886), partly returned by 'girth2_top20 < 0.00531' (+1.031), 'girth2_top40 < 0.00626 and girth2_top3 < 0.00292' (+0.866) and 'girth2_top50 < 0.00743' (+0.853). 'mass > 78.3' passes for only 1.5%, so the neuron ends at 1.079 (on for 98%), adding 0.809 to W and taking 1.483 from Z. The formula calls them q.
- **heavy (91 GeV), average width, pT spread over several particles** — 22.0% of jets, neuron 0.09. Mostly Z (75%), with about 9% each of gluon and top jets; mass about 91 GeV and width 0.0080, with ordinary pT sharing and more pT at 0.05-0.1 from the axis (41.5%) than the 26.5% for all jets. 'mass > 78.3' passes for 99.2% and takes away 1.681, and 'mass_over_sum_pt > 0.077' (98.3%) takes 0.526 more; 'mass_over_sum_pt > 0.0833' (+0.511), 'log_sum_pt < 7.02' (+0.302) and 'mass_over_sum_pt_sq < 0.00818' (72.9% pass, +0.233) give only part of it back. The value is 0.086 and the neuron is on for only 34.4%, so it barely moves the scores (+0.064 W, -0.118 Z). The formula calls them Z.
- **very heavy (171 GeV), very wide, pT spread over several particles** — 8.6% of jets, neuron 0.00. Mostly top (89.6%); mass about 171 GeV and broad (width 0.0284, three times the average), with pT shared among many particles (hardest only 161 GeV) and 20% of the pT at 0.2-0.3 from the axis against 4.5% for all jets. All the mass cuts pass and each adds more the heavier the jet: 'mass > 78.3' (-12.159), 'mass > 92.9' (-9.874), 'mass > 91' (-6.847) and 'mass_over_sum_pt > 0.077' (-3.918), with only 'mass_over_sum_pt > 0.0833' (+7.022) pushing back. The sum sits at -24.327, so the neuron is off for every jet and adds nothing to any class score. The formula calls them t.
- **heavy (111 GeV), average width, pT spread over several particles** — 5.0% of jets, neuron 0.00. A gluon-top mixture (40.4% gluon, 39.1% top, 17.1% quark); mass about 111 GeV and width 0.0120, a bit broader than average, with a softer leading particle (192 GeV). The mass cuts all pass here too, 'mass > 78.3' (-4.293), 'mass > 92.9' (-2.298), 'mass > 91' (-1.713) and 'mass_over_sum_pt > 0.077' (-1.373), against +2.137 from 'mass_over_sum_pt > 0.0833' and +0.413 from 'mass > 92.9 and n_dr_0p2_0p4 < 7' (18% pass). The sum is -5.976, so the neuron is on for only 0.1% and adds nothing to the scores. The formula calls them g slightly more often than t (45.9% against 43.5%).
- **very heavy (153 GeV), very wide, pT spread over several particles** — 4.8% of jets, neuron 0.00. Mostly top (74.1%) with 18.6% gluon; mass about 153 GeV, broad (width 0.0234), pT shared out (hardest 168 GeV) and much of it at 0.1-0.3 from the axis. 'mass > 78.3' (-9.810), 'mass > 92.9' (-7.612), 'mass > 91' (-5.314) and 'mass_over_sum_pt > 0.077' (-3.243) all pass, only partly offset by 'mass_over_sum_pt > 0.0833' (+5.722). The sum is -18.955, the neuron is off and adds nothing to the scores. The formula calls them t.
- **heavy (132 GeV), wide, pT spread over several particles** — 4.2% of jets, neuron 0.00. A top-gluon mixture (53.3% top, 33.1% gluon, 12% quark); mass about 132 GeV, width 0.0170, with pT shared out (hardest 181 GeV) and 37% of it at 0.05-0.1 from the axis. 'mass > 78.3' (-7.012), 'mass > 92.9' (-4.916), 'mass > 91' (-3.487) and 'mass_over_sum_pt > 0.077' (-2.272) pass for all, offset only by 'mass_over_sum_pt > 0.0833' (+3.852) and, for 8.2%, 'mass > 92.9 and n_dr_0p2_0p4 < 7' (+0.260). The sum is -12.351, the neuron is off and adds nothing to the scores. The formula calls them t.
- **very heavy (200 GeV), very wide, pT spread over several particles** — 1.4% of jets, neuron 0.00. A top-gluon mixture (47% top, 42.9% gluon); mass about 200 GeV, the broadest here (width 0.0347), with 26% of the pT at 0.2-0.3 from the axis and a soft leading particle (155 GeV). The mass cuts grow with the mass and dominate: 'mass > 78.3' (-15.878), 'mass > 92.9' (-13.456), 'mass > 91' (-9.274), 'mass_over_sum_pt > 0.077' (-4.649), against +8.430 from 'mass_over_sum_pt > 0.0833'. The sum is -33.434, the neuron is off and adds nothing to the scores. The formula calls them t.
- **very heavy (156 GeV), wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00. Mostly gluon (60.8%) with 32.8% top; mass about 156 GeV, width 0.0156, high total pT (1304 GeV) and pT concentrated at 0.1-0.2 from the axis (41% at 0.1-0.15) with almost nothing beyond 0.2 (1.3% at 0.2-0.3). What sets them apart is 'mass > 92.9 and n_dr_0p2_0p4 < 7', which passes for all (few particles at 0.2-0.4) and adds 11.038; still, 'mass > 78.3' (-10.153), 'mass > 92.9' (-7.941), 'mass > 91' (-5.537) and 'mass_over_sum_pt > 0.077' (-1.981) win. The sum is -9.892, the neuron is off and adds nothing to the scores. The formula calls them g.
- **very heavy (257 GeV), very wide, pT spread over several particles, high pT** — 0.3% of jets, neuron 0.00. Mostly gluon (73%) with 17% top; mass about 257 GeV, very broad (width 0.0361), very high total pT (1395 GeV) and 28% of the pT at 0.2-0.3 from the axis. The mass cuts reach their largest values, 'mass > 78.3' (-23.381), 'mass > 92.9' (-20.684), 'mass > 91' (-14.172), against +8.684 from 'mass_over_sum_pt > 0.0833'. The sum is -52.803, the neuron is off and adds nothing to the scores. The formula calls them g.

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

- **medium-mass (86 GeV), average width, pT spread over several particles** — 27.3% of jets, neuron 1.32. A W-Z mixture, mostly W (42.7%) with 39.8% Z; mass about 86 GeV and width 0.0074, with a fairly hard leading particle (273 GeV) and pT at 0.025-0.1 from the axis; fewer particles than average ('n_particles > 38' passes for only 38.8%). 'z_top50_slots > 0.959' passes for all and removes 2.147, 'max_dr < 0.436' (-0.829) and 'n_pt_above_10 < 31' (-0.747) remove more, while 'LHA < 0.404' (+1.132), 'sum_pt_top2 < 689' (+0.974) and 'log_sum_pt > 6.89' (+0.782) add; with the intercept 1.7525 the value ends at 1.316 (on for 95.5%), the lowest of this neuron, adding 0.823 to g and taking 0.247 from q. The formula calls them W.
- **heavy (119 GeV), wide, pT spread over several particles** — 17.6% of jets, neuron 2.24. Mostly top (53.1%), with 18.6% Z and 13.8% W; mass about 119 GeV and width 0.0161, with pT shared out (hardest particle 168 GeV) and much of it at 0.05-0.15 from the axis. 'sum_pt_top2 < 689' (+1.475), 'sum_pt_top5 < 902' (+0.845), 'n_particles > 38' (99.5% pass, +0.711) and 'sum_pt_top30 < 1.07e+03' (+0.509) add, but 'z_top50_slots > 0.959' (-1.816) and 'z_top20_slots < 0.958' (always passes, -1.298) take off; 'log_sum_pt > 6.89' passes for only 64.9%. The value is 2.243, adding 1.402 to g and taking 0.421 from q. The formula calls them t.
- **light (34 GeV), very narrow, pT spread over several particles** — 14.4% of jets, neuron 2.35. Mostly quark (73.1%) with 12.9% gluon; light (about 34 GeV), very narrow (width 0.0013), a hard leading particle (331 GeV) and 89% of the pT within 0.025 of the axis; few particles ('n_particles > 38' passes for 21.2%). The light-core tests add most: 'mass_top30 < 80.4' (+2.701) and 'LHA < 0.404' (+2.567), but 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.970), 'mass_top50 < 117' (-1.381), 'mass_top20 < 40.2' (93.1% pass, -1.052) and 'z_top50_slots > 0.959' (-2.148) take nearly as much back. The value is 2.349, adding 1.468 to g and taking 0.441 from q. The formula calls them q.
- **heavy (97 GeV), average width, pT spread over several particles** — 10.8% of jets, neuron 3.31. A mixture: 31.9% each W and Z, 15.5% gluon and 12.9% top; mass about 97 GeV, width 0.0087, total pT 1090 GeV, above the 1044 GeV average. The total-pT steps decide it: 'log_sum_pt > 6.89' (+2.424) and 'log_sum_pt > 6.91' (+2.011) pass for all, while 'log_sum_pt > 6.96' passes for 98.9% but adds only -0.483 (just above that cut); 'sum_pt_top50 > 959' (-1.094) and 'z_top50_slots > 0.959' (-1.965) pull down. The value is 3.311, adding 2.069 to g and taking 0.621 from q. The formula calls them W slightly more often than Z.
- **heavy (140 GeV), very wide, pT spread over several particles** — 10.1% of jets, neuron 4.28. Mostly top (56.7%) with 28.1% gluon; mass about 140 GeV and width 0.0219, with pT shared very evenly (hardest only 115 GeV, second 77 GeV) and many particles, spread broadly around the axis. The soft-leader tests all pass: 'sum_pt_top2 < 689' (+1.765), 'sum_pt_top5 < 902' (+1.085), 'n_particles > 38' (+1.043), 'sum_pt_top30 < 1.07e+03' (+0.837) and 'n_particles > 38 and z_top50_slots < 0.985' (99%, +0.715), set apart from the other groups because 'z_top50_slots > 0.959' passes for only 63.3% (-0.381); 'z_top20_slots < 0.958' (-2.753) is the one large debit. The value is 4.275, adding 2.672 to g and taking 0.801 from q. The formula calls them t.
- **medium-mass (59 GeV), narrow, pT spread over several particles** — 6.0% of jets, neuron 4.76. Mostly gluon (54.4%) with 29.8% quark; mass about 59 GeV and width 0.0039, a soft leading particle (162 GeV against 240 for all jets), many particles, and 49% of the pT within 0.025 of the axis. 'mass_top30 < 80.4' (+2.117), 'LHA < 0.404' (+1.804), 'sum_pt_top2 < 689' (+1.502), 'n_particles > 38' (98.8%, +0.756) and 'n_particles > 38 and dr_0 < 0.144' (+0.568) add, against 'z_top50_slots > 0.959' (-1.637), 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.467) and 'z_top20_slots < 0.958' (-1.359). The value is 4.765, adding 2.978 to g and taking 0.893 from q. The formula calls them g.
- **heavy (117 GeV), average width, pT spread over several particles, high pT** — 5.1% of jets, neuron 5.77. Mostly gluon (54.3%), mixed with top (14.7%), W (13.2%) and Z (11.7%); mass about 117 GeV, width 0.0108, total pT 1202 GeV. The total-pT steps are large here: 'log_sum_pt > 6.89' (+4.782) and 'log_sum_pt > 6.91' (+4.353), only partly cancelled by 'sum_pt_top50 > 959' (-1.908), 'log_sum_pt > 6.96' (-1.853), 'log_sum_pt > 6.81' (-1.382) and 'log_sum_pt > 6.99' (-0.826), all of which pass. The value is 5.767, adding 3.604 to g and taking 1.081 from q. The formula calls them g.
- **light (47 GeV), very narrow, pT spread over several particles** — 4.2% of jets, neuron 5.81. Mostly gluon (66.7%) with 28.4% quark; mass about 47 GeV, very narrow (width 0.0018), total pT 1158 GeV with a hard leading particle (293 GeV) and 80% of the pT within 0.025 of the axis. Both kinds of boost add up: the total-pT steps 'log_sum_pt > 6.89' (+3.880) and 'log_sum_pt > 6.91' (+3.457) and the light-core tests 'mass_top30 < 80.4' (+2.514) and 'LHA < 0.404' (+2.364), against 'mass_top30 < 80.4 and mass_top5 < 68.4' (-1.809), 'sum_pt_top50 > 959' (-1.676) and 'log_sum_pt > 6.96' (-1.329). The value is 5.813, adding 3.633 to g and taking 1.09 from q. The formula calls them g.
- **heavy (97 GeV), narrow, pT spread over several particles, high pT** — 3.5% of jets, neuron 6.99. Mostly gluon (74.9%); mass about 97 GeV, width 0.0064, very high total pT (1363 GeV), with 46% of the pT within 0.025 of the axis. 'log_sum_pt > 6.89' (+7.835) and 'log_sum_pt > 6.91' (+7.384) grow with the pT and outweigh 'log_sum_pt > 6.96' (-3.627), 'sum_pt_top50 > 959' (-3.326), 'log_sum_pt > 6.81' (-2.003) and 'log_sum_pt > 6.99' (-1.852). The value is 6.993, adding 4.37 to g and taking 1.311 from q. The formula calls them g.
- **heavy (111 GeV), narrow, pT spread over several particles, high pT** — 1.1% of jets, neuron 7.56. Mostly gluon (82.4%); mass about 111 GeV, width 0.0059, the highest total pT (1667 GeV) and a very hard leading particle (355 GeV). 'log_sum_pt > 6.89' (+12.632) and 'log_sum_pt > 6.91' (+12.148) are huge, and 'log_sum_pt > 6.96' (-6.413), 'sum_pt_top50 > 959' (-5.878), 'log_sum_pt > 6.99' (-3.464), 'log_sum_pt > 6.81' (-2.980) and 'log_sum_pt > 7.14' (-1.383) cancel most of it. The value is 7.563, the neuron's largest (well under 16), adding 4.727 to g and taking 1.418 from q. The formula calls them g.

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

- **heavy (96 GeV), average width, pT spread over several particles** — 22.1% of jets, neuron 1.07. Mostly Z (65.6%), with 14.2% gluon and 11.9% top; mass about 96 GeV and width 0.0086, with ordinary pT sharing and 46% of the pT at 0.05-0.1 from the axis (only 3.7% within 0.025, against 28.8% for all jets). 'mass < 121' (94.7% pass, +0.689), 'girth2_top15 < 0.0156' (+0.555), 'D2 < 6.92' (+0.539), 'mass < 101' (79.9%, +0.510) and 'n_dr_0p2_0p4 < 15' (+0.396) add, against 'LHA > 0.115' (-1.965), 'girth < 0.121' (-1.047) and 'mass_top30 < 153' (-0.628). With the intercept 2.5973 the value is 1.07 (on for 81.8%), taking 0.936 from g and 1.137 from q and adding 0.368 to W and 0.201 to t; this neuron does not enter the Z score. The formula calls them Z.
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 17.2% of jets, neuron 1.33. Mostly W (81%); mass about 80 GeV, width 0.0062, with 47% of the pT at 0.05-0.1 from the axis and almost none beyond 0.2 (0.57% at 0.2-0.3). Both mass windows pass for all, 'mass < 101' (+1.393) and 'mass < 121' (+1.117), helped by 'girth2_top15 < 0.0156' (+0.656), 'D2 < 6.92' (+0.570) and 'n_dr_0p2_0p4 < 15' (+0.522); 'LHA > 0.115' (-1.796), 'girth < 0.121' (-1.277), 'mass_top30 < 153' (-0.751), 'mass_top40 < 94.6' (-0.698) and 'mass < 86.4' (-0.363) take off. The value is 1.333, the neuron's largest (on for 96.6%), taking 1.166 from g and 1.416 from q and adding 0.458 to W and 0.25 to t. The formula calls them W.
- **very heavy (173 GeV), very wide, pT spread over several particles** — 11.3% of jets, neuron 0.75. Mostly top (84.8%) with 10.9% gluon; mass about 173 GeV, broad (width 0.0299), pT shared among many particles (hardest 156 GeV) and 21% of it at 0.2-0.3 from the axis. The mass windows fail here; instead 'width > 0.00961' passes for all and adds 3.381, nearly cancelling 'LHA > 0.115' (-3.297), while 'lam1 > 0.00767' (-1.721), 'mass_top20 > 104' (-0.869) and 'mass_over_sum_pt > 0.098' (-0.685) are offset by 'e2 > 0.0368' (+0.983), 'mass > 144' (+0.623) and 'D2 < 6.92' (+0.491). The value is 0.754 (on for 91.6%), taking 0.66 from g and 0.801 from q and adding 0.259 to W and 0.141 to t. The formula calls them t.
- **medium-mass (85 GeV), narrow, pT spread over several particles** — 10.7% of jets, neuron 0.74. A mixture: 34.2% W, 29.2% Z and 23.3% gluon; mass about 85 GeV, width 0.0064, total pT 1088 GeV with a hard leading particle (294 GeV) and 42% of the pT within 0.025 and 40% at 0.025-0.05 from the axis, i.e. close-together prongs. 'mass < 101' (+1.074), 'mass < 121' (+0.975) and 'girth2_top15 < 0.0156' (+0.773) add, but because the pT is so central 'girth < 0.121' removes more here (-1.790) while 'LHA > 0.115' removes less (-1.079), and 'mass_top30 < 153' (-0.753), 'mass_top40 < 94.6' (-0.593) and 'girth2_top15 < 0.00728' (-0.546) take off too. The value is 0.737 (on for 80.9%), taking 0.645 from g and 0.783 from q and adding 0.253 to W. The formula calls them W.
- **very heavy (141 GeV), very wide, pT spread over several particles** — 8.8% of jets, neuron 0.23. Mostly top (63.4%) with 25.5% gluon; mass about 141 GeV, width 0.0189, pT shared out (hardest 171 GeV) and 38% of it at 0.05-0.1 from the axis. 'LHA > 0.115' (-2.677) is only partly offset by 'width > 0.00961' (+1.548), and 'lam1 > 0.00767' (-0.855) and 'n_particles > 29 and n_dr_0_0p05 < 25' (-0.440) outweigh 'D2 < 6.92' (+0.475) and 'e2 > 0.0368' (81.2%, +0.384); the mass windows mostly fail. The sum is near zero, so the value is 0.226 and the neuron is on for 49.2%, taking 0.198 from g and 0.24 from q. The formula calls them t.
- **medium-mass (51 GeV), very narrow, pT spread over several particles** — 8.1% of jets, neuron 0.01. A gluon-quark mixture (46.8% gluon, 40.5% quark); mass about 51 GeV, narrow (width 0.0025), hardest particle 275 GeV and 74% of the pT within 0.025 of the axis. All the 'mass below' tests pass, and each adds more the lighter the jet: 'mass < 101' (+3.319), 'mass < 121' (+1.914), 'mass_top40 < 83.3' (+1.781), but the lower cuts 'mass < 86.4' (-2.038), 'mass_top40 < 94.6' (-2.028), 'mass < 80.8' (-0.964) and 'girth < 0.121' (-2.330) take back more. The sum is -0.923, so the neuron is on for only 5% and adds essentially nothing to the scores. The formula splits them almost evenly between g and q.
- **medium-mass (68 GeV), narrow, pT spread over several particles** — 7.4% of jets, neuron 0.08. Mostly gluon (48.7%), mixed with quark (27%) and W (13.1%); mass about 68 GeV, width 0.0042, with 53% of the pT within 0.025 of the axis. As for the other light groups, 'mass < 101' (+2.218), 'mass < 121' (+1.458) and 'mass_top40 < 83.3' (+1.119) are outweighed by 'girth < 0.121' (-1.982), 'mass_top40 < 94.6' (-1.453), 'mass < 86.4' (-1.080) and 'mass_top30 < 153' (-0.966). The sum is -0.327, so the value is 0.081 (on for 25.9%) and the effect on the scores is small (-0.071 g, -0.087 q). The formula calls them g.
- **light (29 GeV), very narrow, pT spread over several particles** — 7.1% of jets, neuron 0.00. Mostly quark (77.8%); very light (about 29 GeV), extremely narrow (width 0.0009), a very hard leading particle (386 GeV) and 94% of the pT within 0.025 of the axis. 'mass < 101' (+4.775), 'mass_top40 < 83.3' (+2.653) and 'mass < 121' (+2.517) are cancelled by the lower mass cuts 'mass < 86.4' (-3.305), 'mass_top40 < 94.6' (-2.783) and 'mass < 80.8' (-1.679) plus 'girth < 0.121' (-2.719); 'LHA > 0.115' passes for only 9.3%. The sum is -0.690, the neuron is on for 0.4% and adds nothing to the scores. The formula calls them q.
- **light (37 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 0.00. A quark-gluon mixture (51.4% quark, 40.8% gluon); mass about 37 GeV, width 0.0014, 76% of the pT within 0.025 of the axis but an ordinary leading particle (227 GeV). The same light-jet balance as the neighbouring groups, 'mass < 101' (+4.262) against 'mass < 86.4' (-2.859), 'mass_top40 < 94.6' (-2.588) and 'girth < 0.121' (-2.437), but here 'mass_top40 < 83.3 and D2 < 6.92' passes for all and removes 2.259 (it passes for only 22.3% of the group above). The sum is -2.317, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (22 GeV), very narrow, pT spread over several particles** — 1.8% of jets, neuron 0.00. Mostly quark (83.7%); very light (about 22 GeV), the narrowest here (width 0.0005), with a hard leading particle (356 GeV) and 91% of the pT within 0.025 of the axis. 'mass < 74.3 and max_dr < 0.394' passes for all and removes 3.987, which sets this group apart; with 'mass < 86.4' (-3.738) and 'mass_top40 < 94.6' (-3.107) it outweighs 'mass < 101' (+5.272), 'mass_top40 < 83.3' (+3.026) and 'mass < 121' (+2.723). The sum is -3.470, the neuron is off and adds nothing to the scores. The formula calls them q.

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

- **medium-mass (68 GeV), narrow, pT spread over several particles** — 43.3% of jets, neuron 2.02. The largest group (43.3%), a W-Z-quark mixture (29.4% W, 28.6% Z, 28.3% quark); mass about 68 GeV, width 0.0052, total pT 998 GeV (a bit below the 1044 GeV average) with a fairly hard leading particle (261 GeV). 'z_top50_slots > 0.959' (+1.292), 'log_sum_pt > 6.86' (90.9% pass, +0.651), 'sum_pt_top40 > 907' (+0.556) and 'n_particles < 64 and e2 < 0.0388' (+0.523) add, against 'z_top40_slots > 0.93' (-1.700); 'log_sum_pt > 6.91' passes for only 64.1% and costs just 0.403. With the intercept 0.6732 the value is 2.015, the neuron's largest (on for 99.9%), adding 1.165 to W and 1.385 to Z and taking 0.63 from g and 0.944 from t. The formula calls them q slightly more often than W.
- **medium-mass (75 GeV), narrow, pT spread over several particles** — 18.4% of jets, neuron 1.38. A mixture: 29.8% W, 29.2% Z, 20% gluon and 18.2% quark; mass about 75 GeV, width 0.0053, total pT 1072 GeV with a hard leading particle (284 GeV). Here 'log_sum_pt > 6.91' passes for all and removes 2.304, and 'z_top40_slots > 0.93' removes 1.540, against 'log_sum_pt > 6.86' (+1.398), 'z_top50_slots > 0.959' (+1.206) and 'sum_pt_top40 > 907' (+0.928). The value is 1.377 (on for 99.5%), adding 0.796 to W and 0.946 to Z and taking 0.43 from g and 0.645 from t. The formula calls them W, closely followed by Z.
- **very heavy (159 GeV), very wide, pT spread over several particles** — 12.2% of jets, neuron 0.43. Mostly top (87.2%); mass about 159 GeV, broad (width 0.0260), pT shared out (hardest 161 GeV) and spread to 0.1-0.3 from the axis. The mass tests pass for all: 'mass_top30 > 68.3' (+2.505) and 'mass > 64.5' (+1.533) against 'mass_over_sum_pt > 0.0905' (-2.685), 'mass_top50 > 91.2' (-1.469) and 'mass_over_sum_pt > 0.08' (-0.860), and 'z_top40_slots > 0.93' (74.6%, -0.757) takes more. The value is 0.434 (on for 64.2%), adding 0.251 to W and 0.298 to Z and taking 0.203 from t. The formula calls them t.
- **heavy (95 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 1.18. A mixture, mostly gluon (35.5%) with 32% top, 14% quark and 11.7% Z; mass about 95 GeV, width 0.0105, low total pT (958 GeV) shared among many particles (hardest 142 GeV). The low pT keeps 'log_sum_pt > 6.91' mostly off (34.3% pass, -0.248), and partial passes of 'z_top50_slots > 0.959' (+0.632), 'mass > 64.5' (+0.508) and 'mass_top30 > 68.3' (+0.448) outweigh 'mass_over_sum_pt > 0.0905' (60.2%, -0.571) and 'z_top40_slots > 0.93' (69.1%, -0.529). The value is 1.179 (on for 91.5%), adding 0.682 to W and 0.811 to Z and taking 0.369 from g and 0.553 from t. The formula calls them g.
- **medium-mass (73 GeV), narrow, pT spread over several particles** — 7.1% of jets, neuron 0.57. Mostly gluon (55.8%), with W, Z and quark at 13-15% each; mass about 73 GeV, width 0.0043, total pT 1179 GeV and 47% of the pT within 0.025 of the axis. 'log_sum_pt > 6.91' removes 5.606 and 'z_top40_slots > 0.93' 1.303, against 'log_sum_pt > 6.86' (+2.507), 'sum_pt_top40 > 907' (+1.498), 'z_top50_slots > 0.959' (+1.075) and 'log_sum_pt > 6.99' (+0.824). The value is 0.566 (on for 86.1%), adding 0.327 to W and 0.389 to Z and taking 0.177 from g and 0.266 from t. The formula calls them g.
- **heavy (96 GeV), narrow, pT spread over several particles, high pT** — 3.9% of jets, neuron 0.24. Mostly gluon (73.1%); mass about 96 GeV, width 0.0062, high total pT (1346 GeV) and 44% of the pT within 0.025 of the axis. 'log_sum_pt > 6.91' grows with pT and removes 10.163, against 'log_sum_pt > 6.86' (+4.037), 'sum_pt_top40 > 907' (+2.384), 'log_sum_pt > 6.99' (+2.136) and 'sum_pt > 908 and girth2_top2 < 0.014' (+1.309), with 'log_sum_pt > 6.92 and girth2_top2 < 0.014' (-1.567) also taking off. The sum is -0.087, so the value is 0.239 (on for 44.6%) and it moves the scores little (+0.165 Z, -0.112 t). The formula calls them g.
- **very heavy (166 GeV), very wide, pT spread over several particles** — 2.5% of jets, neuron 0.28. A top-gluon mixture (50% top, 41.5% gluon); mass about 166 GeV, width 0.0208, total pT 1157 GeV. Both the pT and the mass tests fire: 'log_sum_pt > 6.91' (-4.915), 'mass_over_sum_pt > 0.0905' (-2.017) and 'mass_top50 > 91.2' (-1.581) against 'mass_top30 > 68.3' (+2.592), 'log_sum_pt > 6.86' (+2.275), 'mass > 64.5' (+1.635) and 'sum_pt_top40 > 907' (+1.143). The sum is near zero (-0.126), so the value is 0.277 (on for 50.4%) with small effects on the scores. The formula calls them t.
- **very heavy (182 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 0.23. Mostly top (74.8%) with 18% gluon; mass about 182 GeV, very broad (width 0.0350), low total pT (974 GeV) shared out (hardest 140 GeV) and 27% of it at 0.2-0.3 from the axis. The very large m/pT turns on 'mass_over_sum_pt > 0.171' (+5.443), mostly cancelled by 'mass_over_sum_pt_sq > 0.0292' (-3.840); 'mass_over_sum_pt > 0.0905' (-3.692), 'mass_top50 > 91.2' (-1.937) and 'mass_over_sum_pt > 0.08' (-1.140) balance 'mass_top30 > 68.3' (+3.057) and 'mass > 64.5' (+1.901). The sum is near zero, so the value is 0.234 (on for 56.3%) with small effects on the scores. The formula calls them t.
- **heavy (109 GeV), narrow, pT spread over several particles, high pT** — 1.2% of jets, neuron 0.13. Mostly gluon (82.3%); mass about 109 GeV, width 0.0058, the highest total pT here (1648 GeV) with a very hard leading particle (362 GeV). 'log_sum_pt > 6.91' removes 17.064 and 'log_sum_pt > 6.92 and girth2_top2 < 0.014' 2.729, against 'log_sum_pt > 6.86' (+6.354), 'sum_pt_top40 > 907' (+4.129), 'log_sum_pt > 6.99' (+4.124) and 'sum_pt > 908 and girth2_top2 < 0.014' (+2.262). The sum is -0.765, so the neuron is on for only 19.5% (value 0.126) and barely touches the scores. The formula calls them g.
- **very heavy (209 GeV), very wide, pT spread over several particles** — 0.5% of jets, neuron 0.19. Mostly top (59%) with 31.9% gluon; mass about 209 GeV, the broadest here (width 0.0449), with 39% of the pT at 0.2-0.3 from the axis and a soft leading particle (126 GeV). 'mass_over_sum_pt > 0.171' adds 13.740 and 'mass_over_sum_pt_sq > 0.0292' removes 10.371, both growing with m/pT; 'mass_over_sum_pt > 0.0905' (-4.626), 'mass_top50 > 91.2' (-2.486) and 'mass_over_sum_pt > 0.08' (-1.401) balance 'mass_top30 > 68.3' (+3.682) and 'mass > 64.5' (+2.347). The sum is -0.486, so the value is 0.191 (on for 41%) with small effects on the scores. The formula calls them t.

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

- **heavy (134 GeV), wide, pT spread over several particles** — 34.2% of jets, neuron 0.30. The largest group (34.2%): mostly top (51.4%), with 20.2% Z and 19.5% gluon; mass about 135 GeV, width 0.0185, pT shared out (hardest 188 GeV) and spread to 0.1-0.3 from the axis. Being above most mass cuts, the tests pass only partly: 'mass < 121' (42.1%, +0.316) and 'e2_sq < 0.00961' (27.7%, +0.194) against 'girth < 0.0859' (33.9%, -0.255) and 'z_top50_slots < 0.991' (50.7%, -0.252). With the intercept -0.5094 the value is 0.3, on for 27.1% (the lighter, narrower ones, which include the Z jets), adding 0.272 to Z and taking 0.187 from W. The formula calls them t.
- **medium-mass (82 GeV), narrow, pT spread over several particles** — 17.3% of jets, neuron 0.15. Mostly W (45.8%), mixed with gluon (22.8%) and Z (18%); mass about 82 GeV, width 0.0060, total pT 1077 GeV with a hard core: 32% of the pT within 0.025 and 39% at 0.025-0.05 from the axis. 'e2_sq < 0.00961' (+1.684), 'mass < 121' (+1.270), 'LHA < 0.32' (+0.845) and 'mass < 101' (+0.804) add, but the narrowness tests 'girth < 0.0859' (-1.646), 'girth2_top40 < 0.00803' (-1.360) and 'girth2_top20 < 0.00803' (-1.340) plus 'mass < 91.2' (91.4% pass, -1.341) and 'mass < 82.9' (65.9%, -0.524) take off more. The value is 0.146 (on for 27.8%), a small push towards Z (+0.132) and away from W (-0.091). The formula calls them W.
- **medium-mass (87 GeV), average width, pT spread over several particles** — 13.1% of jets, neuron 1.66. Mostly Z (64.4%) with 27.7% W; mass about 87 GeV, width 0.0073, with 54% of the pT at 0.05-0.1 from the axis and almost none within 0.025 (1.5%). The clean two-prong pattern passes 'mass < 101 and D2 < 1.6' (+2.415), which almost cancels 'mass_top50 < 97.9 and D2 < 1.6' (-2.587); the lift comes from 'e2_sq < 0.00961' (+1.094), 'mass < 121' (+1.093), 'D2 < 1.79 and n_dr_0p2_0p4 < 9' (+0.923) and 'mass < 101 and max_dr < 0.391' (+0.917), while 'mass < 91.2' passes for 73.9% and costs only 0.555, being close to the cut. The value is 1.658, the neuron's largest (on for 73%), adding 1.503 to Z and taking 1.036 from W and 0.466 from t. The formula calls them Z.
- **light (47 GeV), very narrow, pT spread over several particles** — 8.1% of jets, neuron 0.00. A gluon-quark mixture (45.3% gluon, 43.3% quark); mass about 47 GeV, narrow (width 0.0021), with 77% of the pT within 0.025 of the axis. The 'mass below' cuts grow with the distance below them: 'mass < 82.9' (-6.108), 'mass < 91.2' (-6.093) and 'mass_top50 < 71.8' (-4.495), with 'girth2_top40 < 0.00803' (-3.356) and 'girth < 0.0859' (-3.127), far outweighing 'e2_sq < 0.00961' (+3.465), 'mass < 121' (+2.434) and 'mass < 101' (+2.273). The sum is -13.273, the neuron is off and adds nothing to the scores. The formula splits them almost evenly between q and g.
- **medium-mass (79 GeV), narrow, pT spread over several particles** — 7.8% of jets, neuron 0.06. Mostly W (86.5%); mass about 79 GeV, width 0.0058, with 61% of the pT at 0.05-0.1 from the axis and nearly nothing beyond 0.15. The same clean two-prong tests as the Z-like group pass, 'mass < 101 and D2 < 1.6' (+5.161) against 'mass_top50 < 97.9 and D2 < 1.6' (-6.312), with 'mass < 101 and max_dr < 0.391' (+2.029), 'e2_sq < 0.00961' (+1.736), 'mass < 121' (+1.387) and 'D2 < 1.79 and n_dr_0p2_0p4 < 9' (+1.201); but the lower mass now fails the window: 'mass < 91.2' passes for all (-1.731), 'mass < 91.2 and max_dr < 0.391' (-1.379) and 'mass < 82.9' (92.9%, -0.736). The sum is -1.655, so the neuron is on for only 11.1% and barely moves the scores. The formula calls them W.
- **medium-mass (63 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.00. Mostly gluon (52.8%) with 29.4% quark; mass about 63 GeV, width 0.0036, 60% of the pT within 0.025 of the axis. 'mass < 91.2' (-3.845), 'mass < 82.9' (-3.332), 'girth2_top40 < 0.00803' (-2.683), 'girth < 0.0859' (-2.521) and 'mass_top50 < 71.8' (-1.933) outweigh 'e2_sq < 0.00961' (+2.757), 'mass < 121' (+1.894) and 'mass < 101' (+1.587). The sum is -6.855, the neuron is off and adds nothing to the scores. The formula calls them g.
- **light (30 GeV), very narrow, pT spread over several particles** — 6.4% of jets, neuron 0.00. Mostly quark (73.1%) with 14.9% gluon; very light (about 30 GeV), width 0.0009, a very hard leading particle (344 GeV) and 91% of the pT within 0.025 of the axis. The mass cuts are deep in the negative: 'mass < 82.9' (-9.011), 'mass < 91.2' (-8.445) and 'mass_top50 < 71.8' (-7.262), against 'e2_sq < 0.00961' (+4.000) and the upper windows; 'mass < 101 and max_dr < 0.391' passes for only 62.9%. The sum is -19.768, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (32 GeV), very narrow, pT spread over several particles** — 3.6% of jets, neuron 0.00. Mostly quark (68.9%) with 20.2% gluon; mass about 32 GeV, width 0.0011, 85% of the pT within 0.025 of the axis and every particle within the 0.391 radius used by the tests. Unlike the group above, 'mass < 101 and max_dr < 0.391' (+4.034) and 'mass < 91.2 and max_dr < 0.391' (-4.263) both pass and cancel, and 'mass_top50 < 71.8 and max_dr < 0.394' adds 1.523; the mass cuts 'mass < 82.9' (-8.581), 'mass < 91.2' (-8.096) and 'mass_top50 < 71.8' (-6.868) still dominate. The sum is -18.112, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (26 GeV), very narrow, pT spread over several particles** — 1.6% of jets, neuron 0.00. Mostly quark (79.3%); mass about 26 GeV, width 0.0007, 87% of the pT within 0.025 of the axis and no particle beyond the 0.391 radius. The compact pair grows larger, 'mass < 101 and max_dr < 0.391' (+8.277) against 'mass < 91.2 and max_dr < 0.391' (-8.888), with 'mass_top50 < 71.8 and max_dr < 0.394' (+3.260); 'mass < 82.9' (-9.651), 'mass < 91.2' (-8.963) and 'mass_top50 < 71.8' (-7.932) dominate. The sum is -19.612, the neuron is off and adds nothing to the scores. The formula calls them q.
- **very light (18 GeV), very narrow, pT spread over several particles** — 0.5% of jets, neuron 0.00. Mostly quark (87.7%); the lightest here (about 18 GeV), width 0.0004, a very hard leading particle (376 GeV) and 92% of the pT within 0.025 of the axis. 'mass < 101 and max_dr < 0.391' (+15.497) and 'mass < 91.2 and max_dr < 0.391' (-16.925) are at their largest and nearly cancel, 'mass_top50 < 71.8 and max_dr < 0.394' adds 6.450, and 'mass < 82.9' (-10.962), 'mass < 91.2' (-10.025) and 'mass_top50 < 71.8' (-9.235) decide it. The sum is -20.773, the neuron is off and adds nothing to the scores. The formula calls them q.

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

- **medium-mass (80 GeV), narrow, pT spread over several particles** — 28.1% of jets, neuron 0.25. Mostly W (64.7%), with 11.8% gluon and 11% Z; mass about 80 GeV, width 0.0059, with pT at 0.025-0.1 from the axis and little beyond 0.2. 'girth < 0.0975' (+1.041), 'mass > 64.5' (99% pass, +0.810) and 'z_dr_0p2_0p4 < 0.0912' (+0.676) add, but 'lam1 < 0.0205' (-1.363), 'n_dr_0p2_0p4 < 21' (-0.911) and 'mass_over_sum_pt_sq < 0.014' (-0.526) take off more; 'mass > 80.8' passes for 38.8% and adds little. With the intercept -0.4386 the value is 0.246 (on for 29.5%), taking 0.215 from W and 0.23 from Z. The formula calls them W.
- **light (43 GeV), very narrow, pT spread over several particles** — 26.1% of jets, neuron 0.11. A quark-gluon mixture (53.7% quark, 33.2% gluon); mass about 43 GeV, width 0.0018, a hard leading particle (304 GeV) and 80% of the pT within 0.025 of the axis. The mass cuts fail ('mass > 64.5' passes for 9%); 'girth < 0.0975' (+2.134), 'z_dr_0p2_0p4 < 0.0912' (+0.783) and 'lam1 < 0.00824' (+0.765) are outweighed by 'lam1 < 0.0205' (-1.712), 'mass_over_sum_pt_sq < 0.00617' (-1.103), 'n_dr_0p2_0p4 < 21' (-0.887) and 'mass_over_sum_pt_sq < 0.014' (-0.793). The value is 0.106 (on for 15.4%), barely touching the scores (-0.093 W, -0.099 Z). The formula calls them q.
- **heavy (93 GeV), average width, pT spread over several particles** — 21.6% of jets, neuron 0.80. Mostly Z (71.6%), with 11.5% gluon and 10.2% top; mass about 93 GeV, width 0.0081, with 43% of the pT at 0.05-0.1 from the axis. 'mass > 64.5' (+1.504), 'mass > 80.8' (+0.648), 'z_dr_0p2_0p4 < 0.0912' (+0.631), 'girth < 0.0975' (+0.622) and 'mass > 87.4' (95.5%, +0.197) add, against 'lam1 < 0.0205' (-1.179), 'n_dr_0p2_0p4 < 21' (-0.882) and 'mass_over_sum_pt_sq < 0.014' (-0.383); 'mass > 101' does not pass. The value is 0.803 (on for 57.8%), taking 0.702 from W and 0.752 from Z and adding 0.169 to t. The formula calls them Z.
- **very heavy (171 GeV), very wide, pT spread over several particles** — 8.0% of jets, neuron 4.27. Mostly top (91.5%); mass about 171 GeV, broad (width 0.0288), pT shared out (hardest 158 GeV) and 20% of it at 0.2-0.3 from the axis. The mass steps all pass and grow with mass: 'mass > 64.5' (+5.612), 'mass > 80.8' (+4.768), 'mass > 87.4' (+2.867), with 'girth2_top20 > 0.00803' (+4.167), against 'mass > 101' (-7.512), 'mass_over_sum_pt > 0.098' (-5.067), 'girth2_top20 > 0.00572' (-2.519) and 'mass_top50 > 82' (-2.215). The value is 4.268 (always on), taking 3.735 from W and 4.001 from Z and adding 0.9 to t. The formula calls them t.
- **heavy (122 GeV), wide, pT spread over several particles** — 5.4% of jets, neuron 3.00. A top-gluon mixture (43.4% top, 40.5% gluon, 13.9% quark); mass about 122 GeV, width 0.0138, 38% of the pT at 0.05-0.1 from the axis. 'mass > 64.5' (+3.030), 'mass > 80.8' (+2.178) and 'mass > 87.4' (+1.188) outweigh 'mass > 101' (-2.252), 'mass_over_sum_pt > 0.098' (91.7%, -1.373) and 'mass_top50 > 82' (-0.926); 'girth2_top20 > 0.00803' passes for 78.2% (+0.688). The value is 2.995 (always on), taking 2.621 from W and 2.808 from Z and adding 0.632 to t. The formula calls them t, only slightly more often than g.
- **very heavy (143 GeV), very wide, pT spread over several particles** — 4.0% of jets, neuron 4.12. Mostly top (73.2%) with 15.5% gluon and 10.6% quark; mass about 143 GeV, width 0.0215, total pT 980 GeV, spread to 0.05-0.3 from the axis. 'mass > 64.5' (+4.132), 'mass > 80.8' (+3.284), 'girth2_top20 > 0.00803' (+2.295), 'mass > 87.4' (+1.904) and 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (+1.816) against 'mass > 101' (-4.497), 'mass_over_sum_pt > 0.098' (-3.427) and 'girth2_top20 > 0.00572' (-1.514). The value is 4.124, taking 3.609 from W and 3.866 from Z and adding 0.87 to t. The formula calls them t.
- **very heavy (167 GeV), very wide, pT spread over several particles, high pT** — 2.6% of jets, neuron 2.53. A top-gluon mixture (53.9% top, 40.5% gluon); mass about 167 GeV, width 0.0198, high total pT (1213 GeV). The mass steps 'mass > 64.5' (+5.412), 'mass > 80.8' (+4.568) and 'mass > 87.4' (+2.737) against 'mass > 101' (-7.105), 'mass_over_sum_pt > 0.098' (-2.983) and 'mass_top50 > 82' (-2.062); because of the high pT, 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' passes for only 33.1%, which keeps the value lower than in the other top-rich groups. The value is 2.534, taking 2.217 from W and 2.376 from Z and adding 0.535 to t. The formula calls them t.
- **medium-mass (64 GeV), average width, pT spread over several particles, low pT** — 1.7% of jets, neuron 3.95. A mixture: 46.7% gluon, 29.4% quark and 23.8% top; mass about 64 GeV, width 0.0085, the lowest total pT here (738 GeV) with a soft leading particle (152 GeV). The low pT decides it: 'sum_pt < 1.01e+03' passes for all and adds 3.557, 'sum_pt_top40 < 1e+03' adds 1.830 and 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (59.5%) 1.340, against 'sum_pt_top40 < 1e+03 and log_sum_pt < 6.81' (-1.499), 'lam1 < 0.0205' (-1.224) and 'sum_pt_top30 < 1.01e+03' (-0.963); 'mass > 64.5' passes for only 51.6%. The value is 3.951 (on for 99.7%), taking 3.457 from W and 3.704 from Z and adding 0.833 to t. The formula calls them g.
- **very heavy (161 GeV), very wide, pT spread over several particles** — 1.6% of jets, neuron 5.83. Mostly top (84.1%); mass about 161 GeV, very broad (width 0.0332), low total pT (889 GeV) shared out (hardest 123 GeV) and 25% of it at 0.2-0.3 from the axis. Everything that raises the neuron passes: 'girth2_top20 > 0.00803' (+5.110), 'mass > 64.5' (+5.094), 'mass_over_sum_pt > 0.077 and sum_pt < 1.12e+03' (+4.441), 'mass > 80.8' (+4.249) and 'mass > 87.4' (+2.530), against 'mass > 101' (-6.459), 'mass_over_sum_pt > 0.098' (-5.936), 'mass > 87.4 and log_sum_pt < 6.9' (-3.142) and 'girth2_top20 > 0.00572' (-3.025). The value is 5.826, the neuron's largest, taking 5.098 from W and 5.462 from Z and adding 1.229 to t. The formula calls them t.
- **very heavy (221 GeV), very wide, pT spread over several particles** — 1.0% of jets, neuron 3.67. Mostly gluon (53.7%) with 34.8% top; mass about 221 GeV, width 0.0364, total pT 1182 GeV and 28% of the pT at 0.2-0.3 from the axis. The largest mass steps, 'mass > 64.5' (+8.217), 'mass > 80.8' (+7.381), 'girth2_top20 > 0.00803' (+5.786) and 'mass > 87.4' (+4.561), are held back by 'mass > 101' (-12.817), 'mass_over_sum_pt > 0.098' (-6.489), 'girth2_top20 > 0.00572' (-3.388), 'mass_top50 > 82' (-3.374) and 'mass > 144' (-1.696). The value is 3.667 (always on), taking 3.209 from W and 3.438 from Z and adding 0.773 to t. The formula calls them g.

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

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 26.9% of jets, neuron 0.53. Mostly W (66.7%) with 12.7% gluon; mass about 79 GeV, width 0.0058, pT at 0.025-0.1 from the axis. 'mass_top50 < 137' adds 3.843 for all, but 'mass_top50 < 161' (-2.100), 'girth2_top15 < 0.0215' (-1.825), 'mass > 62.5' (99.1%, -0.846) and 'e2 < 0.0476' (-0.662) take most of it back. With the intercept 0.9079 the value is 0.527 (on for 84.8%), adding 0.272 to g and 0.296 to q and taking 0.115 from W. The formula calls them W.
- **heavy (92 GeV), average width, pT spread over several particles** — 22.1% of jets, neuron 0.08. Mostly Z (75.4%); mass about 92 GeV, width 0.0079, with 41% of the pT at 0.05-0.1 from the axis. 'mass_top50 < 137' (+2.997) is cancelled by 'mass_top50 < 161' (-1.772), 'girth2_top15 < 0.0215' (-1.609) and 'mass > 62.5' (-1.496), with 'mass > 82.9' (+0.443) giving a little back. The value is 0.084 (on for 42.9%), with only tiny effects on the g and q scores. The formula calls them Z.
- **light (32 GeV), very narrow, pT spread over several particles** — 13.2% of jets, neuron 3.93. Mostly quark (69.3%) with 18.4% gluon; light (about 32 GeV), extremely narrow (width 0.0010), a very hard leading particle (355 GeV) and 92% of the pT within 0.025 of the axis. 'mass_top50 < 137' adds 6.892, 'LHA < 0.209' 2.823 and 'girth2 < 0.00364' 1.105, and 'mass > 62.5' does not pass; the narrowness tests 'mass_top50 < 161' (-3.283), 'girth < 0.0436' (-2.494), 'girth2_top15 < 0.0215' (-2.262), 'mass_top40 < 91.3' (-1.445), 'e2 < 0.0476' (-1.404) and 'girth < 0.0278' (-1.040) take back about half. The value is 3.932 (always on), adding 2.027 to g and 2.212 to q and taking 0.86 from W. The formula calls them q.
- **medium-mass (54 GeV), very narrow, pT spread over several particles** — 12.1% of jets, neuron 2.94. A gluon-quark mixture (49.6% gluon, 37.6% quark); mass about 54 GeV, width 0.0027, with 69% of the pT within 0.025 of the axis. 'mass_top50 < 137' (+5.520) and 'LHA < 0.209' (90% pass, +1.117) with several small narrowness credits outweigh 'mass_top50 < 161' (-2.751), 'girth2_top15 < 0.0215' (-2.197), 'e2 < 0.0476' (-1.207), 'girth < 0.0436' (-1.157) and 'mass_top40 < 91.3' (-1.016). The value is 2.94 (always on), adding 1.516 to g and 1.654 to q and taking 0.643 from W. The formula calls them g.
- **very heavy (172 GeV), very wide, pT spread over several particles** — 10.2% of jets, neuron 0.24. Mostly top (86%); mass about 172 GeV, broad (width 0.0281), pT shared out (hardest 162 GeV) and 20% of it at 0.2-0.3 from the axis. 'mass_top50 < 137' fails here, and the mass steps balance: 'mass > 62.5' (-5.564) and 'mass > 144' (-2.845) against 'mass > 82.9' (+4.351), 'mass > 92.9' (+2.652) and 'girth > 0.0975' (98%, +0.615). The sum is close to zero (0.010), so the value is 0.245 (on for 57.4%) with small effects on the scores. The formula calls them t.
- **very heavy (146 GeV), very wide, pT spread over several particles** — 6.2% of jets, neuron 0.72. Mostly top (66.5%) with 23% gluon; mass about 146 GeV, width 0.0213, pT spread to 0.05-0.3 from the axis. 'mass > 62.5' (-4.273) is offset by 'mass > 82.9' (+3.111) and 'mass > 92.9' (+1.800); 'mass_top50 < 137' passes for only 31.6% and 'mass > 144' for 60.5% (-0.521), and 'girth2_top15 < 0.0215' (79.5%, -0.552) takes a little more. The value is 0.717 (on for 77.7%), adding 0.37 to g and 0.403 to q and taking 0.157 from W. The formula calls them t.
- **heavy (118 GeV), wide, pT spread over several particles** — 5.9% of jets, neuron 0.28. A top-gluon mixture (42.3% top, 39.6% gluon, 15.4% quark); mass about 118 GeV, width 0.0131, 38% of the pT at 0.05-0.1 from the axis. 'mass > 82.9' (+1.712), 'mass_top50 < 137' (+1.527) and 'mass > 92.9' (+0.839) against 'mass > 62.5' (-2.817), 'girth2_top15 < 0.0215' (-1.307) and 'mass_top50 < 161' (-1.202) leave a small value of 0.281 (on for 52.2%), with small effects on the scores. The formula calls them t slightly more often than g.
- **medium-mass (73 GeV), average width, pT spread over several particles, low pT** — 1.8% of jets, neuron 3.15. A mixture: 39.6% gluon, 33.1% top and 26% quark; mass about 73 GeV, width 0.0097, low total pT (805 GeV) with a soft leading particle (165 GeV). The low pT turns on 'sum_pt < 950' (+4.713), only partly cancelled by 'log_sum_pt < 6.86' (-3.316); with 'mass_top50 < 137' (+4.332) against 'mass_top50 < 161' (-2.289) and 'girth2_top15 < 0.0215' (-1.644), the value is 3.154 (always on), adding 1.626 to g and 1.774 to q and taking 0.69 from W. The formula calls them g.
- **very heavy (221 GeV), very wide, pT spread over several particles, high pT** — 1.1% of jets, neuron 0.00. Mostly gluon (58.5%) with 30.4% top; mass about 221 GeV, very broad (width 0.0349), total pT 1222 GeV and 26% of the pT at 0.2-0.3 from the axis. 'mass > 62.5' (-8.057) and 'mass > 144' (-7.830) outweigh 'mass > 82.9' (+6.746) and 'mass > 92.9' (+4.297), and 'mass_top50 < 137' fails. The sum is -3.187, the neuron is on for only 0.2% and adds nothing to the scores. The formula calls them g.
- **medium-mass (55 GeV), average width, pT spread over several particles, low pT** — 0.6% of jets, neuron 4.86. A gluon-quark mixture (52.8% gluon, 34% quark, 13.2% top); mass about 55 GeV, width 0.0098, the lowest total pT here (625 GeV) with a soft leading particle (130 GeV). 'sum_pt < 950' is at its largest (+10.546), cancelled only partly by 'log_sum_pt < 6.86' (-8.468), and 'mass_top50 < 137' (+5.477) beats 'mass_top50 < 161' (-2.734) and 'girth2_top15 < 0.0215' (-1.680). The value is 4.855, the neuron's largest (on for 99.7%), adding 2.503 to g and 2.731 to q and taking 1.062 from W. The formula calls them g.

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

- **medium-mass (87 GeV), average width, pT spread over several particles** — 30.5% of jets, neuron 1.61. The largest group (30.5%), a W-Z mixture (43.9% Z, 43.5% W) with 7.3% top; mass about 87 GeV, width 0.0071, with 52% of the pT at 0.05-0.1 from the axis and only 2.5% within 0.025. 'e2 < 0.0652' (-1.281), 'girth2_top5 < 0.0244' (-1.209), 'mass < 121' (-0.566), 'n_dr_0p2_0p4 < 13' (-0.504) and 'max_dr < 0.402' (-0.439) take the intercept 3.7825 down, partly offset by 'girth2_top10 > 0.000237' (+0.747), 'z_dr_0p2_0p4 < 0.0912' (+0.656), 'girth2_top5 < 0.0244 and sum_pt_top3 < 656' (+0.604) and 'mass < 121 and tau21 < 0.47' (+0.367). The value is 1.608 (on for 97.7%), adding 1.583 to the t score; this neuron does not enter the W or Z score. The formula calls them W, closely followed by Z.
- **medium-mass (89 GeV), average width, pT spread over several particles** — 23.0% of jets, neuron 0.79. A mixture: 32.2% gluon, 23.6% W, 22.8% Z and 12.4% quark; mass about 89 GeV, width 0.0072, total pT 1069 GeV, with 30% of the pT within 0.025 and 41% at 0.025-0.05 from the axis. The central pT makes 'e2 < 0.0652' (-1.876) and 'girth2_top5 < 0.0244' (-1.458) bite harder than for the wide two-prong group, and 'z_dr_0_0p05 > 0.767' (54.1%, -0.206) adds to it; 'girth2_top5 < 0.0244 and sum_pt_top3 < 656' (+0.762), 'girth2_top10 > 0.000237' (+0.345) and 'z_dr_0p2_0p4 < 0.0912' (83.1%, +0.312) give back less. The value is 0.788 (on for 73.8%), adding 0.775 to t. The formula calls them g.
- **light (30 GeV), very narrow, pT spread over several particles** — 13.2% of jets, neuron 0.06. Mostly quark (72.6%) with 15.3% gluon; very light (about 30 GeV), width 0.0009, a very hard leading particle (347 GeV) and 90% of the pT within 0.025 of the axis. 'e2 < 0.0652' (-2.577), 'girth2_top5 < 0.0244' (-1.567), 'mass < 121' (-1.502) and 'z_dr_0_0p05 > 0.767' (-0.841) outweigh 'mass < 80.4' (+2.679), 'mass < 86.4' (+0.790) and 'z_dr_0p2_0p4 < 0.0912' (+0.674); 'girth2_top10 > 0.000237' passes for only 24.8%. The sum is -0.578, so the value is 0.06 (on for 15.8%), barely touching the t score (+0.059). The formula calls them q.
- **medium-mass (54 GeV), narrow, pT spread over several particles** — 13.0% of jets, neuron 0.53. Mostly gluon (50.9%) with 36.3% quark; mass about 54 GeV, width 0.0028, 68% of the pT within 0.025 of the axis. 'e2 < 0.0652' (-2.340), 'girth2_top5 < 0.0244' (-1.542) and 'mass < 121' (-1.104) against 'mass < 80.4' (+1.401), 'girth2_top5 < 0.0244 and sum_pt_top3 < 656' (+0.777) and 'z_dr_0p2_0p4 < 0.0912' (+0.593). The value is 0.529 (on for 64.1%), adding 0.52 to t. The formula calls them g.
- **heavy (130 GeV), wide, pT spread over several particles** — 7.0% of jets, neuron 2.76. Mostly top (65.2%), with 20.9% gluon and 12.2% quark; mass about 130 GeV, width 0.0179, total pT 982 GeV shared out (hardest 173 GeV), 40% of it at 0.05-0.1 from the axis. 'girth2_top10 > 0.000237' (+1.613) and 'girth2_top5 < 0.0244 and sum_pt_top3 < 656' (+0.639) help, and the larger e2 makes 'e2 < 0.0652' cost less (-0.853); 'girth2_top5 < 0.0244' (-0.846), 'lam1 > 0.00187' (-0.799) and 'girth2_top10 > 0.00768' (-0.537) take off, while 'mass < 121' passes for only 28.5%. The value is 2.762 (on for 99.6%), adding 2.719 to t. The formula calls them t.
- **very heavy (174 GeV), very wide, pT spread over several particles** — 5.2% of jets, neuron 2.65. Mostly top (91.3%); mass about 174 GeV, broad (width 0.0299), pT shared out (hardest 155 GeV) and spread to 0.1-0.3 from the axis. 'girth2_top10 > 0.000237' (+3.398) and 'mass_top50 > 137' (+2.722) add, and 'e2 < 0.0652' passes for only 44.6% (-0.152); 'mass > 144' (-2.559), 'girth2_top10 > 0.00768' (-1.942), 'lam1 > 0.00187' (-1.335) and 'mass > 163' (-1.071) take off. The value is 2.651 (on for 97.9%), adding 2.61 to t. The formula calls them t.
- **very heavy (156 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 3.18. Mostly top (85.6%); mass about 156 GeV, width 0.0269, total pT 961 GeV shared among many particles (hardest 144 GeV) and 31% of it at 0.1-0.15 from the axis. 'girth2_top10 > 0.000237' (+2.958) and 'mass_top50 > 137' (94.7%, +1.244) outweigh 'girth2_top10 > 0.00768' (-1.595), 'lam1 > 0.00187' (-1.201) and 'mass > 144' (93.8%, -1.133), and being below 163 GeV it avoids most of 'mass > 163'. The value is 3.182, the neuron's largest (on for 99.8%), adding 3.133 to t. The formula calls them t.
- **very heavy (169 GeV), very wide, pT spread over several particles** — 3.3% of jets, neuron 2.37. Mostly top (78.6%) with 17.4% gluon; mass about 169 GeV, width 0.0246, total pT 1100 GeV, 34% of the pT at 0.05-0.1 from the axis. 'mass_top50 > 137' (+2.233) and 'girth2_top10 > 0.000237' (+1.894) against 'mass > 144' (-2.150), 'lam1 > 0.00187' (-1.105), 'girth2_top5 < 0.0244' (-0.847), 'girth2_top10 > 0.00768' (-0.771), 'mass > 163' (85%, -0.645) and 'e2 < 0.0652' (89%, -0.610): the higher pT makes the prongs closer, so more of the narrowness debits apply. The value is 2.37 (on for 94.4%), adding 2.333 to t. The formula calls them t.
- **very heavy (204 GeV), very wide, pT spread over several particles** — 1.1% of jets, neuron 0.19. A gluon-top mixture (47.4% gluon, 41.2% top); mass about 204 GeV, width 0.0345, 26% of the pT at 0.2-0.3 from the axis. The mass terms grow with mass and now favour going down: 'mass > 144' (-5.077) and 'mass > 163' (-3.915) against 'mass_top50 > 137' (+4.766), with 'girth2_top10 > 0.000237' (+3.675) against 'girth2_top10 > 0.00768' (-2.167) and 'lam1 > 0.00187' (-1.608). The sum is -1.145, so the value is 0.187 (on for 26.2%), adding only 0.184 to t. The formula splits them almost evenly between t and g.
- **very heavy (259 GeV), very wide, pT spread over several particles, high pT** — 0.2% of jets, neuron 0.00. Mostly gluon (74%) with 16% top; mass about 259 GeV, width 0.0360, high total pT (1405 GeV) and 28% of the pT at 0.2-0.3 from the axis. 'mass > 144' (-9.719) and 'mass > 163' (-9.159) outweigh 'mass_top50 > 137' (+9.050) and 'girth2_top10 > 0.000237' (+3.903), with 'girth2_top10 > 0.00768' (-2.341) on top. The sum is -6.753, the neuron is off and adds nothing to the scores. The formula calls them g.

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

- **heavy (133 GeV), wide, pT spread over several particles** — 33.9% of jets, neuron 0.01. Mostly top (51.4%), with 22.7% gluon and 13.8% Z; mass about 133 GeV, width 0.0182, pT shared out (hardest 178 GeV) and spread to 0.1-0.3 from the axis. Almost none of the tests pass: 'mass_over_sum_pt_sq < 0.00751' passes for 13.3% and 'n_particles < 46' for 9.6%, so there is nothing to lift the intercept -0.3058. The value is 0.009 and the neuron is on for 7.4%, so it leaves the scores alone. The formula calls them t.
- **light (50 GeV), very narrow, pT spread over several particles** — 11.1% of jets, neuron 0.26. Mostly gluon (59.6%) with 32.8% quark; mass about 50 GeV, narrow (width 0.0021), total pT 1112 GeV and 72% of the pT within 0.025 of the axis; 'n_particles < 46' passes for only 39.1%. 'mass_over_sum_pt_sq < 0.00751' adds 1.240 for all, but 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' (-0.719) and 'lam1 < 0.00767' (-0.521) take most of it back, and the many particles keep 'n_particles < 46' from helping (+0.116). The value is 0.256 (on for 67.2%), taking 0.192 from g and adding 0.112 to each of W and Z. The formula calls them g.
- **medium-mass (75 GeV), narrow, pT spread over several particles** — 10.4% of jets, neuron 0.11. A mixture: 39.1% gluon, 30.3% W, 18.3% quark; mass about 75 GeV, width 0.0049, with 41% of the pT within 0.025 of the axis; 'n_particles < 46' passes for 28.4%. 'mass_over_sum_pt_sq < 0.00751' (+0.591) and 'n_dr_0p1_0p2 < 15' (+0.189) are cancelled by 'lam1 < 0.00767' (-0.316) and 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' (-0.276). The value is 0.106, on for 34.1%, so it only slightly lowers g (-0.079) and raises W and Z (+0.046 each). The formula calls them g.
- **heavy (90 GeV), average width, pT spread over several particles** — 8.6% of jets, neuron 0.25. Mostly Z (44.2%) with 33.5% W; mass about 90 GeV, width 0.0081, a hard leading particle (326 GeV) and 42% of the pT at 0.025-0.05 from the axis; all have fewer than 46 particles. 'n_particles < 46' passes for all and adds 0.741, with 'n_dr_0p1_0p2 < 15' (+0.324); 'tau21 < 0.428 and lam1 < 0.0165' (-0.416) and 'n_particles < 46 and mass_top30 > 73.3' (-0.328) take off, and 'mass_over_sum_pt_sq < 0.00751' passes for only 56.2%. The value is 0.25 (on for 58.1%), taking 0.188 from g and adding 0.11 to each of W and Z. The formula calls them Z.
- **medium-mass (84 GeV), narrow, pT spread over several particles** — 8.3% of jets, neuron 0.77. Mostly W (60.1%) with 36.6% Z; mass about 84 GeV, width 0.0065, with 54% of the pT at 0.05-0.1 from the axis, only 2.1% within 0.025 and almost nothing beyond 0.2 (0.19%). The few-particle tests all pass: 'n_particles < 46' (+0.586), 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.551) and 'n_dr_0p2_0p4 < 7' (+0.466), with 'tau21 < 0.428 and lam1 < 0.0165' (-0.584) the main debit. The value is 0.772 (on for 96.8%), taking 0.579 from g and adding 0.338 to each of W and Z. The formula calls them W.
- **light (38 GeV), very narrow, pT spread over several particles** — 7.6% of jets, neuron 1.29. Mostly quark (71.2%) with 10.7% gluon; light (about 38 GeV), very narrow (width 0.0015), a very hard leading particle (364 GeV) and 89% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00751' (+1.376), 'n_particles < 46' (+0.977) and 'n_dr_0p1_0p2 < 15' (+0.464) all pass, minus 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' (-0.771) and 'lam1 < 0.00767' (-0.550). The value is 1.295 (on for 99.9%), taking 0.971 from g and adding 0.567 to each of W and Z. The formula calls them q.
- **medium-mass (86 GeV), average width, pT spread over several particles** — 7.5% of jets, neuron 0.29. A W-Z mixture (46.3% W, 44.8% Z); mass about 86 GeV, width 0.0069, a softer leading particle (185 GeV) and 53% of the pT at 0.05-0.1 from the axis; 'n_particles < 46' passes for 42.1%. 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.427) and 'n_dr_0p2_0p4 < 7' (+0.420) add, but 'tau21 < 0.428 and lam1 < 0.0165' (-0.313) and 'n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988' (59.8%, -0.209) take off. The value is 0.289 (on for 60.1%), taking 0.217 from g and adding 0.127 to each of W and Z. The formula calls them W.
- **medium-mass (86 GeV), average width, pT spread over several particles** — 6.0% of jets, neuron 1.06. A W-Z mixture (49% W, 48.1% Z); mass about 86 GeV, width 0.0069, a hard leading particle (312 GeV), 52% of the pT at 0.05-0.1 from the axis, and all with fewer than 46 particles. 'n_particles < 46' (+1.151), 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.550), 'n_dr_0p2_0p4 < 7' (+0.464) and 'n_dr_0p1_0p2 < 15' (+0.303) add, against 'tau21 < 0.428 and lam1 < 0.0165' (-0.697) and 'n_particles < 46 and mass_top30 > 73.3' (-0.432). The value is 1.058 (on for 98.9%), taking 0.793 from g and adding 0.463 to each of W and Z. The formula calls them W.
- **light (25 GeV), very narrow, pT spread over several particles** — 5.2% of jets, neuron 1.84. Mostly quark (82.6%); very light (about 25 GeV), extremely narrow (width 0.0006), a very hard leading particle (387 GeV) and 93% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00751' (+1.575), 'n_particles < 46' (+1.320), 'n_dr_0p1_0p2 < 15' (+0.509), 'n_dr_0p2_0p4 < 7' (+0.417) and 'n_dr_0p2_0p4 < 5 and z_top50_slots > 0.979' (+0.459) all pass, minus 'mass_over_sum_pt_sq < 0.00751 and mass_top20 < 137' (-0.944), 'lam1 < 0.00767' (-0.606) and 'n_dr_0p2_0p4 < 5 and dr_0 < 0.0412' (-0.495). The value is 1.842, the neuron's largest, taking 1.381 from g and adding 0.806 to each of W and Z. The formula calls them q.
- **medium-mass (77 GeV), average width, pT spread over several particles** — 1.4% of jets, neuron 0.00. A mixture: 30.5% gluon, 25.2% W, 21.2% Z, 15.2% quark; mass about 77 GeV, total pT 971 GeV shared among many particles (hardest only 119 GeV), with almost nothing beyond 0.2 from the axis. 'n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988' passes for all and removes 1.793, which sets this group apart and outweighs 'n_dr_0p2_0p4 < 7' (+0.473) and 'mass_over_sum_pt_sq < 0.00751' (+0.381). The sum is -1.482 and the neuron is on for only 0.6%, so it adds nothing to the scores. The formula calls them g.

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

- **heavy (137 GeV), very wide, pT spread over several particles** — 30.7% of jets, neuron 0.85. The largest group (30.7%): mostly top (56.8%), with 19.7% gluon and 14.6% Z; mass about 137 GeV, width 0.0194, pT shared out (hardest 168 GeV) and spread to 0.1-0.3 from the axis. Almost none of the if-statements pass for these heavy broad jets (the largest pass rate is 37.5%, for 'girth2_top5 < 0.00716'), so the neuron sits near its intercept 0.8778: the value is 0.846 (on for 99.6%), adding 0.502 to W and taking 0.212 from q. The formula calls them t.
- **medium-mass (86 GeV), average width, pT spread over several particles** — 21.9% of jets, neuron 2.32. A W-Z mixture, mostly W (51.3%) with 41.9% Z; mass about 86 GeV, width 0.0068, with 56% of the pT at 0.05-0.1 from the axis and nearly nothing beyond 0.2 (0.16% at 0.2-0.3). 'n_dr_0p2_0p4 < 10 and z_dr_0p2_0p4 < 0.0685' passes for all and adds 1.079, with 'z_dr_0p2_0p4 < 0.00638' (93.1%, +0.380), 'mass < 92.9' (92.1%, +0.300), 'mass_over_sum_pt_sq < 0.00818' (+0.265) and 'mass < 101' (+0.242); 'z_top5_slots > 0.535' (57%, -0.259) and 'mass_top50 < 86.4' (58.1%, -0.224) take a little off. The value is 2.319, the neuron's largest (always on), adding 1.377 to W and taking 0.58 from q. The formula calls them W.
- **medium-mass (89 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 0.66. A Z-W mixture, mostly Z (42.2%) with 35.2% W and 10.2% quark; mass about 89 GeV, width 0.0075, with a very lopsided pT sharing (hardest particle 413 GeV against 240 for all jets) and 47% of the pT at 0.025-0.05 from the axis. The lopsided sharing makes 'z_top5_slots > 0.535' pass for all (-1.061), which sets this group apart, with 'girth2_top5 < 0.00716' (-0.325) and 'e2 < 0.0388' (-0.271) also taking off; 'sum_pt_top3 > 439' (+0.527), 'mass < 92.9' (+0.262), 'n_dr_0p2_0p4 < 10 and z_dr_0p2_0p4 < 0.0685' (47.2%, +0.262) and 'mass_over_sum_pt_sq < 0.00818' (+0.242) add. The value is 0.664 (on for 88.8%), adding 0.394 to W and taking 0.166 from q. The formula calls them Z.
- **medium-mass (83 GeV), narrow, pT spread over several particles** — 8.9% of jets, neuron 0.46. A W-gluon mixture (37.8% W, 36.3% gluon, with about 10% each of Z and quark); mass about 83 GeV, width 0.0061, total pT 1083 GeV, with 33% of the pT within 0.025 and 39% at 0.025-0.05 from the axis. The narrowness tests take off: 'e2 < 0.0388' (-0.560), 'girth2_top30 < 0.00636' (-0.509), 'mass_top50 < 86.4' (86.8%, -0.468) and 'girth2_top5 < 0.00716' (-0.445), against 'mass < 92.9' (+0.419), 'mass_over_sum_pt_sq < 0.00818' (+0.405) and 'e2 < 0.0279' (85.6%, +0.385). The value is 0.457 (on for 77.5%), adding 0.272 to W and taking 0.114 from q. The formula calls them W.
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 5.5% of jets, neuron 0.03. Mostly gluon (51.9%), with 26.6% quark and 11.7% W; mass about 67 GeV, width 0.0041, 61% of the pT within 0.025 of the axis. 'mass_top50 < 86.4' (-1.382), 'girth2_top30 < 0.00636' (-1.042), 'e2 < 0.0388' (-0.756) and 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' (79.2%, -0.620) outweigh 'mass < 92.9' (+0.968), 'mass_over_sum_pt_sq < 0.00818' (+0.753) and 'e2 < 0.0279' (+0.706). The sum is -0.853, so the neuron is on for only 7.2% (value 0.026) and barely touches the scores. The formula calls them g.
- **medium-mass (56 GeV), narrow, pT spread over several particles** — 5.4% of jets, neuron 0.00. Mostly gluon (50.4%) with 34.9% quark; mass about 56 GeV, width 0.0029, 63% of the pT within 0.025 of the axis. 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' passes for all and removes 2.309, with 'mass_top50 < 86.4' (-1.947) and 'girth2_top30 < 0.00636' (-1.255), against 'mass < 92.9' (+1.354) and 'mass_over_sum_pt_sq < 0.00818' (+0.961). The sum is -2.746, the neuron is on for 0.3% and adds nothing to the scores. The formula calls them g.
- **light (47 GeV), very narrow, pT spread over several particles** — 5.3% of jets, neuron 0.00. A gluon-quark mixture (45.4% gluon, 44.2% quark); mass about 47 GeV, width 0.0020, 73% of the pT within 0.025 of the axis. 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' grows as the mass drops (-4.056), with 'mass_top50 < 86.4' (-2.507) and 'girth2_top30 < 0.00636' (-1.442), against 'mass < 92.9' (+1.725) and 'mass_over_sum_pt_sq < 0.00818' (+1.130). The sum is -4.945, the neuron is off and adds nothing to the scores. The formula calls them q, slightly more often than g.
- **light (37 GeV), very narrow, pT spread over several particles** — 5.0% of jets, neuron 0.00. Mostly quark (59.8%) with 28.7% gluon; mass about 37 GeV, width 0.0013, 83% of the pT within 0.025 of the axis. 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' (-5.850) and 'mass_top50 < 86.4' (-3.024) outweigh 'mass < 92.9' (+2.071) and 'mass_over_sum_pt_sq < 0.00818' (+1.253). The sum is -7.066, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (29 GeV), very narrow, pT spread over several particles** — 4.8% of jets, neuron 0.00. Mostly quark (76.6%) with 11.8% gluon; mass about 29 GeV, width 0.0008, a very hard leading particle (346 GeV) and 90% of the pT within 0.025 of the axis. 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' (-7.594), 'mass_top50 < 86.4' (-3.513) and 'mass < 62.5' (-1.312) outweigh 'mass < 92.9' (+2.381). The sum is -9.075, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (20 GeV), very narrow, pT spread over several particles** — 3.0% of jets, neuron 0.00. Mostly quark (84.9%); the lightest here (about 20 GeV), width 0.0004, a very hard leading particle (385 GeV) and 95% of the pT within 0.025 of the axis. 'mass < 80.4 and z_dr_0p2_0p4 < 0.037' is at its largest (-9.569), with 'mass_top50 < 86.4' (-4.051) and 'mass < 62.5' (-1.656), against 'mass < 92.9' (+2.709). The sum is -11.191, the neuron is off and adds nothing to the scores. The formula calls them q.

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

- **medium-mass (75 GeV), narrow, pT spread over several particles** — 69.4% of jets, neuron 1.90. The bulk (69.4%): a mixture of W (27.8%), Z (27.4%), quark (20.7%) and gluon (20.2%) with few tops (4%); mass about 75 GeV, width 0.0055, total pT 1072 GeV. Only small terms apply: 'mass > 74.3' (66.3% pass, +0.251) and 'sum_pt_top40 < 1.05e+03' (68%, +0.232) roughly cancel 'sum_pt < 1.05e+03' (63%, -0.223) and 'log_sum_pt < 7.02' (-0.166), so the value stays near the intercept 2.0711, at 1.904 (always on). This takes 1.725 from the t score; the neuron enters no other class score. The formula calls them W slightly more often than Z.
- **medium-mass (83 GeV), average width, pT spread over several particles** — 11.6% of jets, neuron 0.92. A mixture: 35.3% top, 27.7% quark and 20% gluon; mass about 83 GeV, width 0.0088, total pT 963 GeV, below the 1044 GeV average. The lower pT turns on the pT tests of both signs: 'sum_pt < 1.01e+03' (99.3%, -1.254), 'sum_pt < 1.05e+03' (-0.949) and 'sum_pt_top40 < 1.05e+03 and D2 < 5.38' (-0.543) against 'sum_pt_top50 < 1.01e+03' (+1.066) and 'sum_pt_top40 < 1.05e+03' (+0.956), with 'mass > 74.3' (64.8%, +0.466). The value drops to 0.916 (on for 89.2%), taking only 0.83 from t. The formula calls them t.
- **very heavy (173 GeV), very wide, pT spread over several particles** — 6.4% of jets, neuron 0.21. Mostly top (86.3%) with 10.6% gluon; mass about 173 GeV, broad (width 0.0270), pT shared out (hardest 173 GeV) and spread to 0.05-0.3 from the axis. The mass terms nearly cancel: 'mass_top50 > 137' (+3.878) and 'mass > 74.3' (+2.393) against 'mass > 144' (-3.549), 'mass_top50 > 97.9' (-2.179), 'mass > 137' (-1.078) and 'mass > 161' (-1.007), so the sum is 0.007. The value is 0.207 (on for 49.1%), taking only 0.188 from t. The formula calls them t.
- **very heavy (159 GeV), very wide, pT spread over several particles** — 5.1% of jets, neuron 0.79. Mostly top (79.6%) with 14.6% gluon; mass about 159 GeV, width 0.0248, total pT 1026 GeV, pT spread to 0.05-0.3 from the axis. 'mass > 74.3' (+2.047) and 'mass_top50 > 137' (+1.991) outweigh 'mass > 144' (-1.824) and 'mass_top50 > 97.9' (-1.694), and being mostly below 161 GeV ('mass > 161' passes for 43.1%) they lose less than the heavier group; the pT tests 'sum_pt_top40 < 1.05e+03' (+0.860) and 'sum_pt_top50 < 1.01e+03' (+0.780) against 'sum_pt < 1.01e+03' (-0.674) roughly cancel. The value is 0.787 (on for 79.5%), taking 0.714 from t. The formula calls them t.
- **medium-mass (90 GeV), wide, pT spread over several particles, low pT** — 4.1% of jets, neuron 0.05. A mixture: 48% top, 29.6% gluon, 20.8% quark; mass about 90 GeV, width 0.0123, low total pT (869 GeV) with a soft leading particle (168 GeV). The low pT makes the negative pT tests large: 'sum_pt < 1.01e+03' (-3.623), 'sum_pt < 1.05e+03' (-1.941), 'sum_pt_top40 < 956' (-0.962), 'sum_pt_top40 < 1.05e+03 and D2 < 5.38' (-0.954) and 'sum_pt < 986' (-0.783), more than 'sum_pt_top50 < 1.01e+03' (+2.777) and 'sum_pt_top40 < 1.05e+03' (+1.794) give back. The sum is -1.337, so the neuron is on for only 10.2% (value 0.049) and hardly touches the t score. The formula calls them t.
- **very heavy (179 GeV), very wide, pT spread over several particles** — 1.3% of jets, neuron 0.02. Mostly top (81.6%) with 11.9% gluon; mass about 179 GeV, very broad (width 0.0348), total pT 961 GeV and 27% of the pT at 0.2-0.3 from the axis. Several large pairs cancel: 'mass_over_sum_pt_sq > 0.0292' (-4.467) against 'mass_over_sum_pt > 0.171' (+4.289), and 'mass > 144' (-4.233) against 'mass_top50 > 137' (+4.253); what is left, 'mass_top50 > 97.9' (-2.275), 'mass > 161' (-1.479) and 'sum_pt < 1.01e+03' (81.9%, -1.433), beats 'mass > 74.3' (+2.531) and 'mass > 173' (70.8%, +1.062). The sum is -1.159, so the neuron is on for 6.5% and adds almost nothing. The formula calls them t.
- **medium-mass (64 GeV), average width, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00. A mixture: 48.9% gluon, 30.7% quark and 20.4% top; mass about 64 GeV, width 0.0110, very low total pT (679 GeV) with a soft leading particle (134 GeV). The pT tests grow as the pT drops: 'sum_pt < 1.01e+03' (-8.422), 'sum_pt < 1.05e+03' (-3.950), 'sum_pt_top40 < 956' (-2.375) and 'sum_pt < 986' (-2.056) outweigh 'sum_pt_top50 < 1.01e+03' (+6.002) and 'sum_pt_top40 < 1.05e+03' (+3.258). The sum is -6.550, the neuron is off and adds nothing to the scores. The formula calls them g.
- **very heavy (218 GeV), very wide, pT spread over several particles, high pT** — 0.6% of jets, neuron 0.44. Mostly gluon (76.4%) with 13.4% top; mass about 218 GeV, width 0.0284, high total pT (1332 GeV). Only the mass tests act, in large balancing steps: 'mass_top50 > 137' (+8.571), 'mass > 173' (+6.159) and 'mass > 74.3' (+3.475) against 'mass > 144' (-8.936), 'mass > 161' (-4.626), 'mass_top50 > 97.9' (-3.384) and 'mass > 137' (-2.400). The sum is about zero (0.030), so the value is 0.439 (on for 55.8%), taking 0.398 from t. The formula calls them g.
- **very heavy (200 GeV), very wide, pT spread over several particles** — 0.5% of jets, neuron 0.01. Mostly top (65.3%) with 25.6% gluon; mass about 200 GeV, the broadest here (width 0.0441), total pT 955 GeV and 37% of the pT at 0.2-0.3 from the axis. 'mass_over_sum_pt_sq > 0.0292' (-11.879) against 'mass_over_sum_pt > 0.171' (+10.695), and 'mass > 144' (-6.727) against 'mass_top50 > 137' (+6.337), with 'mass > 161' (-3.167) and 'mass_top50 > 97.9' (-2.809) outweighing 'mass > 173' (+3.753) and 'mass > 74.3' (+3.030). The sum is -2.392, so the neuron is on for only 3.6% and adds almost nothing. The formula calls them t.
- **very heavy (271 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.43. Mostly gluon (71.1%) with 19.7% top; mass about 271 GeV, width 0.0426, total pT 1332 GeV and 38% of the pT at 0.2-0.3 from the axis. The largest mass terms: 'mass > 144' (-15.326), 'mass > 161' (-8.918) and 'mass_over_sum_pt_sq > 0.0292' (-10.745) against 'mass_top50 > 137' (+14.494), 'mass > 173' (+13.392) and 'mass_over_sum_pt > 0.171' (+9.625). The sum is -0.641, so the value is 0.429 (on for 32.9%), taking 0.389 from t. The formula calls them g.

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

- **medium-mass (85 GeV), average width, pT spread over several particles** — 87.4% of jets, neuron 0.33. The bulk of all jets (87.4%), a full mixture of every class (W 22.5%, Z 22.3%, quark 20.8%, top 19%, gluon 15.4%); mass about 85 GeV, width 0.0088 and pT sharing and spread all close to the all-jet averages. 'mass_over_sum_pt < 0.098' (+0.886) and 'z_top30_slots > 0.905' (+0.643) roughly cancel 'mass < 92.9' (74.9% pass, -0.864), 'log_sum_pt > 6.9 and n_real_top30 > 22' (-0.417) and 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' (-0.337); with the intercept -0.6637 the value is small (0.334), on for 59.6%. It adds 0.042 to q and takes 0.125 from Z; the formula's decisions here are spread over all five classes.
- **very heavy (156 GeV), very wide, pT spread over several particles** — 4.9% of jets, neuron 0.00. Mostly top (55.5%) with 35.2% gluon; mass about 156 GeV, width 0.0210, total pT 1109 GeV, with pT shared out (hardest 189 GeV) and spread broadly. 'log_sum_pt > 6.9 and mass_top50 > 85.9' passes for all and removes 9.184, and 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' (96.4%) removes 4.192 more; the pT steps 'sum_pt > 1.02e+03' (+1.150) and 'sum_pt > 996' (+0.904) cannot compensate. The sum is -15.721, the neuron is off and adds nothing to the scores. The formula calls them t.
- **medium-mass (69 GeV), narrow, pT spread over several particles, high pT** — 4.4% of jets, neuron 0.19. Mostly gluon (65.8%), with quark, W and Z at 10-12% each; mass about 69 GeV, narrow (width 0.0031), high total pT (1299 GeV) with a hard leading particle (306 GeV) and 58% of the pT within 0.025 of the axis. Large high-pT terms of both signs nearly cancel: 'sum_pt > 1.07e+03' (-6.264), 'sum_pt_top50 > 1.11e+03 and z_top20_slots > 0.807' (-5.999) and 'sum_pt_top50 > 1.04e+03' (-4.230) against 'sum_pt_top50 > 1.16e+03' (+4.854), 'sum_pt > 1.02e+03' (+3.545) and 'sum_pt_top20 > 1.13e+03' (+3.487), and 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' (-2.893) tips it negative. The mean value is 0.191, on for only 7.7%, so it barely touches the scores (+0.024 q, -0.072 Z). The formula calls them g.
- **very heavy (169 GeV), very wide, pT spread over several particles, high pT** — 1.7% of jets, neuron 0.01. Mostly gluon (59.7%) with 33.4% top; mass about 169 GeV, width 0.0192, total pT 1268 GeV, pT spread broadly. 'log_sum_pt > 6.9 and mass_top50 > 85.9' grows with mass and pT and removes 26.428, with 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' (-8.768) and 'sum_pt > 1.07e+03' (-5.446) adding to it. The sum is -39.816 and the neuron is on for only 0.2%, adding essentially nothing. The formula calls them g.
- **medium-mass (67 GeV), very narrow, pT spread over several particles, high pT** — 0.9% of jets, neuron 0.52. Mostly gluon (64.8%), with W and quark near 12-13% each; mass about 67 GeV, very narrow (width 0.0020), extremely high total pT (1577 GeV) and a very hard leading particle (468 GeV), with 71% of the pT within 0.025 of the axis. Huge pT terms cancel pairwise: 'sum_pt_top20 > 1.13e+03' (+22.545) against 'sum_pt_top50 > 1.11e+03 and z_top20_slots > 0.807' (-22.527), and 'sum_pt_top50 > 1.16e+03' (+15.193) against 'sum_pt > 1.07e+03' (-13.763) and 'sum_pt_top50 > 1.04e+03' (-9.005). The mean sum is -12.848 but the value varies widely between jets: it is on for 15% (mean 0.517), adding 0.065 to q and taking 0.194 from Z. The formula calls them g.
- **very heavy (197 GeV), very wide, pT spread over several particles, high pT** — 0.5% of jets, neuron 0.00. Mostly gluon (76.1%) with 16.4% top; mass about 197 GeV, width 0.0206, total pT 1443 GeV, pT spread broadly. 'log_sum_pt > 6.9 and mass_top50 > 85.9' removes 56.353 and 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' 14.332, far more than the pT steps give back. The sum is -73.651, on for 0.3%, so the neuron adds nothing. The formula calls them g.
- **very heavy (249 GeV), very wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. Mostly gluon (88.9%); mass about 249 GeV, width 0.0238, the highest total pT here (1720 GeV). 'log_sum_pt > 6.9 and mass_top50 > 85.9' removes 125.877, backed by 'log_sum_pt > 6.9 and n_dr_0p2_0p4 > 4' (-23.309) and 'sum_pt > 1.07e+03' (-17.617). The sum is -141.123, the neuron is off and adds nothing. The formula calls them g.
- **very heavy (162 GeV), wide, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. A small mixed set (0.06% of jets): 50% gluon, 21% quark, 13.2% top; mass about 162 GeV, total pT 1309 GeV and a hard leading pair (356 and 193 GeV). 'sum_pt_top40 > 1.01e+03 and eta_5 < -0.114' passes for all and removes 102.700 on its own, the one test that defines this group, with 'log_sum_pt > 6.9 and mass_top50 > 85.9' removing 36.542 more. The sum is -154.978, the neuron is off and adds nothing. The formula calls them g.

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

- **heavy (95 GeV), average width, pT spread over several particles** — 19.4% of jets, neuron 0.17. Mostly Z (69.4%) with 13% gluon and 10.7% top; mass about 95 GeV, width 0.0084, with 48% of the pT at 0.05-0.1 from the axis. 'mass_over_sum_pt > 0.0508' (-2.007), 'mass < 121' (96.5% pass, -1.106) and 'mass < 101' (84.5%, -0.540) take off, and 'mass < 92.9' passes for only 63.5% (+0.119), so it gives little back; 'LHA > 0.228' (+1.359), 'girth2_top20 > 0.00182' (+0.641), 'mass_over_sum_pt > 0.0508 and n_dr_0p2_0p4 < 21' (+0.637) and 'mass_over_sum_pt > 0.0508 and lam2 < 0.0024' (+0.531) nearly compensate. The value is 0.173 (on for 21.6%), taking 0.168 from the Z score. The formula calls them Z.
- **medium-mass (80 GeV), narrow, pT spread over several particles** — 14.9% of jets, neuron 0.00. Mostly W (82.7%); mass about 80 GeV, width 0.0062, with 54% of the pT at 0.05-0.1 from the axis. The mass windows pass for all, and 'mass < 121' (-1.721), 'mass < 101' (-1.406) and 'mass_over_sum_pt > 0.0508' (-1.357) take off more than 'LHA > 0.228' (+1.133), 'mass < 92.9' (+0.897) and 'mass < 173' (+0.614) give back. The sum is -0.686, the neuron is on for only 0.1% and adds nothing to the scores. The formula calls them W.
- **medium-mass (85 GeV), narrow, pT spread over several particles** — 14.2% of jets, neuron 0.05. A mixture, mostly W (36.2%) with 30% Z and 21.8% gluon; mass about 85 GeV, width 0.0063, total pT 1085 GeV with 33% of the pT within 0.025 and 46% at 0.025-0.05 from the axis. 'mass < 121' (-1.505), 'mass_over_sum_pt > 0.0508' (-1.404) and 'mass < 101' (-1.086) outweigh 'e2 < 0.0476' (+0.766), 'mass < 92.9' (90%, +0.595) and 'mass < 173' (+0.580); the central pT means 'LHA > 0.228' passes for only 53.4% (+0.189). The value is 0.046 (on for 8.3%), a small cut to the Z score (-0.045). The formula calls them W.
- **light (28 GeV), very narrow, pT spread over several particles** — 10.8% of jets, neuron 0.69. Mostly quark (76.1%) with 12.3% gluon; very light (about 28 GeV), extremely narrow (width 0.0008), a very hard leading particle (346 GeV) and 91% of the pT within 0.025 of the axis. The mass tests grow with the distance below each cut: 'mass < 101' (-4.907) and 'mass < 121' (-3.938) against 'mass < 92.9' (+4.562), 'mass < 86.4' (+2.172), 'e2 < 0.0476' (+1.275) and 'lam1 < 0.00726' (+1.270); 'mass_over_sum_pt > 0.0508' passes for only 0.6%. The value is 0.687 (on for 96.1%), taking 0.666 from Z and adding 0.15 to q and 0.107 to g. The formula calls them q.
- **light (47 GeV), very narrow, pT spread over several particles** — 10.6% of jets, neuron 0.36. A quark-gluon mixture (44.5% quark, 43.6% gluon); mass about 47 GeV, width 0.0021, with 75% of the pT within 0.025 of the axis. The same light-jet balance: 'mass < 101' (-3.613) and 'mass < 121' (-3.118) against 'mass < 92.9' (+3.207), 'mass < 86.4' (+1.455), 'e2 < 0.0476' (+1.136) and 'lam1 < 0.00726' (+1.087), with 'mass_over_sum_pt > 0.0508' passing for 22.6%. The value is 0.361 (on for 75.7%), taking 0.35 from Z. The formula calls them q.
- **medium-mass (67 GeV), narrow, pT spread over several particles** — 8.6% of jets, neuron 0.11. Mostly gluon (48.4%), mixed with quark (26.4%) and W (14.9%); mass about 67 GeV, width 0.0040, 55% of the pT within 0.025 of the axis. 'mass < 101' (-2.317), 'mass < 121' (-2.298) and 'mass_over_sum_pt > 0.0508' (90.6%, -0.595) slightly outweigh 'mass < 92.9' (+1.850), 'e2 < 0.0476' (+0.989), 'lam1 < 0.00726' (+0.793) and 'mass < 86.4' (+0.738). The value is 0.11 (on for 31%), a small cut to the Z score (-0.107). The formula calls them g.
- **heavy (134 GeV), wide, pT spread over several particles** — 8.0% of jets, neuron 1.22. Mostly top (55.6%), with 30.8% gluon and 11.8% quark; mass about 134 GeV, width 0.0168, pT shared out (hardest 181 GeV) and 41% of it at 0.05-0.1 from the axis. Above the mass windows ('mass < 121' passes for only 28.6%), 'LHA > 0.228' (+2.269) and 'girth2_top20 > 0.00182' (+1.366) with smaller shape terms beat 'mass_over_sum_pt > 0.0508' (-3.854). The value is 1.223, the neuron's largest mean (on for 88.1%), taking 1.185 from Z and adding 0.268 to q and 0.191 to g. The formula calls them t.
- **very heavy (166 GeV), very wide, pT spread over several particles** — 6.2% of jets, neuron 0.32. Mostly top (85.2%); mass about 166 GeV, width 0.0273, pT shared out (hardest 165 GeV) and 17% of it at 0.2-0.3 from the axis. 'mass_over_sum_pt > 0.0508' (-5.628), 'lam1 > 0.0117' (-1.159) and 'girth2_top20 > 0.0166' (-1.144) balance 'LHA > 0.228' (+3.228) and 'girth2_top20 > 0.00182' (+2.574); 'girth2_top3 > 0.01' passes for 60.1%. The sum is near zero, so the value is 0.318 (on for 42.3%), taking 0.308 from Z. The formula calls them t.
- **very heavy (169 GeV), very wide, pT spread over several particles** — 5.2% of jets, neuron 0.74. Mostly top (84.1%); mass about 169 GeV, width 0.0283, a soft leading particle (149 GeV) and 33% of the pT at 0.1-0.15 from the axis. What sets it apart from the group above is that the three hardest particles are far apart: 'girth2_top3 > 0.01 and D2 < 4.45' (+2.611) and 'girth2_top3 > 0.01' (-2.577) both pass and cancel. The rest is the same balance, 'mass_over_sum_pt > 0.0508' (-5.762), 'girth2_top20 > 0.0166' (-1.618) and 'lam1 > 0.0117' (-1.203) against 'LHA > 0.228' (+3.753) and 'girth2_top20 > 0.00182' (+2.919). The value is 0.74 (on for 52.5%), taking 0.716 from Z. The formula calls them t.
- **very heavy (184 GeV), very wide, pT spread over several particles** — 2.1% of jets, neuron 0.94. Mostly top (76.5%) with 16.4% gluon; mass about 184 GeV, width 0.0337, with 30% of the pT at 0.15-0.2 and 28% at 0.2-0.3 from the axis. The shape tests are at their largest: 'girth2_top3 > 0.01 and D2 < 4.45' (+5.497) against 'girth2_top3 > 0.01' (-5.041), and 'mass_over_sum_pt > 0.0508' (-6.503) and 'girth2_top20 > 0.0166' (-2.644) against 'LHA > 0.228' (+4.148), 'girth2_top20 > 0.00182' (+3.665) and 'e2 > 0.0556' (+1.413). The value is 0.938 (on for 51.4%), taking 0.909 from Z. The formula calls them t.

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

- **heavy (110 GeV), wide, pT spread over several particles** — 65.1% of jets, neuron 0.06. The bulk (65.1%): a W-Z-top mixture (28.2% Z, 27% top, 25.9% W, 13.4% gluon); mass about 110 GeV, width 0.0123, ordinary pT. Being heavier than the mass cuts, almost none of the tests pass ('girth2_top50 < 0.00743' at 42.3% is the most common), so the neuron stays at about its intercept -0.2279: the value is 0.055 and it is on for 15.2%, with only tiny effects on the scores. The formula's decisions here are spread over W, Z and t.
- **light (35 GeV), very narrow, pT spread over several particles** — 6.6% of jets, neuron 1.19. Mostly quark (65.4%) with 22.5% gluon; mass about 35 GeV, width 0.0012, a hard leading particle (318 GeV) and 87% of the pT within 0.025 of the axis. 'mass < 82.9' (+3.843), 'mass_top50 < 80.4' (+2.270), 'mass < 74.3' (+2.138) and 'girth2_top50 < 0.00743' (+0.754) add, against 'mass_top50 < 77.4' (-1.803), 'girth < 0.0505' (-1.761), 'mass < 62.5' (-1.728), 'mass < 86.4' (-1.445) and 'mass_top40 < 77.9' (-0.710). The value is 1.191 (on for 98.5%), adding 0.409 to q, 0.279 to g and 0.372 to Z and taking 0.484 from W and 0.447 from t. The formula calls them q.
- **light (47 GeV), very narrow, pT spread over several particles** — 6.5% of jets, neuron 1.00. A gluon-quark mixture (45.3% gluon, 43.7% quark); mass about 47 GeV, width 0.0020, 77% of the pT within 0.025 of the axis. 'mass < 82.9' (+2.870), 'mass_top50 < 80.4' (+1.716) and 'mass < 74.3' (+1.478) against 'girth < 0.0505' (-1.355), 'mass_top50 < 77.4' (-1.332), 'mass < 86.4' (-1.104) and 'mass < 62.5' (-0.968). The value is 1.005 (on for 97.4%), adding 0.346 to q and 0.236 to g and taking 0.408 from W. The formula calls them q, slightly more often than g.
- **medium-mass (73 GeV), narrow, pT spread over several particles** — 6.4% of jets, neuron 0.28. A mixture: 38.7% gluon, 30.4% W and 19.1% quark; mass about 73 GeV, width 0.0047, total pT 1089 GeV and 47% of the pT within 0.025 of the axis. 'mass < 82.9' (98.1%, +0.784), 'mass_top50 < 80.4' (+0.493) and 'girth2_top50 < 0.00743' (+0.355) against 'girth < 0.0505' (69.7%, -0.458), 'mass < 86.4' (-0.373), 'girth2_top50 < 0.00743 and pt_dispersion < 0.429' (-0.348) and 'mass_top50 < 77.4' (-0.294); 'mass < 74.3' passes for 58.2%. The value is 0.278 (on for 58.3%), with small effects on the scores (-0.113 W). The formula calls them g.
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 5.9% of jets, neuron 0.85. Mostly gluon (52.5%) with 32.6% quark; mass about 60 GeV, width 0.0032, 62% of the pT within 0.025 of the axis. 'mass < 82.9' (+1.869), 'mass_top50 < 80.4' (+1.155) and 'mass < 74.3' (+0.799) against 'girth < 0.0505' (-0.883), 'mass_top50 < 77.4' (-0.855) and 'mass < 86.4' (-0.754); 'mass < 62.5' passes for 73.7% but costs little. The value is 0.852 (on for 93.6%), adding 0.293 to q and 0.2 to g and taking 0.346 from W. The formula calls them g.
- **light (23 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 1.39. Mostly quark (83.8%); mass about 23 GeV, width 0.0005, a very hard leading particle (382 GeV) and 94% of the pT within 0.025 of the axis. 'mass < 82.9' (+4.795), 'mass_top50 < 80.4' (+2.855) and 'mass < 74.3' (+2.784) are at their largest, against 'mass < 62.5' (-2.472), 'mass_top50 < 77.4' (-2.301), 'girth < 0.0505' (-2.084) and 'mass < 86.4' (-1.778). The value is 1.393, the neuron's largest (on for 98.6%), adding 0.479 to q and 0.327 to g and taking 0.566 from W and 0.522 from t. The formula calls them q.
- **heavy (115 GeV), very wide, pT spread over several particles, low pT** — 2.4% of jets, neuron 0.00. Mostly top (64.8%) with 24.2% gluon; mass about 115 GeV, width 0.0190, low total pT (864 GeV) with a soft leading particle (150 GeV). 'log_sum_pt < 6.86' passes for all and removes 1.953, and the mass cuts mostly fail ('mass < 82.9' passes for 19.7%). The sum is -1.816, so the neuron is on for only 0.8% and adds nothing to the scores. The formula calls them t.
- **light (37 GeV), very narrow, pT spread over several particles, low pT** — 0.7% of jets, neuron 0.00. A quark-gluon mixture (48.5% quark, 46.2% gluon); mass about 37 GeV, width 0.0025, low total pT (764 GeV), 74% of the pT within 0.025 of the axis. The light-jet tests add as in the light groups, 'mass < 82.9' (+3.688), 'mass_top50 < 80.4' (+2.196) and 'mass < 74.3' (+2.033) against 'mass_top50 < 77.4' (-1.740), 'mass < 62.5' (-1.607) and 'mass < 86.4' (-1.391), but 'log_sum_pt < 6.86' passes for all and removes 4.488. The sum is -2.699, the neuron is off and adds nothing to the scores. The formula splits them almost evenly between g and q.
- **heavy (91 GeV), wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.00. A mixture: 46% gluon, 35.3% top and 18.7% quark; mass about 91 GeV, width 0.0174, low total pT (719 GeV) with a soft leading particle (122 GeV). 'log_sum_pt < 6.86' passes for all and removes 5.720; 'mass < 82.9' passes for 45.5% (+0.520). The sum is -5.192, the neuron is off and adds nothing to the scores. The formula calls them g.
- **light (45 GeV), average width, pT spread over several particles, low pT** — 0.2% of jets, neuron 0.00. A gluon-quark mixture (51.3% gluon, 37.8% quark, 10.9% top); mass about 45 GeV, width 0.0086, the lowest total pT here (544 GeV). 'log_sum_pt < 6.86' grows as the pT drops and removes 11.582, far more than 'mass < 82.9' (95%, +3.077), 'mass_top50 < 80.4' (+1.825) and 'mass < 74.3' (+1.652) add. The sum is -9.435, the neuron is off and adds nothing to the scores. The formula calls them g.

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

- **medium-mass (79 GeV), narrow, pT spread over several particles** — 22.1% of jets, neuron 0.07. Mostly W (74.8%); mass about 79 GeV, width 0.0060, pT at 0.025-0.1 from the axis. 'mass_over_sum_pt < 0.0905' (99.9% pass, -2.098), 'girth2_top30 < 0.0122' (-1.580), 'mass < 91' (-1.182) and 'girth2_top50 < 0.00926' (-1.127) outweigh the upper windows 'mass_over_sum_pt < 0.098' (+1.699), 'mass_over_sum_pt < 0.118' (+1.644), 'mass_over_sum_pt < 0.141' (+1.256) and 'mass < 137' (+1.159). With the intercept 0.7369 the sum is -0.489, so the value is 0.07 (on for 18.4%), taking only 0.097 from W; the neuron enters no other class score. The formula calls them W.
- **heavy (91 GeV), average width, pT spread over several particles** — 21.6% of jets, neuron 1.02. Mostly Z (74.4%); mass about 91 GeV, width 0.0080, with 41% of the pT at 0.05-0.1 from the axis. Sitting at the cuts, 'mass_over_sum_pt < 0.0905' passes for 72% and 'mass < 91' for 54.4%, so they cost little (-0.369, -0.158), while 'mass_over_sum_pt < 0.118' (+1.166), 'mass_over_sum_pt < 0.141' (+1.022), 'mass < 137' (+0.923), 'lam2 < 0.0024' (+0.830) and 'mass_over_sum_pt < 0.098' (+0.732) add; 'girth2_top30 < 0.0122' (-1.140), 'girth2_top20 < 0.0166 and z_top50_slots > 0.97' (-0.684) and 'sum_pt_top50 > 976' (-0.625) take off. The value is 1.023, the neuron's largest (on for 96.2%), taking 1.407 from W. The formula calls them Z.
- **very heavy (150 GeV), very wide, pT spread over several particles** — 21.4% of jets, neuron 0.41. Mostly top (74%) with 16.4% gluon; mass about 150 GeV, width 0.0236, pT shared out (hardest 162 GeV) and spread to 0.05-0.3 from the axis. Being above all the mass and m/pT windows, few tests pass; 'max_dr > 0.24' (99.5%, -0.538) and 'sum_pt_top50 > 976' (51.4%, -0.335) against 'lam2 < 0.0024' (40.8%, +0.221) leave the value near the intercept, at 0.411 (on for 79.1%), taking 0.565 from W. The formula calls them t.
- **light (26 GeV), very narrow, pT spread over several particles** — 8.5% of jets, neuron 0.00. Mostly quark (79%); mass about 26 GeV, width 0.0007, a very hard leading particle (355 GeV) and 92% of the pT within 0.025 of the axis. The low m/pT and mass make 'mass_over_sum_pt < 0.0905' (-10.470), 'mass < 91' (-6.599), 'mass < 89.7' (-3.751) and 'mass < 80.4' (-3.219) large, more than 'mass_over_sum_pt < 0.098' (+6.006), 'mass_top40 < 67.7' (+4.422), 'mass_over_sum_pt < 0.118' (+3.740) and 'mass_top50 < 89.1' (+3.021) return. The sum is -5.459, the neuron is off and adds nothing to the scores. The formula calls them q.
- **light (42 GeV), very narrow, pT spread over several particles** — 8.5% of jets, neuron 0.00. Mostly quark (55.8%) with 32.3% gluon; mass about 42 GeV, width 0.0018, 79% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-7.834) and 'mass < 91' (-4.939) outweigh 'mass_over_sum_pt < 0.098' (+4.650), 'mass_over_sum_pt < 0.118' (+3.080) and 'mass_top40 < 67.7' (+2.930). The sum is -3.996, the neuron is off and adds nothing to the scores. The formula calls them q.
- **medium-mass (60 GeV), narrow, pT spread over several particles** — 7.3% of jets, neuron 0.00. A gluon-quark mixture (43.2% gluon, 37.3% quark); mass about 60 GeV, width 0.0035, 59% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-5.051) and 'mass < 91' (-3.163) outweigh 'mass_over_sum_pt < 0.098' (+3.219) and 'mass_over_sum_pt < 0.118' (+2.383). The sum is -2.548, the neuron is on for 0.5% and adds nothing to the scores. The formula splits them almost evenly between g and q.
- **medium-mass (83 GeV), narrow, pT spread over several particles, high pT** — 4.2% of jets, neuron 0.30. A mixture: 43.7% gluon, 31.7% W and 17.6% Z; mass about 83 GeV, width 0.0047, high total pT (1215 GeV), 33% of the pT within 0.025 and 33% at 0.025-0.05 from the axis. The high pT turns on 'log_sum_pt > 6.94' (+2.864) and 'sum_pt_top50 > 976' (-3.000), which cancel; then 'mass_over_sum_pt < 0.0905' (-3.527) and 'girth2_top30 < 0.0122' (-1.984) against 'mass_over_sum_pt < 0.098' (+2.435) and 'mass_over_sum_pt < 0.118' (+2.002). The sum is -0.186, so the value is 0.298 (on for 41.9%), taking 0.41 from W. The formula calls them g.
- **very heavy (159 GeV), wide, pT spread over several particles, high pT** — 3.1% of jets, neuron 0.91. Mostly gluon (71.4%) with 20.6% top; mass about 159 GeV, width 0.0157, high total pT (1304 GeV). The high pT keeps m/pT moderate, so the windows 'mass_over_sum_pt < 0.141' (75.5%, +0.482) and 'mass_over_sum_pt < 0.118' (50.6%, +0.411) partly pass, and 'log_sum_pt > 6.94' (+4.043) slightly outweighs 'sum_pt_top50 > 976' (-3.874); 'girth2_top30 < 0.0122' (-0.532) and 'max_dr > 0.24' (-0.480) take off. The value is 0.906 (on for 92.7%), taking 1.245 from W. The formula calls them g.
- **light (50 GeV), very narrow, pT spread over several particles, high pT** — 2.4% of jets, neuron 0.00. Mostly gluon (81.3%) with 16.4% quark; mass about 50 GeV, width 0.0015, high total pT (1313 GeV) and 81% of the pT within 0.025 of the axis. 'mass_over_sum_pt < 0.0905' (-8.411), 'sum_pt_top50 > 976' (-4.339) and 'mass < 91' (-4.170) outweigh 'mass_over_sum_pt < 0.098' (+4.947), 'log_sum_pt > 6.94' (+4.175) and 'mass_over_sum_pt < 0.118' (+3.224). The sum is -3.074, the neuron is on for 0.3% and adds nothing to the scores. The formula calls them g.
- **heavy (94 GeV), narrow, pT spread over several particles, high pT** — 0.9% of jets, neuron 0.19. Mostly gluon (75.7%) with 12.1% W; mass about 94 GeV, narrow (width 0.0036), the highest total pT here (1628 GeV) and 50% of the pT within 0.025 of the axis. 'sum_pt_top50 > 976' (-8.260) against 'log_sum_pt > 6.94' (+7.840), and 'mass_over_sum_pt < 0.0905' (96.1%, -5.568) against 'mass_over_sum_pt < 0.098' (+3.463) and 'mass_over_sum_pt < 0.118' (+2.484). The sum is -0.842, so the value is 0.186 (on for 28.9%), taking 0.255 from W. The formula calls them g.

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

- **medium-mass (86 GeV), average width, pT spread over several particles** — 40.0% of jets, neuron 0.52. The largest group (40%), a W-Z mixture, mostly W (41.2%) with 37.3% Z; mass about 86 GeV, width 0.0068, pT at 0.025-0.1 from the axis. 'girth2_top50 < 0.0135 and z_dr_0p2_0p4 < 0.129' (+1.490) and 'girth2_top50 < 0.0135' (-1.371) cancel, leaving 'girth < 0.121' (+1.244), 'lam2 < 0.00178' (92.1%, +0.440) and 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (56.8%, +0.401) against 'girth < 0.121 and z_dr_0p2_0p4 < 0.129' (-0.383), 'lam1 < 0.0165' (-0.346) and 'e2_sq < 0.00694' (-0.304). With the intercept -0.7328 the value is 0.517 (on for 85.9%), adding 0.291 to Z and taking 0.194 from t and 0.097 from W. The formula calls them W.
- **very heavy (150 GeV), very wide, pT spread over several particles** — 23.5% of jets, neuron 0.12. Mostly top (68.6%) with 21% gluon; mass about 150 GeV, width 0.0229, pT spread to 0.05-0.3 from the axis. The narrowness tests mostly fail ('girth < 0.121' passes for 39.2%); the small pT terms 'sum_pt < 1e+03' (51.7%, -0.481), 'sum_pt < 986' (40.9%, +0.333) and 'log_sum_pt < 7.02' (+0.308) leave the sum at -0.282. The value is 0.115 (on for 18.6%), with small effects on the scores. The formula calls them t.
- **light (36 GeV), very narrow, pT spread over several particles** — 18.9% of jets, neuron 0.36. Mostly quark (60.8%) with 28.4% gluon; mass about 36 GeV, width 0.0012, a hard leading particle (322 GeV) and 87% of the pT within 0.025 of the axis. Every narrowness test passes, in pairs that largely cancel: 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (+4.585) against 'e2_sq < 0.00694' (-3.205), 'girth2_top50 < 0.0135 and z_dr_0p2_0p4 < 0.129' (+3.128) against 'girth2_top50 < 0.0135' (-2.490), 'girth2_top5 < 0.0015 and z_dr_0p2_0p4 < 0.194' (+2.279) against 'girth2_top5 < 0.0015' (-2.544); 'girth < 0.121' (+2.421) is offset by 'mass_top50 < 71.8' (-1.488) and 'girth < 0.0505' (-1.347). The value is 0.365 (on for 84.1%), adding 0.206 to Z and taking 0.137 from t. The formula calls them q.
- **medium-mass (66 GeV), narrow, pT spread over several particles** — 12.4% of jets, neuron 0.61. A mixture, mostly gluon (44.1%) with 28.6% quark and 17% W; mass about 66 GeV, width 0.0038, 60% of the pT within 0.025 of the axis. The same cancelling pairs, 'e2_sq < 0.00694 and z_dr_0p2_0p4 < 0.129' (+2.210) against 'e2_sq < 0.00694' (-1.742) and 'girth2_top50 < 0.0135 and z_dr_0p2_0p4 < 0.129' (+2.179) against 'girth2_top50 < 0.0135' (-1.993), but with less taken off by 'girth < 0.0505' (-0.558) and 'mass_top50 < 71.8' (-0.402), so 'girth < 0.121' (+1.946) carries more weight. The value is 0.611, the neuron's largest mean (on for 82.7%), adding 0.343 to Z and taking 0.229 from t and 0.114 from W. The formula calls them g.
- **heavy (92 GeV), average width, pT spread over several particles** — 2.4% of jets, neuron 0.00. Mostly Z (58.7%), with 20.3% W and 13.1% top; mass about 92 GeV, width 0.0082, with 56% of the pT at 0.05-0.1 and 31% at 0.1-0.15 from the axis and almost nothing beyond 0.2. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' passes for all (empty outer ring, heavy 0.1-0.2 ring) and removes 8.269, far more than 'girth2_top50 < 0.0135 and z_dr_0p2_0p4 < 0.129' (+1.421) and 'girth < 0.121' (+0.857) add. The sum is -8.109, the neuron is off and adds nothing to the scores. The formula calls them Z.
- **heavy (96 GeV), average width, pT spread over several particles** — 1.6% of jets, neuron 0.00. Mostly Z (74.1%) with 10.9% top; mass about 96 GeV, width 0.0088, with 36% of the pT at 0.1-0.15 from the axis and almost nothing beyond 0.2. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' grows with the pT share in the 0.1-0.2 ring and removes 18.562. The sum is -18.498, the neuron is off and adds nothing to the scores. The formula calls them Z.
- **heavy (106 GeV), average width, pT spread over several particles** — 0.6% of jets, neuron 0.00. Mostly Z (54.7%) with 26.6% top and 11.2% gluon; mass about 106 GeV, width 0.0109, with 44% of the pT at 0.1-0.15 from the axis. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' removes 33.690 here. The sum is -33.815, the neuron is off and adds nothing to the scores. The formula calls them Z.
- **heavy (128 GeV), wide, pT spread over several particles** — 0.3% of jets, neuron 0.00. Mostly top (56.7%), with 19.4% gluon and 14.4% Z; mass about 128 GeV, width 0.0146, with 58% of the pT at 0.1-0.15 from the axis. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' removes 59.972. The sum is -60.396, the neuron is off and adds nothing to the scores. The formula calls them t.
- **heavy (131 GeV), wide, pT spread over several particles** — 0.2% of jets, neuron 0.00. Mostly top (74.5%); mass about 131 GeV, width 0.0166, with 68% of the pT at 0.1-0.15 from the axis and very little within 0.05. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' removes 101.592. The sum is -102.101, the neuron is off and adds nothing to the scores. The formula calls them t.
- **very heavy (142 GeV), wide, pT spread over several particles** — 0.1% of jets, neuron 0.00. Mostly top (66.7%) with 23.2% gluon; mass about 142 GeV, width 0.0169, with 72% of the pT at 0.1-0.15 from the axis and only 0.29% within 0.025. 'z_dr_0p2_0p4 < 0.0195 and z_dr_0p1_0p2 > 0.286' is at its largest and removes 147.060. The sum is -147.740, the neuron is off and adds nothing to the scores. The formula calls them t.
