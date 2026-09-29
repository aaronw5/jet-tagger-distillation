# What each part of the 399-term formula does (64 particles)

*the simplified formula closest to the start formula (within 0.1 point on validation jets)*. Validation accuracy 81.39%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Each jet is placed on 16 scales, and each class score adds some scales and subtracts others. Gluon and quark jets are told apart mainly by gluon-likeness (many particles sharing the pT thinly, neuron 1), which the g score adds and the q score subtracts; both light-QCD scores subtract heavy two-prong-ness (neuron 4) and add light one-prong scales (9, 6, 12). W and Z jets are marked by compact, few-particle, two-prong structure (neurons 5, 3) and by the absence of busy, wide radiation (neuron 8, which both boson scores subtract strongly). W and Z are then split by narrow mass windows around 80.4, 91.2 and 101 GeV: W-side mass (0) and the 80-91 GeV scale (11) raise the W score (0 also lowers the Z score), while the 91-101 GeV window (7) raises the Z score and mass just above the Z (14) lowers the W score. Top jets are recognised by hard particles spread far from the axis in a heavy jet (neuron 10) and by busy wide radiation (8), and the t score is pulled down strongly by the high-pT, below-top-mass scale (13).

## The 5 class scores

### score g: Many soft particles, no heavy prongs

High for jets with many particles, one-prong and without a heavy two-prong mass: gluon jets score highest (mean 3.05 for g, AUC 0.93), quark jets next, top jets near zero and W and Z jets far below.

Adds gluon-likeness (neuron 1, +43%) and light one-prong (9, +10%), with small additions from 6 and 12; subtracts heavy two-prong-ness (4, -22%), the sparse-jet scale (3, -9%) and the hard-core scale (5, -8%).

*computed:* largest for g (3.05), then q (0.76), then t (-0.03), then Z (-2.24), then W (-2.56); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light, one-prong jet with few particles

High for light one-prong jets that are not particle-rich: quark jets score highest (mean 2.68 for q, AUC 0.89), gluon jets next, top jets near zero, Z and W jets below.

Subtracts heavy two-prong-ness (neuron 4, -40%), gluon-likeness (1, -20%) and the 80-91 GeV two-prong scale (11, -7%); adds light one-prong (9, +17%), one-prong outside the Z mass (6, +7%) and lightness (12, +7%).

*computed:* largest for q (2.68), then g (1.14), then t (0.08), then Z (-0.75), then W (-0.77); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact two-prong jet at the W mass

High for compact two-prong jets with mass up to about the W: W jets score highest (mean 3.55 for W, AUC 0.97), quark and Z jets below zero, gluon and especially top jets far below.

Subtracts the busy wide-radiation scale (neuron 8, -34%) and mass just above the Z (14, -14%), plus small amounts of 7, 12 and 9; adds the hard-core scale (5, +11%), the 80-91 GeV two-prong scale (11, +8%), W-side mass (0, +7%), heavy two-prong-ness (4, +6%) and the sparse-jet scale (3, +4%).

*computed:* largest for W (3.55), then q (-0.29), then Z (-1.40), then g (-2.28), then t (-5.36); it separates W jets from the rest best (AUC 0.97: large for W)

### score Z: Compact two-prong jet at the Z mass

High for compact two-prong jets at or just above 91 GeV: Z jets score highest (mean 3.49 for Z, AUC 0.95), quark and W jets slightly below zero, gluon and especially top jets far below.

Subtracts the busy wide-radiation scale (neuron 8, -37%), one-prong outside the Z mass (6, -15%), W-side mass (0, -14%) and a little of 2; adds the hard-core scale (5, +13%), the 91-101 GeV window (7, +9%) and small amounts of 3, 12 and 15.

*computed:* largest for Z (3.49), then q (-0.17), then W (-0.29), then g (-2.07), then t (-5.71); it separates Z jets from the rest best (AUC 0.95: large for Z)

### score t: Heavy jet with widely spread hard prongs

High for heavy jets whose hard particles are spread far from the axis: top jets score highest (mean 3.36 for t, AUC 0.95), W and gluon jets slightly below zero, Z and quark jets further below.

Subtracts the high-pT, below-top-mass scale (neuron 13, -38%), the hard-core scale (5, -10%) and small amounts of 12 and 7; adds spread-out hard particles (10, +27%), the busy wide-radiation scale (8, +10%) and a little heavy two-prong-ness (4, +4%).

*computed:* largest for t (3.36), then W (-0.20), then g (-0.36), then Z (-1.10), then q (-1.35); it separates t jets from the rest best (AUC 0.95: large for t)

## How the scores are assembled (from the neurons' regimes)

The formula builds five scores from 16 neurons, each of which is effectively a set of mass, width and particle-count windows; the class with the highest score wins. The q score leans on a large bias (+1.36) and light-core neurons 9 and 12, the g score (bias -1.08) is driven by the particle-count neuron 1 (+4.85 for the 90%-gluon, many-particle regime), and both are cut by neuron 4 as soon as the 40 hardest particles exceed 74.5 GeV (-2.35 on q, -1.94 on g). W and Z share a baseline +0.865 / +1.03 from neuron 5 on 59% of jets and are then separated by mass windows: neuron 0 below 85.2 GeV (+1.52 W, -2.79 Z), neuron 7 above 86.5 GeV (+3.20 Z, -2.21 W) and neuron 14 above 87.4 GeV (-2.29 W). The top score comes from spread-out hard particles (neuron 10, up to +2.60) and busy wide jets (neuron 8, +1.96), and is braked by neuron 13 for hard jets lighter than 161 GeV (-1.90 and -2.52). The strongest single term is neuron 8's veto of top-like wide jets in the W and Z scores (-8.11 and -8.69 on 11% of jets), which is why tops average -5.36 and -5.71 on those scores.

**What goes into score g:** The gluon score starts from a bias of -1.08 and is built almost entirely from neuron 1 (many soft particles): its high-pT, more-than-47.5-particle, light-core regime adds +4.85 (5.9% of jets, 90% g), the many-particle low-pT regimes add +3.15 (7.1%) and +1.94 (19%, though that one is 63% top), and even the sparser 24% and 33% of jets get +1.23 and +0.645. Neuron 9 (light hard core, mass_top50 ≤ 65.3 GeV) adds +1.39 on 16% of jets. It is lowered by neuron 4 on heavy two-prong jets (-1.94 for compact mass_top40 > 74.5 GeV jets, -1.41 for wider ones, -0.754 on 25% many-particle heavy jets), by neuron 5's baseline -0.467 on 59% of jets and by neuron 3 (-0.891 for thin jets with little outer radiation). So a high gluon score means many particles, a light core and no heavy prongs: true gluons average 3.05, quarks 0.763, tops -0.029, W -2.56 and Z -2.24.

**What goes into score q:** The quark score has the largest bias, +1.36, so quark is the default for light jets. It is raised by neuron 9 (light hard core, mass_top50 ≤ 65.3 GeV and pT > 937 GeV: +1.51 on 16% of jets, +1.05 on 8.3%) and neuron 12 (mass ≤ 64.5 GeV with hard leading pair: +0.62 on 13%). It is lowered mainly by neuron 4 on any heavy jet with mass_top40 > 74.5 GeV (-2.35 compact, -1.71 wider, -1.87 on the 8% three-prong top regime, -0.916 on 25%), by neuron 1 for many-particle jets (-1.46 in the 90%-gluon regime, -0.944, -0.583) and by neuron 11 for clean two-prong W/Z jets (-0.68). Physically a high quark score means a light, narrow jet with few particles; quarks average 2.68 but gluons still reach 1.14, so quark/gluon is the least separated pair, while W (-0.773), Z (-0.748) and top (0.077) stay low.

**What goes into score W:** The W score starts near zero (bias 0.094) and gets a broad +0.865 from neuron 5 on 59% of jets (hard, not top-heavy), then +1.52 from neuron 0 for not-too-narrow jets with mass ≤ 85.2 GeV and a quiet outer ring (15% of jets, 87% W), +1.62 / +1.17 from neuron 11 for clean elongated two-prong jets, +0.552 from neuron 4 on heavy compact jets and +0.168 from neuron 11 on the 64% bulk. The biggest subtractions are top vetoes from neuron 8 (-8.11 for m/pT > 0.13, 11% of jets, 84% top; -5.44 and -4.86 in the next wide heavy regimes) and Z vetoes: neuron 14 takes -2.29 for mass > 87.4 GeV (24%, 68% Z) and neuron 7 -2.21 for two-prong jets at the Z mass (8.2%, 95% Z). The W score is thus high for a compact two-prong jet just below 85 GeV: W jets average 3.55, Z -1.41, g -2.28, q -0.291 and top -5.36.

**What goes into score Z:** The Z score has a bias of 0.984 and a broad +1.03 from neuron 5 on 59% of jets; its specific Z signal is neuron 7, +3.20 for two-prong jets (D2 ≤ 1.24) with mass_top50 > 86.5 GeV and a quiet outer ring (8.2% of jets, 95% Z) and +1.11 for 84.1 < mass ≤ 95.8 GeV (11%), plus +0.875 from neuron 3 on thin sparse jets. It is lowered by the same top veto as W from neuron 8 (-8.69 on 11% of jets, 84% top; -5.83, -5.21, -2.94, -1.07 in the other wide regimes), by neuron 0 on the W side (-2.79 for mass ≤ 85.2 GeV compact jets, 87% W; -1.56 on 7.4%) and by neuron 6 (-1.58 for mass > 98.1 GeV thin jets, -1.98 and -2.34 for very light cores). So a high Z score needs a clean two-prong jet between about 86.5 and 95.8 GeV: Z jets average 3.49, W -0.294, q -0.171, g -2.07 and top -5.71.

**What goes into score t:** The top score has a bias of 0.781 and is raised by neuron 10 when the pT is spread away from the core (less than 0.782 within ΔR < 0.05): +2.60 for e2 > 0.0456 and mass ≤ 179 GeV (15% of jets, 81% top), +1.67 for max_dr > 0.336 (26%) and +1.20 otherwise (20%), plus +1.96 from neuron 8 for wide heavy jets with m/pT > 0.13 (11%, 84% top). The main subtractions come from neuron 13 on hard jets lighter than 161 GeV (-1.90 on 41% of jets, -2.52 on 28%, -1.26 on 9.9% with pT ≤ 993 GeV), from neuron 5's -0.701 on 59%, from neuron 7 (-0.994 at the Z mass) and neuron 12 (-0.676 for light jets). Physically a high top score means a heavy (above 161 GeV), wide jet with hard particles spread out and busy radiation: tops average 3.36, W -0.204, g -0.363, Z -1.10 and q -1.35.

## The 16 neurons (most important first)

### neuron 1: Gluon-likeness: many soft particles (major)

- **What it measures:** Grows with the number of particles and with how thinly the pT is shared among them (a small pT share held by the 30-40 hardest particles). Gluon jets sit far highest (mean 5.89 for g, AUC 0.93), top jets next, then quark jets, with W and Z jets lowest.
- *computed — its value:* largest for g (5.89), then t (2.76), then q (2.23), then Z (1.55), then W (1.39); it separates g jets from the rest best (AUC 0.93: large for g)
- **How the class scores use it:** It raises the g score (+43%) and lowers the q score (-20%), making it the main gluon-versus-quark handle; freezing it costs 12.064 points. The W, Z and t scores hardly use it, even though top jets are fairly high on it.
- *computed — used by:* raises the score of g (+43%); lowers the score of q (-20%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** Never really off (on in 82-100% of jets); highest (7.76) for high-pT jets (log total pT > 7) with more than 47.5 particles and a light hard core (mass_top10 ≤ 35.1 GeV), which are 90% gluons (5.9% of jets), and lowest (1.03) for jets with at most 54.5 particles of which at most 39.5 are real among the top 40 (33% of jets). Every regime adds to g and subtracts from q in a fixed ratio: +4.85 / -1.46 in the gluon-rich regime, +1.94 / -0.583 in the many-particle, heavier-core regime (19% of jets, 63% top), +1.23 in the 24% with 39.5+ real particles, +0.645 in the sparse 33%. The particle count is thus the backbone of the gluon score; regime_r2 0.69.

Regimes (a small tree on its quantities; R² 0.69):

- `` — 5.9% of jets, value 7.76 (5.75…9.72), formula right 92%
- `` — 5.4% of jets, value 5.83 (3.53…8.03), formula right 78%
- `` — 2.2% of jets, value 5.11 (3.28…7.19), formula right 78%
- `` — 7.1% of jets, value 5.04 (3.16…6.88), formula right 69%
- `` — 4.7% of jets, value 3.28 (1.92…5.12), formula right 78%
- `` — 18.6% of jets, value 3.11 (1.38…4.94), formula right 79%
- `` — 23.6% of jets, value 1.97 (0.34…3.81), formula right 80%
- `` — 32.6% of jets, value 1.03 (0.00…2.31), formula right 85%

```
z = 2.19
if log_sum_pt > 6.91: z += 30.00 × (log_sum_pt − 6.91)
if log_sum_pt > 6.89: z += 18.70 × (log_sum_pt − 6.89)
if z_top50_slots > 0.959: z += -36.50 × (z_top50_slots − 0.959)
if LHA < 0.411: z += 7.47 × (0.411 − LHA)
if sum_pt_top2 < 668: z += 0.0035 × (668 − sum_pt_top2)
if log_sum_pt > 6.81: z += -6.45 × (log_sum_pt − 6.81)
if sum_pt_top50 > 960: z += -0.010 × (sum_pt_top50 − 960)
if tau32 > 0.328: z += 2.09 × (tau32 − 0.328)
if z_top20_slots < 0.952: z += -10.70 × (0.952 − z_top20_slots)
if n_pt_above_10 < 31.30: z += -0.063 × (31.30 − n_pt_above_10)
if mass_top30 < 80.70: z += 0.050 × (80.70 − mass_top30)
if mass_top50 < 117: z += -0.020 × (117 − mass_top50)
if max_dr < 0.435: z += -7.22 × (0.435 − max_dr)
if sum_pt_top30 < 1080: z += 0.0051 × (1080 − sum_pt_top30)
if mass_top30 < 80.80 and mass_top5 < 68.20: z += -0.00053 × (80.80 − mass_top30) × (68.20 − mass_top5)
if log_sum_pt > 6.99: z += -20.40 × (log_sum_pt − 6.99)
if max_dr < 0.433 and z_dr_0p2_0p4 < 0.191: z += 28.80 × (0.433 − max_dr) × (0.191 − z_dr_0p2_0p4)
if n_dr_0p2_0p4 < 13.30: z += -0.045 × (13.30 − n_dr_0p2_0p4)
if n_particles > 37.80 and dr_0 < 0.143: z += 0.297 × (n_particles − 37.80) × (0.143 − dr_0)
if z_top30_slots > 0.940 and mass_top10 < 92.70: z += -0.169 × (z_top30_slots − 0.940) × (92.70 − mass_top10)
if pt_9 < 35.20: z += -0.026 × (35.20 − pt_9)
if mass_over_sum_pt < 0.074: z += 21.70 × (0.074 − mass_over_sum_pt)
if mass_top20 < 41.10: z += -0.048 × (41.10 − mass_top20)
if lam2 < 0.00084: z += -719 × (0.00084 − lam2)
if n_particles > 37.70: z += 0.015 × (n_particles − 37.70)
if n_particles > 38.20 and z_top50_slots < 0.986: z += 1.71 × (n_particles − 38.20) × (0.986 − z_top50_slots)
if girth2_top3 < 0.00085: z += 766 × (0.00085 − girth2_top3)
if sum_pt_top2 < 701 and n_dr_0p2_0p4 < 6.95: z += -0.00024 × (701 − sum_pt_top2) × (6.95 − n_dr_0p2_0p4)
if girth2_top20 < 0.0011: z += 1080 × (0.0011 − girth2_top20)
if girth2_top3 < 0.00084 and n_dr_0p05_0p1 < 10.40: z += -102 × (0.00084 − girth2_top3) × (10.40 − n_dr_0p05_0p1)
if mass_top20 < 40.50 and n_real_top40 > 26.60: z += 0.0029 × (40.50 − mass_top20) × (n_real_top40 − 26.60)
if z_top30_slots > 0.943 and m012 > 13.70: z += 0.373 × (z_top30_slots − 0.943) × (m012 − 13.70)
if girth2_top15 < 0.0029: z += 120 × (0.0029 − girth2_top15)
if n_dr_0p1_0p2 < 7.22 and dr_6 < 0.078: z += -1.59 × (7.22 − n_dr_0p1_0p2) × (0.078 − dr_6)
if D2 < 1.18: z += 0.651 × (1.18 − D2)
if log_sum_pt > 7.13: z += -2.33 × (log_sum_pt − 7.13)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W/Z jets near 85 GeV** — 24.4% of jets, neuron 0.86, formula right for 86%. 42% W and 40% Z; mass about 85 GeV, width 0.0072, slightly below-average total pT (1008 GeV) and a two-prong radial profile with little pT within 0.025 (12%). Many tests pass for nearly all of them and largely cancel: 'z_top50_slots > 0.959' (-1.492), 'n_pt_above_10 < 31.3' (-0.899) and 'max_dr < 0.435' (-0.846) against 'LHA < 0.411' (+1.003), 'sum_pt_top2 < 668' (+0.922) and 'tau32 > 0.328' (+0.908); 'log_sum_pt > 6.91' passes for 74% and adds only 0.409. The value stays low (0.86, on for 80%) and adds 0.538 to the gluon score and removes 0.161 from the quark score, too little to matter. The formula splits them between W (46%) and Z (39%) and is right for 85.9%.
- **Wide, evenly shared top-like jets** — 17.7% of jets, neuron 2.23, formula right for 81%. 57% top, the rest spread over the other four classes; mass about 122 GeV, width 0.017, pT spread evenly (hardest particle 164 GeV against 240 for all jets) and far from the axis. 'sum_pt_top2 < 668' (+1.383) and 'sum_pt_top30 < 1.08e+03' (+0.879) reward the even pT sharing, and 'tau32 > 0.328' (+0.581) and 'n_particles > 37.8 and dr_0 < 0.143' (+0.368) add more, while 'z_top20_slots < 0.952' (-1.198, 99% pass) and 'z_top50_slots > 0.959' (-1.195) pull it down. The value, 2.229 (on for 98.2%), adds 1.393 to the gluon score and removes 0.418 from the quark score. The formula still calls them t and is right for 80.9%.
- **Light narrow quark jets** — 16.6% of jets, neuron 2.21, formula right for 74%. 66% quark and 20% gluon; light (about 38 GeV) and very narrow (width 0.0017), with a hard leading particle (296 GeV) and 83% of the pT within 0.025 of the axis. The light-mass tests 'mass_top30 < 80.7' (+2.370) and 'LHA < 0.411' (+2.111) and 'tau32 > 0.328' (+1.058) pass for all of them, while 'mass_top30 < 80.8 and mass_top5 < 68.2' (-1.611), 'mass_top50 < 117' (-1.583), 'z_top50_slots > 0.959' (-1.451) and 'n_pt_above_10 < 31.3' (-1.056) pull back. The value is 2.215 (on for 90.4%), which adds 1.384 to the gluon score and removes 0.415 from the quark score, the wrong direction for these jets. The formula still calls them q and is right for 74.3%.
- **W/Z jets with higher total pT** — 13.9% of jets, neuron 2.42, formula right for 83%. 36% W and 36% Z, with about 10% each of gluon and top; mass about 94 GeV, width 0.0084, and a total pT of 1066 GeV, a bit above average. Here 'log_sum_pt > 6.91' and 'log_sum_pt > 6.89' pass for all and add 1.857 and 1.531, which the first W/Z group mostly lacks; 'z_top50_slots > 0.959' (-1.406), 'sum_pt_top50 > 960' (-1.069) and 'log_sum_pt > 6.81' (-1.044) take part of it back. The value, 2.425 (on for 99.2%), adds 1.516 to the gluon score, which the formula overrides: it splits them between W (39%) and Z (35%) and is right for 83.1%.
- **Many-particle wide top/gluon jets** — 9.0% of jets, neuron 4.41, formula right for 75%. 52% top and 32% gluon; mass about 135 GeV, width 0.0209, and very evenly shared pT (hardest particle only 111 GeV against 240 for all jets), with most of the pT far from the axis. 'sum_pt_top2 < 668' (+1.668), 'sum_pt_top30 < 1.08e+03' (+1.372) and 'n_particles > 38.2 and z_top50_slots < 0.986' (+1.195, passes for all here) reward many soft particles, against 'z_top20_slots < 0.952' (-2.550). The value is 4.408 (always on), adding 2.755 to the gluon score and removing 0.827 from the quark score. The formula splits them between t (57%) and g (35%) and is right for 75.3%, a group where tops and gluons are often confused.
- **High-pT mixed jets near 110 GeV** — 5.8% of jets, neuron 4.93, formula right for 80%. 43% gluon with about 17% each of W, Z and top; mass about 112 GeV, width 0.0109, and a high total pT (1149 GeV). The total-pT hinges decide it: 'log_sum_pt > 6.91' (+4.101) and 'log_sum_pt > 6.89' (+2.930) against 'sum_pt_top50 > 960' (-1.779), 'log_sum_pt > 6.81' (-1.527) and 'log_sum_pt > 6.99' (-1.157); the shape tests add about 1 more net. The value, 4.927 (always on), adds 3.079 to the gluon score. The formula calls them g and is right for 80%.
- **High-pT narrow gluon/quark jets** — 4.9% of jets, neuron 4.96, formula right for 79%. 54% gluon and 39% quark; light (about 44 GeV) and narrow (width 0.0018), with a hard leading particle (303 GeV), 81% of the pT within 0.025, and a total pT of 1111 GeV. It gets both the high-pT bonus ('log_sum_pt > 6.91' +3.086, 'log_sum_pt > 6.89' +2.297) and the light-mass bonus ('mass_top30 < 80.7' +2.303, 'LHA < 0.411' +2.073), against 'mass_top30 < 80.8 and mass_top5 < 68.2' (-1.554), 'sum_pt_top50 > 960' (-1.522) and 'mass_top50 < 117' (-1.476). The value, 4.956, adds 3.097 to the gluon score and removes 0.929 from the quark score. The formula splits them between g (55%) and q (45%) and is right for 78.8%, so the quark jets here are often misread.
- **High-pT gluon jets, 90 GeV** — 4.5% of jets, neuron 6.40, formula right for 84%. 67% gluon, the rest small shares of all other classes; mass about 90 GeV, width 0.0064, and a total pT of 1256 GeV, well above the 1044 average. 'log_sum_pt > 6.91' (+6.750) and 'log_sum_pt > 6.89' (+4.582) grow with the total pT and outweigh 'log_sum_pt > 6.99' (-2.958), 'sum_pt_top50 > 960' (-2.895) and 'log_sum_pt > 6.81' (-2.096). The value, 6.398, adds 3.999 to the gluon score and removes 1.2 from the quark score. The formula calls them g and is right for 84.4%.
- **Very high-pT gluon jets** — 2.4% of jets, neuron 7.47, formula right for 88%. 77% gluon; mass about 105 GeV, width 0.0071, and a total pT of 1411 GeV, with harder particles at every rank than average. Same total-pT hinges, now larger: 'log_sum_pt > 6.91' (+10.233) and 'log_sum_pt > 6.89' (+6.752) against 'log_sum_pt > 6.99' (-5.326), 'sum_pt_top50 > 960' (-4.427) and 'log_sum_pt > 6.81' (-2.845). The value, 7.474, adds 4.671 to the gluon score and removes 1.401 from the quark score. The formula calls them g and is right for 87.5%.
- **Highest-pT gluon jets** — 0.9% of jets, neuron 8.42, formula right for 91%. 82% gluon; mass about 112 GeV, width 0.0058, and the highest total pT (1700 GeV), with a hardest particle of 364 GeV. 'log_sum_pt > 6.91' (+15.725) and 'log_sum_pt > 6.89' (+10.176) dominate 'log_sum_pt > 6.99' (-9.061), 'sum_pt_top50 > 960' (-7.343) and 'log_sum_pt > 6.81' (-4.026), giving the neuron's highest value, 8.417. This adds 5.261 to the gluon score and removes 1.578 from the quark score; the formula calls them g and is right for 90.6%.

### neuron 4: Heavy two-prong-ness (major)

- **What it measures:** Grows with the mass of the hardest 10-20 particles and with e2, and with small D2 (a two-prong pattern); jets lighter than 80.4 GeV or with a broad, soft spread (large LHA) are pushed down. Z and W jets sit highest, top jets next, gluon jets low and quark jets lowest (AUC 0.16 for q: small for q).
- *computed — its value:* largest for Z (1.68), then W (1.63), then t (1.15), then g (0.31), then q (0.15); it separates q jets from the rest best (AUC 0.16: small for q)
- **How the class scores use it:** It lowers the q score (-40%) and the g score (-22%) strongly, since a heavy pronged jet is not a light QCD jet, and it raises the W (+6%) and t (+4%) scores a little. The Z score hardly uses it.
- *computed — used by:* raises the score of W (+6%), t (+4%); lowers the score of g (-22%), q (-40%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** Off (0.0) for light jets (mass_top40 ≤ 74.5 GeV and mass ≤ 69.1 GeV, 27% of jets); high (2.21) when mass_top40 > 74.5 GeV with at most 47.5 particles and all particles within ΔR 0.317 of the axis (14%, 51% W and 46% Z), 1.61 for the same with max_dr > 0.317 (19%), 1.76 for many-particle jets with a three-prong look (tau32 ≤ 0.441, 8%, 86% top) and 0.862 for many-particle jets with tau32 > 0.441 (25%). It is the main penalty on gluon and quark scores: -1.94 and -2.35 in the compact heavy regime, -1.41 and -1.71 in the wider one, while adding +0.76 to W and +0.415 to t. regime_r2 0.748.

Regimes (a small tree on its quantities; R² 0.748):

- `` — 13.6% of jets, value 2.21 (1.75…2.69), formula right 96%
- `` — 8.0% of jets, value 1.76 (1.06…2.44), formula right 93%
- `` — 19.2% of jets, value 1.61 (0.75…2.19), formula right 85%
- `` — 24.9% of jets, value 0.86 (0.00…1.69), formula right 76%
- `` — 2.0% of jets, value 0.79 (0.07…1.44), formula right 78%
- `` — 3.4% of jets, value 0.27 (0.00…0.75), formula right 70%
- `` — 2.0% of jets, value 0.07 (0.00…0.25), formula right 60%
- `` — 26.9% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = 1.87
if mass < 80.40: z += -0.128 × (80.40 − mass)
if LHA > 0.114: z += -7.88 × (LHA − 0.114)
if girth < 0.121: z += -20.30 × (0.121 − girth)
if mass < 101: z += 0.038 × (101 − mass)
if D2 < 6.92: z += 0.185 × (6.92 − D2)
if girth2_top15 < 0.016: z += 72.30 × (0.016 − girth2_top15)
if width > 0.0097: z += 163 × (width − 0.0097)
if mass < 120: z += 0.013 × (120 − mass)
if girth2_top40 < 0.013: z += 73.60 × (0.013 − girth2_top40)
if mass_top40 < 84.30 and D2 < 6.61: z += -0.014 × (84.30 − mass_top40) × (6.61 − D2)
if lam1 > 0.0076: z += -158 × (lam1 − 0.0076)
if z_dr_0p2_0p4 < 0.069: z += -10.30 × (0.069 − z_dr_0p2_0p4)
if mass_top40 < 95.90: z += -0.017 × (95.90 − mass_top40)
if girth2_top15 < 0.0073: z += -132 × (0.0073 − girth2_top15)
if mass < 122 and max_dr < 0.403: z += 0.141 × (122 − mass) × (0.403 − max_dr)
if girth < 0.061: z += -26.50 × (0.061 − girth)
if n_dr_0p2_0p4 < 15.40: z += 0.042 × (15.40 − n_dr_0p2_0p4)
if sum_pt < 1070: z += -0.0054 × (1070 − sum_pt)
if n_particles > 28.20: z += -0.013 × (n_particles − 28.20)
if mass < 74.40 and max_dr < 0.388: z += -0.535 × (74.40 − mass) × (0.388 − max_dr)
if e2 > 0.037: z += 40.80 × (e2 − 0.037)
if n_dr_0p2_0p4 < 14.60 and max_dr < 0.436: z += -0.209 × (14.60 − n_dr_0p2_0p4) × (0.436 − max_dr)
if z_dr_0p2_0p4 < 0.194: z += 1.09 × (0.194 − z_dr_0p2_0p4)
if tau21 < 0.465: z += -1.24 × (0.465 − tau21)
if planar_flow < 0.425: z += 0.978 × (0.425 − planar_flow)
if lam2 > 0.0024: z += -173 × (lam2 − 0.0024)
if tau32 < 0.577: z += 2.23 × (0.577 − tau32)
if C2 > 0.110: z += 15.40 × (C2 − 0.110)
if mass > 143: z += 0.015 × (mass − 143)
if mass_top20 > 106: z += -0.016 × (mass_top20 − 106)
if z_top5_slots > 0.569: z += -0.777 × (z_top5_slots − 0.569)
if z_top5_slots < 0.548: z += 0.793 × (0.548 − z_top5_slots)
if planar_flow < 0.303 and n_dr_0p05_0p1 > 6.21: z += -0.104 × (0.303 − planar_flow) × (n_dr_0p05_0p1 − 6.21)
if n_dr_0p2_0p4 < 15.20 and n_dr_0p1_0p2 > 11.90: z += -0.0012 × (15.20 − n_dr_0p2_0p4) × (n_dr_0p1_0p2 − 11.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W/Z two-prong jets** — 32.6% of jets, neuron 1.79, formula right for 90%. 44% Z and 42% W; mass about 87 GeV, width 0.0071, and a two-prong profile with 51% of the pT at 0.05-0.1 from the axis and only 2% within 0.025. 'D2 < 6.92' (+1.059), 'girth2_top15 < 0.0159' (+0.692), 'mass < 101' (+0.550), 'mass < 122 and max_dr < 0.403' (+0.515) and 'n_dr_0p2_0p4 < 15.4' (+0.512) outweigh 'LHA > 0.114' (-1.436) and 'girth < 0.121' (-0.942); 'mass < 80.4' passes for only 26%. The value, 1.793 (on for 99.1%), adds 0.616 to W and 0.336 to top and removes 1.569 from the gluon and 1.905 from the quark scores. The formula splits them between Z (45%) and W (43%) and is right for 89.9%.
- **Boson-mass jets with hard core** — 17.4% of jets, neuron 0.98, formula right for 76%. A mixture: 30% gluon, 28% W, 25% Z and 11% quark; mass about 87 GeV, width 0.0066, but a harder leading particle (269 GeV) and 36% of the pT within 0.025, more core-dominated than true two-prong jets. Similar tests to the W/Z group, but 'girth < 0.121' (-1.409) and 'LHA > 0.114' (-0.887) are only partly offset by 'girth2_top15 < 0.0159' (+0.886), 'D2 < 6.92' (+0.616) and 'mass < 101' (+0.602), and 'girth2_top15 < 0.00725' (98% pass, -0.478) marks the narrow core. The value, 0.979, adds 0.336 to W and removes 0.856 from the gluon score. The formula splits them between W (34%) and g (33%) and is right for only 75.6%, one of the least clean groups.
- **Top-mass three-prong jets** — 10.8% of jets, neuron 1.38, formula right for 90%. 87% top; mass about 165 GeV, width 0.0269, pT shared out (hardest particle 160 GeV) and mostly far from the axis. 'width > 0.00971' (+2.807), 'e2 > 0.0369' (+0.922) and 'D2 < 6.92' (+0.880) outweigh 'LHA > 0.114' (-2.336) and 'lam1 > 0.00757' (-2.264), which grow with the spread. The value, 1.379 (on for 94.7%), adds 0.474 to W and 0.259 to top and removes 1.207 from the gluon and 1.465 from the quark scores. The formula calls them t and is right for 90.2%.
- **130 GeV top/gluon mixture** — 8.5% of jets, neuron 0.55, formula right for 70%. 55% top and 31% gluon; mass about 131 GeV, width 0.0163, pT fairly evenly shared (hardest particle 178 GeV) and spread wide. 'LHA > 0.114' (-1.857) and 'lam1 > 0.00757' (-0.992) are balanced by 'width > 0.00971' (+1.070) and 'D2 < 6.92' (+0.857); 'sum_pt < 1.07e+03' (-0.418) and 'n_particles > 28.2' (-0.387) take a little off. The value, 0.554 (on for 79.9%), adds 0.191 to W and 0.104 to top. The formula calls them t but is right for only 70.2%, with many gluons mistaken.
- **Sub-80 GeV gluon/quark jets** — 7.8% of jets, neuron 0.00, formula right for 68%. 53% gluon and 29% quark; mass about 62 GeV, width 0.0036, and 53% of the pT within 0.025 of the axis. 'mass < 80.4' grows with how far the mass is below 80.4 GeV and subtracts 2.285, joined by 'girth < 0.121' (-1.682) and 'mass_top40 < 84.3 and D2 < 6.61' (-1.327), against 'mass < 101' (+1.457) and 'girth2_top15 < 0.0159' (+1.034). The sum stays below zero almost always (value 0.003, on for 1.1%), so this neuron gives nothing. The formula calls them g, but it is right for only 68.2%.
- **Very light quark jets** — 6.6% of jets, neuron 0.00, formula right for 80%. 80% quark; very light (about 28 GeV), extremely narrow (width 0.0008), a very hard leading particle (386 GeV) and 94% of the pT within 0.025. 'mass < 80.4' subtracts 6.692, far more than 'mass < 101' (+2.762) and the other positive tests can return, and 'girth < 0.121' (-2.253) adds to the deficit. The neuron is never on here (value 0.0), so it has no effect. The formula calls them q and is right for 80.3%.
- **Light quark/gluon jets, 40 GeV** — 6.6% of jets, neuron 0.00, formula right for 75%. 49% quark and 43% gluon; light (about 40 GeV), narrow (width 0.0016), and 73% of the pT within 0.025. 'mass < 80.4' (-5.208) and 'mass_top40 < 84.3 and D2 < 6.61' (-2.681, passing for all) outweigh 'mass < 101' (+2.323) and the smaller positive tests; the value is 0.0. The formula splits them between q (57%) and g (43%) and is right for 74.6%; the neuron does not help separate them.
- **Light hard-core quark jets** — 5.2% of jets, neuron 0.00, formula right for 70%. 51% quark and 34% gluon; mass about 48 GeV, narrow (width 0.0022), a hard leading particle (329 GeV) and 88% of the pT within 0.025. 'mass < 80.4' (-4.174) and 'girth < 0.121' (-2.065) outweigh 'mass < 101' (+2.017); unlike the 40 GeV group, 'mass_top40 < 84.3 and D2 < 6.61' passes for only 45%. The value is 0.0 so the neuron is silent. The formula calls them q and is right for 70.3%.
- **Heavy wide top jets** — 2.8% of jets, neuron 1.44, formula right for 80%. 69% top and 23% gluon; mass about 188 GeV, width 0.0358, and almost all pT far from the axis (1% within 0.025). 'width > 0.00971' (+4.251), 'e2 > 0.0369' (+1.213), 'D2 < 6.92' (+0.901) and 'mass > 143' (+0.665) outweigh 'lam1 > 0.00757' (-3.601) and 'LHA > 0.114' (-2.650). The value, 1.435, adds 0.493 to W and 0.269 to top and removes 1.256 from the gluon and 1.525 from the quark scores. The formula calls them t and is right for 80.4%.
- **Very light compact quark jets** — 1.8% of jets, neuron 0.00, formula right for 84%. 83% quark; very light (about 22 GeV), extremely narrow (width 0.0005), and 90% of the pT within 0.025. 'mass < 80.4' (-7.406) and 'mass < 74.4 and max_dr < 0.388' (-4.604, passing for all only here) far outweigh 'mass < 101' (+2.974) and 'mass < 122 and max_dr < 0.403' (+2.534); the value is 0.0. The formula calls them q and is right for 84.5%.

### neuron 5: Hard-core, few-particle jets (major)

- **What it measures:** Rises when a few of the hardest particles carry most of the pT (fewer particles in total) and for mass above 61.6 GeV; it is also shaped by many thresholds on the total pT between about 900 and 1020 GeV. Quark and Z jets sit highest, W jets next, top and gluon jets lowest (AUC 0.27 for g: small for g).
- *computed — its value:* largest for q (1.44), then Z (1.40), then W (1.13), then t (0.56), then g (0.50); it separates g jets from the rest best (AUC 0.27: small for g)
- **How the class scores use it:** It raises the W (+11%) and Z (+13%) scores and lowers the g (-8%) and t (-10%) scores: a compact, few-particle jet with some mass is boson-like, not gluon- or top-like. The q score hardly uses it, although quark jets sit high on it.
- *computed — used by:* raises the score of W (+11%), Z (+13%); lowers the score of g (-8%), t (-10%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** On (1.50) for the bulk of jets: log total pT ≤ 7 with the 50 hardest carrying more than 969 GeV and mass_top50 ≤ 160 GeV (59% of jets, mixed W/Z/q); it falls to 0.26-0.78 when the 50 hardest carry less than 969 GeV, to 0.138 when mass_top50 > 160 GeV (92% top) and is nearly off (0.024) for jets with log total pT > 7.03 and max_dr ≤ 0.396 (11%). Because the big 59% regime adds +1.03 to Z, +0.865 to W, -0.701 to t and -0.467 to g, it acts as a broad baseline tilt toward W/Z and away from top and gluon. regime_r2 0.567.

Regimes (a small tree on its quantities; R² 0.567):

- `` — 59.0% of jets, value 1.50 (0.81…2.25), formula right 82%
- `` — 6.6% of jets, value 0.78 (0.00…1.44), formula right 76%
- `` — 2.0% of jets, value 0.67 (0.00…1.38), formula right 76%
- `` — 3.6% of jets, value 0.36 (0.00…0.94), formula right 81%
- `` — 10.8% of jets, value 0.26 (0.00…0.69), formula right 73%
- `` — 2.0% of jets, value 0.24 (0.00…0.75), formula right 82%
- `` — 5.1% of jets, value 0.14 (0.00…0.50), formula right 94%
- `` — 10.9% of jets, value 0.02 (0.00…0.00), formula right 84%

```
z = -1.06
if sum_pt_top40 > 905: z += 0.014 × (sum_pt_top40 − 905)
if log_sum_pt > 6.91: z += -32.90 × (log_sum_pt − 6.91)
if log_sum_pt > 6.85: z += 12.70 × (log_sum_pt − 6.85)
if sum_pt_top40 > 1010: z += -0.019 × (sum_pt_top40 − 1010)
if sum_pt_top40 < 1010: z += 0.022 × (1010 − sum_pt_top40)
if mass > 61.60: z += 0.019 × (mass − 61.60)
if sum_pt_top50 < 1020: z += -0.022 × (1020 − sum_pt_top50)
if mass_top30 > 68.90: z += 0.026 × (mass_top30 − 68.90)
if sum_pt_top50 > 1020: z += 0.011 × (sum_pt_top50 − 1020)
if mass_top50 > 90.50: z += -0.021 × (mass_top50 − 90.50)
if mass_top50 > 156: z += -0.166 × (mass_top50 − 156)
if n_particles < 61.70 and e2 < 0.037: z += 1.27 × (61.70 − n_particles) × (0.037 − e2)
if z_top30_slots > 0.933: z += -5.52 × (z_top30_slots − 0.933)
if sum_pt_top50 > 1060: z += -0.0061 × (sum_pt_top50 − 1060)
if max_dr > 0.435: z += 20.40 × (max_dr − 0.435)
if mass_top30 > 107: z += -0.024 × (mass_top30 − 107)
if mass_over_sum_pt > 0.089: z += -8.94 × (mass_over_sum_pt − 0.089)
if sum_pt_top10 > 840: z += 0.0034 × (sum_pt_top10 − 840)
if mass_top10 < 62.50: z += 0.0054 × (62.50 − mass_top10)
if girth2_top30 < 0.019 and tau32 < 0.866: z += 92.70 × (0.019 − girth2_top30) × (0.866 − tau32)
if mass_over_sum_pt_sq > 0.029: z += -444 × (mass_over_sum_pt_sq − 0.029)
if n_particles < 64.00 and mass_top5 > 11.10: z += -0.00026 × (64.00 − n_particles) × (mass_top5 − 11.10)
if mass > 173: z += -0.107 × (mass − 173)
if log_sum_pt > 7.14: z += 11.40 × (log_sum_pt − 7.14)
if sum_pt_top3 > 318 and n_dr_0p1_0p2 > 7.87: z += 7.6e-05 × (sum_pt_top3 − 318) × (n_dr_0p1_0p2 − 7.87)
if max_dr > 0.278 and tau21 < 0.582: z += -2.61 × (max_dr − 0.278) × (0.582 − tau21)
if z_top30_slots < 0.864: z += -7.78 × (0.864 − z_top30_slots)
if sum_pt_top10 > 994: z += -0.0032 × (sum_pt_top10 − 994)
if mass_top50 > 157 and z_dr_0p05_0p1 > 0.589: z += 1.55 × (mass_top50 − 157) × (z_dr_0p05_0p1 − 0.589)
if sum_pt_top3 > 794: z += -0.0031 × (sum_pt_top3 − 794)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Typical-pT boson and quark jets** — 45.7% of jets, neuron 1.56, formula right for 82%. 30% W, 30% Z and 25% quark; mass about 74 GeV, width 0.0059, a slightly low total pT (1015 GeV), and ordinary pT sharing; 45.7% of all jets. 'sum_pt_top40 > 905' (+1.446) and 'log_sum_pt > 6.85' (+0.918) pass for all, while 'log_sum_pt > 6.91' passes for 75% and costs 0.523; 'mass > 61.6' (+0.359) and 'n_particles < 61.7 and e2 < 0.0374' (+0.343) add a bit. The value, 1.56, adds 0.902 to W and 1.073 to Z and removes 0.488 from the gluon and 0.731 from the top scores. The formula spreads them over W (31%), q (29%) and Z (28%) and is right for 81.6%.
- **Mixed jets, slightly high pT** — 17.2% of jets, neuron 0.90, formula right for 81%. A mixture: 26% W, 26% Z, 25% gluon and 18% quark; mass about 80 GeV, width 0.0062, and a total pT of 1083 GeV, a bit above average. 'log_sum_pt > 6.91' passes for all and subtracts 2.546, with 'sum_pt_top40 > 1.01e+03' (-1.129), against 'sum_pt_top40 > 905' (+2.318), 'log_sum_pt > 6.85' (+1.745) and 'sum_pt_top50 > 1.02e+03' (+0.661). The value, 0.902 (on for 88.5%), adds 0.522 to W and 0.62 to Z. The formula spreads them over g and W (26% each) and Z (24%) and is right for 80.9%.
- **Low-pT top-like jets** — 13.8% of jets, neuron 0.63, formula right for 75%. 55% top and 22% gluon; mass about 108 GeV, width 0.0148, low total pT (949 GeV) and evenly shared pT (hardest particle 167 GeV). 'sum_pt_top40 < 1.01e+03' (+2.385) and 'sum_pt_top50 < 1.02e+03' (-1.945) nearly cancel; 'mass > 61.6' (+0.952) and 'mass_top30 > 68.9' (+0.739) add, and 'mass_top50 > 90.5' (-0.509) subtracts. The value, 0.632 (on for 72.4%), adds 0.366 to W and 0.435 to Z and removes 0.296 from the top score. The formula calls them t and is right for 74.7%.
- **High-pT gluon jets, neuron suppressed** — 7.5% of jets, neuron 0.07, formula right for 82%. 55% gluon, the rest spread thin; mass about 87 GeV, width 0.0064, and a total pT of 1198 GeV. 'log_sum_pt > 6.91' (-5.861) and 'sum_pt_top40 > 1.01e+03' (-3.082) grow with the total pT and outweigh 'sum_pt_top40 > 905' (+3.762), 'log_sum_pt > 6.85' (+3.025) and 'sum_pt_top50 > 1.02e+03' (+1.919). The value is 0.065 (on for 10.5%), so these jets do not get the W/Z push. The formula calls them g and is right for 82.3%.
- **Top-mass jets** — 7.3% of jets, neuron 0.14, formula right for 92%. 89% top; mass about 175 GeV, width 0.03, pT shared out (hardest particle 163 GeV) and mostly far from the axis. The mass tests fight: 'mass_top50 > 156' (-2.583), 'mass_top50 > 90.5' (-1.662) and 'mass_top30 > 107' (-1.161) against 'mass_top30 > 68.9' (+2.250) and 'mass > 61.6' (+2.173); the pT tests largely cancel too. The value is small (0.145, on for 20.6%) and removes only 0.068 from the top score. The formula calls them t and is right for 91.9%.
- **Very high-pT gluon jets** — 3.2% of jets, neuron 0.03, formula right for 87%. 72% gluon; mass about 91 GeV, width 0.0054, and a total pT of 1369 GeV. 'log_sum_pt > 6.91' (-10.216) and 'sum_pt_top40 > 1.01e+03' (-6.199) outweigh 'sum_pt_top40 > 905' (+6.047), 'log_sum_pt > 6.85' (+4.705) and 'sum_pt_top50 > 1.02e+03' (+3.826), so the value is 0.032 and the neuron has no effect. The formula calls them g and is right for 86.6%.
- **Low-pT gluon/top/quark mixture** — 2.6% of jets, neuron 0.17, formula right for 64%. 43% gluon, 33% top and 23% quark; mass about 86 GeV, width 0.0145, and a very low total pT (773 GeV), with a hardest particle of only 136 GeV. 'sum_pt_top40 < 1.01e+03' (+6.232) and 'sum_pt_top50 < 1.02e+03' (-5.724) grow as the pT falls and nearly cancel, leaving the mass tests ('mass > 61.6' +0.607) to set a small value of 0.171 (on for 37.3%). The scores barely move; the formula splits them between g (48%) and t (35%) and is right for only 63.6%, one of the harder groups.
- **Jets with a far-out particle** — 1.3% of jets, neuron 1.97, formula right for 82%. A mixture of W (26%), quark (25%), Z (23%) and top (15%); mass about 81 GeV, width 0.0075, ordinary pT; only 1.3% of jets. 'max_dr > 0.435' passes for all and adds 8.113, which is what sets this group apart (elsewhere it passes for at most 12%); with 'sum_pt_top40 > 905' (+1.590) the value reaches 1.971. That adds 1.14 to W and 1.355 to Z and removes 0.924 from top and 0.616 from gluon. The formula calls them q (30%), with W 26%, and is right for 81.8%.
- **Very high-pT gluon jets** — 0.9% of jets, neuron 0.02, formula right for 90%. 80% gluon; mass about 94 GeV, width 0.0038, total pT 1679 GeV, and 54% of the pT within 0.025. 'log_sum_pt > 6.91' (-16.819) and 'sum_pt_top40 > 1.01e+03' (-12.033) outweigh 'sum_pt_top40 > 905' (+10.325), 'sum_pt_top50 > 1.02e+03' (+7.336) and 'log_sum_pt > 6.85' (+7.254), so the value is 0.023 and the neuron stays out of the decision. The formula calls them g and is right for 89.9%.
- **Very heavy high-pT gluon jets** — 0.5% of jets, neuron 0.02, formula right for 77%. 76% gluon and 14% top; very heavy (about 237 GeV), wide (width 0.0328), total pT 1337 GeV and almost no pT near the axis. 'mass_top50 > 156' (-11.937), 'log_sum_pt > 6.91' (-9.175) and 'mass > 173' (-6.904) swamp all the positive pT and mass tests, so the value is 0.024. The formula calls them g and is right for 77.2%.

### neuron 8: Busy, wide radiation pattern (major)

- **What it measures:** Grows with the number and pT share of particles at 0.2 <= ΔR < 0.4, with the minor-axis width lam2 and with the total particle count: radiation spread over a broad area. Top jets sit far highest (mean 6.60 for t, AUC 0.91), gluon jets next, quark jets lower, Z and W jets near zero.
- *computed — its value:* largest for t (6.60), then g (2.24), then q (1.00), then Z (0.28), then W (0.08); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W (-34%) and Z (-37%) scores, because a two-prong boson is compact with an empty outer ring, and raises the t score (+10%). The g and q scores hardly use it.
- *computed — used by:* raises the score of t (+10%); lowers the score of W (-34%), Z (-37%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** Very high (9.27) for heavy wide jets with m/pT > 0.13 and the 50 hardest carrying ≤ 1 TeV (11% of jets, 84% top), 6.22 for 0.0992 < m/pT ≤ 0.13 (4.2%, 56% top) and 5.55 when the 50 hardest carry 1-1.05 TeV (4.5%, 72% top); low (0.177) for m/pT ≤ 0.0992, 40 hardest above 871 GeV and thin minor axis lam2 ≤ 0.00094 (59% of jets), 1.14 when lam2 > 0.00094 (14%). It is the strongest veto in the formula: -8.11 on W and -8.69 on Z in the top regime (with only +1.96 to t), -5.44 / -5.83 and -4.86 / -5.21 in the next ones. regime_r2 0.807.

Regimes (a small tree on its quantities; R² 0.807):

- `` — 11.0% of jets, value 9.27 (5.06…13.69), formula right 85%
- `` — 4.2% of jets, value 6.22 (3.50…9.69), formula right 70%
- `` — 4.5% of jets, value 5.55 (3.19…7.31), formula right 81%
- `` — 2.0% of jets, value 4.52 (0.00…9.75), formula right 65%
- `` — 4.8% of jets, value 3.14 (1.88…4.50), formula right 77%
- `` — 13.9% of jets, value 1.14 (0.00…2.62), formula right 74%
- `` — 59.5% of jets, value 0.18 (0.00…0.62), formula right 84%

```
z = 0.672
if mass > 101: z += -0.107 × (mass − 101)
if mass > 80.40: z += 0.058 × (mass − 80.40)
if mass_over_sum_pt < 0.099: z += -46.70 × (0.099 − mass_over_sum_pt)
if girth < 0.097: z += 24.60 × (0.097 − girth)
if mass_over_sum_pt > 0.075 and sum_pt < 1110: z += 0.355 × (mass_over_sum_pt − 0.075) × (1110 − sum_pt)
if mass_top50 > 82.60: z += -0.044 × (mass_top50 − 82.60)
if n_dr_0p2_0p4 < 20.50: z += -0.061 × (20.50 − n_dr_0p2_0p4)
if mass > 91.20: z += 0.048 × (mass − 91.20)
if mass_over_sum_pt > 0.089: z += 34.00 × (mass_over_sum_pt − 0.089)
if mass_over_sum_pt_sq < 0.013: z += -70.90 × (0.013 − mass_over_sum_pt_sq)
if lam2 < 0.0015: z += -585 × (0.0015 − lam2)
if mass_top50 > 98.10: z += 0.035 × (mass_top50 − 98.10)
if girth2_top30 < 0.0076: z += 162 × (0.0076 − girth2_top30)
if girth2_top20 > 0.0081: z += 130 × (girth2_top20 − 0.0081)
if z_dr_0p2_0p4 < 0.091: z += 6.20 × (0.091 − z_dr_0p2_0p4)
if sum_pt < 1010: z += 0.015 × (1010 − sum_pt)
if n_particles > 50.70: z += 0.068 × (n_particles − 50.70)
if e2 > 0.026: z += -20.90 × (e2 − 0.026)
if mass < 64.60: z += 0.033 × (64.60 − mass)
if mass > 140: z += -0.037 × (mass − 140)
if sum_pt_top50 < 1080: z += -0.0021 × (1080 − sum_pt_top50)
if girth2_top40 > 0.0052 and sum_pt_top30 < 912: z += 0.710 × (girth2_top40 − 0.0052) × (912 − sum_pt_top30)
if n_particles > 51.10 and z_top50_slots > 0.971: z += -2.94 × (n_particles − 51.10) × (z_top50_slots − 0.971)
if sum_pt_top40 < 995: z += 0.0044 × (995 − sum_pt_top40)
if sum_pt_top40 < 967 and log_sum_pt < 6.82: z += 0.068 × (967 − sum_pt_top40) × (6.82 − log_sum_pt)
if z_dr_0p1_0p2 < 0.144: z += 1.52 × (0.144 − z_dr_0p1_0p2)
if D2 < 1.81: z += 0.305 × (1.81 − D2)
if sum_pt_top40 < 1010 and max_dr < 0.392: z += -0.085 × (1010 − sum_pt_top40) × (0.392 − max_dr)
if sum_pt < 1010 and n_real_top50 < 43.50: z += -0.0012 × (1010 − sum_pt) × (43.50 − n_real_top50)
if lam2 < 0.0014 and n_dr_0p05_0p1 > 4.52: z += 17.50 × (0.0014 − lam2) × (n_dr_0p05_0p1 − 4.52)
if C2 > 0.072: z += 4.39 × (C2 − 0.072)
if mass > 85.70 and log_sum_pt < 6.91: z += 0.112 × (mass − 85.70) × (6.91 − log_sum_pt)
if sum_pt_top40 > 1060: z += 0.0013 × (sum_pt_top40 − 1060)
if n_dr_0p2_0p4 < 15.90 and z_dr_0p1_0p2 > 0.300: z += 0.150 × (15.90 − n_dr_0p2_0p4) × (z_dr_0p1_0p2 − 0.300)
if z_dr_0p1_0p2 > 0.212 and mean_phi > 0.00054: z += -1950 × (z_dr_0p1_0p2 − 0.212) × (mean_phi − 0.00054)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W/Z jets, quiet outer ring** — 50.2% of jets, neuron 0.53, formula right for 84%. 37% W and 37% Z, with 12% gluon; mass about 84 GeV, width 0.0067, ordinary pT, and pT held at 0.025-0.1 from the axis; half of all jets. 'n_dr_0p2_0p4 < 20.5' (-0.893), 'mass_over_sum_pt < 0.0988' (-0.818), 'lam2 < 0.00149' (-0.497) and 'mass_over_sum_pt_sq < 0.0135' (-0.482) outweigh 'girth < 0.0968' (+0.772) and 'z_dr_0p2_0p4 < 0.0913' (+0.422); 'mass > 80.4' passes for 65% (+0.326). The value, 0.533 (on for 41.5%), removes 0.467 from W and 0.5 from Z. The formula splits them between W (40%) and Z (37%) and is right for 83.6%.
- **Light gluon/quark jets** — 24.7% of jets, neuron 0.25, formula right for 77%. 55% quark and 33% gluon; light (about 42 GeV), narrow (width 0.0017), a hard leading particle (307 GeV) and 81% of the pT within 0.025. 'mass_over_sum_pt < 0.0988' (-2.798), 'n_dr_0p2_0p4 < 20.5' (-0.898) and 'mass_over_sum_pt_sq < 0.0135' (-0.839) are balanced by 'girth < 0.0968' (+1.891), 'girth2_top30 < 0.00763' (+1.036) and 'mass < 64.6' (+0.776). The value is 0.25 (on for 43.6%), with only small score changes. The formula calls them q and is right for 76.7%.
- **Top-mass jets, neuron maximal** — 7.4% of jets, neuron 9.40, formula right for 94%. 92% top; mass about 171 GeV, width 0.0298, pT shared out (hardest particle 152 GeV) and mostly far from the axis. The mass tests pass for all: 'mass > 80.4' (+5.245), 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+4.094), 'mass > 91.2' (+3.798), 'mass_over_sum_pt > 0.089' (+2.833), 'girth2_top20 > 0.00813' (+2.495) and 'mass_top50 > 98.1' (+2.423) outweigh 'mass > 101' (-7.472) and 'mass_top50 > 82.6' (-3.693). The value, 9.401, adds 1.983 to top and removes 8.226 from W and 8.813 from Z. The formula calls them t and is right for 93.5%.
- **120 GeV gluon/top mixture** — 6.1% of jets, neuron 4.65, formula right for 71%. 41% gluon and 41% top, with 15% quark; mass about 119 GeV, width 0.013, and pT fairly evenly shared (hardest particle 191 GeV). 'mass > 80.4' (+2.264), 'mass > 91.2' (+1.347) and 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+1.322, 74% pass) outweigh 'mass > 101' (-1.974) and 'mass_top50 > 82.6' (-1.431). The value, 4.649 (always on), adds 0.981 to top and removes about 4 from W and Z each. The formula splits them between t (47%) and g (44%) and is right for only 71.3%, so the top push here often misleads.
- **High-pT top/gluon jets** — 4.3% of jets, neuron 4.21, formula right for 82%. 64% top and 30% gluon; mass about 168 GeV, width 0.022, and a total pT of 1154 GeV, above average. The mass tests behave as in the main top group ('mass > 101' -7.134, 'mass > 80.4' +5.062, 'mass > 91.2' +3.648), but 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' passes for only 53% (+0.745 against +4.094 there) because the pT is high, so the value stops at 4.209. It adds 0.888 to top and removes 3.683 from W; the formula calls them t and is right for 82.4%.
- **140 GeV low-pT top jets** — 3.8% of jets, neuron 8.71, formula right for 76%. 76% top, 14% gluon and 10% quark; mass about 141 GeV, width 0.0215, and a low total pT (967 GeV). 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+3.629), 'mass > 80.4' (+3.526), 'mass > 91.2' (+2.385) and 'mass_over_sum_pt > 0.089' (+1.942) outweigh 'mass > 101' (-4.301) and 'mass_top50 > 82.6' (-2.390); 'sum_pt < 1.01e+03' adds 0.706. The value, 8.707, adds 1.837 to top and removes 7.619 from W and 8.163 from Z. The formula calls them t and is right for 76.1%.
- **Low-pT light gluon jets** — 1.4% of jets, neuron 7.46, formula right for 64%. 44% gluon, 30% quark and 25% top; mass about 66 GeV, width 0.0085, and a very low total pT (756 GeV), with a hardest particle of only 160 GeV. The low-pT tests decide it: 'sum_pt < 1.01e+03' (+3.804), 'sum_pt_top40 < 967 and log_sum_pt < 6.82' (+3.440), 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+2.399, 62% pass) and 'sum_pt_top40 < 995' (+1.152), against 'sum_pt < 1.01e+03 and n_real_top50 < 43.5' (-1.273). The value, 7.456, adds 1.573 to top and removes about 6.5-7 from W and Z. The formula calls them g, but it is right for only 63.6%: a hard, low-pT mixture.
- **Low-pT wide top jets** — 1.0% of jets, neuron 6.13, formula right for 77%. 77% top and 19% gluon; mass about 156 GeV, very wide (width 0.0337), low total pT (853 GeV) and a hardest particle of only 110 GeV. 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+9.670), 'mass > 80.4' (+4.369), 'girth2_top40 > 0.00516 and sum_pt_top30 < 912' (+3.626) and 'mass_over_sum_pt > 0.089' (+3.173) outweigh 'mass > 101' (-5.858) and 'mass_top50 > 82.6' (-2.919). The value, 6.131, adds 1.293 to top; the formula calls them t and is right for 76.6%.
- **Very heavy gluon jets** — 0.9% of jets, neuron 4.35, formula right for 75%. 66% gluon and 22% top; mass about 225 GeV, width 0.0335, total pT 1265 GeV, and almost all pT far from the axis. 'mass > 101' (-13.249), 'mass_top50 > 82.6' (-5.860) and 'mass > 140' (-3.130) are more than offset by 'mass > 80.4' (+8.377), 'mass > 91.2' (+6.374), 'mass_top50 > 98.1' (+4.167) and 'mass_over_sum_pt > 0.089' (+3.130). The value, 4.351, adds 0.918 to top, which is the wrong way here. The formula still calls them g and is right for 74.6%.
- **Very low-pT gluon/quark jets** — 0.2% of jets, neuron 8.04, formula right for 60%. 55% gluon, 35% quark and 11% top; mass about 52 GeV, width 0.0117, and the lowest total pT (532 GeV), with a hardest particle of 107 GeV. 'sum_pt_top40 < 967 and log_sum_pt < 6.82' grows as the pT falls and adds 17.643, with 'sum_pt < 1.01e+03' (+7.176) and 'mass_over_sum_pt > 0.0754 and sum_pt < 1.11e+03' (+6.465), against 'sum_pt < 1.01e+03 and n_real_top50 < 43.5' (-3.292). The value, 8.044, adds 1.697 to top and removes about 7 from W and Z. The formula calls them g but is right for only 59.6%, the worst group of this neuron.

### neuron 10: Spread-out hard particles, heavy mass (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (large girth of the 5 hardest, little pT within ΔR < 0.05, large LHA and e2) and for mass above 121 GeV. Top jets sit highest (AUC 0.85), W, Z and gluon jets in the middle, quark jets lowest.
- *computed — its value:* largest for t (2.30), then Z (1.10), then W (1.02), then g (1.02), then q (0.66); it separates t jets from the rest best (AUC 0.85: large for t)
- **How the class scores use it:** Only the t score uses it, raising it (+27%): three spread-out prongs in a heavy jet are the main positive top signature; freezing it costs 1.736 points.
- *computed — used by:* raises the score of t (+27%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** High (2.64) when less than 0.782 of the pT sits within ΔR < 0.05, e2 > 0.0456 and mass ≤ 179 GeV (15% of jets, 81% top); 1.69 for spread jets with e2 ≤ 0.0456 and max_dr > 0.336 (26%, mixed) and 1.22 with max_dr ≤ 0.336 (20%, W and Z); low (0.169) when more than 0.833 of the pT is in the core and the 10 hardest carry above 810 GeV (23%, 49% q). It feeds almost only the top score: +2.60, +1.67 and +1.20 in those three spread regimes (g and q get at most -0.041). regime_r2 0.643.

Regimes (a small tree on its quantities; R² 0.643):

- `` — 14.8% of jets, value 2.64 (1.62…3.62), formula right 89%
- `` — 25.7% of jets, value 1.69 (0.75…2.56), formula right 76%
- `` — 19.6% of jets, value 1.22 (0.50…1.88), formula right 90%
- `` — 7.9% of jets, value 0.99 (0.00…1.88), formula right 73%
- `` — 2.0% of jets, value 0.62 (0.00…1.88), formula right 77%
- `` — 3.4% of jets, value 0.62 (0.00…1.50), formula right 83%
- `` — 3.4% of jets, value 0.26 (0.00…0.81), formula right 73%
- `` — 23.2% of jets, value 0.17 (0.00…0.62), formula right 78%

```
z = 4.78
if girth2_top5 < 0.025: z += -69.30 × (0.025 − girth2_top5)
if mass < 121: z += -0.032 × (121 − mass)
if e2 < 0.066: z += -24.80 × (0.066 − e2)
if mass < 86.40: z += 0.050 × (86.40 − mass)
if z_dr_0_0p05 > 0.767: z += -10.40 × (z_dr_0_0p05 − 0.767)
if mass < 80.40: z += 0.050 × (80.40 − mass)
if z_dr_0p2_0p4 < 0.089: z += 9.17 × (0.089 − z_dr_0p2_0p4)
if mass > 144: z += -0.107 × (mass − 144)
if girth < 0.123: z += -6.56 × (0.123 − girth)
if n_dr_0p2_0p4 < 13.10: z += -0.062 × (13.10 − n_dr_0p2_0p4)
if mass_top50 > 137: z += 0.087 × (mass_top50 − 137)
if girth2_top5 < 0.024 and sum_pt_top3 < 655: z += 0.107 × (0.024 − girth2_top5) × (655 − sum_pt_top3)
if mass < 120 and tau21 < 0.464: z += 0.102 × (120 − mass) × (0.464 − tau21)
if lam1 > 0.0022: z += -51.80 × (lam1 − 0.0022)
if mass_top10 < 81.30: z += -0.0081 × (81.30 − mass_top10)
if girth2_top30 < 0.0053: z += 198 × (0.0053 − girth2_top30)
if e2 < 0.067 and z_dr_0p1_0p2 < 0.218: z += -37.50 × (0.067 − e2) × (0.218 − z_dr_0p1_0p2)
if max_dr < 0.402: z += -3.11 × (0.402 − max_dr)
if mass < 63.00: z += -0.032 × (63.00 − mass)
if n_dr_0p2_0p4 < 5.96: z += -0.084 × (5.96 − n_dr_0p2_0p4)
if sum_pt_top10 > 677: z += -0.001 × (sum_pt_top10 − 677)
if dr_0 < 0.064 and n_dr_0p2_0p4 > 2.93: z += 0.885 × (0.064 − dr_0) × (n_dr_0p2_0p4 − 2.93)
if mass > 162: z += -0.059 × (mass − 162)
if sum_pt_top50 < 959: z += -0.0086 × (959 − sum_pt_top50)
if m012 < 47.40: z += -0.0024 × (47.40 − m012)
if mass < 114 and D2 < 2.16: z += -0.0061 × (114 − mass) × (2.16 − D2)
if z_dr_0p2_0p4 < 0.085 and max_dr > 0.265: z += 11.60 × (0.085 − z_dr_0p2_0p4) × (max_dr − 0.265)
if max_dr < 0.240: z += 8.14 × (0.240 − max_dr)
if mass_top50 > 162 and D2 > -0.418: z += 0.012 × (mass_top50 − 162) × (D2 − -0.418)
if sum_pt_top10 < 671: z += 0.00093 × (671 − sum_pt_top10)
if mass_top50 > 172 and sum_pt < 1240: z += -0.00032 × (mass_top50 − 172) × (1240 − sum_pt)
if lam1 > 0.0052 and pt_dispersion > 0.276: z += -194 × (lam1 − 0.0052) × (pt_dispersion − 0.276)
if mass_top10 > 89.10: z += 0.0085 × (mass_top10 − 89.10)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W/Z two-prong jets** — 31.7% of jets, neuron 1.41, formula right for 90%. 45% W and 43% Z; mass about 85 GeV, width 0.0068, and a two-prong profile with half the pT at 0.05-0.1 and only 3% within 0.025. Negatives such as 'girth2_top5 < 0.0253' (-1.382), 'mass < 121' (-1.144), 'e2 < 0.0663' (-0.753) and 'n_dr_0p2_0p4 < 13.1' (-0.632) are offset by 'mass < 120 and tau21 < 0.464' (+0.767), 'z_dr_0p2_0p4 < 0.089' (+0.758) and 'girth2_top5 < 0.0236 and sum_pt_top3 < 655' (+0.394). The value, 1.406 (on for 98.1%), adds 1.384 to the top score, which the other neurons must overcome. The formula splits them between W (48%) and Z (44%) and is right for 90.2%.
- **Boson-mass jets with hard core** — 14.9% of jets, neuron 0.35, formula right for 76%. A mixture: 32% Z, 27% W, 25% gluon and 11% quark; mass about 85 GeV, width 0.0066, a harder leading particle (278 GeV) and 37% of the pT within 0.025. Compared with the W/Z group, 'z_dr_0_0p05 > 0.767' now passes for 74% and subtracts 0.707, and 'mass < 120 and tau21 < 0.464' passes for only 57% (+0.311), on top of 'girth2_top5 < 0.0253' (-1.659), 'mass < 121' (-1.142) and 'e2 < 0.0663' (-1.071). The value falls to 0.355 (on for 46.1%), adding only 0.35 to top. The formula spreads them over W (33%), Z (31%) and g (28%) and is right for only 76%.
- **120 GeV top/gluon jets** — 12.0% of jets, neuron 2.15, formula right for 71%. 50% top and 33% gluon; mass about 123 GeV, width 0.0155, pT fairly evenly shared (hardest particle 176 GeV) and spread wide. 'mass < 121' passes for only 46% (-0.202), so fewer negatives build up: 'girth2_top5 < 0.0253' (-1.130), 'e2 < 0.0663' (-0.643) and 'lam1 > 0.00223' (-0.564) against 'girth2_top5 < 0.0236 and sum_pt_top3 < 655' (+0.448). The value, 2.145 (on for 97.8%), adds 2.111 to top. The formula calls them t but is right for only 71.2%, since the gluons here get the same push.
- **Very light quark jets** — 10.1% of jets, neuron 0.14, formula right for 80%. 77% quark; very light (about 27 GeV), extremely narrow (width 0.0008), a hard leading particle (353 GeV) and 92% of the pT within 0.025. 'mass < 121' (-2.998), 'z_dr_0_0p05 > 0.767' (-2.097), 'girth2_top5 < 0.0253' (-1.748), 'e2 < 0.0663' (-1.457) and 'mass < 63' (-1.150) outweigh 'mass < 86.4' (+2.937) and 'mass < 80.4' (+2.676). The value is 0.142 (on for 32.5%), a negligible top push. The formula calls them q and is right for 80.3%.
- **Light quark/gluon jets** — 9.8% of jets, neuron 0.61, formula right for 73%. 46% quark and 42% gluon; light (about 46 GeV), narrow (width 0.0019), and 80% of the pT within 0.025. 'mass < 121' (-2.412), 'girth2_top5 < 0.0253' (-1.738), 'z_dr_0_0p05 > 0.767' (-1.630) and 'e2 < 0.0663' (-1.366) against 'mass < 86.4' (+2.026) and 'mass < 80.4' (+1.752). The value, 0.615 (on for 69.9%), adds 0.605 to top. The formula splits them between q (58%) and g (42%) and is right for 73%.
- **Mid-mass gluon/quark jets** — 8.3% of jets, neuron 1.27, formula right for 68%. 52% gluon and 30% quark; mass about 62 GeV, width 0.0037, and 54% of the pT within 0.025. 'mass < 121' (-1.881), 'girth2_top5 < 0.0253' (-1.693) and 'e2 < 0.0663' (-1.238) against 'mass < 86.4' (+1.202) and 'mass < 80.4' (+0.916); the core test passes for only 70% (-0.711). The value, 1.267 (on for 83.2%), adds 1.247 to the top score for jets that are rarely tops. The formula calls them g and is right for only 68.3%.
- **Top-mass jets** — 6.6% of jets, neuron 2.31, formula right for 93%. 88% top; mass about 174 GeV, width 0.0287, pT shared out and mostly far from the axis. 'mass > 144' (-3.231) is almost repaid by 'mass_top50 > 137' (+2.891), while 'lam1 > 0.00223' (-1.100) and 'mass > 162' (-0.722) subtract; the spread-out hard particles make 'girth2_top5 < 0.0253' pass for only 63% (-0.380). The value, 2.312 (on for 96.3%), adds 2.276 to top. The formula calls them t and is right for 92.8%.
- **160 GeV top jets** — 5.3% of jets, neuron 2.55, formula right for 87%. 83% top and 12% gluon; mass about 161 GeV, width 0.0257, pT shared out and mostly far from the axis. 'mass > 144' (-1.825) against 'mass_top50 > 137' (+1.632), with 'lam1 > 0.00223' (-0.977) and 'girth2_top5 < 0.0253' (-0.616, 81% pass). The value, 2.548 (on for 99.1%), is the neuron's highest and adds 2.508 to top. The formula calls them t and is right for 86.6%.
- **Heavy gluon/top mixture** — 1.2% of jets, neuron 0.18, formula right for 76%. 47% gluon and 42% top; mass about 203 GeV, width 0.0342, total pT 1136 GeV, and almost all pT far from the axis. 'mass > 144' (-6.341), 'mass > 162' (-2.443), 'lam1 > 0.00223' (-1.368) and 'mass_top50 > 172 and sum_pt < 1.24e+03' (-0.966) outweigh 'mass_top50 > 137' (+5.040) and 'mass_top50 > 162 and D2 > -0.418' (+0.989). The value is 0.176 (on for 27.5%). The formula splits them between g (51%) and t (45%) and is right for 75.6%.
- **Very heavy gluon jets** — 0.2% of jets, neuron 0.00, formula right for 77%. 74% gluon and 16% top; very heavy (about 258 GeV), wide (width 0.0358), total pT 1406 GeV. 'mass > 144' (-12.236) and 'mass > 162' (-5.704) outweigh 'mass_top50 > 137' (+9.619) and 'mass_top50 > 162 and D2 > -0.418' (+2.449); the value is 0.0 and the neuron has no effect. The formula calls them g and is right for 77%.

### neuron 0: W-side mass, below the Z (moderate)

- **What it measures:** Pushed up for jets lighter than 80.4 GeV (and m/pT below about 0.077) and cut back sharply for masses between 80.4 and 91.2 GeV, the Z side of the W peak; overall it falls as mass and width grow. W jets sit far highest on it (mean 1.78, AUC 0.96), quark and gluon jets low, Z and top jets lowest.
- *computed — its value:* largest for W (1.78), then q (0.37), then g (0.25), then Z (0.12), then t (0.09); it separates W jets from the rest best (AUC 0.96: large for W)
- **How the class scores use it:** It raises the W score (+7%) and lowers the Z score (-14%): being high on this scale is evidence for a W rather than a Z. The g, q and t scores hardly use it.
- *computed — used by:* raises the score of W (+7%); lowers the score of Z (-14%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** High (mean 2.03) for jets with mass ≤ 85.2 GeV that are not too narrow (girth2_top20 > 0.00326) and have at most 5 particles at 0.2 < ΔR < 0.4 (15% of jets, 87% W); it drops to 1.14 when that outer ring holds more particles (7.4%, 53% W), to 0.79 or 0.25 for narrow light jets (girth2_top20 ≤ 0.00326, split at LHA 0.173), and is essentially off (0.004) above 87.2 GeV with m/pT > 0.0807 (43% of jets). The regime that matters is the first: it adds +1.52 to W, -2.79 to Z and +0.254 to t, so this neuron is the W-versus-Z mass separator below 85.2 GeV. The regimes describe it well (regime_r2 0.835).

Regimes (a small tree on its quantities; R² 0.835):

- `` — 15.2% of jets, value 2.03 (1.38…2.53), formula right 91%
- `` — 7.4% of jets, value 1.14 (0.06…1.91), formula right 68%
- `` — 9.0% of jets, value 0.79 (0.19…1.34), formula right 69%
- `` — 2.2% of jets, value 0.29 (0.00…0.81), formula right 71%
- `` — 20.9% of jets, value 0.25 (0.00…0.59), formula right 77%
- `` — 2.0% of jets, value 0.07 (0.00…0.31), formula right 88%
- `` — 43.2% of jets, value 0.00 (0.00…0.00), formula right 84%

```
z = 0.972
if mass > 80.40: z += -0.154 × (mass − 80.40)
if mass > 91.20: z += 0.156 × (mass − 91.20)
if mass_over_sum_pt_sq < 0.0083: z += 539 × (0.0083 − mass_over_sum_pt_sq)
if mass_over_sum_pt > 0.077: z += -65.80 × (mass_over_sum_pt − 0.077)
if mass_over_sum_pt > 0.084: z += 72.10 × (mass_over_sum_pt − 0.084)
if girth < 0.057: z += -48.80 × (0.057 − girth)
if girth2_top40 < 0.0062: z += -238 × (0.0062 − girth2_top40)
if mass_top50 < 81.90: z += -0.025 × (81.90 − mass_top50)
if girth2_top20 < 0.0063: z += -146 × (0.0063 − girth2_top20)
if lam1 < 0.0059: z += -187 × (0.0059 − lam1)
if log_sum_pt < 7.02: z += 2.53 × (7.02 − log_sum_pt)
if z_top30_slots > 0.918: z += -3.34 × (z_top30_slots − 0.918)
if LHA < 0.255: z += 3.98 × (0.255 − LHA)
if n_dr_0p2_0p4 < 10.20: z += 0.037 × (10.20 − n_dr_0p2_0p4)
if girth2_top40 < 0.006 and girth2_top3 < 0.0029: z += 41900 × (0.006 − girth2_top40) × (0.0029 − girth2_top3)
if sum_pt < 1010 and z_dr_0p2_0p4 < 0.207: z += -0.038 × (1010 − sum_pt) × (0.207 − z_dr_0p2_0p4)
if mass > 80.40 and n_dr_0p2_0p4 < 6.83: z += -0.010 × (mass − 80.40) × (6.83 − n_dr_0p2_0p4)
if n_particles < 47.80: z += 0.0098 × (47.80 − n_particles)
if mass_top50 < 80.00 and z_dr_0p05_0p1 < 0.208: z += 0.039 × (80.00 − mass_top50) × (0.208 − z_dr_0p05_0p1)
if log_sum_pt < 6.99 and sum_pt_top40 > 961: z += 0.045 × (6.99 − log_sum_pt) × (sum_pt_top40 − 961)
if z_dr_0_0p05 > 0.878: z += -3.23 × (z_dr_0_0p05 − 0.878)
if sum_pt < 1020 and z_dr_0p1_0p2 > 0.097: z += -0.013 × (1020 − sum_pt) × (z_dr_0p1_0p2 − 0.097)
if girth2_top40 < 0.0042: z += 51.00 × (0.0042 − girth2_top40)
if z_top30_slots > 0.916 and C2 > 0.059: z += -69.50 × (z_top30_slots − 0.916) × (C2 − 0.059)
if n_dr_0p2_0p4 < 9.29 and z_top50_slots < 0.987: z += -2.01 × (9.29 − n_dr_0p2_0p4) × (0.987 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W-mass jets, W-dominated** — 25.1% of jets, neuron 1.62, formula right for 82%. 71% W, with about 9% each of gluon and Z jets; mass about 80 GeV and width 0.006, narrower than the 0.0093 average, with ordinary pT sharing (hardest particle 244 GeV). The pT sits at 0.025-0.1 from the axis rather than in a single hard core (only 15% within 0.025, against 29% for all jets), as expected for two resolved prongs. 'mass_over_sum_pt_sq < 0.00825' passes for all of them and adds 1.224, helped by 'log_sum_pt < 7.02' (+0.221) and 'n_dr_0p2_0p4 < 10.2' (+0.210); 'mass > 80.4' passes for only 42% and costs 0.166 on average, and several narrowness tests take off about 0.1-0.2 each. The neuron ends at 1.616 and is on for 94.8%, which adds 1.212 to the W score and removes 2.222 from the Z score. The formula calls them W and is right for 82.3%.
- **Z-mass jets, neuron silenced** — 21.3% of jets, neuron 0.01, formula right for 88%. 76% Z, with about 9% each of gluon and top jets; mass about 91 GeV and width 0.008, with ordinary pT sharing and more pT at 0.05-0.1 from the axis (41%) than the 27% seen for all jets. 'mass > 80.4' passes for 99% and takes away 1.673, and 'mass_over_sum_pt > 0.0769' (98%) takes away 0.813 more; 'mass_over_sum_pt > 0.0836' (+0.431), 'mass_over_sum_pt_sq < 0.00825' (+0.282) and 'mass > 91.2' (+0.197, 46% pass) give only part of it back. The value is 0.015 and the neuron is on for only 6.3%, so it barely touches the scores (-0.021 on Z); this is exactly the W/Z separation the neuron exists for. The formula calls them Z and is right for 87.9%.
- **Light narrow quark-like jets** — 17.0% of jets, neuron 0.24, formula right for 78%. 63% quark and 25% gluon; light (about 34 GeV) and very narrow (width 0.0011), with a hard leading particle (331 GeV against 240 for all jets) and 89% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00825' passes for all and adds a large 3.844, but the narrowness tests that also always pass pull it back: 'girth < 0.0567' (-2.053), 'girth2_top40 < 0.00621' (-1.231), 'mass_top50 < 81.9' (-1.190), 'lam1 < 0.00592' (-0.960) and 'girth2_top20 < 0.00631' (-0.835). The net 0.243 (on for 73%) adds only 0.182 to W and removes 0.334 from Z, so this neuron plays little part here. The formula calls them q and is right for 78.5%.
- **Mid-mass gluon/quark mixture** — 11.8% of jets, neuron 0.62, formula right for 70%. 49% gluon and 32% quark, with 10% W; mass about 61 GeV, width 0.0033, pT sharing close to average, and 62% of the pT within 0.025 of the axis (more core-dominated than average). The same tug of war as the light quark group, but weaker: 'mass_over_sum_pt_sq < 0.00825' adds 2.671 while 'girth < 0.0567' (-1.108), 'girth2_top40 < 0.00621' (-0.806), 'girth2_top20 < 0.00631' (-0.673), 'lam1 < 0.00592' (-0.638) and 'mass_top50 < 81.9' (-0.570) all pass and subtract. The result, 0.624 (on for 86.1%), adds 0.468 to W and removes 0.858 from Z, a push these gluon/quark jets do not need. The formula splits them between g (51%) and q (41%) and is right for only 70.4%.
- **Top-mass jets, neuron silenced** — 8.9% of jets, neuron 0.00, formula right for 93%. 89% top; mass about 171 GeV and width 0.0282, three times the average, with pT shared out among many particles (hardest particle 161 GeV) and only 2% of the pT within 0.025 of the axis. 'mass > 80.4' grows with the mass above 80.4 GeV and takes away 13.956, while 'mass > 91.2' gives back 12.453; the pair 'mass_over_sum_pt > 0.0836' (+6.037) and 'mass_over_sum_pt > 0.0769' (-5.950) cancel as well. What is left sits below zero, so the value is 0.003 (on for 1.6%) and the class scores are untouched. The formula calls them t and is right for 92.7%.
- **110 GeV gluon/top mixture** — 5.1% of jets, neuron 0.00, formula right for 70%. 40% gluon, 39% top and 17% quark; mass about 111 GeV, width 0.0119, with pT more evenly shared than average (hardest particle 193 GeV) and spread to 0.05-0.1 and beyond. The mass tests again cancel: 'mass > 80.4' takes away 4.666 and 'mass_over_sum_pt > 0.0769' 2.063, against 'mass > 91.2' (+3.043) and 'mass_over_sum_pt > 0.0836' (+1.796); the value is 0.003 and the neuron is on for only 0.6%. It gives nothing to the scores, so the gluon-versus-top call is left to other neurons: the formula splits them between t (45%) and g (43%) and is right for only 70.2%.
- **150 GeV top-like jets** — 4.8% of jets, neuron 0.00, formula right for 79%. 72% top and 21% gluon; mass about 153 GeV, width 0.023, pT shared out (hardest particle 170 GeV) and mostly beyond 0.05 from the axis, with only 3% within 0.025. 'mass > 80.4' (-11.143) is nearly balanced by 'mass > 91.2' (+9.603), and 'mass_over_sum_pt > 0.0769' (-4.855) by 'mass_over_sum_pt > 0.0836' (+4.837); the rest cannot lift the sum above zero, so the value is 0.002 (on for 1.3%) and the scores are untouched. The formula calls them t and is right for 79.1%.
- **130 GeV top/gluon mixture** — 4.3% of jets, neuron 0.00, formula right for 72%. 52% top and 34% gluon; mass about 131 GeV, width 0.0168, pT fairly evenly shared (hardest particle 179 GeV) and spread wide. 'mass > 80.4' (-7.863) and 'mass_over_sum_pt > 0.0769' (-3.414) outweigh 'mass > 91.2' (+6.281) and 'mass_over_sum_pt > 0.0836' (+3.260), giving 0.002 (on for 0.7%) and no effect on the scores. The formula calls them t, but it is right for only 72%, because of the large gluon share.
- **Heavy wide gluon/top mixture** — 1.5% of jets, neuron 0.02, formula right for 77%. 46% top and 44% gluon; heavy (about 200 GeV) and wide (width 0.0343), with a slightly higher total pT (1110 GeV) and almost all pT far from the axis (1% within 0.025). The largest mass hinges nearly cancel: 'mass > 80.4' takes away 18.362 and 'mass > 91.2' gives back 16.916, with 'mass_over_sum_pt > 0.0836' (+7.197) against 'mass_over_sum_pt > 0.0769' (-7.009). The value is 0.024 (on for 16.7%), too small to matter for the scores. The formula splits them between t (50%) and g (46%) and is right for 76.6%.
- **Very heavy high-pT gluon jets** — 0.3% of jets, neuron 0.04, formula right for 77%. 73% gluon and 17% top; very heavy (about 257 GeV) and wide (width 0.0361), with a high total pT (1395 GeV against 1044 on average) and almost no pT near the axis. 'mass > 80.4' (-27.213) against 'mass > 91.2' (+25.881), and 'mass_over_sum_pt > 0.0836' (+7.513) against 'mass_over_sum_pt > 0.0769' (-7.298), leave 0.035 (on for 18.9%), so the neuron hardly affects the scores. The formula calls them g and is right for 77.4%; this is a very small group (0.27% of jets).

### neuron 3: Sparse jet, empty outer ring (moderate)

- **What it measures:** Rises for jets with few particles, little activity at 0.2 <= ΔR < 0.4, a thin minor axis (small lam2) and low m/pT. W jets and quark jets sit highest, Z jets in the middle, gluon jets low and top jets almost at zero (AUC 0.18 for top: small for t).
- *computed — its value:* largest for W (1.02), then q (0.77), then Z (0.50), then g (0.22), then t (0.03); it separates t jets from the rest best (AUC 0.18: small for t)
- **How the class scores use it:** It raises the W and Z scores (+4% each) and lowers the g score (-9%): a sparse, clean jet looks like a boson or a quark, not a gluon. The q and t scores hardly use it.
- *computed — used by:* raises the score of W (+4%), Z (+4%); lowers the score of g (-9%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** High (2.0) for jets with a thin minor axis (lam2 ≤ 0.00034), an almost empty 0.2 < ΔR < 0.4 ring (pT share ≤ 0.00105) and low (m/pT)² ≤ 0.00721 (6.7% of jets, 66% W); still 1.19-1.27 for thin jets with a slightly fuller ring or heavier (the heavier thin-and-empty regime is 99% Z), and nearly off (0.073) for wide jets (lam2 > 0.00034) with ring share > 0.00165 and (m/pT)² > 0.00354, which is 54% of jets. It subtracts from g (-1.50 at the top, -0.891 in the 15% thin, quark/W-rich regime) and adds the same amount to W and Z (+0.875 at the top). regime_r2 0.752.

Regimes (a small tree on its quantities; R² 0.752):

- `` — 6.7% of jets, value 2.00 (1.28…2.69), formula right 94%
- `` — 2.3% of jets, value 1.27 (0.81…1.69), formula right 99%
- `` — 14.8% of jets, value 1.19 (0.56…1.78), formula right 82%
- `` — 2.8% of jets, value 1.13 (0.41…1.89), formula right 96%
- `` — 2.1% of jets, value 0.60 (0.00…1.38), formula right 92%
- `` — 13.2% of jets, value 0.48 (0.00…0.94), formula right 76%
- `` — 4.5% of jets, value 0.42 (0.00…0.91), formula right 94%
- `` — 53.6% of jets, value 0.07 (0.00…0.25), formula right 78%

```
z = -0.185
if n_dr_0p2_0p4 < 4.94 and z_top50_slots > 0.979: z += 10.60 × (4.94 − n_dr_0p2_0p4) × (z_top50_slots − 0.979)
if lam2 < 0.00062: z += 1350 × (0.00062 − lam2)
if mass_over_sum_pt_sq < 0.0073: z += 94.70 × (0.0073 − mass_over_sum_pt_sq)
if tau21 < 0.424 and lam1 < 0.016: z += -255 × (0.424 − tau21) × (0.016 − lam1)
if n_particles < 46.10 and sum_pt_top40 > 841: z += 0.00013 × (46.10 − n_particles) × (sum_pt_top40 − 841)
if n_dr_0p1_0p2 < 14.40: z += 0.017 × (14.40 − n_dr_0p1_0p2)
if girth2_top5 < 0.00012: z += -7230 × (0.00012 − girth2_top5)
if mass_over_sum_pt_sq < 0.0075 and girth2_top15 > 0.0049: z += 412000 × (0.0075 − mass_over_sum_pt_sq) × (girth2_top15 − 0.0049)
if n_particles < 45.40 and mass_top30 > 73.00: z += -0.0015 × (45.40 − n_particles) × (mass_top30 − 73.00)
if n_dr_0p2_0p4 < 5.19 and dr_0 < 0.042: z += -5.37 × (5.19 − n_dr_0p2_0p4) × (0.042 − dr_0)
if lam2 < 0.00035: z += 1180 × (0.00035 − lam2)
if z_dr_0p2_0p4 < 0.0014: z += 233 × (0.0014 − z_dr_0p2_0p4)
if n_dr_0p2_0p4 < 5.16 and sum_pt_top30 < 991: z += -0.0014 × (5.16 − n_dr_0p2_0p4) × (991 − sum_pt_top30)
if max_dr < 0.293 and tau21 > 0.430: z += -15.50 × (0.293 − max_dr) × (tau21 − 0.430)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Busy heavy jets, neuron idle** — 37.4% of jets, neuron 0.02, formula right for 78%. 46% top and 26% gluon, with about 10% each of W and Z; mass about 127 GeV, width 0.017, pT shared out (hardest particle 179 GeV) and spread far from the axis. None of the if-statements passes for most of these jets: 'n_dr_0p1_0p2 < 14.4' passes for 39% (+0.034) and 'mass_over_sum_pt_sq < 0.00728' for 22% (+0.027), so the value is only 0.022 (on for 12.7%) and the scores barely move. The formula calls them t and is right for 77.6%.
- **Mid-mass gluon/quark jets** — 14.1% of jets, neuron 0.42, formula right for 74%. 53% gluon and 33% quark; mass about 56 GeV, width 0.0028, total pT 1081 GeV, and 68% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00728' passes for all and adds 0.420, with 'n_dr_0p1_0p2 < 14.4' (+0.112) and 'lam2 < 0.000624' (+0.094, 43% pass); 'girth2_top5 < 0.000115' (-0.074) takes a little off. The value, 0.424, removes 0.318 from the gluon score and adds 0.185 each to W and Z, which works against the gluons here. The formula calls them g, but it is right for only 73.7%.
- **W/Z/top jets, two-prong penalty** — 10.2% of jets, neuron 0.10, formula right for 77%. 39% Z, 29% W and 15% top; mass about 87 GeV, width 0.0072, and pT at 0.025-0.1 from the axis. 'tau21 < 0.424 and lam1 < 0.0159' passes for all and subtracts 0.400, which eats the gains from 'lam2 < 0.000624' (+0.171), 'n_particles < 46.1 and sum_pt_top40 > 841' (+0.105) and the sparse-outer-ring test (+0.074, 26% pass). The value is only 0.103 (on for 34.1%), adding 0.045 each to W and Z. The formula splits them between Z (40%) and W (34%) and is right for 77.4%.
- **Z/W jets, hard leading particle** — 6.6% of jets, neuron 0.45, formula right for 90%. 51% Z and 36% W; mass about 90 GeV, width 0.0076, but a very hard leading particle (342 GeV against 240 for all jets), with 43% of the pT at 0.025-0.05 from the axis. 'lam2 < 0.000624' (+0.456) and 'n_particles < 46.1 and sum_pt_top40 > 841' (+0.422) pass for nearly all, against 'tau21 < 0.424 and lam1 < 0.0159' (-0.506) and 'n_particles < 45.4 and mass_top30 > 73' (-0.352). The value, 0.453 (on for 76%), adds 0.198 each to W and Z and removes 0.34 from the gluon score. The formula calls them Z and is right for 89.5%.
- **Light narrow quark jets** — 6.6% of jets, neuron 0.90, formula right for 76%. 64% quark and 21% gluon; light (about 36 GeV), very narrow (width 0.0012), a hard leading particle (369 GeV) and 93% of the pT within 0.025 of the axis. 'mass_over_sum_pt_sq < 0.00728' (+0.573), 'n_particles < 46.1 and sum_pt_top40 > 841' (+0.418) and 'lam2 < 0.000624' (+0.409) add, while 'girth2_top5 < 0.000115' (-0.561, 98% pass) takes much of it back. The value, 0.904, adds 0.395 each to W and Z and removes 0.678 from the gluon score, which helps against gluons but not towards quarks. The formula calls them q and is right for 75.9%.
- **W/Z jets, empty outer ring** — 6.5% of jets, neuron 0.84, formula right for 94%. 50% W and 43% Z; mass about 84 GeV, width 0.0067, pT shared more evenly than average (hardest particle 200 GeV) and 52% of the pT at 0.05-0.1 from the axis. 'n_dr_0p2_0p4 < 4.94 and z_top50_slots > 0.979' passes for all and adds 0.744, the main difference from the other W/Z groups; 'tau21 < 0.424 and lam1 < 0.0159' (-0.270) and the smaller tests roughly balance. The value, 0.836, adds 0.366 each to W and Z and removes 0.627 from the gluon score. The formula splits them between W (52%) and Z (43%) and is right for 94.2%.
- **Clean W jets** — 6.2% of jets, neuron 1.90, formula right for 97%. 88% W; mass about 81 GeV, width 0.0062, and 60% of the pT at 0.05-0.1 from the axis, with almost none within 0.025 (2%). Everything passes: the empty-outer-ring test (+0.854), 'mass_over_sum_pt_sq < 0.00749 and girth2_top15 > 0.00486' (+0.580, passing for all only here) and 'lam2 < 0.000624' (+0.561) outweigh 'tau21 < 0.424 and lam1 < 0.0159' (-0.639). The neuron's top value, 1.901, adds 0.832 each to W and Z and removes 1.425 from the gluon score. The formula calls them W and is right for 96.6%.
- **Clean Z jets** — 5.1% of jets, neuron 1.27, formula right for 97%. 74% Z and 23% W; mass about 88 GeV, width 0.007, and 53% of the pT at 0.05-0.1 from the axis, very little within 0.025. The empty-outer-ring test (+0.886), 'lam2 < 0.000624' (+0.572) and 'n_particles < 46.1 and sum_pt_top40 > 841' (+0.429) add, against 'tau21 < 0.424 and lam1 < 0.0159' (-0.610) and 'n_particles < 45.4 and mass_top30 > 73' (-0.327); unlike the clean W group, the girth2_top15 test passes for only 22%. The value, 1.273, adds 0.557 each to W and Z; the formula calls them Z and is right for 97.2%.
- **Very light pencil-like quark jets** — 3.7% of jets, neuron 1.53, formula right for 83%. 82% quark; very light (about 23 GeV) and extremely narrow (width 0.0005), with a very hard leading particle (396 GeV) and 95% of the pT within 0.025. All the sparse and thin tests pass: the empty-outer-ring test (+0.694), 'lam2 < 0.000624' (+0.658), 'n_particles < 46.1 and sum_pt_top40 > 841' (+0.644) and 'mass_over_sum_pt_sq < 0.00728' (+0.640), while 'n_dr_0p2_0p4 < 5.19 and dr_0 < 0.042' (-0.688) and 'girth2_top5 < 0.000115' (-0.600) take some back. The value, 1.533, adds 0.67 each to W and Z and removes 1.149 from the gluon score, a push towards the bosons that other neurons must undo. The formula calls them q and is right for 82.8%.
- **Light sparse quark jets** — 3.5% of jets, neuron 1.21, formula right for 75%. 65% quark and 19% gluon; light (about 38 GeV), narrow (width 0.0014), a hard leading particle (278 GeV) and 76% of the pT within 0.025. 'mass_over_sum_pt_sq < 0.00728' (+0.555), the empty-outer-ring test (+0.552, 99% pass) and 'lam2 < 0.000624' (+0.426) add, against 'n_dr_0p2_0p4 < 5.19 and dr_0 < 0.042' (-0.436). The value, 1.211, adds 0.53 each to W and Z and removes 0.908 from the gluon score. The formula calls them q and is right for 75.4%.

### neuron 6: One-prong, outside the Z mass (moderate)

- **What it measures:** Follows one-prong-ness (large D2 and τ21, a round rather than elongated pT pattern) and is pushed down for masses between about 91 and 120 GeV and for m/pT above 0.0513. Quark jets sit highest, gluon jets next, top jets in between, Z and W jets near zero.
- *computed — its value:* largest for q (1.56), then g (1.26), then t (0.89), then Z (0.21), then W (0.16); it separates q jets from the rest best (AUC 0.81: large for q)
- **How the class scores use it:** It raises the q (+7%) and g (+3%) scores and lowers the Z score (-15%): a one-prong jet away from the Z mass is not a Z. The W and t scores hardly use it.
- *computed — used by:* raises the score of g (+3%), q (+7%); lowers the score of Z (-15%); does not (or hardly) enter the score of W, t (share of each class score’s average input)
- **Boundaries:** Highest for very light hard cores: 2.42 for mass_top50 ≤ 25.7 GeV (3.8% of jets, 85% q) and 2.05 for 25.7-36.3 GeV (6.4%, 71% q); a second high island is heavy, thin jets (mass > 98.1 GeV, lam2 ≤ 0.00235, 1.63, 12%, 48% top and 35% gluon); it is low (0.131) for mass_top50 > 68.7 GeV with mass ≤ 98.1 GeV, the W/Z window (46% of jets). Its main job is to lower Z (-2.34, -1.98 and -1.58 in those regimes) while mildly adding to q (+0.529 at most) and g (+0.378). regime_r2 0.657.

Regimes (a small tree on its quantities; R² 0.657):

- `` — 3.8% of jets, value 2.42 (2.12…2.62), formula right 85%
- `` — 6.4% of jets, value 2.05 (1.75…2.25), formula right 78%
- `` — 3.7% of jets, value 1.77 (1.38…2.06), formula right 74%
- `` — 11.9% of jets, value 1.63 (0.62…2.75), formula right 75%
- `` — 6.6% of jets, value 1.43 (1.00…1.81), formula right 74%
- `` — 7.5% of jets, value 0.88 (0.38…1.31), formula right 68%
- `` — 13.9% of jets, value 0.77 (0.00…1.88), formula right 85%
- `` — 46.2% of jets, value 0.13 (0.00…0.44), formula right 85%

```
z = 0.607
if mass < 120: z += -0.038 × (120 − mass)
if mass < 101: z += -0.055 × (101 − mass)
if mass < 173: z += 0.015 × (173 − mass)
if mass_over_sum_pt > 0.051: z += -30.50 × (mass_over_sum_pt − 0.051)
if mass < 91.20: z += 0.054 × (91.20 − mass)
if width < 0.0096: z += -177 × (0.0096 − width)
if mass < 86.10: z += 0.030 × (86.10 − mass)
if n_dr_0p2_0p4 < 16.10: z += -0.046 × (16.10 − n_dr_0p2_0p4)
if lam1 < 0.0074: z += 137 × (0.0074 − lam1)
if lam1 > 0.0081: z += 116 × (lam1 − 0.0081)
if e2 < 0.048: z += 15.70 × (0.048 − e2)
if mass_top10 < 85.30: z += 0.0075 × (85.30 − mass_top10)
if mass_top50 < 71.70: z += 0.035 × (71.70 − mass_top50)
if mass_over_sum_pt > 0.040 and z_dr_0_0p05 < 0.945: z += 8.95 × (mass_over_sum_pt − 0.040) × (0.945 − z_dr_0_0p05)
if mass_over_sum_pt > 0.048 and lam2 < 0.0025: z += 5870 × (mass_over_sum_pt − 0.048) × (0.0025 − lam2)
if mass_over_sum_pt > 0.053 and n_dr_0p2_0p4 < 19.00: z += 0.954 × (mass_over_sum_pt − 0.053) × (19.00 − n_dr_0p2_0p4)
if mass_over_sum_pt > 0.050 and n_dr_0p1_0p2 < 17.30: z += -1.11 × (mass_over_sum_pt − 0.050) × (17.30 − n_dr_0p1_0p2)
if mass_over_sum_pt > 0.052 and z_dr_0p1_0p2 < 0.282: z += 44.80 × (mass_over_sum_pt − 0.052) × (0.282 − z_dr_0p1_0p2)
if girth2_top20 > 0.0024 and max_dr < 0.425: z += 357 × (girth2_top20 − 0.0024) × (0.425 − max_dr)
if girth2_top3 > 0.0096 and D2 < 5.09: z += 20.30 × (girth2_top3 − 0.0096) × (5.09 − D2)
if sum_pt_top10 > 625: z += 0.00049 × (sum_pt_top10 − 625)
if e2 > 0.056: z += -67.50 × (e2 − 0.056)
if m012 > 20.40: z += -0.011 × (m012 − 20.40)
if log_sum_pt < 6.83: z += 6.54 × (6.83 − log_sum_pt)
if lam1 > 0.024: z += 94.10 × (lam1 − 0.024)
if e2 > 0.055 and max_dr < 0.366: z += -1540 × (e2 − 0.055) × (0.366 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W-mass jets** — 24.6% of jets, neuron 0.12, formula right for 83%. 73% W; mass about 80 GeV, width 0.0059, ordinary pT sharing and pT spread to 0.025-0.1 from the axis. The mass window tests all pass and nearly cancel: 'mass < 120' (-1.527) and 'mass < 101' (-1.178) against 'mass < 173' (+1.353) and 'mass < 91.2' (+0.634), with 'mass_over_sum_pt > 0.0513' (-0.765), 'width < 0.00959' (-0.660) and 'n_dr_0p2_0p4 < 16.1' (-0.507) keeping it low. The value, 0.117 (on for 35%), removes only 0.113 from the Z score. The formula calls them W and is right for 83.4%.
- **Z-mass jets** — 15.0% of jets, neuron 0.13, formula right for 92%. 85% Z; mass about 91 GeV, width 0.0079, and 54% of the pT at 0.05-0.1 from the axis. 'mass < 173' (+1.186) is cancelled by 'mass_over_sum_pt > 0.0513' (-1.135), 'mass < 120' (-1.092), 'n_dr_0p2_0p4 < 16.1' (-0.599) and 'mass < 101' (-0.549); 'mass < 91.2' passes for 58% and adds only 0.066. The value stays at 0.127 (on for 39.1%), so the neuron's strong anti-Z weight hardly touches these Z jets (-0.123). The formula calls them Z and is right for 91.5%.
- **Light gluon/quark jets** — 9.9% of jets, neuron 1.59, formula right for 73%. 48% quark and 40% gluon; mass about 44 GeV, width 0.0019, and 78% of the pT within 0.025 of the axis. The light-mass tests 'mass < 91.2' (+2.534), 'mass < 173' (+1.861), 'mass < 86.1' (+1.233) and 'mass_top50 < 71.7' (+0.994) beat 'mass < 101' (-3.099), 'mass < 120' (-2.848) and 'width < 0.00959' (-1.364). The value, 1.587 (always on), removes 1.537 from the Z score and adds 0.347 to quark and 0.248 to gluon. The formula splits them between q (60%) and g (40%) and is right for 73.1%.
- **Very light quark jets** — 9.6% of jets, neuron 2.20, formula right for 80%. 78% quark; very light (about 27 GeV), extremely narrow (width 0.0007), a hard leading particle (351 GeV) and 92% of the pT within 0.025. The same light-mass tests, larger: 'mass < 91.2' (+3.489), 'mass < 173' (+2.117), 'mass < 86.1' (+1.754) and 'mass_top50 < 71.7' (+1.589) against 'mass < 101' (-4.064), 'mass < 120' (-3.512) and 'width < 0.00959' (-1.565). The value, 2.199, is the highest of this neuron; it removes 2.13 from the Z score and adds 0.481 to the quark score. The formula calls them q and is right for 80.5%.
- **Z-mass jets with narrow core** — 9.0% of jets, neuron 0.41, formula right for 78%. 47% Z and 29% gluon, with 11% top and 10% quark; mass about 94 GeV, width 0.0078, harder leading particle (257 GeV) and 48% of the pT at 0.025-0.05. 'mass < 173' (+1.147) against 'mass_over_sum_pt > 0.0513' (-1.118) and 'mass < 120' (-0.997), with several small positive tests ('e2 < 0.0481', 'mass_top10 < 85.3', 'mass_over_sum_pt > 0.0519 and z_dr_0p1_0p2 < 0.282', about +0.33 each). The value, 0.415 (on for 73.3%), removes 0.402 from the Z score, working against the Z jets here. The formula calls them Z, but is right for only 78.5%, with gluons the main confusion.
- **Mid-mass gluon/quark jets** — 8.2% of jets, neuron 0.93, formula right for 68%. 51% gluon and 31% quark; mass about 62 GeV, width 0.0035, and 59% of the pT within 0.025. 'mass < 120' (-2.184) and 'mass < 101' (-2.133) against 'mass < 173' (+1.605) and 'mass < 91.2' (+1.578), with 'width < 0.00959' (-1.074) and smaller positive shape tests. The value, 0.926, removes 0.897 from the Z score and adds 0.203 to quark and 0.145 to gluon. The formula calls them g but is right for only 68.5%.
- **120-130 GeV top/gluon jets** — 8.2% of jets, neuron 1.27, formula right for 71%. 46% top and 39% gluon; mass about 126 GeV, width 0.0143, pT fairly evenly shared (hardest particle 186 GeV) and spread beyond 0.05. 'mass_over_sum_pt > 0.0513' subtracts 2.066, but 'mass < 173' (+0.683), 'mass_over_sum_pt > 0.0403 and z_dr_0_0p05 < 0.945' (+0.515) and 'lam1 > 0.00815' (+0.464) and the rest keep it positive. The value, 1.272 (on for 96.4%), removes 1.232 from the Z score. The formula splits them between t (52%) and g (41%) and is right for only 70.9%.
- **Top jets** — 8.1% of jets, neuron 1.02, formula right for 87%. 83% top; mass about 163 GeV, width 0.026, pT shared out (hardest particle 149 GeV) and almost nothing within 0.025. 'mass_over_sum_pt > 0.0513' (-3.332) is offset by 'lam1 > 0.00815' (+1.476), 'mass_over_sum_pt > 0.0403 and z_dr_0_0p05 < 0.945' (+0.964), 'girth2_top3 > 0.00959 and D2 < 5.09' (+0.449) and 'girth2_top20 > 0.00238 and max_dr < 0.425' (+0.435). The value, 1.017 (on for 77.1%), removes 0.985 from the Z score. The formula calls them t and is right for 87%.
- **Heavy wide top jets** — 4.2% of jets, neuron 1.59, formula right for 87%. 80% top; mass about 181 GeV, width 0.0328, and almost all pT far from the axis. 'mass_over_sum_pt > 0.0513' (-3.933) and 'e2 > 0.0555' (-1.007) are outweighed by 'lam1 > 0.00815' (+2.252), 'girth2_top3 > 0.00959 and D2 < 5.09' (+1.385) and 'mass_over_sum_pt > 0.0403 and z_dr_0_0p05 < 0.945' (+1.137). The value, 1.595, removes 1.545 from the Z score. The formula calls them t and is right for 86.6%.
- **Top jets, moderate spread** — 3.1% of jets, neuron 0.84, formula right for 84%. 78% top; mass about 162 GeV, width 0.0256, a harder leading particle (198 GeV) than other top groups, and 45% of the pT at 0.05-0.1. 'mass_over_sum_pt > 0.0513' (-3.288) is offset by 'lam1 > 0.00815' (+1.601) and 'mass_over_sum_pt > 0.0519 and z_dr_0p1_0p2 < 0.282' (+1.060), which here nearly cancels with 'mass_over_sum_pt > 0.0497 and n_dr_0p1_0p2 < 17.3' (-1.030). The value, 0.845 (on for 72.3%), removes 0.818 from the Z score. The formula calls them t and is right for 84.1%.

### neuron 7: Mass in the 91-101 GeV window (moderate)

- **What it measures:** Switches on mostly for mass between 91.2 and 101 GeV, helped by an elongated two-prong pattern (large eccentricity, small τ21 and D2) and a compact jet. Almost only Z jets sit high on it (mean 2.18 for Z, AUC 0.91); all other types stay near zero.
- *computed — its value:* largest for Z (2.18), then W (0.11), then g (0.07), then t (0.06), then q (0.04); it separates Z jets from the rest best (AUC 0.91: large for Z)
- **How the class scores use it:** It raises the Z score (+9%) and lowers the W (-6%) and t (-3%) scores: a mass just above the Z peak points to a Z. The g and q scores hardly use it.
- *computed — used by:* raises the score of Z (+9%); lowers the score of W (-6%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** High (3.54) for two-prong jets (D2 ≤ 1.24) with mass_top50 > 86.5 GeV and at most 4.5 particles at 0.2 < ΔR < 0.4 (8.2% of jets, 95% Z), medium (1.23) for less two-prong jets (D2 > 1.24) with 84.1 < mass ≤ 95.8 GeV (11%, 67% Z), and off for D2 > 1.24 with mass above 95.8 GeV (0.022, 22%) or with mass ≤ 84.1 GeV and mass_top50 ≤ 78.2 GeV (0.002, 34%). The key regime adds +3.20 to Z and takes -2.21 from W and -0.994 from t: it is the Z-mass switch. regime_r2 0.775.

Regimes (a small tree on its quantities; R² 0.775):

- `` — 8.2% of jets, value 3.54 (2.06…5.06), formula right 97%
- `` — 11.5% of jets, value 1.23 (0.06…2.56), formula right 82%
- `` — 6.1% of jets, value 0.51 (0.00…2.00), formula right 78%
- `` — 2.0% of jets, value 0.49 (0.00…1.44), formula right 83%
- `` — 7.0% of jets, value 0.13 (0.00…0.50), formula right 83%
- `` — 9.2% of jets, value 0.06 (0.00…0.25), formula right 92%
- `` — 21.9% of jets, value 0.02 (0.00…0.00), formula right 81%
- `` — 34.1% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = -0.238
if mass < 91.20: z += -0.242 × (91.20 − mass)
if mass < 101: z += 0.087 × (101 − mass)
if girth2_top40 < 0.013: z += 240 × (0.013 − girth2_top40)
if girth2_top20 < 0.008: z += -461 × (0.008 − girth2_top20)
if width < 0.0094: z += 344 × (0.0094 − width)
if mass_top50 < 98.10 and D2 < 1.61: z += -0.467 × (98.10 − mass_top50) × (1.61 − D2)
if mass < 101 and max_dr < 0.391: z += 0.905 × (101 − mass) × (0.391 − max_dr)
if mass < 91.20 and max_dr < 0.392: z += -1.32 × (91.20 − mass) × (0.392 − max_dr)
if mass < 101 and D2 < 1.61: z += 0.362 × (101 − mass) × (1.61 − D2)
if girth2_top40 < 0.008: z += -318 × (0.008 − girth2_top40)
if girth2_top20 < 0.0064: z += 273 × (0.0064 − girth2_top20)
if z_dr_0p2_0p4 < 0.093: z += -8.94 × (0.093 − z_dr_0p2_0p4)
if girth < 0.086: z += -16.30 × (0.086 − girth)
if mass < 80.40: z += 0.037 × (80.40 − mass)
if mass_top20 < 66.00: z += 0.026 × (66.00 − mass_top20)
if e2 < 0.028: z += 47.00 × (0.028 − e2)
if D2 < 1.82 and n_dr_0p2_0p4 < 9.16: z += 0.136 × (1.82 − D2) × (9.16 − n_dr_0p2_0p4)
if n_dr_0p2_0p4 < 13.00 and z_1st < 0.504: z += 0.140 × (13.00 − n_dr_0p2_0p4) × (0.504 − z_1st)
if z_top50_slots < 0.991: z += -36.20 × (0.991 − z_top50_slots)
if z_dr_0p2_0p4 < 0.0047: z += 173 × (0.0047 − z_dr_0p2_0p4)
if D2 < 1.79 and girth2_top50 < 0.0079: z += -504 × (1.79 − D2) × (0.0079 − girth2_top50)
if C2 < 0.056: z += -10.90 × (0.056 − C2)
if n_dr_0p2_0p4 < 12.90 and planar_flow < 0.549: z += 0.060 × (12.90 − n_dr_0p2_0p4) × (0.549 − planar_flow)
if mass < 91.20 and pt_dispersion < 0.340: z += -0.120 × (91.20 − mass) × (0.340 − pt_dispersion)
if n_dr_0p2_0p4 < 1.99: z += -0.288 × (1.99 − n_dr_0p2_0p4)
if mass < 91.20 and z_dr_0p05_0p1 > 0.393: z += -0.127 × (91.20 − mass) × (z_dr_0p05_0p1 − 0.393)
if D2 < 1.77 and girth2_top50 < 0.0061: z += 659 × (1.77 − D2) × (0.0061 − girth2_top50)
if mass_top30 < 76.10 and D2 < 1.61: z += 0.210 × (76.10 − mass_top30) × (1.61 − D2)
if max_dr < 0.196: z += -8.83 × (0.196 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Heavy jets outside the window** — 36.3% of jets, neuron 0.39, formula right for 81%. 48% top, 21% gluon and 21% Z; mass about 132 GeV, width 0.0178, pT shared out and spread far from the axis; 36.3% of jets. No test is decisive: 'girth2_top40 < 0.0127' passes for 45% (+0.485), 'z_top50_slots < 0.991' for 51% (-0.398), 'girth2_top20 < 0.00798' for 35% (-0.322) and 'mass < 101' for 32% (+0.255). The value, 0.391 (on for 30.6%), adds 0.354 to Z and removes 0.244 from W. The formula calls them t (52%), with Z at 23% and gluon at 22%, and is right for 81.1%.
- **W-mass jets, window shut** — 16.0% of jets, neuron 0.12, formula right for 75%. 51% W and 24% gluon; mass about 79 GeV, width 0.0057, and a harder core than true two-prong jets (34% of the pT within 0.025). 'mass < 91.2' passes for 99% and subtracts 3.038, more than 'mass < 101' (+1.946), 'girth2_top40 < 0.0127' (+1.792) and 'width < 0.00943' (+1.300) return, with 'girth2_top20 < 0.00798' (-1.817) also pulling down. The value is 0.125 (on for 22.7%) and the scores barely change. The formula calls them W but is right for only 75.3%, with gluons the main confusion.
- **Z jets in the window** — 10.4% of jets, neuron 2.90, formula right for 94%. 83% Z; mass about 90 GeV, width 0.0076, and 55% of the pT at 0.05-0.1 from the axis, very little in a hard core. 'mass < 101 and D2 < 1.61' (+2.754) and 'mass_top50 < 98.1 and D2 < 1.61' (-2.578) nearly cancel; 'girth2_top40 < 0.0127' (+1.238), 'mass < 101' (+0.964), 'D2 < 1.82 and n_dr_0p2_0p4 < 9.16' (+0.847) and 'mass < 101 and max_dr < 0.391' (+0.818) add, and 'mass < 91.2' passes for only 62%, costing just 0.433. The value, 2.903 (on for 90.9%), adds 2.631 to Z and removes 1.815 from W and 0.817 from top. The formula calls them Z and is right for 93.9%.
- **Sub-W gluon/quark jets** — 10.4% of jets, neuron 0.00, formula right for 72%. 51% gluon and 35% quark; mass about 56 GeV, width 0.0029, and 69% of the pT within 0.025. 'mass < 91.2' grows with the distance below 91.2 GeV and subtracts 8.529, swamping 'mass < 101' (+3.928), 'girth2_top40 < 0.0127' (+2.453) and 'width < 0.00943' (+2.244). The neuron is never on (value 0.0) and has no effect. The formula calls them g, but is right for only 71.5%.
- **Light quark jets** — 8.7% of jets, neuron 0.00, formula right for 76%. 66% quark and 22% gluon; light (about 34 GeV), narrow (width 0.0012), and 89% of the pT within 0.025. 'mass < 91.2' (-13.915) far outweighs 'mass < 101' (+5.869) and the other positive tests, so the value is 0.0. The formula calls them q and is right for 76.1%.
- **W jets with low D2** — 6.7% of jets, neuron 0.37, formula right for 90%. 79% W; mass about 81 GeV, width 0.0062, and 49% of the pT at 0.05-0.1 from the axis. 'mass_top50 < 98.1 and D2 < 1.61' (-5.173) outweighs 'mass < 101 and D2 < 1.61' (+4.701), and 'mass < 91.2' (-2.524) outweighs 'mass < 101' (+1.764); 'girth2_top40 < 0.0127' (+1.582) and 'mass < 101 and max_dr < 0.391' (+1.581) keep it slightly positive. The value, 0.374 (on for 34.2%), adds 0.339 to Z and removes 0.234 from W, a small push the wrong way. The formula calls them W and is right for 89.8%.
- **Clean W jets** — 4.8% of jets, neuron 0.05, formula right for 92%. 85% W; mass about 77 GeV, width 0.0056, and 66% of the pT at 0.05-0.1 from the axis. 'mass_top50 < 98.1 and D2 < 1.61' (-8.543) and 'mass < 91.2' (-3.473) and 'mass < 91.2 and max_dr < 0.392' (-2.153) outweigh 'mass < 101 and D2 < 1.61' (+7.581) and 'mass < 101 and max_dr < 0.391' (+2.598). The value is 0.053 (on for 13.2%), so the neuron stays out of it. The formula calls them W and is right for 91.5%.
- **Light compact quark jets** — 4.3% of jets, neuron 0.00, formula right for 78%. 68% quark and 21% gluon; light (about 33 GeV), narrow (width 0.0011), and 85% of the pT within 0.025. 'mass < 91.2' (-14.043) and 'mass < 91.2 and max_dr < 0.392' (-5.168) outweigh 'mass < 101' (+5.915) and 'mass < 101 and max_dr < 0.391' (+4.103); the value is 0.0. The formula calls them q and is right for 77.6%.
- **Very light quark jets** — 1.8% of jets, neuron 0.00, formula right for 82%. 78% quark; very light (about 27 GeV), extremely narrow (width 0.0008), and 86% of the pT within 0.025. 'mass < 91.2' (-15.446) and 'mass < 91.2 and max_dr < 0.392' (-11.259) outweigh 'mass < 101 and max_dr < 0.391' (+8.867) and 'mass < 101' (+6.42); the value is 0.0. The formula calls them q and is right for 82.5%.
- **Lightest quark jets** — 0.6% of jets, neuron 0.00, formula right for 87%. 87% quark; the lightest group (about 19 GeV), width 0.0004, a hardest particle of 379 GeV and 92% of the pT within 0.025. 'mass < 91.2 and max_dr < 0.392' (-21.858) and 'mass < 91.2' (-17.540) outweigh 'mass < 101 and max_dr < 0.391' (+16.937) and 'mass < 101' (+7.175); the value is 0.0. The formula calls them q and is right for 86.9%.

### neuron 9: Light one-prong jet, off the W mass (moderate)

- **What it measures:** Large when the hardest 50 particles have mass below 135 GeV, and it follows τ21 (one-prong-ness); it is pushed down for mass between 63 and 80.4 GeV, the W side. Quark jets sit highest, gluon jets next, top, W and Z jets low.
- *computed — its value:* largest for q (2.03), then g (1.26), then t (0.34), then W (0.19), then Z (0.16); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** It raises the q (+17%) and g (+10%) scores and lowers the W score slightly (-3%): a light, one-prong jet off the W mass is a light-QCD jet. The Z and t scores hardly use it.
- *computed — used by:* raises the score of g (+10%), q (+17%); lowers the score of W (-3%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** High for light jets: 4.06 for mass_top50 ≤ 65.3 GeV and total pT ≤ 937 GeV (2.5%), 2.69 for mass_top50 ≤ 65.3 GeV, pT > 937 GeV and mass_top40 ≤ 44.6 GeV (16% of jets, 63% q) and 1.86 when mass_top40 > 44.6 GeV (8.3%); it fades to 0.908 for 65.3-71.7 GeV and is nearly off (0.044) above 71.7 GeV with girth2_top40 ≤ 0.0142 (52% of jets). The 16% light-quark regime matters most: +1.51 to q, +1.39 to g and -0.589 to W. The regimes describe it very well (regime_r2 0.871).

Regimes (a small tree on its quantities; R² 0.871):

- `` — 2.5% of jets, value 4.06 (1.75…6.44), formula right 68%
- `` — 15.6% of jets, value 2.69 (2.19…3.19), formula right 79%
- `` — 8.3% of jets, value 1.86 (1.25…2.44), formula right 71%
- `` — 3.1% of jets, value 0.91 (0.31…1.44), formula right 61%
- `` — 18.6% of jets, value 0.33 (0.00…0.94), formula right 84%
- `` — 51.8% of jets, value 0.04 (0.00…0.06), formula right 84%

```
z = -0.968
if mass_top50 < 135: z += 0.041 × (135 − mass_top50)
if mass > 80.40: z += 0.097 × (mass − 80.40)
if mass > 63.00: z += -0.053 × (mass − 63.00)
if girth2_top15 < 0.021: z += -74.40 × (0.021 − girth2_top15)
if girth2_top40 < 0.006: z += 346 × (0.006 − girth2_top40)
if girth < 0.043: z += -70.70 × (0.043 − girth)
if LHA < 0.210: z += 14.30 × (0.210 − LHA)
if mass > 145: z += -0.054 × (mass − 145)
if sum_pt < 951: z += 0.020 × (951 − sum_pt)
if log_sum_pt < 6.85: z += -17.50 × (6.85 − log_sum_pt)
if width > 0.026: z += -240 × (width − 0.026)
if girth2_top15 < 0.018 and n_particles > 40.20: z += 1.25 × (0.018 − girth2_top15) × (n_particles − 40.20)
if girth > 0.098: z += 12.60 × (girth − 0.098)
if girth < 0.028: z += -26.60 × (0.028 − girth)
if z_top50_slots < 0.973: z += -34.80 × (0.973 − z_top50_slots)
if lam2 > 0.0015: z += -90.90 × (lam2 − 0.0015)
if girth2_top40 < 0.0061 and sum_pt < 1020: z += 1.71 × (0.0061 − girth2_top40) × (1020 − sum_pt)
if sum_pt_top40 < 965: z += 0.0028 × (965 − sum_pt_top40)
if n_dr_0p1_0p2 > 21.20: z += -0.027 × (n_dr_0p1_0p2 − 21.20)
if mass_top40 < 120 and max_dr > 0.380: z += 0.052 × (120 − mass_top40) × (max_dr − 0.380)
if log_sum_pt < 6.85 and dr_7 < 0.078: z += 159 × (6.85 − log_sum_pt) × (0.078 − dr_7)
if log_sum_pt < 6.85 and z_top20_slots > 0.898: z += 172 × (6.85 − log_sum_pt) × (z_top20_slots − 0.898)
if sum_pt_top40 < 965 and z_top30_slots > 0.970: z += -0.275 × (965 − sum_pt_top40) × (z_top30_slots − 0.970)
if e2 > 0.066: z += -45.40 × (e2 − 0.066)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W-mass jets** — 26.1% of jets, neuron 0.13, formula right for 82%. 69% W; mass about 79 GeV, width 0.0059, ordinary pT sharing, pT at 0.025-0.1 from the axis. 'mass_top50 < 135' passes for all and adds 2.304, but 'girth2_top15 < 0.0208' (-1.205) and 'mass > 63' (-0.865) cancel it. The value is 0.133 (on for 19.9%), removing only 0.029 from the W score. The formula calls them W and is right for 81.5%.
- **Z-mass jets** — 22.2% of jets, neuron 0.03, formula right for 88%. 75% Z; mass about 92 GeV, width 0.0079, and 41% of the pT at 0.05-0.1. 'mass_top50 < 135' (+1.805) and 'mass > 80.4' (+1.111) are cancelled by 'mass > 63' (-1.525) and 'girth2_top15 < 0.0208' (-1.074); the value is 0.025 (on for 6.3%) and the scores do not move. The formula calls them Z and is right for 88%.
- **Light narrow quark jets** — 14.4% of jets, neuron 2.69, formula right for 78%. 67% quark and 21% gluon; light (about 33 GeV), very narrow (width 0.001), a hard leading particle (351 GeV) and 92% of the pT within 0.025. 'mass_top50 < 135' (+4.176), 'girth2_top40 < 0.00603' (+1.743) and 'LHA < 0.21' (+1.522) outweigh 'girth < 0.0435' (-2.174) and 'girth2_top15 < 0.0208' (-1.521), and 'mass > 63' fails so there is no mass penalty. The value, 2.688 (always on), adds 1.386 to gluon and 1.512 to quark and removes 0.588 from W. The formula calls them q and is right for 78.4%.
- **Mid-mass gluon/quark jets** — 12.1% of jets, neuron 2.12, formula right for 72%. 50% gluon and 36% quark; mass about 56 GeV, width 0.0029, and 65% of the pT within 0.025. 'mass_top50 < 135' (+3.306) and 'girth2_top40 < 0.00603' (+1.256) outweigh 'girth2_top15 < 0.0208' (-1.468) and 'girth < 0.0435' (-0.889); 'mass > 63' passes for only 26%. The value, 2.122, adds 1.094 to gluon and 1.194 to quark. The formula splits them between g (52%) and q (45%) and is right for 72.1%.
- **Top-mass jets** — 10.0% of jets, neuron 0.24, formula right for 92%. 87% top; mass about 169 GeV, width 0.0273, pT shared out and mostly far from the axis. 'mass > 80.4' (+8.592) against 'mass > 63' (-5.605) and 'mass > 145' (-1.297), with 'width > 0.0258' (-0.611) and 'girth > 0.0981' (+0.606); 'mass_top50 < 135' fails here. The value, 0.243 (on for 51.2%), gives only about 0.13 to gluon and quark. The formula calls them t and is right for 91.6%.
- **115 GeV top/gluon mixture** — 6.0% of jets, neuron 0.15, formula right for 72%. 43% top and 40% gluon, with 15% quark; mass about 117 GeV, width 0.0131, pT fairly evenly shared. 'mass > 80.4' (+3.499) and 'mass_top50 < 135' (+0.929) against 'mass > 63' (-2.827) and 'girth2_top15 < 0.0208' (-0.857); the value is 0.152 (on for 35%), too small to matter. The formula splits them between t (50%) and g (41%) and is right for only 72%.
- **145 GeV top jets** — 5.8% of jets, neuron 0.47, formula right for 75%. 63% top and 26% gluon; mass about 144 GeV, width 0.0205, pT shared out and mostly far from the axis. 'mass > 80.4' (+6.176) outweighs 'mass > 63' (-4.288), with small terms from 'girth2_top15 < 0.0208' (-0.390) and 'girth > 0.0981' (+0.324). The value, 0.473 (on for 75.6%), adds 0.244 to gluon and 0.266 to quark. The formula calls them t and is right for 74.8%.
- **Heavy wide top/gluon jets** — 1.7% of jets, neuron 0.03, formula right for 79%. 54% top and 37% gluon; mass about 194 GeV, width 0.0342, and almost all pT far from the axis. 'mass > 80.4' (+11.007) is offset by 'mass > 63' (-6.923), 'mass > 145' (-2.637) and 'width > 0.0258' (-2.203); the value is 0.026 (on for 7%). The formula calls them t and is right for 78.6%.
- **Low-pT gluon/quark jets** — 1.4% of jets, neuron 3.12, formula right for 63%. 48% gluon, 34% quark and 17% top; mass about 59 GeV, width 0.0086, and a low total pT (711 GeV), with a hardest particle of 155 GeV. 'log_sum_pt < 6.85' (-5.135) and 'sum_pt < 951' (+4.882) pass for all only here and nearly cancel; 'mass_top50 < 135' (+3.184) and the two 'log_sum_pt < 6.85 and ...' tests (+1.568, +1.541) give the value 3.117, the neuron's highest. That adds 1.607 to gluon and 1.753 to quark and removes 0.682 from W. The formula calls them g, but is right for only 63.1%.
- **Very heavy gluon jets** — 0.4% of jets, neuron 0.01, formula right for 75%. 72% gluon and 19% top; very heavy (about 247 GeV), wide (width 0.0363), total pT 1336 GeV. 'mass > 80.4' (+16.077) is cancelled by 'mass > 63' (-9.688), 'mass > 145' (-5.450) and 'width > 0.0258' (-2.684); the value is 0.011. The formula calls them g and is right for 75.1%.

### neuron 12: Lightness: mass below 80 GeV (moderate)

- **What it measures:** Essentially on for jets lighter than 80.4 GeV, but pushed down for very light jets (hardest-50 mass below 60.1 GeV), very narrow jets and large C2. Quark jets sit highest (AUC 0.79), gluon jets next, W, top and Z jets low.
- *computed — its value:* largest for q (1.27), then g (0.60), then W (0.31), then t (0.18), then Z (0.17); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** It raises the q (+7%), g (+3%) and Z (+3%) scores and lowers the W (-4%) and t (-4%) scores: small corrections that credit light jets to the light-QCD scores and fine-tune the W/Z balance below the W mass.
- *computed — used by:* raises the score of g (+3%), q (+7%), Z (+3%); lowers the score of W (-4%), t (-4%) (share of each class score’s average input)
- **Boundaries:** High (1.80) for light jets (mass ≤ 64.5 GeV, C2 ≤ 0.0864) whose 2 hardest particles carry more than 350 GeV (13% of jets, 65% q), 1.26 when they carry less (8%, 52% g), 0.767 for C2 > 0.0864; low (0.127) for mass > 74 GeV with mass_top30 ≤ 163 GeV (68% of jets), with an odd island at 1.27 for mass_top30 > 163 GeV (2%, 61% top). In the main regime it adds +0.62 to q, +0.564 to Z and +0.423 to g and takes -0.733 from W and -0.676 from t. regime_r2 0.659.

Regimes (a small tree on its quantities; R² 0.659):

- `` — 13.0% of jets, value 1.80 (1.25…2.38), formula right 78%
- `` — 2.0% of jets, value 1.27 (0.00…4.25), formula right 84%
- `` — 8.0% of jets, value 1.26 (0.50…1.88), formula right 73%
- `` — 3.9% of jets, value 0.77 (0.00…1.38), formula right 71%
- `` — 4.7% of jets, value 0.50 (0.00…1.12), formula right 62%
- `` — 68.5% of jets, value 0.13 (0.00…0.38), formula right 84%

```
z = 0.119
if mass < 80.40: z += 0.113 × (80.40 − mass)
if C2 > 0.058: z += -13.40 × (C2 − 0.058)
if mass_top50 < 60.10: z += -0.046 × (60.10 − mass_top50)
if girth < 0.049: z += -23.50 × (0.049 − girth)
if mass < 84.70 and z_dr_0p1_0p2 < 0.065: z += -0.312 × (84.70 − mass) × (0.065 − z_dr_0p1_0p2)
if planar_flow > 0.266: z += -0.490 × (planar_flow − 0.266)
if mass_top40 < 94.40 and sum_pt_top2 < 471: z += -4.3e-05 × (94.40 − mass_top40) × (471 − sum_pt_top2)
if mass_top20 > 127 and C2 > 0.060: z += -1.34 × (mass_top20 − 127) × (C2 − 0.060)
if mass_top30 > 137: z += 0.035 × (mass_top30 − 137)
if log_sum_pt < 6.86: z += -6.56 × (6.86 − log_sum_pt)
if LHA > 0.400: z += 15.10 × (LHA − 0.400)
if z_dr_0_0p05 > 0.918: z += -7.28 × (z_dr_0_0p05 − 0.918)
if mass_top20 > 127: z += 0.040 × (mass_top20 − 127)
if mass < 91.20 and z_dr_0p05_0p1 > 0.320: z += 0.058 × (91.20 − mass) × (z_dr_0p05_0p1 − 0.320)
if z_dr_0p1_0p2 > 0.685 and min_pair_mass > 0.762: z += -1.78 × (z_dr_0p1_0p2 − 0.685) × (min_pair_mass − 0.762)
if LHA > 0.406 and lam2 < 0.0037: z += 12700 × (LHA − 0.406) × (0.0037 − lam2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Boson-mass jets, neuron near idle** — 53.3% of jets, neuron 0.14, formula right for 84%. 34% W and 34% Z, with 14% top and 13% gluon; mass about 93 GeV, width 0.0083; 53.3% of all jets. 'mass < 80.4' passes for only 26% (+0.067) and 'C2 > 0.0576' for 35% (-0.082), so nothing adds much and the value stays at 0.141 (on for 54.6%), with tiny score changes. The formula splits them between W (37%) and Z (34%) and is right for 84.5%.
- **Heavy high-C2 top jets** — 12.9% of jets, neuron 0.02, formula right for 80%. 60% top and 26% gluon; mass about 141 GeV, width 0.021, pT shared out (hardest particle 164 GeV). 'C2 > 0.0576' passes for all and subtracts 1.011, and 'mass < 80.4' never passes, so the value is 0.022 (on for 4.1%) and the neuron is out of play. The formula calls them t and is right for 79.8%.
- **Light narrow quark jets** — 7.6% of jets, neuron 1.52, formula right for 74%. 61% quark and 27% gluon; light (about 37 GeV), narrow (width 0.0013), and 85% of the pT within 0.025. 'mass < 80.4' passes for all and adds 4.901, trimmed by 'mass_top50 < 60.1' (-1.065), 'mass < 84.7 and z_dr_0p1_0p2 < 0.0652' (-0.760) and 'girth < 0.0488' (-0.739). The value, 1.519, adds 0.522 to quark, 0.356 to gluon and 0.475 to Z and removes 0.617 from W and 0.57 from top. The formula calls them q and is right for 73.5%.
- **Light gluon/quark mixture** — 7.4% of jets, neuron 1.27, formula right for 72%. 48% gluon and 40% quark; mass about 50 GeV, width 0.0024, and 72% of the pT within 0.025. 'mass < 80.4' (+3.382) with smaller subtractions from 'girth < 0.0488' (-0.519), 'mass_top50 < 60.1' (-0.501) and 'mass < 84.7 and z_dr_0p1_0p2 < 0.0652' (-0.419). The value, 1.273, adds 0.438 to quark and 0.298 to gluon. The formula splits them between q (51%) and g (49%) and is right for 72%.
- **Sub-W gluon-rich jets** — 6.9% of jets, neuron 0.77, formula right for 66%. 48% gluon, 29% quark and 11% W; mass about 65 GeV, width 0.004, and 55% of the pT within 0.025. 'mass < 80.4' adds 1.745, less than for lighter jets since it shrinks towards 80.4 GeV, with 'girth < 0.0488' (-0.279) and 'mass_top40 < 94.4 and sum_pt_top2 < 471' (-0.213) trimming. The value, 0.768 (on for 83.2%), adds 0.264 to quark and 0.18 to gluon and removes 0.312 from W. The formula calls them g but is right for only 66.4%.
- **Very light quark jets** — 6.6% of jets, neuron 1.83, formula right for 83%. 82% quark; very light (about 24 GeV), extremely narrow (width 0.0006), and 93% of the pT within 0.025. 'mass < 80.4' (+6.382) dominates 'mass_top50 < 60.1' (-1.654), 'mass < 84.7 and z_dr_0p1_0p2 < 0.0652' (-1.119) and 'girth < 0.0488' (-0.915). The value, 1.829, is the neuron's highest and adds 0.629 to quark and removes 0.743 from W. The formula calls them q and is right for 82.8%.
- **Top jets, heavy hard subset** — 3.9% of jets, neuron 0.92, formula right for 92%. 81% top; mass about 179 GeV, width 0.0292, and pT mostly far from the axis. 'mass_top30 > 137' (+0.859), 'mass_top20 > 127' (+0.664) and 'LHA > 0.4' (+0.488) are offset by 'mass_top20 > 127 and C2 > 0.0598' (-0.859, 90% pass) and 'C2 > 0.0576' (-0.699). The value, 0.92 (on for 46.6%), removes 0.345 from top and 0.374 from W and adds to the others, a small push against the right answer. The formula still calls them t and is right for 91.8%.
- **Top jets, high C2** — 1.2% of jets, neuron 0.25, formula right for 88%. 82% top; mass about 191 GeV, width 0.0327, pT far from the axis. 'mass_top20 > 127 and C2 > 0.0598' passes for all and subtracts 2.685, more than 'mass_top30 > 137' (+1.267) and 'mass_top20 > 127' (+1.122) return. The value is 0.248 (on for 16%), with little effect. The formula calls them t and is right for 88.5%.
- **Coreless top jets** — 0.2% of jets, neuron 0.00, formula right for 96%. 96% top; mass about 169 GeV, width 0.0276, with no pT within 0.05 of the axis and only 3% at 0.05-0.1, so no hard core at all; only 0.19% of jets. 'z_dr_0p1_0p2 > 0.685 and min_pair_mass > 0.762' passes for all and subtracts 7.462, forcing the value to 0.0. The formula calls them t and is right for 95.6%.
- **Very heavy gluon/top jets** — 0.1% of jets, neuron 0.14, formula right for 66%. 47% gluon and 43% top; very heavy (about 262 GeV), width 0.0428, total pT 1282 GeV; only 0.09% of jets. 'mass_top20 > 127 and C2 > 0.0598' (-9.472) outweighs 'mass_top30 > 137' (+2.986) and 'mass_top20 > 127' (+2.628); the value is 0.144 (on for 7.5%). The formula splits them between g (53%) and t (47%) and is right for only 66%.

### neuron 13: High pT, mass below the top (moderate)

- **What it measures:** Grows with the total jet pT and is pushed down for mass above 143 GeV (and hardest-50 mass above 98.1 GeV), while mass above 74.9 GeV pushes it up. All non-top types sit at similar, high values; top jets sit lowest (AUC 0.19 for t: small for t).
- *computed — its value:* largest for g (2.37), then Z (2.16), then W (1.97), then q (1.97), then t (1.01); it separates t jets from the rest best (AUC 0.19: small for t)
- **How the class scores use it:** Only the t score uses it, lowering it strongly (-38%): a high-pT jet whose mass is short of the top is not a top. It is the main negative top handle.
- *computed — used by:* lowers the score of t (-38%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** High for hard jets below the top mass: 2.78 for total pT > 993 GeV, mass ≤ 161 GeV and log pT > 6.95 (28% of jets, 40% g) and 2.10 for log pT ≤ 6.95 (41%, W/Z/q); lower (0.74-1.18) for mass > 161 GeV (58-92% top) and lowest (0.151) for pT ≤ 993 GeV with log pT ≤ 6.83 (6.7%). It enters only the top score, always subtracting: -2.52 and -1.90 in the two large regimes, making it the main brake on t for anything lighter than 161 GeV. regime_r2 0.765.

Regimes (a small tree on its quantities; R² 0.765):

- `` — 28.4% of jets, value 2.78 (2.22…3.28), formula right 82%
- `` — 40.9% of jets, value 2.10 (1.53…2.62), formula right 83%
- `` — 9.9% of jets, value 1.39 (0.69…2.06), formula right 73%
- `` — 3.6% of jets, value 1.18 (0.59…1.84), formula right 84%
- `` — 4.2% of jets, value 0.74 (0.22…1.31), formula right 94%
- `` — 4.1% of jets, value 0.59 (0.00…1.34), formula right 77%
- `` — 2.2% of jets, value 0.28 (0.00…0.76), formula right 93%
- `` — 6.7% of jets, value 0.15 (0.00…0.59), formula right 70%

```
z = 2.37
if mass > 143: z += -0.162 × (mass − 143)
if mass > 74.90: z += 0.025 × (mass − 74.90)
if log_sum_pt < 7.02: z += -5.92 × (7.02 − log_sum_pt)
if sum_pt < 1010: z += -0.026 × (1010 − sum_pt)
if sum_pt < 1060: z += -0.0092 × (1060 − sum_pt)
if mass_top50 > 98.10: z += -0.033 × (mass_top50 − 98.10)
if mass_top50 > 137: z += 0.090 × (mass_top50 − 137)
if sum_pt_top40 < 1050: z += 0.0065 × (1050 − sum_pt_top40)
if mass_over_sum_pt_sq < 0.0095: z += 104 × (0.0095 − mass_over_sum_pt_sq)
if sum_pt_top50 < 1010: z += 0.01 × (1010 − sum_pt_top50)
if mass_over_sum_pt > 0.171: z += 312 × (mass_over_sum_pt − 0.171)
if mass_over_sum_pt_sq > 0.029: z += -815 × (mass_over_sum_pt_sq − 0.029)
if mass_top10 > 57.10: z += 0.015 × (mass_top10 − 57.10)
if mass > 173: z += 0.159 × (mass − 173)
if sum_pt_top40 < 1030 and D2 < 4.94: z += -0.0012 × (1030 − sum_pt_top40) × (4.94 − D2)
if n_dr_0p2_0p4 < 3.96: z += -0.133 × (3.96 − n_dr_0p2_0p4)
if n_particles < 46.40: z += -0.013 × (46.40 − n_particles)
if n_particles > 46.00: z += 0.013 × (n_particles − 46.00)
if log_sum_pt < 6.81: z += 17.30 × (6.81 − log_sum_pt)
if sum_pt < 1070 and tau21 > 0.241: z += 0.004 × (1070 − sum_pt) × (tau21 − 0.241)
if mass > 161 and D2 < 6.47: z += -0.0053 × (mass − 161) × (6.47 − D2)
if sum_pt_top50 < 953 and D2 < 4.76: z += 0.0022 × (953 − sum_pt_top50) × (4.76 − D2)
if mass_top50 > 169: z += -0.046 × (mass_top50 − 169)
if sum_pt > 1250 and dr_7 < 0.080: z += -0.041 × (sum_pt − 1250) × (0.080 − dr_7)
if sum_pt > 1250: z += -0.0016 × (sum_pt − 1250)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Ordinary non-top jets** — 72.1% of jets, neuron 2.32, formula right for 82%. An even mixture of W (27%), Z (27%), quark (21%) and gluon (20%), with only 5% top; mass about 75 GeV, width 0.0056, total pT 1069 GeV; 72.1% of all jets. The tests are small here: 'mass_over_sum_pt_sq < 0.00947' (+0.434), 'log_sum_pt < 7.02' (-0.404), 'sum_pt < 1.06e+03' (-0.253) and 'mass > 74.9' (+0.249, 66% pass) roughly cancel. The neuron stays high at 2.323, which removes 2.105 from the top score for all of them. The formula spreads them over all four non-top classes and is right for 81.9%.
- **Low-pT mixed jets** — 10.4% of jets, neuron 0.90, formula right for 71%. 43% top, 26% quark and 21% gluon; mass about 86 GeV, width 0.0097, low total pT (947 GeV) and a softer leading particle (203 GeV). The low-pT tests subtract: 'sum_pt < 1.01e+03' (-1.653), 'sum_pt < 1.06e+03' (-1.035) and 'log_sum_pt < 7.02' (-0.988), only partly repaid by 'sum_pt_top40 < 1.05e+03' (+0.850), 'sum_pt_top50 < 1.01e+03' (+0.734) and 'mass > 74.9' (+0.524). The value, 0.904 (on for 82.5%), removes 0.82 from top. The formula calls them t but is right for only 71.1%.
- **Top-mass jets** — 6.7% of jets, neuron 0.68, formula right for 93%. 86% top; mass about 173 GeV, width 0.0271, pT shared out and mostly far from the axis. 'mass > 143' (-4.855) and 'mass_top50 > 98.1' (-2.354) outweigh 'mass_top50 > 137' (+2.875) and 'mass > 74.9' (+2.422), pulling the value down to 0.679 (on for 88.5%). It removes only 0.615 from the top score, much less than for other jets, so tops keep their score. The formula calls them t and is right for 92.8%.
- **Top jets below top mass** — 5.0% of jets, neuron 1.23, formula right for 84%. 78% top and 16% gluon; mass about 158 GeV, width 0.0245, pT shared out. 'mass > 143' (-2.429) and 'mass_top50 > 98.1' (-1.811) against 'mass > 74.9' (+2.052) and 'mass_top50 > 137' (+1.406), with 'sum_pt < 1.01e+03' (-0.690, 62% pass). The value, 1.225, removes 1.11 from top. The formula calls them t and is right for 84.3%.
- **Low-pT top/gluon/quark mixture** — 2.6% of jets, neuron 0.11, formula right for 68%. 40% top, 34% gluon and 25% quark; mass about 81 GeV, width 0.0112, and a low total pT (834 GeV). 'sum_pt < 1.01e+03' (-4.641), 'sum_pt < 1.06e+03' (-2.077) and 'log_sum_pt < 7.02' (-1.749) outweigh 'sum_pt_top50 < 1.01e+03' (+1.869), 'sum_pt_top40 < 1.05e+03' (+1.589) and 'log_sum_pt < 6.81' (+1.480). The value is 0.112 (on for 26.3%), so the top score is not held down. The formula calls them t (43%), with g at 34%, and is right for only 67.5%.
- **Wide low-pT top jets** — 1.3% of jets, neuron 0.41, formula right for 86%. 80% top; mass about 180 GeV, width 0.035, low total pT (962 GeV) and almost all pT far from the axis. 'mass > 143' (-5.914) and 'mass_over_sum_pt_sq > 0.0292' (-4.654) outweigh 'mass_over_sum_pt > 0.171' (+4.919), 'mass_top50 > 137' (+3.249) and 'mass > 74.9' (+2.584). The value is 0.408 (on for 44.1%), removing only 0.37 from top. The formula calls them t and is right for 85.6%.
- **Very low-pT gluon/quark jets** — 0.7% of jets, neuron 0.07, formula right for 60%. 52% gluon, 32% quark and 17% top; mass about 61 GeV, width 0.0107, very low total pT (647 GeV). 'sum_pt < 1.01e+03' (-9.540), 'sum_pt < 1.06e+03' (-3.785) and 'log_sum_pt < 7.02' (-3.293) outweigh 'log_sum_pt < 6.81' (+5.990), 'sum_pt_top50 < 1.01e+03' (+3.697) and 'sum_pt_top40 < 1.05e+03' (+2.757). The value is 0.065, so these jets lose the anti-top protection; the formula calls them g but is right for only 59.5%.
- **Very heavy high-pT gluon jets** — 0.5% of jets, neuron 1.43, formula right for 80%. 79% gluon and 12% top; mass about 218 GeV, width 0.028, total pT 1343 GeV. 'mass > 143' (-12.219) and 'mass_top50 > 98.1' (-3.691) are outweighed by 'mass > 173' (+7.255), 'mass_top50 > 137' (+6.490) and 'mass > 74.9' (+3.545), so the value returns to 1.431. That removes 1.297 from top, which helps here; the formula calls them g and is right for 79.8%.
- **Very wide low-pT top jets** — 0.5% of jets, neuron 0.65, formula right for 67%. 65% top and 26% gluon; mass about 200 GeV, the widest group (width 0.0441), low total pT (957 GeV). 'mass_over_sum_pt_sq > 0.0292' (-12.057) and 'mass_over_sum_pt > 0.171' (+12.015) cancel, and 'mass > 143' (-9.261) is offset by 'mass_top50 > 137' (+4.831), 'mass > 173' (+4.480) and 'mass > 74.9' (+3.093). The value, 0.649 (on for 57.8%), removes 0.588 from top. The formula calls them t but is right for only 67.3%.
- **Heaviest gluon jets** — 0.1% of jets, neuron 2.23, formula right for 74%. 71% gluon and 20% top; very heavy (about 271 GeV), width 0.0426, total pT 1332 GeV; only 0.13% of jets. 'mass > 143' (-20.740) and 'mass_over_sum_pt_sq > 0.0292' (-10.894) are offset by 'mass > 173' (+15.617), 'mass_top50 > 137' (+10.955) and 'mass_over_sum_pt > 0.171' (+10.799), so the value is 2.225. That removes 2.017 from top; the formula calls them g and is right for 73.7%.

### neuron 14: Mass just above the Z (moderate)

- **What it measures:** Large for mass between 91.2 and 136 GeV, especially with m/pT between 0.0907 and 0.0984. Z jets sit highest (AUC 0.88), gluon and top jets well below, quark and W jets lowest.
- *computed — its value:* largest for Z (1.52), then g (0.52), then t (0.41), then q (0.16), then W (0.15); it separates Z jets from the rest best (AUC 0.88: large for Z)
- **How the class scores use it:** Only the W score uses it, lowering it (-14%): a jet heavier than the Z peak is not a W. The Z score hardly uses it even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-14%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** High for masses above the W: 1.67 for mass > 87.4 GeV with mass_top40 ≤ 104 GeV (24% of jets, 68% Z) and 1.76 for mass > 83.8 GeV, mass_top40 > 104 GeV and the 50 hardest slots holding ≤ 0.952 of the pT (2%); 0.93 for 83.8-87.4 GeV, low (0.206) for 79.8-83.8 GeV with m/pT ≤ 0.0809 (9.2%, 86% W) and off (0.007) below 79.8 GeV with m/pT ≤ 0.077 (35%). It only subtracts from W: -2.29 in the 24% Z-mass regime, -1.28 at 83.8-87.4 GeV. regime_r2 0.777.

Regimes (a small tree on its quantities; R² 0.777):

- `` — 2.0% of jets, value 1.76 (0.75…3.00), formula right 76%
- `` — 23.5% of jets, value 1.67 (0.97…2.25), formula right 87%
- `` — 3.9% of jets, value 0.93 (0.34…1.56), formula right 70%
- `` — 2.0% of jets, value 0.53 (0.09…0.97), formula right 65%
- `` — 19.6% of jets, value 0.28 (0.00…0.88), formula right 83%
- `` — 9.2% of jets, value 0.21 (0.00…0.50), formula right 92%
- `` — 4.8% of jets, value 0.11 (0.00…0.31), formula right 84%
- `` — 35.0% of jets, value 0.01 (0.00…0.00), formula right 75%

```
z = 0.205
if mass_over_sum_pt < 0.091: z += -165 × (0.091 − mass_over_sum_pt)
if mass_over_sum_pt < 0.098: z += 116 × (0.098 − mass_over_sum_pt)
if mass < 91.20: z += -0.132 × (91.20 − mass)
if mass < 136: z += 0.030 × (136 − mass)
if girth2_top50 < 0.0094: z += -207 × (0.0094 − girth2_top50)
if lam2 < 0.0024: z += 433 × (0.0024 − lam2)
if lam1 < 0.0063: z += 260 × (0.0063 − lam1)
if girth2_top20 < 0.017 and z_top50_slots > 0.969: z += -1310 × (0.017 − girth2_top20) × (z_top50_slots − 0.969)
if mass_top40 < 80.70: z += -0.026 × (80.70 − mass_top40)
if max_dr > 0.249: z += -2.43 × (max_dr − 0.249)
if n_dr_0p2_0p4 < 21.40: z += 0.019 × (21.40 − n_dr_0p2_0p4)
if sum_pt_top50 > 978: z += -0.0031 × (sum_pt_top50 − 978)
if log_sum_pt > 6.94: z += 5.53 × (log_sum_pt − 6.94)
if girth2_top20 < 0.016 and tau21 < 0.637: z += -94.90 × (0.016 − girth2_top20) × (0.637 − tau21)
if n_dr_0p2_0p4 < 19.90 and n_dr_0p1_0p2 < 21.30: z += -0.0011 × (19.90 − n_dr_0p2_0p4) × (21.30 − n_dr_0p1_0p2)
if lam1 < 0.0062 and z_top50_slots > 0.987: z += 8090 × (0.0062 − lam1) × (z_top50_slots − 0.987)
if max_dr > 0.201 and mass_top10 < 52.10: z += 0.040 × (max_dr − 0.201) × (52.10 − mass_top10)
if z_top50_slots < 0.959: z += 89.60 × (0.959 − z_top50_slots)
if m012 > 36.20: z += -0.011 × (m012 − 36.20)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Heavy top/gluon jets** — 25.4% of jets, neuron 0.54, formula right for 80%. 66% top and 23% gluon; mass about 149 GeV, width 0.0222, pT shared out and mostly far from the axis. 'max_dr > 0.249' passes for 99% and subtracts 0.315, and the window test 'mass < 136' passes for only 34% (+0.224); the value, 0.542 (on for 64.2%), removes 0.746 from the W score. The formula calls them t and is right for 79.8%.
- **Z jets in the window** — 19.9% of jets, neuron 1.71, formula right for 88%. 78% Z; mass about 91 GeV, width 0.0079, and 42% of the pT at 0.05-0.1. 'mass < 136' (+1.366), 'mass_over_sum_pt < 0.0984' (+1.119) and 'lam2 < 0.00238' (+0.725) add, while 'mass_over_sum_pt < 0.0907' passes for 79% (-0.409) and 'mass < 91.2' for 58% (-0.215), so the low-mass penalties stay small. The value, 1.705 (on for 98.6%), removes 2.345 from W. The formula calls them Z and is right for 88.1%.
- **W-mass jets** — 18.1% of jets, neuron 0.35, formula right for 86%. 72% W and 12% Z; mass about 82 GeV, width 0.0062, pT at 0.025-0.1 from the axis. 'mass_over_sum_pt < 0.0984' (+2.251) and 'mass < 136' (+1.646) are cut back by 'mass_over_sum_pt < 0.0907' (-1.933) and 'mass < 91.2' (-1.290, 96% pass). The value, 0.346 (on for 69.4%), removes 0.476 from W, a push against their true class. The formula still calls them W and is right for 86%.
- **Sub-W W/gluon jets** — 7.7% of jets, neuron 0.11, formula right for 77%. 59% W and 20% gluon; mass about 77 GeV, width 0.0053, and 30% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-3.011) and 'mass_over_sum_pt < 0.0984' (+3.010) cancel, and 'mass < 91.2' (-1.885) about balances 'mass < 136' (+1.788); the value is 0.107 (on for 22.4%). The formula calls them W but is right for only 76.7%.
- **Light quark jets** — 6.2% of jets, neuron 0.00, formula right for 78%. 69% quark and 20% gluon; light (about 32 GeV), narrow (width 0.001), and 89% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-9.858) and 'mass < 91.2' (-7.786) outweigh 'mass_over_sum_pt < 0.0984' (+7.823) and 'mass < 136' (+3.145); the value is 0.0. The formula calls them q and is right for 77.5%.
- **Light quark/gluon mixture** — 6.0% of jets, neuron 0.00, formula right for 75%. 48% quark and 41% gluon; mass about 43 GeV, width 0.0016, and 80% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-8.290) and 'mass < 91.2' (-6.346) outweigh 'mass_over_sum_pt < 0.0984' (+6.721) and 'mass < 136' (+2.814); the value is 0.0. The formula splits them between q (59%) and g (41%) and is right for 75.1%.
- **Mid-light gluon/quark jets** — 5.8% of jets, neuron 0.00, formula right for 72%. 50% gluon and 38% quark; mass about 54 GeV, width 0.0026, and 70% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-6.648) and 'mass < 91.2' (-4.955) outweigh 'mass_over_sum_pt < 0.0984' (+5.567) and 'mass < 136' (+2.495); the value is 0.0. The formula splits them between g (51%) and q (48%) and is right for 72.3%.
- **Sub-W gluon/quark jets** — 4.7% of jets, neuron 0.00, formula right for 64%. 43% gluon, 35% quark and 10% W; mass about 63 GeV, width 0.0039, and 55% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-4.679) and 'mass < 91.2' (-3.704) outweigh 'mass_over_sum_pt < 0.0984' (+4.183) and 'mass < 136' (+2.208); the value is 0.003. The formula splits them almost evenly between g (47%) and q (46%) and is right for only 64%, the hardest group here.
- **Very light quark jets** — 4.3% of jets, neuron 0.00, formula right for 84%. 84% quark; very light (about 22 GeV), extremely narrow (width 0.0005), and 95% of the pT within 0.025. 'mass_over_sum_pt < 0.0907' (-11.497) and 'mass < 91.2' (-9.179) outweigh 'mass_over_sum_pt < 0.0984' (+8.976) and 'mass < 136' (+3.464); the value is 0.0. The formula calls them q and is right for 84%.
- **High-pT gluons at W mass** — 1.9% of jets, neuron 0.39, formula right for 88%. 60% gluon, 26% W and 10% Z; mass about 84 GeV, width 0.0037, and a high total pT (1393 GeV), which makes m/pT small. 'mass_over_sum_pt < 0.0907' (-5.006) is mostly repaid by 'mass_over_sum_pt < 0.0984' (+4.413); 'log_sum_pt > 6.94' (+1.609) and 'mass < 136' (+1.592) add, and 'mass < 91.2' (-1.266), 'sum_pt_top50 > 978' (-1.252) and 'girth2_top50 < 0.00935' (-1.225) subtract. The value, 0.392 (on for 43.9%), removes 0.539 from W, which helps here. The formula calls them g and is right for 88.3%.

### neuron 2: Total pT in the hardest particles (minor)

- **What it measures:** Follows the total pT carried by the 20-30 hardest particles and rises when the 30 hardest carry more than 0.919 of the jet pT; jets lighter than 91.2 GeV are pushed down. It varies little between jet types: slightly highest for gluons and Z jets, lowest for top jets.
- *computed — its value:* largest for g (0.56), then Z (0.46), then q (0.36), then W (0.32), then t (0.25); it separates t jets from the rest best (AUC 0.32: small for t)
- **How the class scores use it:** Only the Z score uses it, lowering it slightly (-3%); it is a small correction.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** Rises only for very hard jets with total pT above 1.24 TeV (three regimes of about 2% each, 68-80% gluons), reaching 2.17 when C2 > 0.07; below 1.24 TeV it stays between 0.105 and 0.559, set by whether the 50 hardest carry more than 1.03 TeV and the 30 hardest slots hold more than about 0.96 of the pT. It only lowers Z (-0.813 at most) and raises q slightly (+0.271 at most), so its effect on the scores is small; the regimes describe it only partly (regime_r2 0.513).

Regimes (a small tree on its quantities; R² 0.513):

- `` — 2.0% of jets, value 2.17 (0.62…5.12), formula right 83%
- `` — 2.0% of jets, value 1.46 (0.88…2.12), formula right 90%
- `` — 2.0% of jets, value 0.85 (0.50…1.12), formula right 86%
- `` — 23.2% of jets, value 0.56 (0.38…0.75), formula right 82%
- `` — 12.5% of jets, value 0.34 (0.00…0.62), formula right 82%
- `` — 32.0% of jets, value 0.32 (0.12…0.50), formula right 82%
- `` — 26.2% of jets, value 0.10 (0.00…0.25), formula right 78%

```
z = 0.112
if z_top30_slots > 0.919: z += 7.92 × (z_top30_slots − 0.919)
if mass < 91.20: z += -0.019 × (91.20 − mass)
if sum_pt > 1090: z += -0.011 × (sum_pt − 1090)
if sum_pt > 1010: z += 0.005 × (sum_pt − 1010)
if mass_over_sum_pt < 0.073: z += 23.70 × (0.073 − mass_over_sum_pt)
if lam2 < 0.00096: z += -408 × (0.00096 − lam2)
if log_sum_pt > 7.05: z += 11.70 × (log_sum_pt − 7.05)
if mass_top40 > 158: z += -0.046 × (mass_top40 − 158)
if sum_pt_top50 > 1110 and C2 > 0.096: z += 0.754 × (sum_pt_top50 − 1110) × (C2 − 0.096)
if sum_pt_top50 > 1030 and mean_phi > 3.2e-05: z += 3.93 × (sum_pt_top50 − 1030) × (mean_phi − 3.2e-05)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **W/Z jets, pT in top 30** — 37.6% of jets, neuron 0.42, formula right for 85%. 40% W and 37% Z, with 11% top; mass about 87 GeV, width 0.0075, average total pT, and pT held mostly at 0.025-0.1 from the axis. 'z_top30_slots > 0.919' passes for all and adds 0.543, trimmed by 'lam2 < 0.000964' (-0.202) and 'mass < 91.2' (-0.156, 76% pass). The value, 0.419 (on for 95.4%), adds only 0.052 to the quark score and removes 0.157 from the Z score. The formula splits them between W (43%) and Z (36%) and is right for 85%.
- **Wide jets, pT shared thinly** — 33.3% of jets, neuron 0.16, formula right for 79%. 46% top and 23% gluon, with 10-13% each of W and Z; mass about 122 GeV, width 0.0166, and evenly shared pT (hardest particle 156 GeV), spread far from the axis. No test decides this group: 'sum_pt > 1.01e+03' passes for half (+0.113) and 'z_top30_slots > 0.919' for only 40% (+0.071), while 'mass < 91.2' (-0.085) and 'mass_top40 > 158' (-0.084) take off a little. The value is 0.16 (on for 66.5%) and changes the Z score by only -0.06. The formula calls them t and is right for 78.8%.
- **Light narrow quark jets** — 18.9% of jets, neuron 0.34, formula right for 75%. 65% quark and 22% gluon; light (about 36 GeV) and narrow (width 0.0014), a hard leading particle (310 GeV) and 84% of the pT within 0.025 of the axis. 'mass < 91.2' passes for all and subtracts 1.063, but 'mass_over_sum_pt < 0.0732' (+0.881) and 'z_top30_slots > 0.919' (+0.547) make up for it. The value, 0.342, adds 0.043 to the quark score and removes 0.128 from the Z score, a minor push. The formula calls them q and is right for 75.3%.
- **High-pT gluon jets, 90 GeV** — 6.0% of jets, neuron 0.65, formula right for 83%. 62% gluon, the rest spread thin; mass about 90 GeV, width 0.0067, total pT 1220 GeV (well above average) and 40% of the pT within 0.025. The total-pT tests fight: 'sum_pt > 1.09e+03' takes away 1.408 while 'sum_pt > 1.01e+03' (+1.052) and 'log_sum_pt > 7.05' (+0.659) add, with 'mass < 91.2' (-0.318) and 'mass_over_sum_pt < 0.0732' (+0.295) as minor terms. The value, 0.652, adds 0.082 to the quark score and removes 0.245 from the Z score. The formula calls them g and is right for 83.2%.
- **Very high-pT gluon jets** — 2.6% of jets, neuron 1.23, formula right for 88%. 77% gluon; mass about 94 GeV, width 0.0057, and a total pT of 1392 GeV. 'sum_pt > 1.09e+03' (-3.259) is outweighed by 'log_sum_pt > 7.05' (+2.192) and 'sum_pt > 1.01e+03' (+1.909), so the value rises with the jet pT to 1.228. That removes 0.46 from the Z score and adds 0.153 to the quark score; the formula calls them g and is right for 87.9%.
- **Highest-pT gluon jets** — 0.8% of jets, neuron 1.93, formula right for 91%. 82% gluon; mass about 104 GeV, width 0.0049, and a total pT of 1698 GeV, with the hardest particle at 365 GeV. Same three total-pT tests, larger: 'sum_pt > 1.09e+03' (-6.562) against 'log_sum_pt > 7.05' (+4.478) and 'sum_pt > 1.01e+03' (+3.438), giving 1.932. This removes 0.725 from the Z score and adds 0.242 to the quark score; the formula calls them g and is right for 91.2%.
- **Heavy high-pT gluon/top jets** — 0.5% of jets, neuron 3.65, formula right for 80%. 64% gluon and 25% top; mass about 160 GeV, width 0.0168, and a total pT of 1290 GeV. 'sum_pt_top50 > 1.11e+03 and C2 > 0.0963' passes for all and adds 3.054, which marks this group and the two heavier ones after it; with 'sum_pt > 1.01e+03' (+1.400) and 'log_sum_pt > 7.05' (+1.298) against 'sum_pt > 1.09e+03' (-2.159), the value reaches 3.647. That removes 1.368 from the Z score and adds 0.456 to the quark score. The formula splits them between g (69%) and t (27%) and is right for 80.5%.
- **Top-mass high-pT gluon jets** — 0.2% of jets, neuron 4.15, formula right for 82%. 76% gluon and 18% top; mass about 179 GeV, width 0.0174, and a total pT of 1435 GeV. 'sum_pt_top50 > 1.11e+03 and C2 > 0.0963' adds 7.775 and outweighs 'sum_pt > 1.09e+03' (-3.730), with 'log_sum_pt > 7.05' (+2.534) and 'sum_pt > 1.01e+03' (+2.127) on top; 'mass_top40 > 158' (-0.886) trims it. The value, 4.15, removes 1.556 from the Z score; the formula calls them g and is right for 81.5%.
- **Rare very heavy gluon jets** — 0.0% of jets, neuron 4.01, formula right for 88%. 88% gluon; mass about 238 GeV, width 0.023, and a total pT of 1687 GeV; only 0.04% of jets. 'sum_pt_top50 > 1.11e+03 and C2 > 0.0963' adds 18.714, against 'sum_pt > 1.09e+03' (-6.452) and 'mass_top40 > 158' (-2.455), for a value of 4.01. This removes 1.504 from the Z score; the formula calls them g and is right for 88.5%.

### neuron 11: Mass 80-91 GeV, clean two-prong (minor)

- **What it measures:** Large for mass between 80.4 and 91.2 GeV, especially with few particles at 0.2 <= ΔR < 0.4 and an elongated (two-prong) pattern; very narrow jets are pushed down. W jets sit highest (AUC 0.82), Z jets next, top, quark and gluon jets low.
- *computed — its value:* largest for W (1.68), then Z (1.07), then t (0.38), then q (0.27), then g (0.25); it separates W jets from the rest best (AUC 0.82: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the q score (-7%): a clean two-prong jet at the W mass is a W, not a quark jet. The g, Z and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of q (-7%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** High (2.72) for jets with an almost empty 0.2 < ΔR < 0.4 ring (pT share ≤ 0.00251), not too narrow (girth2_top15 > 0.00345) and flat, elongated (planar_flow ≤ 0.181) (9.7% of jets, 52% W and 45% Z), 1.97 with planar_flow > 0.181 (6.4%, 57% W); for the 64% with a fuller ring, D2 > 1.24 and max_dr > 0.265 it sits low (0.284) but never fully off. It raises W (+1.62 at the top, +1.17 next, +0.168 on the 64% bulk) and lowers q (-0.68 at the top). regime_r2 0.775.

Regimes (a small tree on its quantities; R² 0.775):

- `` — 9.7% of jets, value 2.72 (1.81…3.69), formula right 97%
- `` — 6.4% of jets, value 1.97 (1.25…2.78), formula right 95%
- `` — 6.2% of jets, value 1.38 (0.56…2.25), formula right 90%
- `` — 3.0% of jets, value 0.80 (0.12…1.56), formula right 86%
- `` — 2.6% of jets, value 0.68 (0.31…1.06), formula right 84%
- `` — 6.2% of jets, value 0.47 (0.00…0.94), formula right 76%
- `` — 2.0% of jets, value 0.32 (0.00…1.13), formula right 79%
- `` — 63.9% of jets, value 0.28 (0.00…0.62), formula right 77%

```
z = -0.037
if mass < 91.20: z += 0.064 × (91.20 − mass)
if mass < 80.40: z += -0.056 × (80.40 − mass)
if girth2_top30 < 0.0064: z += -306 × (0.0064 − girth2_top30)
if e2 < 0.026: z += 72.10 × (0.026 − e2)
if n_dr_0p2_0p4 < 9.59 and n_dr_0p1_0p2 < 22.00: z += 0.0051 × (9.59 − n_dr_0p2_0p4) × (22.00 − n_dr_0p1_0p2)
if z_dr_0p2_0p4 < 0.0063: z += 124 × (0.0063 − z_dr_0p2_0p4)
if mass_top50 < 89.60 and girth2_top15 < 0.0036: z += 4.37 × (89.60 − mass_top50) × (0.0036 − girth2_top15)
if n_dr_0p2_0p4 < 9.84 and girth2 < 0.0057: z += -30.50 × (9.84 − n_dr_0p2_0p4) × (0.0057 − girth2)
if n_dr_0p2_0p4 < 8.15 and z_dr_0p2_0p4 < 0.061: z += 1.02 × (8.15 − n_dr_0p2_0p4) × (0.061 − z_dr_0p2_0p4)
if mass < 80.40 and z_dr_0p2_0p4 < 0.034: z += -0.540 × (80.40 − mass) × (0.034 − z_dr_0p2_0p4)
if z_top5_slots > 0.545: z += -1.80 × (z_top5_slots − 0.545)
if z_dr_0_0p05 < 0.099: z += 5.36 × (0.099 − z_dr_0_0p05)
if mass < 106 and planar_flow < 0.368: z += 0.071 × (106 − mass) × (0.368 − planar_flow)
if n_dr_0p2_0p4 < 10.70 and z_dr_0p05_0p1 > 0.593: z += -0.331 × (10.70 − n_dr_0p2_0p4) × (z_dr_0p05_0p1 − 0.593)
if mass < 80.40 and D2 < 3.33: z += -0.019 × (80.40 − mass) × (3.33 − D2)
if mass < 103 and mass_top10 > 45.10: z += 0.00068 × (103 − mass) × (mass_top10 − 45.10)
if mass < 80.40 and sum_pt_top2 < 467: z += -6.8e-05 × (80.40 − mass) × (467 − sum_pt_top2)
if mass < 98.70 and D2 < 1.30: z += 0.033 × (98.70 − mass) × (1.30 − D2)
if max_dr < 0.296: z += 3.27 × (0.296 − max_dr)
if mass < 98.90 and n_dr_0p2_0p4 > 10.10: z += -0.003 × (98.90 − mass) × (n_dr_0p2_0p4 − 10.10)
if z_dr_0p05_0p1 > 0.851: z += 6.62 × (z_dr_0p05_0p1 − 0.851)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **110 GeV Z/top/gluon jets** — 21.2% of jets, neuron 0.23, formula right for 79%. 40% Z, 27% top and 22% gluon; mass about 111 GeV, width 0.0122, and 39% of the pT at 0.025-0.05 from the axis. No test passes for most of them: 'z_top5_slots > 0.545' passes for 44% (-0.101) and 'mass < 91.2' for 34% (+0.060), so the value stays near its base at 0.234 (on for 51.5%), adding only 0.139 to W. The formula calls them Z (43%), with t at 28%, and is right for 79.3%.
- **Top jets without a core** — 17.6% of jets, neuron 0.48, formula right for 82%. 70% top and 15% gluon; mass about 151 GeV, width 0.0231, pT shared out, with essentially no pT within 0.025 of the axis. Only 'z_dr_0_0p05 < 0.0985' passes for all (+0.463), because these jets have almost no pT in the core; the others rarely pass. The value, 0.477 (on for 99.3%), adds 0.283 to W and removes 0.119 from quark. The formula calls them t and is right for 82.3%.
- **W/Z jets, quiet outer ring** — 13.3% of jets, neuron 2.26, formula right for 96%. 49% W and 48% Z; mass about 86 GeV, width 0.0068, and 47% of the pT at 0.05-0.1 from the axis. 'z_dr_0p2_0p4 < 0.00633' (+0.640), 'n_dr_0p2_0p4 < 9.59 and n_dr_0p1_0p2 < 22' (+0.415), 'n_dr_0p2_0p4 < 8.15 and z_dr_0p2_0p4 < 0.0608' (+0.413), 'mass < 91.2' (+0.391, 80% pass) and 'mass < 106 and planar_flow < 0.368' (+0.268) all add. The value, 2.263, adds 1.344 to W and removes 0.566 from quark. The formula splits them between W (50%) and Z (48%) and is right for 96.2%; the W push is offset elsewhere for the Z jets.
- **W-mass jets with harder core** — 10.6% of jets, neuron 0.76, formula right for 75%. 68% W; mass about 79 GeV, width 0.0061, a harder leading particle (269 GeV) and 24% of the pT within 0.025. 'mass < 91.2' (+0.803) carries it, trimmed by 'girth2_top30 < 0.00644' (-0.294), 'z_top5_slots > 0.545' (-0.180) and 'mass < 80.4' (-0.127, 67% pass). The value, 0.756 (on for 89.7%), adds 0.449 to W. The formula calls them W and is right for 75.3%.
- **Light quark jets** — 7.6% of jets, neuron 0.17, formula right for 75%. 64% quark and 25% gluon; light (about 35 GeV), narrow (width 0.0012), and 86% of the pT within 0.025. 'mass < 91.2' (+3.584) is cancelled by 'mass < 80.4' (-2.539) and 'girth2_top30 < 0.00644' (-1.662), with 'e2 < 0.0258' (+1.169) adding back. The value is 0.171 (on for 57.6%), only a slight W push. The formula calls them q and is right for 75.2%.
- **Light gluon/quark mixture** — 7.6% of jets, neuron 0.18, formula right for 73%. 48% gluon and 41% quark; mass about 49 GeV, width 0.0022, and 74% of the pT within 0.025. 'mass < 91.2' (+2.715) against 'mass < 80.4' (-1.776) and 'girth2_top30 < 0.00644' (-1.488), with 'e2 < 0.0258' (+0.975). The value, 0.176, adds just 0.105 to W. The formula splits them between q (52%) and g (48%) and is right for 72.6%.
- **Mid-mass gluon/quark jets** — 6.7% of jets, neuron 0.24, formula right for 68%. 50% gluon and 31% quark; mass about 63 GeV, width 0.0037, and 58% of the pT within 0.025. 'mass < 91.2' (+1.804) against 'girth2_top30 < 0.00644' (-1.164) and 'mass < 80.4' (-0.976), with 'e2 < 0.0258' (+0.721). The value, 0.237, adds 0.141 to W. The formula calls them g but is right for only 67.6%.
- **Very light quark jets** — 5.6% of jets, neuron 0.45, formula right for 83%. 83% quark; very light (about 23 GeV), extremely narrow (width 0.0005), and 94% of the pT within 0.025. 'mass < 91.2' (+4.369) and 'e2 < 0.0258' (+1.380) beat 'mass < 80.4' (-3.229), 'girth2_top30 < 0.00644' (-1.813) and 'n_dr_0p2_0p4 < 9.84 and girth2 < 0.0057' (-1.159). The value, 0.449 (on for 87.1%), adds 0.266 to W, a push the wrong way. The formula calls them q and is right for 83.3%.
- **Clean two-prong W jets** — 5.2% of jets, neuron 2.62, formula right for 96%. 69% W and 28% Z; mass about 83 GeV, width 0.0064, pT evenly shared between the leading particles (218 and 140 GeV) and 91% of the pT at 0.05-0.1 from the axis. 'n_dr_0p2_0p4 < 10.7 and z_dr_0p05_0p1 > 0.593' passes for all and subtracts 0.958, but 'n_dr_0p2_0p4 < 9.59 and n_dr_0p1_0p2 < 22' (+0.661), 'z_dr_0p2_0p4 < 0.00633' (+0.621), 'mass < 91.2' (+0.533), 'n_dr_0p2_0p4 < 8.15 and z_dr_0p2_0p4 < 0.0608' (+0.410) and 'z_dr_0p05_0p1 > 0.851' (+0.405) outweigh it. The value, 2.616, is the neuron's highest and adds 1.553 to W. The formula calls them W and is right for 95.7%.
- **High-pT gluons at W mass** — 4.5% of jets, neuron 0.31, formula right for 78%. 67% gluon and 15% W; mass about 83 GeV, width 0.0052, total pT 1187 GeV and 49% of the pT within 0.025. 'girth2_top30 < 0.00644' (-0.974) is offset by 'e2 < 0.0258' (+0.741) and 'mass < 91.2' (+0.639); the value is 0.309 (on for 64.1%), adding 0.184 to W. The formula calls them g and is right for 77.8%.

### neuron 15: Narrow hard core (weak) (minor)

- **What it measures:** Grows linearly with the pT of the 30 hardest particles and with pT concentrated close to the axis (large share within ΔR < 0.05, small LHA). It is small for all types: slightly highest for quark jets, lowest for top jets.
- *computed — its value:* largest for q (0.31), then g (0.21), then W (0.19), then Z (0.17), then t (0.11); it separates q jets from the rest best (AUC 0.65: large for q)
- **How the class scores use it:** It raises the Z score slightly (+2%); the g, q, W and t scores hardly use it.
- *computed — used by:* raises the score of Z (+2%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** Weak and mostly low: 1.43 only for jets whose 5 hardest particles are tightly packed (girth2_top5 ≤ 0.00404) at low pT (log pT ≤ 6.79, 2.1% of jets), 0.393 for the same at log pT > 6.79 and pT ≤ 1.04 TeV (28%), and near off (0.005) when the 5 hardest are spread and more than 0.174 of the pT sits at 0.1 < ΔR < 0.2 (31%). It adds to Z (+0.802 at most, +0.221 in the 28% regime) and takes from t (-0.535) and W (-0.267); its effect is small and the regimes describe it poorly (regime_r2 0.458).

Regimes (a small tree on its quantities; R² 0.458):

- `` — 2.1% of jets, value 1.43 (0.56…2.44), formula right 66%
- `` — 28.2% of jets, value 0.39 (0.00…0.88), formula right 74%
- `` — 4.9% of jets, value 0.29 (0.00…0.75), formula right 84%
- `` — 23.2% of jets, value 0.14 (0.00…0.50), formula right 80%
- `` — 6.2% of jets, value 0.06 (0.00…0.25), formula right 85%
- `` — 3.9% of jets, value 0.06 (0.00…0.25), formula right 92%
- `` — 31.5% of jets, value 0.01 (0.00…0.00), formula right 86%

```
z = -3.06
z += 0.0024 × sum_pt_top30
if girth2_top50 < 0.013: z += -245 × (0.013 − girth2_top50)
if girth2_top50 < 0.013 and z_dr_0p2_0p4 < 0.129: z += 1850 × (0.013 − girth2_top50) × (0.129 − z_dr_0p2_0p4)
if log_sum_pt < 7.02: z += 6.82 × (7.02 − log_sum_pt)
if girth < 0.123: z += 9.18 × (0.123 − girth)
if z_dr_0p1_0p2 < 0.177: z += 4.55 × (0.177 − z_dr_0p1_0p2)
if girth < 0.050: z += -38.80 × (0.050 − girth)
if girth2_top5 < 0.0071: z += 68.90 × (0.0071 − girth2_top5)
if z_dr_0p2_0p4 < 0.019: z += -30.70 × (0.019 − z_dr_0p2_0p4)
if lam2 < 0.0021: z += 184 × (0.0021 − lam2)
if mass_top10 > 26.60: z += -0.0081 × (mass_top10 − 26.60)
if sum_pt < 1000: z += -0.0075 × (1000 − sum_pt)
if girth2_top5 < 0.0018 and n_dr_0p2_0p4 > 2.31: z += -28.90 × (0.0018 − girth2_top5) × (n_dr_0p2_0p4 − 2.31)
if z_dr_0p1_0p2 < 0.127 and sum_pt_top3 < 758: z += -0.006 × (0.127 − z_dr_0p1_0p2) × (758 − sum_pt_top3)
if girth2_top5 < 0.006 and max_dr < 0.332: z += -1640 × (0.006 − girth2_top5) × (0.332 − max_dr)
if z_dr_0p1_0p2 < 0.117 and planar_flow < 0.818: z += 5.76 × (0.117 − z_dr_0p1_0p2) × (0.818 − planar_flow)
if girth2_top5 < 0.0073 and sum_pt_top40 < 975: z += 1.23 × (0.0073 − girth2_top5) × (975 − sum_pt_top40)
if girth > 0.086: z += 3.75 × (girth − 0.086)
if sum_pt_top50 > 1120: z += -0.0021 × (sum_pt_top50 − 1120)
if sum_pt < 1000 and tau32 < 0.664: z += -0.034 × (1000 − sum_pt) × (0.664 − tau32)
if z_dr_0p05_0p1 > 0.849: z += -6.44 × (z_dr_0p05_0p1 − 0.849)
if girth < 0.051 and z_dr_0p2_0p4 > 0.067: z += -7980 × (0.051 − girth) × (z_dr_0p2_0p4 − 0.067)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Top jets, wide** — 17.9% of jets, neuron 0.10, formula right for 81%. 69% top and 21% gluon; mass about 158 GeV, width 0.0233, pT shared out and mostly far from the axis. The linear term 'sum_pt_top30' (+2.313) and 'log_sum_pt < 7.02' (+0.561) are offset by 'mass_top10 > 26.6' (-0.454); the narrow-core tests mostly fail. The value, 0.095 (on for 18.3%), barely changes the scores. The formula calls them t and is right for 81.3%.
- **W-mass jets** — 16.1% of jets, neuron 0.14, formula right for 86%. 69% W and 12% Z; mass about 81 GeV, width 0.0059, and 41% of the pT at 0.025-0.05. 'sum_pt_top30' (+2.476) and 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+1.571) are cancelled by 'girth2_top50 < 0.0135' (-1.887) and 'z_dr_0p2_0p4 < 0.0195' (-0.408). The value, 0.138 (on for 37.2%), has almost no effect. The formula calls them W and is right for 86%.
- **Light narrow quark jets** — 15.6% of jets, neuron 0.15, formula right for 79%. 65% quark and 24% gluon; light (about 34 GeV), narrow (width 0.0011), a hard leading particle (337 GeV) and 89% of the pT within 0.025. 'girth2_top50 < 0.0135' (-3.054) and 'girth < 0.0498' (-1.374) are balanced by 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+2.682), 'sum_pt_top30' (+2.582) and 'girth < 0.123' (+0.997). The value is 0.154 (on for 55.2%), a very small effect. The formula calls them q and is right for 79.1%.
- **Z-mass jets** — 14.3% of jets, neuron 0.05, formula right for 87%. 75% Z and 10% top; mass about 92 GeV, width 0.008, and half the pT at 0.05-0.1. 'sum_pt_top30' (+2.405) and 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+1.074) against 'girth2_top50 < 0.0135' (-1.365); the value is 0.054 (on for 19.3%). The formula calls them Z and is right for 86.8%.
- **Boson-mass jets with hard core** — 12.0% of jets, neuron 0.40, formula right for 78%. 39% Z, 27% W and 19% gluon; mass about 88 GeV, width 0.0071, a harder leading particle (282 GeV) and 34% of the pT within 0.025. 'sum_pt_top30' (+2.473) with 'girth < 0.123' (+0.644), 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+0.644) and 'z_dr_0p1_0p2 < 0.177' (+0.615) outweigh 'girth2_top50 < 0.0135' (-1.602). The value, 0.403 (on for 73.8%), adds 0.226 to Z and removes 0.151 from top. The formula splits them between Z (39%) and W (31%) and is right for 78.4%.
- **Mid-mass gluon/quark jets** — 11.0% of jets, neuron 0.28, formula right for 72%. 51% gluon, 31% quark and 10% W; mass about 62 GeV, width 0.0032, total pT 1118 GeV and 64% of the pT within 0.025. 'sum_pt_top30' (+2.600) and 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+2.009) against 'girth2_top50 < 0.0135' (-2.572) and 'girth < 0.0498' (-0.668); the value, 0.277 (on for 61.9%), adds 0.156 to Z. The formula calls them g and is right for 72.4%.
- **Low-pT top jets** — 5.7% of jets, neuron 0.15, formula right for 78%. 76% top and 17% gluon; mass about 131 GeV, width 0.0223, a low total pT (898 GeV) and pT shared out. 'sum_pt_top30' (+1.952) and 'log_sum_pt < 7.02' (+1.513) against 'sum_pt < 1e+03' (-0.766) and 'sum_pt < 1e+03 and tau32 < 0.664' (-0.364); the value is 0.154 (on for 21.2%). The formula calls them t and is right for 78.1%.
- **Clean two-prong W jets** — 4.6% of jets, neuron 0.12, formula right for 93%. 71% W and 22% Z; mass about 83 GeV, width 0.0062, and 91% of the pT at 0.05-0.1 from the axis. 'sum_pt_top30' (+2.527) and 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+1.551) against 'girth2_top50 < 0.0135' (-1.791), with 'z_dr_0p05_0p1 > 0.849' (-0.464, 89% pass) marking the two-prong ring. The value is 0.119 (on for 42%). The formula calls them W and is right for 92.7%.
- **Low-pT gluon/quark jets** — 2.1% of jets, neuron 0.92, formula right for 67%. 46% gluon, 42% quark and 11% top; mass about 48 GeV, width 0.0033, a low total pT (872 GeV) and 63% of the pT within 0.025. 'sum_pt_top30' (+2.034), 'girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129' (+2.008), 'log_sum_pt < 7.02' (+1.716) and 'girth2_top5 < 0.00727 and sum_pt_top40 < 975' (+0.957) outweigh 'girth2_top50 < 0.0135' (-2.544) and 'sum_pt < 1e+03' (-0.965). The value, 0.916 (on for 96.3%), adds 0.516 to Z and removes 0.344 from top, a push towards Z for jets that are not Z. The formula splits them between g (52%) and q (46%) and is right for only 66.9%.
- **Very low-pT gluon/quark jets** — 0.7% of jets, neuron 1.92, formula right for 61%. 56% gluon, 32% quark and 12% top; mass about 55 GeV, width 0.0085, and a very low total pT (655 GeV). 'log_sum_pt < 7.02' (+3.729) and 'girth2_top5 < 0.00727 and sum_pt_top40 < 975' (+2.156) grow as the pT falls and outweigh 'sum_pt < 1e+03' (-2.598) and 'girth2_top50 < 0.0135' (-1.616). The value, 1.922, the neuron's highest, adds 1.081 to Z and removes 0.721 from top. The formula calls them g but is right for only 60.9%.
