# What each part of the 99-term formula does (64 particles)

*the simpler version of the simplified formula*. Validation accuracy 80.46%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet gets a value on 16 scales, and each class score adds some scales and subtracts others. Gluon and quark jets are split mainly by the particle-count scale (neuron 1), which the g score adds and the q score subtracts, while both light-jet scores subtract the 80-110 GeV boson-mass scale (4) and add the narrow one-prong scale (9). W and Z jets are marked by the absence of a busy, wide radiation pattern (neuron 8, subtracted strongly by both boson scores) and by the lower-pT, moderate-mass scale (5), which both add. W and Z are then split by mass: the light side of the W mass (0) and the clean two-prong scale (11) raise the W score (0 also lowers the Z score), the 91-130 GeV window (14) lowers the W score, and the Z score adds the 91-120 GeV window (7) and subtracts the one-prong scale that is lowest at 91-100 GeV (6). Top jets are recognised by hard particles spread far from the axis (neuron 10) and by the busy wide pattern (8), and the t score is pulled down strongly by the high-pT scale (13), on which top jets sit lowest.

## The 5 class scores

### score g: Many particles, outside the boson mass range

High for particle-rich one-prong jets outside the boson mass range: gluon jets score highest (mean 2.91 for g, AUC 0.93), quark jets next (0.78), top jets near zero (-0.03) and Z (-1.81) and W jets (-1.94) far below.

Adds the particle-count scale (neuron 1, +45%), the narrow one-prong scale (9, +10%) and a little of 12 and 6 (+3% each); subtracts the 80-110 GeV boson-mass scale (4, -20%), the lower-pT moderate-mass scale (5, -8%) and the slim sparse-jet scale (3, -8%).

*computed:* largest for g (2.91), then q (0.78), then t (-0.03), then Z (-1.81), then W (-1.94); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Narrow, light one-prong jet, few particles

High for narrow, light one-prong jets without many particles: quark jets score highest (mean 2.58 for q, AUC 0.89), gluon jets next (1.16), top jets near zero (0.18), W (-0.57) and Z jets (-0.61) below.

Subtracts the 80-110 GeV boson-mass scale (neuron 4, -38%), the particle-count scale (1, -21%) and the clean two-prong scale (11, -7%); adds the narrow one-prong scale (9, +17%), the light below-80 GeV scale (12, +7%) and the one-prong scale (6, +7%).

*computed:* largest for q (2.58), then g (1.16), then t (0.18), then W (-0.57), then Z (-0.61); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Compact jet at the W mass

High for compact two-prong jets at or just below the W mass with a quiet outer ring: W jets score highest (mean 2.95 for W, AUC 0.96), quark jets near zero (-0.08), Z (-0.87) and gluon jets (-1.96) below, top jets far below (-5.15).

Subtracts the busy wide-radiation scale (neuron 8, -35%), the 91-130 GeV window (14, -12%) and small amounts of 7, 12 and 9 (-5%, -4%, -3%); adds the lower-pT moderate-mass scale (5, +12%), the clean two-prong scale (11, +8%), the light side of the W mass (0, +8%), the 80-110 GeV boson-mass scale (4, +6%) and the slim sparse-jet scale (3, +4%).

*computed:* largest for W (2.95), then q (-0.08), then Z (-0.87), then g (-1.96), then t (-5.15); it separates W jets from the rest best (AUC 0.96: large for W)

### score Z: Compact jet in the Z mass window

High for compact jets at or just above 91 GeV with a quiet outer ring: Z jets score highest (mean 2.84 for Z, AUC 0.95), quark jets near zero (-0.03), W jets below (-0.60), gluon (-1.49) and top jets far below (-5.42).

Subtracts the busy wide-radiation scale (neuron 8, -37%), the one-prong scale that is lowest at 91-100 GeV (6, -15%), the light side of the W mass (0, -14%) and a little of 2 (-3%); adds the lower-pT moderate-mass scale (5, +14%), the 91-120 GeV window (7, +8%) and small amounts of 3, 12 and 15 (+3% each).

*computed:* largest for Z (2.84), then q (-0.03), then W (-0.60), then g (-1.49), then t (-5.42); it separates Z jets from the rest best (AUC 0.95: large for Z)

### score t: Widely spread hard particles in heavy jet

High for jets whose hardest particles are spread far from the axis and that sit low on the high-pT, below-140 GeV scale: top jets score highest (mean 2.96 for t, AUC 0.95), W (-0.37) and gluon jets (-0.43) slightly below zero, Z (-0.82) and quark jets (-1.08) further below.

Subtracts the high-pT scale (neuron 13, -38%), the lower-pT moderate-mass scale (5, -11%) and small amounts of 12, 7 and 15 (-4%, -3%, -2%); adds the spread-out hard-particle scale (10, +28%), the busy wide-radiation scale (8, +9%) and a little of the 80-110 GeV boson-mass scale (4, +4%).

*computed:* largest for t (2.96), then W (-0.37), then g (-0.43), then Z (-0.82), then q (-1.08); it separates t jets from the rest best (AUC 0.95: large for t)

## The 16 neurons (most important first)

### neuron 1: Particle count: gluon-likeness (major)

- **What it measures:** Grows with the number of particles and with how thinly the pT is shared among them; it is pushed down when the 50 hardest particles carry more than 0.96 of the pT, and pushed up when the 3 hardest carry less than 760 GeV or τ32 is above 0.28. Gluon jets sit far highest (5.77 for g, AUC 0.92), then top (2.83) and quark jets (2.37), with Z (1.72) and W jets lowest (1.59 for W).
- *computed — its value:* largest for g (5.77), then t (2.83), then q (2.37), then Z (1.72), then W (1.59); it separates g jets from the rest best (AUC 0.92: large for g)
- **How the class scores use it:** It raises the g score (+45%) and lowers the q score (-21%), making it the main gluon-versus-quark handle; freezing it costs 11.916 accuracy points. The W, Z and t scores hardly use it, even though top jets sit fairly high on it.
- *computed — used by:* raises the score of g (+45%); lowers the score of q (-21%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.7):

- `` — 4.0% of jets, value 7.84 (6.19…9.59), formula right 94%
- `` — 7.4% of jets, value 6.16 (4.06…8.19), formula right 79%
- `` — 5.2% of jets, value 5.20 (3.53…6.88), formula right 68%
- `` — 3.3% of jets, value 4.72 (2.81…6.66), formula right 70%
- `` — 20.4% of jets, value 3.32 (1.62…5.19), formula right 77%
- `` — 3.7% of jets, value 3.00 (1.78…4.28), formula right 83%
- `` — 27.3% of jets, value 2.18 (0.78…3.78), formula right 79%
- `` — 28.7% of jets, value 0.99 (0.00…2.28), formula right 85%

```
z = 1.97
if log_sum_pt > 6.90: z += 37.30 × (log_sum_pt − 6.90)
if z_top50_slots > 0.960: z += -49.50 × (z_top50_slots − 0.960)
if sum_pt_top3 < 760: z += 0.0051 × (760 − sum_pt_top3)
if log_sum_pt > 6.80: z += -10.00 × (log_sum_pt − 6.80)
if tau32 > 0.280: z += 2.12 × (tau32 − 0.280)
if z_top20_slots < 0.960: z += -6.69 × (0.960 − z_top20_slots)
if n_particles > 38.00 and dr_0 < 0.110: z += 0.733 × (n_particles − 38.00) × (0.110 − dr_0)
if n_particles < 43.00: z += -0.076 × (43.00 − n_particles)
if log_sum_pt > 7.00: z += -18.90 × (log_sum_pt − 7.00)
if sum_pt_top2 < 820 and n_dr_0p2_0p4 < 7.70: z += -0.0003 × (820 − sum_pt_top2) × (7.70 − n_dr_0p2_0p4)
if girth2_top20 < 0.0021: z += 500 × (0.0021 − girth2_top20)
if mass_top30 < 73.00: z += 0.014 × (73.00 − mass_top30)
if n_particles > 40.00 and z_top50_slots < 0.980: z += 1.62 × (n_particles − 40.00) × (0.980 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 20.8% of jets, neuron 2.27, formula right for 74%.  
- **group 2** — 19.8% of jets, neuron 2.14, formula right for 85%.  
- **group 3** — 18.4% of jets, neuron 0.76, formula right for 83%.  
- **group 4** — 10.4% of jets, neuron 2.20, formula right for 80%.  
- **group 5** — 10.3% of jets, neuron 4.11, formula right for 75%.  
- **group 6** — 7.0% of jets, neuron 5.21, formula right for 79%.  
- **group 7** — 6.6% of jets, neuron 5.19, formula right for 79%.  
- **group 8** — 4.1% of jets, neuron 6.34, formula right for 84%.  
- **group 9** — 2.0% of jets, neuron 7.35, formula right for 87%.  
- **group 10** — 0.7% of jets, neuron 8.82, formula right for 90%.  

### neuron 4: Boson mass window, 80-110 GeV (major)

- **What it measures:** Masses below 110 GeV get an upward push and masses below 80.4 GeV a downward one, so jets of 80.4-110 GeV sit highest; a two-prong pattern (D2 below 7.7) pushes it up, while total pT below 1100 GeV pushes it down, and it grows with the mass of the 10-20 hardest particles. Z (1.51) and W jets (1.50) sit highest, top jets next (1.05), gluon jets low (0.33) and quark jets lowest (0.16 for q; AUC 0.16: small for q).
- *computed — its value:* largest for Z (1.51), then W (1.50), then t (1.05), then g (0.33), then q (0.16); it separates q jets from the rest best (AUC 0.16: small for q)
- **How the class scores use it:** It lowers the q (-38%) and g (-20%) scores strongly, since a jet in the boson mass range is not a light-QCD jet, and raises the W (+6%) and t (+4%) scores a little. The Z score hardly uses it, even though Z jets sit highest on it.
- *computed — used by:* raises the score of W (+6%), t (+4%); lowers the score of g (-20%), q (-38%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.719):

- `` — 10.7% of jets, value 2.19 (1.69…2.69), formula right 96%
- `` — 8.3% of jets, value 1.70 (1.19…2.19), formula right 92%
- `` — 2.0% of jets, value 1.49 (0.50…2.25), formula right 88%
- `` — 45.5% of jets, value 1.07 (0.25…1.75), formula right 80%
- `` — 3.5% of jets, value 0.54 (0.00…1.12), formula right 73%
- `` — 2.0% of jets, value 0.19 (0.00…0.62), formula right 64%
- `` — 28.1% of jets, value 0.00 (0.00…0.00), formula right 73%

```
z = -0.532
if mass < 80.40: z += -0.139 × (80.40 − mass)
if mass < 110: z += 0.044 × (110 − mass)
if D2 < 7.70: z += 0.209 × (7.70 − D2)
if mass_top40 < 85.00 and D2 < 8.20: z += -0.012 × (85.00 − mass_top40) × (8.20 − D2)
if sum_pt < 1100: z += -0.0043 × (1100 − sum_pt)
if mass < 120 and max_dr < 0.400: z += 0.118 × (120 − mass) × (0.400 − max_dr)
if tau32 < 0.610: z += 2.89 × (0.610 − tau32)
if C2 > 0.100: z += 16.90 × (C2 − 0.100)
if mass < 75.00 and max_dr < 0.330: z += -0.627 × (75.00 − mass) × (0.330 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 26.8% of jets, neuron 1.56, formula right for 86%.  
- **group 2** — 19.2% of jets, neuron 1.34, formula right for 84%.  
- **group 3** — 15.1% of jets, neuron 0.55, formula right for 73%.  
- **group 4** — 10.6% of jets, neuron 1.47, formula right for 89%.  
- **group 5** — 6.7% of jets, neuron 0.01, formula right for 65%.  
- **group 6** — 5.9% of jets, neuron 0.00, formula right for 74%.  
- **group 7** — 5.8% of jets, neuron 0.00, formula right for 82%.  
- **group 8** — 4.8% of jets, neuron 0.00, formula right for 75%.  
- **group 9** — 4.1% of jets, neuron 0.00, formula right for 68%.  
- **group 10** — 1.0% of jets, neuron 0.00, formula right for 86%.  

### neuron 5: Lower-pT jet with moderate mass (major)

- **What it measures:** Rises when the 30 hardest particles have mass above 49 GeV, when e2 is small (below 0.038) and when a particle lies beyond ΔR 0.44; it falls with total pT (total pT above roughly 1000 GeV pushes it down) and is pushed down when the 50 hardest particles are heavier than 150 GeV, when the 30 hardest carry more than 0.93 of the pT, or when (mass/pT)² is above 0.029. Quark jets sit highest (1.55 for q, AUC 0.75), Z (1.06) and W jets (1.01) next, top jets lower (0.90) and gluon jets lowest (0.69 for g).
- *computed — its value:* largest for q (1.55), then Z (1.06), then W (1.01), then t (0.90), then g (0.69); it separates q jets from the rest best (AUC 0.75: large for q)
- **How the class scores use it:** It raises the W (+12%) and Z (+14%) scores and lowers the g (-8%) and t (-11%) scores: a moderately massive jet that is neither very heavy nor very high in pT looks like a boson rather than a gluon or a top. The q score hardly uses it, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+12%), Z (+14%); lowers the score of g (-8%), t (-11%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.539):

- `` — 8.6% of jets, value 2.05 (1.38…2.75), formula right 81%
- `` — 66.6% of jets, value 1.20 (0.62…1.81), formula right 78%
- `` — 2.0% of jets, value 0.78 (0.00…1.62), formula right 74%
- `` — 5.1% of jets, value 0.45 (0.00…1.00), formula right 80%
- `` — 2.0% of jets, value 0.37 (0.00…0.81), formula right 93%
- `` — 3.8% of jets, value 0.24 (0.00…0.69), formula right 81%
- `` — 6.0% of jets, value 0.06 (0.00…0.06), formula right 86%
- `` — 5.9% of jets, value 0.04 (0.00…0.00), formula right 92%

```
z = 0.871
if mass_top30 > 49.00: z += 0.016 × (mass_top30 − 49.00)
if log_sum_pt > 6.90: z += -7.36 × (log_sum_pt − 6.90)
if n_particles < 66.00 and e2 < 0.038: z += 1.57 × (66.00 − n_particles) × (0.038 − e2)
if mass_top50 > 150: z += -0.144 × (mass_top50 − 150)
if z_top30_slots > 0.930: z += -9.11 × (z_top30_slots − 0.930)
if mass_over_sum_pt_sq > 0.029: z += -1240 × (mass_over_sum_pt_sq − 0.029)
if max_dr > 0.440: z += 19.80 × (max_dr − 0.440)
if girth2_top30 < 0.020 and tau32 < 0.860: z += 96.10 × (0.020 − girth2_top30) × (0.860 − tau32)
if mass_top50 > 160 and z_dr_0p05_0p1 > 0.550: z += 0.750 × (mass_top50 − 160) × (z_dr_0p05_0p1 − 0.550)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 62.1% of jets, neuron 1.10, formula right for 80%.  
- **group 2** — 16.7% of jets, neuron 1.74, formula right for 74%.  
- **group 3** — 8.9% of jets, neuron 0.10, formula right for 86%.  
- **group 4** — 7.1% of jets, neuron 0.13, formula right for 92%.  
- **group 5** — 1.6% of jets, neuron 1.88, formula right for 78%.  
- **group 6** — 1.6% of jets, neuron 0.00, formula right for 87%.  
- **group 7** — 0.7% of jets, neuron 0.00, formula right for 73%.  
- **group 8** — 0.5% of jets, neuron 1.94, formula right for 76%.  
- **group 9** — 0.4% of jets, neuron 0.02, formula right for 84%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 67%.  

### neuron 8: Busy, wide radiation, massive jet (major)

- **What it measures:** Grows with the number and pT share of particles at 0.2 <= ΔR < 0.4, with the jet's spread in its narrower direction and with the particle count; mass above 71 GeV and a large mass/pT (above 0.085, with total pT under 1100 GeV) push it up, while mass above 120 GeV pushes it down. Top jets sit far highest (6.57 for t, AUC 0.91), gluon jets next (2.11), quark jets lower (0.97), Z (0.36) and W jets near zero (0.10 for W).
- *computed — its value:* largest for t (6.57), then g (2.11), then q (0.97), then Z (0.36), then W (0.10); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W (-35%) and Z (-37%) scores strongly, because a two-prong boson is compact with a quiet outer ring, and raises the t score (+9%). The g and q scores hardly use it.
- *computed — used by:* raises the score of t (+9%); lowers the score of W (-35%), Z (-37%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.809):

- `` — 12.3% of jets, value 9.13 (5.12…13.50), formula right 86%
- `` — 3.6% of jets, value 6.28 (3.50…10.01), formula right 68%
- `` — 2.6% of jets, value 5.14 (3.38…6.75), formula right 79%
- `` — 2.0% of jets, value 4.88 (0.75…9.94), formula right 64%
- `` — 4.3% of jets, value 2.84 (2.00…3.50), formula right 74%
- `` — 3.6% of jets, value 2.55 (1.12…4.00), formula right 65%
- `` — 71.6% of jets, value 0.32 (0.00…1.06), formula right 81%

```
z = -0.035
if mass > 71.00: z += 0.049 × (mass − 71.00)
if mass_over_sum_pt > 0.085 and sum_pt < 1100: z += 0.758 × (mass_over_sum_pt − 0.085) × (1100 − sum_pt)
if n_dr_0p2_0p4 < 19.00: z += -0.058 × (19.00 − n_dr_0p2_0p4)
if girth < 0.096: z += 16.90 × (0.096 − girth)
if mass > 120: z += -0.047 × (mass − 120)
if lam2 < 0.0015: z += -508 × (0.0015 − lam2)
if sum_pt < 1000: z += 0.017 × (1000 − sum_pt)
if n_particles > 58.00: z += 0.110 × (n_particles − 58.00)
if sum_pt_top40 < 930 and log_sum_pt < 6.90: z += 0.056 × (930 − sum_pt_top40) × (6.90 − log_sum_pt)
if sum_pt < 1000 and n_real_top50 < 44.00: z += -0.0012 × (1000 − sum_pt) × (44.00 − n_real_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 42.1% of jets, neuron 0.58, formula right for 86%.  
- **group 2** — 33.6% of jets, neuron 0.28, formula right for 73%.  
- **group 3** — 6.4% of jets, neuron 8.34, formula right for 90%.  
- **group 4** — 5.3% of jets, neuron 6.15, formula right for 69%.  
- **group 5** — 4.9% of jets, neuron 3.29, formula right for 77%.  
- **group 6** — 4.2% of jets, neuron 11.81, formula right for 87%.  
- **group 7** — 1.7% of jets, neuron 4.01, formula right for 66%.  
- **group 8** — 1.3% of jets, neuron 4.90, formula right for 79%.  
- **group 9** — 0.3% of jets, neuron 7.59, formula right for 55%.  
- **group 10** — 0.2% of jets, neuron 7.63, formula right for 62%.  

### neuron 10: Hard particles spread from axis (major)

- **What it measures:** Grows when the hardest particles sit far from the jet axis (the 5 hardest spread out, little pT within ΔR < 0.05, large LHA); in mass it prefers 120-150 GeV, since masses between 80.4 and 120 GeV and above 150 GeV get a downward push, while a hardest-50 mass above 140 GeV pushes it up. Top jets sit highest (2.10 for t, AUC 0.82), Z (1.25), gluon (1.09) and W jets (1.06) in the middle, quark jets lowest (0.85 for q).
- *computed — its value:* largest for t (2.10), then Z (1.25), then g (1.09), then W (1.06), then q (0.85); it separates t jets from the rest best (AUC 0.82: large for t)
- **How the class scores use it:** It raises the t score (+28%): hard particles spread far from the axis, as in a three-prong top decay, are the main positive top signature, and freezing it costs 1.632 accuracy points. The g, q, W and Z scores hardly use it.
- *computed — used by:* raises the score of t (+28%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.638):

- `` — 20.6% of jets, value 2.28 (1.49…3.12), formula right 86%
- `` — 23.5% of jets, value 1.65 (1.06…2.19), formula right 87%
- `` — 16.7% of jets, value 1.33 (0.56…2.19), formula right 74%
- `` — 11.3% of jets, value 0.94 (0.19…1.69), formula right 75%
- `` — 2.3% of jets, value 0.55 (0.00…1.19), formula right 76%
- `` — 2.0% of jets, value 0.54 (0.00…1.56), formula right 77%
- `` — 13.2% of jets, value 0.42 (0.00…0.94), formula right 79%
- `` — 10.4% of jets, value 0.09 (0.00…0.38), formula right 73%

```
z = 3.08
if girth2_top5 < 0.021: z += -104 × (0.021 − girth2_top5)
if mass < 120: z += -0.033 × (120 − mass)
if mass < 80.40: z += 0.092 × (80.40 − mass)
if z_dr_0p2_0p4 < 0.088: z += 15.20 × (0.088 − z_dr_0p2_0p4)
if z_dr_0_0p05 > 0.770: z += -13.20 × (z_dr_0_0p05 − 0.770)
if n_dr_0p2_0p4 < 11.00: z += -0.144 × (11.00 − n_dr_0p2_0p4)
if mass > 150: z += -0.159 × (mass − 150)
if girth2_top5 < 0.021 and sum_pt_top3 < 720: z += 0.113 × (0.021 − girth2_top5) × (720 − sum_pt_top3)
if mass_top50 > 140: z += 0.099 × (mass_top50 − 140)
if mass < 120 and tau21 < 0.450: z += 0.092 × (120 − mass) × (0.450 − tau21)
if sum_pt_top50 < 960: z += -0.0081 × (960 − sum_pt_top50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 32.8% of jets, neuron 1.48, formula right for 89%.  
- **group 2** — 17.3% of jets, neuron 1.88, formula right for 71%.  
- **group 3** — 10.8% of jets, neuron 0.11, formula right for 75%.  
- **group 4** — 10.8% of jets, neuron 0.56, formula right for 79%.  
- **group 5** — 9.8% of jets, neuron 0.63, formula right for 72%.  
- **group 6** — 6.0% of jets, neuron 1.55, formula right for 66%.  
- **group 7** — 6.0% of jets, neuron 1.94, formula right for 92%.  
- **group 8** — 5.1% of jets, neuron 2.33, formula right for 88%.  
- **group 9** — 1.1% of jets, neuron 0.29, formula right for 74%.  
- **group 10** — 0.2% of jets, neuron 0.00, formula right for 79%.  

### neuron 13: High total pT, mass below 140 (major)

- **What it measures:** Grows with the total jet pT: it is pushed down when the total pT is below roughly 1000-1100 GeV and when the mass is above 140 GeV. Gluon (2.34), Z (2.26), W (2.22) and quark jets (1.86) all sit high, top jets far lowest (0.97 for t; AUC 0.16: small for t).
- *computed — its value:* largest for g (2.34), then Z (2.26), then W (2.22), then q (1.86), then t (0.97); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** It lowers the t score strongly (-38%): a jet with high total pT and mass below 140 GeV is unlikely to be a top; freezing it costs 1.188 accuracy points. The g, q, W and Z scores hardly use it, even though those types all sit high on it.
- *computed — used by:* lowers the score of t (-38%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.862):

- `` — 26.5% of jets, value 2.95 (2.56…3.09), formula right 81%
- `` — 43.7% of jets, value 2.09 (1.72…2.44), formula right 82%
- `` — 3.0% of jets, value 1.38 (0.78…2.06), formula right 78%
- `` — 9.4% of jets, value 1.33 (0.75…2.00), formula right 71%
- `` — 5.6% of jets, value 0.69 (0.16…1.31), formula right 92%
- `` — 2.7% of jets, value 0.48 (0.00…1.19), formula right 76%
- `` — 2.5% of jets, value 0.33 (0.00…0.94), formula right 93%
- `` — 6.7% of jets, value 0.11 (0.00…0.44), formula right 68%

```
z = 3.10
if log_sum_pt < 7.00: z += -15.10 × (7.00 − log_sum_pt)
if sum_pt_top40 < 1000: z += 0.015 × (1000 − sum_pt_top40)
if sum_pt < 1000: z += -0.024 × (1000 − sum_pt)
if mass > 140: z += -0.077 × (mass − 140)
if log_sum_pt < 6.80: z += 22.80 × (6.80 − log_sum_pt)
if mass_top10 > 67.00: z += 0.018 × (mass_top10 − 67.00)
if mass > 173: z += 0.084 × (mass − 173)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 44.2% of jets, neuron 2.05, formula right for 82%.  
- **group 2** — 26.9% of jets, neuron 2.94, formula right for 81%.  
- **group 3** — 9.5% of jets, neuron 1.33, formula right for 69%.  
- **group 4** — 7.4% of jets, neuron 0.81, formula right for 89%.  
- **group 5** — 4.7% of jets, neuron 0.31, formula right for 71%.  
- **group 6** — 3.7% of jets, neuron 0.44, formula right for 91%.  
- **group 7** — 1.7% of jets, neuron 0.07, formula right for 66%.  
- **group 8** — 0.9% of jets, neuron 1.59, formula right for 75%.  
- **group 9** — 0.8% of jets, neuron 0.06, formula right for 58%.  
- **group 10** — 0.2% of jets, neuron 0.20, formula right for 56%.  

### neuron 0: Light side of the W mass (moderate)

- **What it measures:** It is pushed down for mass above 80.4 GeV and partly pushed back up above 91.2 GeV, so a mass between 80.4 and 91.2 GeV gets only the downward push; a small (mass/pT)² (below 0.0079) pushes it up and very narrow jets (girth below 0.057) push it down, and overall it falls as mass and width grow. W jets sit far highest (1.66 for W, AUC 0.95), then quark (0.37) and gluon jets (0.30), with Z (0.13) and top jets lowest (0.09 for t).
- *computed — its value:* largest for W (1.66), then q (0.37), then g (0.30), then Z (0.13), then t (0.09); it separates W jets from the rest best (AUC 0.95: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the Z score (-14%): a jet on the light side of the W mass is evidence for a W rather than a Z. The g, q and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of Z (-14%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.849):

- `` — 21.7% of jets, value 1.74 (1.06…2.22), formula right 84%
- `` — 10.0% of jets, value 0.71 (0.00…1.22), formula right 70%
- `` — 2.2% of jets, value 0.41 (0.00…1.19), formula right 56%
- `` — 2.0% of jets, value 0.32 (0.00…0.84), formula right 67%
- `` — 18.6% of jets, value 0.24 (0.00…0.53), formula right 77%
- `` — 3.3% of jets, value 0.08 (0.00…0.31), formula right 88%
- `` — 42.2% of jets, value 0.00 (0.00…0.00), formula right 84%

```
z = 1.32
if mass > 80.40: z += -0.200 × (mass − 80.40)
if mass > 91.20: z += 0.204 × (mass − 91.20)
if mass_over_sum_pt_sq < 0.0079: z += 532 × (0.0079 − mass_over_sum_pt_sq)
if girth < 0.057: z += -45.80 × (0.057 − girth)
if lam1 < 0.0058: z += -316 × (0.0058 − lam1)
if girth2_top20 < 0.0062: z += -195 × (0.0062 − girth2_top20)
if sum_pt < 1000 and z_dr_0p2_0p4 < 0.210: z += -0.052 × (1000 − sum_pt) × (0.210 − z_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.1% of jets, neuron 1.46, formula right for 79%.  
- **group 2** — 25.3% of jets, neuron 0.37, formula right for 76%.  
- **group 3** — 21.3% of jets, neuron 0.03, formula right for 88%.  
- **group 4** — 6.8% of jets, neuron 0.00, formula right for 92%.  
- **group 5** — 5.0% of jets, neuron 0.00, formula right for 86%.  
- **group 6** — 4.3% of jets, neuron 0.00, formula right for 68%.  
- **group 7** — 3.8% of jets, neuron 0.00, formula right for 72%.  
- **group 8** — 3.8% of jets, neuron 0.00, formula right for 73%.  
- **group 9** — 1.3% of jets, neuron 0.00, formula right for 75%.  
- **group 10** — 0.3% of jets, neuron 0.01, formula right for 79%.  

### neuron 3: Slim, sparse, low mass-to-pT jet (moderate)

- **What it measures:** Rises for jets with a small (mass/pT)² (below 0.008), a thin spread in the jet's narrower direction, and few particles (under 45) while the 40 hardest carry over 830 GeV; jets whose 5 hardest particles sit almost on the axis are pushed down. Quark jets sit highest (0.76 for q), then W (0.60) and Z jets (0.39), gluon jets low (0.24) and top jets almost at zero (0.03 for t; AUC 0.18: small for t).
- *computed — its value:* largest for q (0.76), then W (0.60), then Z (0.39), then g (0.24), then t (0.03); it separates t jets from the rest best (AUC 0.18: small for t)
- **How the class scores use it:** It raises the W (+4%) and Z (+3%) scores and lowers the g score (-8%): a slim, sparse jet is more boson-like than gluon-like. The q and t scores hardly use it, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+4%), Z (+3%); lowers the score of g (-8%); does not (or hardly) enter the score of q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.847):

- `` — 5.7% of jets, value 1.56 (1.16…2.03), formula right 83%
- `` — 8.5% of jets, value 1.09 (0.69…1.50), formula right 77%
- `` — 7.0% of jets, value 1.03 (0.72…1.38), formula right 96%
- `` — 4.2% of jets, value 0.70 (0.38…1.06), formula right 73%
- `` — 9.0% of jets, value 0.59 (0.28…0.91), formula right 93%
- `` — 10.7% of jets, value 0.36 (0.06…0.59), formula right 74%
- `` — 6.0% of jets, value 0.27 (0.00…0.59), formula right 82%
- `` — 48.9% of jets, value 0.03 (0.00…0.09), formula right 77%

```
z = -0.287
if mass_over_sum_pt_sq < 0.008: z += 126 × (0.008 − mass_over_sum_pt_sq)
if lam2 < 0.00052: z += 2090 × (0.00052 − lam2)
if n_particles < 45.00 and sum_pt_top40 > 830: z += 0.00013 × (45.00 − n_particles) × (sum_pt_top40 − 830)
if girth2_top5 < 0.00011: z += -7010 × (0.00011 − girth2_top5)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 36.0% of jets, neuron 0.00, formula right for 78%.  
- **group 2** — 13.0% of jets, neuron 0.08, formula right for 74%.  
- **group 3** — 10.3% of jets, neuron 0.45, formula right for 74%.  
- **group 4** — 10.0% of jets, neuron 0.52, formula right for 92%.  
- **group 5** — 9.0% of jets, neuron 1.10, formula right for 95%.  
- **group 6** — 6.0% of jets, neuron 1.27, formula right for 81%.  
- **group 7** — 5.2% of jets, neuron 0.36, formula right for 84%.  
- **group 8** — 4.7% of jets, neuron 1.26, formula right for 74%.  
- **group 9** — 4.5% of jets, neuron 0.35, formula right for 76%.  
- **group 10** — 1.1% of jets, neuron 2.06, formula right for 76%.  

### neuron 6: One-prong, away from 91-100 GeV (moderate)

- **What it measures:** Built from two mass steps only: masses between 91.2 and 100 GeV get the downward push without the upward one given below 91.2 GeV, so jets in that window sit lowest; it also moves with one-prong-ness (larger τ21 and D2, a rounder pT pattern). Quark jets sit highest (1.46 for q), then gluon (0.95) and top jets (0.89), with W (0.39) and Z jets lowest (0.19 for Z; AUC 0.11: small for Z).
- *computed — its value:* largest for q (1.46), then g (0.95), then t (0.89), then W (0.39), then Z (0.19); it separates Z jets from the rest best (AUC 0.11: small for Z)
- **How the class scores use it:** It raises the q (+7%) and g (+3%) scores and lowers the Z score (-15%): a high value means a one-prong jet outside the 91-100 GeV window, which is evidence against a Z. The W and t scores hardly use it.
- *computed — used by:* raises the score of g (+3%), q (+7%); lowers the score of Z (-15%); does not (or hardly) enter the score of W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.96):

- `` — 4.4% of jets, value 2.27 (2.12…2.44), formula right 84%
- `` — 5.4% of jets, value 1.94 (1.81…2.06), formula right 77%
- `` — 5.1% of jets, value 1.63 (1.50…1.75), formula right 72%
- `` — 5.1% of jets, value 1.31 (1.19…1.44), formula right 72%
- `` — 24.1% of jets, value 1.00 (1.00…1.00), formula right 80%
- `` — 2.0% of jets, value 0.91 (0.74…1.00), formula right 66%
- `` — 8.6% of jets, value 0.85 (0.56…1.12), formula right 65%
- `` — 45.3% of jets, value 0.19 (0.00…0.38), formula right 86%

```
z = 0.985
if mass < 100: z += -0.123 × (100 − mass)
if mass < 91.20: z += 0.157 × (91.20 − mass)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.1% of jets, neuron 0.98, formula right for 79%.  
- **group 2** — 17.8% of jets, neuron 0.30, formula right for 88%.  
- **group 3** — 17.5% of jets, neuron 0.04, formula right for 91%.  
- **group 4** — 7.2% of jets, neuron 0.12, formula right for 72%.  
- **group 5** — 5.8% of jets, neuron 1.30, formula right for 72%.  
- **group 6** — 5.6% of jets, neuron 1.98, formula right for 77%.  
- **group 7** — 5.6% of jets, neuron 1.65, formula right for 72%.  
- **group 8** — 4.9% of jets, neuron 0.95, formula right for 69%.  
- **group 9** — 4.9% of jets, neuron 0.59, formula right for 63%.  
- **group 10** — 3.6% of jets, neuron 2.30, formula right for 85%.  

### neuron 7: Z mass window, 91-120 GeV (moderate)

- **What it measures:** Switches on mostly for masses between 91.2 and 120 GeV, most strongly below 100 GeV in a compact jet (no particle beyond ΔR 0.4) or with a two-prong pattern (D2 below 1.6); few particles at 0.2 <= ΔR < 0.4 and small e2 push it up a little more. Almost only Z jets sit high (1.77 for Z, AUC 0.91); W (0.14), gluon (0.08), top (0.07) and quark jets (0.04) stay near zero.
- *computed — its value:* largest for Z (1.77), then W (0.14), then g (0.08), then t (0.07), then q (0.04); it separates Z jets from the rest best (AUC 0.91: large for Z)
- **How the class scores use it:** It raises the Z score (+8%) and lowers the W (-5%) and t (-3%) scores: a compact two-prong jet just above 91 GeV points to a Z. The g and q scores hardly use it.
- *computed — used by:* raises the score of Z (+8%); lowers the score of W (-5%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.764):

- `` — 10.1% of jets, value 2.56 (1.31…3.94), formula right 94%
- `` — 12.0% of jets, value 1.06 (0.12…2.19), formula right 80%
- `` — 2.1% of jets, value 0.49 (0.00…1.31), formula right 83%
- `` — 6.2% of jets, value 0.16 (0.00…0.56), formula right 82%
- `` — 8.9% of jets, value 0.08 (0.00…0.31), formula right 92%
- `` — 4.5% of jets, value 0.08 (0.00…0.25), formula right 74%
- `` — 21.7% of jets, value 0.03 (0.00…0.00), formula right 80%
- `` — 34.7% of jets, value 0.00 (0.00…0.00), formula right 73%

```
z = -0.612
if mass < 91.20: z += -0.190 × (91.20 − mass)
if mass < 120: z += 0.044 × (120 − mass)
if mass < 100 and max_dr < 0.400: z += 1.02 × (100 − mass) × (0.400 − max_dr)
if mass_top50 < 97.00 and D2 < 1.60: z += -0.589 × (97.00 − mass_top50) × (1.60 − D2)
if mass < 91.20 and max_dr < 0.390: z += -1.60 × (91.20 − mass) × (0.390 − max_dr)
if mass < 100 and D2 < 1.60: z += 0.459 × (100 − mass) × (1.60 − D2)
if n_dr_0p2_0p4 < 9.40: z += 0.123 × (9.40 − n_dr_0p2_0p4)
if e2 < 0.026: z += 45.30 × (0.026 − e2)
if z_top50_slots < 0.990: z += -35.50 × (0.990 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 38.8% of jets, neuron 0.44, formula right for 81%.  
- **group 2** — 12.7% of jets, neuron 0.08, formula right for 71%.  
- **group 3** — 12.5% of jets, neuron 1.78, formula right for 92%.  
- **group 4** — 9.3% of jets, neuron 0.00, formula right for 70%.  
- **group 5** — 8.2% of jets, neuron 0.00, formula right for 76%.  
- **group 6** — 7.8% of jets, neuron 0.14, formula right for 91%.  
- **group 7** — 4.6% of jets, neuron 0.00, formula right for 77%.  
- **group 8** — 3.9% of jets, neuron 0.21, formula right for 85%.  
- **group 9** — 1.8% of jets, neuron 0.00, formula right for 82%.  
- **group 10** — 0.6% of jets, neuron 0.00, formula right for 87%.  

### neuron 9: Narrow one-prong jet (moderate)

- **What it measures:** Mostly a narrowness scale: large when the 40 hardest particles sit close to the axis (pT-weighted mean ΔR² below 0.0056), with small extra pushes up for mass above 91.2 GeV and for lower pT, and a push down for (mass/pT)² above 0.026; it moves with one-prong-ness (τ21) and against e2 and mass. Quark jets sit highest (2.00 for q, AUC 0.83), gluon jets next (1.33), top (0.28), W (0.23) and Z jets (0.17) low.
- *computed — its value:* largest for q (2.00), then g (1.33), then t (0.28), then W (0.23), then Z (0.17); it separates q jets from the rest best (AUC 0.83: large for q)
- **How the class scores use it:** It raises the q (+17%) and g (+10%) scores and lowers the W score (-3%): a narrow, one-prong jet is a light-quark or gluon jet. The Z and t scores hardly use it.
- *computed — used by:* raises the score of g (+10%), q (+17%); lowers the score of W (-3%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.889):

- `` — 2.8% of jets, value 4.06 (1.75…6.44), formula right 66%
- `` — 15.4% of jets, value 2.65 (2.25…3.00), formula right 80%
- `` — 8.0% of jets, value 1.68 (1.06…2.06), formula right 66%
- `` — 2.0% of jets, value 1.59 (1.19…2.19), formula right 88%
- `` — 2.1% of jets, value 0.88 (0.62…1.06), formula right 74%
- `` — 4.4% of jets, value 0.49 (0.00…1.00), formula right 69%
- `` — 65.4% of jets, value 0.10 (0.00…0.38), formula right 83%

```
z = -0.155
if girth2_top40 < 0.0056: z += 584 × (0.0056 − girth2_top40)
if mass > 91.20: z += 0.0085 × (mass − 91.20)
if girth2_top40 < 0.0094 and sum_pt < 1000: z += 2.97 × (0.0094 − girth2_top40) × (1000 − sum_pt)
if mass_over_sum_pt_sq > 0.026: z += -159 × (mass_over_sum_pt_sq − 0.026)
if sum_pt_top40 < 920: z += 0.0042 × (920 − sum_pt_top40)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 50.5% of jets, neuron 0.07, formula right for 82%.  
- **group 2** — 17.3% of jets, neuron 2.54, formula right for 79%.  
- **group 3** — 12.7% of jets, neuron 0.38, formula right for 81%.  
- **group 4** — 9.4% of jets, neuron 1.36, formula right for 69%.  
- **group 5** — 5.2% of jets, neuron 0.05, formula right for 91%.  
- **group 6** — 1.8% of jets, neuron 3.86, formula right for 74%.  
- **group 7** — 1.0% of jets, neuron 0.00, formula right for 72%.  
- **group 8** — 1.0% of jets, neuron 2.34, formula right for 59%.  
- **group 9** — 0.8% of jets, neuron 5.59, formula right for 69%.  
- **group 10** — 0.2% of jets, neuron 2.85, formula right for 62%.  

### neuron 12: Light jet, below 80 GeV (moderate)

- **What it measures:** Mainly on for jets lighter than 80.4 GeV, but pushed down when C2 is above 0.055 or the 50 hardest particles are lighter than 63 GeV; overall it falls as mass and width grow. Quark jets sit highest (1.24 for q, AUC 0.79), gluon jets next (0.75), W (0.25), top (0.19) and Z jets (0.18) low.
- *computed — its value:* largest for q (1.24), then g (0.75), then W (0.25), then t (0.19), then Z (0.18); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** It raises the q (+7%), g (+3%) and Z (+3%) scores and lowers the W (-4%) and t (-4%) scores. These are small corrections: light jets are credited to the light-jet types, and the small plus for Z and minus for W and t fine-tune the balance among the heavy types.
- *computed — used by:* raises the score of g (+3%), q (+7%), Z (+3%); lowers the score of W (-4%), t (-4%) (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.763):

- `` — 11.3% of jets, value 1.89 (1.62…2.12), formula right 78%
- `` — 9.3% of jets, value 1.43 (1.00…1.75), formula right 73%
- `` — 2.0% of jets, value 1.37 (0.00…4.00), formula right 83%
- `` — 3.0% of jets, value 1.06 (0.75…1.38), formula right 70%
- `` — 4.2% of jets, value 0.57 (0.00…1.00), formula right 61%
- `` — 2.1% of jets, value 0.51 (0.10…0.88), formula right 68%
- `` — 68.2% of jets, value 0.12 (0.00…0.25), formula right 84%

```
z = 0.085
if mass < 80.40: z += 0.082 × (80.40 − mass)
if C2 > 0.055: z += -21.10 × (C2 − 0.055)
if mass_top50 < 63.00: z += -0.067 × (63.00 − mass_top50)
if mass_top30 > 130: z += 0.039 × (mass_top30 − 130)
if LHA > 0.410 and lam2 < 0.0044: z += 17700 × (LHA − 0.410) × (0.0044 − lam2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 45.4% of jets, neuron 0.14, formula right for 86%.  
- **group 2** — 12.7% of jets, neuron 0.01, formula right for 70%.  
- **group 3** — 6.9% of jets, neuron 0.00, formula right for 83%.  
- **group 4** — 6.5% of jets, neuron 1.62, formula right for 76%.  
- **group 5** — 6.1% of jets, neuron 1.49, formula right for 73%.  
- **group 6** — 5.6% of jets, neuron 0.87, formula right for 62%.  
- **group 7** — 5.5% of jets, neuron 1.25, formula right for 70%.  
- **group 8** — 5.3% of jets, neuron 0.25, formula right for 93%.  
- **group 9** — 4.7% of jets, neuron 1.96, formula right for 84%.  
- **group 10** — 1.1% of jets, neuron 2.93, formula right for 74%.  

### neuron 2: High-pT, high-C2 step (weak) (minor)

- **What it measures:** It varies little between jets: its only if-statement pushes it up when the 50 hardest particles carry more than 1100 GeV and C2 is above 0.094. Gluon jets sit slightly highest (0.50 for g), then top (0.42), quark (0.39), W and Z jets (0.38), so it hardly separates the types (AUC 0.53).
- *computed — its value:* largest for g (0.50), then t (0.42), then q (0.39), then W (0.38), then Z (0.38); it separates g jets from the rest best (AUC 0.53: large for g)
- **How the class scores use it:** It lowers the Z score slightly (-3%), a small correction; the g, q, W and t scores hardly use it.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.401):

- `` — 2.0% of jets, value 1.96 (0.38…4.75), formula right 79%
- `` — 2.0% of jets, value 0.63 (0.38…1.25), formula right 76%
- `` — 2.1% of jets, value 0.41 (0.38…0.50), formula right 78%
- `` — 84.8% of jets, value 0.38 (0.38…0.38), formula right 80%
- `` — 2.2% of jets, value 0.38 (0.38…0.38), formula right 82%
- `` — 7.0% of jets, value 0.38 (0.38…0.38), formula right 85%

```
z = 0.381
if sum_pt_top50 > 1100 and C2 > 0.094: z += 0.663 × (sum_pt_top50 − 1100) × (C2 − 0.094)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 98.4% of jets, neuron 0.38, formula right for 80%.  
- **group 2** — 0.8% of jets, neuron 1.31, formula right for 80%.  
- **group 3** — 0.4% of jets, neuron 2.58, formula right for 74%.  
- **group 4** — 0.2% of jets, neuron 4.10, formula right for 76%.  
- **group 5** — 0.1% of jets, neuron 6.06, formula right for 83%.  
- **group 6** — 0.1% of jets, neuron 3.07, formula right for 67%.  
- **group 7** — 0.0% of jets, neuron 4.73, formula right for 90%.  

### neuron 11: Clean two-prong, quiet outer ring (minor)

- **What it measures:** Large for jets with almost no pT at 0.2 <= ΔR < 0.4 (share below 0.0048) and a clean two-prong shape (D2 below 1.6 with mass under 100 GeV, or an elongated pT pattern, planar flow below 0.39, with mass under 120 GeV); two-prong jets lighter than 80.4 GeV are pushed down. W jets sit highest (1.46 for W, AUC 0.79), Z jets next (1.10 for Z), quark (0.36), top (0.33) and gluon jets (0.25) low.
- *computed — its value:* largest for W (1.46), then Z (1.10), then q (0.36), then t (0.33), then g (0.25); it separates W jets from the rest best (AUC 0.79: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the q score (-7%): a clean two-prong jet with a quiet outer ring is a boson, not a quark jet. The g, Z and t scores hardly use it.
- *computed — used by:* raises the score of W (+8%); lowers the score of q (-7%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.859):

- `` — 9.9% of jets, value 2.53 (1.88…3.19), formula right 97%
- `` — 3.6% of jets, value 1.86 (1.38…2.38), formula right 96%
- `` — 3.2% of jets, value 1.27 (0.81…1.75), formula right 95%
- `` — 13.3% of jets, value 1.03 (0.56…1.69), formula right 87%
- `` — 4.5% of jets, value 0.69 (0.00…1.25), formula right 80%
- `` — 3.1% of jets, value 0.61 (0.25…1.19), formula right 75%
- `` — 7.1% of jets, value 0.28 (0.25…0.38), formula right 75%
- `` — 55.3% of jets, value 0.25 (0.00…0.38), formula right 75%

```
z = 0.241
if z_dr_0p2_0p4 < 0.0048: z += 206 × (0.0048 − z_dr_0p2_0p4)
if mass < 100 and D2 < 1.60: z += 0.064 × (100 − mass) × (1.60 − D2)
if mass < 120 and planar_flow < 0.390: z += 0.070 × (120 − mass) × (0.390 − planar_flow)
if mass < 80.40 and D2 < 3.10: z += -0.048 × (80.40 − mass) × (3.10 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 58.1% of jets, neuron 0.27, formula right for 76%.  
- **group 2** — 9.1% of jets, neuron 0.83, formula right for 86%.  
- **group 3** — 8.0% of jets, neuron 2.08, formula right for 98%.  
- **group 4** — 7.5% of jets, neuron 1.03, formula right for 88%.  
- **group 5** — 5.4% of jets, neuron 2.86, formula right for 95%.  
- **group 6** — 4.9% of jets, neuron 1.39, formula right for 85%.  
- **group 7** — 3.1% of jets, neuron 0.02, formula right for 66%.  
- **group 8** — 2.2% of jets, neuron 0.01, formula right for 69%.  
- **group 9** — 1.1% of jets, neuron 0.00, formula right for 73%.  
- **group 10** — 0.5% of jets, neuron 0.17, formula right for 68%.  

### neuron 14: Mass window 91-130 GeV (minor)

- **What it measures:** Set by the mass: jets between 91.2 and 130 GeV get the upward push without the downward one given below 91.2 GeV. Z jets sit highest (1.06 for Z, AUC 0.87), then gluon (0.40) and top jets (0.35), with W (0.19) and quark jets lowest (0.13 for q).
- *computed — its value:* largest for Z (1.06), then g (0.40), then t (0.35), then W (0.19), then q (0.13); it separates Z jets from the rest best (AUC 0.87: large for Z)
- **How the class scores use it:** It lowers the W score (-12%): a jet heavier than the Z peak is not a W. The g, q, Z and t scores hardly use it, even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-12%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.842):

- `` — 2.1% of jets, value 1.90 (1.06…3.16), formula right 75%
- `` — 25.3% of jets, value 1.18 (0.84…1.34), formula right 85%
- `` — 6.1% of jets, value 0.62 (0.41…0.84), formula right 71%
- `` — 4.4% of jets, value 0.32 (0.25…0.38), formula right 87%
- `` — 4.3% of jets, value 0.19 (0.16…0.22), formula right 92%
- `` — 18.5% of jets, value 0.13 (0.06…0.34), formula right 83%
- `` — 3.5% of jets, value 0.10 (0.06…0.12), formula right 92%
- `` — 35.9% of jets, value 0.01 (0.00…0.00), formula right 74%

```
z = 0.061
if mass < 91.20: z += -0.138 × (91.20 − mass)
if mass < 130: z += 0.033 × (130 − mass)
if z_top50_slots < 0.960: z += 99.30 × (0.960 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.1% of jets, neuron 1.18, formula right for 85%.  
- **group 2** — 20.6% of jets, neuron 0.19, formula right for 82%.  
- **group 3** — 17.1% of jets, neuron 0.32, formula right for 84%.  
- **group 4** — 8.2% of jets, neuron 0.02, formula right for 77%.  
- **group 5** — 6.5% of jets, neuron 0.00, formula right for 75%.  
- **group 6** — 6.0% of jets, neuron 0.00, formula right for 73%.  
- **group 7** — 5.4% of jets, neuron 0.00, formula right for 70%.  
- **group 8** — 5.1% of jets, neuron 0.00, formula right for 84%.  
- **group 9** — 4.8% of jets, neuron 0.01, formula right for 64%.  
- **group 10** — 2.1% of jets, neuron 2.19, formula right for 75%.  

### neuron 15: Constant (no terms left) (minor)

- **What it measures:** It has no if-statements left, so it equals 0.25 for every jet; all five types sit at the same level (AUC 0.50).
- *computed — its value:* largest for g (0.25), then q (0.25), then W (0.25), then Z (0.25), then t (0.25); it separates g jets from the rest best (AUC 0.50: small for g)
- **How the class scores use it:** Being the same for every jet, it only shifts scores by fixed amounts: it raises the Z score (+3%) and lowers the t score (-2%). The g, q and W scores hardly use it.
- *computed — used by:* raises the score of Z (+3%); lowers the score of t (-2%); does not (or hardly) enter the score of g, q, W (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.265
h = max(0, z)
```
