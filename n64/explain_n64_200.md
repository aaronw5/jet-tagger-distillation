# What each part of the 200-term formula does (64 particles)

*the formula with the fewest quantities (31) at the network's accuracy*. Validation accuracy 81.23%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Each jet gets a value on 16 scales, and each class score adds some scales and subtracts others. Gluon and quark jets are separated mainly by gluon-likeness (many particles sharing the pT thinly, neuron 1), which the g score adds and the q score subtracts, while both light-jet scores subtract heavy two-prong-ness (4) and add the light one-prong scale (9). W and Z jets are recognised as clean two-prong jets with no wide radiation: both boson scores subtract the busy wide-radiation scale (8) strongly and add the few-particle pT-band scale (5). W and Z are then split by mass windows: W-side mass (0) raises the W score and lowers the Z score, the 91-120 GeV window (7) raises the Z score, and mass just above the Z (14) lowers the W score. Top jets are picked out by hard particles spread wide in a heavy jet (10) and by busy wide radiation (8), and the t score subtracts the high-pT, below-top-mass scale (13), on which every other jet type sits higher than top jets.

## The 5 class scores

### score g: Many particles, one prong, no heavy mass

High for jets with many particles sharing the pT thinly and no heavy two-prong structure: gluon jets score highest (mean 3.02, AUC 0.93), quark jets next (0.79), top jets near zero, Z and W jets far below (-2.25 and -2.36).

Adds gluon-likeness (neuron 1, +43%), light one-prong (9, +10%) and small amounts of 6 and 12 (+3% each); subtracts heavy two-prong-ness (4, -21%), the sparse-jet scale (3, -9%) and the few-particle pT-band scale (5, -8%).

*computed:* largest for g (3.02), then q (0.79), then t (0.02), then Z (-2.25), then W (-2.36); it separates g jets from the rest best (AUC 0.93: large for g)

### score q: Light one-prong jet, not particle-rich

High for light one-prong jets without the many soft particles of a gluon: quark jets score highest (mean 2.66, AUC 0.89), gluon jets next (1.14), top jets near zero (0.22), W and Z jets below zero.

Subtracts heavy two-prong-ness (neuron 4, -39%), gluon-likeness (1, -20%) and the clean two-prong scale (11, -7%); adds light one-prong (9, +17%), lighter-than-the-W (12, +7%) and one-prong outside the Z mass (6, +7%).

*computed:* largest for q (2.66), then g (1.14), then t (0.22), then W (-0.68), then Z (-0.76); it separates q jets from the rest best (AUC 0.89: large for q)

### score W: Clean two-prong jet at the W mass

High for clean, compact two-prong jets up to about the W mass with no wide radiation: W jets score highest (mean 3.27, AUC 0.97), quark jets slightly below zero, Z jets further below, gluon and especially top jets far below.

Subtracts busy wide radiation (neuron 8, -35%), mass just above the Z (14, -13%) and small amounts of 7, 12 and 9; adds the few-particle pT-band scale (5, +11%), the clean two-prong scale (11, +8%), W-side mass (0, +8%), heavy two-prong-ness (4, +6%) and the sparse-jet scale (3, +4%).

*computed:* largest for W (3.27), then q (-0.24), then Z (-0.96), then g (-2.13), then t (-5.46); it separates W jets from the rest best (AUC 0.97: large for W)

### score Z: Clean two-prong jet at the Z mass

High for clean two-prong jets at or just above the Z mass: Z jets score highest (mean 3.45, AUC 0.96), quark and W jets slightly below zero, gluon and especially top jets far below.

Subtracts busy wide radiation (neuron 8, -37%), one-prong outside the Z mass (6, -15%), W-side mass (0, -14%) and a little of 2 (-3%); adds the few-particle pT-band scale (5, +13%), the Z mass window (7, +9%) and small amounts of 3, 12 and 15.

*computed:* largest for Z (3.45), then q (-0.16), then W (-0.37), then g (-1.91), then t (-5.90); it separates Z jets from the rest best (AUC 0.96: large for Z)

### score t: Heavy jet with widely spread hard prongs

High for heavy jets with busy radiation and hard particles spread far from the axis: top jets score highest (mean 3.28, AUC 0.94), W and gluon jets slightly below zero, Z and quark jets further below.

Subtracts the high-pT, below-top-mass scale (neuron 13, -38%), the few-particle pT-band scale (5, -10%) and small amounts of 12 and 7; adds wide-spread hard particles (10, +27%), busy wide radiation (8, +10%) and a little heavy two-prong-ness (4, +4%).

*computed:* largest for t (3.28), then W (-0.28), then g (-0.42), then Z (-0.89), then q (-1.15); it separates t jets from the rest best (AUC 0.94: large for t)

## The 16 neurons (most important first)

### neuron 1: Gluon-likeness: many particles, thin pT (major)

- **What it measures:** Grows with the number of particles (pushed up above 26.7) and with the pT being spread thinly, so that the 3 hardest particles carry less than 761 GeV and the 50 hardest hold a smaller share; several total-pT thresholds also shape it. Gluon jets sit far highest (mean 5.85, AUC 0.93), top jets next (2.74), then quark jets (2.28), with Z and W jets lowest.
- *computed — its value:* largest for g (5.85), then t (2.74), then q (2.28), then Z (1.55), then W (1.41); it separates g jets from the rest best (AUC 0.93: large for g)
- **How the class scores use it:** It raises the g score (+43%) and lowers the q score (-20%), making it the main handle for telling gluons from quarks; freezing it costs 11.972 points of accuracy. It hardly enters the W, Z and t scores, even though top jets sit fairly high on it.
- *computed — used by:* raises the score of g (+43%); lowers the score of q (-20%); does not (or hardly) enter the score of W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.688):

- `` — 3.1% of jets, value 8.20 (6.56…9.91), formula right 94%
- `` — 7.5% of jets, value 6.33 (4.12…8.38), formula right 81%
- `` — 10.3% of jets, value 4.74 (2.91…6.69), formula right 72%
- `` — 3.1% of jets, value 4.74 (2.66…6.78), formula right 71%
- `` — 3.1% of jets, value 3.03 (1.66…4.47), formula right 83%
- `` — 15.8% of jets, value 3.02 (1.38…4.84), formula right 78%
- `` — 27.8% of jets, value 1.99 (0.41…3.78), formula right 80%
- `` — 29.2% of jets, value 0.95 (0.00…2.25), formula right 86%

```
z = 0.703
if log_sum_pt > 6.91: z += 32.80 × (log_sum_pt − 6.91)
if sum_pt_top3 < 761: z += 0.0045 × (761 − sum_pt_top3)
if log_sum_pt > 6.80: z += -8.65 × (log_sum_pt − 6.80)
if z_top50_slots > 0.958: z += -34.40 × (z_top50_slots − 0.958)
if log_sum_pt > 6.89: z += 17.60 × (log_sum_pt − 6.89)
if sum_pt_top50 > 962: z += -0.010 × (sum_pt_top50 − 962)
if tau32 > 0.326: z += 2.04 × (tau32 − 0.326)
if LHA < 0.398: z += 5.81 × (0.398 − LHA)
if n_particles > 26.70: z += 0.036 × (n_particles − 26.70)
if mass_top50 < 114: z += -0.019 × (114 − mass_top50)
if z_top30_slots < 0.997: z += -12.00 × (0.997 − z_top30_slots)
if log_sum_pt > 6.98: z += -20.00 × (log_sum_pt − 6.98)
if mass_top30 < 79.30: z += 0.033 × (79.30 − mass_top30)
if n_particles > 34.80 and dr_0 < 0.131: z += 0.358 × (n_particles − 34.80) × (0.131 − dr_0)
if n_dr_0p2_0p4 < 7.51: z += -0.100 × (7.51 − n_dr_0p2_0p4)
if max_dr < 0.429: z += -2.73 × (0.429 − max_dr)
if sum_pt_top40 > 1120: z += 0.012 × (sum_pt_top40 − 1120)
if n_particles > 38.80 and z_top50_slots < 0.985: z += 2.08 × (n_particles − 38.80) × (0.985 − z_top50_slots)
if lam2 < 0.00089: z += -591 × (0.00089 − lam2)
if sum_pt_top50 > 1150: z += -0.012 × (sum_pt_top50 − 1150)
if girth2_top20 < 0.0013: z += 1050 × (0.0013 − girth2_top20)
if D2 < 1.15: z += 1.34 × (1.15 − D2)
if z_dr_0_0p05 > 0.877: z += -6.58 × (z_dr_0_0p05 − 0.877)
if girth < 0.016: z += -92.70 × (0.016 − girth)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 23.7% of jets, neuron 0.81, formula right for 87%.  
- **group 2** — 20.2% of jets, neuron 2.41, formula right for 78%.  
- **group 3** — 15.6% of jets, neuron 2.78, formula right for 82%.  
- **group 4** — 15.1% of jets, neuron 1.97, formula right for 75%.  
- **group 5** — 8.6% of jets, neuron 4.44, formula right for 75%.  
- **group 6** — 8.1% of jets, neuron 4.91, formula right for 79%.  
- **group 7** — 4.6% of jets, neuron 6.02, formula right for 83%.  
- **group 8** — 2.6% of jets, neuron 7.02, formula right for 86%.  
- **group 9** — 1.2% of jets, neuron 7.86, formula right for 89%.  
- **group 10** — 0.3% of jets, neuron 8.44, formula right for 90%.  

### neuron 4: Heavy two-prong-ness (major)

- **What it measures:** Grows with the mass of the 10-20 hardest particles, with e2 and with small D2 (below 6.81, a two-prong pattern); masses between 80.4 and 103 GeV with a compact core are pushed up, and jets lighter than 80.4 GeV are pushed down. Z and W jets sit highest (1.64 and 1.57), top jets next (1.10), gluon jets low and quark jets lowest (AUC 0.16: small for q).
- *computed — its value:* largest for Z (1.64), then W (1.57), then t (1.10), then g (0.31), then q (0.16); it separates q jets from the rest best (AUC 0.16: small for q)
- **How the class scores use it:** It lowers the q score (-39%) and the g score (-21%) strongly, since a heavy pronged jet is not a light quark or gluon jet, and it raises the W score (+6%) and the t score (+4%) a little. It hardly enters the Z score, even though Z jets sit highest on it.
- *computed — used by:* raises the score of W (+6%), t (+4%); lowers the score of g (-21%), q (-39%); does not (or hardly) enter the score of Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.762):

- `` — 12.7% of jets, value 2.23 (1.75…2.69), formula right 97%
- `` — 7.4% of jets, value 1.65 (1.00…2.25), formula right 92%
- `` — 24.8% of jets, value 1.50 (0.69…2.12), formula right 85%
- `` — 2.0% of jets, value 0.78 (0.06…1.44), formula right 76%
- `` — 20.8% of jets, value 0.76 (0.00…1.50), formula right 75%
- `` — 3.4% of jets, value 0.26 (0.00…0.75), formula right 69%
- `` — 2.0% of jets, value 0.06 (0.00…0.25), formula right 61%
- `` — 26.9% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = -0.431
if mass < 80.40: z += -0.131 × (80.40 − mass)
if girth2_top20 < 0.023: z += 75.70 × (0.023 − girth2_top20)
if mass < 103: z += 0.045 × (103 − mass)
if mass_over_sum_pt_sq > 0.0084: z += 313 × (mass_over_sum_pt_sq − 0.0084)
if mass_over_sum_pt > 0.091: z += -71.70 × (mass_over_sum_pt − 0.091)
if D2 < 6.81: z += 0.224 × (6.81 − D2)
if mass_top40 < 84.60 and D2 < 6.72: z += -0.015 × (84.60 − mass_top40) × (6.72 − D2)
if z_dr_0p2_0p4 < 0.070: z += -10.10 × (0.070 − z_dr_0p2_0p4)
if sum_pt < 1080: z += -0.0055 × (1080 − sum_pt)
if girth2_top20 < 0.0078: z += -115 × (0.0078 − girth2_top20)
if n_particles > 23.70: z += -0.013 × (n_particles − 23.70)
if girth < 0.062: z += -21.80 × (0.062 − girth)
if mass < 118 and max_dr < 0.411: z += 0.104 × (118 − mass) × (0.411 − max_dr)
if n_dr_0p2_0p4 < 18.00: z += 0.026 × (18.00 − n_dr_0p2_0p4)
if tau21 < 0.454: z += -1.50 × (0.454 − tau21)
if mass < 74.30 and max_dr < 0.360: z += -0.564 × (74.30 − mass) × (0.360 − max_dr)
if e2 > 0.036: z += 18.60 × (e2 − 0.036)
if C2 > 0.107: z += 17.40 × (C2 − 0.107)
if tau32 < 0.573: z += 2.27 × (0.573 − tau32)
if mass_over_sum_pt_sq > 0.026: z += -119 × (mass_over_sum_pt_sq − 0.026)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 47.7% of jets, neuron 1.53, formula right for 86%.  
- **group 2** — 11.2% of jets, neuron 1.35, formula right for 90%.  
- **group 3** — 8.3% of jets, neuron 0.54, formula right for 70%.  
- **group 4** — 7.7% of jets, neuron 0.08, formula right for 65%.  
- **group 5** — 6.5% of jets, neuron 0.00, formula right for 81%.  
- **group 6** — 5.7% of jets, neuron 0.00, formula right for 73%.  
- **group 7** — 4.8% of jets, neuron 0.00, formula right for 75%.  
- **group 8** — 4.8% of jets, neuron 0.00, formula right for 68%.  
- **group 9** — 2.1% of jets, neuron 1.41, formula right for 79%.  
- **group 10** — 1.3% of jets, neuron 0.00, formula right for 85%.  

### neuron 5: Few-particle jet in a pT band (major)

- **What it measures:** Jumps up when the log of the total pT lies between 6.85 and 6.91 (a narrow band around 1 TeV) and when m/pT lies between 0.0786 and 0.0906; beyond that it is higher for jets with fewer particles whose 10-20 hardest particles carry more of the pT. Quark and Z jets sit highest (1.43 and 1.34), W jets next, top and gluon jets lowest (AUC 0.27: small for g).
- *computed — its value:* largest for q (1.43), then Z (1.34), then W (1.09), then t (0.60), then g (0.50); it separates g jets from the rest best (AUC 0.27: small for g)
- **How the class scores use it:** It raises the W score (+11%) and the Z score (+13%) and lowers the g score (-8%) and the t score (-10%): a compact, few-particle jet with moderate m/pT looks like a boson, not a gluon or top. It hardly enters the q score, although quark jets sit highest on it.
- *computed — used by:* raises the score of W (+11%), Z (+13%); lowers the score of g (-8%), t (-10%); does not (or hardly) enter the score of q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.56):

- `` — 61.5% of jets, value 1.45 (0.75…2.19), formula right 81%
- `` — 2.1% of jets, value 0.71 (0.00…1.44), formula right 75%
- `` — 2.9% of jets, value 0.64 (0.00…1.25), formula right 75%
- `` — 4.1% of jets, value 0.42 (0.00…0.94), formula right 80%
- `` — 2.0% of jets, value 0.26 (0.00…0.88), formula right 81%
- `` — 10.8% of jets, value 0.23 (0.00…0.62), formula right 72%
- `` — 6.0% of jets, value 0.14 (0.00…0.56), formula right 93%
- `` — 10.6% of jets, value 0.04 (0.00…0.06), formula right 84%

```
z = -0.546
if log_sum_pt > 6.85: z += 31.60 × (log_sum_pt − 6.85)
if log_sum_pt > 6.91: z += -44.10 × (log_sum_pt − 6.91)
if mass_over_sum_pt > 0.079: z += 39.60 × (mass_over_sum_pt − 0.079)
if mass_over_sum_pt > 0.091: z += -39.70 × (mass_over_sum_pt − 0.091)
if n_particles < 65.00 and e2 < 0.038: z += 1.72 × (65.00 − n_particles) × (0.038 − e2)
if mass_top50 > 155: z += -0.213 × (mass_top50 − 155)
if z_top30_slots > 0.936: z += -8.88 × (z_top30_slots − 0.936)
if tau32 < 0.881: z += 1.14 × (0.881 − tau32)
if sum_pt_top3 > 239: z += 0.00069 × (sum_pt_top3 − 239)
if max_dr > 0.437: z += 19.70 × (max_dr − 0.437)
if mass_over_sum_pt_sq < 0.0064: z += -103 × (0.0064 − mass_over_sum_pt_sq)
if mass_over_sum_pt_sq > 0.029: z += -450 × (mass_over_sum_pt_sq − 0.029)
if sum_pt_top3 > 791: z += -0.0048 × (sum_pt_top3 − 791)
if mass_top50 > 155 and z_dr_0p05_0p1 > 0.581: z += 1.23 × (mass_top50 − 155) × (z_dr_0p05_0p1 − 0.581)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 32.1% of jets, neuron 1.29, formula right for 76%.  
- **group 2** — 28.2% of jets, neuron 1.35, formula right for 83%.  
- **group 3** — 11.5% of jets, neuron 0.61, formula right for 79%.  
- **group 4** — 10.4% of jets, neuron 0.84, formula right for 82%.  
- **group 5** — 6.3% of jets, neuron 0.07, formula right for 82%.  
- **group 6** — 5.7% of jets, neuron 0.04, formula right for 90%.  
- **group 7** — 3.1% of jets, neuron 0.04, formula right for 86%.  
- **group 8** — 1.3% of jets, neuron 1.98, formula right for 81%.  
- **group 9** — 1.0% of jets, neuron 0.01, formula right for 90%.  
- **group 10** — 0.5% of jets, neuron 0.02, formula right for 80%.  

### neuron 8: Busy, wide radiation (major)

- **What it measures:** Follows the activity away from the core: many particles and much pT at 0.2 <= ΔR < 0.4, a broad minor axis (lam2), many particles in total and a small pT share held by the 30-40 hardest; its main thresholds sit near an m/pT of 0.16, and masses between 65.2 and 86.5 GeV push it down. Top jets sit far highest (mean 6.74, AUC 0.91), gluon jets next (2.16), then quark jets (0.98), with Z and W jets near zero.
- *computed — its value:* largest for t (6.74), then g (2.16), then q (0.98), then Z (0.29), then W (0.08); it separates t jets from the rest best (AUC 0.91: large for t)
- **How the class scores use it:** It lowers the W score (-35%) and the Z score (-37%) strongly, since W and Z jets are clean two-prong jets with little wide radiation, and it raises the t score (+10%); freezing it costs 2.444 points of accuracy. It hardly enters the g and q scores.
- *computed — used by:* raises the score of t (+10%); lowers the score of W (-35%), Z (-37%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.813):

- `` — 11.0% of jets, value 9.50 (4.94…13.88), formula right 88%
- `` — 4.5% of jets, value 6.86 (3.94…10.62), formula right 70%
- `` — 2.9% of jets, value 5.52 (3.48…7.44), formula right 79%
- `` — 2.0% of jets, value 4.51 (0.75…9.36), formula right 65%
- `` — 4.3% of jets, value 3.16 (1.94…4.44), formula right 75%
- `` — 3.6% of jets, value 2.67 (1.19…4.25), formula right 66%
- `` — 71.6% of jets, value 0.29 (0.00…1.00), formula right 82%

```
z = 5.90
if mass_over_sum_pt < 0.161: z += -146 × (0.161 − mass_over_sum_pt)
if mass_over_sum_pt_sq < 0.025: z += 441 × (0.025 − mass_over_sum_pt_sq)
if mass > 65.20: z += -0.072 × (mass − 65.20)
if mass > 86.50: z += 0.089 × (mass − 86.50)
if mass_over_sum_pt > 0.076 and sum_pt < 1090: z += 0.635 × (mass_over_sum_pt − 0.076) × (1090 − sum_pt)
if mass > 105: z += -0.073 × (mass − 105)
if n_dr_0p2_0p4 < 20.60: z += -0.059 × (20.60 − n_dr_0p2_0p4)
if girth2_top20 > 0.0079: z += 243 × (girth2_top20 − 0.0079)
if mass < 64.50: z += 0.118 × (64.50 − mass)
if e2 > 0.020: z += -43.50 × (e2 − 0.020)
if sum_pt_top40 < 1040: z += -0.0078 × (1040 − sum_pt_top40)
if lam2 < 0.0015: z += -487 × (0.0015 − lam2)
if sum_pt_top40 < 1000: z += 0.012 × (1000 − sum_pt_top40)
if n_particles > 50.70: z += 0.074 × (n_particles − 50.70)
if log_sum_pt > 6.94: z += 5.70 × (log_sum_pt − 6.94)
if C2 > 0.070: z += 9.62 × (C2 − 0.070)
if n_particles < 46.10: z += -0.020 × (46.10 − n_particles)
if D2 < 1.80: z += 0.411 × (1.80 − D2)
if n_particles > 51.30 and z_top50_slots > 0.970: z += -2.79 × (n_particles − 51.30) × (z_top50_slots − 0.970)
if sum_pt_top40 < 1010 and max_dr < 0.390: z += -0.093 × (1010 − sum_pt_top40) × (0.390 − max_dr)
if sum_pt_top40 < 883 and log_sum_pt < 6.89: z += 0.068 × (883 − sum_pt_top40) × (6.89 − log_sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.5% of jets, neuron 0.30, formula right for 82%.  
- **group 2** — 22.7% of jets, neuron 1.03, formula right for 85%.  
- **group 3** — 13.2% of jets, neuron 0.16, formula right for 79%.  
- **group 4** — 13.0% of jets, neuron 0.50, formula right for 73%.  
- **group 5** — 8.1% of jets, neuron 7.71, formula right for 90%.  
- **group 6** — 4.7% of jets, neuron 7.31, formula right for 71%.  
- **group 7** — 4.5% of jets, neuron 10.23, formula right for 88%.  
- **group 8** — 3.8% of jets, neuron 3.54, formula right for 69%.  
- **group 9** — 1.0% of jets, neuron 4.50, formula right for 76%.  
- **group 10** — 0.5% of jets, neuron 8.73, formula right for 61%.  

### neuron 10: Hard particles spread wide, heavy (major)

- **What it measures:** Rises when the 5 hardest particles are spread far from the axis (girth2_top5 at or above 0.0254), e2 is large, little pT sits within ΔR 0.05 of the axis, and the mass is above 123 GeV (masses above 144 GeV give back a little); jets below 80.4 GeV get a partial boost back. Top jets sit highest (mean 2.25, AUC 0.84), Z jets next (1.23), gluon and W jets at about 1.00, and quark jets lowest (0.76).
- *computed — its value:* largest for t (2.25), then Z (1.23), then g (1.00), then W (1.00), then q (0.76); it separates t jets from the rest best (AUC 0.84: large for t)
- **How the class scores use it:** It raises the t score (+27%) and hardly enters any other class score: widely spread hard prongs in a heavy jet are the main sign of a top. Freezing it costs 1.924 points of accuracy.
- *computed — used by:* raises the score of t (+27%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.636):

- `` — 20.5% of jets, value 2.41 (1.44…3.38), formula right 88%
- `` — 27.2% of jets, value 1.62 (0.81…2.38), formula right 77%
- `` — 12.4% of jets, value 1.14 (0.44…1.75), formula right 92%
- `` — 10.0% of jets, value 0.93 (0.12…1.75), formula right 75%
- `` — 2.2% of jets, value 0.65 (0.00…1.44), formula right 80%
- `` — 2.0% of jets, value 0.60 (0.00…1.75), formula right 77%
- `` — 4.8% of jets, value 0.31 (0.00…0.94), formula right 75%
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
- **group 2** — 14.2% of jets, neuron 1.23, formula right for 73%.  
- **group 3** — 9.7% of jets, neuron 2.37, formula right for 91%.  
- **group 4** — 9.6% of jets, neuron 0.38, formula right for 80%.  
- **group 5** — 9.5% of jets, neuron 0.61, formula right for 73%.  
- **group 6** — 9.4% of jets, neuron 0.06, formula right for 76%.  
- **group 7** — 9.0% of jets, neuron 2.49, formula right for 73%.  
- **group 8** — 7.7% of jets, neuron 1.16, formula right for 69%.  
- **group 9** — 1.6% of jets, neuron 0.50, formula right for 78%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 78%.  

### neuron 13: High total pT, below top mass (major)

- **What it measures:** Rises with the total pT of the jet (log of total pT at or above 7.0, total pT at or above 1020 GeV) and is pushed down for masses above 143 GeV, where top jets lie. Gluon, Z, W and quark jets all sit between 1.86 and 2.32, while top jets sit far lower (mean 1.04, AUC 0.19: small for t).
- *computed — its value:* largest for g (2.32), then Z (2.12), then W (2.03), then q (1.86), then t (1.04); it separates t jets from the rest best (AUC 0.19: small for t)
- **How the class scores use it:** It lowers the t score (-38%) and hardly enters any other class score: a high-pT jet without a top-like mass is unlikely to be a top. Freezing it costs 1.012 points of accuracy.
- *computed — used by:* lowers the score of t (-38%); does not (or hardly) enter the score of g, q, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.792):

- `` — 24.1% of jets, value 2.83 (2.38…3.06), formula right 82%
- `` — 41.6% of jets, value 2.12 (1.59…2.56), formula right 83%
- `` — 12.8% of jets, value 1.43 (0.81…2.00), formula right 74%
- `` — 5.4% of jets, value 1.19 (0.72…1.78), formula right 85%
- `` — 2.0% of jets, value 0.78 (0.41…1.25), formula right 94%
- `` — 4.5% of jets, value 0.61 (0.00…1.31), formula right 77%
- `` — 2.7% of jets, value 0.33 (0.00…0.84), formula right 94%
- `` — 6.7% of jets, value 0.14 (0.00…0.53), formula right 70%

```
z = 3.07
if log_sum_pt < 7.00: z += -14.00 × (7.00 − log_sum_pt)
if sum_pt_top40 < 1040: z += 0.015 × (1040 − sum_pt_top40)
if mass > 143: z += -0.145 × (mass − 143)
if sum_pt < 1020: z += -0.023 × (1020 − sum_pt)
if mass_top50 > 137: z += 0.060 × (mass_top50 − 137)
if sum_pt_top40 < 1040 and D2 < 5.03: z += -0.0015 × (1040 − sum_pt_top40) × (5.03 − D2)
if n_dr_0p2_0p4 < 4.41: z += -0.156 × (4.41 − n_dr_0p2_0p4)
if log_sum_pt < 6.80: z += 18.80 × (6.80 − log_sum_pt)
if mass > 173: z += 0.104 × (mass − 173)
if e2 > 0.037: z += 16.00 × (e2 − 0.037)
if sum_pt_top50 < 961 and D2 < 5.16: z += 0.0019 × (961 − sum_pt_top50) × (5.16 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 36.4% of jets, neuron 2.01, formula right for 82%.  
- **group 2** — 32.8% of jets, neuron 2.67, formula right for 82%.  
- **group 3** — 11.8% of jets, neuron 1.29, formula right for 71%.  
- **group 4** — 6.7% of jets, neuron 0.91, formula right for 90%.  
- **group 5** — 4.8% of jets, neuron 0.33, formula right for 71%.  
- **group 6** — 4.0% of jets, neuron 0.46, formula right for 92%.  
- **group 7** — 1.6% of jets, neuron 0.09, formula right for 64%.  
- **group 8** — 1.3% of jets, neuron 1.07, formula right for 76%.  
- **group 9** — 0.4% of jets, neuron 0.06, formula right for 59%.  
- **group 10** — 0.3% of jets, neuron 2.19, formula right for 79%.  

### neuron 0: W-side mass, below the Z (moderate)

- **What it measures:** Pushed up for jets lighter than 80.4 GeV (or with small m/pT) and pushed down for masses between 80.4 and 91.2 GeV, the Z side of the W peak; overall it falls as mass and width grow. W jets sit far highest (mean 1.75, AUC 0.96), quark and gluon jets low, Z and top jets lowest.
- *computed — its value:* largest for W (1.75), then q (0.37), then g (0.26), then Z (0.12), then t (0.09); it separates W jets from the rest best (AUC 0.96: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the Z score (-14%): sitting high on this scale is evidence for a W rather than a Z. It hardly enters the g, q and t scores.
- *computed — used by:* raises the score of W (+8%); lowers the score of Z (-14%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.839):

- `` — 15.2% of jets, value 2.00 (1.41…2.44), formula right 91%
- `` — 7.3% of jets, value 1.13 (0.00…1.88), formula right 67%
- `` — 10.1% of jets, value 0.76 (0.06…1.34), formula right 69%
- `` — 2.0% of jets, value 0.33 (0.00…0.88), formula right 69%
- `` — 19.8% of jets, value 0.25 (0.00…0.59), formula right 77%
- `` — 2.0% of jets, value 0.09 (0.00…0.34), formula right 84%
- `` — 43.6% of jets, value 0.00 (0.00…0.00), formula right 84%

```
z = 1.30
if mass > 80.40: z += -0.159 × (mass − 80.40)
if mass > 91.20: z += 0.154 × (mass − 91.20)
if mass_over_sum_pt > 0.078: z += -66.90 × (mass_over_sum_pt − 0.078)
if mass_over_sum_pt > 0.086: z += 79.90 × (mass_over_sum_pt − 0.086)
if mass_over_sum_pt_sq < 0.0078: z += 340 × (0.0078 − mass_over_sum_pt_sq)
if girth2_top20 < 0.0061: z += -162 × (0.0061 − girth2_top20)
if girth < 0.056: z += -27.20 × (0.056 − girth)
if lam1 < 0.0058: z += -197 × (0.0058 − lam1)
if mass_top50 < 80.80: z += -0.016 × (80.80 − mass_top50)
if n_dr_0p2_0p4 < 10.80: z += 0.040 × (10.80 − n_dr_0p2_0p4)
if log_sum_pt < 6.99 and sum_pt_top40 > 936: z += 0.051 × (6.99 − log_sum_pt) × (sum_pt_top40 − 936)
if sum_pt < 1010: z += -0.0048 × (1010 − sum_pt)
if mass > 80.40 and n_dr_0p2_0p4 < 7.00: z += -0.0092 × (mass − 80.40) × (7.00 − n_dr_0p2_0p4)
if mass_top50 < 70.30 and z_dr_0p05_0p1 < 0.231: z += 0.045 × (70.30 − mass_top50) × (0.231 − z_dr_0p05_0p1)
if z_dr_0_0p05 > 0.871: z += -2.97 × (z_dr_0_0p05 − 0.871)
if z_top30_slots > 0.913 and C2 > 0.064: z += -64.40 × (z_top30_slots − 0.913) × (C2 − 0.064)
if n_dr_0p2_0p4 < 9.75 and z_top50_slots < 0.980: z += -2.13 × (9.75 − n_dr_0p2_0p4) × (0.980 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.4% of jets, neuron 1.51, formula right for 80%.  
- **group 2** — 25.0% of jets, neuron 0.35, formula right for 76%.  
- **group 3** — 21.2% of jets, neuron 0.02, formula right for 88%.  
- **group 4** — 7.1% of jets, neuron 0.00, formula right for 92%.  
- **group 5** — 4.8% of jets, neuron 0.00, formula right for 85%.  
- **group 6** — 4.4% of jets, neuron 0.00, formula right for 68%.  
- **group 7** — 3.8% of jets, neuron 0.00, formula right for 72%.  
- **group 8** — 3.6% of jets, neuron 0.00, formula right for 71%.  
- **group 9** — 1.3% of jets, neuron 0.01, formula right for 76%.  
- **group 10** — 0.3% of jets, neuron 0.01, formula right for 78%.  

### neuron 3: Sparse jet, empty outer ring (moderate)

- **What it measures:** Rises for jets with few particles, few particles at 0.2 <= ΔR < 0.4 while the 50 hardest carry over 0.979 of the pT, a thin minor axis (lam2 below 0.000569) and small m/pT; a compact two-prong pattern (tau21 below 0.419) pulls it down a little. W and quark jets sit highest (0.83 and 0.77), Z jets next, gluon jets low and top jets almost at zero (AUC 0.17: small for t).
- *computed — its value:* largest for W (0.83), then q (0.77), then Z (0.58), then g (0.23), then t (0.04); it separates t jets from the rest best (AUC 0.17: small for t)
- **How the class scores use it:** It raises the W score (+4%) and the Z score (+4%) and lowers the g score (-9%): a sparse, clean jet looks like a boson rather than a gluon. It hardly enters the q and t scores, even though quark jets sit high on it.
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

### neuron 6: One-prong jet outside the Z mass (moderate)

- **What it measures:** Follows one-prong-ness (large D2 and tau21, a round rather than elongated pT pattern) and is pushed down for m/pT above 0.0491 and for masses below 110 GeV, most of all between 91.2 and 110 GeV. Quark jets sit highest (1.53), gluon jets next (1.23), top jets in between (0.93), Z and W jets near zero (AUC 0.17: small for W).
- *computed — its value:* largest for q (1.53), then g (1.23), then t (0.93), then Z (0.20), then W (0.16); it separates W jets from the rest best (AUC 0.17: small for W)
- **How the class scores use it:** It raises the q score (+7%) and the g score (+3%) and lowers the Z score (-15%): a one-prong jet away from the Z mass is not a Z. It hardly enters the W and t scores.
- *computed — used by:* raises the score of g (+3%), q (+7%); lowers the score of Z (-15%); does not (or hardly) enter the score of W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.722):

- `` — 2.0% of jets, value 2.54 (1.88…3.69), formula right 80%
- `` — 8.2% of jets, value 2.08 (1.75…2.44), formula right 80%
- `` — 3.9% of jets, value 1.81 (1.38…2.12), formula right 74%
- `` — 3.5% of jets, value 1.72 (0.75…2.69), formula right 83%
- `` — 6.8% of jets, value 1.44 (1.06…1.75), formula right 73%
- `` — 21.6% of jets, value 1.08 (0.00…1.88), formula right 80%
- `` — 6.5% of jets, value 0.88 (0.44…1.31), formula right 68%
- `` — 47.5% of jets, value 0.13 (0.00…0.38), formula right 84%

```
z = 2.55
if mass_over_sum_pt > 0.049: z += -40.30 × (mass_over_sum_pt − 0.049)
if mass < 110: z += -0.051 × (110 − mass)
if mass < 91.20: z += 0.047 × (91.20 − mass)
if mass_over_sum_pt_sq < 0.010: z += -166 × (0.010 − mass_over_sum_pt_sq)
if n_dr_0p2_0p4 < 18.50: z += -0.059 × (18.50 − n_dr_0p2_0p4)
if lam1 > 0.0079: z += 228 × (lam1 − 0.0079)
if mass_over_sum_pt > 0.050 and n_dr_0p2_0p4 < 18.60: z += 1.77 × (mass_over_sum_pt − 0.050) × (18.60 − n_dr_0p2_0p4)
if e2 < 0.048: z += 23.00 × (0.048 − e2)
if mass_top50 < 70.50: z += 0.053 × (70.50 − mass_top50)
if e2 > 0.053: z += -51.90 × (e2 − 0.053)
if log_sum_pt < 6.82: z += 6.95 × (6.82 − log_sum_pt)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 23.5% of jets, neuron 0.09, formula right for 85%.  
- **group 2** — 15.9% of jets, neuron 0.13, formula right for 91%.  
- **group 3** — 10.2% of jets, neuron 1.59, formula right for 73%.  
- **group 4** — 9.8% of jets, neuron 2.18, formula right for 80%.  
- **group 5** — 8.6% of jets, neuron 0.38, formula right for 75%.  
- **group 6** — 8.3% of jets, neuron 0.86, formula right for 68%.  
- **group 7** — 8.2% of jets, neuron 1.04, formula right for 91%.  
- **group 8** — 6.4% of jets, neuron 1.18, formula right for 78%.  
- **group 9** — 6.4% of jets, neuron 1.16, formula right for 69%.  
- **group 10** — 2.7% of jets, neuron 1.71, formula right for 81%.  

### neuron 7: Z mass window, 91-120 GeV (moderate)

- **What it measures:** Switched on mainly for jet masses between 91.2 and 120 GeV (masses below 91.2 GeV push it down), further tuned by m/pT and by the spread of the 20 hardest particles. Z jets sit far highest (mean 2.12, AUC 0.91); all other types stay near zero (0.12 or less).
- *computed — its value:* largest for Z (2.12), then W (0.12), then t (0.06), then g (0.06), then q (0.04); it separates Z jets from the rest best (AUC 0.91: large for Z)
- **How the class scores use it:** It raises the Z score (+9%) and lowers the W score (-6%) and the t score (-3%): a mass in the Z window speaks for a Z and against a W or top. It hardly enters the g and q scores.
- *computed — used by:* raises the score of Z (+9%); lowers the score of W (-6%), t (-3%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.757):

- `` — 8.2% of jets, value 3.37 (2.00…4.94), formula right 97%
- `` — 11.5% of jets, value 1.22 (0.06…2.56), formula right 82%
- `` — 2.0% of jets, value 0.55 (0.00…1.50), formula right 88%
- `` — 6.0% of jets, value 0.53 (0.00…2.00), formula right 77%
- `` — 7.0% of jets, value 0.12 (0.00…0.50), formula right 82%
- `` — 9.1% of jets, value 0.07 (0.00…0.31), formula right 91%
- `` — 22.0% of jets, value 0.03 (0.00…0.00), formula right 81%
- `` — 34.1% of jets, value 0.00 (0.00…0.00), formula right 74%

```
z = -0.571
if mass < 91.20: z += -0.190 × (91.20 − mass)
if mass_over_sum_pt_sq < 0.0095: z += 788 × (0.0095 − mass_over_sum_pt_sq)
if mass < 120: z += 0.065 × (120 − mass)
if mass_over_sum_pt < 0.091: z += -126 × (0.091 − mass_over_sum_pt)
if girth2_top20 < 0.008: z += -570 × (0.008 − girth2_top20)
if mass_over_sum_pt < 0.080: z += 100 × (0.080 − mass_over_sum_pt)
if mass < 101 and max_dr < 0.397: z += 0.814 × (101 − mass) × (0.397 − max_dr)
if mass < 91.20 and max_dr < 0.392: z += -1.33 × (91.20 − mass) × (0.392 − max_dr)
if girth2_top20 < 0.0063: z += 534 × (0.0063 − girth2_top20)
if mass_top50 < 97.40 and D2 < 1.65: z += -0.388 × (97.40 − mass_top50) × (1.65 − D2)
if mass < 100 and D2 < 1.64: z += 0.273 × (100 − mass) × (1.64 − D2)
if mass_over_sum_pt_sq < 0.0064: z += -514 × (0.0064 − mass_over_sum_pt_sq)
if z_dr_0p2_0p4 < 0.090: z += -10.20 × (0.090 − z_dr_0p2_0p4)
if girth < 0.088: z += -19.40 × (0.088 − girth)
if n_dr_0p2_0p4 < 13.00: z += 0.077 × (13.00 − n_dr_0p2_0p4)
if e2 < 0.029: z += 66.50 × (0.029 − e2)
if z_top50_slots < 0.991: z += -36.20 × (0.991 − z_top50_slots)
if D2 < 1.80 and n_dr_0p2_0p4 < 9.09: z += 0.083 × (1.80 − D2) × (9.09 − n_dr_0p2_0p4)
if z_dr_0p2_0p4 < 0.0059: z += 99.20 × (0.0059 − z_dr_0p2_0p4)
if mass < 91.20 and z_dr_0p05_0p1 > 0.422: z += -0.149 × (91.20 − mass) × (z_dr_0p05_0p1 − 0.422)
if mass_top30 < 75.60 and D2 < 1.61: z += 0.183 × (75.60 − mass_top30) × (1.61 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.9% of jets, neuron 0.03, formula right for 79%.  
- **group 2** — 14.1% of jets, neuron 1.31, formula right for 82%.  
- **group 3** — 13.3% of jets, neuron 0.15, formula right for 78%.  
- **group 4** — 11.1% of jets, neuron 2.31, formula right for 93%.  
- **group 5** — 8.7% of jets, neuron 0.00, formula right for 72%.  
- **group 6** — 8.6% of jets, neuron 0.14, formula right for 92%.  
- **group 7** — 7.3% of jets, neuron 0.00, formula right for 79%.  
- **group 8** — 7.3% of jets, neuron 0.01, formula right for 69%.  
- **group 9** — 3.8% of jets, neuron 0.00, formula right for 80%.  
- **group 10** — 0.9% of jets, neuron 0.00, formula right for 86%.  

### neuron 9: Light one-prong, low m/pT (moderate)

- **What it measures:** Falls sharply once m/pT exceeds 0.0588 and rises when the 40 hardest particles weigh less than 80.4 GeV, with a partial offset for heavier jets (mass above 84.5 GeV); overall it tracks one-prong-ness (large tau21) and low mass. Quark jets sit highest (mean 2.05, AUC 0.82), gluon jets next (1.20), top, W and Z jets low.
- *computed — its value:* largest for q (2.05), then g (1.20), then t (0.39), then W (0.20), then Z (0.16); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** It raises the q score (+17%) and the g score (+10%) and lowers the W score (-3%) slightly: a light one-prong jet is a quark or gluon, not a boson. It hardly enters the Z and t scores.
- *computed — used by:* raises the score of g (+10%), q (+17%); lowers the score of W (-3%); does not (or hardly) enter the score of Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.873):

- `` — 2.0% of jets, value 4.14 (2.81…6.12), formula right 73%
- `` — 14.4% of jets, value 2.71 (2.19…3.25), formula right 79%
- `` — 2.0% of jets, value 2.62 (1.49…4.06), formula right 63%
- `` — 7.3% of jets, value 1.83 (1.31…2.31), formula right 74%
- `` — 2.3% of jets, value 1.19 (0.69…1.69), formula right 56%
- `` — 2.0% of jets, value 0.68 (0.00…1.25), formula right 71%
- `` — 17.9% of jets, value 0.38 (0.00…0.94), formula right 84%
- `` — 52.1% of jets, value 0.05 (0.00…0.06), formula right 84%

```
z = 0.389
if mass_over_sum_pt > 0.059: z += -59.50 × (mass_over_sum_pt − 0.059)
if mass_over_sum_pt_sq > 0.0075: z += 216 × (mass_over_sum_pt_sq − 0.0075)
if mass_top40 < 80.40: z += 0.060 × (80.40 − mass_top40)
if mass > 84.50: z += 0.037 × (mass − 84.50)
if sum_pt_top40 < 1010: z += 0.0084 × (1010 − sum_pt_top40)
if girth < 0.042: z += -29.40 × (0.042 − girth)
if mass_over_sum_pt_sq > 0.026: z += -315 × (mass_over_sum_pt_sq − 0.026)
if mass > 151: z += -0.038 × (mass − 151)
if z_top50_slots < 0.979: z += -42.50 × (0.979 − z_top50_slots)
if mass_top40 < 115 and max_dr > 0.375: z += 0.058 × (115 − mass_top40) × (max_dr − 0.375)
if sum_pt_top40 < 931 and z_top30_slots > 0.977: z += 0.769 × (931 − sum_pt_top40) × (z_top30_slots − 0.977)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 25.9% of jets, neuron 0.09, formula right for 84%.  
- **group 2** — 21.5% of jets, neuron 0.02, formula right for 86%.  
- **group 3** — 14.3% of jets, neuron 2.81, formula right for 79%.  
- **group 4** — 12.8% of jets, neuron 1.84, formula right for 71%.  
- **group 5** — 6.4% of jets, neuron 0.65, formula right for 87%.  
- **group 6** — 5.9% of jets, neuron 0.10, formula right for 92%.  
- **group 7** — 5.8% of jets, neuron 0.15, formula right for 69%.  
- **group 8** — 5.3% of jets, neuron 0.42, formula right for 71%.  
- **group 9** — 1.1% of jets, neuron 0.00, formula right for 73%.  
- **group 10** — 1.0% of jets, neuron 4.64, formula right for 68%.  

### neuron 11: Clean elongated two-prong jet (moderate)

- **What it measures:** Rises when m/pT is above 0.0797 while the ring at 0.2 <= ΔR < 0.4 is nearly empty (a count below 14.3, pT share below 0.0049), e2 is below 0.0234 and lam2 is small: an elongated two-prong jet with no stray radiation. W jets sit highest (mean 1.50, AUC 0.80), Z jets next (1.27), gluon, quark and top jets low.
- *computed — its value:* largest for W (1.50), then Z (1.27), then g (0.32), then q (0.29), then t (0.24); it separates W jets from the rest best (AUC 0.80: large for W)
- **How the class scores use it:** It raises the W score (+8%) and lowers the q score (-7%): a clean, elongated two-prong jet is W-like, not quark-like. It hardly enters the g, Z and t scores, even though Z jets sit high on it.
- *computed — used by:* raises the score of W (+8%); lowers the score of q (-7%); does not (or hardly) enter the score of g, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.844):

- `` — 10.8% of jets, value 2.60 (2.06…3.19), formula right 97%
- `` — 6.4% of jets, value 1.75 (1.25…2.25), formula right 95%
- `` — 5.9% of jets, value 1.39 (0.81…2.00), formula right 88%
- `` — 2.9% of jets, value 0.81 (0.50…1.06), formula right 83%
- `` — 40.5% of jets, value 0.42 (0.00…0.88), formula right 76%
- `` — 4.5% of jets, value 0.38 (0.06…0.88), formula right 74%
- `` — 2.0% of jets, value 0.27 (0.00…0.88), formula right 74%
- `` — 27.0% of jets, value 0.15 (0.06…0.44), formula right 78%

```
z = 0.032
if mass_over_sum_pt < 0.080: z += -41.70 × (0.080 − mass_over_sum_pt)
if n_dr_0p2_0p4 < 14.30: z += 0.063 × (14.30 − n_dr_0p2_0p4)
if e2 < 0.023: z += 73.90 × (0.023 − e2)
if lam2 < 0.00083: z += 748 × (0.00083 − lam2)
if z_dr_0p2_0p4 < 0.0049: z += 183 × (0.0049 − z_dr_0p2_0p4)
if mass < 98.10 and D2 < 1.27: z += 0.091 × (98.10 − mass) × (1.27 − D2)
if mass < 80.40 and D2 < 3.24: z += -0.028 × (80.40 − mass) × (3.24 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 30.2% of jets, neuron 0.17, formula right for 77%.  
- **group 2** — 18.8% of jets, neuron 0.73, formula right for 80%.  
- **group 3** — 11.1% of jets, neuron 2.01, formula right for 96%.  
- **group 4** — 9.2% of jets, neuron 0.19, formula right for 76%.  
- **group 5** — 8.1% of jets, neuron 0.44, formula right for 82%.  
- **group 6** — 7.5% of jets, neuron 0.40, formula right for 71%.  
- **group 7** — 5.8% of jets, neuron 2.85, formula right for 95%.  
- **group 8** — 4.3% of jets, neuron 1.60, formula right for 89%.  
- **group 9** — 2.6% of jets, neuron 0.04, formula right for 63%.  
- **group 10** — 2.4% of jets, neuron 0.00, formula right for 75%.  

### neuron 12: Lighter than the W (moderate)

- **What it measures:** Strongly pushed up for jets lighter than 80.4 GeV, but pulled down again when the 50 hardest particles weigh less than 62.5 GeV or C2 is above 0.0556; overall it falls as mass, width and m/pT grow. Quark jets sit highest (mean 1.25, AUC 0.79), gluon jets next (0.74), W, top and Z jets low.
- *computed — its value:* largest for q (1.25), then g (0.74), then W (0.25), then t (0.19), then Z (0.18); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** It raises the q score (+7%), the g score (+3%) and the Z score (+3%), and lowers the W score (-4%) and the t score (-4%): a jet lighter than the W speaks for a light quark or gluon and against a W or top. The small Z share is only a minor correction.
- *computed — used by:* raises the score of g (+3%), q (+7%), Z (+3%); lowers the score of W (-4%), t (-4%) (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.739):

- `` — 12.4% of jets, value 1.86 (1.50…2.12), formula right 78%
- `` — 8.3% of jets, value 1.39 (1.00…1.75), formula right 73%
- `` — 2.0% of jets, value 1.33 (0.00…4.00), formula right 83%
- `` — 3.1% of jets, value 1.03 (0.62…1.38), formula right 70%
- `` — 3.9% of jets, value 0.54 (0.00…1.00), formula right 62%
- `` — 2.0% of jets, value 0.48 (0.00…0.88), formula right 68%
- `` — 68.2% of jets, value 0.12 (0.00…0.25), formula right 84%

```
z = 0.082
if mass < 80.40: z += 0.084 × (80.40 − mass)
if C2 > 0.056: z += -21.60 × (C2 − 0.056)
if mass_top50 < 62.50: z += -0.070 × (62.50 − mass_top50)
if mass_top30 > 128: z += 0.036 × (mass_top30 − 128)
if log_sum_pt < 6.85: z += -5.35 × (6.85 − log_sum_pt)
if LHA > 0.406 and lam2 < 0.0046: z += 15700 × (LHA − 0.406) × (0.0046 − lam2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 43.5% of jets, neuron 0.15, formula right for 87%.  
- **group 2** — 11.8% of jets, neuron 0.02, formula right for 72%.  
- **group 3** — 7.5% of jets, neuron 1.58, formula right for 74%.  
- **group 4** — 7.2% of jets, neuron 1.37, formula right for 72%.  
- **group 5** — 6.7% of jets, neuron 1.01, formula right for 67%.  
- **group 6** — 6.7% of jets, neuron 0.00, formula right for 75%.  
- **group 7** — 6.3% of jets, neuron 1.88, formula right for 83%.  
- **group 8** — 5.1% of jets, neuron 0.23, formula right for 94%.  
- **group 9** — 3.9% of jets, neuron 0.00, formula right for 86%.  
- **group 10** — 1.3% of jets, neuron 2.75, formula right for 74%.  

### neuron 14: Mass just above the Z (moderate)

- **What it measures:** Switched on for masses between 91.2 and 135 GeV (lighter jets are pushed down) and pushed down further for m/pT below 0.0888; it grows with mass and with small tau21. Z jets sit highest (mean 1.30, AUC 0.88), gluon jets next (0.46), top, W and quark jets low.
- *computed — its value:* largest for Z (1.30), then g (0.46), then t (0.32), then W (0.18), then q (0.14); it separates Z jets from the rest best (AUC 0.88: large for Z)
- **How the class scores use it:** It lowers the W score (-13%): a mass above the Z is evidence against a W. It hardly enters the g, q, Z and t scores, even though Z jets sit highest on it.
- *computed — used by:* lowers the score of W (-13%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.819):

- `` — 2.0% of jets, value 1.70 (0.72…2.97), formula right 75%
- `` — 25.4% of jets, value 1.42 (0.94…1.81), formula right 86%
- `` — 4.3% of jets, value 0.75 (0.34…1.19), formula right 70%
- `` — 2.0% of jets, value 0.46 (0.09…0.78), formula right 69%
- `` — 11.0% of jets, value 0.19 (0.00…0.44), formula right 93%
- `` — 2.6% of jets, value 0.13 (0.00…0.34), formula right 78%
- `` — 18.6% of jets, value 0.10 (0.00…0.44), formula right 83%
- `` — 34.2% of jets, value 0.01 (0.00…0.00), formula right 75%

```
z = 0.049
if mass < 135: z += 0.048 × (135 − mass)
if mass < 91.20: z += -0.135 × (91.20 − mass)
if mass_over_sum_pt < 0.089: z += -52.00 × (0.089 − mass_over_sum_pt)
if max_dr > 0.225: z += -2.24 × (max_dr − 0.225)
if girth2_top20 < 0.013 and tau21 < 0.600: z += -118 × (0.013 − girth2_top20) × (0.600 − tau21)
if log_sum_pt > 6.94: z += 2.54 × (log_sum_pt − 6.94)
if z_top50_slots < 0.960: z += 89.40 × (0.960 − z_top50_slots)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 25.1% of jets, neuron 1.42, formula right for 85%.  
- **group 2** — 21.0% of jets, neuron 0.26, formula right for 86%.  
- **group 3** — 20.1% of jets, neuron 0.16, formula right for 83%.  
- **group 4** — 6.0% of jets, neuron 0.00, formula right for 77%.  
- **group 5** — 5.6% of jets, neuron 0.03, formula right for 66%.  
- **group 6** — 5.5% of jets, neuron 0.00, formula right for 73%.  
- **group 7** — 5.4% of jets, neuron 0.00, formula right for 70%.  
- **group 8** — 4.9% of jets, neuron 0.00, formula right for 70%.  
- **group 9** — 4.4% of jets, neuron 0.00, formula right for 84%.  
- **group 10** — 2.0% of jets, neuron 2.00, formula right for 75%.  

### neuron 2: Very hard jet with large C2 (minor)

- **What it measures:** Stays at a fixed base value for almost every jet and rises only when the 50 hardest particles carry more than 1090 GeV and C2 (a three-point energy correlation ratio) is above 0.0936. It barely differs between jet types: means run from 0.50 for g down to 0.38 for Z (AUC at most 0.53).
- *computed — its value:* largest for g (0.50), then t (0.42), then q (0.39), then W (0.38), then Z (0.38); it separates g jets from the rest best (AUC 0.53: large for g)
- **How the class scores use it:** It lowers the Z score only slightly (-3%), a small correction. It hardly enters the g, q, W and t scores.
- *computed — used by:* lowers the score of Z (-3%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.492):

- `` — 2.0% of jets, value 2.19 (0.38…4.88), formula right 81%
- `` — 3.9% of jets, value 0.47 (0.38…0.75), formula right 80%
- `` — 2.1% of jets, value 0.40 (0.38…0.50), formula right 77%
- `` — 83.0% of jets, value 0.38 (0.38…0.38), formula right 80%
- `` — 9.0% of jets, value 0.38 (0.38…0.38), formula right 84%

```
z = 0.380
if sum_pt_top50 > 1090 and C2 > 0.094: z += 0.637 × (sum_pt_top50 − 1090) × (C2 − 0.094)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 98.2% of jets, neuron 0.38, formula right for 81%.  
- **group 2** — 0.9% of jets, neuron 1.35, formula right for 81%.  
- **group 3** — 0.4% of jets, neuron 2.67, formula right for 78%.  
- **group 4** — 0.2% of jets, neuron 4.21, formula right for 78%.  
- **group 5** — 0.1% of jets, neuron 6.10, formula right for 87%.  
- **group 6** — 0.1% of jets, neuron 2.94, formula right for 65%.  
- **group 7** — 0.0% of jets, neuron 4.61, formula right for 91%.  

### neuron 15: Narrow, core-dominated jet (minor)

- **What it measures:** Pulled down once girth exceeds 0.0293, only partly offset by an LHA term, so overall it is highest for narrow jets with much of the pT within ΔR 0.05 of the axis. Values are small and similar across types: quark jets highest (0.32), then gluon and W jets, with Z and top jets lowest (0.15 and 0.13).
- *computed — its value:* largest for q (0.32), then g (0.25), then W (0.24), then Z (0.15), then t (0.13); it separates t jets from the rest best (AUC 0.35: small for t)
- **How the class scores use it:** It raises the Z score only slightly (+2%), a small correction. It hardly enters the g, q, W and t scores.
- *computed — used by:* raises the score of Z (+2%); does not (or hardly) enter the score of g, q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.426):

- `` — 2.5% of jets, value 0.95 (0.50…1.44), formula right 65%
- `` — 2.0% of jets, value 0.44 (0.00…1.12), formula right 67%
- `` — 30.3% of jets, value 0.38 (0.06…0.69), formula right 75%
- `` — 18.6% of jets, value 0.17 (0.00…0.44), formula right 81%
- `` — 12.4% of jets, value 0.17 (0.00…0.38), formula right 83%
- `` — 34.2% of jets, value 0.05 (0.00…0.19), formula right 87%

```
z = 2.19
if girth > 0.029: z += -120 × (girth − 0.029)
if LHA > 0.163: z += 34.50 × (LHA − 0.163)
if lam1 < 0.018: z += -136 × (0.018 − lam1)
if mass_over_sum_pt > 0.093: z += 28.90 × (mass_over_sum_pt − 0.093)
if log_sum_pt < 7.05: z += 2.61 × (7.05 − log_sum_pt)
if girth2_top5 < 0.011 and max_dr < 0.328: z += -583 × (0.011 − girth2_top5) × (0.328 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 25.7% of jets, neuron 0.28, formula right for 76%.  
- **group 2** — 17.7% of jets, neuron 0.15, formula right for 87%.  
- **group 3** — 13.0% of jets, neuron 0.31, formula right for 81%.  
- **group 4** — 12.6% of jets, neuron 0.08, formula right for 87%.  
- **group 5** — 10.4% of jets, neuron 0.47, formula right for 73%.  
- **group 6** — 5.5% of jets, neuron 0.01, formula right for 88%.  
- **group 7** — 5.3% of jets, neuron 0.00, formula right for 91%.  
- **group 8** — 4.7% of jets, neuron 0.10, formula right for 77%.  
- **group 9** — 4.2% of jets, neuron 0.26, formula right for 70%.  
- **group 10** — 1.1% of jets, neuron 0.00, formula right for 74%.  
