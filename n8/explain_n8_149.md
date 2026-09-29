# What each part of the 149-term formula does (8 particles)

*the simpler version of the simplified formula*. Validation accuracy 64.52%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a sum of if-statements on jet quantities, clipped at zero), and each class score adds some scales and subtracts others; the largest score wins. Tops are recognised by size: the t score mostly subtracts the 'narrower than a top' scale (neuron 13, on which tops sit far below every other type) and adds the overall size-and-mass scale (neuron 10, highest for tops), so it behaves like a measure of jet width. Quarks and gluons both sit high on the narrow, light-parton scale (neuron 9), which is the largest input of the q score and a large input of the g score; they are told apart by how the pT is shared, since the g score adds the gluon-like scale (neuron 2: even the 8th-hardest particle is hard) and subtracts the 'pT held by a few particles' scale (neuron 5), on which quarks sit highest, while the q score subtracts overall size (neuron 10). W and Z jets sit high on the compact, elongated two-prong scales (neurons 0, 7 and 11) and both boson scores subtract the broad, off-centre scale (neuron 6) and the wide-angle radiation scale (neuron 3), on which tops and gluons sit higher. W and Z are split mainly by the Z-sized scale (neuron 14: wider and heavier than a W), which the Z score adds and the W score subtracts, while the W score leans most on the compact, centred scale (neuron 11) and the Z score on the two-prong mass-window scale (neuron 7).

## The 5 class scores

### score g: gluon: hard particles spread out, light

High for light jets whose pT is spread over many hard particles rather than held by a few; it averages 2.17 for g jets, 1.49 for q and 1.18 for t, and is near zero for W (0.27) and Z (0.11) (AUC 0.80 for gluons against the rest).

Adds gluon-likeness (neuron 2, +35%) and the narrow light-parton scale (neuron 9, +25%), plus small amounts of neurons 1 and 6; subtracts the 'pT held by a few particles' scale (neuron 5, -15%) and, less, the two-prong scale (neuron 0) and neuron 4.

*computed:* largest for g (2.17), then q (1.49), then t (1.18), then W (0.27), then Z (0.11); it separates g jets from the rest best (AUC 0.80: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets that are small in mass and width; it averages 2.21 for q jets and 1.50 for g, with tops lower (0.37), W near zero (0.03) and Z slightly negative (-0.15) (AUC 0.83 for quarks). Gluons are its main confusion.

Mostly adds the narrow light-parton scale (neuron 9, +49%); subtracts overall size and mass (neuron 10, -18%) and neuron 4 (-16%); adds a little of the broad off-centre scale (neuron 6, +9%) and of neuron 5 (+5%).

*computed:* largest for q (2.21), then g (1.50), then t (0.37), then W (0.03), then Z (-0.15); it separates q jets from the rest best (AUC 0.83: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets that are not wider or heavier than a W; it averages 2.46 for W jets and 0.65 for Z, is near zero for q (0.12), slightly negative for g (-0.25) and strongly negative for t (-2.34) (AUC 0.89).

Adds the compact-centred scale (neuron 11, +25%, its largest input), the two-prong scales (neurons 0 and 7) and 'narrower than a top' (neuron 13); subtracts the broad off-centre scale (neuron 6, -13%), the wider-than-W scales (neurons 15 and 14), wide-angle radiation (neuron 3) and a little of neurons 8 and 9.

*computed:* largest for W (2.46), then Z (0.65), then q (0.12), then g (-0.25), then t (-2.34); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: two-prong jet, wider than a W

High for two-prong jets in the boson mass-to-pT window with an intermediate, Z-sized width; it averages 2.27 for Z jets and 1.58 for W (W is its main confusion), is near zero for q (0.20) and g (-0.05), and strongly negative for t (-1.62) (AUC 0.85).

Adds the two-prong mass-window scale (neuron 7, +28%, its largest input), neuron 4, 'narrower than a top' (neuron 13), the Z-sized scale (neuron 14) and a little of neuron 1; subtracts the broad off-centre scale (neuron 6, -22%), wide-angle radiation (neuron 3) and a little of neurons 9 and 15.

*computed:* largest for Z (2.27), then W (1.58), then q (0.20), then g (-0.05), then t (-1.62); it separates Z jets from the rest best (AUC 0.85: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and girth2 with rank correlations of about 0.91. It averages 2.78 for t jets, 0.41 for Z and 0.16 for W, is near zero for g (-0.07) and negative for q (-1.45) (AUC 0.91).

Subtracts 'narrower than a top' (neuron 13, -47%) and adds overall size and mass (neuron 10, +26%); also subtracts the 'pT held by a few particles' scale (neuron 5, -13%) and adds smaller amounts of neurons 4 and 8.

*computed:* largest for t (2.78), then Z (0.41), then W (0.16), then g (-0.07), then q (-1.45); it separates t jets from the rest best (AUC 0.91: large for t)

## The 16 neurons (most important first)

### neuron 0: massive, elongated two-prong jet (major)

- **What it measures:** Pushed down for light jets (mass < 29.0 GeV and mass < 74.0 GeV are two of its strongest terms) and for very thin jets (width < 0.005), and pushed up for compact jets (girth2 < 0.013); it rises with eccentricity (0.632) and mass (0.555) and falls with τ21 (-0.633). Z jets sit highest (2.20), then W (1.91), with tops in the middle (0.82) and gluons (0.31) and quarks (0.23) lowest; it separates Z jets from the rest best (AUC 0.75).
- *computed — its value:* largest for Z (2.20), then W (1.91), then t (0.82), then g (0.31), then q (0.23); it separates Z jets from the rest best (AUC 0.75: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, it raises the W score (+10%) and lowers the g score (-6%). It does not (or hardly) enter the q, Z or t scores; the Z score takes its two-prong information from neuron 7 instead.
- *computed — used by:* raises the score of W (+10%); lowers the score of g (-6%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.765):

- `` — 24.6% of jets, value 3.07 (1.88…4.25), formula right 74%
- `` — 14.6% of jets, value 1.43 (0.12…2.50), formula right 57%
- `` — 2.0% of jets, value 1.28 (0.62…2.00), formula right 74%
- `` — 4.9% of jets, value 0.87 (0.00…2.38), formula right 50%
- `` — 12.6% of jets, value 0.38 (0.00…1.12), formula right 83%
- `` — 2.9% of jets, value 0.37 (0.00…1.00), formula right 67%
- `` — 2.2% of jets, value 0.34 (0.00…1.00), formula right 48%
- `` — 36.2% of jets, value 0.00 (0.00…0.00), formula right 58%

```
z = -0.139
if mass < 29.00: z += -0.375 × (29.00 − mass)
if girth2 < 0.013: z += 262 × (0.013 − girth2)
if mass < 74.00: z += -0.052 × (74.00 − mass)
if width < 0.005: z += -743 × (0.005 − width)
if centroid_offset < 0.033: z += 59.60 × (0.033 − centroid_offset)
if mass < 65.00 and centroid_offset > 0.008: z += 2.47 × (65.00 − mass) × (centroid_offset − 0.008)
if sum_pt > 880: z += -0.033 × (sum_pt − 880)
if girth2 < 0.018 and eccentricity > 0.970: z += 4970 × (0.018 − girth2) × (eccentricity − 0.970)
if mass < 66.00 and pt_7 < 39.00: z += -0.0014 × (66.00 − mass) × (39.00 − pt_7)
if sum_pt > 860 and pt_7 < 33.00: z += 0.0015 × (sum_pt − 860) × (33.00 − pt_7)
if lam1 < 0.0062 and D2 < 0.950: z += -1980 × (0.0062 − lam1) × (0.950 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.4% of jets, neuron 2.81, formula right for 72%.  
- **group 2** — 18.1% of jets, neuron 0.48, formula right for 78%.  
- **group 3** — 17.2% of jets, neuron 0.00, formula right for 59%.  
- **group 4** — 14.7% of jets, neuron 1.06, formula right for 50%.  
- **group 5** — 8.1% of jets, neuron 0.00, formula right for 53%.  
- **group 6** — 4.4% of jets, neuron 0.00, formula right for 72%.  
- **group 7** — 4.2% of jets, neuron 0.07, formula right for 42%.  
- **group 8** — 3.2% of jets, neuron 1.71, formula right for 65%.  
- **group 9** — 1.0% of jets, neuron 0.10, formula right for 77%.  
- **group 10** — 0.6% of jets, neuron 0.00, formula right for 60%.  

### neuron 2: gluon-likeness: hard 8th particle, light jet (major)

- **What it measures:** Pushed up when even the 8th-hardest particle is hard (pT_7 > 28.0 GeV, its strongest term), for total pT below 780 GeV and for a small major axis (lam1 < 0.005), and pushed down for large angularity (LHA > 0.13); overall it falls with jet mass (rank correlation -0.725) and with the largest particle distance from the axis (-0.612). Gluons sit highest (3.76), then quarks (2.88), with W (2.07), Z (1.79) and tops (1.67) lower; it separates gluons from the rest best (AUC 0.79).
- *computed — its value:* largest for g (3.76), then q (2.88), then W (2.07), then Z (1.79), then t (1.67); it separates g jets from the rest best (AUC 0.79: large for g)
- **How the class scores use it:** Since gluons sit highest on it, it raises the g score, where it is the largest input (+35%). It does not enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+35%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.727):

- `` — 9.0% of jets, value 5.00 (4.00…5.88), formula right 57%
- `` — 13.2% of jets, value 4.10 (3.06…5.00), formula right 51%
- `` — 5.4% of jets, value 3.64 (2.44…4.94), formula right 52%
- `` — 11.1% of jets, value 2.93 (2.31…3.56), formula right 61%
- `` — 27.5% of jets, value 2.04 (1.00…3.19), formula right 71%
- `` — 10.7% of jets, value 1.77 (0.00…3.00), formula right 58%
- `` — 21.1% of jets, value 0.75 (0.00…1.56), formula right 76%
- `` — 2.0% of jets, value 0.52 (0.00…1.50), formula right 80%

```
z = 1.33
if pt_7 > 28.00: z += 0.224 × (pt_7 − 28.00)
if LHA > 0.130: z += -7.67 × (LHA − 0.130)
if sum_pt < 780: z += 0.007 × (780 − sum_pt)
if pt_7 > 28.00 and C2 < 0.054: z += -2.23 × (pt_7 − 28.00) × (0.054 − C2)
if lam1 < 0.005: z += 317 × (0.005 − lam1)
if z_7 > 0.047: z += -42.80 × (z_7 − 0.047)
if pt_7 > 28.00 and max_dr > 0.070: z += -0.584 × (pt_7 − 28.00) × (max_dr − 0.070)
if log_sum_pt < 6.40: z += 4.30 × (6.40 − log_sum_pt)
if z_7 < 0.018: z += -429 × (0.018 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 19.2% of jets, neuron 0.79, formula right for 71%.  
- **group 2** — 17.3% of jets, neuron 2.74, formula right for 58%.  
- **group 3** — 13.7% of jets, neuron 1.51, formula right for 72%.  
- **group 4** — 9.9% of jets, neuron 3.04, formula right for 66%.  
- **group 5** — 9.8% of jets, neuron 4.11, formula right for 50%.  
- **group 6** — 8.5% of jets, neuron 2.10, formula right for 71%.  
- **group 7** — 8.4% of jets, neuron 2.71, formula right for 66%.  
- **group 8** — 6.8% of jets, neuron 4.83, formula right for 53%.  
- **group 9** — 3.4% of jets, neuron 4.62, formula right for 63%.  
- **group 10** — 2.9% of jets, neuron 0.09, formula right for 76%.  

### neuron 6: broad, off-centre jet (major)

- **What it measures:** Switched off for narrow jets (width < 0.012, its strongest term, pushes it down) and grows with the largest particle distance from the axis; it follows the offset of the pT centroid from the jet axis and the jet width (rank correlations 0.575 and 0.548). Tops sit far highest (4.74), then gluons (1.79) and quarks (0.85), with Z (0.61) and W (0.23) lowest; it separates tops from the rest best (AUC 0.83).
- *computed — its value:* largest for t (4.74), then g (1.79), then q (0.85), then Z (0.61), then W (0.23); it separates t jets from the rest best (AUC 0.83: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so it lowers the W score (-13%) and the Z score (-22%); among the non-top jets a broad spread looks more like a quark or gluon, so it raises the g score (+6%) and the q score (+9%). It does not enter the t score, although tops sit highest on it; the t score gets its width information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+6%), q (+9%); lowers the score of W (-13%), Z (-22%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.717):

- `` — 6.2% of jets, value 8.67 (4.38…13.12), formula right 74%
- `` — 9.4% of jets, value 6.06 (2.50…9.62), formula right 73%
- `` — 2.0% of jets, value 4.06 (1.62…7.25), formula right 46%
- `` — 3.5% of jets, value 2.21 (0.00…6.00), formula right 94%
- `` — 5.0% of jets, value 2.03 (0.25…3.88), formula right 44%
- `` — 3.8% of jets, value 1.82 (0.00…4.75), formula right 57%
- `` — 70.2% of jets, value 0.29 (0.00…1.00), formula right 64%

```
z = 9.45
if width < 0.012: z += -991 × (0.012 − width)
if mass_over_sum_pt > 0.0089: z += -70.50 × (mass_over_sum_pt − 0.0089)
z += 12.90 × max_dr
if lam2 < 0.00066: z += -2250 × (0.00066 − lam2)
if mass < 42.00 and z_dr_0p05_0p1 < 0.940: z += 0.088 × (42.00 − mass) × (0.940 − z_dr_0p05_0p1)
if girth > 0.090: z += 124 × (girth − 0.090)
if mass_over_sum_pt > 0.015 and tau32 < 0.550: z += -67.30 × (mass_over_sum_pt − 0.015) × (0.550 − tau32)
if centroid_offset > 0.0059 and lam2 < 0.0026: z += 17000 × (centroid_offset − 0.0059) × (0.0026 − lam2)
if lam2 > 0.0027: z += -1280 × (lam2 − 0.0027)
if log_sum_pt < 6.50 and pt_7 < 44.00: z += 0.267 × (6.50 − log_sum_pt) × (44.00 − pt_7)
if log_sum_pt < 6.70 and z_7 < 0.049: z += 838 × (6.70 − log_sum_pt) × (0.049 − z_7)
if e2 < 0.049 and z_dr_0p1_0p2 > 0.180: z += 744 × (0.049 − e2) × (z_dr_0p1_0p2 − 0.180)
if log_sum_pt < 6.30 and z_7 < 0.071: z += 1250 × (6.30 − log_sum_pt) × (0.071 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.9% of jets, neuron 0.42, formula right for 59%.  
- **group 2** — 20.0% of jets, neuron 0.46, formula right for 65%.  
- **group 3** — 17.6% of jets, neuron 1.06, formula right for 71%.  
- **group 4** — 12.2% of jets, neuron 0.62, formula right for 49%.  
- **group 5** — 6.2% of jets, neuron 6.14, formula right for 70%.  
- **group 6** — 4.2% of jets, neuron 8.81, formula right for 75%.  
- **group 7** — 3.1% of jets, neuron 3.94, formula right for 94%.  
- **group 8** — 2.8% of jets, neuron 8.85, formula right for 76%.  
- **group 9** — 1.8% of jets, neuron 1.18, formula right for 95%.  
- **group 10** — 0.3% of jets, neuron 8.00, formula right for 65%.  

### neuron 7: two-prong jet in boson mass window (major)

- **What it measures:** Pushed up for mass/pT above 0.073 but down again above 0.09 (a window where boosted W and Z jets sit), pushed up for small e2 (< 0.038) and down for very thin jets (width < 0.0056); it rises with eccentricity (0.563) and falls with planar flow (-0.563) and τ21 (-0.539), i.e. it likes elongated two-prong jets. Z jets sit highest (3.42), then W (2.31), with tops (1.16), gluons (0.89) and quarks (0.45) low; it separates Z jets from the rest best (AUC 0.81).
- *computed — its value:* largest for Z (3.42), then W (2.31), then t (1.16), then g (0.89), then q (0.45); it separates Z jets from the rest best (AUC 0.81: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the Z score, where it is the largest input (+28%), and raises the W score (+9%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+9%), Z (+28%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.828):

- `` — 24.4% of jets, value 4.15 (2.75…5.75), formula right 70%
- `` — 2.0% of jets, value 2.58 (1.50…3.75), formula right 41%
- `` — 18.7% of jets, value 2.40 (1.25…3.50), formula right 62%
- `` — 4.5% of jets, value 1.45 (0.62…2.25), formula right 42%
- `` — 2.1% of jets, value 1.11 (0.00…2.38), formula right 70%
- `` — 2.0% of jets, value 1.00 (0.00…2.00), formula right 50%
- `` — 32.6% of jets, value 0.10 (0.00…0.38), formula right 60%
- `` — 13.7% of jets, value 0.02 (0.00…0.00), formula right 83%

```
z = 3.96
if width < 0.0056: z += -1020 × (0.0056 − width)
if e2 < 0.038: z += 132 × (0.038 − e2)
if mass_over_sum_pt > 0.073: z += 150 × (mass_over_sum_pt − 0.073)
if mass_over_sum_pt > 0.090: z += -202 × (mass_over_sum_pt − 0.090)
if girth < 0.087: z += -44.10 × (0.087 − girth)
if girth2 > 0.0046: z += -442 × (girth2 − 0.0046)
if girth2 > 0.0047 and eccentricity > 0.940: z += 11800 × (girth2 − 0.0047) × (eccentricity − 0.940)
if planar_flow < 0.220 and width > 0.0078: z += -4770 × (0.220 − planar_flow) × (width − 0.0078)
if planar_flow < 0.190: z += -7.33 × (0.190 − planar_flow)
if e2_sq < 0.0011: z += 1430 × (0.0011 − e2_sq)
if mass > 80.40: z += -0.329 × (mass − 80.40)
if planar_flow < 0.180 and sum_pt > 590: z += 0.036 × (0.180 − planar_flow) × (sum_pt − 590)
if width < 0.00069: z += -2730 × (0.00069 − width)
if pt_7 < 45.00 and planar_flow < 0.600: z += -0.089 × (45.00 − pt_7) × (0.600 − planar_flow)
if centroid_offset > 0.031 and pt_0 > 370: z += -7.69 × (centroid_offset − 0.031) × (pt_0 − 370)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.5% of jets, neuron 0.29, formula right for 59%.  
- **group 2** — 24.6% of jets, neuron 3.22, formula right for 66%.  
- **group 3** — 16.9% of jets, neuron 1.39, formula right for 52%.  
- **group 4** — 11.7% of jets, neuron 4.49, formula right for 72%.  
- **group 5** — 3.5% of jets, neuron 0.00, formula right for 92%.  
- **group 6** — 3.4% of jets, neuron 0.22, formula right for 73%.  
- **group 7** — 3.2% of jets, neuron 0.29, formula right for 82%.  
- **group 8** — 2.2% of jets, neuron 0.00, formula right for 72%.  
- **group 9** — 2.1% of jets, neuron 0.00, formula right for 89%.  
- **group 10** — 0.8% of jets, neuron 0.00, formula right for 66%.  

### neuron 9: narrow, light quark-or-gluon jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.0065, its strongest term) with a centred pT (centroid offset < 0.019), and pushed down for narrow jets whose centroid is off the axis and for high total pT (log of total pT > 6.3); it falls with lam1, width and girth (rank correlations -0.664, -0.653, -0.645). Quarks sit highest (9.02), then gluons (6.65), far above tops (2.43), W (2.34) and Z (1.42), and it is zero for 64% of Z jets; it separates quarks from the rest best (AUC 0.79).
- *computed — its value:* largest for q (9.02), then g (6.65), then t (2.43), then W (2.34), then Z (1.42); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** High values mean a light-parton jet, so it raises the q score (its largest input, +49%) and the g score (+25%), and lowers the W score (-3%) and the Z score (-5%) slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+25%), q (+49%); lowers the score of W (-3%), Z (-5%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.747):

- `` — 21.1% of jets, value 11.59 (8.25…14.50), formula right 65%
- `` — 4.2% of jets, value 9.23 (6.25…12.50), formula right 57%
- `` — 3.5% of jets, value 7.91 (4.00…13.00), formula right 94%
- `` — 2.2% of jets, value 6.85 (3.00…10.25), formula right 54%
- `` — 14.0% of jets, value 3.92 (0.00…9.25), formula right 46%
- `` — 4.6% of jets, value 2.92 (0.00…6.25), formula right 40%
- `` — 50.4% of jets, value 0.78 (0.00…2.75), formula right 71%

```
z = -1.18
if width < 0.0065: z += 2350 × (0.0065 − width)
if width < 0.0061 and centroid_offset > 0.0016: z += -69600 × (0.0061 − width) × (centroid_offset − 0.0016)
if log_sum_pt > 6.30: z += -5.02 × (log_sum_pt − 6.30)
if centroid_offset < 0.019: z += 129 × (0.019 − centroid_offset)
if mass < 56.00 and log_sum_pt < 6.80: z += 0.172 × (56.00 − mass) × (6.80 − log_sum_pt)
if mass < 25.00: z += -0.101 × (25.00 − mass)
if lam2 > 0.0015: z += 1460 × (lam2 − 0.0015)
if width < 0.00029: z += 7350 × (0.00029 − width)
if n_dr_0p2_0p4 > 1.50: z += 1.20 × (n_dr_0p2_0p4 − 1.50)
if width < 0.0091 and C2 > 0.026: z += -7340 × (0.0091 − width) × (C2 − 0.026)
if girth2 > 0.019 and pt_7 < 30.00: z += 57.90 × (girth2 − 0.019) × (30.00 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 35.0% of jets, neuron 0.43, formula right for 71%.  
- **group 2** — 14.5% of jets, neuron 12.41, formula right for 67%.  
- **group 3** — 12.8% of jets, neuron 1.56, formula right for 66%.  
- **group 4** — 9.4% of jets, neuron 3.14, formula right for 52%.  
- **group 5** — 6.9% of jets, neuron 8.65, formula right for 55%.  
- **group 6** — 6.3% of jets, neuron 4.42, formula right for 44%.  
- **group 7** — 5.9% of jets, neuron 10.29, formula right for 62%.  
- **group 8** — 4.1% of jets, neuron 1.88, formula right for 42%.  
- **group 9** — 3.6% of jets, neuron 5.10, formula right for 91%.  
- **group 10** — 1.5% of jets, neuron 9.97, formula right for 95%.  

### neuron 10: overall jet size and mass (major)

- **What it measures:** Pushed down for small e2 (< 0.037) and up for large e2 (> 0.037), and pushed up for elongated jets with little pT at 0.2-0.4 in ΔR and for a small pT share at 0.05-0.1 in ΔR; it follows mass, width and e2 very closely (rank correlations 0.836, 0.822, 0.821). It is on for nearly every jet: tops sit highest (5.79), W (3.24) and Z (3.22) in the middle, gluons (2.03) and quarks (1.51) lowest; its clearest separation is that it is small for quarks (AUC 0.19).
- *computed — its value:* largest for t (5.79), then W (3.24), then Z (3.22), then g (2.03), then q (1.51); it separates q jets from the rest best (AUC 0.19: small for q)
- **How the class scores use it:** A large, massive jet marks a top, so it raises the t score (+26%); a small jet marks a quark, so it lowers the q score (-18%). It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+26%); lowers the score of q (-18%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.853):

- `` — 2.4% of jets, value 13.09 (10.44…15.94), formula right 95%
- `` — 3.5% of jets, value 8.75 (6.06…11.19), formula right 88%
- `` — 2.5% of jets, value 6.72 (3.88…9.34), formula right 79%
- `` — 59.1% of jets, value 3.43 (2.38…4.31), formula right 64%
- `` — 8.5% of jets, value 1.96 (1.31…2.62), formula right 52%
- `` — 24.0% of jets, value 0.77 (0.44…1.31), formula right 62%

```
z = 2.67
if e2 < 0.037: z += -41.90 × (0.037 − e2)
if eccentricity > 0.900 and z_dr_0p2_0p4 < 0.041: z += 337 × (eccentricity − 0.900) × (0.041 − z_dr_0p2_0p4)
if e2 > 0.037: z += 79.20 × (e2 − 0.037)
if z_dr_0p05_0p1 < 0.340: z += 2.28 × (0.340 − z_dr_0p05_0p1)
if girth2 < 0.0018: z += -865 × (0.0018 − girth2)
if LHA > 0.290 and tau21 < 0.690: z += -40.20 × (LHA − 0.290) × (0.690 − tau21)
if lam2 > 0.00016: z += 532 × (lam2 − 0.00016)
if log_sum_pt < 6.30: z += -4.67 × (6.30 − log_sum_pt)
if lam2 > 0.00022 and tau21 < 0.520: z += 2380 × (lam2 − 0.00022) × (0.520 − tau21)
if C2 > 0.059: z += 31.50 × (C2 − 0.059)
if n_dr_0p2_0p4 > 1.50: z += 0.599 × (n_dr_0p2_0p4 − 1.50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 28.7% of jets, neuron 0.92, formula right for 61%.  
- **group 2** — 22.0% of jets, neuron 3.65, formula right for 72%.  
- **group 3** — 15.4% of jets, neuron 3.43, formula right for 57%.  
- **group 4** — 12.9% of jets, neuron 2.73, formula right for 54%.  
- **group 5** — 6.3% of jets, neuron 3.11, formula right for 60%.  
- **group 6** — 4.9% of jets, neuron 3.49, formula right for 73%.  
- **group 7** — 3.4% of jets, neuron 8.19, formula right for 90%.  
- **group 8** — 2.6% of jets, neuron 5.00, formula right for 72%.  
- **group 9** — 2.0% of jets, neuron 10.04, formula right for 92%.  
- **group 10** — 1.8% of jets, neuron 13.39, formula right for 95%.  

### neuron 13: narrower than a top (major)

- **What it measures:** Driven mostly by girth < 0.14 (its dominant term, pushing it up), i.e. by the jet being narrower than a typical top, with small e2 (< 0.047) pulling it down a little; it falls with width, girth2, lam1 and girth (rank correlations -0.848 to -0.839). Quarks (7.20), W (6.07), gluons (5.95) and Z (5.57) all sit high, tops far lower (2.05); it separates tops from the rest best, as small for tops (AUC 0.09).
- *computed — its value:* largest for q (7.20), then W (6.07), then g (5.95), then Z (5.57), then t (2.05); it separates t jets from the rest best (AUC 0.09: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-47%); it also raises the W score (+10%) and the Z score (+11%). It does not enter the g or q scores.
- *computed — used by:* raises the score of W (+10%), Z (+11%); lowers the score of t (-47%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.805):

- `` — 24.3% of jets, value 8.37 (6.75…9.75), formula right 61%
- `` — 4.9% of jets, value 6.37 (4.50…7.75), formula right 62%
- `` — 36.8% of jets, value 5.64 (4.38…7.12), formula right 67%
- `` — 16.8% of jets, value 4.50 (2.75…6.00), formula right 52%
- `` — 3.5% of jets, value 2.94 (1.62…4.38), formula right 67%
- `` — 2.3% of jets, value 1.22 (0.00…2.62), formula right 78%
- `` — 2.5% of jets, value 1.09 (0.38…2.00), formula right 78%
- `` — 8.9% of jets, value 0.25 (0.00…0.88), formula right 84%

```
z = 2.17
if girth < 0.140: z += 83.10 × (0.140 − girth)
if e2 < 0.047: z += -82.40 × (0.047 − e2)
if girth < 0.150 and pt_7 < 35.00: z += -1.72 × (0.150 − girth) × (35.00 − pt_7)
if tau21 < 0.520 and max_dr > 0.031: z += -24.90 × (0.520 − tau21) × (max_dr − 0.031)
if girth < 0.150 and log_sum_pt < 6.70: z += -48.00 × (0.150 − girth) × (6.70 − log_sum_pt)
if sum_pt_top5 > 660 and pt_7 < 48.00: z += 0.00048 × (sum_pt_top5 − 660) × (48.00 − pt_7)
if C2 > 0.065: z += -66.10 × (C2 − 0.065)
if sum_pt > 980: z += -0.017 × (sum_pt − 980)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 19.2% of jets, neuron 5.09, formula right for 66%.  
- **group 2** — 16.7% of jets, neuron 8.55, formula right for 54%.  
- **group 3** — 13.3% of jets, neuron 4.14, formula right for 72%.  
- **group 4** — 12.8% of jets, neuron 5.65, formula right for 52%.  
- **group 5** — 8.3% of jets, neuron 5.66, formula right for 55%.  
- **group 6** — 8.1% of jets, neuron 0.82, formula right for 76%.  
- **group 7** — 7.6% of jets, neuron 7.41, formula right for 64%.  
- **group 8** — 5.4% of jets, neuron 0.21, formula right for 93%.  
- **group 9** — 4.6% of jets, neuron 5.59, formula right for 71%.  
- **group 10** — 3.9% of jets, neuron 7.88, formula right for 75%.  

### neuron 14: Z-sized: wider and heavier than W (major)

- **What it measures:** Pushed up for girth2 < 0.014 and small e2 (< 0.038) but pushed down for the narrowest jets (width < 0.0075, girth < 0.091) and for mass < 80.4 GeV, so it peaks for jets of intermediate width that are heavier than a W; it rises with the pT share and number of particles at 0.1-0.2 in ΔR and with mass (rank correlations 0.366, 0.364, 0.36). Z jets sit highest (1.42), well above tops (0.46), gluons (0.22), W (0.19) and quarks (0.16); it separates Z jets from the rest best (AUC 0.77).
- *computed — its value:* largest for Z (1.42), then t (0.46), then g (0.22), then W (0.19), then q (0.16); it separates Z jets from the rest best (AUC 0.77: large for Z)
- **How the class scores use it:** It is the main W-versus-Z separator: Z jets sit high and W jets low on it, so it raises the Z score (+7%) and lowers the W score (-9%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+7%); lowers the score of W (-9%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.656):

- `` — 12.5% of jets, value 2.36 (0.88…3.62), formula right 71%
- `` — 5.9% of jets, value 1.16 (0.00…2.38), formula right 65%
- `` — 9.3% of jets, value 0.91 (0.00…2.50), formula right 67%
- `` — 2.0% of jets, value 0.40 (0.00…1.25), formula right 69%
- `` — 5.0% of jets, value 0.27 (0.00…1.00), formula right 62%
- `` — 4.0% of jets, value 0.26 (0.00…0.94), formula right 58%
- `` — 14.1% of jets, value 0.05 (0.00…0.19), formula right 82%
- `` — 47.2% of jets, value 0.01 (0.00…0.00), formula right 58%

```
z = -0.051
if girth2 < 0.014: z += 741 × (0.014 − girth2)
if width < 0.0075: z += -1170 × (0.0075 − width)
if girth < 0.091: z += -88.80 × (0.091 − girth)
if mass < 80.40: z += -0.051 × (80.40 − mass)
if e2 < 0.038: z += 116 × (0.038 − e2)
if max_dr < 0.180: z += -20.60 × (0.180 − max_dr)
if z_dr_0p05_0p1 < 0.630 and C2 < 0.067: z += -73.20 × (0.630 − z_dr_0p05_0p1) × (0.067 − C2)
if z_dr_0p05_0p1 < 0.640: z += 1.68 × (0.640 − z_dr_0p05_0p1)
if width < 0.0083 and log_sum_pt < 6.90: z += 572 × (0.0083 − width) × (6.90 − log_sum_pt)
if girth2 < 0.014 and eccentricity > 0.970: z += 8210 × (0.014 − girth2) × (eccentricity − 0.970)
if planar_flow < 0.120 and centroid_offset > 0.011: z += 1070 × (0.120 − planar_flow) × (centroid_offset − 0.011)
if planar_flow < 0.120 and max_dr < 0.170: z += -194 × (0.120 − planar_flow) × (0.170 − max_dr)
if width < 0.0078 and planar_flow < 0.097: z += -4740 × (0.0078 − width) × (0.097 − planar_flow)
if planar_flow < 0.120 and centroid_offset > 0.018: z += -1220 × (0.120 − planar_flow) × (centroid_offset − 0.018)
if width < 0.0087 and D2 < 1.00: z += -454 × (0.0087 − width) × (1.00 − D2)
if lam1 > 0.0075: z += -61.60 × (lam1 − 0.0075)
if lam1 > 0.0064 and max_dr < 0.160: z += 16900 × (lam1 − 0.0064) × (0.160 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.4% of jets, neuron 0.00, formula right for 62%.  
- **group 2** — 15.3% of jets, neuron 0.14, formula right for 82%.  
- **group 3** — 12.5% of jets, neuron 0.45, formula right for 69%.  
- **group 4** — 12.0% of jets, neuron 2.35, formula right for 71%.  
- **group 5** — 10.4% of jets, neuron 0.00, formula right for 51%.  
- **group 6** — 8.0% of jets, neuron 1.04, formula right for 70%.  
- **group 7** — 6.7% of jets, neuron 0.17, formula right for 49%.  
- **group 8** — 6.5% of jets, neuron 0.38, formula right for 56%.  
- **group 9** — 2.4% of jets, neuron 0.00, formula right for 42%.  
- **group 10** — 2.0% of jets, neuron 0.72, formula right for 62%.  

### neuron 1: wide, massive jet (low for quarks) (moderate)

- **What it measures:** Pushed down for narrow jets (width < 0.01, its strongest term) and pushed up for high-pT jets with a small minor axis (log of total pT > 6.5 with lam2 < 0.0015) and when the 8th-hardest particle is hard (pT_7 > 32.0 GeV); it follows lam1, width and mass (rank correlations 0.644, 0.638, 0.634). Tops sit highest (0.98), then Z (0.87), W (0.52) and gluons (0.49), with quarks clearly lowest (0.21); its clearest separation is that it is small for quarks (AUC 0.26).
- *computed — its value:* largest for t (0.98), then Z (0.87), then W (0.52), then g (0.49), then q (0.21); it separates q jets from the rest best (AUC 0.26: small for q)
- **How the class scores use it:** Gluons sit more than twice as high on it as quarks, so it raises the g score (+8%) as a gluon-versus-quark hint, and it raises the Z score only slightly (+3%). It does not (or hardly) enter the q, W or t scores.
- *computed — used by:* raises the score of g (+8%), Z (+3%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.563):

- `` — 4.0% of jets, value 2.48 (1.62…3.75), formula right 77%
- `` — 3.5% of jets, value 1.78 (0.50…3.12), formula right 56%
- `` — 3.8% of jets, value 1.52 (0.88…2.38), formula right 74%
- `` — 7.3% of jets, value 1.21 (0.62…1.75), formula right 81%
- `` — 3.5% of jets, value 1.08 (0.38…1.88), formula right 46%
- `` — 18.8% of jets, value 0.81 (0.38…1.50), formula right 79%
- `` — 12.4% of jets, value 0.54 (0.00…1.50), formula right 51%
- `` — 46.7% of jets, value 0.12 (0.00…0.50), formula right 60%

```
z = 0.550
if width < 0.010: z += -259 × (0.010 − width)
if log_sum_pt > 6.50 and lam2 < 0.0015: z += 6760 × (log_sum_pt − 6.50) × (0.0015 − lam2)
if pt_7 > 32.00: z += 0.123 × (pt_7 − 32.00)
if log_sum_pt > 6.50 and centroid_offset < 0.028: z += -238 × (log_sum_pt − 6.50) × (0.028 − centroid_offset)
if pt_7 > 34.00 and mass < 98.00: z += -0.0017 × (pt_7 − 34.00) × (98.00 − mass)
if lam1 < 0.012 and centroid_offset > 0.018: z += 10900 × (0.012 − lam1) × (centroid_offset − 0.018)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 20.8% of jets, neuron 0.70, formula right for 74%.  
- **group 2** — 16.3% of jets, neuron 0.05, formula right for 52%.  
- **group 3** — 11.7% of jets, neuron 1.09, formula right for 70%.  
- **group 4** — 11.3% of jets, neuron 1.04, formula right for 78%.  
- **group 5** — 10.8% of jets, neuron 0.29, formula right for 60%.  
- **group 6** — 9.7% of jets, neuron 0.43, formula right for 71%.  
- **group 7** — 7.0% of jets, neuron 0.23, formula right for 56%.  
- **group 8** — 5.3% of jets, neuron 1.31, formula right for 45%.  
- **group 9** — 5.1% of jets, neuron 0.96, formula right for 60%.  
- **group 10** — 1.8% of jets, neuron 0.48, formula right for 63%.  

### neuron 3: radiation far from the jet axis (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.01, its strongest term, pushes it down) and pushed up for small e2 (< 0.047) and large angularity (LHA > 0.27); it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.573 and 0.572). Tops sit clearly highest (1.29), then gluons (0.46), quarks (0.23) and Z (0.16), while W jets are almost never on it (0.02, zero for 97% of them); it separates tops from the rest best (AUC 0.73).
- *computed — its value:* largest for t (1.29), then g (0.46), then q (0.23), then Z (0.16), then W (0.02); it separates t jets from the rest best (AUC 0.73: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so it lowers the W score (-5%) and the Z score (-9%). It does not (or hardly) enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-5%), Z (-9%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.711):

- `` — 2.0% of jets, value 4.55 (2.25…7.38), formula right 68%
- `` — 3.1% of jets, value 2.70 (0.75…4.62), formula right 62%
- `` — 7.3% of jets, value 2.17 (0.75…3.25), formula right 68%
- `` — 2.2% of jets, value 1.10 (0.00…2.88), formula right 85%
- `` — 2.4% of jets, value 0.59 (0.00…1.75), formula right 42%
- `` — 6.9% of jets, value 0.33 (0.00…1.25), formula right 86%
- `` — 76.1% of jets, value 0.05 (0.00…0.00), formula right 63%

```
z = 0.508
if girth2 < 0.010: z += -1040 × (0.010 − girth2)
if e2 < 0.047: z += 188 × (0.047 − e2)
if LHA > 0.270: z += 19.80 × (LHA − 0.270)
if mass_over_sum_pt > 0.071 and tau32 < 0.550: z += -109 × (mass_over_sum_pt − 0.071) × (0.550 − tau32)
if centroid_offset > 0.013: z += 22.50 × (centroid_offset − 0.013)
if lam1 > 0.016: z += -189 × (lam1 − 0.016)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.9% of jets, neuron 0.01, formula right for 60%.  
- **group 2** — 12.5% of jets, neuron 0.07, formula right for 71%.  
- **group 3** — 10.8% of jets, neuron 0.20, formula right for 63%.  
- **group 4** — 10.6% of jets, neuron 0.06, formula right for 51%.  
- **group 5** — 9.8% of jets, neuron 0.21, formula right for 54%.  
- **group 6** — 9.8% of jets, neuron 2.12, formula right for 73%.  
- **group 7** — 8.1% of jets, neuron 0.16, formula right for 75%.  
- **group 8** — 5.7% of jets, neuron 0.17, formula right for 93%.  
- **group 9** — 2.5% of jets, neuron 2.00, formula right for 68%.  
- **group 10** — 2.1% of jets, neuron 4.35, formula right for 56%.  

### neuron 4: low C2, mass below top range (moderate)

- **What it measures:** Pushed up for light jets (mass < 55.0 GeV), for width > 0.0016 and for a two-prong shape (τ21 < 0.27), and pushed down for very thin jets (lam2 < 0.00042) and for large mass/pT (> 0.086); it falls with C2 (rank correlation -0.52) and with the largest particle distance (-0.377). It is on for nearly all g, q, W and Z jets (Z highest at 4.31, then g 4.11, W 3.85, q 3.75) but zero for 36% of tops, which sit lowest (2.84); its clearest separation is that it is small for tops (AUC 0.35).
- *computed — its value:* largest for Z (4.31), then g (4.11), then W (3.85), then q (3.75), then t (2.84); it separates t jets from the rest best (AUC 0.35: small for t)
- **How the class scores use it:** It raises the Z score (+11%) and the t score (+10%), and it lowers the q score (-16%) and the g score (-4%); it does not enter the W score. Z jets sit highest on it, while in the t score it is not a sign of a top (tops sit lowest on it) but a correction next to that score's much larger inputs from neurons 13 and 10.
- *computed — used by:* raises the score of Z (+11%), t (+10%); lowers the score of g (-4%), q (-16%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.495):

- `` — 18.7% of jets, value 5.37 (4.00…6.50), formula right 76%
- `` — 10.9% of jets, value 4.95 (2.25…7.25), formula right 57%
- `` — 56.6% of jets, value 3.66 (1.75…4.75), formula right 59%
- `` — 2.8% of jets, value 3.50 (1.25…6.25), formula right 74%
- `` — 3.1% of jets, value 1.58 (0.00…4.75), formula right 54%
- `` — 2.0% of jets, value 0.42 (0.00…1.50), formula right 81%
- `` — 5.9% of jets, value 0.07 (0.00…0.00), formula right 93%

```
z = 0.821
if mass < 55.00: z += 0.144 × (55.00 − mass)
if lam2 < 0.00042: z += -9010 × (0.00042 − lam2)
if width > 0.0016: z += 427 × (width − 0.0016)
if tau21 < 0.270: z += 29.40 × (0.270 − tau21)
if mass_over_sum_pt > 0.086: z += -184 × (mass_over_sum_pt − 0.086)
if C2 > 0.063: z += -105 × (C2 − 0.063)
if tau21 < 0.260 and mass < 57.00: z += -0.848 × (0.260 − tau21) × (57.00 − mass)
if C2 > -0.00087 and pt_7 > 38.00: z += 4.15 × (C2 − -0.00087) × (pt_7 − 38.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.0% of jets, neuron 4.28, formula right for 59%.  
- **group 2** — 22.8% of jets, neuron 4.95, formula right for 75%.  
- **group 3** — 10.6% of jets, neuron 3.21, formula right for 59%.  
- **group 4** — 10.3% of jets, neuron 3.38, formula right for 51%.  
- **group 5** — 9.0% of jets, neuron 3.50, formula right for 52%.  
- **group 6** — 4.1% of jets, neuron 4.70, formula right for 74%.  
- **group 7** — 4.0% of jets, neuron 0.08, formula right for 94%.  
- **group 8** — 3.5% of jets, neuron 0.68, formula right for 81%.  
- **group 9** — 2.7% of jets, neuron 3.10, formula right for 72%.  
- **group 10** — 2.0% of jets, neuron 0.52, formula right for 80%.  

### neuron 5: pT held by the few hardest particles (moderate)

- **What it measures:** Pushed up when the 8th-hardest particle carries little of the pT (z_7 < 0.066, its strongest term) and for small e2 (< 0.04), and pushed down for jets that also have a centred pT (z_7 < 0.072 with centroid offset < 0.032); it falls with e2 and z_7 (rank correlations -0.634 and -0.626) and rises with the summed pT of the 3 hardest particles (0.613). Quarks sit far highest (4.70); W (2.10), Z (2.07) and gluons (1.95) are similar, and tops lowest (1.08); it separates quarks from the rest best (AUC 0.77).
- *computed — its value:* largest for q (4.70), then W (2.10), then Z (2.07), then g (1.95), then t (1.08); it separates q jets from the rest best (AUC 0.77: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops well below, it raises the q score (+5%) and lowers the g score (-15%) and the t score (-13%). It does not (or hardly) enter the W or Z scores.
- *computed — used by:* raises the score of q (+5%); lowers the score of g (-15%), t (-13%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.746):

- `` — 8.8% of jets, value 8.05 (5.62…10.50), formula right 69%
- `` — 3.5% of jets, value 5.05 (3.00…7.12), formula right 60%
- `` — 2.0% of jets, value 4.68 (0.00…9.25), formula right 70%
- `` — 17.8% of jets, value 3.73 (1.75…6.00), formula right 55%
- `` — 4.2% of jets, value 3.12 (1.25…5.12), formula right 77%
- `` — 17.0% of jets, value 1.53 (0.00…3.25), formula right 52%
- `` — 17.3% of jets, value 1.16 (0.00…2.50), formula right 75%
- `` — 29.4% of jets, value 0.44 (0.00…1.00), formula right 69%

```
z = 0.310
if z_7 < 0.066: z += 91.60 × (0.066 − z_7)
if z_7 < 0.072 and centroid_offset < 0.032: z += -3020 × (0.072 − z_7) × (0.032 − centroid_offset)
if e2 < 0.040: z += 59.60 × (0.040 − e2)
if width < 0.0026 and centroid_offset < 0.021: z += 57000 × (0.0026 − width) × (0.021 − centroid_offset)
if LHA < 0.220 and log_sum_pt < 6.80: z += -132 × (0.220 − LHA) × (6.80 − log_sum_pt)
if z_7 < 0.035: z += 186 × (0.035 − z_7)
if e2 < 0.025 and lam2 < 9.2e-05: z += 874000 × (0.025 − e2) × (9.2e-05 − lam2)
if log_sum_pt > 6.90: z += -71.30 × (log_sum_pt − 6.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 37.3% of jets, neuron 0.74, formula right for 67%.  
- **group 2** — 22.4% of jets, neuron 1.99, formula right for 66%.  
- **group 3** — 10.4% of jets, neuron 2.54, formula right for 50%.  
- **group 4** — 8.4% of jets, neuron 5.14, formula right for 59%.  
- **group 5** — 8.2% of jets, neuron 4.67, formula right for 67%.  
- **group 6** — 5.6% of jets, neuron 8.90, formula right for 78%.  
- **group 7** — 5.5% of jets, neuron 0.68, formula right for 62%.  
- **group 8** — 1.3% of jets, neuron 1.67, formula right for 63%.  
- **group 9** — 0.7% of jets, neuron 0.01, formula right for 60%.  
- **group 10** — 0.2% of jets, neuron 0.00, formula right for 64%.  

### neuron 8: narrow, centred single core (moderate)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.0049, its strongest term) and pushed down when the jet is very thin in girth (girth < 0.063) or narrow but with its pT centroid off the axis; it falls with centroid offset and width (rank correlations -0.447 and -0.409). Quarks (1.43) and gluons (1.24) sit highest, with W (0.44), Z (0.27) and tops (0.22) low; it separates quarks from the rest best (AUC 0.72).
- *computed — its value:* largest for q (1.43), then g (1.24), then W (0.44), then Z (0.27), then t (0.22); it separates q jets from the rest best (AUC 0.72: large for q)
- **How the class scores use it:** A single narrow core is not a two-prong W, so it lowers the W score (-5%); it also raises the t score slightly (+3%), a small correction since tops sit lowest on it. It does not (or hardly) enter the g, q or Z scores, even though quarks and gluons sit highest.
- *computed — used by:* raises the score of t (+3%); lowers the score of W (-5%); does not (or hardly) enter the score of g, q, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.654):

- `` — 13.9% of jets, value 2.99 (1.00…5.25), formula right 58%
- `` — 16.6% of jets, value 1.02 (0.12…2.12), formula right 64%
- `` — 3.3% of jets, value 0.94 (0.00…2.25), formula right 57%
- `` — 3.4% of jets, value 0.74 (0.00…1.75), formula right 47%
- `` — 4.0% of jets, value 0.20 (0.00…0.75), formula right 58%
- `` — 44.6% of jets, value 0.13 (0.12…0.12), formula right 74%
- `` — 8.0% of jets, value 0.07 (0.00…0.25), formula right 43%
- `` — 6.3% of jets, value 0.02 (0.00…0.12), formula right 64%

```
z = 0.139
if width < 0.0049: z += 2310 × (0.0049 − width)
if girth < 0.063: z += -107 × (0.063 − girth)
if width < 0.0051 and centroid_offset > 0.0014: z += -71600 × (0.0051 − width) × (centroid_offset − 0.0014)
if LHA < 0.170 and lam2 < 0.00039: z += -117000 × (0.170 − LHA) × (0.00039 − lam2)
if girth2 < 0.0051 and planar_flow < 0.540: z += 845 × (0.0051 − girth2) × (0.540 − planar_flow)
if mass < 20.00 and centroid_offset > 0.015: z += -9.90 × (20.00 − mass) × (centroid_offset − 0.015)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 50.1% of jets, neuron 0.13, formula right for 73%.  
- **group 2** — 11.1% of jets, neuron 0.99, formula right for 70%.  
- **group 3** — 8.6% of jets, neuron 2.14, formula right for 60%.  
- **group 4** — 7.3% of jets, neuron 0.63, formula right for 57%.  
- **group 5** — 5.5% of jets, neuron 0.92, formula right for 48%.  
- **group 6** — 5.4% of jets, neuron 1.19, formula right for 54%.  
- **group 7** — 4.9% of jets, neuron 3.91, formula right for 54%.  
- **group 8** — 4.7% of jets, neuron 0.12, formula right for 42%.  
- **group 9** — 2.2% of jets, neuron 0.00, formula right for 41%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 58%.  

### neuron 11: moderately compact, centred jet (moderate)

- **What it measures:** Pushed up for width < 0.0088 and a pT centroid near the axis (centroid offset < 0.05), and pushed down for narrow jets (girth < 0.088), so it favours jets that are compact but not a single thin core; it falls with centroid offset (rank correlation -0.396) and planar flow (-0.33). W jets sit highest (4.21), then Z (3.20), gluons (2.40) and quarks (2.28), while tops sit lowest (0.87) and are at zero for 69% of them; its clearest separation is that it is small for tops (AUC 0.16).
- *computed — its value:* largest for W (4.21), then Z (3.20), then g (2.40), then q (2.28), then t (0.87); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** Since W jets sit highest on it, it raises the W score, where it is the largest input (+25%). It does not enter the g, q, Z or t scores; the Z score does not use it although Z jets sit second on it.
- *computed — used by:* raises the score of W (+25%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.727):

- `` — 32.1% of jets, value 4.47 (2.94…6.06), formula right 67%
- `` — 27.8% of jets, value 2.70 (1.12…4.19), formula right 53%
- `` — 10.0% of jets, value 2.31 (1.75…3.00), formula right 60%
- `` — 9.9% of jets, value 1.45 (1.00…1.88), formula right 70%
- `` — 2.2% of jets, value 1.03 (0.00…2.69), formula right 60%
- `` — 2.4% of jets, value 0.41 (0.00…1.62), formula right 76%
- `` — 15.5% of jets, value 0.02 (0.00…0.00), formula right 79%

```
z = -2.21
if width < 0.0088: z += 1350 × (0.0088 − width)
if girth < 0.088: z += -127 × (0.088 − girth)
if centroid_offset < 0.050: z += 136 × (0.050 − centroid_offset)
if width < 0.0038: z += -668 × (0.0038 − width)
if girth > 0.074: z += -51.60 × (girth − 0.074)
if planar_flow < 0.390 and width > 0.0078: z += -959 × (0.390 − planar_flow) × (width − 0.0078)
if LHA < 0.150: z += -22.10 × (0.150 − LHA)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 19.6% of jets, neuron 1.92, formula right for 66%.  
- **group 2** — 14.5% of jets, neuron 4.74, formula right for 73%.  
- **group 3** — 13.4% of jets, neuron 3.29, formula right for 51%.  
- **group 4** — 10.8% of jets, neuron 2.98, formula right for 73%.  
- **group 5** — 10.3% of jets, neuron 4.19, formula right for 59%.  
- **group 6** — 8.4% of jets, neuron 0.01, formula right for 78%.  
- **group 7** — 7.2% of jets, neuron 2.59, formula right for 44%.  
- **group 8** — 6.7% of jets, neuron 0.06, formula right for 81%.  
- **group 9** — 5.9% of jets, neuron 2.51, formula right for 53%.  
- **group 10** — 3.1% of jets, neuron 0.00, formula right for 70%.  

### neuron 15: wider-than-W two-prong jet (moderate)

- **What it measures:** Switched off for narrow jets (width < 0.0069, its strongest term) and pushed up for small e2 (< 0.041); its two lam1 terms (< 0.0065 up, < 0.0081 down) partly cancel for small lam1, so overall it rises with lam1 and width (rank correlations 0.482 and 0.477) and falls with τ21 (-0.47). Z jets sit highest (1.22), then tops (0.87), with gluons (0.37), W (0.24) and quarks (0.21) low; it separates Z jets from the rest best (AUC 0.73).
- *computed — its value:* largest for Z (1.22), then t (0.87), then g (0.37), then W (0.24), then q (0.21); it separates Z jets from the rest best (AUC 0.73: large for Z)
- **How the class scores use it:** W jets sit low on it, so it lowers the W score (-10%); it also lowers the Z score, but with a much smaller share (-3%), so on balance it moves wider two-prong jets away from W more than from Z. It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-10%), Z (-3%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.681):

- `` — 16.4% of jets, value 2.35 (1.12…3.38), formula right 73%
- `` — 3.5% of jets, value 1.15 (0.12…2.50), formula right 59%
- `` — 4.8% of jets, value 1.14 (0.00…2.38), formula right 64%
- `` — 2.1% of jets, value 0.64 (0.00…1.50), formula right 79%
- `` — 19.4% of jets, value 0.37 (0.00…1.12), formula right 68%
- `` — 6.3% of jets, value 0.18 (0.00…0.62), formula right 50%
- `` — 9.6% of jets, value 0.03 (0.00…0.00), formula right 82%
- `` — 37.9% of jets, value 0.01 (0.00…0.00), formula right 58%

```
z = 1.79
if width < 0.0069: z += -2050 × (0.0069 − width)
if lam1 < 0.0065: z += 2180 × (0.0065 − lam1)
if lam1 < 0.0081: z += -1210 × (0.0081 − lam1)
if e2 < 0.041: z += 176 × (0.041 − e2)
if tau21 < 0.290 and z_dr_0p2_0p4 < 0.200: z += 32.80 × (0.290 − tau21) × (0.200 − z_dr_0p2_0p4)
if LHA > 0.340: z += -40.20 × (LHA − 0.340)
if tau21 < 0.310 and z_dr_0p05_0p1 < 0.680: z += -6.46 × (0.310 − tau21) × (0.680 − z_dr_0p05_0p1)
if width < 0.0065 and log_sum_pt > 6.90: z += -7460 × (0.0065 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 37.70 × (log_sum_pt − 6.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 25.3% of jets, neuron 0.00, formula right for 61%.  
- **group 2** — 21.5% of jets, neuron 2.01, formula right for 72%.  
- **group 3** — 10.3% of jets, neuron 0.67, formula right for 70%.  
- **group 4** — 9.9% of jets, neuron 0.03, formula right for 82%.  
- **group 5** — 9.0% of jets, neuron 0.01, formula right for 49%.  
- **group 6** — 8.8% of jets, neuron 0.49, formula right for 66%.  
- **group 7** — 7.3% of jets, neuron 0.35, formula right for 57%.  
- **group 8** — 6.4% of jets, neuron 0.13, formula right for 47%.  
- **group 9** — 1.2% of jets, neuron 0.03, formula right for 65%.  
- **group 10** — 0.3% of jets, neuron 0.11, formula right for 69%.  

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs a very wide jet (width > 0.018 pushes it up) and is pushed down strongly for large e2 (> 0.062); it follows the number of particles above 10 GeV (rank correlation 0.84). Only tops reach it with some frequency (non-zero for 17% of them), so tops sit highest (0.23), then gluons (0.09) and quarks (0.03), with W and Z at 0.00.
- *computed — its value:* largest for t (0.23), then g (0.09), then q (0.03), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.58: large for t)
- **How the class scores use it:** It does not (or hardly) enter any of the five class scores, so it has essentially no effect on which type is chosen.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.438):

- `` — 2.1% of jets, value 2.15 (0.00…4.62), formula right 78%
- `` — 2.0% of jets, value 0.93 (0.00…2.25), formula right 83%
- `` — 2.4% of jets, value 0.28 (0.00…1.00), formula right 88%
- `` — 93.5% of jets, value 0.00 (0.00…0.00), formula right 63%

```
z = -1.12
if width > 0.018: z += 310 × (width − 0.018)
if e2 > 0.062: z += -107 × (e2 − 0.062)
if mass > 91.20: z += 0.083 × (mass − 91.20)
if girth2 > 0.019 and lam2 > 0.0016: z += 15500 × (girth2 − 0.019) × (lam2 − 0.0016)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 90.5% of jets, neuron 0.00, formula right for 63%.  
- **group 2** — 2.3% of jets, neuron 0.10, formula right for 81%.  
- **group 3** — 2.2% of jets, neuron 0.26, formula right for 87%.  
- **group 4** — 1.9% of jets, neuron 0.37, formula right for 89%.  
- **group 5** — 1.4% of jets, neuron 0.74, formula right for 85%.  
- **group 6** — 0.5% of jets, neuron 1.32, formula right for 75%.  
- **group 7** — 0.4% of jets, neuron 3.29, formula right for 78%.  
- **group 8** — 0.4% of jets, neuron 2.57, formula right for 94%.  
- **group 9** — 0.4% of jets, neuron 2.70, formula right for 63%.  
- **group 10** — 0.1% of jets, neuron 8.45, formula right for 67%.  
