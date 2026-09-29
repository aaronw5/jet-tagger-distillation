# What each part of the 300-term formula does (8 particles)

*the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula)*. Validation accuracy 65.28%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a clipped sum of if-statements on 24 jet quantities), and each class score adds some scales and subtracts others. Tops are recognised by size: the t score subtracts compactness (neuron 13, on which tops sit far below every other type) and adds the two-direction spread scale (neuron 10), so it follows e2, girth and width with rank correlations near 0.89. Quarks and gluons both sit high on the narrow, light-jet scale (neuron 9), which feeds both of their scores; the g score also adds a light, low-angularity scale on which gluons sit highest (neuron 2) and subtracts the few-hard-particles scale on which quarks sit highest (neuron 5), while the q score subtracts neuron 4 and the spread scale (neuron 10). W and Z jets sit lowest on the wide-angle-radiation and broad-off-centre scales (neurons 3 and 6), which both boson scores subtract, and high on the two-prong mass-window scale (neuron 7) and on compactness (neuron 13), which both boson scores add. W and Z are split mainly by width and centring: the W score's largest input is the centred, energetic scale (neuron 11), which the Z score does not use, and the W score subtracts the intermediate-width and wider-than-W scales (neurons 14 and 15) on which Z jets sit highest, while the Z score adds neuron 14.

## The 5 class scores

### score g: gluon: light, round, one-prong jet

High for light, one-prong jets whose pT is spread roundly rather than along one line (it rises with planar flow and τ21, rank correlations 0.573 and 0.522, and falls with mass, -0.454); it averages 2.19 for g jets, 1.40 for q and 0.94 for t, and is near zero for W (0.14) and Z (-0.03) (AUC 0.82 for gluons against the rest).

Adds the light, low-angularity scale (neuron 2, +30%, its largest input), the narrow-light scale (neuron 9, +26%), the mass-and-spread-at-high-pT scale (neuron 1, +12%) and a little broad spread (neuron 6, +6%); subtracts the few-hard-particles scale (neuron 5, -13%), neuron 4 (-7%) and the two-prong-with-mass scale (neuron 0, -6%).

*computed:* largest for g (2.19), then q (1.40), then t (0.94), then W (0.14), then Z (-0.03); it separates g jets from the rest best (AUC 0.82: large for g)

### score q: quark: narrow, light single-core jet

High for narrow jets: it falls with girth, LHA, lam1 and width (rank correlations -0.66 to -0.665). It averages 2.23 for q jets and 1.49 for g (gluons come next), with tops (0.11) and W (0.03) near zero and Z slightly below (-0.18) (AUC 0.85).

Mostly adds the narrow-light scale (neuron 9, +48%), plus broad spread (neuron 6, +8%), the few-hard-particles scale (neuron 5, +4%) and the narrow-core scale (neuron 8, +3%); subtracts neuron 4 (-26%) and the two-direction spread scale (neuron 10, -9%).

*computed:* largest for q (2.23), then g (1.49), then t (0.11), then W (0.03), then Z (-0.18); it separates q jets from the rest best (AUC 0.85: large for q)

### score W: W: centred, energetic two-prong jet

High for narrow, energetic, well-centred two-prong jets that are not wider than a W; it averages 2.55 for W jets and 0.69 for Z, is near zero for q (0.02), negative for g (-0.48) and strongly negative for t (-3.81) (AUC 0.89). Its clearest single trends are with width (rank correlation -0.353) and total pT (0.352).

Adds the centred, energetic scale (neuron 11, +27%, its largest input), the two-prong-with-mass scale (neuron 0, +8%), compactness (neuron 13, +8%) and the two-prong window scale (neuron 7, +6%); subtracts wide-angle radiation (neuron 3, -14%), broad off-centre spread (neuron 6, -10%), the intermediate-width and wider-than-W scales (neurons 14 and 15, -8% each), the narrow-core scale (neuron 8, -6%) and a little of neuron 9 (-3%).

*computed:* largest for W (2.55), then Z (0.69), then q (0.02), then g (-0.48), then t (-3.81); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: elongated two-prong jet, Z-sized width

High for elongated, energetic two-prong jets of intermediate width: it rises with eccentricity (rank correlation 0.416) and total pT (0.378) and falls with planar flow and τ21. It averages 2.33 for Z jets and 1.59 for W (W jets come next), is near zero for q (0.14), negative for g (-0.30) and strongly negative for t (-3.14) (AUC 0.84).

Adds the two-prong window scale (neuron 7, +19%, its largest positive input), neuron 4 (+15%), compactness (neuron 13, +9%), the intermediate-width scale (neuron 14, +6%) and a little of neuron 1 (+3.5%); subtracts wide-angle radiation (neuron 3, -22%), broad off-centre spread (neuron 6, -18%) and a little of the narrow-light scale (neuron 9, -4%) and the wider-than-W scale (neuron 15, -2.5%).

*computed:* largest for Z (2.33), then W (1.59), then q (0.14), then g (-0.30), then t (-3.14); it separates Z jets from the rest best (AUC 0.84: large for Z)

### score t: top: wide, massive jet

High for wide jets: it follows e2, girth and width with rank correlations of about 0.89 (0.894, 0.89, 0.885). It averages 2.72 for t jets, 0.38 for Z and 0.22 for W, is near zero for g (0.08) and negative for q (-1.10) (AUC 0.90).

Subtracts compactness (neuron 13, -48%, its largest input) and the few-hard-particles scale (neuron 5, -11%); adds neuron 4 (+18%), the two-direction spread scale (neuron 10, +14%) and a little of the narrow-core scale (neuron 8, +5%).

*computed:* largest for t (2.72), then Z (0.38), then W (0.22), then g (0.08), then q (-1.10); it separates t jets from the rest best (AUC 0.90: large for t)

## The 16 neurons (most important first)

### neuron 1: mass and spread at high pT (major)

- **What it measures:** Pushed up at high total pT (log of total pT > 6.4, its strongest term) and when the 8th-hardest particle is hard (pT_7 > 33.5 GeV), and down when every particle lies within ΔR 0.252 of the axis or the 8th particle carries little of the pT (z_7 < 0.0515); it follows mass, lam1, LHA and width (rank correlations about 0.38-0.40). Tops (1.14), Z (1.06) and gluons (1.03) sit highest, W jets in between (0.70) and quarks lowest (0.50), so its clearest separation is that it is small for quarks (AUC 0.35).
- *computed — its value:* largest for t (1.14), then Z (1.06), then g (1.03), then W (0.70), then q (0.50); it separates q jets from the rest best (AUC 0.35: small for q)
- **How the class scores use it:** Gluons sit about twice as high as quarks on it, so the g score reads it as evidence for a gluon rather than a quark and it raises the g score (+12%); it also raises the Z score slightly (+3%). It does not (or hardly) enter the q, W or t scores; the tops and Z jets that also sit high on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+12%), Z (+3%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.413):

- `` — 3.2% of jets, value 3.76 (1.62…6.50), formula right 66%
- `` — 3.3% of jets, value 2.50 (1.25…3.88), formula right 77%
- `` — 3.1% of jets, value 2.11 (1.00…3.38), formula right 65%
- `` — 10.1% of jets, value 1.31 (0.38…2.38), formula right 68%
- `` — 9.4% of jets, value 0.91 (0.00…2.25), formula right 60%
- `` — 29.6% of jets, value 0.85 (0.00…2.12), formula right 70%
- `` — 41.3% of jets, value 0.38 (0.00…1.12), formula right 62%

```
z = 1.09
if log_sum_pt > 6.40: z += 3.28 × (log_sum_pt − 6.40)
if max_dr < 0.252: z += -4.55 × (0.252 − max_dr)
if lam1 < 0.0091 and z_7 > 0.027: z += -5070 × (0.0091 − lam1) × (z_7 − 0.027)
if z_7 < 0.051: z += -63.30 × (0.051 − z_7)
if log_sum_pt > 6.61 and lam2 < 0.0015: z += 4690 × (log_sum_pt − 6.61) × (0.0015 − lam2)
if pt_7 > 33.50: z += 0.093 × (pt_7 − 33.50)
if log_sum_pt > 6.47 and centroid_offset < 0.027: z += -173 × (log_sum_pt − 6.47) × (0.027 − centroid_offset)
if width < 0.0095 and planar_flow < 0.084: z += -5320 × (0.0095 − width) × (0.084 − planar_flow)
if LHA > 0.282: z += 13.90 × (LHA − 0.282)
if e2 > 0.032: z += -41.20 × (e2 − 0.032)
if log_sum_pt > 6.36 and max_dr < 0.195: z += 12.40 × (log_sum_pt − 6.36) × (0.195 − max_dr)
if C2 < 0.039 and tau21 < 0.263: z += -191 × (0.039 − C2) × (0.263 − tau21)
if tau21 < 0.223 and lam2 < 0.00022: z += 29600 × (0.223 − tau21) × (0.00022 − lam2)
if lam1 < 0.0099 and centroid_offset > 0.020: z += 14200 × (0.0099 − lam1) × (centroid_offset − 0.020)
if log_sum_pt > 6.55 and D2 < 1.20: z += 7.17 × (log_sum_pt − 6.55) × (1.20 − D2)
if e2 < 0.0075: z += -161 × (0.0075 − e2)
if e2 < 0.035 and eccentricity > 0.979: z += 3750 × (0.035 − e2) × (eccentricity − 0.979)
if pt_7 > 35.40 and centroid_offset > 0.019: z += 1.56 × (pt_7 − 35.40) × (centroid_offset − 0.019)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 19.7% of jets, neuron 0.68, formula right for 59%.  
- **group 2** — 13.6% of jets, neuron 0.26, formula right for 56%.  
- **group 3** — 12.8% of jets, neuron 1.32, formula right for 83%.  
- **group 4** — 12.5% of jets, neuron 0.90, formula right for 72%.  
- **group 5** — 11.7% of jets, neuron 0.48, formula right for 58%.  
- **group 6** — 9.3% of jets, neuron 0.58, formula right for 72%.  
- **group 7** — 6.3% of jets, neuron 1.72, formula right for 82%.  
- **group 8** — 5.4% of jets, neuron 0.72, formula right for 66%.  
- **group 9** — 4.5% of jets, neuron 1.70, formula right for 44%.  
- **group 10** — 4.3% of jets, neuron 2.50, formula right for 60%.  

### neuron 2: light, one-prong, low-angularity jet (major)

- **What it measures:** Pushed down as the angularity (LHA) grows, its strongest term, and up for total pT below 794 GeV and for thin jets (lam1 < 0.00591); overall it falls with mass (rank correlation -0.683) and rises with τ21 (0.57) and with the number of particles within ΔR 0.05 of the axis (0.565). Gluons sit highest (3.20), then quarks (2.41) and W (1.56), with tops and Z jets lowest (1.29 each).
- *computed — its value:* largest for g (3.20), then q (2.41), then W (1.56), then t (1.29), then Z (1.29); it separates g jets from the rest best (AUC 0.77: large for g)
- **How the class scores use it:** Gluons sit highest on it, so the g score reads it as direct evidence for a gluon: it raises the g score (+30%), its largest input. It does not (or hardly) enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+30%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.723):

- `` — 23.1% of jets, value 3.90 (2.69…5.06), formula right 56%
- `` — 12.8% of jets, value 2.72 (1.38…4.12), formula right 46%
- `` — 5.6% of jets, value 2.16 (1.19…3.19), formula right 68%
- `` — 8.5% of jets, value 2.11 (1.00…3.06), formula right 61%
- `` — 3.6% of jets, value 1.35 (0.44…2.19), formula right 86%
- `` — 13.8% of jets, value 1.20 (0.31…2.19), formula right 66%
- `` — 5.0% of jets, value 0.79 (0.00…1.62), formula right 75%
- `` — 27.7% of jets, value 0.53 (0.00…1.25), formula right 79%

```
z = 5.16
z += -9.84 × LHA
if mass < 36.70: z += -0.125 × (36.70 − mass)
if sum_pt < 794: z += 0.0087 × (794 − sum_pt)
if mass < 38.00 and lam2 < 0.0011: z += 84.60 × (38.00 − mass) × (0.0011 − lam2)
if lam1 < 0.0059: z += 377 × (0.0059 − lam1)
if sum_pt_top5 < 714: z += -0.0053 × (714 − sum_pt_top5)
if pt_7 < 54.00: z += -0.037 × (54.00 − pt_7)
if pt_7 > 31.00: z += 0.098 × (pt_7 − 31.00)
if planar_flow < 0.748: z += -1.21 × (0.748 − planar_flow)
if pt_7 > 31.20 and C2 < 0.050: z += -2.27 × (pt_7 − 31.20) × (0.050 − C2)
if mass < 72.60 and z_7 < 0.064: z += -0.455 × (72.60 − mass) × (0.064 − z_7)
if log_sum_pt < 6.43: z += 3.62 × (6.43 − log_sum_pt)
if pt_7 > 29.50 and max_dr > 0.111: z += -0.513 × (pt_7 − 29.50) × (max_dr − 0.111)
if lam1 < 0.0058 and max_dr > 0.082: z += -2440 × (0.0058 − lam1) × (max_dr − 0.082)
if lam1 < 0.0034 and centroid_offset < 0.0071: z += -30100 × (0.0034 − lam1) × (0.0071 − centroid_offset)
if mass < 73.70 and max_pair_mass > 12.40: z += -0.0016 × (73.70 − mass) × (max_pair_mass − 12.40)
if pt_7 > 28.10 and centroid_offset > 0.013: z += -1.07 × (pt_7 − 28.10) × (centroid_offset − 0.013)
if z_7 < 0.030 and width < 0.0047: z += -11500 × (0.030 − z_7) × (0.0047 − width)
if girth2 < 4.9e-05: z += -34000 × (4.9e-05 − girth2)
if log_sum_pt > 6.93: z += -9.11 × (log_sum_pt − 6.93)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 17.7% of jets, neuron 1.26, formula right for 69%.  
- **group 2** — 17.2% of jets, neuron 0.46, formula right for 77%.  
- **group 3** — 15.0% of jets, neuron 2.25, formula right for 64%.  
- **group 4** — 9.5% of jets, neuron 2.19, formula right for 70%.  
- **group 5** — 8.9% of jets, neuron 1.43, formula right for 71%.  
- **group 6** — 8.2% of jets, neuron 1.19, formula right for 57%.  
- **group 7** — 7.2% of jets, neuron 4.12, formula right for 56%.  
- **group 8** — 6.6% of jets, neuron 4.00, formula right for 57%.  
- **group 9** — 5.5% of jets, neuron 2.65, formula right for 51%.  
- **group 10** — 4.2% of jets, neuron 4.01, formula right for 52%.  

### neuron 3: pT at wide angle (ΔR 0.2-0.4) (major)

- **What it measures:** Switched off for narrow jets (girth < 0.121 and girth2 < 0.00863, its two strongest terms, both push down); it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.671 and 0.664). Tops sit far highest (4.58), then gluons (1.04) and quarks (0.50); Z jets are low (0.21) and W jets almost always at zero (0.02, zero for 98% of them).
- *computed — its value:* largest for t (4.58), then g (1.04), then q (0.50), then Z (0.21), then W (0.02); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** A clean two-prong W or Z has little pT that far from the axis, so the boson scores read this scale as evidence against a boson: it lowers the W score (-14%) and the Z score (-22%). It does not (or hardly) enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-14%), Z (-22%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.639):

- `` — 11.2% of jets, value 7.79 (1.38…14.12), formula right 76%
- `` — 3.7% of jets, value 5.13 (0.00…12.38), formula right 94%
- `` — 2.1% of jets, value 3.35 (0.00…8.38), formula right 75%
- `` — 3.0% of jets, value 1.86 (0.00…5.25), formula right 65%
- `` — 2.9% of jets, value 0.99 (0.00…3.30), formula right 46%
- `` — 77.1% of jets, value 0.07 (0.00…0.00), formula right 63%

```
z = 7.47
if girth < 0.121: z += -144 × (0.121 − girth)
if girth2 < 0.0086: z += -856 × (0.0086 − girth2)
if width < 0.0054: z += 1170 × (0.0054 − width)
if e2 < 0.040: z += 110 × (0.040 − e2)
if e2 > 0.026: z += -140 × (e2 − 0.026)
if width > 0.018: z += 1730 × (width − 0.018)
if LHA > 0.326: z += 105 × (LHA − 0.326)
if lam1 > 0.0088 and pt_7 < 36.30: z += 123 × (lam1 − 0.0088) × (36.30 − pt_7)
if mass > 35.40 and eccentricity > 0.705: z += 0.228 × (mass − 35.40) × (eccentricity − 0.705)
if lam1 > 0.0055 and z_7 < 0.081: z += -13600 × (lam1 − 0.0055) × (0.081 − z_7)
if max_dr > 0.122: z += 18.10 × (max_dr − 0.122)
if LHA > 0.322 and planar_flow > 0.0023: z += -116 × (LHA − 0.322) × (planar_flow − 0.0023)
if LHA > 0.313 and eccentricity > 0.957: z += 2440 × (LHA − 0.313) × (eccentricity − 0.957)
if mass > 75.60: z += -0.295 × (mass − 75.60)
if centroid_offset > 0.011: z += 46.40 × (centroid_offset − 0.011)
if e2 > 0.032 and n_pt_above_50 > 1.68: z += -15.00 × (e2 − 0.032) × (n_pt_above_50 − 1.68)
if LHA > 0.312 and max_dr < 0.149: z += -4360 × (LHA − 0.312) × (0.149 − max_dr)
if C2 > 0.089: z += -194 × (C2 − 0.089)
if lam1 > 0.015 and eccentricity > 0.957: z += 13800 × (lam1 − 0.015) × (eccentricity − 0.957)
if centroid_offset > 0.050: z += -177 × (centroid_offset − 0.050)
if LHA > 0.424 and pt_7 > 38.50: z += -33.40 × (LHA − 0.424) × (pt_7 − 38.50)
if mass > 75.10 and max_dr < 0.185: z += 4.98 × (mass − 75.10) × (0.185 − max_dr)
if mass_over_sum_pt > 0.090 and max_dr < 0.147: z += 9190 × (mass_over_sum_pt − 0.090) × (0.147 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 32.2% of jets, neuron 0.00, formula right for 61%.  
- **group 2** — 24.2% of jets, neuron 0.29, formula right for 68%.  
- **group 3** — 16.9% of jets, neuron 0.07, formula right for 52%.  
- **group 4** — 12.7% of jets, neuron 1.56, formula right for 73%.  
- **group 5** — 6.3% of jets, neuron 7.00, formula right for 78%.  
- **group 6** — 3.1% of jets, neuron 6.50, formula right for 94%.  
- **group 7** — 1.7% of jets, neuron 7.92, formula right for 79%.  
- **group 8** — 1.5% of jets, neuron 7.21, formula right for 91%.  
- **group 9** — 0.9% of jets, neuron 7.98, formula right for 70%.  
- **group 10** — 0.5% of jets, neuron 7.80, formula right for 69%.  

### neuron 4: low C2, no wide-angle particles (major)

- **What it measures:** Pushed up once the jet is not ultra-thin (width > 0.000872, its strongest term), for τ21 < 0.238 and for C2 < 0.0602, and down for mass/pT > 0.0906; it falls with C2 and D2 (rank correlations -0.485 and -0.473) and with the number of particles at 0.2 ≤ ΔR < 0.4. It is on for 96-98% of g, q, W and Z jets, with Z highest (7.14), then g (6.74), W (6.66) and q (5.82), but zero for 35% of tops, which sit lowest (4.62).
- *computed — its value:* largest for Z (7.14), then g (6.74), then W (6.66), then q (5.82), then t (4.62); it separates t jets from the rest best (AUC 0.35: small for t)
- **How the class scores use it:** Z jets sit highest and quarks second-lowest on it, so it raises the Z score (+15%) and lowers the q score (-26%), where it is the largest subtraction; it also lowers the g score (-7%). It raises the t score (+18%) even though tops sit lowest on it: there it is not evidence of a top but a counterweight that partly cancels the much larger subtraction of compactness (neuron 13), which is also high for non-top jets; it does not (or hardly) enter the W score.
- *computed — used by:* raises the score of Z (+15%), t (+18%); lowers the score of g (-7%), q (-26%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.597):

- `` — 10.3% of jets, value 9.76 (7.50…12.25), formula right 56%
- `` — 57.6% of jets, value 7.13 (5.25…9.50), formula right 66%
- `` — 2.1% of jets, value 6.48 (2.75…10.50), formula right 69%
- `` — 18.5% of jets, value 4.59 (1.75…7.25), formula right 62%
- `` — 2.7% of jets, value 2.74 (0.00…7.25), formula right 63%
- `` — 2.0% of jets, value 0.98 (0.00…3.75), formula right 62%
- `` — 6.7% of jets, value 0.10 (0.00…0.00), formula right 92%

```
z = 1.85
if width > 0.00087: z += 447 × (width − 0.00087)
if mass_over_sum_pt > 0.091: z += -284 × (mass_over_sum_pt − 0.091)
if tau21 < 0.238: z += 28.10 × (0.238 − tau21)
if mass < 47.90: z += 0.088 × (47.90 − mass)
if C2 < 0.060: z += 38.10 × (0.060 − C2)
if girth > 0.065: z += -77.10 × (girth − 0.065)
if centroid_offset < 0.043: z += -34.70 × (0.043 − centroid_offset)
if e2 > 0.033: z += 105 × (e2 − 0.033)
if sum_pt < 762: z += 0.0071 × (762 − sum_pt)
if mass_over_sum_pt > 0.109: z += 110 × (mass_over_sum_pt − 0.109)
if C2 > 0.060: z += -141 × (C2 − 0.060)
if tau21 < 0.278 and mass < 63.50: z += -0.599 × (0.278 − tau21) × (63.50 − mass)
if max_dr > 0.110 and eccentricity > 0.982: z += -1520 × (max_dr − 0.110) × (eccentricity − 0.982)
if sum_pt_top5 < 439: z += 0.019 × (439 − sum_pt_top5)
if tau21 < 0.294 and planar_flow > 0.0036: z += 44.40 × (0.294 − tau21) × (planar_flow − 0.0036)
if C2 > 0.0046 and pt_7 > 38.60: z += 4.37 × (C2 − 0.0046) × (pt_7 − 38.60)
if C2 > 0.067 and pt_7 < 36.40: z += -12.30 × (C2 − 0.067) × (36.40 − pt_7)
if tau21 < 0.267 and pt_7 < 26.30: z += -1.24 × (0.267 − tau21) × (26.30 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 36.2% of jets, neuron 6.62, formula right for 59%.  
- **group 2** — 18.7% of jets, neuron 8.79, formula right for 78%.  
- **group 3** — 15.6% of jets, neuron 5.51, formula right for 56%.  
- **group 4** — 12.9% of jets, neuron 6.48, formula right for 59%.  
- **group 5** — 4.6% of jets, neuron 5.84, formula right for 72%.  
- **group 6** — 3.9% of jets, neuron 0.76, formula right for 90%.  
- **group 7** — 3.1% of jets, neuron 4.70, formula right for 74%.  
- **group 8** — 2.7% of jets, neuron 0.19, formula right for 91%.  
- **group 9** — 1.6% of jets, neuron 1.03, formula right for 78%.  
- **group 10** — 0.7% of jets, neuron 0.05, formula right for 68%.  

### neuron 6: broad, off-centre pT spread (major)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and width < 0.0135 both push it down, its two strongest terms) and pushed up for mass/pT < 0.132 and lam1 < 0.00819; it follows the offset of the pT centroid from the jet axis (rank correlation 0.519) and the width (0.428). Tops sit far highest (3.92), then gluons (1.84) and quarks (0.91), with Z (0.61) and W (0.25) lowest.
- *computed — its value:* largest for t (3.92), then g (1.84), then q (0.91), then Z (0.61), then W (0.25); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** W and Z jets sit lowest on it, so the boson scores read a broad, off-centre spread as evidence against a boson: it lowers the W score (-10%) and the Z score (-18%). It raises the g score (+6%) and the q score (+8%), since among non-top jets such a spread is more typical of quarks and gluons; it does not enter the t score, even though tops sit highest on it.
- *computed — used by:* raises the score of g (+6%), q (+8%); lowers the score of W (-10%), Z (-18%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.641):

- `` — 4.5% of jets, value 7.98 (4.00…12.38), formula right 73%
- `` — 10.9% of jets, value 4.93 (2.00…7.62), formula right 73%
- `` — 3.1% of jets, value 4.38 (1.38…8.38), formula right 46%
- `` — 2.1% of jets, value 3.09 (0.38…6.25), formula right 92%
- `` — 3.6% of jets, value 2.20 (0.00…5.53), formula right 54%
- `` — 2.1% of jets, value 0.93 (0.00…3.16), formula right 95%
- `` — 73.7% of jets, value 0.42 (0.00…1.38), formula right 64%

```
z = 0.530
if girth2 < 0.0087: z += -1670 × (0.0087 − girth2)
if width < 0.013: z += -843 × (0.013 − width)
if mass_over_sum_pt < 0.132: z += 76.80 × (0.132 − mass_over_sum_pt)
if lam1 < 0.0082: z += 1120 × (0.0082 − lam1)
if e2 < 0.051: z += 101 × (0.051 − e2)
if max_dr > 0.087: z += 14.00 × (max_dr − 0.087)
if girth2 < 0.0037: z += 572 × (0.0037 − girth2)
if log_sum_pt < 6.62 and pt_7 < 46.10: z += 0.376 × (6.62 − log_sum_pt) × (46.10 − pt_7)
if LHA > 0.313: z += 32.60 × (LHA − 0.313)
if log_sum_pt < 6.72: z += -1.81 × (6.72 − log_sum_pt)
if lam2 > 0.00076: z += -935 × (lam2 − 0.00076)
if centroid_offset > 0.0071 and C2 < 0.097: z += 515 × (centroid_offset − 0.0071) × (0.097 − C2)
if mass_over_sum_pt > 0.0072 and pt_7 < 44.90: z += -0.538 × (mass_over_sum_pt − 0.0072) × (44.90 − pt_7)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.159: z += 1150 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.159)
if C2 > 0.0034 and pt_7 > 31.00: z += 1.59 × (C2 − 0.0034) × (pt_7 − 31.00)
if planar_flow < 0.075: z += -9.51 × (0.075 − planar_flow)
if log_sum_pt < 6.72 and z_7 < 0.050: z += 647 × (6.72 − log_sum_pt) × (0.050 − z_7)
if C2 > 0.042: z += -24.30 × (C2 − 0.042)
if max_dr < 0.045: z += -26.10 × (0.045 − max_dr)
if log_sum_pt < 6.32 and z_7 < 0.071: z += 1040 × (6.32 − log_sum_pt) × (0.071 − z_7)
if log_sum_pt < 6.33: z += 2.72 × (6.33 − log_sum_pt)
if centroid_offset > 0.052: z += 88.10 × (centroid_offset − 0.052)
if sum_pt > 997: z += 0.013 × (sum_pt − 997)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.7% of jets, neuron 0.46, formula right for 62%.  
- **group 2** — 14.2% of jets, neuron 0.58, formula right for 70%.  
- **group 3** — 12.3% of jets, neuron 0.51, formula right for 65%.  
- **group 4** — 10.1% of jets, neuron 5.56, formula right for 78%.  
- **group 5** — 10.1% of jets, neuron 2.14, formula right for 73%.  
- **group 6** — 10.1% of jets, neuron 1.06, formula right for 50%.  
- **group 7** — 9.1% of jets, neuron 0.61, formula right for 53%.  
- **group 8** — 3.7% of jets, neuron 1.55, formula right for 95%.  
- **group 9** — 2.6% of jets, neuron 8.25, formula right for 69%.  
- **group 10** — 0.3% of jets, neuron 7.67, formula right for 65%.  

### neuron 9: narrow, light QCD-like jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00617, its strongest term), for all particles within ΔR 0.194 of the axis and for light, centred jets (mass < 50.9 GeV with centroid offset < 0.0277), and pushed down slightly at high total pT (log of total pT > 6.37); it falls with lam1 and width (rank correlations -0.735 and -0.726). Quarks sit highest (9.07), then gluons (7.03), far above W (2.52), tops (1.72) and Z (1.46).
- *computed — its value:* largest for q (9.07), then g (7.03), then W (2.52), then t (1.72), then Z (1.46); it separates q jets from the rest best (AUC 0.80: large for q)
- **How the class scores use it:** High values mean a light-parton jet, so it raises the q score (+48%, its largest input) and the g score (+26%), and lowers the W score (-3%) and the Z score (-4%) slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+26%), q (+48%); lowers the score of W (-3%), Z (-4%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.847):

- `` — 22.4% of jets, value 11.77 (9.50…13.75), formula right 65%
- `` — 2.0% of jets, value 7.83 (2.75…11.50), formula right 65%
- `` — 5.6% of jets, value 7.48 (2.50…11.75), formula right 50%
- `` — 3.7% of jets, value 6.87 (4.00…10.00), formula right 50%
- `` — 2.9% of jets, value 5.06 (2.50…8.25), formula right 94%
- `` — 2.1% of jets, value 3.41 (1.00…6.00), formula right 42%
- `` — 9.2% of jets, value 2.43 (0.00…6.00), formula right 47%
- `` — 52.1% of jets, value 0.81 (0.00…2.50), formula right 71%

```
z = -2.21
if width < 0.0062: z += 2400 × (0.0062 − width)
if mass < 38.70 and centroid_offset < 0.026: z += -11.10 × (38.70 − mass) × (0.026 − centroid_offset)
if e2 < 0.019 and centroid_offset < 0.024: z += 29100 × (0.019 − e2) × (0.024 − centroid_offset)
if mass < 50.90 and centroid_offset < 0.028: z += 5.67 × (50.90 − mass) × (0.028 − centroid_offset)
if girth < 0.038: z += -183 × (0.038 − girth)
if log_sum_pt > 6.37: z += -6.26 × (log_sum_pt − 6.37)
if max_dr < 0.194: z += 15.00 × (0.194 − max_dr)
if width < 0.0062 and centroid_offset > 0.0033: z += -56500 × (0.0062 − width) × (centroid_offset − 0.0033)
if sum_pt_top5 > 446: z += 0.007 × (sum_pt_top5 − 446)
if mass < 54.30 and log_sum_pt < 6.79: z += 0.150 × (54.30 − mass) × (6.79 − log_sum_pt)
if mass > 40.00: z += 0.042 × (mass − 40.00)
if mass < 57.90 and lam1 < 0.00061: z += -81.50 × (57.90 − mass) × (0.00061 − lam1)
if mass < 60.50 and planar_flow < 0.317: z += 0.131 × (60.50 − mass) × (0.317 − planar_flow)
if e2 < 0.018 and pt_7 < 58.10: z += -2.18 × (0.018 − e2) × (58.10 − pt_7)
if girth2 < 0.004 and centroid_offset > 0.012: z += -54500 × (0.004 − girth2) × (centroid_offset − 0.012)
if lam2 > 0.001: z += 629 × (lam2 − 0.001)
if e2 < 0.033 and tau21 < 0.430: z += -272 × (0.033 − e2) × (0.430 − tau21)
if sum_pt > 863: z += -0.0095 × (sum_pt − 863)
if e2 < 0.020 and eccentricity > 0.907: z += -1530 × (0.020 − e2) × (eccentricity − 0.907)
if girth2 > 0.024: z += 243 × (girth2 − 0.024)
if width < 0.0067 and C2 > 0.031: z += -11300 × (0.0067 − width) × (C2 − 0.031)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 24.2% of jets, neuron 0.56, formula right for 68%.  
- **group 2** — 15.6% of jets, neuron 0.24, formula right for 79%.  
- **group 3** — 11.6% of jets, neuron 11.99, formula right for 71%.  
- **group 4** — 11.0% of jets, neuron 2.08, formula right for 62%.  
- **group 5** — 9.0% of jets, neuron 11.60, formula right for 61%.  
- **group 6** — 7.7% of jets, neuron 3.42, formula right for 52%.  
- **group 7** — 6.5% of jets, neuron 2.43, formula right for 42%.  
- **group 8** — 6.1% of jets, neuron 7.48, formula right for 52%.  
- **group 9** — 4.4% of jets, neuron 4.27, formula right for 94%.  
- **group 10** — 4.0% of jets, neuron 10.73, formula right for 57%.  

### neuron 10: spread in both directions (major)

- **What it measures:** Pushed down for jets thin across their minor axis (lam2 < 0.00362, its strongest term) or along their major axis (lam1 < 0.00516), and up for girth2 > 0.00809; it rises with the largest particle distance from the axis (rank correlation 0.622), mass (0.597), C2 (0.575) and width (0.549). Tops sit far highest (3.94), W (1.40) and Z (1.24) in the middle, gluons (1.00) and quarks (0.65) lowest, so its clearest separation is that it is small for quarks (AUC 0.25).
- *computed — its value:* largest for t (3.94), then W (1.40), then Z (1.24), then g (1.00), then q (0.65); it separates q jets from the rest best (AUC 0.25: small for q)
- **How the class scores use it:** Large spread marks a top, so it raises the t score (+14%); quarks sit lowest, so the q score reads a large value as evidence against a quark and it lowers the q score (-9%). It does not (or hardly) enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+14%); lowers the score of q (-9%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.826):

- `` — 2.6% of jets, value 10.54 (8.44…12.69), formula right 96%
- `` — 2.0% of jets, value 8.38 (6.00…10.69), formula right 92%
- `` — 2.0% of jets, value 5.66 (2.94…8.24), formula right 81%
- `` — 2.3% of jets, value 5.43 (1.77…8.69), formula right 79%
- `` — 62.3% of jets, value 1.39 (0.31…2.12), formula right 64%
- `` — 28.8% of jets, value 0.36 (0.00…0.75), formula right 62%

```
z = 5.34
if lam2 < 0.0036: z += -906 × (0.0036 − lam2)
if girth < 0.063: z += 61.80 × (0.063 − girth)
if lam1 < 0.0052: z += -518 × (0.0052 − lam1)
if girth2 > 0.0081: z += 310 × (girth2 − 0.0081)
if LHA > 0.240: z += -13.80 × (LHA − 0.240)
if girth2 < 0.0016: z += -1540 × (0.0016 − girth2)
if max_dr > 0.113: z += -10.60 × (max_dr − 0.113)
if LHA > 0.299 and tau21 < 0.663: z += -52.20 × (LHA − 0.299) × (0.663 − tau21)
if mass > 39.20: z += 0.031 × (mass − 39.20)
if C2 > 0.049: z += 42.40 × (C2 − 0.049)
if sum_pt > 792: z += -0.005 × (sum_pt − 792)
if log_sum_pt < 6.28: z += -7.51 × (6.28 − log_sum_pt)
if lam2 > 0.00022 and tau21 < 0.562: z += 1630 × (lam2 − 0.00022) × (0.562 − tau21)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 30.6% of jets, neuron 1.33, formula right for 70%.  
- **group 2** — 27.7% of jets, neuron 0.34, formula right for 62%.  
- **group 3** — 13.6% of jets, neuron 1.39, formula right for 47%.  
- **group 4** — 9.3% of jets, neuron 1.80, formula right for 62%.  
- **group 5** — 4.7% of jets, neuron 1.18, formula right for 73%.  
- **group 6** — 3.7% of jets, neuron 9.46, formula right for 95%.  
- **group 7** — 3.5% of jets, neuron 6.78, formula right for 88%.  
- **group 8** — 3.1% of jets, neuron 1.94, formula right for 74%.  
- **group 9** — 2.4% of jets, neuron 0.27, formula right for 50%.  
- **group 10** — 1.4% of jets, neuron 6.04, formula right for 73%.  

### neuron 11: centred, energetic, below top width (major)

- **What it measures:** Its strongest term pushes it down for girth < 0.124, but width < 0.00909 and a pT centroid close to the axis (centroid offset < 0.0501) push it back up; overall it falls with centroid offset (rank correlation -0.615) and lam2 (-0.461) and rises with total pT (0.501). W jets sit highest (4.78), then Z (3.56), quarks (3.50) and gluons (3.04); tops sit far lowest (0.99) and are at zero for 69% of them.
- *computed — its value:* largest for W (4.78), then Z (3.56), then q (3.50), then g (3.04), then t (0.99); it separates t jets from the rest best (AUC 0.15: small for t)
- **How the class scores use it:** W jets sit highest on it, so it raises the W score (+27%), its largest input. It does not (or hardly) enter the g, q, Z or t scores; in particular the Z score leaves it out although Z jets sit second on it.
- *computed — used by:* raises the score of W (+27%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.695):

- `` — 23.4% of jets, value 5.15 (3.19…7.06), formula right 73%
- `` — 41.3% of jets, value 3.79 (2.81…4.94), formula right 61%
- `` — 10.7% of jets, value 2.63 (1.50…3.81), formula right 51%
- `` — 2.0% of jets, value 2.29 (0.00…7.38), formula right 81%
- `` — 4.3% of jets, value 1.07 (0.00…2.12), formula right 46%
- `` — 2.1% of jets, value 0.97 (0.00…2.88), formula right 65%
- `` — 16.2% of jets, value 0.07 (0.00…0.00), formula right 79%

```
z = 1.49
if girth < 0.124: z += -89.80 × (0.124 − girth)
if width < 0.0091: z += 1010 × (0.0091 − width)
if centroid_offset < 0.050: z += 133 × (0.050 − centroid_offset)
if centroid_offset < 0.055 and log_sum_pt < 6.82: z += -146 × (0.055 − centroid_offset) × (6.82 − log_sum_pt)
if planar_flow < 0.289: z += 8.49 × (0.289 − planar_flow)
if sum_pt_top5 < 674: z += 0.0082 × (674 − sum_pt_top5)
if girth > 0.079 and pt_7 < 41.80: z += -11.30 × (girth − 0.079) × (41.80 − pt_7)
if girth > 0.071: z += -54.10 × (girth − 0.071)
if centroid_offset > 0.014: z += -62.30 × (centroid_offset − 0.014)
if planar_flow < 0.195 and width > 0.0059: z += -2230 × (0.195 − planar_flow) × (width − 0.0059)
if planar_flow < 0.276 and mass < 80.40: z += -0.110 × (0.276 − planar_flow) × (80.40 − mass)
if width < 0.0034: z += -343 × (0.0034 − width)
if mass_over_sum_pt > 0.093: z += -34.40 × (mass_over_sum_pt − 0.093)
if mass > 91.20: z += 0.357 × (mass − 91.20)
if girth < 0.020: z += -74.10 × (0.020 − girth)
if planar_flow < 0.361 and max_dr > 0.121: z += -23.10 × (0.361 − planar_flow) × (max_dr − 0.121)
if planar_flow < 0.188 and girth2 > 0.016: z += 3050 × (0.188 − planar_flow) × (girth2 − 0.016)
if z_7 < 0.042: z += -31.30 × (0.042 − z_7)
if centroid_offset > 0.019 and tau21 < 0.103: z += 2070 × (centroid_offset − 0.019) × (0.103 − tau21)
if LHA < 0.139 and z_7 < 0.026: z += 1330 × (0.139 − LHA) × (0.026 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 22.2% of jets, neuron 3.82, formula right for 63%.  
- **group 2** — 20.9% of jets, neuron 4.76, formula right for 68%.  
- **group 3** — 15.6% of jets, neuron 3.16, formula right for 72%.  
- **group 4** — 13.7% of jets, neuron 3.02, formula right for 48%.  
- **group 5** — 10.1% of jets, neuron 3.74, formula right for 58%.  
- **group 6** — 6.2% of jets, neuron 0.03, formula right for 73%.  
- **group 7** — 5.6% of jets, neuron 0.22, formula right for 82%.  
- **group 8** — 3.5% of jets, neuron 0.00, formula right for 86%.  
- **group 9** — 1.4% of jets, neuron 0.08, formula right for 69%.  
- **group 10** — 0.9% of jets, neuron 4.15, formula right for 78%.  

### neuron 13: compactness: narrower than a top (major)

- **What it measures:** Driven mostly by girth < 0.143 (its dominant term) and lam1 < 0.0157, both pushing it up, i.e. by the jet being narrower than a typical top; a very small width (< 0.00741) pulls it down a little. It falls with lam1, width, girth2 and girth (rank correlations -0.912 to -0.926); quarks (7.10), gluons (5.98), W (5.45) and Z (4.99) all sit high, tops far below (1.85).
- *computed — its value:* largest for q (7.10), then g (5.98), then W (5.45), then Z (4.99), then t (1.85); it separates t jets from the rest best (AUC 0.10: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score (-48%), its largest input; it also raises the W score (+8%) and the Z score (+9%). It does not (or hardly) enter the g or q scores.
- *computed — used by:* raises the score of W (+8%), Z (+9%); lowers the score of t (-48%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.878):

- `` — 23.9% of jets, value 8.28 (7.00…9.38), formula right 62%
- `` — 6.7% of jets, value 6.77 (5.62…7.75), formula right 60%
- `` — 8.2% of jets, value 6.46 (5.12…7.75), formula right 47%
- `` — 9.2% of jets, value 5.54 (4.38…6.62), formula right 53%
- `` — 16.3% of jets, value 4.65 (3.75…5.62), formula right 68%
- `` — 19.5% of jets, value 3.74 (2.75…4.62), formula right 71%
- `` — 5.0% of jets, value 1.51 (0.50…2.50), formula right 74%
- `` — 11.1% of jets, value 0.20 (0.00…0.75), formula right 84%

```
z = 1.02
if girth < 0.143: z += 67.50 × (0.143 − girth)
if lam1 < 0.016: z += 171 × (0.016 − lam1)
if lam1 < 0.0065: z += 548 × (0.0065 − lam1)
if width < 0.0074: z += -447 × (0.0074 − width)
if e2 < 0.049: z += -39.90 × (0.049 − e2)
if girth < 0.133 and log_sum_pt < 6.88: z += -43.90 × (0.133 − girth) × (6.88 − log_sum_pt)
if lam1 < 0.017 and centroid_offset < 0.038: z += -2490 × (0.017 − lam1) × (0.038 − centroid_offset)
if centroid_offset < 0.039: z += 30.50 × (0.039 − centroid_offset)
if tau21 < 0.498 and max_dr > -0.029: z += -17.50 × (0.498 − tau21) × (max_dr − -0.029)
if sum_pt_top5 > 653 and pt_7 < 44.90: z += 0.00055 × (sum_pt_top5 − 653) × (44.90 − pt_7)
if girth < 0.147 and pt_7 < 38.20: z += -0.833 × (0.147 − girth) × (38.20 − pt_7)
if lam2 < 0.00031 and centroid_offset < 0.042: z += -71500 × (0.00031 − lam2) × (0.042 − centroid_offset)
if z_7 < 0.029: z += -226 × (0.029 − z_7)
if sum_pt_top5 > 667 and z_7 > 0.023: z += -0.659 × (sum_pt_top5 − 667) × (z_7 − 0.023)
if C2 > 0.066: z += -51.60 × (C2 − 0.066)
if sum_pt_top5 > 878 and n_pt_above_50 > 6.08: z += 0.073 × (sum_pt_top5 − 878) × (n_pt_above_50 − 6.08)
if sum_pt > 977 and D2 < 4.88: z += -0.0035 × (sum_pt − 977) × (4.88 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 27.0% of jets, neuron 3.81, formula right for 71%.  
- **group 2** — 20.0% of jets, neuron 7.27, formula right for 53%.  
- **group 3** — 17.5% of jets, neuron 5.22, formula right for 56%.  
- **group 4** — 14.5% of jets, neuron 0.48, formula right for 82%.  
- **group 5** — 11.8% of jets, neuron 8.27, formula right for 62%.  
- **group 6** — 5.3% of jets, neuron 7.91, formula right for 74%.  
- **group 7** — 3.7% of jets, neuron 4.93, formula right for 74%.  
- **group 8** — 0.2% of jets, neuron 9.40, formula right for 65%.  
- **group 9** — 0.1% of jets, neuron 7.25, formula right for 68%.  

### neuron 14: intermediate, Z-sized width (major)

- **What it measures:** Pushed up for girth2 < 0.0137, e2 < 0.0437 and lam1 > 0.00419, but down for the narrowest jets (girth < 0.0879, width < 0.00735) and again for lam1 > 0.00616, so it peaks for jets of intermediate width; it rises with mass (rank correlation 0.453), width and mass/pT (0.429). Z jets sit highest (1.44), well above tops (0.56), gluons (0.20), W (0.19) and quarks (0.16).
- *computed — its value:* largest for Z (1.44), then t (0.56), then g (0.20), then W (0.19), then q (0.16); it separates Z jets from the rest best (AUC 0.77: large for Z)
- **How the class scores use it:** It is a W/Z separator: Z jets sit high and W jets near the bottom on it, so it raises the Z score (+6%) and lowers the W score (-8%). It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+6%); lowers the score of W (-8%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.492):

- `` — 13.3% of jets, value 2.04 (0.31…3.44), formula right 71%
- `` — 2.0% of jets, value 0.88 (0.00…2.08), formula right 62%
- `` — 22.1% of jets, value 0.71 (0.00…2.38), formula right 69%
- `` — 9.7% of jets, value 0.56 (0.00…1.31), formula right 81%
- `` — 2.4% of jets, value 0.15 (0.00…0.62), formula right 55%
- `` — 2.3% of jets, value 0.14 (0.00…0.56), formula right 59%
- `` — 7.0% of jets, value 0.06 (0.00…0.19), formula right 80%
- `` — 41.2% of jets, value 0.00 (0.00…0.00), formula right 57%

```
z = -2.44
if girth2 < 0.014: z += 437 × (0.014 − girth2)
if e2 < 0.044: z += 180 × (0.044 − e2)
if girth < 0.088: z += -94.60 × (0.088 − girth)
if lam1 > 0.0042: z += 937 × (lam1 − 0.0042)
if width < 0.0073: z += -881 × (0.0073 − width)
if lam1 > 0.0062: z += -932 × (lam1 − 0.0062)
if lam1 > 0.0042 and D2 > 0.395: z += -829 × (lam1 − 0.0042) × (D2 − 0.395)
if C2 < 0.036: z += -67.10 × (0.036 − C2)
if lam1 > 0.0026 and D2 > 0.419: z += 474 × (lam1 − 0.0026) × (D2 − 0.419)
if girth < 0.033: z += -114 × (0.033 − girth)
if girth2 < 0.013 and eccentricity > 0.968: z += 8900 × (0.013 − girth2) × (eccentricity − 0.968)
if C2 > 0.036: z += 60.50 × (C2 − 0.036)
if lam1 > 0.0071 and D2 > 0.378: z += 371 × (lam1 − 0.0071) × (D2 − 0.378)
if width < 0.0079 and planar_flow < 0.117: z += -5010 × (0.0079 − width) × (0.117 − planar_flow)
if width < 0.0077 and D2 < 1.00: z += -1710 × (0.0077 − width) × (1.00 − D2)
if girth2 < 0.0046: z += -219 × (0.0046 − girth2)
if planar_flow < 0.104 and centroid_offset > 0.0091: z += 1360 × (0.104 − planar_flow) × (centroid_offset − 0.0091)
if planar_flow < 0.110 and centroid_offset > 0.018: z += -1680 × (0.110 − planar_flow) × (centroid_offset − 0.018)
if centroid_offset > 0.050: z += -248 × (centroid_offset − 0.050)
if D2 < 1.09 and centroid_offset < 0.032: z += 48.00 × (1.09 − D2) × (0.032 − centroid_offset)
if C2 > 0.067: z += -58.20 × (C2 − 0.067)
if tau21 < 0.136: z += 9.22 × (0.136 − tau21)
if centroid_offset > 0.030: z += -56.10 × (centroid_offset − 0.030)
if planar_flow < 0.100 and max_dr < 0.170: z += -131 × (0.100 − planar_flow) × (0.170 − max_dr)
if e2 < 0.041 and D2 < 0.984: z += 190 × (0.041 − e2) × (0.984 − D2)
if sum_pt_top5 < 457: z += -0.0064 × (457 − sum_pt_top5)
if mass < 75.60 and D2 < 0.734: z += -0.072 × (75.60 − mass) × (0.734 − D2)
if planar_flow < 0.110 and sum_pt < 741: z += -0.042 × (0.110 − planar_flow) × (741 − sum_pt)
if lam1 > 0.0027 and D2 > 1.65: z += -319 × (lam1 − 0.0027) × (D2 − 1.65)
if lam1 > 0.0066 and max_dr < 0.165: z += 7610 × (lam1 − 0.0066) × (0.165 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 30.2% of jets, neuron 0.00, formula right for 61%.  
- **group 2** — 21.9% of jets, neuron 0.69, formula right for 68%.  
- **group 3** — 19.2% of jets, neuron 0.32, formula right for 53%.  
- **group 4** — 12.6% of jets, neuron 1.90, formula right for 72%.  
- **group 5** — 4.3% of jets, neuron 0.15, formula right for 76%.  
- **group 6** — 3.6% of jets, neuron 0.51, formula right for 75%.  
- **group 7** — 3.6% of jets, neuron 0.60, formula right for 90%.  
- **group 8** — 1.9% of jets, neuron 0.16, formula right for 71%.  
- **group 9** — 1.4% of jets, neuron 0.23, formula right for 87%.  
- **group 10** — 1.2% of jets, neuron 0.42, formula right for 84%.  

### neuron 0: elongated two-prong jet with mass (moderate)

- **What it measures:** Pushed up for compact jets (girth2 < 0.0128, its strongest term) and down for light jets (mass < 29.3 GeV, mass < 59.8 GeV) and for extremely thin ones (lam1 < 0.00142); it rises with eccentricity (rank correlation 0.628) and falls with planar flow and τ21 (-0.628, -0.607). Z (2.25) and W (1.99) jets sit highest, tops in the middle (0.65), gluons (0.32) and quarks (0.24) lowest.
- *computed — its value:* largest for Z (2.25), then W (1.99), then t (0.65), then g (0.32), then q (0.24); it separates Z jets from the rest best (AUC 0.76: large for Z)
- **How the class scores use it:** W jets sit high and gluons low on it, so it raises the W score (+8%) and lowers the g score (-6%). It does not (or hardly) enter the q, Z or t scores, even though Z jets sit highest; the Z score takes its two-prong information from neuron 7 instead.
- *computed — used by:* raises the score of W (+8%); lowers the score of g (-6%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.779):

- `` — 25.2% of jets, value 3.02 (1.88…4.12), formula right 74%
- `` — 16.1% of jets, value 1.65 (0.00…2.75), formula right 56%
- `` — 2.2% of jets, value 0.96 (0.25…1.75), formula right 73%
- `` — 4.1% of jets, value 0.64 (0.00…2.38), formula right 50%
- `` — 2.1% of jets, value 0.30 (0.00…1.12), formula right 48%
- `` — 2.9% of jets, value 0.25 (0.00…0.88), formula right 67%
- `` — 12.6% of jets, value 0.07 (0.00…0.12), formula right 84%
- `` — 34.7% of jets, value 0.00 (0.00…0.00), formula right 60%

```
z = -0.915
if girth2 < 0.013: z += 517 × (0.013 − girth2)
if lam1 < 0.0014: z += -4920 × (0.0014 − lam1)
if girth < 0.077: z += -54.50 × (0.077 − girth)
if mass < 29.30: z += -0.209 × (29.30 − mass)
if width < 0.0046: z += -855 × (0.0046 − width)
if mass < 59.80: z += -0.056 × (59.80 − mass)
if lam1 < 0.00071: z += 6750 × (0.00071 − lam1)
if e2 < 0.025: z += 97.40 × (0.025 − e2)
if planar_flow < 0.158 and centroid_offset < 0.056: z += 272 × (0.158 − planar_flow) × (0.056 − centroid_offset)
if sum_pt > 816: z += -0.016 × (sum_pt − 816)
if sum_pt_top5 > 699: z += 0.013 × (sum_pt_top5 − 699)
if mass < 64.60 and pt_7 < 38.60: z += -0.0016 × (64.60 − mass) × (38.60 − pt_7)
if sum_pt > 890: z += -0.016 × (sum_pt − 890)
if mass < 67.80 and centroid_offset > 0.013: z += 1.12 × (67.80 − mass) × (centroid_offset − 0.013)
if mass < 56.80 and C2 > 0.018: z += 1.65 × (56.80 − mass) × (C2 − 0.018)
if sum_pt > 905 and pt_7 < 26.00: z += 0.0023 × (sum_pt − 905) × (26.00 − pt_7)
if lam1 < 0.0065 and D2 < 0.866: z += -2050 × (0.0065 − lam1) × (0.866 − D2)
if D2 > 3.70: z += -0.856 × (D2 − 3.70)
if sum_pt > 901 and pt_7 > 25.90: z += -0.00069 × (sum_pt − 901) × (pt_7 − 25.90)
if eccentricity > 0.997: z += 367 × (eccentricity − 0.997)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 20.2% of jets, neuron 3.07, formula right for 76%.  
- **group 2** — 18.5% of jets, neuron 0.26, formula right for 79%.  
- **group 3** — 17.0% of jets, neuron 0.00, formula right for 60%.  
- **group 4** — 14.8% of jets, neuron 2.01, formula right for 57%.  
- **group 5** — 9.3% of jets, neuron 0.00, formula right for 49%.  
- **group 6** — 9.2% of jets, neuron 0.61, formula right for 50%.  
- **group 7** — 6.0% of jets, neuron 0.00, formula right for 73%.  
- **group 8** — 3.8% of jets, neuron 1.94, formula right for 70%.  
- **group 9** — 0.7% of jets, neuron 0.07, formula right for 76%.  
- **group 10** — 0.4% of jets, neuron 0.00, formula right for 69%.  

### neuron 5: pT held by the few hardest particles (moderate)

- **What it measures:** Pushed up when the 8th-hardest particle is soft (pT_7 < 54.4 GeV, its strongest term) and for small e2 (< 0.0405); it rises with the summed pT of the 3 and 5 hardest particles (rank correlations 0.646 and 0.647) and falls with e2 (-0.692) and girth (-0.639). Quarks sit far highest (4.45); W (1.63), Z (1.61) and gluons (1.59) are close together, and tops lowest (0.56).
- *computed — its value:* largest for q (4.45), then W (1.63), then Z (1.61), then g (1.59), then t (0.56); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** Quarks sit far above gluons and tops on it, so it raises the q score (+4%) and lowers the g score (-13%) and the t score (-11%): for the gluon and top classes it is evidence against their type. It does not (or hardly) enter the W or Z scores.
- *computed — used by:* raises the score of q (+4%); lowers the score of g (-13%), t (-11%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.752):

- `` — 7.2% of jets, value 7.66 (4.62…10.50), formula right 76%
- `` — 11.0% of jets, value 4.98 (2.75…7.38), formula right 58%
- `` — 6.9% of jets, value 3.18 (0.88…5.12), formula right 65%
- `` — 10.5% of jets, value 2.62 (1.00…4.25), formula right 51%
- `` — 8.8% of jets, value 1.12 (0.00…2.75), formula right 79%
- `` — 16.9% of jets, value 0.99 (0.00…2.38), formula right 53%
- `` — 5.7% of jets, value 0.85 (0.00…2.38), formula right 74%
- `` — 33.0% of jets, value 0.12 (0.00…0.50), formula right 72%

```
z = -0.638
if pt_7 < 54.40: z += 0.122 × (54.40 − pt_7)
if e2 < 0.041: z += 81.90 × (0.041 − e2)
if z_7 < 0.072 and centroid_offset < 0.030: z += -2690 × (0.072 − z_7) × (0.030 − centroid_offset)
if width < 0.0023 and centroid_offset < 0.026: z += 78100 × (0.0023 − width) × (0.026 − centroid_offset)
if sum_pt_top5 < 721: z += -0.0051 × (721 − sum_pt_top5)
if LHA < 0.217 and log_sum_pt < 6.83: z += -121 × (0.217 − LHA) × (6.83 − log_sum_pt)
if sum_pt > 935: z += -0.046 × (sum_pt − 935)
if z_7 < 0.076 and sum_pt < 803: z += -0.258 × (0.076 − z_7) × (803 − sum_pt)
if log_sum_pt > 6.27 and centroid_offset > 0.0024: z += 125 × (log_sum_pt − 6.27) × (centroid_offset − 0.0024)
if sum_pt > 864 and centroid_offset < 0.011: z += 1.57 × (sum_pt − 864) × (0.011 − centroid_offset)
if z_7 < 0.024: z += 169 × (0.024 − z_7)
if LHA < 0.160 and centroid_offset > 0.0037: z += -4360 × (0.160 − LHA) × (centroid_offset − 0.0037)
if girth2 < 5.6e-05: z += 62400 × (5.6e-05 − girth2)
if log_sum_pt > 6.58 and centroid_offset > 0.015: z += -580 × (log_sum_pt − 6.58) × (centroid_offset − 0.015)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 22.0% of jets, neuron 0.43, formula right for 69%.  
- **group 2** — 21.8% of jets, neuron 0.24, formula right for 70%.  
- **group 3** — 15.0% of jets, neuron 2.83, formula right for 49%.  
- **group 4** — 13.5% of jets, neuron 1.97, formula right for 73%.  
- **group 5** — 8.8% of jets, neuron 6.15, formula right for 66%.  
- **group 6** — 8.0% of jets, neuron 2.76, formula right for 56%.  
- **group 7** — 5.1% of jets, neuron 0.41, formula right for 63%.  
- **group 8** — 3.6% of jets, neuron 8.39, formula right for 78%.  
- **group 9** — 1.8% of jets, neuron 1.62, formula right for 64%.  
- **group 10** — 0.3% of jets, neuron 0.10, formula right for 62%.  

### neuron 7: two-prong shape in Z-like window (moderate)

- **What it measures:** Pushed up for mass/pT above 0.0849 but down again above 0.0905 (a narrow window where boosted bosons sit); it rises with eccentricity (rank correlation 0.585) and falls with planar flow (-0.585), τ21 (-0.533) and D2 (-0.4), i.e. it likes an elongated two-prong jet. Z jets sit highest (3.12), then W (1.90), with tops (0.74), gluons (0.54) and quarks (0.28) low.
- *computed — its value:* largest for Z (3.12), then W (1.90), then t (0.74), then g (0.54), then q (0.28); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** High values mark a two-prong boson, most of all a Z, so it raises the W score (+6%) and the Z score (+19%), where it is the largest positive input. It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* raises the score of W (+6%), Z (+19%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.711):

- `` — 15.3% of jets, value 4.20 (2.50…6.12), formula right 79%
- `` — 15.7% of jets, value 2.09 (0.25…3.88), formula right 57%
- `` — 14.5% of jets, value 1.84 (0.00…3.50), formula right 61%
- `` — 5.5% of jets, value 0.84 (0.00…1.75), formula right 42%
- `` — 2.0% of jets, value 0.56 (0.00…2.00), formula right 68%
- `` — 32.3% of jets, value 0.07 (0.00…0.00), formula right 61%
- `` — 12.2% of jets, value 0.06 (0.00…0.00), formula right 84%
- `` — 2.6% of jets, value 0.02 (0.00…0.00), formula right 72%

```
z = 5.45
if girth2 > 0.0074: z += -1710 × (girth2 − 0.0074)
if e2 < 0.050: z += -154 × (0.050 − e2)
if e2 < 0.038: z += 211 × (0.038 − e2)
if girth2 > 0.0087: z += 1420 × (girth2 − 0.0087)
if mass_over_sum_pt > 0.090: z += -312 × (mass_over_sum_pt − 0.090)
if lam1 < 0.0084: z += -434 × (0.0084 − lam1)
if mass_over_sum_pt > 0.085: z += 198 × (mass_over_sum_pt − 0.085)
if width < 0.0052: z += -586 × (0.0052 − width)
if centroid_offset < 0.040 and sum_pt > 559: z += 0.207 × (0.040 − centroid_offset) × (sum_pt − 559)
if LHA < 0.236: z += -21.60 × (0.236 − LHA)
if mass < 27.60: z += 0.124 × (27.60 − mass)
if girth2 > 0.0078 and log_sum_pt > 6.13: z += -1510 × (girth2 − 0.0078) × (log_sum_pt − 6.13)
if girth2 > 0.0046 and eccentricity > 0.942: z += 7960 × (girth2 − 0.0046) × (eccentricity − 0.942)
if pt_7 < 47.20 and planar_flow < 0.801: z += -0.080 × (47.20 − pt_7) × (0.801 − planar_flow)
if centroid_offset < 0.020: z += -65.70 × (0.020 − centroid_offset)
if lam1 < 0.0085 and D2 < 1.17: z += -1010 × (0.0085 − lam1) × (1.17 − D2)
if e2 < 0.049 and D2 < 1.14: z += 132 × (0.049 − e2) × (1.14 − D2)
if eccentricity > 0.955: z += -16.30 × (eccentricity − 0.955)
if girth2 < 0.00056: z += -2650 × (0.00056 − girth2)
if planar_flow < 0.174 and sum_pt > 627: z += 0.026 × (0.174 − planar_flow) × (sum_pt − 627)
if max_dr < 0.197 and D2 < 1.21: z += 10.80 × (0.197 − max_dr) × (1.21 − D2)
if centroid_offset > 0.028: z += -58.40 × (centroid_offset − 0.028)
if mass > 80.40: z += -0.132 × (mass − 80.40)
if centroid_offset < 0.025 and C2 > 0.021: z += 1700 × (0.025 − centroid_offset) × (C2 − 0.021)
if centroid_offset < 0.046 and C2 > 0.065: z += -1660 × (0.046 − centroid_offset) × (C2 − 0.065)
if e2 < 0.022 and tau21 < 0.469: z += 319 × (0.022 − e2) × (0.469 − tau21)
if mass > 80.40 and eccentricity > 0.934: z += -2.21 × (mass − 80.40) × (eccentricity − 0.934)
if lam1 < 0.0083 and n_pt_above_50 < 3.92: z += -82.50 × (0.0083 − lam1) × (3.92 − n_pt_above_50)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 31.5% of jets, neuron 0.21, formula right for 60%.  
- **group 2** — 29.5% of jets, neuron 3.21, formula right for 70%.  
- **group 3** — 20.9% of jets, neuron 1.26, formula right for 56%.  
- **group 4** — 3.8% of jets, neuron 1.13, formula right for 66%.  
- **group 5** — 3.5% of jets, neuron 0.13, formula right for 77%.  
- **group 6** — 3.3% of jets, neuron 0.01, formula right for 82%.  
- **group 7** — 3.1% of jets, neuron 0.00, formula right for 86%.  
- **group 8** — 2.6% of jets, neuron 0.00, formula right for 88%.  
- **group 9** — 1.4% of jets, neuron 0.00, formula right for 81%.  
- **group 10** — 0.3% of jets, neuron 0.00, formula right for 66%.  

### neuron 8: narrow, centred single core (moderate)

- **What it measures:** Pushed up for narrow jets (width < 0.0055, its strongest term) with every particle within ΔR 0.188 of the axis, and down when such a narrow jet has its pT centroid off the axis, for C2 < 0.0262 and for LHA < 0.234; it falls with width, girth and LHA (rank correlations -0.683 to -0.693). Quarks sit highest (2.99), then gluons (1.76), with W (0.57), Z (0.36) and tops (0.14) low.
- *computed — its value:* largest for q (2.99), then g (1.76), then W (0.57), then Z (0.36), then t (0.14); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** The W score reads a single narrow core as evidence against a W, so it lowers the W score (-6%); it raises the q score slightly (+3%), since quarks sit highest, and the t score (+5%), a small correction given that tops sit lowest on it. It does not (or hardly) enter the g or Z scores.
- *computed — used by:* raises the score of q (+3%), t (+5%); lowers the score of W (-6%); does not (or hardly) enter the score of g, Z (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.835):

- `` — 11.8% of jets, value 4.28 (2.75…5.75), formula right 63%
- `` — 13.2% of jets, value 3.25 (2.12…4.50), formula right 66%
- `` — 2.0% of jets, value 2.47 (0.25…4.45), formula right 46%
- `` — 3.8% of jets, value 1.71 (0.50…3.25), formula right 53%
- `` — 6.0% of jets, value 0.55 (0.00…1.88), formula right 51%
- `` — 8.1% of jets, value 0.37 (0.00…1.25), formula right 62%
- `` — 7.4% of jets, value 0.30 (0.00…1.12), formula right 43%
- `` — 47.8% of jets, value 0.03 (0.00…0.00), formula right 74%

```
z = -1.26
if width < 0.0055: z += 1260 × (0.0055 − width)
if max_dr < 0.188: z += 12.20 × (0.188 − max_dr)
if width < 0.0058 and centroid_offset > 0.0063: z += -58200 × (0.0058 − width) × (centroid_offset − 0.0063)
if C2 < 0.026: z += -97.90 × (0.026 − C2)
if LHA < 0.234: z += -19.40 × (0.234 − LHA)
if log_sum_pt > 6.67 and girth2 < 0.016: z += -1190 × (log_sum_pt − 6.67) × (0.016 − girth2)
if max_dr < 0.184 and lam2 < 0.00019: z += 65700 × (0.184 − max_dr) × (0.00019 − lam2)
if mass < 24.10: z += -0.114 × (24.10 − mass)
if girth < 0.044 and log_sum_pt > 6.66: z += 336 × (0.044 − girth) × (log_sum_pt − 6.66)
if log_sum_pt > 6.60 and pt_7 < 55.80: z += 0.172 × (log_sum_pt − 6.60) × (55.80 − pt_7)
if girth2 < 0.0066 and planar_flow < 0.401: z += 559 × (0.0066 − girth2) × (0.401 − planar_flow)
if centroid_offset < 0.0035: z += 533 × (0.0035 − centroid_offset)
if mass < 31.30 and centroid_offset > 0.0079: z += 1.84 × (31.30 − mass) × (centroid_offset − 0.0079)
if mass < 21.80 and max_pair_mass > 13.00: z += 66.80 × (21.80 − mass) × (max_pair_mass − 13.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 46.4% of jets, neuron 0.07, formula right for 73%.  
- **group 2** — 11.0% of jets, neuron 3.39, formula right for 60%.  
- **group 3** — 9.2% of jets, neuron 0.64, formula right for 53%.  
- **group 4** — 8.5% of jets, neuron 3.71, formula right for 71%.  
- **group 5** — 6.2% of jets, neuron 3.54, formula right for 53%.  
- **group 6** — 5.9% of jets, neuron 0.10, formula right for 75%.  
- **group 7** — 5.5% of jets, neuron 0.73, formula right for 48%.  
- **group 8** — 4.3% of jets, neuron 0.03, formula right for 42%.  
- **group 9** — 2.9% of jets, neuron 3.66, formula right for 71%.  

### neuron 15: wider than a W, below top size (moderate)

- **What it measures:** Pushed up for e2 < 0.0627 (its strongest term) and width < 0.0136, but down for lam1 < 0.00855 and lam2 < 0.00323, so it responds to jets fairly wide in both directions yet not as broad as a typical top; it rises with the number and pT share of particles at 0.05 ≤ ΔR < 0.1 (rank correlations 0.405 and 0.403). Z jets sit highest (1.13), then tops (0.74), with gluons (0.30), W (0.19) and quarks (0.17) low.
- *computed — its value:* largest for Z (1.13), then t (0.74), then g (0.30), then W (0.19), then q (0.17); it separates Z jets from the rest best (AUC 0.74: large for Z)
- **How the class scores use it:** W jets sit near the bottom on it, so the W score reads it as evidence against a W and it lowers the W score (-8%); it also lowers the Z score slightly (-2.5%), even though Z jets sit highest. It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-8%), Z (-2%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.606):

- `` — 11.9% of jets, value 2.36 (0.88…3.50), formula right 73%
- `` — 6.4% of jets, value 1.17 (0.00…2.12), formula right 66%
- `` — 3.4% of jets, value 0.82 (0.00…2.38), formula right 78%
- `` — 9.9% of jets, value 0.62 (0.00…1.75), formula right 57%
- `` — 19.5% of jets, value 0.22 (0.00…0.75), formula right 66%
- `` — 2.4% of jets, value 0.21 (0.00…0.75), formula right 47%
- `` — 11.4% of jets, value 0.12 (0.00…0.38), formula right 82%
- `` — 35.2% of jets, value 0.01 (0.00…0.00), formula right 60%

```
z = -5.86
if e2 < 0.063: z += 237 × (0.063 − e2)
if width < 0.014: z += 546 × (0.014 − width)
if lam1 < 0.0086: z += -941 × (0.0086 − lam1)
if lam2 < 0.0032: z += -1150 × (0.0032 − lam2)
if e2 > 0.024: z += 147 × (e2 − 0.024)
if girth > 0.033: z += 50.00 × (girth − 0.033)
if mass < 80.40: z += -0.031 × (80.40 − mass)
if LHA < 0.305: z += -9.22 × (0.305 − LHA)
if LHA > 0.343: z += -73.40 × (LHA − 0.343)
if lam2 < 0.00034: z += -2470 × (0.00034 − lam2)
if tau21 < 0.250: z += 6.19 × (0.250 − tau21)
if e2 > 0.050: z += -82.40 × (e2 − 0.050)
if width < 0.0076 and e2 > 0.024: z += -59700 × (0.0076 − width) × (e2 − 0.024)
if width < 0.0063 and log_sum_pt > 6.90: z += -9720 × (0.0063 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 44.30 × (log_sum_pt − 6.90)
if width < 0.0082 and planar_flow < 0.062: z += -3120 × (0.0082 − width) × (0.062 − planar_flow)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 29.4% of jets, neuron 0.02, formula right for 60%.  
- **group 2** — 16.1% of jets, neuron 0.54, formula right for 70%.  
- **group 3** — 13.0% of jets, neuron 0.61, formula right for 61%.  
- **group 4** — 12.7% of jets, neuron 0.22, formula right for 53%.  
- **group 5** — 11.6% of jets, neuron 2.19, formula right for 72%.  
- **group 6** — 5.8% of jets, neuron 0.17, formula right for 84%.  
- **group 7** — 5.7% of jets, neuron 0.66, formula right for 77%.  
- **group 8** — 4.0% of jets, neuron 0.06, formula right for 85%.  
- **group 9** — 1.4% of jets, neuron 0.22, formula right for 67%.  
- **group 10** — 0.3% of jets, neuron 0.15, formula right for 64%.  

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs a very wide jet (girth2 > 0.0193 pushes up) and is pushed down strongly for e2 > 0.0629; it follows the number of particles above 10 GeV (rank correlation 0.849). Only tops reach it with any frequency (non-zero for 15% of them), so tops sit highest (0.23), then gluons (0.10) and quarks (0.03), with W and Z at 0.00.
- *computed — its value:* largest for t (0.23), then g (0.10), then q (0.03), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.57: large for t)
- **How the class scores use it:** It does not (or hardly) enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

Regimes (a small tree on its quantities; R² 0.429):

- `` — 2.1% of jets, value 2.20 (0.00…5.00), formula right 80%
- `` — 2.0% of jets, value 1.07 (0.00…2.62), formula right 84%
- `` — 2.2% of jets, value 0.28 (0.00…1.12), formula right 88%
- `` — 93.8% of jets, value 0.00 (0.00…0.00), formula right 64%

```
z = -1.34
if e2 > 0.063: z += -120 × (e2 − 0.063)
if girth2 > 0.019: z += 238 × (girth2 − 0.019)
if girth2 > 0.019 and pt_7 > 15.40: z += 6.79 × (girth2 − 0.019) × (pt_7 − 15.40)
if mass > 91.20: z += 0.099 × (mass − 91.20)
if girth2 > 0.015 and lam2 > 5.6e-06: z += 8280 × (girth2 − 0.015) × (lam2 − 5.6e-06)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **group 1** — 91.1% of jets, neuron 0.00, formula right for 64%.  
- **group 2** — 2.1% of jets, neuron 0.14, formula right for 89%.  
- **group 3** — 2.0% of jets, neuron 0.00, formula right for 83%.  
- **group 4** — 1.7% of jets, neuron 0.47, formula right for 89%.  
- **group 5** — 0.8% of jets, neuron 1.73, formula right for 83%.  
- **group 6** — 0.8% of jets, neuron 0.86, formula right for 76%.  
- **group 7** — 0.7% of jets, neuron 2.75, formula right for 83%.  
- **group 8** — 0.6% of jets, neuron 1.41, formula right for 75%.  
- **group 9** — 0.2% of jets, neuron 5.42, formula right for 71%.  
- **group 10** — 0.0% of jets, neuron 10.48, formula right for 78%.  
