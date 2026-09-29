# What each part of the 400-term formula does (8 particles)

*simplified from the formula tuned on the network (624), keeping validation agreement with the network within 0.5 point*. Validation accuracy 65.40%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a clipped sum of if-statements on jet-shape quantities), and each class score adds some scales and subtracts others; this 400-term formula is a simplified version of a formula that makes the same decision as the original network on about 90% of test jets. Tops are recognised by size: the t score subtracts compactness (neuron 13, on which tops sit far below every other type, -46%) and adds overall size (neuron 10, +20%), so it closely follows girth, width and e2. Quarks and gluons both sit high on the narrow-and-light scale (neuron 9), which feeds both of their scores; the g score also adds the light-jet scale (neuron 2) and the moderate-mass scale (neuron 1, on which quarks sit lowest) and subtracts quark-likeness (neuron 5: pT concentrated in the few hardest particles), while the q score subtracts the two-prong substructure scale (neuron 4) and overall size (neuron 10). W and Z jets sit high on scales for a compact, elongated two-prong jet (neurons 7 and 13 feed both boson scores, neuron 0 and the compact-centred neuron 11 only the W score), and both boson scores subtract the broad-spread and wide-angle-radiation scales (neurons 6 and 3), on which tops and gluons sit much higher. W and Z are split mainly by width: Z jets sit highest on an intermediate-width scale (neuron 14), which the Z score adds (+5%) and the W score subtracts (-8%), and the 'slightly wider than a W' scale (neuron 15) takes more out of the W score (-7%) than out of the Z score (-2%).

## The 5 class scores

### score g: gluon: light jet, many hard particles

High for light jets whose pT is spread over many hard particles (it rises with z_7 and planar flow and falls with the pT share of the 5 hardest particles); it averages 2.04 for g jets, 1.15 for q and 1.05 for t, and is near zero for W and Z (AUC 0.82 for gluons against the rest).

Adds the light-jet scale (neuron 2, +32%), narrow-light-ness (neuron 9, +22%) and the moderate-mass scale (neuron 1, +15%), plus a little broad spread (neuron 6, +6%); subtracts quark-likeness (neuron 5, -16%) and, less, the compact two-prong scale (neuron 0, -5%) and the two-prong substructure scale (neuron 4, -4%).

*computed:* largest for g (2.04), then q (1.15), then t (1.05), then W (0.09), then Z (-0.02); it separates g jets from the rest best (AUC 0.82: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets (it falls with girth, LHA and lam1, rank correlations about -0.67); it averages 1.97 for q jets and 1.36 for g, with tops lower (0.18) and W and Z near or below zero (AUC 0.85 for quarks). Gluons are its main confusion.

Mostly adds narrow-light-ness (neuron 9, +49%); subtracts the two-prong substructure scale (neuron 4, -18%) and overall size (neuron 10, -14%); adds a little of broad spread (neuron 6, +10%), quark-likeness (neuron 5, +6%) and the narrow centred core (neuron 8, +2%).

*computed:* largest for q (1.97), then g (1.36), then t (0.18), then W (0.07), then Z (-0.17); it separates q jets from the rest best (AUC 0.85: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets that are not wider than a W; it averages 2.40 for W jets and 0.82 for Z, is slightly negative for q and g, and strongly negative for t (-2.61) (AUC 0.89).

Adds the compact-centred scale (neuron 11, +25%, its largest input), the two-prong scales (neurons 0 and 7, +9% each) and compactness (neuron 13, +8%); subtracts broad off-centre spread (neuron 6, -13%), wide-angle radiation (neuron 3, -12%), the wider-than-a-W scales (neurons 14 and 15, -8% and -7%) and a little of neurons 8 and 9.

*computed:* largest for W (2.40), then Z (0.82), then q (-0.14), then g (-0.48), then t (-2.61); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: two-prong jet, wider than a W

High for compact two-prong jets of W/Z size with an intermediate, Z-sized width; it averages 2.22 for Z jets and 1.36 for W (W is its main confusion), is slightly negative for q and g, and strongly negative for t (-2.11) (AUC 0.86).

Adds the two-prong W/Z scale (neuron 7, +26%, its largest input), two-prong substructure (neuron 4, +10%), compactness (neuron 13, +9%), the intermediate-width scale (neuron 14, +5%) and the moderate-mass scale (neuron 1, +5%); subtracts broad off-centre spread (neuron 6, -20%), wide-angle radiation (neuron 3, -17%) and a little of neurons 9 and 15.

*computed:* largest for Z (2.22), then W (1.36), then q (-0.05), then g (-0.30), then t (-2.11); it separates Z jets from the rest best (AUC 0.86: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and e2 with rank correlations of about 0.89-0.91. It averages 2.85 for t jets, 0.41 for Z and 0.14 for W, is near zero for g and negative for q (-1.38) (AUC 0.91).

Subtracts compactness (neuron 13, -46%) and adds overall size (neuron 10, +20%); also subtracts quark-likeness (neuron 5, -16%) and adds two-prong substructure (neuron 4, +11%) and a little of the narrow-core scale (neuron 8, +3%).

*computed:* largest for t (2.85), then Z (0.41), then W (0.14), then g (0.07), then q (-1.38); it separates t jets from the rest best (AUC 0.91: large for t)

## The 16 neurons (most important first)

### neuron 1: moderate mass and spread, hard jet (major)

- **What it measures:** Pushed up for small e2_sq (< 0.00817, its strongest term) and for high total pT (log of total pT > 6.38), and pushed down for narrow (width < 0.00866), small (max ΔR < 0.249) and light jets (mass < 55.5 GeV); it rises with mass, lam1 and width (rank correlations 0.531, 0.453, 0.451). Tops (1.38) and Z jets (1.31) sit highest, gluons (1.00) and W (0.91) in the middle, quarks lowest (0.53, zero for 58.6% of them).
- *computed — its value:* largest for t (1.38), then Z (1.31), then g (1.00), then W (0.91), then q (0.53); it separates q jets from the rest best (AUC 0.32: small for q)
- **How the class scores use it:** It raises the g score (+15%), where it marks the difference between gluons and the quarks that sit lowest on it, and it raises the Z score a little (+5%). It does not enter the q, W or t scores; the tops that also sit high on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+15%), Z (+5%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.823
if e2_sq < 0.0082: z += 404 × (0.0082 − e2_sq)
if width < 0.0087: z += -295 × (0.0087 − width)
if log_sum_pt > 6.38: z += 4.70 × (log_sum_pt − 6.38)
if max_dr < 0.249: z += -7.11 × (0.249 − max_dr)
if lam1 < 0.0082 and lam2 < 0.00054: z += -428000 × (0.0082 − lam1) × (0.00054 − lam2)
if pt_7 > 34.50: z += 0.155 × (pt_7 − 34.50)
if z_7 < 0.056: z += -57.20 × (0.056 − z_7)
if mass < 55.50: z += -0.025 × (55.50 − mass)
if log_sum_pt > 6.38 and max_dr < 0.200: z += 22.40 × (log_sum_pt − 6.38) × (0.200 − max_dr)
if log_sum_pt > 6.63: z += 7.02 × (log_sum_pt − 6.63)
if log_sum_pt > 6.57 and lam2 < 0.0012: z += 4120 × (log_sum_pt − 6.57) × (0.0012 − lam2)
if pt_7 > 34.80 and mass < 91.20: z += -0.0017 × (pt_7 − 34.80) × (91.20 − mass)
if z_7 < 0.056 and girth2_top2 < 0.014: z += 3150 × (0.056 − z_7) × (0.014 − girth2_top2)
if e2_sq < 0.0085 and planar_flow < 0.083: z += -5800 × (0.0085 − e2_sq) × (0.083 − planar_flow)
if z_7 < 0.043: z += -64.20 × (0.043 − z_7)
if LHA > 0.259: z += 8.82 × (LHA − 0.259)
if log_sum_pt > 6.37 and centroid_offset < 0.025: z += -93.80 × (log_sum_pt − 6.37) × (0.025 − centroid_offset)
if log_sum_pt > 6.64 and girth2_top3 < 0.008: z += -679 × (log_sum_pt − 6.64) × (0.008 − girth2_top3)
if e2 < 0.035 and eccentricity > 0.979: z += 6960 × (0.035 − e2) × (eccentricity − 0.979)
if e2 > 0.031: z += -16.50 × (e2 − 0.031)
if lam1 < 0.0082 and centroid_offset > 0.021: z += 14200 × (0.0082 − lam1) × (centroid_offset − 0.021)
if pt_7 > 35.00 and max_dr < 0.079: z += 0.917 × (pt_7 − 35.00) × (0.079 − max_dr)
if pt_7 > 52.60: z += -0.105 × (pt_7 − 52.60)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (70 GeV), very wide, pT spread over several particles, low pT** — 21.5% of jets, neuron 1.35.  
- **light (44 GeV), average width, pT spread over several particles** — 15.5% of jets, neuron 0.58.  
- **very light (13 GeV), very narrow, pT spread over several particles** — 12.6% of jets, neuron 0.19.  
- **very light (11 GeV), very narrow, leading particle 45% of pT, high pT** — 12.2% of jets, neuron 0.69.  
- **medium-mass (65 GeV), average width, leading particle 44% of pT, high pT** — 10.2% of jets, neuron 1.59.  
- **medium-mass (58 GeV), average width, pT spread over several particles** — 8.8% of jets, neuron 2.25.  
- **very light (14 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 0.19.  
- **light (34 GeV), narrow, pT spread over several particles** — 5.6% of jets, neuron 1.40.  
- **very light (10 GeV), very narrow, pT spread over several particles** — 5.0% of jets, neuron 1.04.  
- **very light (13 GeV), very narrow, pT spread over several particles, high pT** — 1.3% of jets, neuron 3.11.  

### neuron 2: light jet with hard trailing particles (major)

- **What it measures:** Pushed down for large angularity (LHA > 0.119, its strongest term), for a soft 8th-hardest particle (pT_7 < 53.4 GeV) and for very light jets (mass < 36.4 GeV), and pushed up for mass < 69.4 GeV and total pT < 781 GeV; overall it falls steeply with jet mass (rank correlation -0.861). Gluons (3.22) and quarks (3.02) sit highest, well above W (1.43), Z (1.16) and top (1.05) jets.
- *computed — its value:* largest for g (3.22), then q (3.02), then W (1.43), then Z (1.16), then t (1.05); it separates g jets from the rest best (AUC 0.76: large for g)
- **How the class scores use it:** Since gluons sit highest on it, the g score reads it as evidence for a gluon: it raises the g score, where it is the largest input (+32%). It does not enter the q, W, Z or t scores, even though quarks sit almost as high as gluons; quarks are pulled out of the g score by neuron 5 instead.
- *computed — used by:* raises the score of g (+32%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.21
if LHA > 0.119: z += -11.60 × (LHA − 0.119)
if pt_7 < 53.40: z += -0.061 × (53.40 − pt_7)
if mass < 36.40: z += -0.087 × (36.40 − mass)
if mass < 69.40: z += 0.023 × (69.40 − mass)
if pt_7 < 43.40: z += 0.064 × (43.40 − pt_7)
if sum_pt < 781: z += 0.005 × (781 − sum_pt)
if pt_7 > 30.40: z += 0.064 × (pt_7 − 30.40)
if mass < 36.80 and lam2 < 0.0011: z += 37.10 × (36.80 − mass) × (0.0011 − lam2)
if lam1 < 0.0058: z += 135 × (0.0058 − lam1)
if log_sum_pt < 6.49: z += 3.81 × (6.49 − log_sum_pt)
if lam1 < 0.0033: z += 261 × (0.0033 − lam1)
if pt_7 > 30.60 and C2 < 0.051: z += -1.10 × (pt_7 − 30.60) × (0.051 − C2)
if pt_7 > 30.50 and max_dr > 0.093: z += -0.536 × (pt_7 − 30.50) × (max_dr − 0.093)
if lam1 < 0.006 and max_dr > 0.080: z += -2270 × (0.006 − lam1) × (max_dr − 0.080)
if log_sum_pt > 6.90: z += -8.80 × (log_sum_pt − 6.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **light (48 GeV), average width, pT spread over several particles** — 15.5% of jets, neuron 1.28.  
- **medium-mass (65 GeV), average width, pT spread over several particles** — 14.6% of jets, neuron 0.29.  
- **very light (8 GeV), very narrow, pT spread over several particles** — 12.2% of jets, neuron 3.32.  
- **very light (9 GeV), very narrow, pT spread over several particles** — 9.7% of jets, neuron 3.97.  
- **medium-mass (58 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 1.01.  
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 8.5% of jets, neuron 0.23.  
- **very light (9 GeV), very narrow, leading particle 47% of pT, high pT** — 8.3% of jets, neuron 3.21.  
- **light (42 GeV), narrow, leading particle 45% of pT, high pT** — 7.9% of jets, neuron 1.20.  
- **light (50 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 2.56.  
- **very light (15 GeV), narrow, pT spread over several particles, low pT** — 6.0% of jets, neuron 4.34.  

### neuron 5: quark-likeness: pT in few particles (major)

- **What it measures:** Pushed up mainly for a soft 8th-hardest particle (pT_7 < 54.7 GeV, its strongest term), for small angularity (LHA < 0.224) and small e2 (< 0.0346); it rises with the summed pT of the 3 hardest particles and falls with z_7 (rank correlations 0.792 and -0.785). Quarks sit far highest (5.67); W (2.06) and Z (2.03) are in the middle, gluons (1.43) and tops (0.41) lowest.
- *computed — its value:* largest for q (5.67), then W (2.06), then Z (2.03), then g (1.43), then t (0.41); it separates q jets from the rest best (AUC 0.76: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops well below them, it raises the q score (+6%) and lowers the g score (-16%) and the t score (-16%). It does not enter the W or Z scores.
- *computed — used by:* raises the score of q (+6%); lowers the score of g (-16%), t (-16%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.779
if pt_7 < 54.70: z += 0.079 × (54.70 − pt_7)
if log_sum_pt > 6.55 and lam1 < 0.012: z += -985 × (log_sum_pt − 6.55) × (0.012 − lam1)
if z_7 < 0.071 and lam2 < 0.0012: z += 37700 × (0.071 − z_7) × (0.0012 − lam2)
if log_sum_pt > 6.55: z += 8.09 × (log_sum_pt − 6.55)
if LHA < 0.224: z += 20.00 × (0.224 − LHA)
if LHA < 0.228 and log_sum_pt < 6.81: z += -108 × (0.228 − LHA) × (6.81 − log_sum_pt)
if width < 0.0025: z += 736 × (0.0025 − width)
if e2 < 0.035: z += 39.80 × (0.035 − e2)
if sum_pt_top2 < 539: z += -0.0027 × (539 − sum_pt_top2)
if dr_0 < 0.020: z += -113 × (0.020 − dr_0)
if z_6 < 0.051: z += 61.00 × (0.051 − z_6)
if z_7 < 0.075 and sum_pt < 818: z += -0.215 × (0.075 − z_7) × (818 − sum_pt)
if z_7 < 0.054 and centroid_offset < 0.012: z += -5810 × (0.054 − z_7) × (0.012 − centroid_offset)
if LHA < 0.216 and centroid_offset > 0.0024: z += -1910 × (0.216 − LHA) × (centroid_offset − 0.0024)
if log_sum_pt > 6.83: z += -26.90 × (log_sum_pt − 6.83)
if log_sum_pt > 6.58 and dr_0 < 0.021: z += 371 × (log_sum_pt − 6.58) × (0.021 − dr_0)
if e2 < 0.033 and lam2 < 7.7e-05: z += 363000 × (0.033 − e2) × (7.7e-05 − lam2)
if sum_pt > 861 and centroid_offset < 0.012: z += 1.39 × (sum_pt − 861) × (0.012 − centroid_offset)
if z_7 < 0.023: z += 217 × (0.023 − z_7)
if z_7 < 0.032 and pt_5 < 30.90: z += -11.40 × (0.032 − z_7) × (30.90 − pt_5)
if log_sum_pt > 6.57 and mean_phi2 < 0.00015: z += 28800 × (log_sum_pt − 6.57) × (0.00015 − mean_phi2)
if log_sum_pt > 6.90 and centroid_offset < 0.018: z += -2200 × (log_sum_pt − 6.90) × (0.018 − centroid_offset)
if z_6 < 0.028 and n_dr_0p2_0p4 < 1.93: z += 73.50 × (0.028 − z_6) × (1.93 − n_dr_0p2_0p4)
if log_sum_pt > 6.58 and mean_eta2 < 9.1e-05: z += 43800 × (log_sum_pt − 6.58) × (9.1e-05 − mean_eta2)
if mean_phi2 < 0.014 and pt_5 < 24.60: z += 22.90 × (0.014 − mean_phi2) × (24.60 − pt_5)
if z_6 < 0.050 and e2_sq > 0.0022: z += -9060 × (0.050 − z_6) × (e2_sq − 0.0022)
if sum_pt_top5 > 682 and D2 < 1.55: z += -0.0073 × (sum_pt_top5 − 682) × (1.55 − D2)
if log_sum_pt > 6.88 and planar_flow < 0.044: z += -1180 × (log_sum_pt − 6.88) × (0.044 − planar_flow)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (54 GeV), wide, pT spread over several particles** — 30.5% of jets, neuron 0.15.  
- **medium-mass (54 GeV), wide, pT spread over several particles** — 22.1% of jets, neuron 0.85.  
- **medium-mass (57 GeV), average width, leading particle 42% of pT, high pT** — 14.2% of jets, neuron 3.77.  
- **very light (12 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 1.00.  
- **very light (9 GeV), very narrow, pT spread over several particles, high pT** — 9.0% of jets, neuron 5.40.  
- **very light (9 GeV), very narrow, leading particle 44% of pT, high pT** — 5.3% of jets, neuron 11.34.  
- **very light (7 GeV), very narrow, pT spread over several particles, low pT** — 5.2% of jets, neuron 0.27.  
- **light (26 GeV), very narrow, leading particle 42% of pT, high pT** — 1.8% of jets, neuron 3.87.  
- **light (22 GeV), very narrow, leading particle 58% of pT, high pT** — 1.8% of jets, neuron 13.55.  
- **light (21 GeV), very narrow, leading particle 47% of pT, high pT** — 0.5% of jets, neuron 1.91.  

### neuron 6: broad, off-centre spread (major)

- **What it measures:** Switched off for narrow jets (width < 0.0132 and girth2 < 0.00869 both push it down) and pushed up by small e2 (< 0.0501); it follows the pT share at 0.2 ≤ ΔR < 0.4, lam1 and the offset of the pT centroid from the axis (rank correlations 0.485, 0.464, 0.443). Tops sit far highest (3.82), then gluons (1.59) and quarks (0.72), with Z (0.41) and W (0.08) jets lowest.
- *computed — its value:* largest for t (3.82), then g (1.59), then q (0.72), then Z (0.41), then W (0.08); it separates t jets from the rest best (AUC 0.77: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so it lowers the W score (-13%) and the Z score (-20%). It raises the g score (+6%) and the q score (+10%), since among non-top jets a broad spread is QCD-like rather than boson-like; it does not enter the t score, which gets its width information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+6%), q (+10%); lowers the score of W (-13%), Z (-20%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = 8.53
if width < 0.013: z += -664 × (0.013 − width)
if girth2 < 0.0087: z += -1030 × (0.0087 − girth2)
if mass_over_sum_pt > 0.0066: z += -79.30 × (mass_over_sum_pt − 0.0066)
if e2 < 0.050: z += 132 × (0.050 − e2)
if centroid_offset > 0.0082 and lam2 < 0.0035: z += 45800 × (centroid_offset − 0.0082) × (0.0035 − lam2)
if centroid_offset > 0.019: z += -205 × (centroid_offset − 0.019)
if girth2_top2 < 0.0095: z += 147 × (0.0095 − girth2_top2)
if log_sum_pt < 6.69: z += -4.64 × (6.69 − log_sum_pt)
if max_dr < 0.172: z += -12.80 × (0.172 − max_dr)
if mass < 49.80 and z_dr_0p05_0p1 < 0.750: z += 0.062 × (49.80 − mass) × (0.750 − z_dr_0p05_0p1)
if lam2 < 0.00072 and z_dr_0p2_0p4 < 0.199: z += -6250 × (0.00072 − lam2) × (0.199 − z_dr_0p2_0p4)
if D2 < 1.66: z += 1.16 × (1.66 − D2)
if log_sum_pt < 6.56: z += 5.67 × (6.56 − log_sum_pt)
if girth > 0.088: z += 77.00 × (girth − 0.088)
if log_sum_pt < 6.70 and pt_7 < 46.50: z += 0.242 × (6.70 − log_sum_pt) × (46.50 − pt_7)
if lam2 > 0.0035: z += -2560 × (lam2 − 0.0035)
if pt1_dr01 < 5.76: z += -0.109 × (5.76 − pt1_dr01)
if LHA > 0.313 and eccentricity > 0.878: z += 418 × (LHA − 0.313) × (eccentricity − 0.878)
if z_dr_0_0p05 > 0.755: z += -3.02 × (z_dr_0_0p05 − 0.755)
if centroid_offset > 0.0032 and planar_flow > 0.018: z += 85.10 × (centroid_offset − 0.0032) × (planar_flow − 0.018)
if D2 < 1.66 and pt_4 < 89.20: z += -0.016 × (1.66 − D2) × (89.20 − pt_4)
if sum_pt_top5 > 837: z += -0.028 × (sum_pt_top5 − 837)
if max_dr > 0.172: z += 12.20 × (max_dr − 0.172)
if log_sum_pt < 6.68 and mean_eta2 > 0.0043: z += 417 × (6.68 − log_sum_pt) × (mean_eta2 − 0.0043)
if sum_pt > 987: z += 0.048 × (sum_pt − 987)
if centroid_offset > 0.019 and mean_phi2 < 0.0088: z += 9700 × (centroid_offset − 0.019) × (0.0088 − mean_phi2)
if C2 > 0.0034 and pt_7 > 28.90: z += 1.12 × (C2 − 0.0034) × (pt_7 − 28.90)
if mass_over_sum_pt > 0.0063 and tau32 < 0.524: z += -24.60 × (mass_over_sum_pt − 0.0063) × (0.524 − tau32)
if girth2_top5 > 0.0021 and z_dr_0p05_0p1 > 0.275: z += 187 × (girth2_top5 − 0.0021) × (z_dr_0p05_0p1 − 0.275)
if pt_6 < 24.00: z += 0.226 × (24.00 − pt_6)
if log_sum_pt < 6.71 and pt_6 < 36.70: z += 0.270 × (6.71 − log_sum_pt) × (36.70 − pt_6)
if girth2_top5 > 0.011 and mean_eta > 0.014: z += -9630 × (girth2_top5 − 0.011) × (mean_eta − 0.014)
if e2 < 0.051 and z_dr_0p1_0p2 > 0.156: z += 410 × (0.051 − e2) × (z_dr_0p1_0p2 − 0.156)
if centroid_offset > 0.050: z += 112 × (centroid_offset − 0.050)
if z_4 < 0.038: z += -185 × (0.038 − z_4)
if pt_4 < 31.10: z += 0.261 × (31.10 − pt_4)
if log_sum_pt < 6.69 and z_7 < 0.049: z += 296 × (6.69 − log_sum_pt) × (0.049 − z_7)
if sum_pt > 989 and pt_6 > 29.90: z += -0.00034 × (sum_pt − 989) × (pt_6 − 29.90)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 32.3% of jets, neuron 0.07.  
- **medium-mass (58 GeV), average width, pT spread over several particles** — 24.2% of jets, neuron 0.52.  
- **light (43 GeV), narrow, pT spread over several particles** — 17.7% of jets, neuron 0.25.  
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 7.0% of jets, neuron 5.02.  
- **very light (17 GeV), narrow, pT spread over several particles** — 5.2% of jets, neuron 1.99.  
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 4.3% of jets, neuron 6.89.  
- **medium-mass (56 GeV), very wide, pT spread over several particles, low pT** — 4.3% of jets, neuron 5.89.  
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.09.  
- **medium-mass (61 GeV), very wide, pT spread over several particles, low pT** — 1.5% of jets, neuron 7.56.  
- **light (23 GeV), very narrow, leading particle 46% of pT, high pT** — 1.2% of jets, neuron 0.78.  

### neuron 7: two-prong shape of W/Z size (major)

- **What it measures:** Built from many small size steps: pushed down for very narrow jets (lam1 < 0.00846, girth < 0.0883, width < 0.00544, max ΔR < 0.158) and for wider ones (girth2 > 0.00751), and pushed up for mass/pT above 0.0726 and 0.0842 but down again above 0.108, so it peaks in a W/Z-like mass/pT window; it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.538, -0.538, -0.423). Z jets sit highest (3.19), then W (1.66), with tops (0.88), gluons (0.78) and quarks (0.40) low.
- *computed — its value:* largest for Z (3.19), then W (1.66), then t (0.88), then g (0.78), then q (0.40); it separates Z jets from the rest best (AUC 0.83: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the W score (+9%) and the Z score (+26%), where it is the largest input. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+9%), Z (+26%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 11.30
if girth2 > 0.0015: z += -680 × (girth2 − 0.0015)
if mass_over_sum_pt > 0.108: z += -576 × (mass_over_sum_pt − 0.108)
if lam1 < 0.0085: z += -707 × (0.0085 − lam1)
if girth2 > 0.0075: z += -1030 × (girth2 − 0.0075)
if girth < 0.088: z += -63.80 × (0.088 − girth)
if width < 0.0054: z += -1090 × (0.0054 − width)
if max_dr < 0.158: z += -32.80 × (0.158 − max_dr)
if max_dr < 0.157 and z_dr_0p05_0p1 < 0.671: z += 46.10 × (0.157 − max_dr) × (0.671 − z_dr_0p05_0p1)
if e2 > 0.017: z += -80.40 × (e2 − 0.017)
if mass_over_sum_pt > 0.073: z += 87.70 × (mass_over_sum_pt − 0.073)
if C2 < 0.067: z += -28.20 × (0.067 − C2)
if centroid_offset < 0.040 and sum_pt > 567: z += 0.229 × (0.040 − centroid_offset) × (sum_pt − 567)
if mass_over_sum_pt > 0.084: z += 109 × (mass_over_sum_pt − 0.084)
if mass < 29.50: z += 0.136 × (29.50 − mass)
if e2 < 0.025: z += 132 × (0.025 − e2)
if max_dr < 0.196 and z_dr_0p05_0p1 > 0.050: z += 40.10 × (0.196 − max_dr) × (z_dr_0p05_0p1 − 0.050)
if pt_7 < 49.20 and planar_flow < 0.686: z += -0.091 × (49.20 − pt_7) × (0.686 − planar_flow)
if width < 0.0058 and n_dr_0p1_0p2 < 2.77: z += 87.70 × (0.0058 − width) × (2.77 − n_dr_0p1_0p2)
if girth2 > 0.0079 and log_sum_pt > 6.18: z += -1430 × (girth2 − 0.0079) × (log_sum_pt − 6.18)
if lam1 < 0.0081 and D2 < 1.14: z += -1160 × (0.0081 − lam1) × (1.14 − D2)
if centroid_offset < 0.021: z += -53.50 × (0.021 − centroid_offset)
if girth2 > 0.004 and eccentricity > 0.951: z += 6120 × (girth2 − 0.004) × (eccentricity − 0.951)
if mass > 80.40: z += -0.316 × (mass − 80.40)
if width < 0.00055: z += -3790 × (0.00055 − width)
if girth2_top2 < 0.001 and log_sum_pt > 6.40: z += -3890 × (0.001 − girth2_top2) × (log_sum_pt − 6.40)
if mass_over_sum_pt < 0.131 and z_dr_0p05_0p1 > 0.277: z += -36.30 × (0.131 − mass_over_sum_pt) × (z_dr_0p05_0p1 − 0.277)
if e2 < 0.039 and D2 < 1.13: z += 287 × (0.039 − e2) × (1.13 − D2)
if centroid_offset > 0.031: z += 76.80 × (centroid_offset − 0.031)
if planar_flow < 0.191 and sum_pt > 621: z += 0.012 × (0.191 − planar_flow) × (sum_pt − 621)
if mass_top5 > 53.50: z += 0.031 × (mass_top5 − 53.50)
if centroid_offset > 0.031 and pt_2 > 57.70: z += -1.44 × (centroid_offset − 0.031) × (pt_2 − 57.70)
if centroid_offset < 0.023 and C2 > 0.028: z += 1290 × (0.023 − centroid_offset) × (C2 − 0.028)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (55 GeV), average width, pT spread over several particles** — 28.8% of jets, neuron 3.20.  
- **very light (7 GeV), very narrow, pT spread over several particles, high pT** — 20.4% of jets, neuron 0.03.  
- **light (42 GeV), narrow, pT spread over several particles** — 17.3% of jets, neuron 1.68.  
- **very light (14 GeV), very narrow, pT spread over several particles** — 15.5% of jets, neuron 0.78.  
- **medium-mass (61 GeV), very wide, pT spread over several particles, low pT** — 5.5% of jets, neuron 0.85.  
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 0.00.  
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 3.7% of jets, neuron 0.00.  
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 2.5% of jets, neuron 0.00.  
- **heavy (114 GeV), very wide, pT spread over several particles** — 1.2% of jets, neuron 0.00.  
- **heavy (93 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.00.  

### neuron 9: narrow, light single-core jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00624, its strongest term, girth2 < 0.00415, max ΔR < 0.211 and centroid offset < 0.0194), and pushed down for small mass/pT (< 0.0767) and for the very narrowest jets (girth < 0.0535); it falls with lam1, width and girth (rank correlations -0.669, -0.659, -0.635). Quarks sit highest (6.56), then gluons (5.25), far above W (2.00), top (1.86) and Z (1.10) jets.
- *computed — its value:* largest for q (6.56), then g (5.25), then W (2.00), then t (1.86), then Z (1.10); it separates q jets from the rest best (AUC 0.78: large for q)
- **How the class scores use it:** High values mean a light-parton jet, so it raises the q score (its largest input, +49%) and the g score (+22%), and lowers the W score (-3%) and the Z score (-4%) slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+22%), q (+49%); lowers the score of W (-3%), Z (-4%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = -2.15
if width < 0.0062: z += 2140 × (0.0062 − width)
if mass_over_sum_pt < 0.077: z += -64.60 × (0.077 − mass_over_sum_pt)
if girth2 < 0.0042: z += 1000 × (0.0042 − girth2)
if max_dr < 0.211: z += 14.00 × (0.211 − max_dr)
if girth < 0.053: z += -90.80 × (0.053 − girth)
if centroid_offset < 0.019: z += 181 × (0.019 − centroid_offset)
if e2 < 0.020: z += 227 × (0.020 − e2)
if mass < 51.90 and log_sum_pt < 6.83: z += 0.227 × (51.90 − mass) × (6.83 − log_sum_pt)
if width < 0.0061 and centroid_offset > 0.0034: z += -44100 × (0.0061 − width) × (centroid_offset − 0.0034)
if mass < 39.00 and centroid_offset < 0.030: z += -3.62 × (39.00 − mass) × (0.030 − centroid_offset)
if mass < 27.90: z += -0.116 × (27.90 − mass)
if log_sum_pt > 6.34: z += -2.25 × (log_sum_pt − 6.34)
if lam1 < 0.0059 and mean_phi2 < 0.002: z += -122000 × (0.0059 − lam1) × (0.002 − mean_phi2)
if centroid_offset < 0.018 and z_4 > 0.044: z += -1900 × (0.018 − centroid_offset) × (z_4 − 0.044)
if e2 < 0.032 and dr01 < 0.056: z += 794 × (0.032 − e2) × (0.056 − dr01)
if max_dr < 0.238 and z_top5 < 0.903: z += -40.50 × (0.238 − max_dr) × (0.903 − z_top5)
if e2 < 0.019 and pt_7 < 53.00: z += -3.77 × (0.019 − e2) × (53.00 − pt_7)
if girth2 < 0.0044 and centroid_offset > 0.011: z += -59200 × (0.0044 − girth2) × (centroid_offset − 0.011)
if mass < 61.80 and planar_flow < 0.323: z += 0.144 × (61.80 − mass) × (0.323 − planar_flow)
if lam2 > 0.0011: z += 1060 × (lam2 − 0.0011)
if girth2_top2 < 0.00072 and z_7 > 0.023: z += -82700 × (0.00072 − girth2_top2) × (z_7 − 0.023)
if centroid_offset < 0.018 and pt_4 > 47.20: z += 2.33 × (0.018 − centroid_offset) × (pt_4 − 47.20)
if width < 0.00018: z += 11100 × (0.00018 − width)
if e2 < 0.035 and tau21 < 0.432: z += -191 × (0.035 − e2) × (0.432 − tau21)
if e2 < 0.021 and eccentricity > 0.904: z += -1110 × (0.021 − e2) × (eccentricity − 0.904)
if girth2 > 0.017: z += 138 × (girth2 − 0.017)
if girth2_top2 < 0.00067 and pt_7 > 32.40: z += 131 × (0.00067 − girth2_top2) × (pt_7 − 32.40)
if pt_5 < 35.80: z += 0.073 × (35.80 − pt_5)
if sum_pt > 853: z += -0.0051 × (sum_pt − 853)
if log_sum_pt > 6.48 and dr_5 < 0.026: z += 100 × (log_sum_pt − 6.48) × (0.026 − dr_5)
if sum_pt > 986 and dr_3 < 0.055: z += 0.511 × (sum_pt − 986) × (0.055 − dr_3)
if width < 0.0062 and C2 > 0.029: z += -9180 × (0.0062 − width) × (C2 − 0.029)
if girth2 > 0.019 and pt_7 < 29.10: z += 55.10 × (girth2 − 0.019) × (29.10 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (63 GeV), very wide, pT spread over several particles, low pT** — 22.1% of jets, neuron 0.45.  
- **medium-mass (63 GeV), average width, pT spread over several particles** — 17.4% of jets, neuron 0.68.  
- **very light (8 GeV), very narrow, leading particle 40% of pT, high pT** — 14.4% of jets, neuron 8.32.  
- **light (46 GeV), average width, pT spread over several particles** — 11.3% of jets, neuron 1.57.  
- **light (36 GeV), narrow, pT spread over several particles** — 7.7% of jets, neuron 2.65.  
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.7% of jets, neuron 8.58.  
- **light (25 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 7.30.  
- **very light (8 GeV), very narrow, pT spread over several particles** — 5.4% of jets, neuron 2.57.  
- **very light (10 GeV), very narrow, pT spread over several particles, low pT** — 4.2% of jets, neuron 1.88.  
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 4.0% of jets, neuron 5.82.  

### neuron 10: overall jet size (e2, width, mass) (major)

- **What it measures:** Grows steadily with e2 (its strongest term) and is pushed up for mass > 15.3 GeV and for mass/pT < 0.108, and down for a very thin jet (lam2 < 0.00352, lam1 < 0.00422); it follows width, e2 and mass/pT very closely (rank correlations 0.872, 0.871, 0.87). Tops sit highest (4.60), Z (1.74) and W (1.70) in the middle, gluons (0.97) and quarks (0.49) lowest.
- *computed — its value:* largest for t (4.60), then Z (1.74), then W (1.70), then g (0.97), then q (0.49); it separates t jets from the rest best (AUC 0.85: large for t)
- **How the class scores use it:** Large size marks a top, so it raises the t score (+20%); small size marks a quark, so it lowers the q score (-14%). It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+20%); lowers the score of q (-14%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.45
z += 62.20 × e2
if lam2 < 0.0035: z += -512 × (0.0035 − lam2)
if mass > 15.30: z += 0.058 × (mass − 15.30)
if mass_over_sum_pt < 0.108: z += 20.10 × (0.108 − mass_over_sum_pt)
if lam1 < 0.0042: z += -653 × (0.0042 − lam1)
if mass_over_sum_pt_sq < 0.004: z += 376 × (0.004 − mass_over_sum_pt_sq)
if centroid_offset < 0.038: z += 23.30 × (0.038 − centroid_offset)
if girth2_top2 < 0.004: z += 237 × (0.004 − girth2_top2)
if girth2 > 0.0087: z += 201 × (girth2 − 0.0087)
if girth2 < 0.0016: z += -1030 × (0.0016 − girth2)
if tau21 < 0.385: z += -3.11 × (0.385 − tau21)
if log_sum_pt > 6.70: z += -11.70 × (log_sum_pt − 6.70)
if eccentricity > 0.898 and z_dr_0p2_0p4 < 0.057: z += 155 × (eccentricity − 0.898) × (0.057 − z_dr_0p2_0p4)
if girth2_top5 < 0.0023: z += 464 × (0.0023 − girth2_top5)
if lam2 > 0.0002: z += 755 × (lam2 − 0.0002)
if mass > 53.20: z += -0.051 × (mass − 53.20)
if n_dr_0p05_0p1 < 3.02: z += 0.179 × (3.02 − n_dr_0p05_0p1)
if LHA > 0.347: z += -27.10 × (LHA − 0.347)
if LHA > 0.303: z += -10.30 × (LHA − 0.303)
if lam1 < 0.0043 and sum_pt > 994: z += -7.66 × (0.0043 − lam1) × (sum_pt − 994)
if lam2 > 0.00021 and n_pt_above_50 < 8.04: z += -61.20 × (lam2 − 0.00021) × (8.04 − n_pt_above_50)
if lam2 > 0.00023 and D2 < 2.00: z += -192 × (lam2 − 0.00023) × (2.00 − D2)
if centroid_offset > 0.037: z += 17.20 × (centroid_offset − 0.037)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (51 GeV), average width, pT spread over several particles** — 30.9% of jets, neuron 2.34.  
- **very light (9 GeV), very narrow, pT spread over several particles** — 22.6% of jets, neuron 0.09.  
- **light (30 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 1.17.  
- **very light (10 GeV), very narrow, leading particle 43% of pT, high pT** — 9.5% of jets, neuron 0.00.  
- **medium-mass (70 GeV), average width, leading particle 42% of pT, high pT** — 7.5% of jets, neuron 1.36.  
- **medium-mass (74 GeV), very wide, pT spread over several particles, low pT** — 6.4% of jets, neuron 3.81.  
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 3.3% of jets, neuron 6.16.  
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.1% of jets, neuron 7.49.  
- **medium-mass (88 GeV), very wide, pT spread over several particles, low pT** — 2.0% of jets, neuron 10.77.  
- **very light (15 GeV), very narrow, leading particle 44% of pT, high pT** — 0.9% of jets, neuron 0.00.  

### neuron 11: compact, centred, non-top jet (major)

- **What it measures:** Pushed up for compact jets (width < 0.0086, its strongest term) with a pT centroid close to the axis (centroid offset < 0.0499), and pushed down for girth < 0.0879; it rises with eccentricity and total pT (rank correlations 0.388 and 0.375). W jets sit highest (4.33), then Z (2.77), gluons (1.67) and quarks (1.63); tops are lowest (0.65) and zero for 78.0% of them.
- *computed — its value:* largest for W (4.33), then Z (2.77), then g (1.67), then q (1.63), then t (0.65); it separates W jets from the rest best (AUC 0.84: large for W)
- **How the class scores use it:** It is the largest input of the W score, which it raises (+25%), since W jets sit highest on it. It does not enter the g, q, Z or t scores; in particular the Z score does not use it although Z jets sit second on it.
- *computed — used by:* raises the score of W (+25%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.39
if width < 0.0086: z += 1390 × (0.0086 − width)
if centroid_offset < 0.050: z += 124 × (0.050 − centroid_offset)
if girth < 0.088: z += -103 × (0.088 − girth)
if girth > 0.075: z += -189 × (girth − 0.075)
if girth > 0.076 and n_pt_above_50 < 7.11: z += 58.00 × (girth − 0.076) × (7.11 − n_pt_above_50)
if centroid_offset < 0.049 and log_sum_pt < 6.80: z += -166 × (0.049 − centroid_offset) × (6.80 − log_sum_pt)
if planar_flow < 0.268: z += 10.20 × (0.268 − planar_flow)
if girth > 0.099 and n_pt_above_50 < 6.38: z += -76.70 × (girth − 0.099) × (6.38 − n_pt_above_50)
if planar_flow < 0.235 and width > 0.0061: z += -3660 × (0.235 − planar_flow) × (width − 0.0061)
if centroid_offset > 0.014: z += -97.10 × (centroid_offset − 0.014)
if width < 0.0037: z += -486 × (0.0037 − width)
if e2_sq < 0.0057: z += -227 × (0.0057 − e2_sq)
if girth > 0.076 and pt_7 < 40.30: z += -8.16 × (girth − 0.076) × (40.30 − pt_7)
if centroid_offset < 0.051 and pt_7 < 48.50: z += -1.05 × (0.051 − centroid_offset) × (48.50 − pt_7)
if LHA < 0.187: z += -24.30 × (0.187 − LHA)
if log_sum_pt < 6.69: z += 2.54 × (6.69 − log_sum_pt)
if log_sum_pt < 6.72 and z_dr_0p2_0p4 < 0.055: z += 48.60 × (6.72 − log_sum_pt) × (0.055 − z_dr_0p2_0p4)
if planar_flow < 0.260 and mass < 70.60: z += -0.137 × (0.260 − planar_flow) × (70.60 − mass)
if girth > 0.076 and z_dr_0p1_0p2 < 0.197: z += -978 × (girth − 0.076) × (0.197 − z_dr_0p1_0p2)
if C2 < 0.033: z += -18.80 × (0.033 − C2)
if centroid_offset < 0.048 and mean_phi2 < 0.0015: z += -9290 × (0.048 − centroid_offset) × (0.0015 − mean_phi2)
if mass < 22.30: z += 0.047 × (22.30 − mass)
if log_sum_pt < 6.71 and dr_0 < 0.119: z += -19.70 × (6.71 − log_sum_pt) × (0.119 − dr_0)
if LHA < 0.170 and z_7 < 0.028: z += 831 × (0.170 − LHA) × (0.028 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (9 GeV), very narrow, pT spread over several particles, high pT** — 25.4% of jets, neuron 1.38.  
- **light (48 GeV), average width, pT spread over several particles** — 21.4% of jets, neuron 4.29.  
- **medium-mass (59 GeV), average width, pT spread over several particles** — 18.1% of jets, neuron 2.74.  
- **light (21 GeV), very narrow, pT spread over several particles** — 16.9% of jets, neuron 2.68.  
- **medium-mass (64 GeV), very wide, pT spread over several particles, low pT** — 6.5% of jets, neuron 0.02.  
- **medium-mass (79 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 0.00.  
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 2.8% of jets, neuron 0.00.  
- **medium-mass (87 GeV), very wide, pT spread over several particles** — 2.8% of jets, neuron 0.00.  
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00.  
- **medium-mass (72 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.00.  

### neuron 13: compactness (narrower than a top) (major)

- **What it measures:** Driven mostly by girth < 0.149 (its strongest term), lam1 < 0.016 and width < 0.0132, all pushing it up, i.e. by the jet being narrower than a typical top; small e2 (< 0.0501) pulls it down. It falls with max ΔR, lam1 and width (rank correlations -0.62, -0.596, -0.594); W (4.87), quarks (4.68), Z (4.41) and gluons (4.38) all sit high, tops far lower (1.42, zero for 51.4% of them).
- *computed — its value:* largest for W (4.87), then q (4.68), then Z (4.41), then g (4.38), then t (1.42); it separates t jets from the rest best (AUC 0.14: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-46%); it also raises the W score (+8%) and the Z score (+9%). It does not enter the g or q scores.
- *computed — used by:* raises the score of W (+8%), Z (+9%); lowers the score of t (-46%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.629
if girth < 0.149: z += 59.30 × (0.149 − girth)
if e2 < 0.050: z += -121 × (0.050 − e2)
if lam1 < 0.016: z += 256 × (0.016 − lam1)
if centroid_offset < 0.038: z += 60.90 × (0.038 − centroid_offset)
if width < 0.013: z += 163 × (0.013 − width)
if lam2 < 0.00031 and centroid_offset < 0.050: z += -148000 × (0.00031 − lam2) × (0.050 − centroid_offset)
if sum_pt_top5 > 903: z += 0.259 × (sum_pt_top5 − 903)
if lam1 < 0.017 and centroid_offset < 0.037: z += -3430 × (0.017 − lam1) × (0.037 − centroid_offset)
if girth < 0.147 and log_sum_pt < 6.80: z += -48.20 × (0.147 − girth) × (6.80 − log_sum_pt)
if sum_pt > 988: z += 0.190 × (sum_pt − 988)
if girth < 0.152 and pt_7 < 38.10: z += -0.966 × (0.152 − girth) × (38.10 − pt_7)
if lam2 < 0.0003: z += 3140 × (0.0003 − lam2)
if z_top5_slots > 0.931: z += -602 × (z_top5_slots − 0.931)
if sum_pt_top5 > 657 and pt_7 < 42.10: z += 0.00063 × (sum_pt_top5 − 657) × (42.10 − pt_7)
if width < 0.008: z += -129 × (0.008 − width)
if sum_pt_top5 > 902 and D2 < 3.86: z += 0.091 × (sum_pt_top5 − 902) × (3.86 − D2)
if tau21 < 0.515 and max_dr > 0.015: z += -16.50 × (0.515 − tau21) × (max_dr − 0.015)
if z_7 < 0.029: z += -269 × (0.029 − z_7)
if lam1 < 0.017 and pt_6 < 56.90: z += -1.69 × (0.017 − lam1) × (56.90 − pt_6)
if mass > 53.50: z += -0.050 × (mass − 53.50)
if sum_pt > 988 and n_pt_above_50 > 6.00: z += -0.126 × (sum_pt − 988) × (n_pt_above_50 − 6.00)
if sum_pt_top5 > 660 and z_7 > 0.022: z += -0.805 × (sum_pt_top5 − 660) × (z_7 − 0.022)
if sum_pt_top5 > 902 and n_pt_above_50 > 6.00: z += -0.256 × (sum_pt_top5 − 902) × (n_pt_above_50 − 6.00)
if sum_pt_top5 > 840 and n_pt_above_50 > 2.02: z += 0.011 × (sum_pt_top5 − 840) × (n_pt_above_50 − 2.02)
if sum_pt > 989 and D2 < 3.94: z += 0.029 × (sum_pt − 989) × (3.94 − D2)
if e2 < 0.049 and pt_dispersion > 0.397: z += 131 × (0.049 − e2) × (pt_dispersion − 0.397)
if pt_6 < 31.30: z += -0.117 × (31.30 − pt_6)
if C2 > 0.067: z += -57.90 × (C2 − 0.067)
if girth > 0.102 and pt_4 > 81.40: z += -25.00 × (girth − 0.102) × (pt_4 − 81.40)
if sum_pt_top5 > 658 and tau32 < 0.362: z += -0.141 × (sum_pt_top5 − 658) × (0.362 − tau32)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **light (21 GeV), very narrow, pT spread over several particles** — 48.8% of jets, neuron 4.87.  
- **medium-mass (63 GeV), wide, pT spread over several particles** — 45.9% of jets, neuron 2.88.  
- **light (26 GeV), very narrow, leading particle 56% of pT, high pT** — 2.3% of jets, neuron 1.72.  
- **light (30 GeV), very narrow, leading particle 42% of pT, high pT** — 1.2% of jets, neuron 7.39.  
- **light (22 GeV), very narrow, leading particle 59% of pT, high pT** — 0.9% of jets, neuron 6.36.  
- **light (27 GeV), very narrow, leading particle 51% of pT, high pT** — 0.5% of jets, neuron 7.93.  
- **light (29 GeV), very narrow, pT spread over several particles, high pT** — 0.2% of jets, neuron 7.48.  
- **light (33 GeV), very narrow, leading particle 57% of pT, high pT** — 0.1% of jets, neuron 8.79.  
- **light (37 GeV), very narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 6.15.  

### neuron 14: intermediate width (Z-sized spread) (major)

- **What it measures:** Pushed down for the narrowest jets (girth < 0.0872, width < 0.00609 and < 0.00747) but up for girth2 < 0.0133, e2 < 0.0443 and lam1 > 0.00428, so it peaks for jets of intermediate width, a bit wider than a typical W; it falls with τ21 and rises with eccentricity and the pT share at 0.1 ≤ ΔR < 0.2 (rank correlations -0.391, 0.383, 0.355). Z jets sit highest (1.06), well above tops (0.29), W (0.15), gluons (0.13) and quarks (0.10).
- *computed — its value:* largest for Z (1.06), then t (0.29), then W (0.15), then g (0.13), then q (0.10); it separates Z jets from the rest best (AUC 0.78: large for Z)
- **How the class scores use it:** It is the W/Z separator: Z jets sit high and W jets low on it, so it raises the Z score (+5%) and lowers the W score (-8%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+5%); lowers the score of W (-8%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.305
if e2 < 0.044: z += 214 × (0.044 − e2)
if girth < 0.087: z += -109 × (0.087 − girth)
if girth2 < 0.013: z += 463 × (0.013 − girth2)
if width < 0.0061: z += -1050 × (0.0061 − width)
if width < 0.0075: z += -662 × (0.0075 − width)
if lam1 > 0.0043: z += 649 × (lam1 − 0.0043)
if C2 < 0.067: z += -47.20 × (0.067 − C2)
if width < 0.0075 and n_dr_0p1_0p2 < 3.00: z += 166 × (0.0075 − width) × (3.00 − n_dr_0p1_0p2)
if lam1 > 0.006: z += -624 × (lam1 − 0.006)
if mass < 69.50: z += -0.046 × (69.50 − mass)
if mass < 86.40: z += 0.024 × (86.40 − mass)
if LHA < 0.302: z += -13.60 × (0.302 − LHA)
if lam1 > 0.0042 and D2 > 0.451: z += -708 × (lam1 − 0.0042) × (D2 − 0.451)
if e2_sq < 0.0057: z += 385 × (0.0057 − e2_sq)
if max_dr < 0.080: z += -61.40 × (0.080 − max_dr)
if girth2 < 0.013 and n_dr_0p1_0p2 < 2.98: z += -43.60 × (0.013 − girth2) × (2.98 − n_dr_0p1_0p2)
if max_dr < 0.177: z += -11.80 × (0.177 − max_dr)
if z_dr_0p05_0p1 < 0.582: z += 1.74 × (0.582 − z_dr_0p05_0p1)
if n_dr_0p05_0p1 < 5.09: z += -0.151 × (5.09 − n_dr_0p05_0p1)
if lam1 > 0.0023 and D2 > 0.452: z += 241 × (lam1 − 0.0023) × (D2 − 0.452)
if z_dr_0p05_0p1 < 0.601 and C2 < 0.067: z += -19.50 × (0.601 − z_dr_0p05_0p1) × (0.067 − C2)
if centroid_offset > 0.012: z += -37.90 × (centroid_offset − 0.012)
if girth < 0.034: z += -41.80 × (0.034 − girth)
if lam1 > 0.0059 and D2 > 1.68: z += -3670 × (lam1 − 0.0059) × (D2 − 1.68)
if z_dr_0_0p05 > 0.602: z += -1.33 × (z_dr_0_0p05 − 0.602)
if z_dr_0p05_0p1 < 0.588 and n_dr_0p1_0p2 < 3.00: z += 0.275 × (0.588 − z_dr_0p05_0p1) × (3.00 − n_dr_0p1_0p2)
if girth2 < 0.013 and eccentricity > 0.971: z += 3530 × (0.013 − girth2) × (eccentricity − 0.971)
if width < 0.0089 and D2 < 1.03: z += -493 × (0.0089 − width) × (1.03 − D2)
if centroid_offset > 0.050: z += -231 × (centroid_offset − 0.050)
if girth2 < 0.0044 and D2 < 0.884: z += -11800 × (0.0044 − girth2) × (0.884 − D2)
if tau21 < 0.134: z += 10.50 × (0.134 − tau21)
if planar_flow < 0.111 and max_dr < 0.159: z += -150 × (0.111 − planar_flow) × (0.159 − max_dr)
if planar_flow < 0.119 and centroid_offset > 0.0097: z += 441 × (0.119 − planar_flow) × (centroid_offset − 0.0097)
if lam1 > 0.0054 and max_dr < 0.161: z += 8400 × (lam1 − 0.0054) × (0.161 − max_dr)
if planar_flow < 0.115 and centroid_offset > 0.018: z += -656 × (0.115 − planar_flow) × (centroid_offset − 0.018)
if lam1 > 0.0023 and D2 > 1.63: z += -283 × (lam1 − 0.0023) × (D2 − 1.63)
if girth2 < 0.013 and centroid_offset > 0.031: z += -8450 × (0.013 − girth2) × (centroid_offset − 0.031)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles** — 28.5% of jets, neuron 0.00.  
- **medium-mass (57 GeV), average width, pT spread over several particles** — 22.2% of jets, neuron 1.02.  
- **light (47 GeV), average width, pT spread over several particles** — 17.1% of jets, neuron 0.52.  
- **light (27 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 0.10.  
- **medium-mass (71 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 0.16.  
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 5.0% of jets, neuron 0.00.  
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.7% of jets, neuron 0.09.  
- **light (30 GeV), very narrow, pT spread over several particles** — 1.7% of jets, neuron 0.00.  
- **medium-mass (66 GeV), very wide, pT spread over several particles, low pT** — 1.1% of jets, neuron 0.00.  
- **medium-mass (71 GeV), very wide, pT spread over several particles** — 0.3% of jets, neuron 0.00.  

### neuron 0: compact, massive, elongated two-prong jet (moderate)

- **What it measures:** Pushed down for the narrowest jets (girth < 0.0764, its strongest term) and for light jets (mass < 29.1 GeV and < 59.6 GeV), and pushed up for compact jets (girth2 < 0.0133, width < 0.00876); it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.569, -0.569, -0.558). Z (1.84) and W (1.77) jets sit highest, tops (0.43), gluons (0.21) and quarks (0.19) low; it is zero for 83.4% of gluons and 86.6% of quarks.
- *computed — its value:* largest for Z (1.84), then W (1.77), then t (0.43), then g (0.21), then q (0.19); it separates Z jets from the rest best (AUC 0.76: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, it raises the W score (+9%) and lowers the g score (-5%). It does not enter the q, Z or t scores; the Z score, although Z jets sit highest on it, relies on the related two-prong neuron 7 instead.
- *computed — used by:* raises the score of W (+9%); lowers the score of g (-5%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.847
if girth < 0.076: z += -122 × (0.076 − girth)
if girth2 < 0.013: z += 355 × (0.013 − girth2)
if mass < 29.10: z += -0.389 × (29.10 − mass)
if mass < 28.90 and dr_0 < 0.109: z += 3.42 × (28.90 − mass) × (0.109 − dr_0)
if width < 0.0088: z += 465 × (0.0088 − width)
if mass < 59.60: z += -0.072 × (59.60 − mass)
if width < 0.0043: z += -664 × (0.0043 − width)
if girth2_top3 < 0.0081: z += -162 × (0.0081 − girth2_top3)
if e2 < 0.021: z += 134 × (0.021 − e2)
if girth2_top3 < 0.0041: z += 380 × (0.0041 − girth2_top3)
if centroid_offset < 0.033: z += 32.20 × (0.033 − centroid_offset)
if mass < 64.00 and pt_7 < 40.10: z += -0.0021 × (64.00 − mass) × (40.10 − pt_7)
if sum_pt > 821: z += -0.013 × (sum_pt − 821)
if girth2 < 0.019 and eccentricity > 0.955: z += 1590 × (0.019 − girth2) × (eccentricity − 0.955)
if lam1 < 0.00046: z += 4130 × (0.00046 − lam1)
if z_dr_0_0p05 > 0.851: z += 5.86 × (z_dr_0_0p05 − 0.851)
if mass < 63.40 and centroid_offset > 0.011: z += 1.69 × (63.40 − mass) × (centroid_offset − 0.011)
if sum_pt_top5 > 698: z += 0.0083 × (sum_pt_top5 − 698)
if sum_pt > 904 and pt_7 > 29.40: z += -0.0025 × (sum_pt − 904) × (pt_7 − 29.40)
if mass_over_sum_pt_sq < 0.0081 and n_pt_above_50 < 7.90: z += 18.00 × (0.0081 − mass_over_sum_pt_sq) × (7.90 − n_pt_above_50)
if sum_pt > 889: z += -0.011 × (sum_pt − 889)
if lam1 < 0.0066 and D2 < 0.872: z += -1490 × (0.0066 − lam1) × (0.872 − D2)
if sum_pt > 876 and pt_7 < 28.80: z += 0.00097 × (sum_pt − 876) × (28.80 − pt_7)
if log_sum_pt > 6.63 and dr_4 < 0.077: z += 42.60 × (log_sum_pt − 6.63) × (0.077 − dr_4)
if mass < 56.00 and C2 > 0.024: z += 1.00 × (56.00 − mass) × (C2 − 0.024)
if mass < 29.70 and D2 < 0.874: z += -3.17 × (29.70 − mass) × (0.874 − D2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (56 GeV), average width, pT spread over several particles** — 23.6% of jets, neuron 2.47.  
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 19.0% of jets, neuron 0.13.  
- **very light (6 GeV), very narrow, pT spread over several particles, high pT** — 17.6% of jets, neuron 0.00.  
- **light (46 GeV), narrow, pT spread over several particles** — 14.8% of jets, neuron 1.50.  
- **very light (8 GeV), very narrow, pT spread over several particles** — 9.3% of jets, neuron 0.02.  
- **light (36 GeV), narrow, pT spread over several particles** — 7.6% of jets, neuron 0.83.  
- **very light (17 GeV), very narrow, pT spread over several particles** — 7.5% of jets, neuron 0.01.  
- **light (23 GeV), very narrow, pT spread over several particles, high pT** — 0.5% of jets, neuron 0.00.  
- **light (34 GeV), narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00.  

### neuron 3: radiation at wide angle (ΔR 0.2-0.4) (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00872 and girth2 < 0.0126 are its two strongest terms, both pushing down) and pushed up for e2 < 0.0445; it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.633 and 0.625). Tops sit clearly highest (2.61), then gluons (0.72) and quarks (0.35); Z jets are rarely on it (0.16) and W jets almost never (0.03, zero for 97.1% of them).
- *computed — its value:* largest for t (2.61), then g (0.72), then q (0.35), then Z (0.16), then W (0.03); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so it lowers the W score (-12%) and the Z score (-17%). It does not enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-12%), Z (-17%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.94
if girth2 < 0.0087: z += -1270 × (0.0087 − girth2)
if girth2 < 0.013: z += -535 × (0.013 − girth2)
if e2 < 0.044: z += 199 × (0.044 − e2)
if lam1 > 0.0035: z += -541 × (lam1 − 0.0035)
if mass_over_sum_pt > 0.067 and n_dr_0_0p05 < 4.63: z += 23.70 × (mass_over_sum_pt − 0.067) × (4.63 − n_dr_0_0p05)
if girth2 < 0.0039: z += 981 × (0.0039 − girth2)
if z_dr_0p1_0p2 < 0.458: z += 2.43 × (0.458 − z_dr_0p1_0p2)
if e2 > 0.027: z += -54.20 × (e2 − 0.027)
if centroid_offset > 0.015: z += 66.10 × (centroid_offset − 0.015)
if mass > 36.00 and eccentricity > 0.688: z += 0.099 × (mass − 36.00) × (eccentricity − 0.688)
if width > 0.018: z += 387 × (width − 0.018)
if max_dr > 0.140: z += 6.89 × (max_dr − 0.140)
if mass > 69.50: z += -0.079 × (mass − 69.50)
if C2 > 0.093: z += -165 × (C2 − 0.093)
if max_dr > 0.141 and dr_1 < 0.047: z += -673 × (max_dr − 0.141) × (0.047 − dr_1)
if centroid_offset > 0.011 and pt_7 > 25.20: z += 1.16 × (centroid_offset − 0.011) × (pt_7 − 25.20)
if girth2 < 0.008 and pt1_dr01 > 4.22: z += 21.90 × (0.008 − girth2) × (pt1_dr01 − 4.22)
if LHA > 0.312 and max_dr < 0.147: z += -1760 × (LHA − 0.312) × (0.147 − max_dr)
if mass_over_sum_pt > 0.069 and dr_7 < 0.043: z += -4920 × (mass_over_sum_pt − 0.069) × (0.043 − dr_7)
if mass_over_sum_pt > 0.090 and max_dr < 0.149: z += 10100 × (mass_over_sum_pt − 0.090) × (0.149 − max_dr)
if mass_over_sum_pt > 0.109 and dr_7 < 0.049: z += 12100 × (mass_over_sum_pt − 0.109) × (0.049 − dr_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (9 GeV), very narrow, pT spread over several particles** — 30.6% of jets, neuron 0.02.  
- **medium-mass (54 GeV), average width, pT spread over several particles** — 16.5% of jets, neuron 0.20.  
- **light (44 GeV), narrow, pT spread over several particles** — 13.3% of jets, neuron 0.27.  
- **light (26 GeV), narrow, pT spread over several particles** — 11.1% of jets, neuron 0.20.  
- **medium-mass (59 GeV), average width, pT spread over several particles** — 10.9% of jets, neuron 0.63.  
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 5.8% of jets, neuron 3.14.  
- **medium-mass (62 GeV), very wide, pT spread over several particles, low pT** — 5.5% of jets, neuron 3.70.  
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 4.5% of jets, neuron 3.37.  
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 1.6% of jets, neuron 4.09.  
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 0.2% of jets, neuron 2.30.  

### neuron 4: two-prong substructure (low D2, τ21) (moderate)

- **What it measures:** Pushed up for jets that are not too thin (girth2 > 0.00354, its strongest term) and for girth < 0.0766, and pushed down for wider jets (girth2 > 0.00739), for small e2_sq (< 0.00295) and for large mass/pT (above 0.09), so it favours jets with real mass packed into a compact shape; it falls with D2 and τ21 and rises with the pT share at 0.05 ≤ ΔR < 0.1 (rank correlations -0.644, -0.621, 0.577). Z (5.12) and W (4.25) jets sit highest, tops in the middle (3.01), gluons (2.34) and quarks (1.24) lowest.
- *computed — its value:* largest for Z (5.12), then W (4.25), then t (3.01), then g (2.34), then q (1.24); it separates q jets from the rest best (AUC 0.26: small for q)
- **How the class scores use it:** It raises the Z score (+10%) and the t score (+11%), and lowers the q score (-18%) and the g score (-4%), since quarks and gluons sit lowest on it while bosons and tops have hard substructure. It does not enter the W score, although W jets sit second on it.
- *computed — used by:* raises the score of Z (+10%), t (+11%); lowers the score of g (-4%), q (-18%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.765
if girth2 > 0.0035: z += 743 × (girth2 − 0.0035)
if mass_over_sum_pt > 0.108: z += 416 × (mass_over_sum_pt − 0.108)
if mass_over_sum_pt > 0.090: z += -254 × (mass_over_sum_pt − 0.090)
if girth < 0.077: z += 67.60 × (0.077 − girth)
if e2_sq < 0.0029: z += -1680 × (0.0029 − e2_sq)
if girth2 > 0.0074: z += -683 × (girth2 − 0.0074)
if mass_over_sum_pt > 0.108 and D2 < 3.95: z += -96.40 × (mass_over_sum_pt − 0.108) × (3.95 − D2)
if mass < 61.30: z += 0.062 × (61.30 − mass)
if e2 > 0.021: z += 114 × (e2 − 0.021)
if tau21 < 0.236: z += 20.90 × (0.236 − tau21)
if C2 > 0.014: z += -54.10 × (C2 − 0.014)
if centroid_offset < 0.016 and z_dr_0p05_0p1 < 0.541: z += -358 × (0.016 − centroid_offset) × (0.541 − z_dr_0p05_0p1)
if lam2 < 0.0003 and mass_top3 < 43.80: z += -107 × (0.0003 − lam2) × (43.80 − mass_top3)
if sum_pt < 760: z += 0.0039 × (760 − sum_pt)
if C2 > 0.067: z += -110 × (C2 − 0.067)
if tau21 < 0.229 and mass < 60.90: z += -0.618 × (0.229 − tau21) × (60.90 − mass)
if girth2_top2 < 0.011 and centroid_offset > 0.019: z += 10600 × (0.011 − girth2_top2) × (centroid_offset − 0.019)
if max_dr > 0.105 and eccentricity > 0.985: z += -927 × (max_dr − 0.105) × (eccentricity − 0.985)
if tau21 < 0.221 and planar_flow > 0.036: z += 45.70 × (0.221 − tau21) × (planar_flow − 0.036)
if girth2 > 0.019 and pt_6 > 62.00: z += -254 × (girth2 − 0.019) × (pt_6 − 62.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 36.1% of jets, neuron 1.24.  
- **medium-mass (58 GeV), average width, pT spread over several particles** — 25.1% of jets, neuron 7.06.  
- **light (41 GeV), narrow, pT spread over several particles** — 21.1% of jets, neuron 3.06.  
- **medium-mass (60 GeV), wide, pT spread over several particles, low pT** — 4.8% of jets, neuron 4.80.  
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.5% of jets, neuron 2.08.  
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 3.4% of jets, neuron 1.02.  
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 3.2% of jets, neuron 0.27.  
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.04.  
- **heavy (96 GeV), very wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.02.  

### neuron 8: narrow, centred single core (moderate)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00509, its strongest term) and narrow, centred ones (girth2 < 0.00673 with centroid offset < 0.0236), but pulled back for the very narrowest (girth < 0.0604 with width < 0.00498); it falls with centroid offset, width and lam1 (rank correlations -0.546, -0.524, -0.519). Quarks (1.32) and gluons (1.04) sit highest, W (0.50) in the middle, Z (0.22) and tops (0.13) low.
- *computed — its value:* largest for q (1.32), then g (1.04), then W (0.50), then Z (0.22), then t (0.13); it separates q jets from the rest best (AUC 0.73: large for q)
- **How the class scores use it:** A single narrow core is evidence against a two-prong W, so it lowers the W score (-5%); it raises the q score a little (+2%), where quarks sit highest, and raises the t score slightly (+3%), a small correction since tops sit lowest on it. It does not enter the g or Z scores.
- *computed — used by:* raises the score of q (+2%), t (+3%); lowers the score of W (-5%); does not (or hardly) enter the score of g, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.260
if width < 0.0051: z += 2730 × (0.0051 − width)
if girth < 0.060 and width < 0.005: z += -61400 × (0.060 − girth) × (0.005 − width)
if girth2 < 0.0067 and centroid_offset < 0.024: z += 29300 × (0.0067 − girth2) × (0.024 − centroid_offset)
if girth < 0.064 and lam2 < 0.00019: z += 372000 × (0.064 − girth) × (0.00019 − lam2)
if max_dr < 0.183 and lam2 < 0.0002: z += 84200 × (0.183 − max_dr) × (0.0002 − lam2)
if girth < 0.063 and log_sum_pt > 6.67: z += 498 × (0.063 − girth) × (log_sum_pt − 6.67)
if LHA < 0.196 and lam2 < 0.00029: z += -125000 × (0.196 − LHA) × (0.00029 − lam2)
if log_sum_pt > 6.69 and width < 0.0072: z += -4010 × (log_sum_pt − 6.69) × (0.0072 − width)
if mass < 24.70 and lam2 < 0.0002: z += -694 × (24.70 − mass) × (0.0002 − lam2)
if C2 < 0.026: z += -54.30 × (0.026 − C2)
if LHA < 0.202 and width < 0.00048: z += 53000 × (0.202 − LHA) × (0.00048 − width)
if width < 0.0052 and mass_over_sum_pt_sq > 1.2e-05: z += -371000 × (0.0052 − width) × (mass_over_sum_pt_sq − 1.2e-05)
if mass < 21.60 and centroid_offset > 0.032: z += -101 × (21.60 − mass) × (centroid_offset − 0.032)
if C2 < 0.026 and width < 0.0045: z += -16000 × (0.026 − C2) × (0.0045 − width)
if girth < 0.061 and lam1 > 0.00023: z += -33700 × (0.061 − girth) × (lam1 − 0.00023)
if girth < 0.063 and centroid_offset > 0.0069: z += -3320 × (0.063 − girth) × (centroid_offset − 0.0069)
if girth2 < 0.0068 and centroid_offset > 0.018: z += -39800 × (0.0068 − girth2) × (centroid_offset − 0.018)
if C2 < 0.026 and centroid_offset > 0.016: z += -6340 × (0.026 − C2) × (centroid_offset − 0.016)
if sum_pt_top5 > 655 and girth2 < 0.0017: z += -3.96 × (sum_pt_top5 − 655) × (0.0017 − girth2)
if log_sum_pt > 6.66 and pt_7 < 50.20: z += 0.108 × (log_sum_pt − 6.66) × (50.20 − pt_7)
if LHA < 0.196 and girth2 > 0.0068: z += 497000 × (0.196 − LHA) × (girth2 − 0.0068)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (62 GeV), wide, pT spread over several particles** — 51.2% of jets, neuron 0.13.  
- **light (40 GeV), narrow, pT spread over several particles** — 10.1% of jets, neuron 0.77.  
- **very light (7 GeV), very narrow, pT spread over several particles** — 10.0% of jets, neuron 1.30.  
- **very light (12 GeV), very narrow, pT spread over several particles** — 9.6% of jets, neuron 1.45.  
- **very light (8 GeV), very narrow, leading particle 42% of pT, high pT** — 8.3% of jets, neuron 1.08.  
- **light (21 GeV), very narrow, pT spread over several particles** — 7.9% of jets, neuron 1.55.  
- **very light (11 GeV), very narrow, leading particle 46% of pT, high pT** — 1.8% of jets, neuron 0.66.  
- **very light (8 GeV), narrow, pT spread over several particles, low pT** — 0.9% of jets, neuron 0.00.  
- **very light (6 GeV), average width, pT spread over several particles, low pT** — 0.3% of jets, neuron 0.00.  
- **very light (5 GeV), wide, pT spread over several particles, low pT** — 0.0% of jets, neuron 0.00.  

### neuron 15: massive jet slightly wider than a W (moderate)

- **What it measures:** Pushed down for light jets (mass < 36.2 GeV, its strongest term), for small e2_sq (< 0.00854) and for narrow jets (width < 0.0068), but pushed up for width < 0.0141, so it responds to massive jets just wider than the typical W but not broad; it follows the number of particles at 0.2 ≤ ΔR < 0.4 and above 10 GeV (rank correlations 0.476 and 0.422). Z jets sit highest (0.87), then tops (0.46), with gluons (0.18), quarks (0.13) and W (0.06) low.
- *computed — its value:* largest for Z (0.87), then t (0.46), then g (0.18), then q (0.13), then W (0.06); it separates Z jets from the rest best (AUC 0.72: large for Z)
- **How the class scores use it:** W jets sit lowest on it, so the W score reads it as evidence against a W: it lowers the W score (-7%). It also lowers the Z score, but only slightly (-2%), even though Z jets sit highest; it does not enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-7%), Z (-2%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -2.42
if mass < 36.20: z += -0.438 × (36.20 − mass)
if e2_sq < 0.0085: z += -565 × (0.0085 − e2_sq)
if e2 < 0.025: z += 348 × (0.025 − e2)
if width < 0.0068: z += -744 × (0.0068 − width)
if width < 0.014: z += 221 × (0.014 − width)
if mass_over_sum_pt < 0.068: z += -78.10 × (0.068 − mass_over_sum_pt)
if e2 < 0.042: z += 84.60 × (0.042 − e2)
if girth > 0.033: z += 46.30 × (girth − 0.033)
if z_dr_0p1_0p2 < 0.323: z += 4.04 × (0.323 − z_dr_0p1_0p2)
if tau21 < 0.245 and z_dr_0p2_0p4 < 0.221: z += 52.10 × (0.245 − tau21) × (0.221 − z_dr_0p2_0p4)
if mass > 80.40: z += -0.475 × (mass − 80.40)
if LHA > 0.341: z += -47.80 × (LHA − 0.341)
if girth2_top3 < 0.0022: z += -571 × (0.0022 − girth2_top3)
if width < 0.0076 and e2 > 0.024: z += -74100 × (0.0076 − width) × (e2 − 0.024)
if LHA > 0.179 and sum_pt_top3 > 350: z += 0.045 × (LHA − 0.179) × (sum_pt_top3 − 350)
if D2 < 0.744: z += -2.29 × (0.744 − D2)
if planar_flow < 0.080: z += -8.91 × (0.080 − planar_flow)
if tau21 < 0.249 and z_dr_0p05_0p1 < 0.605: z += -12.30 × (0.249 − tau21) × (0.605 − z_dr_0p05_0p1)
if girth2_top2 < 0.0081 and mean_phi < 0.0017: z += 6900 × (0.0081 − girth2_top2) × (0.0017 − mean_phi)
if mass > 80.40 and z_dr_0p2_0p4 < 0.211: z += 1.94 × (mass − 80.40) × (0.211 − z_dr_0p2_0p4)
if width < 0.0059 and log_sum_pt > 6.90: z += -8130 × (0.0059 − width) × (log_sum_pt − 6.90)
if z_dr_0p05_0p1 > 0.754: z += -6.57 × (z_dr_0p05_0p1 − 0.754)
if tau21 < 0.243 and girth2_top5 > 0.0086: z += -1140 × (0.243 − tau21) × (girth2_top5 − 0.0086)
if log_sum_pt > 6.90: z += 32.30 × (log_sum_pt − 6.90)
if LHA > 0.179 and z_top5 > 0.865: z += -213 × (LHA − 0.179) × (z_top5 − 0.865)
if log_sum_pt > 6.90 and mean_phi > 0.018: z += 49200 × (log_sum_pt − 6.90) × (mean_phi − 0.018)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (51 GeV), average width, pT spread over several particles** — 22.6% of jets, neuron 0.23.  
- **very light (6 GeV), very narrow, pT spread over several particles** — 22.1% of jets, neuron 0.00.  
- **medium-mass (61 GeV), wide, pT spread over several particles** — 17.8% of jets, neuron 1.41.  
- **very light (14 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 0.00.  
- **light (42 GeV), narrow, pT spread over several particles** — 8.8% of jets, neuron 0.34.  
- **medium-mass (72 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 0.05.  
- **light (23 GeV), narrow, pT spread over several particles** — 6.4% of jets, neuron 0.00.  
- **heavy (100 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 0.08.  
- **very light (13 GeV), very narrow, leading particle 45% of pT, high pT** — 0.9% of jets, neuron 0.06.  
- **heavy (127 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.11.  

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs girth2 > 0.0188 (pushes up) and is pushed down strongly for e2 > 0.0634; it follows the number of particles above 10 GeV (rank correlation 0.827). Only tops reach it with any frequency (non-zero for 18.8% of them), so tops sit highest (0.22), then gluons (0.07) and quarks (0.02), with W and Z at 0.00.
- *computed — its value:* largest for t (0.22), then g (0.07), then q (0.02), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.59: large for t)
- **How the class scores use it:** It does not enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.931
if girth2 > 0.019: z += 259 × (girth2 − 0.019)
if e2 > 0.063: z += -82.90 × (e2 − 0.063)
if girth2 > 0.019 and pt_7 > 15.60: z += 4.81 × (girth2 − 0.019) × (pt_7 − 15.60)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **light (36 GeV), narrow, pT spread over several particles** — 91.0% of jets, neuron 0.00.  
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 1.7% of jets, neuron 0.08.  
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 1.5% of jets, neuron 0.27.  
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.00.  
- **heavy (90 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.95.  
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 1.2% of jets, neuron 0.32.  
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 1.97.  
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 0.5% of jets, neuron 1.73.  
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 3.70.  
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.1% of jets, neuron 6.05.  
