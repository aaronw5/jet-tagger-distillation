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

- **medium-mass (70 GeV), very wide, pT spread over several particles, low pT** — 21.5% of jets, neuron 1.35. Mostly tops (67%) with 13% gluons and 13% Z, 21.46% of jets: mass 70.3 GeV, width 0.0183, softer than average (sum pT 566 GeV), with the pT spread out (35.89% at 0.05-0.1 and 29.41% at 0.1-0.15 from the axis). These wide jets mostly fail e2_sq < 0.00817, width < 0.00866 and lam1 < 0.00818 and lam2 < 0.00054, so the big bonus and the narrowness penalties both drop out; LHA > 0.259 (+1.023) is the main term, with e2 > 0.0311 (-0.516) and max_dr < 0.249 taking some back and pt_7 > 34.5 adding some. With the 0.823 intercept the neuron sits at 1.35, raising the g score (+0.527) and the Z score; the formula calls them t.
- **light (44 GeV), average width, pT spread over several particles** — 15.5% of jets, neuron 0.58. Mostly W (40%) with 28% Z, 14% tops and 12% gluons, 15.47% of jets: mass 44.1 GeV, width 0.0054, a bit softer than average, with 54.92% of the pT at 0.05-0.1 from the axis. e2_sq < 0.00817 always passes (+1.334) but width < 0.00866 (-0.968) and max_dr < 0.249 (-0.831) nearly cancel it; log_sum_pt > 6.38 (+0.525) and pt_7 > 34.5 add, e2_sq < 0.0085 and planar_flow < 0.0833 and lam1 < 0.00818 and lam2 < 0.00054 subtract. The neuron sits at 0.583 (on for 57.1%), a small push up on the g score and Z score; the formula calls them W.
- **very light (13 GeV), very narrow, pT spread over several particles** — 12.6% of jets, neuron 0.19. A gluon/quark mixture (gluons 38%, quarks 33%, W 13%, Z 10%), 12.63% of jets: mass 12.8 GeV, width 0.0009, average pT, with 63.62% of the pT inside 0.025 of the axis. e2_sq < 0.00817 (+3.086) is outweighed by the narrowness and lightness penalties width < 0.00866 (-2.3), lam1 < 0.00818 and lam2 < 0.00054, max_dr < 0.249 and mass < 55.5, all passing, while pt_7 > 34.5 passes for only 29.7% and LHA > 0.259 almost never. The neuron stays low (on for 22.2%) and barely moves the scores; the formula splits them g 43% / q 38%.
- **very light (11 GeV), very narrow, leading particle 45% of pT, high pT** — 12.2% of jets, neuron 0.69. Mostly quarks (64%) with 15% gluons, 12.25% of jets: mass 11.16 GeV, width 0.0003, very hard (sum pT 955 GeV) with a dominant leading particle (431.73 vs 240.22 GeV) and a soft eighth, and 93.63% of the pT inside 0.025. All the high-pT tests pass: e2_sq < 0.00817 (+3.209), log_sum_pt > 6.38 (+2.242), log_sum_pt > 6.63, log_sum_pt > 6.38 and max_dr < 0.2 and z_7 < 0.0559 and girth2_top2 < 0.0139, while width < 0.00866 (-2.47), z_7 < 0.0555 and lam1 < 0.00818 and lam2 < 0.00054 subtract; z_7 < 0.043 passing for nearly all marks the soft last particle. The neuron is on for 49%, a mild push on the g score (+0.27); the formula calls them q.
- **medium-mass (65 GeV), average width, leading particle 44% of pT, high pT** — 10.2% of jets, neuron 1.59. A Z/W mixture (Z 44%, W 40%), 10.16% of jets: mass 65.3 GeV, width 0.0058, hard (sum pT 884 GeV, leading particle 386.83 GeV) with a soft eighth particle, and the pT at 0.025-0.1 from the axis. The hardness tests pass: log_sum_pt > 6.38 (+1.882), log_sum_pt > 6.63, log_sum_pt > 6.57 and lam2 < 0.00118 and z_7 < 0.0559 and girth2_top2 < 0.0139, plus e2_sq < 0.00817, against z_7 < 0.0555 (-1.524), z_7 < 0.043 and width < 0.00866; mass < 55.5 passes for only 19.8%, sparing them the lightness penalty. The neuron sits at 1.591, raising the g score (+0.621) and the Z score; the formula calls them W.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 8.8% of jets, neuron 2.25. A Z/W mixture (Z 37%, W 36%, tops 14%, gluons 10%), 8.76% of jets: mass 58.2 GeV, width 0.0076, average total pT but spread evenly over the particles (leading 170.68 GeV, eighth 50.49 GeV vs 34.66 on average), with 57.36% of the pT at 0.05-0.1. pt_7 > 34.5 always passes (+2.478), since the eighth particle is hard, and pt_7 > 34.8 and mass < 91.2 (-0.898) gives only part back; log_sum_pt > 6.38, e2_sq < 0.00817 and LHA > 0.259 add, max_dr < 0.249 and width < 0.00866 subtract. The neuron reaches 2.246, raising the g score (+0.878) and the Z score (+0.281) and slightly lowering the W score; the formula calls them W.
- **very light (14 GeV), very narrow, pT spread over several particles** — 7.4% of jets, neuron 0.19. Mostly gluons (50%) with 23% quarks, 7.39% of jets: mass 13.8 GeV, width 0.0011, soft (sum pT 615 GeV) with the pT shared more evenly than usual (leading 151.67 GeV), 54.03% of it inside 0.025. e2_sq < 0.00817 (+3.008) and pt_7 > 34.5 are cancelled by width < 0.00866 (-2.227), max_dr < 0.249, lam1 < 0.00818 and lam2 < 0.00054, pt_7 > 34.8 and mass < 91.2 and mass < 55.5; log_sum_pt > 6.38 passes for only 60.4%. The neuron stays low (mean 0.192, on for 19.6%) with little effect on the scores; the formula calls them g.
- **light (34 GeV), narrow, pT spread over several particles** — 5.6% of jets, neuron 1.40. A mixture led by W (38%) with 28% Z, 14% quarks and 12% gluons, 5.61% of jets: mass 34.4 GeV, width 0.003, harder than average (sum pT 781 GeV), with 61.63% of the pT at 0.025-0.05 from the axis. e2 < 0.0346 and eccentricity > 0.979 always passes (+2.061) and marks this group, together with e2_sq < 0.00817 (+2.389) and log_sum_pt > 6.38; e2_sq < 0.0085 and planar_flow < 0.0833 (-2.063), width < 0.00866 and lam1 < 0.00818 and lam2 < 0.00054 take much of it back. The neuron sits at 1.401 (on for 80.5%), raising the g score (+0.547) and the Z score; the formula calls them W.
- **very light (10 GeV), very narrow, pT spread over several particles** — 5.0% of jets, neuron 1.04. Mostly gluons (49%) with 31% quarks, 5.01% of jets: mass 9.9 GeV, width 0.0004, harder than average (sum pT 785 GeV) and evenly shared (eighth particle 51.13 GeV), with 80.61% of the pT inside 0.025. e2_sq < 0.00817 (+3.19), pt_7 > 34.5 (+2.577) and log_sum_pt > 6.38 add; width < 0.00866, pt_7 > 34.8 and mass < 91.2 (-2.291), lam1 < 0.00818 and lam2 < 0.00054 and max_dr < 0.249 subtract, and pt_7 > 35 and max_dr < 0.0787 sets them apart. The neuron averages 1.038 (on for 63.4%), raising the g score; the formula calls them g.
- **very light (13 GeV), very narrow, pT spread over several particles, high pT** — 1.3% of jets, neuron 3.11. Mostly gluons (63%) with 20% quarks, 1.27% of jets: mass 13.5 GeV, width 0.0005, very hard (sum pT 964 GeV) with the pT spread evenly (eighth particle 66.16 GeV, so pt_7 > 52.6 passes for 98.3%), 81.46% of it inside 0.025. pt_7 > 34.5 (+4.907) is mostly cancelled by pt_7 > 34.8 and mass < 91.2 (-4.175), so the rest decides: e2_sq < 0.00817, log_sum_pt > 6.38, log_sum_pt > 6.38 and max_dr < 0.2 and log_sum_pt > 6.63 add more than width < 0.00866 and the other narrowness penalties take. The neuron reaches 3.106, its largest group mean, raising the g score (+1.213) and the Z score; the formula calls them g.

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

- **light (48 GeV), average width, pT spread over several particles** — 15.5% of jets, neuron 1.28. A W/Z mixture (W 37%, Z 34%, tops 14%, gluons 10%), 15.52% of jets: mass 47.9 GeV, width 0.0058, a bit softer than average (sum pT 674 GeV), with 48.76% of the pT at 0.05-0.1 from the axis. LHA > 0.119 (-1.842) and pt_7 < 53.4 always pass, but the 2.21 intercept plus sum_pt < 781, pt_7 > 30.4 and mass < 69.4 stay ahead; mass < 36.4 passes for only 18.8%, so the light-jet penalty is mostly absent. The neuron sits at 1.285 (on for 85.7%), raising the g score (+0.552); the formula calls them W.
- **medium-mass (65 GeV), average width, pT spread over several particles** — 14.6% of jets, neuron 0.29. A Z/W/top mixture (Z 37%, W 29%, tops 24%), 14.59% of jets: mass 64.7 GeV, width 0.0079, a hard leading particle (303.49 GeV) but a soft eighth (27.22 GeV vs 34.66 on average), with 51.69% of the pT at 0.05-0.1. LHA > 0.119 (-2.095) and pt_7 < 53.4 (-1.602) always pass and are only partly made up by pt_7 < 43.4 (+1.032), since pt_7 > 30.4, sum_pt < 781 and mass < 69.4 each pass for only part of the group. The neuron hovers near zero (on for 46.8%), a small push on the g score; the formula splits them Z 38% / W 36% / t 24%.
- **very light (8 GeV), very narrow, pT spread over several particles** — 12.2% of jets, neuron 3.32. Mostly quarks (40%) with 34% gluons, 12.2% of jets: mass 8.41 GeV, width 0.0004, harder than average (sum pT 789 GeV), with 80.72% of the pT inside 0.025 of the axis. mass < 36.4 (-2.435) and pt_7 < 53.4 subtract, but the light-and-thin bonuses mass < 69.4 (+1.421), mass < 36.8 and lam2 < 0.00114, lam1 < 0.00333, lam1 < 0.00583 and pt_7 < 43.4 all pass, and LHA > 0.119 passes for only about half. With the 2.21 intercept the neuron sits at 3.324, raising the g score (+1.428); the formula calls them q.
- **very light (9 GeV), very narrow, pT spread over several particles** — 9.7% of jets, neuron 3.97. Mostly gluons (52%) with 29% quarks, 9.71% of jets: mass 8.6 GeV, width 0.0004, average total pT but spread evenly (leading 200.82 GeV, eighth 49.93 GeV), with 80.59% of the pT inside 0.025. The same light-jet terms as group 2 apply (mass < 36.4 against mass < 69.4 and mass < 36.8 and lam2 < 0.00114), but the hard eighth particle adds pt_7 > 30.4 (+1.246) while pt_7 < 43.4 passes for only 19.7% and pt_7 > 30.6 and C2 < 0.0507 takes -0.896. The neuron reaches 3.97, raising the g score (+1.706); the formula calls them g.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 9.5% of jets, neuron 1.01. A Z/W mixture (Z 36%, W 35%, tops 14%, gluons 13%), 9.49% of jets: mass 58 GeV, width 0.007, the pT spread evenly (eighth particle 50.88 GeV), with 53.77% of it at 0.05-0.1 from the axis. LHA > 0.119 always passes (-2.037) and is mostly offset by pt_7 > 30.4 (+1.307); pt_7 < 43.4 passes for only 5.8%, and pt_7 > 30.6 and C2 < 0.0507 and pt_7 > 30.5 and max_dr > 0.0929 take a little more. The neuron sits at 1.007 (on for 76%), raising the g score (+0.433); the formula calls them W.
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 8.5% of jets, neuron 0.23. Mostly tops (83%), 8.48% of jets: mass 84.6 GeV, width 0.0235, soft (sum pT 582 GeV), with the pT spread far out (38.23% at 0.1-0.15, 22.37% at 0.15-0.2 from the axis). LHA > 0.119 subtracts most here (-3.408), with pt_7 < 53.4 adding to it, while mass < 69.4 passes for only 19.2%, so the mass bonus is mostly lost; sum_pt < 781 (+0.996) and log_sum_pt < 6.49 are the main offsets. The neuron stays low (mean 0.23) with a small push on the g score; the formula calls them t.
- **very light (9 GeV), very narrow, leading particle 47% of pT, high pT** — 8.3% of jets, neuron 3.21. Mostly quarks (70%), 8.35% of jets: mass 8.79 GeV, width 0.0002, very hard leading particle (431.55 GeV) with a soft tail (eighth 17 GeV), and 94.13% of the pT inside 0.025. pt_7 > 30.4 never passes and LHA > 0.119 passes for only 23.4%; mass < 36.4 (-2.402) and pt_7 < 53.4 (-2.228) are outweighed by pt_7 < 43.4 (+1.684), mass < 69.4, mass < 36.8 and lam2 < 0.00114 and both lam1 tests. The neuron sits at 3.213, raising the g score (+1.381); the formula calls them q.
- **light (42 GeV), narrow, leading particle 45% of pT, high pT** — 7.9% of jets, neuron 1.20. A mixture (W 35%, Z 26%, quarks 22%, gluons 10%), 7.91% of jets: mass 42.2 GeV, width 0.0029, a very hard leading particle (387.15 GeV) and soft tail (eighth 22.71 GeV), with 52.22% of the pT at 0.025-0.05. pt_7 < 53.4 (-1.878) is balanced by pt_7 < 43.4 (+1.32), while LHA > 0.119 and the group-marking lam1 < 0.00595 and max_dr > 0.0801 subtract and mass < 69.4 and lam1 < 0.00583 add; pt_7 > 30.4 passes for only 14.6%. The neuron sits at 1.202, raising the g score (+0.517); the formula calls them W.
- **light (50 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 2.56. Mostly tops (57%) with 23% gluons, 7.76% of jets: mass 49.6 GeV, width 0.0164, very soft (sum pT 436 GeV, leading particle 106.97 GeV), with the pT spread over 0.05-0.15 from the axis. The softness bonuses sum_pt < 781 (+1.713) and log_sum_pt < 6.49 (+1.608) always pass and outweigh LHA > 0.119 (-2.807) and pt_7 < 53.4 (-1.322), helped by pt_7 < 43.4 and mass < 69.4. The neuron sits at 2.561, raising the g score (+1.101); the formula calls them t.
- **very light (15 GeV), narrow, pT spread over several particles, low pT** — 6.0% of jets, neuron 4.34. Mostly gluons (54%) with 17% quarks and 11% tops, 5.99% of jets: mass 15.3 GeV, width 0.002, soft (sum pT 503 GeV) and evenly shared (leading 125.85 GeV), with 46.12% of the pT at 0.025-0.05. Both softness bonuses (sum_pt < 781 +1.379, log_sum_pt < 6.49) and the light-jet bonuses (mass < 69.4 +1.261, mass < 36.8 and lam2 < 0.00114) pass, against mass < 36.4 (-1.837), pt_7 < 53.4 and LHA > 0.119. The neuron reaches 4.336, its largest group mean, and raises the g score most (+1.863); the formula calls them g.

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

- **medium-mass (54 GeV), wide, pT spread over several particles** — 30.5% of jets, neuron 0.15. A mixture (tops 32%, W 23%, Z 23%, gluons 16%), 30.53% of jets: mass 53.8 GeV, width 0.0107, soft (sum pT 609 GeV) with the pT shared evenly (leading 150.51 GeV, eighth 42.75 GeV), 44.42% of it at 0.05-0.1. pt_7 < 54.7 (+0.972) is cancelled by sum_pt_top2 < 539 (-0.783) and the -0.779 intercept, and the narrowness bonuses (LHA < 0.224, width < 0.0025, dr_0 < 0.0199) rarely pass. The neuron stays low (mean 0.152) and barely moves any score; the formula splits them t 34% / W 29% / Z 23%.
- **medium-mass (54 GeV), wide, pT spread over several particles** — 22.1% of jets, neuron 0.85. Mostly tops (39%) with 21% Z, 18% W and 12% gluons, 22.14% of jets: mass 54.3 GeV, width 0.0104, softer than average (sum pT 628 GeV) with a soft eighth particle (30.01 GeV), 41.59% of the pT at 0.05-0.1. pt_7 < 54.7 (+1.951) and z_7 < 0.0707 and lam2 < 0.00124 (+0.729) add, but z_7 < 0.0751 and sum_pt < 818 (-0.89), passing far more often than in other groups, and sum_pt_top2 < 539 take much back. The neuron averages 0.853 (on for 62.8%), slightly lowering the t (-0.213) and g scores; the formula calls them t.
- **medium-mass (57 GeV), average width, leading particle 42% of pT, high pT** — 14.2% of jets, neuron 3.77. A W/Z mixture (W 40%, Z 39%), 14.21% of jets: mass 56.9 GeV, width 0.0049, hard (sum pT 871 GeV, leading particle 365.38 GeV) with a soft eighth (27.33 GeV), 47.48% of the pT at 0.025-0.05. pt_7 < 54.7 (+2.168), log_sum_pt > 6.55 (+1.756), z_7 < 0.0707 and lam2 < 0.00124 and z_6 < 0.0513 add, while log_sum_pt > 6.55 and lam1 < 0.0123 (-1.596) takes some back; z_6 < 0.05 and e2_sq > 0.00225 is nearly unique to this group. The neuron sits at 3.767, lowering the t (-0.942) and g (-0.706) scores and raising the q score; the formula calls them W.
- **very light (12 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 1.00. Mostly gluons (44%) with 28% quarks, 9.54% of jets: mass 11.7 GeV, width 0.0008, a bit softer than average and evenly shared (leading 190.56 GeV), with 67.87% of the pT inside 0.025. LHA < 0.228 and log_sum_pt < 6.81 always passes here (-2.023) and LHA < 0.216 and centroid_offset > 0.00235 also subtracts, against width < 0.0025 (+1.321), LHA < 0.224, pt_7 < 54.7 and e2 < 0.0346. The neuron averages 1.003 (on for 58.7%), lowering the t and g scores a little; the formula calls them g.
- **very light (9 GeV), very narrow, pT spread over several particles, high pT** — 9.0% of jets, neuron 5.40. Mostly quarks (48%) with 29% gluons, 8.99% of jets: mass 8.68 GeV, width 0.0002, hard (sum pT 860 GeV), with 93.19% of the pT inside 0.025. Almost all quark-like tests pass: LHA < 0.224 (+2.211), log_sum_pt > 6.55, width < 0.0025, pt_7 < 54.7, z_7 < 0.0707 and lam2 < 0.00124 and e2 < 0.0346, against log_sum_pt > 6.55 and lam1 < 0.0123 (-2.445) and dr_0 < 0.0199. The neuron sits at 5.396, raising the q score (+0.253) and lowering the t (-1.349) and g scores; the formula calls them q.
- **very light (9 GeV), very narrow, leading particle 44% of pT, high pT** — 5.3% of jets, neuron 11.34. Mostly quarks (73%), 5.31% of jets: mass 9.19 GeV, width 0.0002, a dominant leading particle (420.37 GeV) and a soft tail (eighth 20.39 GeV), with 97.69% of the pT inside 0.025. Nearly every bonus passes: LHA < 0.224 (+2.752), pt_7 < 54.7, log_sum_pt > 6.55, z_7 < 0.0707 and lam2 < 0.00124, log_sum_pt > 6.58 and dr_0 < 0.0208 and sum_pt > 861 and centroid_offset < 0.0117, with log_sum_pt > 6.55 and lam1 < 0.0123 (-3.62) the main penalty. The neuron reaches 11.341, raising the q score (+0.532) and lowering the t (-2.835) and g scores strongly; the formula calls them q.
- **very light (7 GeV), very narrow, pT spread over several particles, low pT** — 5.2% of jets, neuron 0.27. Mostly gluons (62%) with 28% quarks, 5.16% of jets: mass 7.35 GeV, width 0.0003, soft (sum pT 602 GeV) and evenly shared (leading 156.53 GeV), with 91.28% of the pT inside 0.025. LHA < 0.228 and log_sum_pt < 6.81 (-4.681), which needs a soft jet, cancels LHA < 0.224 (+2.114), width < 0.0025, pt_7 < 54.7 and e2 < 0.0346, and dr_0 < 0.0199 subtracts too; log_sum_pt > 6.55 passes for only 16.6%. The neuron stays low (mean 0.265, on for 19.3%) with little effect on the scores; the formula calls them g.
- **light (26 GeV), very narrow, leading particle 42% of pT, high pT** — 1.8% of jets, neuron 3.87. Mostly gluons (42%) with 33% quarks, 1.84% of jets: mass 25.8 GeV, width 0.0013, very hard (sum pT 1099 GeV, leading particle 459.44 GeV), with 75.47% of the pT inside 0.025. The high-pT penalties log_sum_pt > 6.55 and lam1 < 0.0123 (-4.916), log_sum_pt > 6.83 (-4.595) and log_sum_pt > 6.9 and centroid_offset < 0.0182 fight log_sum_pt > 6.55 (+3.647), sum_pt > 861 and centroid_offset < 0.0117 and LHA < 0.224. The neuron is on for only 64.9%, lowering the t and g scores and raising the q score when on; the formula splits them between g and q.
- **light (22 GeV), very narrow, leading particle 58% of pT, high pT** — 1.8% of jets, neuron 13.55. Mostly quarks (71%), 1.82% of jets: mass 21.5 GeV, width 0.0009, an extremely hard leading particle (572.89 GeV) with an almost empty tail (eighth 8.38 GeV), and 89.31% of the pT inside 0.025. The soft-tail tests pass only here: z_7 < 0.0233 (+3.179), mean_phi2 < 0.0137 and pt_5 < 24.6 (+3.113), z_6 < 0.0283 and n_dr_0p2_0p4 < 1.93, on top of pt_7 < 54.7 and z_6 < 0.0513, against z_7 < 0.0323 and pt_5 < 30.9 (-4.556) and log_sum_pt > 6.55 and lam1 < 0.0123. The neuron reaches 13.549, its largest group mean, raising the q score (+0.635) and lowering the t (-3.387) and g scores; the formula calls them q.
- **light (21 GeV), very narrow, leading particle 47% of pT, high pT** — 0.5% of jets, neuron 1.91. Mostly gluons (52%) with 32% quarks, 0.46% of jets: mass 20.9 GeV, width 0.0007, the hardest group (sum pT 1330 GeV, leading particle 627.77 GeV), with 86.84% of the pT inside 0.025. log_sum_pt > 6.83 (-9.603), log_sum_pt > 6.9 and centroid_offset < 0.0182 (-9.395) and log_sum_pt > 6.55 and lam1 < 0.0123 all pass and outweigh sum_pt > 861 and centroid_offset < 0.0117 (+5.538), log_sum_pt > 6.55 and log_sum_pt > 6.58 and dr_0 < 0.0208 for most jets. The neuron is on for only 43% (mean 1.913); the formula calls them g.

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

- **very light (10 GeV), very narrow, pT spread over several particles** — 32.3% of jets, neuron 0.07. Mostly quarks (45%) with 35% gluons, 32.3% of jets: mass 10.2 GeV, width 0.0004, harder than average (sum pT 792 GeV), with 84.26% of the pT inside 0.025 of the axis. The narrowness penalties girth2 < 0.00869 (-8.548) and width < 0.0132 (-8.505) cancel the 8.53 intercept, and e2 < 0.0501 (+5.788), mass < 49.8 and z_dr_0p05_0p1 < 0.75 and girth2_top2 < 0.00947 are undone by max_dr < 0.172, z_dr_0_0p05 > 0.755 and smaller terms. The neuron stays near 0 and barely touches the scores; the formula calls them q.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 24.2% of jets, neuron 0.52. Mostly Z (46%) with 35% W and 11% tops, 24.16% of jets: mass 57.7 GeV, width 0.0067, average pT, with 61.17% of the pT at 0.05-0.1 from the axis. mass_over_sum_pt > 0.00657 (-5.859), width < 0.0132 (-4.315) and girth2 < 0.00869 pass, and e2 < 0.0501 and D2 < 1.66 do not quite make up for them; centroid_offset > 0.0186 passes for only 16.7%, so the off-centre terms hardly act. The neuron is on for 31.8%, slightly lowering the Z (-0.194) and W scores; the formula calls them Z.
- **light (43 GeV), narrow, pT spread over several particles** — 17.7% of jets, neuron 0.25. Mostly W (42%) with 25% Z, 14% gluons and 10% quarks, 17.66% of jets: mass 42.9 GeV, width 0.0037, average pT, with 55.67% of the pT at 0.025-0.05. width < 0.0132 (-6.277), girth2 < 0.00869 (-5.092) and mass_over_sum_pt > 0.00657 (-4.074) always pass and beat the intercept plus e2 < 0.0501 (+3.623), centroid_offset > 0.00824 and lam2 < 0.00351 and girth2_top2 < 0.00947. The neuron stays low (mean 0.251, on for 17.3%), slightly lowering the W and Z scores; the formula calls them W.
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 7.0% of jets, neuron 5.02. Mostly tops (74%) with 16% gluons, 6.96% of jets: mass 74.6 GeV, width 0.0161, softer than average (sum pT 603 GeV), with 42.3% of the pT at 0.05-0.1 and 26.12% at 0.1-0.15. The narrowness penalties vanish (girth2 < 0.00869 passes for 0.1%), so only mass_over_sum_pt > 0.00657 (-9.278) and log_sum_pt < 6.69 pull against the intercept, while girth > 0.0885 (+1.748), LHA > 0.313 and eccentricity > 0.878, log_sum_pt < 6.56 and centroid_offset > 0.00824 and lam2 < 0.00351 add. The neuron sits at 5.016, lowering the Z (-1.881) and W scores and raising the q and g scores; the formula calls them t.
- **very light (17 GeV), narrow, pT spread over several particles** — 5.2% of jets, neuron 1.99. A mixture (gluons 32%, Z 26%, W 16%, tops 13%, quarks 12%), 5.25% of jets: mass 17.2 GeV, width 0.0027, soft (sum pT 616 GeV), with the pT centroid off the axis: 58.05% at 0.025-0.05 and 33.9% at 0.05-0.1. The off-centre tests all pass: centroid_offset > 0.00824 and lam2 < 0.00351 (+4.605) and centroid_offset > 0.0188 and mean_phi2 < 0.00883 add, centroid_offset > 0.0186 (-4.051) subtracts; with e2 < 0.0501 they outweigh width < 0.0132 (-6.969) and girth2 < 0.00869. The neuron sits at 1.989, lowering the Z (-0.746) and W scores and raising the q and g scores; the formula calls them g.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 4.3% of jets, neuron 6.89. Mostly tops (77%), 4.34% of jets: mass 85.9 GeV, width 0.0276, soft (sum pT 533 GeV), with the pT far out (35.1% at 0.1-0.15, 27.43% at 0.15-0.2). width < 0.0132 and e2 < 0.0501 never pass; mass_over_sum_pt > 0.00657 (-12.279) and centroid_offset > 0.0186 are outweighed by the intercept with girth > 0.0885 (+5.118), LHA > 0.313 and eccentricity > 0.878 (+3.292), log_sum_pt < 6.68 and mean_eta2 > 0.00426 and log_sum_pt < 6.56. The neuron reaches 6.895 (on for 87.5%), lowering the Z (-2.586) and W scores and raising the q and g scores; the formula calls them t.
- **medium-mass (56 GeV), very wide, pT spread over several particles, low pT** — 4.3% of jets, neuron 5.89. Mostly tops (66%) with 21% gluons, 4.31% of jets: mass 56.5 GeV, width 0.0131, soft (sum pT 564 GeV), with 51.4% of the pT at 0.05-0.1 and the centroid off the axis. centroid_offset > 0.0186 always passes (-6.415) and with mass_over_sum_pt > 0.00657 (-7.415) is balanced by the intercept, centroid_offset > 0.00824 and lam2 < 0.00351 (+4.289) and the softness bonuses; width < 0.0132 and girth2 < 0.00869 often fail, so the narrowness penalties are weak. The neuron sits at 5.894, lowering the Z (-2.21) and W scores and raising the q and g scores; the formula calls them t.
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.09. Mostly tops (95%), 2.32% of jets: mass 85.2 GeV, width 0.028, soft (sum pT 530 GeV), with 37.71% of the pT at 0.1-0.15 from the axis. lam2 > 0.00354 passes only here (-13.298), and with mass_over_sum_pt > 0.00657 (-12.28) and centroid_offset > 0.0186 it beats girth > 0.0885 (+5.264) and the softness bonuses; centroid_offset > 0.00824 and lam2 < 0.00351 never passes. The neuron stays near 0 (mean 0.085) and adds almost nothing to the scores; the formula calls them t.
- **medium-mass (61 GeV), very wide, pT spread over several particles, low pT** — 1.5% of jets, neuron 7.56. Mostly tops (67%) with 24% gluons, 1.52% of jets: mass 60.9 GeV, width 0.0227, soft (sum pT 510 GeV), with 44.97% of the pT at 0.1-0.15 and a pT centroid far from the axis. centroid_offset > 0.0499 passes for all (+3.42), nearly unique to this group, and with centroid_offset > 0.00824 and lam2 < 0.00351 (+7.97), girth > 0.0885 and LHA > 0.313 and eccentricity > 0.878 it outweighs centroid_offset > 0.0186 (-12.675) and mass_over_sum_pt > 0.00657. The neuron reaches 7.556, its largest group mean, lowering the Z (-2.834) and W scores and raising the q and g scores; the formula calls them t.
- **light (23 GeV), very narrow, leading particle 46% of pT, high pT** — 1.2% of jets, neuron 0.78. Mostly gluons (43%) with 36% quarks, 1.19% of jets: mass 23.1 GeV, width 0.0008, very hard (sum pT 1224 GeV, leading particle 567.02 GeV), with 82.55% of the pT inside 0.025. sum_pt > 987 passes for all (+11.384) and sum_pt_top5 > 837 (-7.186) takes most of it back, both nearly unique to this group; the rest is like group 0, width < 0.0132 (-8.229) and girth2 < 0.00869 against e2 < 0.0501, with sum_pt > 989 and pt_6 > 29.9 subtracting. The neuron is on for 37.2% (mean 0.776), a small pull on the Z and W scores; the formula calls them g.

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

- **medium-mass (55 GeV), average width, pT spread over several particles** — 28.8% of jets, neuron 3.20. Mostly Z (42%) with 33% W and 13% tops, 28.82% of jets: mass 54.8 GeV, width 0.0066, average pT, with 64.1% of the pT at 0.05-0.1 from the axis. On top of the 11.3 intercept, max_dr < 0.196 and z_dr_0p05_0p1 > 0.0498 (+1.9) and centroid_offset < 0.0397 and sum_pt > 567 add, while girth2 > 0.00153 (-3.426), e2 > 0.0174, lam1 < 0.00846, C2 < 0.0673 and lam1 < 0.0081 and D2 < 1.14 (mostly passing only here) subtract. The neuron sits at 3.2, its largest group mean, raising the Z score (+1.5) and the W score; the formula splits them between Z and W.
- **very light (7 GeV), very narrow, pT spread over several particles, high pT** — 20.4% of jets, neuron 0.03. Mostly quarks (53%) with 32% gluons, 20.36% of jets: mass 6.91 GeV, width 0.0001, harder than average (sum pT 860 GeV), with 97.49% of the pT inside 0.025. All the narrowness penalties pass: lam1 < 0.00846 (-5.904), width < 0.00544 (-5.789), girth < 0.0883, max_dr < 0.158 and width < 0.000546, and together with the intercept they are not made up by max_dr < 0.157 and z_dr_0p05_0p1 < 0.671 (+4.019), mass < 29.5 and e2 < 0.0245; girth2 > 0.00153 never passes. The neuron stays near 0 (mean 0.03) and hardly touches the scores; the formula calls them q.
- **light (42 GeV), narrow, pT spread over several particles** — 17.3% of jets, neuron 1.68. Mostly W (40%) with 26% Z, 14% gluons and 12% quarks, 17.33% of jets: mass 41.6 GeV, width 0.0035, a harder leading particle (288 GeV), with 58.33% of the pT at 0.025-0.05. lam1 < 0.00846 (-3.6), girth < 0.0883 (-2.711), width < 0.00544 and girth2 > 0.00153 subtract, against centroid_offset < 0.0397 and sum_pt > 567 (+1.183) and e2 < 0.0245; mass_over_sum_pt > 0.0726 and mass_over_sum_pt > 0.0842 almost never pass, so the mass/pT bonuses are missing. The neuron sits at 1.68 (on for 79.8%), raising the Z (+0.787) and W scores; the formula calls them W.
- **very light (14 GeV), very narrow, pT spread over several particles** — 15.5% of jets, neuron 0.78. Mostly gluons (41%) with 27% quarks, 14% W and 13% Z, 15.49% of jets: mass 13.9 GeV, width 0.001, average pT, with 50.73% of the pT inside 0.025 and 42.66% at 0.025-0.05. The narrowness penalties lam1 < 0.00846 (-5.364), width < 0.00544 (-4.881), girth < 0.0883 and max_dr < 0.158 are nearly balanced by max_dr < 0.157 and z_dr_0p05_0p1 < 0.671 (+2.803), mass < 29.5, e2 < 0.0245 and width < 0.00581 and n_dr_0p1_0p2 < 2.77 with the intercept. The neuron sits near the edge (mean 0.779, on for 43.2%), a small push on the Z and W scores; the formula calls them g.
- **medium-mass (61 GeV), very wide, pT spread over several particles, low pT** — 5.5% of jets, neuron 0.85. Mostly tops (67%) with 20% gluons, 5.55% of jets: mass 60.8 GeV, width 0.0129, soft (sum pT 584 GeV), with 45.63% of the pT at 0.05-0.1 and 30.32% at 0.1-0.15. girth2 > 0.00153 (-7.74), girth2 > 0.00751 (-5.564) and e2 > 0.0174 subtract, the mass/pT window bonuses mass_over_sum_pt > 0.0726 (+2.771) and mass_over_sum_pt > 0.0842 add, and mass_over_sum_pt > 0.108 passes for 39.7%. The neuron is on for 41.9%, a small push on the Z and W scores; the formula calls them t.
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 0.00. Mostly tops (79%), 4.25% of jets: mass 75.4 GeV, width 0.018, soft (sum pT 588 GeV), with 35.99% of the pT at 0.1-0.15. mass_over_sum_pt > 0.108 (-11.632), girth2 > 0.00153 (-11.207) and girth2 > 0.00751 (-10.816) outweigh the intercept and the window bonuses mass_over_sum_pt > 0.0726 and mass_over_sum_pt > 0.0842. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 3.7% of jets, neuron 0.00. Mostly tops (85%), 3.69% of jets: mass 80.6 GeV, width 0.0233, soft (sum pT 548 GeV), with 35.1% of the pT at 0.1-0.15. mass_over_sum_pt > 0.108 (-22.557), girth2 > 0.00751 (-16.257) and girth2 > 0.00153 (-14.799) far outweigh the intercept and mass_over_sum_pt > 0.0842 (+6.863) and mass_over_sum_pt > 0.0726. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 2.5% of jets, neuron 0.00. Mostly tops (87%), 2.51% of jets: mass 80.8 GeV, width 0.0298, very soft (sum pT 482 GeV, leading particle 118.02 GeV), with 30.15% of the pT at 0.15-0.2. mass_over_sum_pt > 0.108 (-34.383), girth2 > 0.00751 (-22.926) and girth2 > 0.00153 dominate, and centroid_offset < 0.0397 and sum_pt > 567 passes for only 8.5%, so the sum stays far below zero. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **heavy (114 GeV), very wide, pT spread over several particles** — 1.2% of jets, neuron 0.00. Mostly tops (78%), 1.16% of jets: mass 114.3 GeV (the heaviest group), width 0.0284, near-average pT (sum pT 691 GeV), with the pT spread over 0.1-0.3 from the axis. Besides mass_over_sum_pt > 0.108 (-33.297) and both girth2 penalties, mass > 80.4 (-10.709) and girth2 > 0.00793 and log_sum_pt > 6.18 (-10.089) pass for all, which marks this group. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **heavy (93 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.00. Mostly tops (69%) with 23% gluons, 0.84% of jets: mass 93.1 GeV, width 0.0397 (the widest group), very soft (sum pT 482 GeV), with 39.57% of the pT at 0.2-0.3. The width and mass/pT penalties are at their largest: mass_over_sum_pt > 0.108 (-49.091), girth2 > 0.00751 (-33.201) and girth2 > 0.00153 (-25.985), against only mass_over_sum_pt > 0.0842 (+11.884) and mass_over_sum_pt > 0.0726. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.

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

- **medium-mass (63 GeV), very wide, pT spread over several particles, low pT** — 22.1% of jets, neuron 0.45. Mostly tops (50%) with 21% Z and 16% gluons, 22.09% of jets: mass 62.9 GeV, width 0.0139, soft (sum pT 600 GeV), with 45.58% of the pT at 0.05-0.1 and 22.87% at 0.1-0.15. The narrowness bonuses mostly fail (width < 0.00624 passes for 14.9%, girth2 < 0.00415 never), so only small terms remain: max_dr < 0.211 (+0.555), mass < 51.9 and log_sum_pt < 6.83, girth2 > 0.0165 and lam2 > 0.00114, which for most jets do not overcome the -2.15 intercept. The neuron stays low (mean 0.447, on for 27%) with little effect on the scores; the formula calls them t.
- **medium-mass (63 GeV), average width, pT spread over several particles** — 17.4% of jets, neuron 0.68. Mostly Z (42%) with 37% W and 15% tops, 17.41% of jets: mass 63.2 GeV, width 0.0078, average pT, with 55.64% of the pT at 0.05-0.1. centroid_offset < 0.0194 always passes (+2.306), with max_dr < 0.211, centroid_offset < 0.0179 and pt_4 > 47.2 and width < 0.00624 adding, against centroid_offset < 0.0176 and z_4 > 0.0442 (-0.869) and log_sum_pt > 6.34; girth2 < 0.00415 and e2 < 0.0202 almost never pass. The neuron averages 0.678 (on for 58.7%), a small push on the q and g scores; the formula splits them W 44% / Z 43%.
- **very light (8 GeV), very narrow, leading particle 40% of pT, high pT** — 14.4% of jets, neuron 8.32. Mostly quarks (64%) with 22% gluons, 14.44% of jets: mass 7.82 GeV, width 0.0001, very hard (sum pT 926 GeV, leading particle 378.95 GeV), with 97.83% of the pT inside 0.025. width < 0.00624 (+13.115), girth2 < 0.00415, e2 < 0.0202, centroid_offset < 0.0194 and max_dr < 0.211 all add, against mass_over_sum_pt < 0.0767 (-4.408), girth < 0.0535 (-4.185), mass < 39 and centroid_offset < 0.0299 and mass < 27.9; width < 0.000176 is nearly unique to this group. The neuron sits at 8.318, raising the q (+2.112) and g scores and lowering the W and Z scores; the formula calls them q.
- **light (46 GeV), average width, pT spread over several particles** — 11.3% of jets, neuron 1.57. Mostly W (49%) with 27% Z, 11.26% of jets: mass 46.5 GeV, width 0.0045, average pT, with the pT at 0.025-0.1 from the axis (47.57% at 0.025-0.05). width < 0.00624 always passes (+3.719), helped by centroid_offset < 0.0194, max_dr < 0.211 and mass < 51.9 and log_sum_pt < 6.83, against width < 0.00609 and centroid_offset > 0.00339 (-0.912), mass_over_sum_pt < 0.0767 and e2 < 0.0348 and tau21 < 0.432. The neuron sits at 1.57 (on for 70.3%), raising the q (+0.399) and g scores a little; the formula calls them W.
- **light (36 GeV), narrow, pT spread over several particles** — 7.7% of jets, neuron 2.65. A mixture led by W (37%) with 22% Z, 19% gluons and 13% quarks, 7.74% of jets: mass 35.9 GeV, width 0.0029, average pT, with 60.32% of the pT at 0.025-0.05. width < 0.00624 (+7.162), mass < 51.9 and log_sum_pt < 6.83 and girth2 < 0.00415 add; width < 0.00609 and centroid_offset > 0.00339 (-2.177), mass_over_sum_pt < 0.0767 (-1.801), girth < 0.0535 and girth2 < 0.00445 and centroid_offset > 0.0105 take back less. The neuron sits at 2.652 (on for 79.6%), raising the q (+0.673) and g scores; the formula calls them W.
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.7% of jets, neuron 8.58. Mostly gluons (58%) with 32% quarks, 7.72% of jets: mass 8 GeV, width 0.0003, soft (sum pT 648 GeV) with the pT shared evenly (leading 167.96 GeV), 91.21% of it inside 0.025. width < 0.00624 (+12.788), girth2 < 0.00415, mass < 51.9 and log_sum_pt < 6.83 (+3.679) and e2 < 0.0202 add, outweighing mass_over_sum_pt < 0.0767 (-4.128), girth < 0.0535 and the light-mass tests; girth2_top2 < 0.000725 and z_7 > 0.0232 marks the evenly shared pT. The neuron reaches 8.579, its largest group mean, raising the q (+2.178) and g (+1.475) scores; the formula calls them g.
- **light (25 GeV), very narrow, pT spread over several particles** — 5.7% of jets, neuron 7.30. A gluon/quark mixture (gluons 39%, quarks 38%, W 12%), 5.71% of jets: mass 24.5 GeV, width 0.0012, average pT, with 59.82% of the pT inside 0.025. width < 0.00624 (+10.805), girth2 < 0.00415, centroid_offset < 0.0194, max_dr < 0.211 and e2 < 0.0202 add, against mass_over_sum_pt < 0.0767 (-2.864), girth < 0.0535, width < 0.00609 and centroid_offset > 0.00339 and mass < 39 and centroid_offset < 0.0299. The neuron sits at 7.298, raising the q (+1.853) and g scores; the formula splits them q 48% / g 45%.
- **very light (8 GeV), very narrow, pT spread over several particles** — 5.4% of jets, neuron 2.57. A mixture (gluons 30%, W 26%, quarks 21%, Z 20%), 5.42% of jets: mass 8.2 GeV, width 0.0006, average pT, with 67.35% of the pT inside 0.025 and 31.61% at 0.025-0.05. width < 0.00624 (+12.101) with girth2 < 0.00415 and e2 < 0.0202 is cut down by mass_over_sum_pt < 0.0767, width < 0.00609 and centroid_offset > 0.00339 (-4.033), girth < 0.0535, mass < 27.9 and girth2 < 0.00445 and centroid_offset > 0.0105 (-2.159), the last two off-centre tests passing for all. The neuron sits at 2.573 (on for 84.1%), raising the q (+0.653) and g scores; the formula calls them g.
- **very light (10 GeV), very narrow, pT spread over several particles, low pT** — 4.2% of jets, neuron 1.88. Mostly gluons (40%) with 21% Z, 15% W, 13% quarks and 10% tops, 4.18% of jets: mass 10.2 GeV, width 0.0017, soft (sum pT 588 GeV), with only 13.93% of the pT inside 0.025 and 69.1% at 0.025-0.05. width < 0.00624 (+9.79) and mass < 51.9 and log_sum_pt < 6.83 (+4.487) add, but the off-centre penalties width < 0.00609 and centroid_offset > 0.00339 (-5.675) and girth2 < 0.00445 and centroid_offset > 0.0105 plus mass_over_sum_pt < 0.0767 take much of it, and centroid_offset < 0.0194 passes for only 4.2%. The neuron averages 1.882 (on for 61.7%), raising the q and g scores a little; the formula calls them g.
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 4.0% of jets, neuron 5.82. Mostly tops (94%), 4.02% of jets: mass 82.4 GeV, width 0.0263, soft (sum pT 536 GeV), with 34.63% of the pT at 0.1-0.15 and the rest spread to 0.3. All the narrowness terms fail (width < 0.00624 and mass_over_sum_pt < 0.0767 never pass), but lam2 > 0.00114 (+6.458) and girth2 > 0.0165, which almost only pass here, lift the neuron over the -2.15 intercept. It sits at 5.818, raising the q (+1.477) and g scores and lowering the W and Z scores; the formula calls them t.

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

- **medium-mass (51 GeV), average width, pT spread over several particles** — 30.9% of jets, neuron 2.34. A Z/W mixture (Z 35%, W 34%, tops 18%), 30.95% of jets: mass 51.4 GeV, width 0.0067, a bit softer than average, with 56.2% of the pT at 0.05-0.1 from the axis. The e2 term (+2.258) and mass > 15.3 (+2.08) carry the neuron, helped by mass_over_sum_pt < 0.108 and eccentricity > 0.898 and z_dr_0p2_0p4 < 0.0572, against lam2 < 0.00352 (-1.681) and tau21 < 0.385. With the -1.45 intercept it sits at 2.344, raising the t score (+0.879) and lowering the q score; the formula calls them W.
- **very light (9 GeV), very narrow, pT spread over several particles** — 22.6% of jets, neuron 0.09. Mostly gluons (41%) with 36% quarks, 22.58% of jets: mass 9.07 GeV, width 0.0004, average pT, with 78.2% of the pT inside 0.025. The thin-jet penalties lam1 < 0.00422 (-2.499), lam2 < 0.00352 and girth2 < 0.00165 all pass and cancel mass_over_sum_pt < 0.108 (+1.908), mass_over_sum_pt_sq < 0.00396 and the two girth2_top tests, while mass > 15.3 passes for only 14.1% and the e2 term is small. The neuron stays near 0 (mean 0.086) and barely moves the scores; the formula calls them g.
- **light (30 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 1.17. A mixture (gluons 27%, W 27%, Z 19%, quarks 17%, tops 10%), 13.69% of jets: mass 29.5 GeV, width 0.0028, average pT, with 54.46% of the pT at 0.025-0.05. mass_over_sum_pt < 0.108 (+1.296), the e2 term (+1.072), mass > 15.3 and mass_over_sum_pt_sq < 0.00396 outweigh lam2 < 0.00352 (-1.711) and lam1 < 0.00422; mass > 53.2 never passes. The neuron sits at 1.174, raising the t score and lowering the q score; the formula splits them between g and W.
- **very light (10 GeV), very narrow, leading particle 43% of pT, high pT** — 9.5% of jets, neuron 0.00. Mostly quarks (63%) with 19% gluons, 9.51% of jets: mass 9.72 GeV, width 0.0002, very hard (sum pT 965 GeV, leading particle 418.91 GeV), with 95.61% of the pT inside 0.025. Like group 1 but harder: lam1 < 0.00422 (-2.639), log_sum_pt > 6.7 (-2.0), lam2 < 0.00352 and girth2 < 0.00165 outweigh mass_over_sum_pt < 0.108 (+1.97) and the other small-size bonuses, and the e2 term is tiny. The neuron is almost always 0 (mean 0.001) and adds nothing to the scores; the formula calls them q.
- **medium-mass (70 GeV), average width, leading particle 42% of pT, high pT** — 7.5% of jets, neuron 1.36. Mostly Z (49%) with 37% W, 7.52% of jets: mass 69.8 GeV, width 0.0061, hard (sum pT 920 GeV, leading particle 389.34 GeV), with the pT at 0.025-0.1 from the axis. mass > 15.3 (+3.14) and the e2 term (+1.969) are pulled back by lam2 < 0.00352, log_sum_pt > 6.7 (-1.429), tau21 < 0.385 and mass > 53.2, the last three passing far more often than in other groups. The neuron sits at 1.365, raising the t score (+0.512) and lowering the q score; the formula splits them evenly between Z and W.
- **medium-mass (74 GeV), very wide, pT spread over several particles, low pT** — 6.4% of jets, neuron 3.81. Mostly tops (76%) with 16% gluons, 6.43% of jets: mass 74 GeV, width 0.0177, soft (sum pT 588 GeV), with 36.69% of the pT at 0.1-0.15. The e2 term (+3.744), mass > 15.3 (+3.383) and girth2 > 0.00871 (always, +1.807) outweigh lam2 < 0.00352, mass > 53.2, LHA > 0.347 and LHA > 0.303; mass_over_sum_pt < 0.108 passes for only 7.9%. The neuron sits at 3.805, raising the t score (+1.427) and lowering the q score; the formula calls them t.
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 3.3% of jets, neuron 6.16. Mostly tops (74%) with 18% gluons, 3.33% of jets: mass 94.2 GeV, width 0.0297, soft (sum pT 569 GeV), with the pT spread over 0.1-0.3 from the axis. The e2 term (+5.313), mass > 15.3 (+4.542) and girth2 > 0.00871 are much larger than LHA > 0.347 (-2.472), mass > 53.2, LHA > 0.303 and lam2 < 0.00352. The neuron sits at 6.162, raising the t score (+2.311) and lowering the q score; the formula calls them t.
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.1% of jets, neuron 7.49. Mostly tops (91%), 3.12% of jets: mass 73.3 GeV, width 0.0207, soft (sum pT 543 GeV), with 35.87% of the pT at 0.05-0.1 and 31.95% at 0.1-0.15. lam2 > 0.000205 always passes (+3.47) while lam2 < 0.00352 passes for only 20.7%, so the thin-jet penalty is gone; with the e2 term (+4.254), mass > 15.3 and girth2 > 0.00871 this outweighs LHA > 0.347, mass > 53.2 and lam2 > 0.000209 and n_pt_above_50 < 8.04. The neuron sits at 7.491, raising the t score (+2.809) and lowering the q score; the formula calls them t.
- **medium-mass (88 GeV), very wide, pT spread over several particles, low pT** — 2.0% of jets, neuron 10.77. Mostly tops (94%), 2.02% of jets: mass 88 GeV, width 0.0303, soft (sum pT 525 GeV), with the pT spread to 0.1-0.3 (36.04% at 0.1-0.15). lam2 > 0.000205 (+6.524), the e2 term (+5.734), girth2 > 0.00871 and mass > 15.3 all add, lam2 < 0.00352 never passes, and only LHA > 0.347 (-2.708), lam2 > 0.000209 and n_pt_above_50 < 8.04 and mass > 53.2 subtract. The neuron reaches 10.771, its largest group mean, raising the t score (+4.039) and lowering the q score (-1.346); the formula calls them t.
- **very light (15 GeV), very narrow, leading particle 44% of pT, high pT** — 0.9% of jets, neuron 0.00. Mostly gluons (51%) with 35% quarks, 0.86% of jets: mass 14.6 GeV, width 0.0003, very hard (sum pT 1250 GeV, leading particle 558.73 GeV), with 91.68% of the pT inside 0.025. lam1 < 0.00426 and sum_pt > 994 passes only here (-7.738), on top of log_sum_pt > 6.7 (-4.981) and the thin-jet penalties lam1 < 0.00422 and lam2 < 0.00352, which the small-size bonuses cannot offset. The neuron is 0 for all of them and adds nothing to any score; the formula calls them g.

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

- **very light (9 GeV), very narrow, pT spread over several particles, high pT** — 25.4% of jets, neuron 1.38. Mostly quarks (51%) with 34% gluons, 25.39% of jets: mass 8.8 GeV, width 0.0002, harder than average (sum pT 837 GeV), with 95.22% of the pT inside 0.025. width < 0.0086 (+11.66) and centroid_offset < 0.0499 (+5.496) outweigh girth < 0.0879 (-7.976) and the thin-jet penalties LHA < 0.187 (-1.974), width < 0.00368 and e2_sq < 0.00575, all passing; LHA < 0.187 and mass < 22.3 pass almost only here. The neuron sits at 1.378, raising the W score (+0.517); the formula calls them q.
- **light (48 GeV), average width, pT spread over several particles** — 21.4% of jets, neuron 4.29. Mostly W (47%) with 28% Z, 10% tops and 10% gluons, 21.4% of jets: mass 47.8 GeV, width 0.0048, average pT, with the pT at 0.025-0.1 from the axis (45.66% at 0.025-0.05). width < 0.0086 (+5.351), centroid_offset < 0.0499 (+4.106) and planar_flow < 0.268 add, against girth < 0.0879 (-3.015) and centroid_offset < 0.0494 and log_sum_pt < 6.8; the thin-jet penalties are small and girth > 0.0754 almost never passes. The neuron reaches 4.288, its largest group mean, raising the W score (+1.608); the formula calls them W.
- **medium-mass (59 GeV), average width, pT spread over several particles** — 18.1% of jets, neuron 2.74. Mostly Z (51%) with 24% W and 16% tops, 18.07% of jets: mass 59.1 GeV, width 0.0074, average pT, with 64% of the pT at 0.05-0.1. centroid_offset < 0.0499 (+4.674), planar_flow < 0.268 and width < 0.0086 (+1.921) add, while centroid_offset < 0.0494 and log_sum_pt < 6.8, girth < 0.0879, girth > 0.0754 and planar_flow < 0.235 and width > 0.00609 subtract. The neuron sits at 2.742 (on for 79.5%), raising the W score (+1.028); the formula calls them Z.
- **light (21 GeV), very narrow, pT spread over several particles** — 16.9% of jets, neuron 2.68. A mixture (gluons 33%, W 23%, quarks 20%, Z 18%), 16.94% of jets: mass 20.9 GeV, width 0.0016, average pT, with 54.16% of the pT at 0.025-0.05. width < 0.0086 (+9.683) and centroid_offset < 0.0499 (+3.704) beat girth < 0.0879 (-5.584), centroid_offset < 0.0494 and log_sum_pt < 6.8, e2_sq < 0.00575 and width < 0.00368, both of which pass for all. The neuron sits at 2.682, raising the W score (+1.006); the formula calls them g.
- **medium-mass (64 GeV), very wide, pT spread over several particles, low pT** — 6.5% of jets, neuron 0.02. Mostly tops (67%) with 19% gluons, 6.5% of jets: mass 63.9 GeV, width 0.0133, soft (sum pT 596 GeV), with 54.65% of the pT at 0.05-0.1. Too wide for width < 0.0086 and girth < 0.0879, so girth > 0.0754 (-5.024), planar_flow < 0.235 and width > 0.00609 and centroid_offset > 0.0144 decide, only partly offset by girth > 0.0761 and n_pt_above_50 < 7.11 (+2.833) and centroid_offset < 0.0499. The neuron is almost always 0 (mean 0.024); the formula calls them t.
- **medium-mass (79 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 0.00. Mostly tops (89%), 4.24% of jets: mass 79 GeV, width 0.0218, soft (sum pT 563 GeV), with 38.98% of the pT at 0.1-0.15. girth > 0.0754 (-11.382), girth > 0.0994 and n_pt_above_50 < 6.38 (-3.407) and girth > 0.0763 and pt_7 < 40.3 outweigh girth > 0.0761 and n_pt_above_50 < 7.11 (+7.137) and centroid_offset < 0.0499; width < 0.0086 never passes. The neuron is almost always 0 (mean 0.003); the formula calls them t.
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 2.8% of jets, neuron 0.00. Mostly tops (88%), 2.82% of jets: mass 76.6 GeV, width 0.0264, very soft (sum pT 493 GeV), with 39.57% of the pT at 0.1-0.15. girth > 0.0761 and n_pt_above_50 < 7.11 (+15.704) is cancelled by girth > 0.0754 (-14.488), and girth > 0.0994 and n_pt_above_50 < 6.38 (always, -11.147) and girth > 0.0763 and pt_7 < 40.3 push the sum well below zero. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (87 GeV), very wide, pT spread over several particles** — 2.8% of jets, neuron 0.00. Mostly tops (71%) with 20% gluons, 2.78% of jets: mass 87.2 GeV, width 0.0222, near-average pT (sum pT 609 GeV), with 41% of the pT at 0.1-0.15. planar_flow < 0.235 and width > 0.00609 passes for all (-11.254) together with girth > 0.0754 (-11.524), and girth > 0.0761 and n_pt_above_50 < 7.11 (+6.211), centroid_offset < 0.0499 and planar_flow < 0.268 cannot make up the gap. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 1.0% of jets, neuron 0.00. Mostly tops (65%) with 22% gluons and 13% quarks, 1.04% of jets: mass 84.1 GeV, width 0.0295, soft (sum pT 517 GeV), with the pT spread over 0.1-0.3. girth > 0.0761 and n_pt_above_50 < 7.11 (+17.434) is swamped by girth > 0.0754 (-16.382), planar_flow < 0.235 and width > 0.00609 (-16.219), girth > 0.0994 and n_pt_above_50 < 6.38 (-13.097) and girth > 0.0763 and pt_7 < 40.3. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (72 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.00. Mostly tops (74%) with 20% gluons, 0.82% of jets: mass 72.1 GeV, width 0.0347 (the widest group), the softest group (sum pT 410 GeV, leading particle 115.87 GeV), with 38.53% of the pT at 0.15-0.2. girth > 0.0761 and n_pt_above_50 < 7.11 (+29.076) is outweighed by girth > 0.0994 and n_pt_above_50 < 6.38 (-25.133), girth > 0.0754 (-19.378) and girth > 0.0763 and pt_7 < 40.3 (-8.878). The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.

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

- **light (21 GeV), very narrow, pT spread over several particles** — 48.8% of jets, neuron 4.87. A mixture (gluons 30%, quarks 30%, W 20%, Z 16%), 48.82% of jets: mass 20.5 GeV, width 0.0016, average pT, with 55.5% of the pT inside 0.025 and 34.43% at 0.025-0.05. The compactness bonuses girth < 0.149 (+7.196), lam1 < 0.016, width < 0.0132 and centroid_offset < 0.0378 all pass and outweigh e2 < 0.0501 (-4.687), lam2 < 0.00031 and centroid_offset < 0.0495, lam1 < 0.0168 and centroid_offset < 0.0374 and girth < 0.147 and log_sum_pt < 6.8; the high-pT tests almost never pass. The neuron sits at 4.871, raising the W and Z scores and lowering the t score (-1.979); the formula splits them over g, q and W.
- **medium-mass (63 GeV), wide, pT spread over several particles** — 45.9% of jets, neuron 2.88. Mostly tops (38%) with 26% Z and 21% W, 45.94% of jets: mass 62.7 GeV, width 0.0121, softer than average (sum pT 643 GeV), with 50.64% of the pT at 0.05-0.1. The compactness bonuses pass less often here (girth < 0.149 89.7%, lam1 < 0.016 77.7%, width < 0.0132 68%), and e2 < 0.0501 with tau21 < 0.515 and max_dr > 0.015 and mass > 53.5 subtract. The neuron sits at 2.878 (on for 72.3%), raising the W and Z scores and lowering the t score (-1.169); the formula calls them t.
- **light (26 GeV), very narrow, leading particle 56% of pT, high pT** — 2.3% of jets, neuron 1.72. Mostly quarks (62%) with 17% Z and 16% W, 2.31% of jets: mass 26.1 GeV, width 0.0015, a very hard leading particle (531.77 GeV) and an almost empty tail (eighth 11.38 GeV), with 79.63% of the pT inside 0.025. z_top5_slots > 0.931 passes for 99.9% (-12.818), nearly unique to this group, and with e2 < 0.0501, z_7 < 0.0289 (-4.523), girth < 0.152 and pt_7 < 38.1 and pt_6 < 31.3 it outweighs girth < 0.149 (+7.729), sum_pt_top5 > 657 and pt_7 < 42.1 and sum_pt_top5 > 903. The neuron is on for only 25.6% (mean 1.72), lowering the t score when on; the formula calls them q.
- **light (30 GeV), very narrow, leading particle 42% of pT, high pT** — 1.2% of jets, neuron 7.39. Mostly gluons (41%) with 32% quarks, 1.18% of jets: mass 29.8 GeV, width 0.0018, very hard (sum pT 1087 GeV, leading particle 449.19 GeV), with 69.88% of the pT inside 0.025. sum_pt > 988 always passes (+18.874), and sum_pt_top5 > 903 (+13.834), sum_pt_top5 > 902 and D2 < 3.86 and sum_pt > 989 and D2 < 3.94 add more on top of girth < 0.149, with sum_pt > 988 and n_pt_above_50 > 6 taking back only part. Their average sum lies well above the largest_value 16, so the neuron's value wraps around and is effectively scrambled (mean 7.389); on average it lowers the t score and raises the W and Z scores, and the formula splits them g 40% / q 35%.
- **light (22 GeV), very narrow, leading particle 59% of pT, high pT** — 0.9% of jets, neuron 6.36. Mostly quarks (69%), 0.91% of jets: mass 22.1 GeV, width 0.001, an extremely hard leading particle (610.8 GeV) and a near-empty tail (eighth 7.71 GeV), with 90.7% of the pT inside 0.025. sum_pt_top5 > 903 passes for all (+25.969) and z_top5_slots > 0.931 (-22.37) nearly cancels it, but sum_pt > 988, girth < 0.149 and sum_pt_top5 > 657 and pt_7 < 42.1 add far more than z_7 < 0.0289 and e2 < 0.0501 take. The average sum is above the largest_value 16, so the value wraps around and is effectively scrambled (mean 6.36, on for 81.1%); on average it lowers the t score, and the formula calls them q.
- **light (27 GeV), very narrow, leading particle 51% of pT, high pT** — 0.5% of jets, neuron 7.93. Mostly quarks (42%) with 33% gluons, 14% Z and 10% W, 0.47% of jets: mass 27.4 GeV, width 0.0012, very hard (sum pT 1193 GeV, leading particle 612.52 GeV), with 77.75% of the pT inside 0.025. The high-pT terms pile up: sum_pt_top5 > 903 (+51.716), sum_pt > 988 (+39.012), sum_pt_top5 > 902 and D2 < 3.86 and sum_pt > 989 and D2 < 3.94, with only small penalties. The sum lies far above the largest_value 16, so the value wraps around and is effectively scrambled (mean 7.93); on average it lowers the t score, and the formula splits them q 49% / g 29%.
- **light (29 GeV), very narrow, pT spread over several particles, high pT** — 0.2% of jets, neuron 7.48. Mostly gluons (60%), 0.18% of jets: mass 28.5 GeV, width 0.0014, very hard (sum pT 1246 GeV) with the pT spread over many hard particles (eighth 56.17 GeV), 70.95% of it inside 0.025. Having more than six particles above 50 GeV switches on sum_pt_top5 > 902 and n_pt_above_50 > 6 (-56.041) and sum_pt > 988 and n_pt_above_50 > 6 (-54.271), which fight the high-pT bonuses sum_pt > 988 (+48.937), sum_pt_top5 > 903, the two D2 tests and sum_pt_top5 > 840 and n_pt_above_50 > 2.02. The average sum still lies above the largest_value 16, so the value wraps around and is effectively scrambled (mean 7.484); the formula calls them g.
- **light (33 GeV), very narrow, leading particle 57% of pT, high pT** — 0.1% of jets, neuron 8.79. Mostly gluons (45%) with 36% quarks, 0.11% of jets: mass 33 GeV, width 0.0015, the hardest leading particle of all (824.23 GeV, sum pT 1440 GeV), with 79.52% of the pT inside 0.025. The high-pT terms are enormous here: sum_pt_top5 > 903 (+113.21), sum_pt > 988 (+85.895), sum_pt_top5 > 902 and D2 < 3.86 and sum_pt > 989 and D2 < 3.94, with no large penalty. The sum lies far above the largest_value 16, so the value wraps around and is effectively scrambled (mean 8.794); the formula calls them g.
- **light (37 GeV), very narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 6.15. Mostly gluons (67%) with 18% W, only 0.06% of jets: mass 37 GeV, width 0.0018, the hardest group (sum pT 1463 GeV) with the pT shared over many hard particles (eighth 58.75 GeV), 68.89% of it inside 0.025. sum_pt_top5 > 902 and n_pt_above_50 > 6 (-146.549) and sum_pt > 988 and n_pt_above_50 > 6 (-100.923) are answered by sum_pt > 988 (+90.273), sum_pt_top5 > 903, the two D2 tests and sum_pt_top5 > 840 and n_pt_above_50 > 2.02. The average sum lies above the largest_value 16, so the value wraps around and is effectively scrambled (mean 6.147, on for 82.1%); the formula calls them g.

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

- **very light (8 GeV), very narrow, pT spread over several particles** — 28.5% of jets, neuron 0.00. Mostly quarks (46%) with 34% gluons, 28.54% of jets: mass 7.9 GeV, width 0.0003, harder than average (sum pT 814 GeV), with 88.19% of the pT inside 0.025. e2 < 0.0443 (+8.422), girth2 < 0.0133 and width < 0.00746 and n_dr_0p1_0p2 < 3 add, but the narrowness penalties girth < 0.0872 (-8.144), width < 0.00609 (-6.12), width < 0.00747 and max_dr < 0.0804 take more, and lam1 > 0.00428 never passes. The neuron is 0 for all of them and adds nothing to any score; the formula calls them q.
- **medium-mass (57 GeV), average width, pT spread over several particles** — 22.2% of jets, neuron 1.02. Mostly Z (45%) with 30% W and 16% tops, 22.24% of jets: mass 57.3 GeV, width 0.0072, average pT, with 67.61% of the pT at 0.05-0.1 from the axis. girth2 < 0.0133 (+2.811), lam1 > 0.00428 (+1.753) and e2 < 0.0443 add, while C2 < 0.0669 (-2.094) and girth < 0.0872 subtract and the width penalties mostly fail; lam1 > 0.00543 and max_dr < 0.161 passes almost only here. The neuron sits at 1.025, raising the Z score and lowering the W score (-0.769); the formula calls them Z.
- **light (47 GeV), average width, pT spread over several particles** — 17.1% of jets, neuron 0.52. Mostly W (42%) with 30% Z, 11% gluons and 11% tops, 17.08% of jets: mass 47.4 GeV, width 0.0048, average pT, with 50.91% of the pT at 0.025-0.05. girth2 < 0.0133 (+3.946) and e2 < 0.0443 (+3.817) are nearly cancelled by girth < 0.0872 (-3.233), width < 0.00747, C2 < 0.0669 and width < 0.00609, and lam1 > 0.006 rarely passes. The neuron is on for 40.4%, lowering the W score (-0.388) and raising the Z score; the formula calls them W.
- **light (27 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 0.10. A mixture (gluons 29%, W 25%, quarks 21%, Z 18%), 13.66% of jets: mass 27 GeV, width 0.002, average pT, with 51.28% of the pT at 0.025-0.05. e2 < 0.0443 (+6.586) and girth2 < 0.0133 (+5.211) are outweighed by girth < 0.0872 (-5.643), width < 0.00609 (-4.247) and width < 0.00747, and lam1 > 0.00428 almost never passes. The neuron stays near 0 (mean 0.103) with little effect on the scores; the formula calls them g.
- **medium-mass (71 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 0.16. Mostly tops (74%) with 18% gluons, 7.76% of jets: mass 71 GeV, width 0.0166, soft (sum pT 582 GeV), with 39.35% of the pT at 0.1-0.15. The narrow-jet tests fail (girth < 0.0872 4.4%, e2 < 0.0443 13%), so lam1 > 0.00428 (+6.98) against lam1 > 0.006 (-5.638), lam1 > 0.00418 and D2 > 0.451 and C2 < 0.0669 decides. The neuron stays low (mean 0.159, on for 26.4%) with little effect on the scores; the formula calls them t.
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 5.0% of jets, neuron 0.00. Mostly tops (90%), 4.95% of jets: mass 82.3 GeV, width 0.0254, soft (sum pT 541 GeV), with the pT spread over 0.05-0.3 from the axis. lam1 > 0.00428 (+10.819) and lam1 > 0.0023 and D2 > 0.452 (+3.525) are outweighed by lam1 > 0.006 (-9.329) and lam1 > 0.00418 and D2 > 0.451 (always, -9.295), which with the small intercept leaves the sum below zero. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.7% of jets, neuron 0.09. Mostly tops (70%) with 22% gluons, 2.73% of jets: mass 90.6 GeV, width 0.0317, soft (sum pT 533 GeV), with 31.78% of the pT at 0.15-0.2. lam1 > 0.00428 (+16.717) and lam1 > 0.006 (-15.0) nearly cancel, and lam1 > 0.00418 and D2 > 0.451 (-3.701) usually tips the sum below zero. The neuron stays near 0 (mean 0.085, on for 17.9%) with almost no effect on the scores; the formula calls them t.
- **light (30 GeV), very narrow, pT spread over several particles** — 1.7% of jets, neuron 0.00. Mostly gluons (37%) with 28% quarks and 18% W, 1.66% of jets: mass 29.7 GeV, width 0.0019, pT shared evenly (eighth particle 41.32 GeV), with 55.09% of it at 0.025-0.05. girth2 < 0.00436 and D2 < 0.884 passes for all (-8.244), almost only here, and together with width < 0.0089 and D2 < 1.03 it turns the group 3 balance (e2 < 0.0443 and girth2 < 0.0133 against girth < 0.0872 and the width tests) clearly negative. The neuron is 0 for all of them and adds nothing to any score; the formula calls them g.
- **medium-mass (66 GeV), very wide, pT spread over several particles, low pT** — 1.1% of jets, neuron 0.00. Mostly tops (72%) with 17% gluons, 1.1% of jets: mass 65.9 GeV, width 0.0148, average pT, with 47.38% of the pT at 0.05-0.1. lam1 > 0.00594 and D2 > 1.68 passes for all (-12.893), almost only here, and with lam1 > 0.00418 and D2 > 0.451 (-10.719) and lam1 > 0.006 it outweighs lam1 > 0.00428 (+5.392) and lam1 > 0.0023 and D2 > 0.452. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **medium-mass (71 GeV), very wide, pT spread over several particles** — 0.3% of jets, neuron 0.00. Mostly tops (83%), 0.27% of jets: mass 71.1 GeV, width 0.0161, a harder leading particle (225.39 GeV), with the pT spread from the core (12.92% inside 0.025) out to 0.1. lam1 > 0.00594 and D2 > 1.68 subtracts -36.703 here and lam1 > 0.00418 and D2 > 0.451 another -17.876, far more than lam1 > 0.0023 and D2 > 0.452 and lam1 > 0.00428 add. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.

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

- **medium-mass (56 GeV), average width, pT spread over several particles** — 23.6% of jets, neuron 2.47. A Z/W mixture (Z 45%, W 35%, tops 11%), 23.56% of jets: mass 56.3 GeV, width 0.0067 and average pT, with 68.24% of the pT at 0.05-0.1 from the axis, as two separated prongs would give. girth2 < 0.0133 always passes (+2.356), helped by width < 0.00876, centroid_offset < 0.0327 and girth2 < 0.019 and eccentricity > 0.955; the lightness penalties are weak here (girth < 0.0764 passes for only 53.4%), and width < 0.00435 never passes. With the intercept the neuron sits at 2.471, which raises the W score (+0.849) and lowers the g score; the formula splits them between Z and W.
- **medium-mass (75 GeV), very wide, pT spread over several particles, low pT** — 19.0% of jets, neuron 0.13. Mostly tops (76%, plus 15% gluons), 18.99% of jets: mass 74.7 GeV, width 0.0202 (about three times the average), softer than average (sum pT 566 GeV), with the pT spread out to 0.1-0.2 from the axis (31.49% at 0.1-0.15). They are too wide for width < 0.00876 (0.2% pass) and girth < 0.0764 and mostly fail girth2 < 0.0133, so neither the big bonuses nor the big penalties apply; the leftovers, centroid_offset < 0.0327 (+0.242) and mass < 63.4 and centroid_offset > 0.0108 (+0.225) against mass < 59.6, stay below the -0.847 intercept for most jets. The neuron averages 0.129 (on for 17.7%) and barely moves any score; the formula calls them t on the strength of other neurons.
- **very light (6 GeV), very narrow, pT spread over several particles, high pT** — 17.6% of jets, neuron 0.00. Mostly quarks (55%) with 31% gluons, 17.62% of jets: mass 6.21 GeV, width 0.0001, harder than average (sum pT 849 GeV, leading particle 317.73 vs 240.22 GeV), with 98.28% of the pT inside 0.025 of the axis. Every narrowness and lightness test passes: the bonuses mass < 28.9 and dr_0 < 0.109 (+7.945), girth2 < 0.0133 and width < 0.00876 are outweighed by mass < 29.1 (-8.905), girth < 0.0764 (-8.336) and mass < 59.6. The sum is far below zero, so the neuron stays at 0 and adds nothing to any score; the formula calls them q.
- **light (46 GeV), narrow, pT spread over several particles** — 14.8% of jets, neuron 1.50. A W/Z mixture (W 41%, Z 29%, gluons 12%, tops 11%), 14.75% of jets: mass 45.6 GeV, width 0.0044, average pT, with 60.6% of the pT at 0.025-0.05 from the axis, so the prongs are closer than in group 0. girth2 < 0.0133 (+3.146) and width < 0.00876 pass, while girth < 0.0764 always passes (-2.7) and mass < 59.6 mostly passes; what separates them from the light groups is that mass < 29.1 and its paired bonus mass < 28.9 and dr_0 < 0.109 rarely pass. The neuron sits at 1.504 (on for 84.6%), raising the W score (+0.517) and lowering the g score; the formula calls them W.
- **very light (8 GeV), very narrow, pT spread over several particles** — 9.3% of jets, neuron 0.02. Mostly gluons (40%) with 22% quarks, 16% W and 16% Z, 9.28% of jets: mass 8.07 GeV, width 0.0009, average pT, with 55.96% of the pT inside 0.025 and 38.21% at 0.025-0.05 from the axis. As in group 2 all the light-and-narrow tests pass: mass < 29.1 (-8.182), girth < 0.0764 (-6.219) and mass < 59.6 outweigh mass < 28.9 and dr_0 < 0.109 (+5.962), girth2 < 0.0133 and width < 0.00876. The neuron stays near 0 (mean 0.018) and adds almost nothing to the scores; the formula calls them g.
- **light (36 GeV), narrow, pT spread over several particles** — 7.6% of jets, neuron 0.83. A mixture (W 31%, quarks 24%, gluons 20%, Z 19%), 7.61% of jets: mass 36.2 GeV, width 0.0023, hard (sum pT 819 GeV, leading particle 336.57 GeV), with 44.18% of the pT inside 0.025 and 46.12% at 0.025-0.05. girth < 0.0764 always passes (-5.222) against girth2 < 0.0133 (+3.908) and width < 0.00876; mass < 29.1 passes for only 32.6%, so the heaviest light-jet penalty is mostly absent, while sum_pt > 821 and mass < 64 and pt_7 < 40.1 take a little. The neuron lands near the edge (on for 54.2%), adding to the W score (+0.286) and lowering the g score; the formula calls them W, with sizeable q and g shares.
- **very light (17 GeV), very narrow, pT spread over several particles** — 7.5% of jets, neuron 0.01. Mostly gluons (41%) with 35% quarks, 7.49% of jets: mass 17.3 GeV, width 0.0011, average pT, with 54.79% of the pT inside 0.025 and 34.65% at 0.025-0.05. girth < 0.0764 (-6.009), mass < 29.1 (-4.591) and mass < 59.6 always pass and beat girth2 < 0.0133 (+4.333), width < 0.00876 and mass < 28.9 and dr_0 < 0.109. The neuron stays near 0 (mean 0.013) and hardly touches the scores; the formula calls them g, with q 35%.
- **light (23 GeV), very narrow, pT spread over several particles, high pT** — 0.5% of jets, neuron 0.00. Mostly gluons (69%), a rare 0.55% of jets: mass 22.8 GeV, width 0.0012, very hard (sum pT 1138 GeV) with even the eighth particle hard enough for sum_pt > 904 and pt_7 > 29.4, and 75.97% of the pT inside 0.025. sum_pt > 904 and pt_7 > 29.4 always passes and alone subtracts -13.797, joined by sum_pt > 821, sum_pt > 889 and the usual girth < 0.0764 (-7.016) penalty. The neuron is pinned at 0 and adds nothing to any score; the formula calls them g.
- **light (34 GeV), narrow, pT spread over several particles, high pT** — 0.1% of jets, neuron 0.00. Mostly gluons (71%) with 15% W, only 0.15% of jets: mass 34.4 GeV, width 0.0021, the hardest group (sum pT 1328 GeV, leading particle 428.95 GeV), with 70.26% of the pT inside 0.025. sum_pt > 904 and pt_7 > 29.4 passes for all and subtracts -40.548 on average here, far more than any bonus, with sum_pt > 821 and sum_pt > 889 adding further penalties. The neuron is pinned at 0 and adds nothing to any score; the formula calls them g.

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

- **very light (9 GeV), very narrow, pT spread over several particles** — 30.6% of jets, neuron 0.02. Mostly quarks (45%) with 34% gluons, 30.56% of jets: mass 8.62 GeV, width 0.0003, harder than average (sum pT 813 GeV), with 86.87% of the pT inside 0.025 of the axis. The two narrowness tests girth2 < 0.00872 (-10.69) and girth2 < 0.0126 (-6.579) pass for all and beat e2 < 0.0445 (+7.825), girth2 < 0.00389 (+3.519) and z_dr_0p1_0p2 < 0.458, while lam1 > 0.00353 and e2 > 0.0269 never pass. The neuron stays near 0 (mean 0.021) and barely touches the scores; the formula calls them q.
- **medium-mass (54 GeV), average width, pT spread over several particles** — 16.5% of jets, neuron 0.20. Mostly W (50%) with 31% Z and 10% tops, 16.51% of jets: mass 53.8 GeV, width 0.0059, average pT, with 58.79% of the pT at 0.05-0.1 and little beyond 0.15. girth2 < 0.0126 (-3.577) and girth2 < 0.00872 (-3.564) pass and e2 < 0.0445 (+1.896) only covers part; lam1 > 0.00353 (-1.195) and e2 > 0.0269 also pass, while z_dr_0p1_0p2 < 0.458, mass > 36 and eccentricity > 0.688 and mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 give some back. The neuron stays low (mean 0.195, on for 12.3%), lowering the W and Z scores slightly; the formula calls them W.
- **light (44 GeV), narrow, pT spread over several particles** — 13.3% of jets, neuron 0.27. Mostly W (41%) with 28% Z, 13% gluons and 10% tops, 13.28% of jets: mass 44.2 GeV, width 0.0042, average pT, with 54.89% of the pT at 0.025-0.05. girth2 < 0.00872 (-5.754) and girth2 < 0.0126 (-4.5) outweigh e2 < 0.0445 (+4.246) and z_dr_0p1_0p2 < 0.458; mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 passes for only 3.3%, unlike the heavier groups. The neuron stays low (mean 0.273, on for 14.4%), slightly lowering the W and Z scores; the formula calls them W.
- **light (26 GeV), narrow, pT spread over several particles** — 11.1% of jets, neuron 0.20. A mixture (gluons 32%, W 23%, quarks 21%, Z 16%), 11.1% of jets: mass 26.1 GeV, width 0.002, average pT, with 55.79% of the pT at 0.025-0.05. girth2 < 0.00872 (-8.579) and girth2 < 0.0126 (-5.69) beat e2 < 0.0445 (+5.99), girth2 < 0.00389 (+1.889) and z_dr_0p1_0p2 < 0.458, and lam1 > 0.00353 and e2 > 0.0269 almost never pass. The neuron stays low (mean 0.204, on for 13.1%) with only a slight pull on the W and Z scores; the formula calls them g.
- **medium-mass (59 GeV), average width, pT spread over several particles** — 10.9% of jets, neuron 0.63. Mostly Z (62%) with 19% tops, 10.91% of jets: mass 59.5 GeV, width 0.008, average pT, with 65.46% of the pT at 0.05-0.1 and 19.43% at 0.1-0.15. girth2 < 0.0126 (-2.47), lam1 > 0.00353 (-2.269) and girth2 < 0.00872 subtract, while mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 (+1.726), mass > 36 and eccentricity > 0.688 and e2 < 0.0445 add; LHA > 0.312 and max_dr < 0.147, passing for 62.4% and rarely elsewhere, marks the group. The neuron is on for 30.2%, lowering the W and Z scores; the formula calls them Z.
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 5.8% of jets, neuron 3.14. Mostly tops (81%), 5.82% of jets: mass 76.9 GeV, width 0.0193, softer than average (sum pT 579 GeV), with 40.08% of the pT at 0.1-0.15 from the axis. Both girth2 tests fail here, so the big penalties vanish; lam1 > 0.00353 (-7.293) is almost fully balanced by mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 (+6.799), and centroid_offset > 0.0146 (+1.281) adds more than e2 > 0.0269 and mass > 69.5 take. With the intercept the neuron sits at 3.139, lowering the W (-1.57) and Z scores and raising the t score; the formula calls them t.
- **medium-mass (62 GeV), very wide, pT spread over several particles, low pT** — 5.5% of jets, neuron 3.70. Mostly tops (70%) with 20% gluons, 5.53% of jets: mass 62.1 GeV, width 0.0135, softer than average, with 44.67% of the pT at 0.05-0.1 and 27.74% at 0.1-0.15. girth2 < 0.00872 never passes and girth2 < 0.0126 only for 42.9%; lam1 > 0.00353 (-4.739) is more than offset by mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 (+3.268) and centroid_offset > 0.0146, with e2 > 0.0269 taking some back. The neuron sits at 3.704, lowering the W (-1.852) and Z scores and raising the t score; the formula calls them t.
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 4.5% of jets, neuron 3.37. Mostly tops (86%), 4.48% of jets: mass 86.9 GeV, width 0.0271, soft (sum pT 546 GeV), with the pT far from the axis (29.28% at 0.15-0.2, 17.95% at 0.2-0.3). width > 0.0182 always passes (+3.445) and, with mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 (+9.704), outweighs lam1 > 0.00353 (-10.594), e2 > 0.0269 and mass > 69.5; the girth2 penalties never apply. The neuron sits at 3.37, lowering the W and Z scores (-1.896 on Z) and raising the t score (+0.211); the formula calls them t.
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 1.6% of jets, neuron 4.09. Mostly tops (68%) with 23% gluons, 1.59% of jets: mass 93.6 GeV, width 0.0362 (the widest group), soft (sum pT 512 GeV), with 34.54% of the pT at 0.2-0.3 from the axis. The width terms dominate: lam1 > 0.00353 (-16.05) is answered by mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63 (+12.389) and width > 0.0182 (+6.971), with e2 > 0.0269 and mass > 69.5 subtracting. The neuron reaches 4.091, its largest group mean, lowering the W (-2.046) and Z (-2.301) scores and raising the t score; the formula calls them t.
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 0.2% of jets, neuron 2.30. Mostly tops (93%), a rare 0.24% of jets: mass 81.4 GeV, width 0.0247, soft, with an unusual split of the pT between the core (25.56% inside 0.025) and far out (24.68% at 0.2-0.3). Only here does mass_over_sum_pt > 0.109 and dr_7 < 0.0487 pass (+13.584), together with its partner mass_over_sum_pt > 0.0689 and dr_7 < 0.0428 (-8.53); with lam1 > 0.00353 (-10.85), e2 > 0.0269 and C2 > 0.0926 the sum lands close to zero. The neuron is on for 43.7% (mean 2.304), lowering the W and Z scores and raising the t score a little; the formula calls them t.

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

- **very light (10 GeV), very narrow, pT spread over several particles** — 36.1% of jets, neuron 1.24. Mostly quarks (42%) with 35% gluons, 36.08% of jets: mass 9.92 GeV, width 0.0005, harder than average (sum pT 787 GeV), with 78.21% of the pT inside 0.025 of the axis. e2_sq < 0.00295 always passes (-4.526) but is outweighed by girth < 0.0766 (+4.05) and mass < 61.3 (+3.206), while centroid_offset < 0.016 and z_dr_0p05_0p1 < 0.541 and lam2 < 0.000297 and mass_top3 < 43.8 subtract; girth2 > 0.00354 and e2 > 0.0214 almost never pass. The neuron sits at 1.244, lowering the q and g scores and raising the t and Z scores a little; the formula splits them q 46% / g 43%.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 25.1% of jets, neuron 7.06. A Z/W mixture (Z 44%, W 36%, tops 13%), 25.06% of jets: mass 57.9 GeV, width 0.0067, average pT, with 62.04% of the pT at 0.05-0.1 from the axis. The two-prong bonuses tau21 < 0.236 (+2.841), girth2 > 0.00354 (+2.323) and e2 > 0.0214 pass, and almost no penalty does: mass_over_sum_pt > 0.108 never passes, and mass_over_sum_pt > 0.09 and girth2 > 0.00739 pass only for a minority. The neuron reaches 7.056, its largest group mean, raising the t (+0.882) and Z scores and lowering the q and g scores; the formula splits them between Z and W.
- **light (41 GeV), narrow, pT spread over several particles** — 21.1% of jets, neuron 3.06. A mixture led by W (36%) with 26% Z, 17% gluons, 11% quarks and 10% tops, 21.09% of jets: mass 40.8 GeV, width 0.004, average pT, with 51.16% of the pT at 0.025-0.05. girth < 0.0766 (+1.753) and mass < 61.3 (+1.295) are partly cancelled by C2 > 0.0136 (-1.332) and lam2 < 0.000297 and mass_top3 < 43.8; the two-prong bonuses tau21 < 0.236 and girth2 > 0.00354 pass for only about half, and the heavy-jet penalties (mass_over_sum_pt > 0.09, girth2 > 0.00739) almost never. The neuron sits at 3.056, raising the t and Z scores and lowering the q (-0.286) and g scores; the formula calls them W.
- **medium-mass (60 GeV), wide, pT spread over several particles, low pT** — 4.8% of jets, neuron 4.80. Mostly tops (66%) with 21% gluons, 4.76% of jets: mass 60.1 GeV, width 0.0127, softer than average (sum pT 581 GeV), with 45.37% of the pT at 0.05-0.1 and 30.47% at 0.1-0.15. girth2 > 0.00354 (+6.836), e2 > 0.0214 and tau21 < 0.236 add more than girth2 > 0.00739 (-3.655), mass_over_sum_pt > 0.09 (-3.556) and C2 > 0.0136 take, and mass_over_sum_pt > 0.108 with its D2 partner passes for only about a third. The neuron sits at 4.802 (on for 83.5%), raising the t (+0.6) and Z scores and lowering the q and g scores; the formula calls them t.
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.5% of jets, neuron 2.08. Mostly tops (78%) with 15% gluons, 3.55% of jets: mass 73.1 GeV, width 0.017, soft (sum pT 589 GeV), with the pT split between 0.05-0.1 and 0.1-0.15 from the axis. Large terms of both signs all pass: girth2 > 0.00354 (+10.003), mass_over_sum_pt > 0.108 and e2 > 0.0214 against mass_over_sum_pt > 0.09 (-8.651), girth2 > 0.00739 and mass_over_sum_pt > 0.108 and D2 < 3.95 (-4.535), plus C2 > 0.0136. They nearly cancel, leaving 2.081 on average (on for 58.5%), a modest push up on the t and Z scores; the formula calls them t.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 3.4% of jets, neuron 1.02. Mostly tops (83%), 3.37% of jets: mass 79.7 GeV, width 0.0212, soft (sum pT 570 GeV), with 37.43% of the pT at 0.1-0.15. The same cancellation as group 4 at larger size: mass_over_sum_pt > 0.108 (+13.305) and girth2 > 0.00354 (+13.12) against mass_over_sum_pt > 0.09 (-12.696), girth2 > 0.00739 and mass_over_sum_pt > 0.108 and D2 < 3.95, with C2 > 0.0674 tipping more jets below zero. The neuron averages 1.022 (on for 39.8%), a small push on the t and Z scores; the formula calls them t.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 3.2% of jets, neuron 0.27. Mostly tops (88%), 3.18% of jets: mass 85.6 GeV, width 0.026, soft (sum pT 547 GeV), with the pT spread to 0.1-0.2 from the axis. mass_over_sum_pt > 0.108 (+20.176) and girth2 > 0.00354 are outweighed by mass_over_sum_pt > 0.09 (-16.891), mass_over_sum_pt > 0.108 and D2 < 3.95 (-13.856) and girth2 > 0.00739, and C2 > 0.0674 passes for 69.7%. The neuron stays low (mean 0.272, on for 16.1%) with little effect on the scores; the formula calls them t.
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.04. Mostly tops (83%), 2.33% of jets: mass 91.2 GeV, width 0.0316, soft (sum pT 526 GeV), with 31.27% of the pT at 0.15-0.2 and 26.12% at 0.2-0.3. mass_over_sum_pt > 0.108 (+27.224) and girth2 > 0.00354 are beaten by mass_over_sum_pt > 0.09 (-21.195), mass_over_sum_pt > 0.108 and D2 < 3.95 (-19.527) and girth2 > 0.00739. The neuron is almost always 0 (mean 0.044) and adds almost nothing to the scores; the formula calls them t.
- **heavy (96 GeV), very wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.02. Mostly tops (60%) with 30% gluons, 0.59% of jets: mass 95.5 GeV, width 0.0414 (the widest group), soft (sum pT 485 GeV), with 42.49% of the pT at 0.2-0.3 from the axis. The largest terms of all cancel: mass_over_sum_pt > 0.108 (+37.22) and girth2 > 0.00354 against mass_over_sum_pt > 0.108 and D2 < 3.95 (-27.712), mass_over_sum_pt > 0.09 and girth2 > 0.00739 (-23.251), leaving the sum below zero. The neuron is almost always 0 (mean 0.018); the formula calls them t.

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

- **medium-mass (62 GeV), wide, pT spread over several particles** — 51.2% of jets, neuron 0.13. A top/Z/W mixture (tops 35%, Z 27%, W 23%, gluons 10%), 51.16% of jets: mass 62.2 GeV, width 0.0114 (wider than average), a bit softer than average, with 46.64% of the pT at 0.05-0.1 from the axis. They are too wide for the big terms (width < 0.00509 passes for 12.9%); what is left, max_dr < 0.183 and lam2 < 0.000197 (+0.339) and girth2 < 0.00673 and centroid_offset < 0.0236, roughly cancels C2 < 0.0264 (-0.281) and the intercept. The neuron stays low (mean 0.129) and barely moves any score; the formula splits them t 39% / W 29% / Z 28%.
- **light (40 GeV), narrow, pT spread over several particles** — 10.1% of jets, neuron 0.77. Mostly W (41%) with 24% Z, 16% gluons and 11% quarks, 10.05% of jets: mass 39.8 GeV, width 0.0033, average pT, with 60.94% of the pT at 0.025-0.05. width < 0.00509 always passes (+4.798), but width < 0.00515 and mass_over_sum_pt_sq > 1.24e-05 (-1.839), girth < 0.0604 and width < 0.00498 and girth < 0.0613 and lam1 > 0.000235 (98.5%, mostly passing only here) take most of it back, with girth2 < 0.00673 and centroid_offset < 0.0236 adding a little. The neuron averages 0.773 (on for 47.2%), lowering the W score (-0.193) and raising the t and q scores; the formula calls them W.
- **very light (7 GeV), very narrow, pT spread over several particles** — 10.0% of jets, neuron 1.30. A quark/gluon mixture (quarks 46%, gluons 43%), 9.98% of jets: mass 6.6 GeV, width 0.0001, average pT, with 97.41% of the pT inside 0.025. width < 0.00509 (+13.524) is outweighed by girth < 0.0604 and width < 0.00498 (-15.132), and the smaller narrow-core bonuses (girth2 < 0.00673 and centroid_offset < 0.0236, girth < 0.0636 and lam2 < 0.000187, LHA < 0.202 and width < 0.000479) beat LHA < 0.196 and lam2 < 0.000294 and mass < 24.7 and lam2 < 0.000203. The neuron sits at 1.3, lowering the W score (-0.325) and raising the t and q scores; the formula splits them g 51% / q 48%.
- **very light (12 GeV), very narrow, pT spread over several particles** — 9.6% of jets, neuron 1.45. Mostly gluons (41%) with 30% quarks, 15% W and 11% Z, 9.58% of jets: mass 11.8 GeV, width 0.0006, average pT, with 75.16% of the pT inside 0.025. width < 0.00509 (+12.308) against girth < 0.0604 and width < 0.00498 (-10.686) leaves a surplus, and girth < 0.0636 and lam2 < 0.000187, girth2 < 0.00673 and centroid_offset < 0.0236 and max_dr < 0.183 and lam2 < 0.000197 add more than mass < 24.7 and lam2 < 0.000203, LHA < 0.196 and lam2 < 0.000294 and girth < 0.0632 and centroid_offset > 0.00688 take. The neuron sits at 1.454 (on for 53.9%), lowering the W score (-0.364) and raising the t and q scores; the formula calls them g.
- **very light (8 GeV), very narrow, leading particle 42% of pT, high pT** — 8.3% of jets, neuron 1.08. Mostly quarks (68%) with 16% gluons, 8.28% of jets: mass 8.08 GeV, width 0.0001, very hard (sum pT 947 GeV, leading particle 402.5 GeV), with 97.94% of the pT inside 0.025. As in group 2, width < 0.00509 (+13.572) is outweighed by girth < 0.0604 and width < 0.00498 (-15.942); the hard-jet pair girth < 0.0634 and log_sum_pt > 6.67 and log_sum_pt > 6.69 and width < 0.00721 nearly cancel, and the narrow-core bonuses tip the sum positive. The neuron sits at 1.083, lowering the W score and raising the t (+0.203) and q scores; the formula calls them q.
- **light (21 GeV), very narrow, pT spread over several particles** — 7.9% of jets, neuron 1.55. A mixture (gluons 34%, quarks 22%, W 22%, Z 16%), 7.93% of jets: mass 21.5 GeV, width 0.0015, average pT, with 58.8% of the pT at 0.025-0.05. width < 0.00509 (+9.722) is only partly cancelled by girth < 0.0604 and width < 0.00498 (-5.796), then chipped at by width < 0.00515 and mass_over_sum_pt_sq > 1.24e-05, girth < 0.0632 and centroid_offset > 0.00688, girth < 0.0613 and lam1 > 0.000235 and girth2 < 0.0068 and centroid_offset > 0.0184. The neuron sits at 1.546 (on for 55.9%), lowering the W score (-0.387) and raising the t (+0.29) and q scores; the formula calls them g.
- **very light (11 GeV), very narrow, leading particle 46% of pT, high pT** — 1.8% of jets, neuron 0.66. Mostly quarks (49%) with 41% gluons, 1.77% of jets: mass 11.1 GeV, width 0.0002, very hard (sum pT 1153 GeV, leading particle 531.46 GeV), with 96.68% of the pT inside 0.025. Like group 4 at higher pT: girth < 0.0634 and log_sum_pt > 6.67 (+10.321) and log_sum_pt > 6.69 and width < 0.00721 (-9.983) cancel, width < 0.00509 is outweighed by girth < 0.0604 and width < 0.00498, and sum_pt_top5 > 655 and girth2 < 0.0017 subtracts more. The neuron is on for 62.2%, a small pull down on the W score and push up on the t and q scores; the formula calls them q.
- **very light (8 GeV), narrow, pT spread over several particles, low pT** — 0.9% of jets, neuron 0.00. Mostly gluons (41%) with 24% Z, 15% tops and 13% quarks, 0.91% of jets: mass only 7.61 GeV yet width 0.0025, soft (sum pT 598 GeV), with just 2.35% of the pT inside 0.025 and 65.34% at 0.025-0.05. mass < 21.6 and centroid_offset > 0.0322 passes for all (-17.493) and, with girth2 < 0.0068 and centroid_offset > 0.0184 and C2 < 0.0264 and centroid_offset > 0.0157, overwhelms width < 0.00509 (+7.376). The neuron is 0 for all of them and adds nothing to any score; the formula calls them g.
- **very light (6 GeV), average width, pT spread over several particles, low pT** — 0.3% of jets, neuron 0.00. Mostly gluons (57%) with 28% tops, 0.31% of jets: mass 6.24 GeV, width 0.0045, soft (sum pT 572 GeV), with 89.24% of the pT at 0.05-0.1 from the axis and almost none in the core. mass < 21.6 and centroid_offset > 0.0322 passes for all and subtracts -48.076 here, far more than width < 0.00509 could add. The neuron is 0 for all of them and adds nothing to any score; the formula calls them g.
- **very light (5 GeV), wide, pT spread over several particles, low pT** — 0.0% of jets, neuron 0.00. Mostly tops (62%) with 19% gluons and 19% quarks, a tiny 0.04% of jets: mass 5.11 GeV but width 0.0106, soft (sum pT 513 GeV), with no pT inside 0.05 of the axis (64.51% at 0.05-0.1). mass < 21.6 and centroid_offset > 0.0322 passes for all (-110.781) and C2 < 0.0264 and centroid_offset > 0.0157 adds another -8.882, while width < 0.00509 never passes. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.

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

- **medium-mass (51 GeV), average width, pT spread over several particles** — 22.6% of jets, neuron 0.23. Mostly W (52%) with 27% Z and 10% tops, 22.57% of jets: mass 51.3 GeV, width 0.0054, average pT, with the pT at 0.025-0.1 from the axis (50.48% at 0.05-0.1). width < 0.0141 (+1.929), girth > 0.0325 and tau21 < 0.245 and z_dr_0p2_0p4 < 0.221 add, but e2_sq < 0.00854 (-1.962), width < 0.00761 and e2 > 0.0242 (88.5%, almost only here) and width < 0.0068 take it back below the -2.42 intercept for most jets. The neuron stays low (mean 0.235, on for 21.4%), a slight pull down on the W score; the formula calls them W.
- **very light (6 GeV), very narrow, pT spread over several particles** — 22.1% of jets, neuron 0.00. Mostly quarks (45%) with 32% gluons, 22.07% of jets: mass 5.92 GeV, width 0.0003, harder than average (sum pT 801 GeV), with 85.34% of the pT inside 0.025. mass < 36.2 (-13.263) dominates, joined by width < 0.0068, e2_sq < 0.00854 and mass_over_sum_pt < 0.0676, far more than e2 < 0.0245 (+7.151), e2 < 0.0424 and width < 0.0141 add; girth > 0.0325 passes for only 6.8%. The neuron is 0 for all of them and adds nothing to any score; the formula calls them q.
- **medium-mass (61 GeV), wide, pT spread over several particles** — 17.8% of jets, neuron 1.41. A Z/top mixture (Z 43%, tops 35%, gluons 11%), 17.85% of jets: mass 60.8 GeV, width 0.0099, softer than average (sum pT 650 GeV), with 59.13% of the pT at 0.05-0.1. These jets escape the lightness and narrowness penalties (width < 0.0068 passes for 6.6%, mass_over_sum_pt < 0.0676 almost never), so girth > 0.0325 (+2.565), tau21 < 0.245 and z_dr_0p2_0p4 < 0.221 and width < 0.0141 lift the neuron over the intercept. It sits at 1.406 (on for 84.9%), its largest group mean, lowering the W score (-0.967) and the Z score and raising the q score slightly; the formula calls them Z.
- **very light (14 GeV), very narrow, pT spread over several particles** — 9.5% of jets, neuron 0.00. Mostly gluons (40%) with 38% quarks, 9.46% of jets: mass 13.6 GeV, width 0.0007, average pT, with 68.6% of the pT inside 0.025. mass < 36.2 (-9.879), e2_sq < 0.00854, width < 0.0068 (-4.512) and mass_over_sum_pt < 0.0676 outweigh e2 < 0.0245 (+5.327), width < 0.0141 and e2 < 0.0424. The neuron is essentially 0 (mean 0.001) and adds nothing to the scores; the formula calls them g.
- **light (42 GeV), narrow, pT spread over several particles** — 8.8% of jets, neuron 0.34. Mostly W (38%) with 28% Z, 16% quarks and 11% gluons, 8.79% of jets: mass 42.2 GeV, width 0.0031, hard (sum pT 817 GeV, leading particle 328.86 GeV), with 58.39% of the pT at 0.025-0.05. e2 < 0.0245 (+2.834), width < 0.0141, e2 < 0.0424 and z_dr_0p1_0p2 < 0.323 add, against e2_sq < 0.00854 (-3.267), width < 0.0068 (-2.72) and mass_over_sum_pt < 0.0676; mass < 36.2 passes for only 32.2%. The neuron stays low (mean 0.338, on for 30.4%), a slight pull down on the W score; the formula calls them W.
- **medium-mass (72 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 0.05. Mostly tops (81%), 7.79% of jets: mass 71.7 GeV, width 0.0237, very soft (sum pT 501 GeV), with 43.29% of the pT at 0.1-0.15. Too wide for width < 0.0141, so girth > 0.0325 (+5.163) must beat LHA > 0.341 (always, -3.619) and the -2.42 intercept, and tau21 < 0.243 and girth2_top5 > 0.00856 and mass > 80.4 usually tip it below zero. The neuron stays near 0 (mean 0.049); the formula calls them t.
- **light (23 GeV), narrow, pT spread over several particles** — 6.4% of jets, neuron 0.00. Mostly gluons (41%) with 26% quarks, 14% W and 11% Z, 6.43% of jets: mass 23.2 GeV, width 0.002, softer than average (sum pT 643 GeV), with 44.09% of the pT at 0.025-0.05. mass < 36.2 (-5.685), e2_sq < 0.00854, width < 0.0068 and mass_over_sum_pt < 0.0676 outweigh e2 < 0.0245 (+2.702), width < 0.0141 and e2 < 0.0424. The neuron is 0 for all of them and adds nothing to any score; the formula calls them g.
- **heavy (100 GeV), very wide, pT spread over several particles** — 3.4% of jets, neuron 0.08. Mostly tops (83%), 3.42% of jets: mass 99.9 GeV, width 0.0251, near-average pT (sum pT 666 GeV), with the pT spread over 0.05-0.3 from the axis. mass > 80.4 always passes (-9.243) and with LHA > 0.341 outweighs girth > 0.0325 (+5.146) and mass > 80.4 and z_dr_0p2_0p4 < 0.211. The neuron stays near 0 (mean 0.077) and hardly touches the scores; the formula calls them t.
- **very light (13 GeV), very narrow, leading particle 45% of pT, high pT** — 0.9% of jets, neuron 0.06. Mostly gluons (51%) with 37% quarks, 0.9% of jets: mass 13.1 GeV, width 0.0002, very hard (sum pT 1241 GeV, leading particle 560.03 GeV), with 93.52% of the pT inside 0.025. width < 0.0059 and log_sum_pt > 6.9 passes for all (-9.927) and cancels log_sum_pt > 6.9 (+7.026), both almost unique to this group, so mass < 36.2 (-10.61) and the other narrowness penalties decide against e2 < 0.0245. The neuron stays near 0 (mean 0.063); the formula calls them g.
- **heavy (127 GeV), very wide, pT spread over several particles** — 0.7% of jets, neuron 0.11. Mostly tops (53%) with 34% gluons, 0.72% of jets: mass 126.5 GeV (the heaviest group), width 0.0269, harder than average (sum pT 807 GeV), with the pT spread over 0.05-0.3. mass > 80.4 always passes and subtracts -21.899 here, with LHA > 0.341 adding to it, while mass > 80.4 and z_dr_0p2_0p4 < 0.211 (+6.164) and girth > 0.0325 give back only part. The neuron stays near 0 (mean 0.114); the formula calls them t.

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

- **light (36 GeV), narrow, pT spread over several particles** — 91.0% of jets, neuron 0.00. Almost every jet, 91.05% of them, an even mixture of all five classes (W 22%, Z 22%, gluons 21%, quarks 21%, tops 14%): mass 35.9 GeV, width 0.0044 and average pT. girth2 > 0.0188, the only term that can lift this neuron, passes for just 1.1%, so nothing offsets the -0.931 intercept (e2 > 0.0634 passes for 1.4%). The neuron is 0 for all of them and adds nothing to any score; the formula's decisions are spread over all classes (W 26%, g 22%).
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 1.7% of jets, neuron 0.08. Mostly tops (89%), 1.73% of jets: mass 83.3 GeV, width 0.0242, soft (sum pT 548 GeV), with 33.9% of the pT at 0.1-0.15 from the axis. All three tests pass: girth2 > 0.0188 (+1.398) and girth2 > 0.0188 and pt_7 > 15.6 (+0.496) against e2 > 0.0634 (-1.332), which with the -0.931 intercept leaves the sum just below zero. The neuron is on for only 20.9% (mean 0.077), a tiny pull down on the t score; the formula calls them t.
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 1.5% of jets, neuron 0.27. Mostly tops (91%), 1.51% of jets: mass 87.3 GeV, width 0.0274, soft (sum pT 537 GeV), with the pT spread over 0.1-0.2 from the axis. girth2 > 0.0188 (+2.221) and girth2 > 0.0188 and pt_7 > 15.6 (+0.798) slightly outweigh e2 > 0.0634 (-1.974) and the intercept. The neuron is on for 52.6% (mean 0.271), a small pull down on the t score (-0.136); the formula calls them t.
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.00. Mostly tops (82%) with 14% gluons, 1.43% of jets: mass 77.3 GeV, width 0.0202, soft (sum pT 555 GeV), with 40.77% of the pT at 0.1-0.15. e2 > 0.0634 always passes (-0.782) while girth2 > 0.0188 passes for 85.1% and adds only +0.405 here, so the sum stays below zero. The neuron is 0 for all of them and adds nothing to any score; the formula calls them t.
- **heavy (90 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.95. Mostly tops (86%), 1.35% of jets: mass 90.4 GeV, width 0.0309, soft (sum pT 525 GeV), with 33.57% of the pT at 0.15-0.2. girth2 > 0.0188 (+3.135) and girth2 > 0.0188 and pt_7 > 15.6 (+1.16) outweigh e2 > 0.0634 (-2.427) and the intercept. The neuron sits at 0.954 (on for 89.3%), lowering the t score (-0.477); the formula calls them t.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 1.2% of jets, neuron 0.32. Mostly tops (80%) with 14% gluons, 1.2% of jets: mass 79.9 GeV, width 0.023, soft (sum pT 567 GeV), with 43.02% of the pT at 0.1-0.15. girth2 > 0.0188 (+1.087) and its pt_7 > 15.6 partner pass, while e2 > 0.0634 passes for only 63.9%, so the sum edges above zero for about half. The neuron is on for 56.3% (mean 0.324), a small pull down on the t score; the formula calls them t.
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 1.97. Mostly tops (78%) with 17% gluons, 0.8% of jets: mass 90.9 GeV, width 0.0354, soft (sum pT 498 GeV), with 33.18% of the pT at 0.2-0.3 from the axis. girth2 > 0.0188 (+4.291) and girth2 > 0.0188 and pt_7 > 15.6 (+1.48) clearly outweigh e2 > 0.0634 (-2.869). The neuron sits at 1.969, lowering the t score (-0.984); the formula calls them t.
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 0.5% of jets, neuron 1.73. Mostly tops (67%) with 24% gluons, 0.55% of jets: mass 84.2 GeV, width 0.0288, soft (sum pT 537 GeV), with the pT at 0.1-0.2 from the axis. girth2 > 0.0188 (+2.587) and girth2 > 0.0188 and pt_7 > 15.6 (+0.958) pass for all, while e2 > 0.0634 takes only -0.882 here. The neuron sits at 1.731, lowering the t score (-0.866); the formula calls them t.
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 3.70. Mostly tops (59%) with 31% gluons, 0.29% of jets: mass 95.2 GeV, width 0.0422, soft (sum pT 483 GeV), with 45.09% of the pT at 0.2-0.3 from the axis. girth2 > 0.0188 (+6.048) and girth2 > 0.0188 and pt_7 > 15.6 (+2.033) outweigh e2 > 0.0634 (-3.45). The neuron sits at 3.7, lowering the t score (-1.85); the formula calls them t.
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.1% of jets, neuron 6.05. Mostly tops (57%) with 30% gluons and 13% quarks, only 0.08% of jets: mass 94.6 GeV, width 0.0525 (the widest group), soft and evenly shared (leading particle 90.76 GeV), with 58.66% of the pT at 0.2-0.3. girth2 > 0.0188 (+8.726) and girth2 > 0.0188 and pt_7 > 15.6 (+3.009) far outweigh e2 > 0.0634 (-4.759). The neuron reaches 6.051, its largest group mean, lowering the t score (-3.025); the formula calls them t.
