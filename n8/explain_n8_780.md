# What each part of the 780-term formula does (8 particles)

*tuned on the network's predictions (from 60 if-statements per neuron, pruned)*. Validation accuracy 65.16%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet gets a value on 16 scales (each a clipped sum of if-statements on jet-shape quantities), and each class score adds some scales and subtracts others; the formula picks the largest score. Tops are recognised by size: the t score mostly subtracts compactness (neuron 13, -49%; tops sit far below every other type on it) and adds overall size (neuron 10, +26%), so it closely follows girth, width and e2. Quarks and gluons both sit high on the narrow single-core scale (neuron 9), the largest input of the q score (+50%) and a large input of the g score (+24%); they are split by how the pT is shared: the g score adds gluon-likeness (neuron 2: a light jet whose 8th-hardest particle is still hard) and a moderately-wide-and-heavy scale on which quarks sit lowest (neuron 1), and subtracts quark-likeness (neuron 5: pT held by the few hardest particles), while the q score subtracts overall size (neuron 10) and the clean two-prong scale (neuron 4). W and Z jets sit high on the two-prong scales (neuron 7 feeds both boson scores, neurons 11 and 0 only the W score), and both boson scores subtract broad off-centre spread and wide-angle radiation (neurons 6 and 3), on which tops and gluons sit much higher. W and Z are split by width and mass: Z jets sit highest on the Z-sized scale (neuron 14), which the Z score adds (+6%) and the W score subtracts (-8%), and the 'slightly wider than a W' scale (neuron 15) takes more out of the W score (-8%) than out of the Z score (-2%).

## The 5 class scores

### score g: gluon: pT spread over many particles

High for light, fairly narrow jets whose pT is spread over many particles (it rises with z_7 and z_6 and falls with the pT share of the 5 hardest particles); it averages 2.00 for g jets, 1.14 for t and 1.11 for q, and is near zero for W and Z (AUC 0.81 for gluons against the rest).

Adds gluon-likeness (neuron 2, +32%), the narrow single-core scale (neuron 9, +24%) and the moderately-wide-and-heavy scale (neuron 1, +15%), plus a little broad spread (neuron 6, +7%); subtracts quark-likeness (neuron 5, -14%) and, less, the compact two-prong scale (neuron 0, -5%) and neuron 4 (-3%).

*computed:* largest for g (2.00), then t (1.14), then q (1.11), then W (0.11), then Z (0.03); it separates g jets from the rest best (AUC 0.81: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets (it falls with lam1, girth, width and mass, rank correlations about -0.64); it averages 1.94 for q jets and 1.31 for g, with tops lower (0.24) and W and Z near or below zero (AUC 0.85 for quarks). Gluons are its main confusion.

Mostly adds the narrow single-core scale (neuron 9, +50%); subtracts overall size (neuron 10, -18%) and the clean two-prong scale (neuron 4, -12%); adds some broad spread (neuron 6, +11%) and a little quark-likeness (neuron 5, +5%) and narrow core (neuron 8, +2%).

*computed:* largest for q (1.94), then g (1.31), then t (0.24), then W (0.08), then Z (-0.14); it separates q jets from the rest best (AUC 0.85: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets that are not wider than a W; it averages 2.43 for W jets and 0.86 for Z, is negative for q (-0.18) and g (-0.54) and strongly negative for t (-2.54) (AUC 0.89).

Adds the compact-centred scale (neuron 11, +25%, its largest input), the two-prong scales (neurons 7 and 0) and compactness (neuron 13); subtracts broad off-centre spread (neuron 6, -14%), wide-angle radiation (neuron 3, -10%), the Z-sized and 'wider than a W' scales (neurons 14 and 15, -8% each) and a little of neurons 8 and 9.

*computed:* largest for W (2.43), then Z (0.86), then q (-0.18), then g (-0.54), then t (-2.54); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: two-prong jet, wider than a W

High for two-prong jets of boson mass with a Z-sized width; it averages 2.23 for Z jets and 1.42 for W (W is its main confusion), is negative for q (-0.09) and g (-0.35) and strongly negative for t (-2.09) (AUC 0.86).

Adds the two-prong W/Z scale (neuron 7, +29%, its largest input), compactness (neuron 13, +9%), the clean two-prong scale (neuron 4, +7%), the Z-sized scale (neuron 14, +6%) and neuron 1 (+5%); subtracts broad off-centre spread (neuron 6, -22%), wide-angle radiation (neuron 3, -15%) and a little of neurons 9 and 15.

*computed:* largest for Z (2.23), then W (1.42), then q (-0.09), then g (-0.35), then t (-2.09); it separates Z jets from the rest best (AUC 0.86: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and e2 with rank correlations of 0.91-0.924. It averages 2.97 for t jets, 0.49 for Z and 0.25 for W, is near zero for g and negative for q (-1.29) (AUC 0.91).

Subtracts compactness (neuron 13, -49%) and adds overall size (neuron 10, +26%); also subtracts quark-likeness (neuron 5, -12%) and adds smaller amounts of the clean two-prong scale (neuron 4, +7%) and neuron 8 (+3%).

*computed:* largest for t (2.97), then Z (0.49), then W (0.25), then g (0.06), then q (-1.29); it separates t jets from the rest best (AUC 0.91: large for t)

## The 16 neurons (most important first)

### neuron 1: moderately wide, heavy, hard jet (major)

- **What it measures:** Pushed up for small e2_sq (< 0.00817, its strongest term), for high total pT (log of total pT > 6.38 and > 6.57) and a hard 8th particle (pT_7 > 34.5 GeV), and pushed down for narrow jets (width < 0.00868), for a soft 8th particle (z_7 < 0.0556) and for light jets (mass < 53.3 GeV); it rises with mass, lam1 and width (rank correlations 0.536, 0.514, 0.513). Tops (1.32) and Z jets (1.17) sit highest, gluons in the middle (0.89), then W (0.69), and quarks lowest (0.40).
- *computed — its value:* largest for t (1.32), then Z (1.17), then g (0.89), then W (0.69), then q (0.40); it separates q jets from the rest best (AUC 0.31: small for q)
- **How the class scores use it:** Its clearest job is to separate gluons from the quarks that sit lowest on it, so it raises the g score (+15%); it also raises the Z score (+5%), since Z jets sit above W jets on it. It does not (or hardly) enter the q, W or t scores; the tops that sit highest on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+15%), Z (+5%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.530
if e2_sq < 0.0082: z += 450 × (0.0082 − e2_sq)
if width < 0.0087: z += -365 × (0.0087 − width)
if z_7 < 0.056: z += -76.60 × (0.056 − z_7)
if log_sum_pt > 6.38: z += 3.78 × (log_sum_pt − 6.38)
if lam1 < 0.0084 and lam2 < 0.00054: z += -352541 × (0.0084 − lam1) × (0.00054 − lam2)
if LHA < 0.347: z += -5.59 × (0.347 − LHA)
if girth < 0.076: z += 21.65 × (0.076 − girth)
if log_sum_pt > 6.57: z += 6.15 × (log_sum_pt − 6.57)
if pt_7 > 34.53: z += 0.118 × (pt_7 − 34.53)
if log_sum_pt > 6.57 and lam2 < 0.0011: z += 4997 × (log_sum_pt − 6.57) × (0.0011 − lam2)
if mass < 53.33: z += -0.023 × (53.33 − mass)
if mass_over_sum_pt > 0.055: z += 19.93 × (mass_over_sum_pt − 0.055)
if log_sum_pt > 6.38 and centroid_offset < 0.024: z += -132 × (log_sum_pt − 6.38) × (0.024 − centroid_offset)
if e2_sq < 0.0082 and planar_flow < 0.084: z += -6114 × (0.0082 − e2_sq) × (0.084 − planar_flow)
if e2 < 0.036 and eccentricity > 0.978: z += 8623 × (0.036 − e2) × (eccentricity − 0.978)
if pt_7 > 34.53 and mass < 91.19: z += -0.0012 × (pt_7 − 34.53) × (91.19 − mass)
if log_sum_pt > 6.57 and girth2_top3 < 0.0068: z += -641 × (log_sum_pt − 6.57) × (0.0068 − girth2_top3)
if log_sum_pt > 6.38 and max_dr < 0.198: z += 13.31 × (log_sum_pt − 6.38) × (0.198 − max_dr)
if z_7 < 0.056 and girth2_top3 < 0.0079: z += 4326 × (0.056 − z_7) × (0.0079 − girth2_top3)
if C2 < 0.042: z += -13.09 × (0.042 − C2)
if z_7 < 0.043: z += -43.89 × (0.043 − z_7)
if e2 > 0.032: z += -26.69 × (e2 − 0.032)
if tau21 < 0.198: z += 4.31 × (0.198 − tau21)
if LHA < 0.347 and planar_flow < 0.084: z += -120 × (0.347 − LHA) × (0.084 − planar_flow)
if z_7 < 0.056 and lam2 < 0.0034: z += 4354 × (0.056 − z_7) × (0.0034 − lam2)
if lam1 < 0.0084: z += -25.67 × (0.0084 − lam1)
if planar_flow < 0.084: z += 5.07 × (0.084 − planar_flow)
if log_sum_pt > 6.57 and max_dr < 0.290: z += 5.66 × (log_sum_pt − 6.57) × (0.290 − max_dr)
if lam1 < 0.0084 and z_7 > 0.040: z += -1585 × (0.0084 − lam1) × (z_7 − 0.040)
if tau21 < 0.198 and pt_6 < 50.25: z += -0.234 × (0.198 − tau21) × (50.25 − pt_6)
if max_dr < 0.047: z += -16.84 × (0.047 − max_dr)
if lam1 < 0.0084 and centroid_offset > 0.021: z += 11183 × (0.0084 − lam1) × (centroid_offset − 0.021)
if pt_7 > 34.53 and max_dr < 0.081: z += 0.912 × (pt_7 − 34.53) × (0.081 − max_dr)
if e2 < 0.036: z += -5.32 × (0.036 − e2)
if width < 0.0087 and planar_flow < 0.084: z += -1220 × (0.0087 − width) × (0.084 − planar_flow)
if lam1 < 0.0084 and n_pt_above_50 > 6.00: z += -35.78 × (0.0084 − lam1) × (n_pt_above_50 − 6.00)
if z_7 < 0.056 and C2 < 0.042: z += 178 × (0.056 − z_7) × (0.042 − C2)
if pt_7 > 53.44: z += -0.116 × (pt_7 − 53.44)
if log_sum_pt > 6.38 and z_dr_0_0p05 < 0.901: z += 0.633 × (log_sum_pt − 6.38) × (0.901 − z_dr_0_0p05)
if pt_7 > 34.53 and pt_6 < 52.91: z += -0.0022 × (pt_7 − 34.53) × (52.91 − pt_6)
if log_sum_pt > 6.57 and D2 < 1.23: z += -1.44 × (log_sum_pt − 6.57) × (1.23 − D2)
if e2_sq < 0.0082 and n_pt_above_50 > 6.00: z += 18.36 × (0.0082 − e2_sq) × (n_pt_above_50 − 6.00)
if log_sum_pt > 6.38 and n_pt_above_50 > 7.00: z += 1.07 × (log_sum_pt − 6.38) × (n_pt_above_50 − 7.00)
if log_sum_pt > 6.57 and n_pt_above_50 > 7.00: z += -2.65 × (log_sum_pt − 6.57) × (n_pt_above_50 − 7.00)
if pt_7 > 34.53 and centroid_offset > 0.014: z += 0.855 × (pt_7 − 34.53) × (centroid_offset − 0.014)
if log_sum_pt > 6.57 and planar_flow < 0.033: z += 45.06 × (log_sum_pt − 6.57) × (0.033 − planar_flow)
if z_7 < 0.056 and dr01 > 0.056: z += -60.83 × (0.056 − z_7) × (dr01 − 0.056)
if pt_7 > 53.44 and mass_top3 < 32.62: z += 0.00088 × (pt_7 − 53.44) × (32.62 − mass_top3)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (71 GeV), very wide, pT spread over several particles, low pT** — 21.3% of jets, neuron 1.42. Mostly tops (68%, with 13% gluons and 12% Z), 21.33% of jets: mass 71.1 GeV, width 0.0186, soft (sum pT 565 GeV), with the pT spread over 0.05-0.2 from the axis. Being heavy for their pT, they all pass mass_over_sum_pt > 0.0549 (+1.443), helped by pt_7 > 34.5 (50%, +0.345), tau21 < 0.198 (49.3%) and log_sum_pt > 6.38 (41.6%); e2 > 0.0323 (97.9%, -0.826) and z_7 < 0.0556 (32.7%) take part back, and the small-e2 bonus e2_sq < 0.00817 passes for only 14.8%. The neuron sits at 1.419 (on for 94.3%), raising the g (+0.554) and Z (+0.177) scores and barely lowering the W score; the formula calls them t (82%).
- **light (44 GeV), average width, pT spread over several particles** — 14.5% of jets, neuron 0.48. Mostly W (39%) with 27% Z, 15% tops and 12% gluons, 14.54% of jets: mass 43.6 GeV, width 0.0055, somewhat soft (sum pT 625 GeV), with 55.83% of the pT at 0.05-0.1 from the axis. e2_sq < 0.00817 (+1.458) is almost cancelled by width < 0.00868 (98.2%, -1.168), while the smaller terms, log_sum_pt > 6.38 (65.9%, +0.381), pt_7 > 34.5 (56.5%) and mass_over_sum_pt > 0.0549 (94%) against e2_sq < 0.00817 and planar_flow < 0.0837 (51.2%, -0.41), LHA < 0.347, lam1 < 0.00838 and lam2 < 0.000537 and z_7 < 0.0556 (49%), roughly balance. The neuron is on for 53.6% (mean 0.484), raising the g score a little; the formula calls them W (56%).
- **very light (15 GeV), very narrow, pT spread over several particles, low pT** — 12.8% of jets, neuron 0.15. Mostly gluons (46%, with 23% quarks), 12.82% of jets: mass 14.8 GeV, width 0.0013, somewhat soft (sum pT 607 GeV), with 48.48% of the pT inside 0.025 and 39.58% at 0.025-0.05 from the axis. e2_sq < 0.00817 (+3.306) and girth < 0.0761 (+1.013) are cancelled by width < 0.00868 (-2.702), lam1 < 0.00838 and lam2 < 0.000537 (96.6%, -1.08), LHA < 0.347 (-0.928) and mass < 53.3 (-0.892), and the hard-jet bonuses pass for only part of them. The neuron is on for 17.4% (mean 0.155) and barely moves the scores; the formula calls them g (71%).
- **very light (12 GeV), very narrow, pT spread over several particles, high pT** — 9.8% of jets, neuron 0.38. Mostly quarks (50%, with 22% gluons and 15% W), 9.81% of jets: mass 12.1 GeV, width 0.0005, hard (sum pT 845 GeV, leading particle 321.28 GeV), 8th particle 29.23 GeV, with 83.48% of the pT inside 0.025 of the axis. e2_sq < 0.00817 (+3.513), log_sum_pt > 6.38 (99.7%, +1.355), girth < 0.0761 (+1.313) and log_sum_pt > 6.57 (98%, +1.016) are cancelled by width < 0.00868 (-2.982), z_7 < 0.0556 (all, -1.616), lam1 < 0.00838 and lam2 < 0.000537 (-1.387), LHA < 0.347 (-1.231) and mass < 53.3 (-0.956). The neuron is on for 37.9% (mean 0.382), raising the g score slightly; the formula calls them q (74%).
- **medium-mass (68 GeV), average width, leading particle 42% of pT, high pT** — 9.7% of jets, neuron 1.20. Mostly Z (45%, with 38% W), 9.72% of jets: mass 67.7 GeV, width 0.0064, hard (sum pT 870 GeV, leading particle 367.96 GeV) with a soft 8th particle (26.53 GeV), pT mostly at 0.025-0.1 from the axis. The hard-jet tests log_sum_pt > 6.38 (+1.462), log_sum_pt > 6.57 (99%, +1.184) and log_sum_pt > 6.57 and lam2 < 0.00113 (97.6%, +1.005) with e2_sq < 0.00817 (92.2%, +1.001) outweigh z_7 < 0.0556 (all, -1.904), width < 0.00868 (94%, -0.939), log_sum_pt > 6.38 and centroid_offset < 0.0236 (94.8%, -0.738) and z_7 < 0.043 (91.1%). The neuron sits at 1.195 (on for 77.6%), raising the g (+0.467) and Z (+0.149) scores; the formula splits them W 47% / Z 44%.
- **very light (13 GeV), very narrow, leading particle 48% of pT, high pT** — 8.0% of jets, neuron 0.54. Mostly quarks (68%, with 13% gluons), 7.98% of jets: mass 12.7 GeV, width 0.0003, very hard (sum pT 991 GeV, leading particle 478.84 GeV) with a soft 8th particle (17.83 GeV), 94.59% of the pT inside 0.025 of the axis. e2_sq < 0.00817 (+3.539) and the hard-jet bonuses log_sum_pt > 6.57 (+1.974), log_sum_pt > 6.38 (+1.948), log_sum_pt > 6.57 and lam2 < 0.00113 (+1.77) and girth < 0.0761 (+1.444) are met by width < 0.00868 (-3.043), z_7 < 0.0556 (-2.883), lam1 < 0.00838 and lam2 < 0.000537 (-1.455) and LHA < 0.347 (-1.408). The neuron is on for 37.7% (mean 0.538), raising the g score a little; the formula calls them q (87%).
- **medium-mass (58 GeV), average width, pT spread over several particles** — 7.8% of jets, neuron 2.16. A Z/W mixture (Z 41%, W 39%, tops 10%), 7.85% of jets: mass 57.7 GeV, width 0.0069, pT shared evenly (leading particle 177.5 GeV, 8th particle 50.28 GeV), with 61.59% of it at 0.05-0.1 from the axis. The hard 8th particle makes pt_7 > 34.5 pass for all (+1.853), joined by e2_sq < 0.00817 (89.5%, +0.872), log_sum_pt > 6.38 (89.5%, +0.747), mass_over_sum_pt > 0.0549 (+0.505) and tau21 < 0.198 (85.6%); width < 0.00868 (91.2%, -0.802), pt_7 > 34.5 and mass < 91.2 (97.6%, -0.635) and e2_sq < 0.00817 and planar_flow < 0.0837 (74.3%) take back part. The neuron is high (2.161, on for 91.4%), raising the g (+0.844) and Z (+0.27) scores; the formula splits them W 49% / Z 41%.
- **light (37 GeV), narrow, pT spread over several particles** — 7.0% of jets, neuron 1.10. Mostly W (39%) with 29% Z, 14% quarks and 12% gluons, 7.03% of jets: mass 36.8 GeV, width 0.0031, somewhat hard (sum pT 797 GeV), with 61.62% of the pT at 0.025-0.05 from the axis. The elongation test e2 < 0.0356 and eccentricity > 0.978 passes for all (+2.716), with e2_sq < 0.00817 (+2.576) and log_sum_pt > 6.38 (94.2%, +1.119), outweighing width < 0.00868 (99.6%, -2.043), e2_sq < 0.00817 and planar_flow < 0.0837 (all, -1.975), z_7 < 0.0556 (79.6%, -1.183) and LHA < 0.347 and planar_flow < 0.0837 (-0.883). The neuron sits at 1.099 (on for 71.1%), raising the g (+0.429) and Z scores; the formula calls them W (51%), with Z 27%.
- **very light (10 GeV), very narrow, pT spread over several particles** — 7.0% of jets, neuron 0.28. Mostly gluons (51%, with 29% quarks), 7.01% of jets: mass 10.4 GeV, width 0.0005, pT shared among many particles (8th particle 48.88 GeV), with 75.91% of it inside 0.025 of the axis. e2_sq < 0.00817 (+3.518), pt_7 > 34.5 (all, +1.688) and girth < 0.0761 (+1.252) are cancelled by width < 0.00868 (-2.967), pt_7 > 34.5 and mass < 91.2 (-1.389), lam1 < 0.00838 and lam2 < 0.000537 (99.5%, -1.356), LHA < 0.347 (-1.153) and mass < 53.3 (-0.994). The neuron is on for 26.2% (mean 0.279) and barely moves the scores; the formula calls them g (78%).
- **very light (12 GeV), very narrow, pT spread over several particles, high pT** — 1.9% of jets, neuron 2.19. Mostly gluons (60%, with 25% quarks), 1.93% of jets: mass 12.2 GeV, width 0.0004, hard (sum pT 956 GeV) with pT shared by many particles (8th particle 61.44 GeV), 84.42% of it inside 0.025 of the axis. Both the hard-8th-particle test pt_7 > 34.5 (+3.167) and the hard-jet tests log_sum_pt > 6.38 (+1.784), log_sum_pt > 6.57 (99%, +1.708) and log_sum_pt > 6.57 and lam2 < 0.00113 (+1.537) pass, with e2_sq < 0.00817 (+3.528) and girth < 0.0761 (+1.339), outweighing width < 0.00868 (-3.011), pt_7 > 34.5 and mass < 91.2 (99.9%, -2.527) and lam1 < 0.00838 and lam2 < 0.000537 (-1.447). The neuron is high (2.19, on for 87%), raising the g (+0.855) and Z (+0.274) scores; the formula calls them g (82%).

### neuron 2: gluon-likeness: light jet, hard 8th particle (major)

- **What it measures:** Pushed up for jets lighter than 69.6 GeV (but pushed down again below 36.2 GeV), for total pT below 788 GeV and when the 8th-hardest particle is still hard (its pT_7 terms favour pT_7 above 30.5 GeV, most of all not below 53.4 GeV), and pushed down for large angularity (LHA > 0.112, its strongest term); overall it falls with mass (rank correlation -0.723). Gluons sit highest (3.11), then quarks (2.15), with W (1.35), top (1.10) and Z (1.10) jets lower.
- *computed — its value:* largest for g (3.11), then q (2.15), then W (1.35), then t (1.10), then Z (1.10); it separates g jets from the rest best (AUC 0.78: large for g)
- **How the class scores use it:** Gluons sit highest on it, so it is direct evidence for a gluon: it raises the g score, where it is the largest input (+32%). It does not (or hardly) enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+32%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.94
if LHA > 0.112: z += -10.29 × (LHA − 0.112)
if mass < 69.61: z += 0.035 × (69.61 − mass)
if pt_7 < 53.44: z += -0.058 × (53.44 − pt_7)
if sum_pt < 788: z += 0.0095 × (788 − sum_pt)
if mass < 36.23: z += -0.090 × (36.23 − mass)
if pt_7 < 43.50: z += 0.070 × (43.50 − pt_7)
if sum_pt_top5 < 687: z += -0.005 × (687 − sum_pt_top5)
if pt_7 > 30.48: z += 0.074 × (pt_7 − 30.48)
if mass < 36.23 and lam2 < 0.0011: z += 34.87 × (36.23 − mass) × (0.0011 − lam2)
if pt_6 < 56.53: z += -0.019 × (56.53 − pt_6)
if lam1 < 0.006: z += 124 × (0.006 − lam1)
if mass < 69.61 and z_7 < 0.068: z += -0.396 × (69.61 − mass) × (0.068 − z_7)
if lam1 < 0.0034: z += 201 × (0.0034 − lam1)
if log_sum_pt < 6.46: z += 3.02 × (6.46 − log_sum_pt)
if log_sum_pt > 6.27: z += -0.677 × (log_sum_pt − 6.27)
if pt_7 > 30.48 and C2 < 0.051: z += -0.886 × (pt_7 − 30.48) × (0.051 − C2)
if planar_flow < 0.695: z += -0.307 × (0.695 − planar_flow)
if log_sum_pt < 6.64: z += -0.812 × (6.64 − log_sum_pt)
if log_sum_pt < 6.57: z += 0.997 × (6.57 − log_sum_pt)
if lam1 < 0.006 and max_dr > 0.081: z += -2544 × (0.006 − lam1) × (max_dr − 0.081)
if max_dr < 0.160: z += -1.96 × (0.160 − max_dr)
if width < 0.00017 and dr_7 < 0.223: z += -37592 × (0.00017 − width) × (0.223 − dr_7)
if log_sum_pt > 6.84: z += 13.65 × (log_sum_pt − 6.84)
if pt_7 > 30.48 and max_dr > 0.093: z += -0.377 × (pt_7 − 30.48) × (max_dr − 0.093)
if width < 0.00017: z += 6245 × (0.00017 − width)
if log_sum_pt > 6.80: z += -4.63 × (log_sum_pt − 6.80)
if lam1 < 0.0034 and centroid_offset < 0.0068: z += -24480 × (0.0034 − lam1) × (0.0068 − centroid_offset)
if pt_7 > 30.48 and z_dr_0p05_0p1 < 0.751: z += -0.015 × (pt_7 − 30.48) × (0.751 − z_dr_0p05_0p1)
if z_7 < 0.028: z += -36.07 × (0.028 − z_7)
if mass < 8.38: z += 0.078 × (8.38 − mass)
if width < 9.1e-05: z += -10085 × (9.1e-05 − width)
if z_7 < 0.017: z += -158 × (0.017 − z_7)
if mass < 69.61 and centroid_offset > 0.011: z += -0.181 × (69.61 − mass) × (centroid_offset − 0.011)
if pt_7 > 30.48 and dr_0 > 0.031: z += -0.191 × (pt_7 − 30.48) × (dr_0 − 0.031)
if z_7 > 0.046: z += -2.92 × (z_7 − 0.046)
if z_7 < 0.028 and width < 0.0075: z += -4066 × (0.028 − z_7) × (0.0075 − width)
if pt_7 < 15.55: z += 0.136 × (15.55 − pt_7)
if log_sum_pt > 6.90: z += -6.88 × (log_sum_pt − 6.90)
if pt_7 > 30.48 and centroid_offset > 0.013: z += -0.570 × (pt_7 − 30.48) × (centroid_offset − 0.013)
if mass < 8.38 and centroid_offset < 0.011: z += 8.21 × (8.38 − mass) × (0.011 − centroid_offset)
if mass < 69.61 and max_pair_mass > 13.05: z += -0.00069 × (69.61 − mass) × (max_pair_mass − 13.05)
if sum_pt_top5 > 840: z += -0.002 × (sum_pt_top5 − 840)
if pt_6 < 56.53 and m012 > 32.62: z += 0.00031 × (56.53 − pt_6) × (m012 − 32.62)
if mass < 36.23 and pt_4 < 55.66: z += 0.00022 × (36.23 − mass) × (55.66 − pt_4)
if girth2 < 4.8e-05: z += -15366 × (4.8e-05 − girth2)
if log_sum_pt > 6.84 and n_pt_above_50 > 5.00: z += -1.89 × (log_sum_pt − 6.84) × (n_pt_above_50 − 5.00)
if width < 9.1e-05 and pt_2 < 154: z += 72.92 × (9.1e-05 − width) × (154 − pt_2)
if mass < 36.23 and mean_phi < -0.0031: z += 0.351 × (36.23 − mass) × (-0.0031 − mean_phi)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (57 GeV), average width, leading particle 41% of pT** — 20.1% of jets, neuron 0.40. An even W/Z mixture (Z 36.32%, W 36.05%, tops 12%), 20.12% of jets: mass 57.5 GeV, width 0.0057, harder than average with a strong leading particle (342.4 GeV) and a soft 8th particle (26.27 GeV vs 34.66 on average), pT mostly 0.025-0.1 from the axis. pt_7 < 53.4 (-1.581) and LHA > 0.112 (99.8%, -1.534) are only partly offset by pt_7 < 43.5 (+1.207) and mass < 69.6 (79.7%, +0.518), and the soft-jet bonus sum_pt < 788 passes for only 39.1%, so the neuron stays low (0.404, on for 54.5%). It adds a little (+0.174) to the g score only; the formula splits them W 48% / Z 33%.
- **medium-mass (60 GeV), average width, pT spread over several particles** — 13.9% of jets, neuron 0.80. A W/Z mixture (Z 37%, W 36%, tops 13%, gluons 10%), 13.94% of jets: mass 59.8 GeV, width 0.007, with pT shared evenly (leading particle only 210.79 GeV, 8th particle 46.2 GeV) and 51.69% of it at 0.05-0.1 from the axis. The hard 8th particle makes pt_7 > 30.5 pass for all of them (+1.161), which with mass < 69.6 (82.3%) and sum_pt < 788 (68%) partly offsets LHA > 0.112 (-1.867), pt_7 < 53.4 and sum_pt_top5 < 687. The neuron sits at 0.798 (on for 74.2%) and adds +0.343 to the g score only; the formula splits them W 47% / Z 35%.
- **very light (9 GeV), very narrow, pT spread over several particles, high pT** — 13.6% of jets, neuron 2.25. Mostly quarks (53%, plus 22% gluons), 13.57% of jets: mass 8.6 GeV, width 0.0003, a hard leading particle (327.29 GeV) and 86.05% of the pT inside 0.025 of the axis. Being light, they collect mass < 69.6 (+2.158), mass < 36.2 and lam2 < 0.00113 (+1.062) and the lam1 < 0.00595 / lam1 < 0.00338 bonuses, which beat mass < 36.2 (-2.498) and pt_7 < 53.4 (-1.531); LHA > 0.112 passes for only 48.9% and costs little. The neuron sits at 2.247 (on for 99.9%) and adds +0.966 to the g score only; the formula calls them q (74%).
- **light (42 GeV), average width, pT spread over several particles, low pT** — 12.6% of jets, neuron 2.09. A mixture of W (30%), Z (28%), tops (20%) and gluons (14%), 12.57% of jets: mass 42.1 GeV, width 0.006, soft (sum pT 589 GeV), with pT mostly at 0.025-0.1 from the axis. Low total pT makes sum_pt < 788 pass for all (+1.901), which together with mass < 69.6 (+0.976) and pt_7 > 30.5 (88.7%) outweighs LHA > 0.112 (-1.749) and sum_pt_top5 < 687 (-1.149). The neuron sits at 2.091 (on for 99.9%) and adds +0.899 to the g score only; the formula splits them W 42% / Z 30% / t 16%.
- **very light (8 GeV), very narrow, pT spread over several particles** — 10.0% of jets, neuron 3.68. Mostly gluons (48%, with 32% quarks), 10.02% of jets: mass 8.5 GeV, width 0.0004, with pT shared among many particles (8th particle 48.03 GeV vs 34.66 on average) and 81.83% of it inside 0.025 of the axis. The light-jet pieces mass < 69.6 (+2.162) and mass < 36.2 and lam2 < 0.00113 (+1.064) cancel mass < 36.2 (-2.509), and what lifts this group is pt_7 > 30.5, passing for all (+1.296), with the lam1 bonuses; LHA > 0.112 passes for 59.6% but costs only -0.274. The neuron sits high at 3.677 and adds +1.58 to the g score; the formula calls them g (66%).
- **medium-mass (81 GeV), very wide, pT spread over several particles, low pT** — 9.2% of jets, neuron 0.31. Mostly tops (83%), 9.21% of jets: mass 81.5 GeV, width 0.0217, soft (sum pT 585 GeV), with most of the pT beyond 0.05 from the axis. LHA > 0.112 always passes and hits hardest here (-2.977), while mass < 69.6 passes for only 25.9%; sum_pt < 788 (+1.932) is the main support, partly cancelled by sum_pt_top5 < 687 (-1.125) and pt_7 < 53.4 (-1.055). The neuron stays low (0.309, on for 50.5%) and adds only +0.133 to the g score; the formula calls them t (97%).
- **very light (12 GeV), very narrow, pT spread over several particles, low pT** — 8.5% of jets, neuron 4.35. Mostly gluons (55%, with 20% quarks), 8.45% of jets: mass 12.0 GeV, width 0.0013, very soft (sum pT 541 GeV, leading particle 137.5 GeV), pT within 0.05 of the axis. Both the soft-jet bonus sum_pt < 788 (+2.359) and the light-jet bonus mass < 69.6 (+2.039) pass for all, with mass < 36.2 and lam2 < 0.00113 (+0.862) and log_sum_pt < 6.46 (+0.554); they outweigh mass < 36.2 (-2.194) and sum_pt_top5 < 687 (-1.359), while LHA > 0.112 costs only -0.756. The neuron reaches its highest value, 4.348, and adds +1.868 to the g score; the formula calls them g (92%).
- **light (50 GeV), very wide, pT spread over several particles, low pT** — 7.8% of jets, neuron 2.67. Mostly tops (57%) mixed with 25% gluons, 7.8% of jets: mass 50.0 GeV, width 0.0173, very soft (sum pT 426 GeV, leading particle 100.3 GeV) and spread out beyond 0.05 from the axis. The softness tests dominate: sum_pt < 788 (+3.45), log_sum_pt < 6.46 (+1.265) and log_sum_pt < 6.57 (+0.526) all pass, beating LHA > 0.112 (-2.599), sum_pt_top5 < 687 (-1.855) and pt_7 < 53.4 (-1.223). The neuron sits at 2.665 (on for 99.7%) and adds +1.145 to the g score; the formula calls them t (72%), with g 19%.
- **very light (12 GeV), very narrow, leading particle 53% of pT, high pT** — 3.3% of jets, neuron 0.35. Mostly quarks (79%), 3.29% of jets: mass 12.1 GeV, width 0.0002, hard (sum pT 979 GeV) with one dominant particle (516.82 GeV) and a very soft 8th particle (10.33 GeV), 97.07% of the pT inside 0.025 of the axis. The two pT_7 tests cancel (pt_7 < 53.4 -2.508, pt_7 < 43.5 +2.323) as do the mass tests (mass < 36.2 -2.2, mass < 69.6 +2.036), and the soft tail adds mass < 69.6 and z_7 < 0.0681 (-1.307) and z_7 < 0.0169 (96.3%, -0.993); pt_7 > 30.5 and sum_pt < 788 essentially never pass. The neuron is low (0.355, on for 54.5%) and adds +0.153 to the g score; the formula calls them q (98%).
- **light (22 GeV), very narrow, leading particle 42% of pT, high pT** — 1.0% of jets, neuron 1.62. Mostly gluons (55%, with 27% quarks), 1.02% of jets: mass 21.9 GeV, width 0.0008, very hard (sum pT 1239 GeV, leading particle 526.27 GeV), with 82.67% of the pT inside 0.025 of the axis. The high-pT test log_sum_pt > 6.84 passes for all and adds +3.731, more than log_sum_pt > 6.9 (-1.514) and log_sum_pt > 6.8 (-1.444) take away; mass < 36.2 (81.6%, -1.87) and mass < 69.6 (+1.757) roughly cancel. The neuron sits at 1.622 (on for 83.1%) and adds +0.697 to the g score; the formula calls them g (70%).

### neuron 6: broad, off-centre spread (major)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and width < 0.0132 both push it down) and pushed up by a large largest particle distance (max ΔR) and small e2 (< 0.0503); it follows width, girth2 and the offset of the pT centroid from the jet axis (rank correlations 0.541, 0.541, 0.511). Tops sit far highest (4.51), then gluons (1.63) and quarks (0.72), with Z (0.41) and W (0.10) jets lowest.
- *computed — its value:* largest for t (4.51), then g (1.63), then q (0.72), then Z (0.41), then W (0.10); it separates t jets from the rest best (AUC 0.83: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so it lowers the W score (-14%) and the Z score (-22%). It raises the g score (+7%) and the q score (+11%), since among non-top jets a broad spread is QCD-like rather than boson-like; it does not (or hardly) enter the t score, which takes its size information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+7%), q (+11%); lowers the score of W (-14%), Z (-22%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = 8.17
if girth2 < 0.0087: z += -1282 × (0.0087 − girth2)
if width < 0.013: z += -656 × (0.013 − width)
if mass_over_sum_pt > 0.0084: z += -89.18 × (mass_over_sum_pt − 0.0084)
z += 17.74 × max_dr
if e2 < 0.050: z += 84.12 × (0.050 − e2)
if lam1 < 0.0084: z += 298 × (0.0084 − lam1)
if centroid_offset > 0.018: z += -136 × (centroid_offset − 0.018)
if girth > 0.087: z += 95.23 × (girth − 0.087)
if mass < 49.67 and z_dr_0p05_0p1 < 0.751: z += 0.065 × (49.67 − mass) × (0.751 − z_dr_0p05_0p1)
if log_sum_pt < 6.70: z += -3.44 × (6.70 − log_sum_pt)
if lam1 < 0.0073: z += 186 × (0.0073 − lam1)
if lam2 < 0.00054: z += -1613 × (0.00054 − lam2)
if log_sum_pt < 6.57: z += 4.12 × (6.57 − log_sum_pt)
if centroid_offset > 0.0081 and lam2 < 0.0034: z += 18047 × (centroid_offset − 0.0081) × (0.0034 − lam2)
if D2 < 1.68: z += 0.814 × (1.68 − D2)
if log_sum_pt < 6.70 and pt_7 < 45.75: z += 0.186 × (6.70 − log_sum_pt) × (45.75 − pt_7)
if girth2_top5 > 0.0023: z += -74.46 × (girth2_top5 − 0.0023)
if e2 < 0.050 and n_dr_0p05_0p1 < 4.00: z += -4.08 × (0.050 − e2) × (4.00 − n_dr_0p05_0p1)
if centroid_offset > 0.0081: z += 29.10 × (centroid_offset − 0.0081)
if D2 < 1.68 and pt_4 < 90.62: z += -0.016 × (1.68 − D2) × (90.62 − pt_4)
if C2 > 0.011: z += -12.64 × (C2 − 0.011)
if LHA > 0.313 and eccentricity > 0.873: z += 258 × (LHA − 0.313) × (eccentricity − 0.873)
if girth2_top5 > 0.011: z += 140 × (girth2_top5 − 0.011)
if log_sum_pt < 6.70 and mean_eta2 > 0.0042: z += 390 × (6.70 − log_sum_pt) × (mean_eta2 − 0.0042)
if centroid_offset > 0.0081 and C2 < 0.095: z += 359 × (centroid_offset − 0.0081) × (0.095 − C2)
if centroid_offset > 0.0081 and planar_flow > 0.008: z += 55.50 × (centroid_offset − 0.0081) × (planar_flow − 0.008)
if lam2 > 0.0034: z += -951 × (lam2 − 0.0034)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.159: z += 624 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.159)
if log_sum_pt < 6.33: z += 3.93 × (6.33 − log_sum_pt)
if centroid_offset > 0.018 and mean_phi2 < 0.0089: z += 5451 × (centroid_offset − 0.018) × (0.0089 − mean_phi2)
if mass_over_sum_pt > 0.0084 and tau32 < 0.519: z += -20.20 × (mass_over_sum_pt − 0.0084) × (0.519 − tau32)
if C2 > 0.011 and pt_7 > 31.86: z += 1.42 × (C2 − 0.011) × (pt_7 − 31.86)
if log_sum_pt < 6.70 and z_7 < 0.049: z += 407 × (6.70 − log_sum_pt) × (0.049 − z_7)
if log_sum_pt < 6.70 and pt_6 < 36.81: z += 0.192 × (6.70 − log_sum_pt) × (36.81 − pt_6)
if lam2 > 0.00054: z += -203 × (lam2 − 0.00054)
if girth2_top5 > 0.0023 and z_dr_0p05_0p1 > 0.292: z += 102 × (girth2_top5 − 0.0023) × (z_dr_0p05_0p1 − 0.292)
if girth2 < 0.0036: z += 57.23 × (0.0036 − girth2)
if centroid_offset > 0.018 and pt_2 > 56.50: z += 0.496 × (centroid_offset − 0.018) × (pt_2 − 56.50)
if girth2_top5 > 0.011 and mean_eta > 0.013: z += -4757 × (girth2_top5 − 0.011) × (mean_eta − 0.013)
if z_7 > 0.062: z += -8.21 × (z_7 − 0.062)
if log_sum_pt < 6.70 and mean_phi > 0.0091: z += -36.69 × (6.70 − log_sum_pt) × (mean_phi − 0.0091)
if log_sum_pt < 6.33 and pt_6 > 27.58: z += -0.099 × (6.33 − log_sum_pt) × (pt_6 − 27.58)
if lam1 < 0.0073 and eta_7 > 0.0083: z += -690 × (0.0073 − lam1) × (eta_7 − 0.0083)
if lam2 < 0.00054 and mean_eta > 0.026: z += -197856 × (0.00054 − lam2) × (mean_eta − 0.026)
if log_sum_pt < 6.33 and z_7 < 0.071: z += 163 × (6.33 − log_sum_pt) × (0.071 − z_7)
if width < 0.013 and mean_eta > 0.026: z += 7782 × (0.013 − width) × (mean_eta − 0.026)
if centroid_offset > 0.0081 and mean_eta2 < 0.0011: z += -6312 × (centroid_offset − 0.0081) × (0.0011 − mean_eta2)
if centroid_offset > 0.018 and pt_5 < 24.58: z += 22.22 × (centroid_offset − 0.018) × (24.58 − pt_5)
if centroid_offset > 0.050 and n_dr_0p05_0p1 > 6.00: z += -44.31 × (centroid_offset − 0.050) × (n_dr_0p05_0p1 − 6.00)
if log_sum_pt < 6.70 and z_4 < 0.037: z += 2064 × (6.70 − log_sum_pt) × (0.037 − z_4)
if e2_sq < 0.0082 and mass_top3 > 50.35: z += 29.03 × (0.0082 − e2_sq) × (mass_top3 − 50.35)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles** — 29.4% of jets, neuron 0.10. Mostly quarks (45%, with 34% gluons), 29.45% of jets: mass 8.2 GeV, width 0.0003, with 87.32% of the pT inside 0.025 of the axis. The narrowness switches girth2 < 0.00868 (-10.773) and width < 0.0132 (-8.496) pass for all and outweigh e2 < 0.0503 (+3.798), lam1 < 0.00838 (+2.421), mass < 49.7 and z_dr_0p05_0p1 < 0.751 (+2.006) and lam1 < 0.00733 (+1.316). The neuron is on for only 8.5% (mean 0.103) and hardly moves the scores; the formula splits them q 51% / g 40%.
- **medium-mass (55 GeV), average width, pT spread over several particles** — 17.7% of jets, neuron 0.32. Mostly W (50%, with 32% Z and 10% tops), 17.66% of jets: mass 54.5 GeV, width 0.0059, with 58.87% of the pT at 0.05-0.1 from the axis. mass_over_sum_pt > 0.00837 (-5.95), width < 0.0132 (-4.806) and girth2 < 0.00868 (-3.554) all pass, and the plain max_dr term (+2.357), e2 < 0.0503 (+1.29) and D2 < 1.68 (95%, +0.819) do not make up for them. The neuron is on for 20.7% (mean 0.319), nudging the W and Z scores down and the g and q scores up a little; the formula calls them W (65%).
- **light (45 GeV), narrow, pT spread over several particles** — 12.8% of jets, neuron 0.31. Mostly W (45%, with 28% Z and 11% gluons), 12.76% of jets: mass 44.6 GeV, width 0.004, with 56.09% of the pT at 0.025-0.05 from the axis. width < 0.0132 (-6.061), girth2 < 0.00868 (-6.008) and mass_over_sum_pt > 0.00837 (-4.589) outweigh max_dr (+2.96), e2 < 0.0503 (+2.307) and lam1 < 0.00838 (+1.351). The neuron is on for 22.2% (mean 0.31) and moves the scores only slightly (W and Z down, g and q up); the formula calls them W (65%).
- **medium-mass (60 GeV), average width, pT spread over several particles** — 11.2% of jets, neuron 1.66. Mostly Z (58%, with 23% tops and 10% gluons), 11.18% of jets: mass 60.5 GeV, width 0.0083, with 60.97% of the pT at 0.05-0.1 and 19.06% at 0.1-0.15 from the axis. Being broader, they escape much of girth2 < 0.00868 (73.7%, only -0.907), so max_dr (+2.656), D2 < 1.68 (94.1%, +0.867) and e2 < 0.0503 can hold against mass_over_sum_pt > 0.00837 (-7.191) and width < 0.0132 (-3.246). The neuron is on for 66.9% (mean 1.655), lowering the Z (-0.621) and W (-0.517) scores and raising the q and g scores; the formula calls them Z (70%), with t 25%.
- **light (27 GeV), very narrow, pT spread over several particles** — 9.2% of jets, neuron 0.20. A mixture of gluons (33%), quarks (28%), W (21%) and Z (12%), 9.17% of jets: mass 26.9 GeV, width 0.0017, with 39.29% of the pT inside 0.025 and 47.53% at 0.025-0.05 from the axis. girth2 < 0.00868 (-8.918) and width < 0.0132 (-7.548) pass for all and beat e2 < 0.0503 (+2.94), max_dr (+2.035), lam1 < 0.00838 (+2.019) and lam1 < 0.00733 (+1.066). The neuron is on for 13.4% (mean 0.205) and barely moves the scores; the formula splits them g 47% / q 27% / W 25%.
- **medium-mass (76 GeV), very wide, pT spread over several particles, low pT** — 7.5% of jets, neuron 6.01. Mostly tops (78%, with 15% gluons), 7.5% of jets: mass 76.0 GeV, width 0.0173, soft (sum pT 596 GeV), with 38.61% of the pT at 0.05-0.1 and 28.84% at 0.1-0.15 from the axis. The narrowness switches no longer pass (girth2 < 0.00868 never, width < 0.0132 for 14.8%), so max_dr (+4.173), girth > 0.0872 (93.7%, +2.706) and LHA > 0.313 and eccentricity > 0.873 (64.7%) keep it high despite mass_over_sum_pt > 0.00837 (-10.641), centroid_offset > 0.0184 (66.4%, -1.259) and log_sum_pt < 6.7 (-1.171). The neuron is high (6.008, on for 98.9%), lowering the Z (-2.253) and W (-1.877) scores and raising the q and g scores; it does not enter the t score, and the formula calls them t (96%).
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 4.0% of jets, neuron 8.64. Mostly tops (75%, with 18% gluons), 3.96% of jets: mass 87.2 GeV, width 0.0292 (over four times the average), soft (sum pT 530 GeV), with most of the pT beyond 0.1 from the axis. The breadth bonuses are large: girth > 0.0872 (+6.919), max_dr (+4.272), girth2_top5 > 0.0115 (+2.448), LHA > 0.313 and eccentricity > 0.873 (82.3%, +2.398) and log_sum_pt < 6.7 and mean_eta2 > 0.00425 (81.3%, +2.039), against mass_over_sum_pt > 0.00837 (-13.937) and centroid_offset > 0.0184 (80.2%, -2.766). The neuron reaches its highest value, 8.642, strongly lowering the Z (-3.241) and W (-2.701) scores and raising the q (+1.08) and g scores; the formula calls them t (93%).
- **medium-mass (53 GeV), very wide, pT spread over several particles, low pT** — 3.0% of jets, neuron 7.84. Mostly tops (69%) mixed with 22% gluons, 2.98% of jets: mass 53.4 GeV, width 0.0152, soft (sum pT 535 GeV), with 37.49% of the pT at 0.05-0.1 and 39.62% at 0.1-0.15 from the axis and the pT centroid off the axis. The off-centre tests set them apart: centroid_offset > 0.0184 passes for all (-6.569), but centroid_offset > 0.00809 and lam2 < 0.00341 (86.9%, +2.254) and centroid_offset > 0.00809 (+1.706) return part, and max_dr (+3.669) and girth > 0.0872 (91.2%, +2.531) offset mass_over_sum_pt > 0.00837 (-8.165). The neuron is high (7.842, on for 99.4%), lowering the Z and W scores and raising the q and g scores; the formula calls them t (93%).
- **very light (12 GeV), narrow, pT spread over several particles, low pT** — 2.8% of jets, neuron 2.72. A mixture of gluons (39%), Z (21%), tops (16%), quarks (12%) and W (11%), 2.79% of jets: mass 12.0 GeV, width 0.0026, soft (sum pT 592 GeV), with only 4.85% of the pT inside 0.025 but 56.55% at 0.025-0.05 and 37.7% at 0.05-0.1 from the axis. Like the other light jets they pay girth2 < 0.00868 (-7.786) and width < 0.0132 (-6.97), and centroid_offset > 0.0184 passes for all (-3.373); but e2 < 0.0503 (+3.393), centroid_offset > 0.00809 and lam2 < 0.00341 (+2.053), lam1 < 0.00838 (+1.767) and a weaker mass_over_sum_pt > 0.00837 (78.7%, -1.175) leave it clearly on. The neuron sits at 2.723 (on for 92.4%), lowering the Z (-1.021) and W scores and raising the q and g scores; the formula splits them g 63% / Z 26%.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 2.6% of jets, neuron 1.74. Mostly tops (95%), 2.56% of jets: mass 86.0 GeV, width 0.0286, soft (sum pT 528 GeV), with almost no pT inside 0.05 and 38.05% at 0.1-0.15 from the axis. They share the breadth bonuses of group 6 (girth > 0.0872 +6.816, max_dr +4.6, girth2_top5 > 0.0115 +2.209) and its penalty mass_over_sum_pt > 0.00837 (-13.834), but the pT is spread in both directions: lam2 > 0.00341 passes for all (-4.685) and LHA > 0.313 and eccentricity > 0.873 never does. The neuron falls to 1.736 (on for 58.3%), lowering the Z and W scores and raising the q and g scores moderately; the formula calls them t (99%).

### neuron 7: two-prong jet of W/Z mass (major)

- **What it measures:** Set mostly by girth2 steps (pushed down above 0.00165, 0.00437 and 0.00752, back up above 0.00868 and 0.0132), and pushed down for very narrow jets (girth < 0.0872, width < 0.00559); its mass/pT terms push up above 0.0727 and 0.0848 but down above 0.0904, so it favours a boson-like mass/pT; it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.587, -0.587, -0.517). Z jets sit highest (3.42), then W (2.10), with top (1.05), gluon (0.76) and quark (0.40) jets low.
- *computed — its value:* largest for Z (3.42), then W (2.10), then t (1.05), then g (0.76), then q (0.40); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the W score (+10%) and the Z score (+29%), where it is the largest input. It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* raises the score of W (+10%), Z (+29%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 9.65
if girth2 > 0.0075: z += -1387 × (girth2 − 0.0075)
if girth < 0.087: z += -88.48 × (0.087 − girth)
if girth2 > 0.0087: z += 1439 × (girth2 − 0.0087)
if girth2 > 0.0017: z += -556 × (girth2 − 0.0017)
if width < 0.0056: z += -1030 × (0.0056 − width)
if mass_over_sum_pt > 0.090: z += -270 × (mass_over_sum_pt − 0.090)
if lam1 < 0.0084: z += -482 × (0.0084 − lam1)
if e2 < 0.038: z += 128 × (0.038 − e2)
if mass_over_sum_pt > 0.085: z += 186 × (mass_over_sum_pt − 0.085)
if e2 < 0.050: z += -66.85 × (0.050 − e2)
if girth2 > 0.0044: z += -423 × (girth2 − 0.0044)
if mass_over_sum_pt > 0.073: z += 85.82 × (mass_over_sum_pt − 0.073)
if girth2 > 0.013: z += 758 × (girth2 − 0.013)
if LHA < 0.293: z += 11.65 × (0.293 − LHA)
if e2 < 0.025: z += 92.84 × (0.025 − e2)
if lam1 < 0.0084 and D2 < 1.12: z += -1226 × (0.0084 − lam1) × (1.12 − D2)
if e2 < 0.050 and D2 < 1.12: z += 191 × (0.050 − e2) × (1.12 − D2)
if e2_sq < 0.0011: z += 1536 × (0.0011 − e2_sq)
if mass_over_sum_pt > 0.108: z += -82.97 × (mass_over_sum_pt − 0.108)
if pt_7 < 48.72 and planar_flow < 0.695: z += -0.070 × (48.72 − pt_7) × (0.695 − planar_flow)
if girth2 > 0.0044 and eccentricity > 0.946: z += 5185 × (girth2 − 0.0044) × (eccentricity − 0.946)
if planar_flow < 0.195 and sum_pt > 616: z += 0.028 × (0.195 − planar_flow) × (sum_pt − 616)
if girth < 0.041: z += -29.78 × (0.041 − girth)
if centroid_offset < 0.021: z += -26.07 × (0.021 − centroid_offset)
if mass < 29.64: z += 0.029 × (29.64 − mass)
if pt_7 < 48.72: z += 0.014 × (48.72 − pt_7)
if mass > 80.40: z += -0.166 × (mass − 80.40)
if planar_flow < 0.195 and width > 0.0075: z += -1358 × (0.195 − planar_flow) × (width − 0.0075)
if centroid_offset < 0.038: z += -7.03 × (0.038 − centroid_offset)
if max_dr < 0.198: z += -1.65 × (0.198 − max_dr)
if planar_flow < 0.195: z += -1.36 × (0.195 − planar_flow)
if width < 0.00056: z += -959 × (0.00056 − width)
if centroid_offset < 0.021 and C2 > 0.024: z += 1630 × (0.021 − centroid_offset) × (C2 − 0.024)
if girth2_top3 < 0.0059: z += 22.49 × (0.0059 − girth2_top3)
if centroid_offset > 0.031: z += 23.96 × (centroid_offset − 0.031)
if planar_flow < 0.195 and mass_top3 > 23.66: z += 0.086 × (0.195 − planar_flow) × (mass_top3 − 23.66)
if planar_flow < 0.195 and pt_6 < 35.28: z += -0.268 × (0.195 − planar_flow) × (35.28 − pt_6)
if girth2 > 0.013 and log_sum_pt > 6.19: z += 234 × (girth2 − 0.013) × (log_sum_pt − 6.19)
if centroid_offset > 0.031 and pt_2 > 56.50: z += -0.845 × (centroid_offset − 0.031) × (pt_2 − 56.50)
if mass > 80.40 and eccentricity > 0.927: z += -0.885 × (mass − 80.40) × (eccentricity − 0.927)
if z_dr_0p05_0p1 > 0.675: z += -0.599 × (z_dr_0p05_0p1 − 0.675)
if z_dr_0p05_0p1 > 0.675 and pt_2 > 84.62: z += 0.020 × (z_dr_0p05_0p1 − 0.675) × (pt_2 − 84.62)
if centroid_offset > 0.031 and n_pt_above_50 > 6.00: z += -34.78 × (centroid_offset − 0.031) × (n_pt_above_50 − 6.00)
if centroid_offset > 0.031 and pt_0 > 376: z += -3.62 × (centroid_offset − 0.031) × (pt_0 − 376)
if width < 0.0056 and mean_eta < -0.0094: z += 2614 × (0.0056 − width) × (-0.0094 − mean_eta)
if e2_sq < 0.0011 and phi_1 < -0.0095: z += -12669 × (0.0011 − e2_sq) × (-0.0095 − phi_1)
if e2 < 0.025 and z_dr_0p05_0p1 > 0.751: z += -191 × (0.025 − e2) × (z_dr_0p05_0p1 − 0.751)
if e2_sq < 0.0011 and phi_0 > 0.040: z += -116468 × (0.0011 − e2_sq) × (phi_0 − 0.040)
if e2_sq < 0.0011 and eta_2 < -0.045: z += -88692 × (0.0011 − e2_sq) × (-0.045 − eta_2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 33.3% of jets, neuron 0.26. Mostly quarks (44%, with 34% gluons), 33.34% of jets: mass 9.6 GeV, width 0.0004, with 83.78% of the pT inside 0.025 of the axis. The narrow-jet penalties girth < 0.0872 (-6.432), width < 0.00559 (-5.356), lam1 < 0.00838 (-3.872) and e2 < 0.0503 (-2.983) all pass and outweigh e2 < 0.0385 (+4.213), LHA < 0.293 (+1.973), e2 < 0.0245 (+1.753) and e2_sq < 0.0011 (97.4%); none of the girth2 steps pass. The neuron is on for 18.3% (mean 0.262) and raises the Z and W scores only slightly; the formula splits them q 49% / g 40%.
- **medium-mass (55 GeV), average width, pT spread over several particles** — 29.7% of jets, neuron 3.67. Mostly Z (42%, with 35% W and 12% tops), 29.68% of jets: mass 55.0 GeV, width 0.0065, with 60.82% of the pT at 0.05-0.1 from the axis. No large term applies in full: girth2 > 0.00165 (-2.672), girth < 0.0872 (90.8%, -1.254) and lam1 < 0.00838 and D2 < 1.12 (81.1%, -1.236) are balanced by e2 < 0.0503 and D2 < 1.12 (85.1%, +1.174) and mass_over_sum_pt > 0.0727 (72.8%), and the costly girth2 > 0.00752 step passes for only 22.3%, so the neuron keeps most of its baseline. It sits at 3.673 (on for 99.3%) and raises the Z (+1.722) and W (+0.803) scores; the formula splits them W 45% / Z 43%.
- **light (37 GeV), narrow, pT spread over several particles** — 18.6% of jets, neuron 1.57. A mixture of W (33%), Z (24%), gluons (21%) and quarks (13%), 18.6% of jets: mass 36.6 GeV, width 0.0032, with 59.56% of the pT at 0.025-0.05 from the axis. The narrow-jet penalties return at half strength, girth < 0.0872 (-3.731), lam1 < 0.00838 (-2.577), width < 0.00559 (99.1%, -2.48) and e2 < 0.0503 (-2.083), and only e2 < 0.0385 (+2.483) and LHA < 0.293 (+0.773) push back. The neuron sits at 1.573 (on for 76.9%), raising the Z (+0.737) and W (+0.344) scores; the formula calls them W (46%), with g 29%.
- **medium-mass (59 GeV), wide, pT spread over several particles, low pT** — 3.8% of jets, neuron 2.03. Mostly tops (62%, with 22% gluons), 3.83% of jets: mass 58.7 GeV, width 0.0114, soft (sum pT 592 GeV), with 51.04% of the pT at 0.05-0.1 from the axis. The girth2 steps cost a lot (girth2 > 0.00165 -5.421, girth2 > 0.00752 -5.397, girth2 > 0.00437 -2.981, only partly returned by girth2 > 0.00868 +3.932), but the mass/pT window pays back: mass_over_sum_pt > 0.0848 (93.8%, +2.762) and mass_over_sum_pt > 0.0727 (+2.277) against mass_over_sum_pt > 0.0904 (87.2%, -2.627). The neuron stays at 2.032 (on for 90.9%), raising the Z (+0.953) and W (+0.445) scores; the formula calls them t (88%).
- **medium-mass (69 GeV), very wide, pT spread over several particles, low pT** — 3.5% of jets, neuron 0.17. Mostly tops (75%, with 17% gluons), 3.5% of jets: mass 69.0 GeV, width 0.0153, soft (sum pT 590 GeV), with pT mostly at 0.05-0.15 from the axis. Large opposite terms nearly cancel: girth2 > 0.00752 (-10.779) against girth2 > 0.00868 (+9.516), and mass_over_sum_pt > 0.0904 (98.6%, -7.124) against mass_over_sum_pt > 0.0848 (+5.939), while girth2 > 0.00165 (-7.577) and girth2 > 0.00437 (-4.624) outweigh mass_over_sum_pt > 0.0727 (+3.775) and girth2 > 0.0132 (95.8%, +1.561). The neuron is on for 28% (mean 0.166) and barely moves the W and Z scores; the formula calls them t (96%).
- **medium-mass (77 GeV), very wide, pT spread over several particles, low pT** — 3.4% of jets, neuron 0.01. Mostly tops (81%), 3.39% of jets: mass 76.6 GeV, width 0.0192, soft (sum pT 578 GeV), with pT mostly at 0.05-0.2 from the axis. The same cancelling pairs, now larger (girth2 > 0.00752 -16.234 vs girth2 > 0.00868 +15.176; mass_over_sum_pt > 0.0904 -11.302 vs mass_over_sum_pt > 0.0848 +8.821), leave girth2 > 0.00165 (-9.762) and girth2 > 0.00437 (-6.29) plus mass_over_sum_pt > 0.108 (-2.023) as a net loss. The neuron is on for only 1.9% (mean 0.009) and adds nothing to the scores; the formula calls them t (97%).
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 3.2% of jets, neuron 0.00. Mostly tops (86%), 3.22% of jets: mass 82.5 GeV, width 0.0237, soft (sum pT 556 GeV), with the pT spread from 0.05 to 0.3 from the axis. girth2 > 0.00752 (-22.442) and girth2 > 0.00868 (+21.616) cancel, as do much of mass_over_sum_pt > 0.0904 (-15.605) and mass_over_sum_pt > 0.0848 (+11.783); girth2 > 0.00165 (-12.248) and girth2 > 0.00437 (-8.185) beat girth2 > 0.0132 (+7.929) and mass_over_sum_pt > 0.0727 (+6.482). The neuron is off for essentially all of them and adds nothing to the scores; the formula calls them t (98%).
- **medium-mass (88 GeV), very wide, pT spread over several particles, low pT** — 2.7% of jets, neuron 0.00. Mostly tops (86%), 2.71% of jets: mass 88.5 GeV, width 0.0285, soft (sum pT 538 GeV), with most of the pT beyond 0.1 from the axis. The step pairs grow further (girth2 > 0.00752 -29.068 vs girth2 > 0.00868 +28.49; mass_over_sum_pt > 0.0904 -19.928 vs mass_over_sum_pt > 0.0848 +14.755), and girth2 > 0.00165 (-14.903) and girth2 > 0.00437 (-10.209) outweigh girth2 > 0.0132 (+11.55). The neuron is off for all of them and adds nothing to the scores; the formula calls them t (97%).
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 1.4% of jets, neuron 0.00. Mostly tops (77%, with 17% gluons), 1.39% of jets: mass 91.1 GeV, width 0.0342, soft (sum pT 508 GeV), with 30.36% of the pT at 0.15-0.2 and 31.34% at 0.2-0.3 from the axis. girth2 > 0.00752 (-37.027) and girth2 > 0.00868 (+36.746) cancel, mass_over_sum_pt > 0.0904 (-24.006) beats mass_over_sum_pt > 0.0848 (+17.559), and girth2 > 0.00165 (-18.09) outweighs girth2 > 0.0132 (+15.898). The neuron is off for all of them and adds nothing to the scores; the formula calls them t (92%).
- **heavy (94 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 0.00. Mostly tops (57%) mixed with 31% gluons, 0.34% of jets: mass 94.1 GeV, width 0.0449 (seven times the average), the softest group (sum pT 465 GeV), with 48.36% of the pT at 0.2-0.3 from the axis. The largest terms of the neuron cancel almost exactly: girth2 > 0.00868 (+52.046) vs girth2 > 0.00752 (-51.775) and girth2 > 0.0132 (+23.957) vs girth2 > 0.00165 (-23.998), while mass_over_sum_pt > 0.0904 (-30.203) outweighs mass_over_sum_pt > 0.0848 (+21.821). The neuron is off for all of them and adds nothing to the scores; the formula calls them t (84%), with g 16%.

### neuron 9: narrow single-core jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.0061, its strongest term, plus girth2 < 0.00752 and < 0.00437, max ΔR < 0.222 and a centred pT centroid), and pushed down for light jets (mass < 53.3 GeV) and small mass/pT (< 0.0764); it falls with lam1, width and girth2 (rank correlations -0.676, -0.666, -0.666). Quarks sit highest (6.73), then gluons (4.89), far above W (1.85), top (1.78) and Z (1.01) jets.
- *computed — its value:* largest for q (6.73), then g (4.89), then W (1.85), then t (1.78), then Z (1.01); it separates q jets from the rest best (AUC 0.80: large for q)
- **How the class scores use it:** High values mean a light-parton (QCD) jet, so it raises the q score (its largest input, +50%) and the g score (+24%), and lowers the W score (-3%) and the Z score (-4%) slightly. It does not (or hardly) enter the t score.
- *computed — used by:* raises the score of g (+24%), q (+50%); lowers the score of W (-3%), Z (-4%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.53
if width < 0.0061: z += 1522 × (0.0061 − width)
if mass < 53.33: z += -0.091 × (53.33 − mass)
if girth2 < 0.0075: z += 437 × (0.0075 − girth2)
if girth2 < 0.0044: z += 904 × (0.0044 − girth2)
if girth < 0.055: z += -84.28 × (0.055 − girth)
if max_dr < 0.222: z += 10.68 × (0.222 − max_dr)
if mass_over_sum_pt < 0.076: z += -40.24 × (0.076 − mass_over_sum_pt)
if e2 < 0.020: z += 197 × (0.020 − e2)
if centroid_offset < 0.018: z += 159 × (0.018 − centroid_offset)
if width < 0.0061 and centroid_offset > 0.0033: z += -46014 × (0.0061 − width) × (centroid_offset − 0.0033)
if mass < 53.33 and centroid_offset < 0.027: z += 2.80 × (53.33 − mass) × (0.027 − centroid_offset)
if log_sum_pt > 6.38: z += -3.79 × (log_sum_pt − 6.38)
if mass < 29.64: z += -0.103 × (29.64 − mass)
if mass < 53.33 and lam1 < 0.00087: z += -79.35 × (53.33 − mass) × (0.00087 − lam1)
if girth2 < 0.00096: z += 3271 × (0.00096 − girth2)
if mass < 41.38: z += 0.052 × (41.38 − mass)
if mass < 53.33 and log_sum_pt < 6.84: z += 0.121 × (53.33 − mass) × (6.84 − log_sum_pt)
if centroid_offset < 0.018 and z_4 > 0.047: z += -2136 × (0.018 − centroid_offset) × (z_4 − 0.047)
if mass < 53.33 and planar_flow < 0.322: z += 0.294 × (53.33 − mass) × (0.322 − planar_flow)
if max_dr < 0.222 and z_top5 < 0.909: z += -37.95 × (0.222 − max_dr) × (0.909 − z_top5)
if lam1 < 0.0015: z += -709 × (0.0015 − lam1)
if centroid_offset < 0.018 and pt_4 > 47.34: z += 2.58 × (0.018 − centroid_offset) × (pt_4 − 47.34)
if lam2 > 0.0011: z += 859 × (lam2 − 0.0011)
if mass < 29.64 and mean_phi2 < 0.0021: z += -19.21 × (29.64 − mass) × (0.0021 − mean_phi2)
if e2 < 0.032 and dr01 < 0.056: z += 457 × (0.032 − e2) × (0.056 − dr01)
if mass < 41.38 and planar_flow < 0.322: z += -0.268 × (41.38 − mass) × (0.322 − planar_flow)
if centroid_offset < 0.0095 and mean_phi2 < 0.0021: z += -71209 × (0.0095 − centroid_offset) × (0.0021 − mean_phi2)
if lam1 < 0.006: z += -75.85 × (0.006 − lam1)
if e2 < 0.020 and pt_7 < 53.44: z += -1.28 × (0.020 − e2) × (53.44 − pt_7)
if girth2 > 0.019: z += 198 × (girth2 − 0.019)
if log_sum_pt > 6.38 and dr_5 < 0.022: z += 162 × (log_sum_pt − 6.38) × (0.022 − dr_5)
if e2 < 0.032 and tau21 < 0.447: z += -185 × (0.032 − e2) × (0.447 − tau21)
if e2 < 0.032: z += -11.35 × (0.032 − e2)
if max_dr > 0.160: z += -5.84 × (max_dr − 0.160)
if girth < 0.055 and mean_phi < 0.0028: z += -1734 × (0.055 − girth) × (0.0028 − mean_phi)
if girth < 0.055 and dr_5 < 0.022: z += -1037 × (0.055 − girth) × (0.022 − dr_5)
if pt_5 < 35.50: z += 0.071 × (35.50 − pt_5)
if width < 0.00017: z += 6817 × (0.00017 − width)
if e2 < 0.020 and eccentricity > 0.903: z += -749 × (0.020 − e2) × (eccentricity − 0.903)
if C2 > 0.051: z += 18.68 × (C2 − 0.051)
if mass < 53.33 and n_dr_0p05_0p1 > 3.00: z += 0.014 × (53.33 − mass) × (n_dr_0p05_0p1 − 3.00)
if n_dr_0p2_0p4 > 1.00: z += 0.474 × (n_dr_0p2_0p4 − 1.00)
if centroid_offset < 0.0095: z += -36.48 × (0.0095 − centroid_offset)
if girth2 < 0.00096 and mean_eta < 0.0068: z += -40853 × (0.00096 − girth2) × (0.0068 − mean_eta)
if width < 0.0061 and C2 > 0.031: z += -10696 × (0.0061 − width) × (C2 − 0.031)
if z_dr_0p2_0p4 > 0.101: z += -3.12 × (z_dr_0p2_0p4 − 0.101)
if girth < 0.055 and mean_phi2 > 0.00054: z += 17513 × (0.055 − girth) × (mean_phi2 − 0.00054)
if C2 > 0.051 and n_dr_0p05_0p1 > 1.00: z += -5.34 × (C2 − 0.051) × (n_dr_0p05_0p1 − 1.00)
if girth2 > 0.019 and pt_7 < 29.04: z += 35.57 × (girth2 − 0.019) × (29.04 − pt_7)
if girth2 < 0.0044 and n_dr_0p05_0p1 > 3.00: z += -276 × (0.0044 − girth2) × (n_dr_0p05_0p1 − 3.00)
if girth2 < 4.8e-05: z += 20193 × (4.8e-05 − girth2)
if lam2 > 0.0011 and mass_top2 > 16.31: z += 4.32 × (lam2 − 0.0011) × (mass_top2 − 16.31)
if girth2 < 0.0044 and mass_top2 > 6.78: z += -28.03 × (0.0044 − girth2) × (mass_top2 − 6.78)
if C2 > 0.051 and eccentricity < 0.621: z += 30.11 × (C2 − 0.051) × (0.621 − eccentricity)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (63 GeV), very wide, pT spread over several particles, low pT** — 22.4% of jets, neuron 0.35. Mostly tops (50%) with 21% Z and 16% gluons, 22.37% of jets: mass 63.1 GeV, width 0.0138, soft (sum pT 601 GeV), with 45.43% of the pT at 0.05-0.1 and much of the rest further out. They are too wide for width < 0.0061 (12.4% pass) and girth2 < 0.00437 (never), so only small terms act: max_dr < 0.222 (66.8%, +0.496) and girth2 > 0.0188 (25.1%, +0.344) against log_sum_pt > 6.38 (52.2%, -0.337) and mass < 53.3 (36.2%, -0.31). The neuron is on for 27.1% (mean 0.352), barely raising the g and q scores; the formula calls them t (62%), with Z 24%.
- **medium-mass (63 GeV), average width, pT spread over several particles** — 17.7% of jets, neuron 0.51. A Z/W mixture (Z 41%, W 39%, tops 14%), 17.74% of jets: mass 63.0 GeV, width 0.0075, with 55.06% of the pT at 0.05-0.1 from the axis. The centred-pT test centroid_offset < 0.0184 passes for all (+1.879), with max_dr < 0.222 (92.4%, +0.928), centroid_offset < 0.0184 and pt_4 > 47.3 (82.2%) and girth2 < 0.00752 (69.9%); but centroid_offset < 0.0184 and z_4 > 0.0475 (93.1%, -0.974) and log_sum_pt > 6.38 (88.4%, -0.925) cancel most of it, and width < 0.0061 passes for only 35.4%. The neuron is on for 46.4% (mean 0.513), nudging the q and g scores up; the formula splits them W 47% / Z 42%.
- **very light (7 GeV), very narrow, pT spread over several particles, high pT** — 15.9% of jets, neuron 9.03. Mostly quarks (60%, with 28% gluons), 15.92% of jets: mass 7.2 GeV, width 0.0001, somewhat hard (sum pT 882 GeV), with 97.99% of the pT inside 0.025 of the axis. Every narrowness bonus passes: width < 0.0061 (+9.131), girth2 < 0.00437 (+3.863), girth2 < 0.00752 (+3.247), e2 < 0.0205 (+3.24), mass < 53.3 and centroid_offset < 0.0269 (+3.072), girth2 < 0.000964 (+2.838) and centroid_offset < 0.0184 (+2.434), far outweighing mass < 53.3 (-4.205), girth < 0.0546 (-3.998) and mass < 53.3 and lam1 < 0.000872 (-2.92). The neuron reaches its highest value, 9.026 (on for all), raising the q (+2.292) and g (+1.551) scores and lowering the W and Z scores (-0.282 each); the formula calls them q (74%).
- **light (45 GeV), average width, pT spread over several particles** — 11.0% of jets, neuron 1.35. Mostly W (46%, with 27% Z and 11% gluons), 11.01% of jets: mass 45.0 GeV, width 0.0045, with 47.86% of the pT at 0.025-0.05 and 35.01% at 0.05-0.1 from the axis. width < 0.0061 (99.3%, +2.518), girth2 < 0.00752 (+1.343), max_dr < 0.222 (88.2%, +0.812) and centroid_offset < 0.0184 (59%, +0.647) are partly cancelled by width < 0.0061 and centroid_offset > 0.00334 (97%, -1.03), mass < 53.3 (77.7%, -0.882), log_sum_pt > 6.38 (78.7%, -0.786) and mass_over_sum_pt < 0.0764 (-0.554). The neuron sits at 1.349 (on for 67.5%), raising the q (+0.342) and g (+0.232) scores; the formula calls them W (67%).
- **light (36 GeV), narrow, pT spread over several particles** — 7.6% of jets, neuron 2.73. A mixture led by W (36%) with Z (21%), gluons (19%) and quarks (16%), 7.62% of jets: mass 35.9 GeV, width 0.0028, with 57.67% of the pT at 0.025-0.05 from the axis. width < 0.0061 (+5.072), girth2 < 0.00752 (+2.08), girth2 < 0.00437 (99.7%, +1.454) and mass < 53.3 and planar_flow < 0.322 (79.6%, +0.899) outweigh width < 0.0061 and centroid_offset > 0.00334 (97.7%, -2.192), mass < 53.3 (97%, -1.606), mass_over_sum_pt < 0.0764 (-1.139) and girth < 0.0546 (93%, -1.092). The neuron sits at 2.728 (on for 86%), raising the q (+0.693) and g (+0.469) scores; the formula splits them W 55% / g 25%.
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.5% of jets, neuron 6.04. Mostly gluons (49%, with 30% quarks), 7.46% of jets: mass 7.9 GeV, width 0.0003, somewhat soft (sum pT 721 GeV), with 91.15% of the pT inside 0.025 of the axis. Like group 2 they collect width < 0.0061 (+8.808), girth2 < 0.00437 (+3.671), girth2 < 0.00752 (+3.154) and e2 < 0.0205 (+2.897) against mass < 53.3 (-4.145), girth < 0.0546 (-3.286), mass_over_sum_pt < 0.0764 (-2.619) and mass < 29.6 (-2.233); the difference is the centroid, as width < 0.0061 and centroid_offset > 0.00334 passes for 99.4% (-2.255) and the centred bonuses are smaller. The neuron sits at 6.042 (on for 99.5%), raising the q (+1.534) and g (+1.038) scores; the formula calls them g (68%).
- **light (22 GeV), very narrow, pT spread over several particles** — 5.6% of jets, neuron 7.98. A quark-gluon mixture (quarks 42%, gluons 41%), 5.58% of jets: mass 22.0 GeV, width 0.001, with 66.38% of the pT inside 0.025 and 26.13% at 0.025-0.05 from the axis. width < 0.0061 (+7.778), girth2 < 0.00437 (+3.06), girth2 < 0.00752 (+2.858), centroid_offset < 0.0184 (+1.843), mass < 53.3 and centroid_offset < 0.0269 (+1.762) and e2 < 0.0205 (93.2%) outweigh mass < 53.3 (-2.861), girth < 0.0546 (-2.553) and mass_over_sum_pt < 0.0764 (-1.88), which cost less than for the lightest jets. The neuron is high (7.981, on for all), raising the q (+2.026) and g (+1.372) scores; the formula splits them q 51% / g 48%.
- **very light (15 GeV), very narrow, pT spread over several particles** — 4.3% of jets, neuron 2.30. Mostly gluons (41%) with W (19%), Z (17%) and quarks (16%), 4.28% of jets: mass 15.2 GeV, width 0.0015, soft (sum pT 610 GeV), with 26.66% of the pT inside 0.025 and 59.42% at 0.025-0.05 from the axis. width < 0.0061 (+7.027), girth2 < 0.00752 (+2.642), girth2 < 0.00437 (+2.614), mass < 53.3 and log_sum_pt < 6.84 (98.9%, +2.149) and e2 < 0.0205 (95%) keep it on despite width < 0.0061 and centroid_offset > 0.00334 (all, -4.628), mass < 53.3 (-3.48), mass_over_sum_pt < 0.0764 (-2.05) and girth < 0.0546 (97.7%, -1.68); centroid_offset < 0.0184 passes for only 18.6%. The neuron sits at 2.303 (on for 80.9%), raising the q and g scores; the formula calls them g (70%).
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 4.2% of jets, neuron 5.71. Mostly tops (93%), 4.17% of jets: mass 82.7 GeV, width 0.0265, soft (sum pT 536 GeV), with almost no pT inside 0.05 and the rest spread out to 0.3 from the axis. None of the narrowness tests pass here; instead lam2 > 0.00113 passes for all (+5.132), with girth2 > 0.0188 (87%, +1.594), C2 > 0.0512 (97.5%, +0.942) and n_dr_0p2_0p4 > 1 (58.9%), and only max_dr > 0.16 (99%, -0.588) pulls back. So this narrow-jet neuron is also high here (5.713, on for 99.6%), raising the q (+1.451) and g (+0.982) scores and lowering the W and Z scores; the formula calls them t (99%).
- **very light (6 GeV), very narrow, pT spread over several particles** — 3.9% of jets, neuron 0.75. A mixture of gluons (28%), Z (26%), W (23%) and quarks (14%), 3.86% of jets: mass 5.9 GeV, width 0.0012, with 31.23% of the pT inside 0.025 and 59.82% at 0.025-0.05 from the axis. width < 0.0061 (+7.494), mass < 53.3 and planar_flow < 0.322 (+3.183), e2 < 0.0205 (+3.16) and the girth2 bonuses are cancelled by width < 0.0061 and centroid_offset > 0.00334 (all, -5.944), mass < 53.3 (-4.328), mass_over_sum_pt < 0.0764 (-2.732) and mass < 29.6 (-2.439); centroid_offset < 0.0184 almost never passes. The neuron is on for 45.9% (mean 0.747), nudging the q and g scores up; the formula splits them g 41% / Z 35%.

### neuron 10: overall jet size (e2, mass, width) (major)

- **What it measures:** Grows steadily with e2 (its strongest term) and is pushed up for mass > 15.5 GeV, and pushed down for very narrow jets (lam1 < 0.00418); it follows e2, mass/pT and width very closely (rank correlations 0.908, 0.9, 0.898). Tops sit highest (5.32), Z (2.49) and W (2.43) in the middle, gluons (1.18) and quarks (0.62) lowest.
- *computed — its value:* largest for t (5.32), then Z (2.49), then W (2.43), then g (1.18), then q (0.62); it separates t jets from the rest best (AUC 0.84: large for t)
- **How the class scores use it:** Large size marks a top, so it raises the t score (+26%); small size marks a quark, so it lowers the q score (-18%). It does not (or hardly) enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+26%); lowers the score of q (-18%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.69
z += 71.16 × e2
if mass > 15.45: z += 0.045 × (mass − 15.45)
if mass_over_sum_pt < 0.090: z += 30.09 × (0.090 − mass_over_sum_pt)
if lam1 < 0.0042: z += -506 × (0.0042 − lam1)
if lam2 > 0.00019: z += 1411 × (lam2 − 0.00019)
if log_sum_pt > 6.70: z += -17.09 × (log_sum_pt − 6.70)
if lam1 < 0.0065: z += -160 × (0.0065 − lam1)
if girth2_top2 < 0.0024: z += 452 × (0.0024 − girth2_top2)
if girth2 < 0.0017: z += -917 × (0.0017 − girth2)
if eccentricity > 0.903 and z_dr_0p2_0p4 < 0.056: z += 177 × (eccentricity − 0.903) × (0.056 − z_dr_0p2_0p4)
if sum_pt > 813: z += 0.011 × (sum_pt − 813)
if n_dr_0p05_0p1 < 3.00: z += 0.199 × (3.00 − n_dr_0p05_0p1)
if girth2 < 0.0067: z += 102 × (0.0067 − girth2)
if girth2_top5 < 0.0023: z += 349 × (0.0023 − girth2_top5)
if lam2 > 0.00019 and planar_flow > 0.013: z += -631 × (lam2 − 0.00019) × (planar_flow − 0.013)
if LHA > 0.303: z += -9.86 × (LHA − 0.303)
if eccentricity > 0.903 and mass_top2 < 36.77: z += -0.131 × (eccentricity − 0.903) × (36.77 − mass_top2)
if eccentricity > 0.903: z += 3.54 × (eccentricity − 0.903)
if lam1 < 0.0042 and log_sum_pt > 6.70: z += 1741 × (0.0042 − lam1) × (log_sum_pt − 6.70)
if centroid_offset > 0.0023: z += -10.89 × (centroid_offset − 0.0023)
if mass > 53.33: z += -0.020 × (mass − 53.33)
if tau21 < 0.392 and pt_4 > 39.81: z += -0.046 × (0.392 − tau21) × (pt_4 − 39.81)
if LHA > 0.303 and tau21 < 0.553: z += -18.32 × (LHA − 0.303) × (0.553 − tau21)
if mass > 15.45 and n_dr_0p2_0p4 < 2.00: z += -0.0026 × (mass − 15.45) × (2.00 − n_dr_0p2_0p4)
if max_dr > 0.122: z += -2.63 × (max_dr − 0.122)
if lam1 < 0.0042 and dr_7 < 0.175: z += -412 × (0.0042 − lam1) × (0.175 − dr_7)
if tau21 < 0.392 and z_4 > 0.075: z += 36.69 × (0.392 − tau21) × (z_4 − 0.075)
if tau21 < 0.392 and dr01 < 0.251: z += 2.62 × (0.392 − tau21) × (0.251 − dr01)
if C2 > 0.051: z += 12.78 × (C2 − 0.051)
if lam2 > 0.00019 and tau21 < 0.501: z += 1044 × (lam2 − 0.00019) × (0.501 − tau21)
if lam2 > 0.00019 and D2 < 2.06: z += -129 × (lam2 − 0.00019) × (2.06 − D2)
if tau32 < 0.269: z += 4.24 × (0.269 − tau32)
if lam2 > 0.00019 and n_pt_above_50 < 8.00: z += -28.24 × (lam2 − 0.00019) × (8.00 − n_pt_above_50)
if z_7 > 0.062: z += -9.26 × (z_7 − 0.062)
if n_dr_0p05_0p1 < 3.00 and D2 < 1.43: z += -0.112 × (3.00 − n_dr_0p05_0p1) × (1.43 − D2)
if centroid_offset > 0.038: z += 22.62 × (centroid_offset − 0.038)
if centroid_offset > 0.0023 and pt_7 < 34.53: z += -0.677 × (centroid_offset − 0.0023) × (34.53 − pt_7)
if eccentricity > 0.903 and z_top2_slots > 0.550: z += -23.00 × (eccentricity − 0.903) × (z_top2_slots − 0.550)
if mass_over_sum_pt < 0.090 and pt_7 > 33.22: z += 0.144 × (0.090 − mass_over_sum_pt) × (pt_7 − 33.22)
if mass_over_sum_pt_sq < 0.0039: z += 14.60 × (0.0039 − mass_over_sum_pt_sq)
if tau32 < 0.269 and n_dr_0p2_0p4 < 2.00: z += -2.54 × (0.269 − tau32) × (2.00 − n_dr_0p2_0p4)
if lam2 > 0.00019 and pt_6 < 38.25: z += -14.70 × (lam2 − 0.00019) × (38.25 − pt_6)
if tau21 < 0.392 and mean_eta < -0.0068: z += -36.64 × (0.392 − tau21) × (-0.0068 − mean_eta)
if lam1 < 0.0042 and mean_phi > 0.0091: z += 9993 × (0.0042 − lam1) × (mean_phi − 0.0091)
if lam2 > 0.00019 and z_top2_slots > 0.477: z += 1380 × (lam2 − 0.00019) × (z_top2_slots − 0.477)
if lam1 < 0.0042 and sum_pt > 988: z += -0.586 × (0.0042 − lam1) × (sum_pt − 988)
if n_dr_0p2_0p4 > 2.00 and dr_7 > 0.223: z += 4.57 × (n_dr_0p2_0p4 − 2.00) × (dr_7 − 0.223)
if n_dr_0p2_0p4 > 2.00 and dr_7 < 0.042: z += 37.07 × (n_dr_0p2_0p4 − 2.00) × (0.042 − dr_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (55 GeV), average width, pT spread over several particles** — 29.0% of jets, neuron 3.26. A Z/W mixture (Z 36%, W 31%) with 20% tops, 29.03% of jets: mass 54.8 GeV, width 0.0074, somewhat soft (sum pT 670 GeV), with 59.8% of the pT at 0.05-0.1 from the axis. The steadily growing e2 term (+2.764) and mass > 15.5 (all, +1.781) carry it, with eccentricity > 0.903 and z_dr_0p2_0p4 < 0.0564 (82.3%, +0.663) and mass_over_sum_pt < 0.0904 (80.2%); the narrow-jet penalty lam1 < 0.00418 almost never passes. The neuron sits at 3.26 (on for 99.9%), raising the t score (+1.223) and lowering the q score (-0.408); the formula splits them W 40% / Z 38% / t 20%.
- **very light (9 GeV), very narrow, pT spread over several particles** — 23.3% of jets, neuron 0.16. Mostly gluons (41%, with 35% quarks), 23.27% of jets: mass 9.4 GeV, width 0.0005, with 75.08% of the pT inside 0.025 of the axis. The e2 term is small here (+0.479) and mass > 15.5 passes for only 15.6%; mass_over_sum_pt < 0.0904 (+2.311) and the small-girth bonuses girth2_top2 < 0.00241 (+0.914), girth2_top5 < 0.00227 and girth2 < 0.00668 are cancelled by lam1 < 0.00418 (-1.887), girth2 < 0.00165 (97.8%, -1.057) and lam1 < 0.00651 (-0.968). The neuron is on for 30% (mean 0.157) and barely moves the scores; the formula splits them g 54% / q 34%.
- **light (34 GeV), narrow, pT spread over several particles** — 16.2% of jets, neuron 1.85. A mixture of W (30%), gluons (23%), Z (22%), quarks (14%) and tops (12%), 16.23% of jets: mass 33.9 GeV, width 0.0035, somewhat soft, with 53.92% of the pT at 0.025-0.05 from the axis. e2 (+1.427), mass_over_sum_pt < 0.0904 (+1.192), mass > 15.5 (94.2%, +0.852) and eccentricity > 0.903 and z_dr_0p2_0p4 < 0.0564 (74.5%) outweigh lam1 < 0.00418 (77.9%, -0.558) and lam1 < 0.00651 (98%, -0.521). The neuron sits at 1.853 (on for 98.2%), raising the t score (+0.695) and lowering the q score; the formula splits them W 42% / g 29%.
- **very light (10 GeV), very narrow, leading particle 43% of pT, high pT** — 9.8% of jets, neuron 0.09. Mostly quarks (62%, with 18% gluons), 9.78% of jets: mass 10.4 GeV, width 0.0003, hard (sum pT 952 GeV, leading particle 410.33 GeV), with 94.4% of the pT inside 0.025 of the axis. The hard-jet test log_sum_pt > 6.7 passes for all (-2.674) and joins lam1 < 0.00418 (-2.002), girth2 < 0.00165 (97.9%, -1.292) and lam1 < 0.00651 (-1.004); mass_over_sum_pt < 0.0904 (+2.393), sum_pt > 813 (+1.524), lam1 < 0.00418 and log_sum_pt > 6.7 (+1.078) and girth2_top2 < 0.00241 (+1.043) do not quite balance them. The neuron is on for 12.4% (mean 0.091) and barely moves the scores; the formula calls them q (83%).
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 6.9% of jets, neuron 4.78. Mostly tops (74%, with 18% gluons), 6.9% of jets: mass 82.7 GeV, width 0.0223, soft (sum pT 582 GeV), with 35.52% of the pT at 0.1-0.15 from the axis. The size terms are large, e2 (+5.045) and mass > 15.5 (+3.04), with lam2 > 0.000195 (62.8%, +0.597); LHA > 0.303 (97.2%, -0.953), LHA > 0.303 and tau21 < 0.553 (91.6%, -0.708) and mass > 53.3 (96.1%, -0.602) take a little back. The neuron is high (4.781, on for all), raising the t score (+1.793) and lowering the q score (-0.598); the formula calls them t (95%).
- **medium-mass (67 GeV), average width, leading particle 45% of pT, high pT** — 6.0% of jets, neuron 2.35. An even Z/W mixture (Z 43%, W 42%), 6.05% of jets: mass 67.3 GeV, narrow for their mass (width 0.0055), hard (sum pT 940 GeV, leading particle 419.49 GeV), with 46.59% of the pT at 0.025-0.05 from the axis. mass > 15.5 (+2.346), e2 (+2.003), sum_pt > 813 (+1.392) and eccentricity > 0.903 and z_dr_0p2_0p4 < 0.0564 (82.6%, +0.622) outweigh the hard-jet penalty log_sum_pt > 6.7 (all, -2.448). The neuron sits at 2.354 (on for 99.6%), raising the t score (+0.883) and lowering the q score; the formula splits them W 56% / Z 37%.
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.6% of jets, neuron 7.06. Mostly tops (83%, with 11% gluons), 3.57% of jets: mass 72.8 GeV, width 0.0208, soft (sum pT 547 GeV), with 36.4% of the pT at 0.05-0.1 and much of the rest further out. Besides e2 (+4.804) and mass > 15.5 (+2.594), the spread across the jet (not just along one line) lets lam2 > 0.000195 pass for all (+4.181); lam2 > 0.000195 and planar_flow > 0.0126 (-1.045) and LHA > 0.303 (95.8%, -0.839) take little back. The neuron is high (7.06, on for all), raising the t score (+2.648) and lowering the q score (-0.883); the formula calls them t (95%).
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 2.5% of jets, neuron 10.24. Mostly tops (94%), 2.51% of jets: mass 81.5 GeV, width 0.025, soft (sum pT 543 GeV), with almost no pT inside 0.025 and 34.94% at 0.1-0.15 from the axis. lam2 > 0.000195 grows to +8.65 and adds to e2 (+5.644) and mass > 15.5 (+2.99), while lam2 > 0.000195 and planar_flow > 0.0126 (-2.94) and LHA > 0.303 (-1.125) take only part back. The neuron is very high (10.245, on for all), raising the t score (+3.842) and lowering the q score (-1.281); the formula calls them t (99%).
- **very light (19 GeV), very narrow, leading particle 45% of pT, high pT** — 1.6% of jets, neuron 0.24. A gluon-quark mixture (gluons 44%, quarks 38%), 1.62% of jets: mass 19.3 GeV, width 0.0006, very hard (sum pT 1186 GeV, leading particle 534.19 GeV), with 85.85% of the pT inside 0.025 of the axis. log_sum_pt > 6.7 hits hard (-6.347) together with lam1 < 0.00418 (98.5%, -1.842) and girth2 < 0.00165 (86.7%, -1.137), against sum_pt > 813 (+4.088), lam1 < 0.00418 and log_sum_pt > 6.7 (98.5%, +2.333) and mass_over_sum_pt < 0.0904 (+2.239). The neuron is on for 26.2% (mean 0.244) and barely moves the scores; the formula splits them g 51% / q 40%.
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 1.0% of jets, neuron 13.28. Mostly tops (96%), 1.04% of jets: mass 86.3 GeV, width 0.0301, soft (sum pT 518 GeV), with no pT inside 0.05 and 37.1% at 0.1-0.15 and 31.8% at 0.15-0.2 from the axis. lam2 > 0.000195 reaches +14.767 and, with e2 (+6.529) and mass > 15.5 (+3.204), far outweighs lam2 > 0.000195 and planar_flow > 0.0126 (-5.942), LHA > 0.303 (-1.416), lam2 > 0.000195 and D2 < 2.06 (98.9%) and lam2 > 0.000195 and n_pt_above_50 < 8 (97%). The neuron reaches its highest value, 13.278, raising the t score (+4.979) and lowering the q score (-1.66); the formula calls them t (99%).

### neuron 11: compact, centred, elongated jet (major)

- **What it measures:** Pushed up for width < 0.00868 (its strongest term), a pT centroid close to the axis (centroid offset < 0.0499) and girth2 < 0.0132, and pushed down for girth < 0.0872; it rises with eccentricity and falls with planar flow, τ21 and centroid offset (rank correlations 0.494, -0.494, -0.427, -0.322). W jets sit highest (4.31), then Z (3.31), gluons (1.55) and quarks (1.33); tops are lowest (0.75) and at zero for 73.8% of them.
- *computed — its value:* largest for W (4.31), then Z (3.31), then g (1.55), then q (1.33), then t (0.75); it separates W jets from the rest best (AUC 0.83: large for W)
- **How the class scores use it:** Since W jets sit highest on it, it raises the W score, where it is the largest input (+25%). It does not (or hardly) enter the g, q, Z or t scores; the Z score does not use it although Z jets sit second on it.
- *computed — used by:* raises the score of W (+25%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -1.37
if width < 0.0087: z += 1093 × (0.0087 − width)
if centroid_offset < 0.050: z += 90.44 × (0.050 − centroid_offset)
if girth < 0.087: z += -65.16 × (0.087 − girth)
if girth2 < 0.013: z += 260 × (0.013 − girth2)
if girth2 < 0.0067: z += 666 × (0.0067 − girth2)
if e2_sq < 0.0064: z += -484 × (0.0064 − e2_sq)
if lam1 < 0.0084: z += -320 × (0.0084 − lam1)
if girth > 0.076: z += -106 × (girth − 0.076)
if e2 < 0.044: z += -49.93 × (0.044 − e2)
if planar_flow < 0.253: z += 8.78 × (0.253 − planar_flow)
if max_dr < 0.222: z += 6.27 × (0.222 − max_dr)
if girth > 0.076 and n_pt_above_50 < 7.00: z += 23.70 × (girth − 0.076) × (7.00 − n_pt_above_50)
if centroid_offset < 0.050 and log_sum_pt < 6.80: z += -81.71 × (0.050 − centroid_offset) × (6.80 − log_sum_pt)
if width < 0.0036: z += -499 × (0.0036 − width)
if lam1 < 0.0048: z += -307 × (0.0048 − lam1)
if centroid_offset > 0.014: z += -64.35 × (centroid_offset − 0.014)
if n_dr_0p1_0p2 < 3.00: z += -0.225 × (3.00 − n_dr_0p1_0p2)
if sum_pt_top5 < 687: z += 0.0028 × (687 − sum_pt_top5)
z += 20.07 × centroid_offset
if planar_flow < 0.253 and mass < 69.61: z += -0.139 × (0.253 − planar_flow) × (69.61 − mass)
if LHA < 0.155: z += -23.62 × (0.155 − LHA)
if centroid_offset < 0.050 and pt_7 < 48.72: z += -0.570 × (0.050 − centroid_offset) × (48.72 − pt_7)
if girth > 0.102: z += -45.19 × (girth − 0.102)
if planar_flow < 0.253 and width > 0.0061: z += -873 × (0.253 − planar_flow) × (width − 0.0061)
if max_dr < 0.112: z += -7.97 × (0.112 − max_dr)
if sum_pt_top5 < 687 and n_dr_0p2_0p4 < 2.00: z += -0.00092 × (687 − sum_pt_top5) × (2.00 − n_dr_0p2_0p4)
if girth2_top3 < 0.0022 and phi_0 > -0.030: z += 8290 × (0.0022 − girth2_top3) × (phi_0 − -0.030)
if centroid_offset < 0.050 and mean_phi2 < 0.0016: z += -7269 × (0.050 − centroid_offset) × (0.0016 − mean_phi2)
if C2 < 0.036: z += -11.66 × (0.036 − C2)
if planar_flow < 0.253 and girth2 > 0.013: z += -1372 × (0.253 − planar_flow) × (girth2 − 0.013)
if n_dr_0_0p05 < 7.00: z += -0.048 × (7.00 − n_dr_0_0p05)
if girth2_top3 < 0.0022: z += -180 × (0.0022 − girth2_top3)
if pt_7 < 29.04: z += -0.058 × (29.04 − pt_7)
if n_dr_0_0p05 < 7.00 and z_1st < 0.501: z += -0.164 × (7.00 − n_dr_0_0p05) × (0.501 − z_1st)
if girth2_top5 < 0.00066: z += -748 × (0.00066 − girth2_top5)
if max_dr < 0.222 and phi_0 < 0.021: z += 34.29 × (0.222 − max_dr) × (0.021 − phi_0)
if planar_flow < 0.253 and D2 < 1.68: z += 1.01 × (0.253 − planar_flow) × (1.68 − D2)
if girth2_top2 < 0.0011: z += -307 × (0.0011 − girth2_top2)
if pt_7 < 29.04 and n_dr_0p2_0p4 < 2.00: z += -0.021 × (29.04 − pt_7) × (2.00 − n_dr_0p2_0p4)
if pt_7 < 29.04 and mass_top2 < 22.84: z += 0.0017 × (29.04 − pt_7) × (22.84 − mass_top2)
if mass_top5 < 9.26: z += 0.033 × (9.26 − mass_top5)
if centroid_offset < 0.050 and mean_phi < -0.00086: z += -675 × (0.050 − centroid_offset) × (-0.00086 − mean_phi)
if e2 < 0.0071: z += 81.84 × (0.0071 − e2)
if LHA < 0.155 and z_7 < 0.028: z += 1019 × (0.155 − LHA) × (0.028 − z_7)
if planar_flow < 0.253 and max_dr > 0.103: z += -9.34 × (0.253 − planar_flow) × (max_dr − 0.103)
if log_sum_pt < 6.70: z += -0.262 × (6.70 − log_sum_pt)
if centroid_offset < 0.050 and n_dr_0p05_0p1 > 2.00: z += 1.38 × (0.050 − centroid_offset) × (n_dr_0p05_0p1 − 2.00)
if e2_sq < 0.0064 and D2 < 1.33: z += -101 × (0.0064 − e2_sq) × (1.33 − D2)
if mass < 15.45: z += 0.013 × (15.45 − mass)
if centroid_offset > 0.014 and pt_5 < 29.88: z += 5.45 × (centroid_offset − 0.014) × (29.88 − pt_5)
if planar_flow < 0.253 and z_6 < 0.034: z += -83.11 × (0.253 − planar_flow) × (0.034 − z_6)
if width < 0.0036 and C2 > 0.031: z += 7516 × (0.0036 − width) × (C2 − 0.031)
if mass_top5 < 9.26 and eta_0 < -0.053: z += -9.08 × (9.26 − mass_top5) × (-0.053 − eta_0)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles, high pT** — 22.2% of jets, neuron 0.93. Mostly quarks (54%, with 33% gluons), 22.2% of jets: mass 8.0 GeV, width 0.0002, somewhat hard (sum pT 848 GeV), with 96.76% of the pT inside 0.025 of the axis. The compactness bonuses width < 0.00868 (+9.313), girth2 < 0.00668 (+4.341), centroid_offset < 0.0499 (+4.08) and girth2 < 0.0132 (+3.407) pass for all, but so do the very-narrow penalties girth < 0.0872 (-5.088), e2_sq < 0.00639 (-3.037), lam1 < 0.00838 (-2.638), e2 < 0.0445 (-1.994) and width < 0.00356 (-1.7). The neuron ends low (0.925, on for 93%) and adds only +0.347 to the W score; the formula splits them q 64% / g 35%.
- **medium-mass (53 GeV), average width, pT spread over several particles** — 18.1% of jets, neuron 4.68. Mostly W (52%, with 30% Z and 10% tops), 18.11% of jets: mass 53.5 GeV, width 0.0056, with 51.98% of the pT at 0.05-0.1 and 31.57% at 0.025-0.05 from the axis. width < 0.00868 (+3.34), centroid_offset < 0.0499 (99.8%, +3.328), girth2 < 0.0132 (+1.983) and the flat-shape test planar_flow < 0.253 (89.4%, +1.609) pass, and the narrow-jet penalties are small (girth < 0.0872 -1.359, lam1 < 0.00838 -0.931, centroid_offset < 0.0499 and log_sum_pt < 6.8 at 89.3%). The neuron reaches its highest value, 4.684 (on for 97.3%), and adds +1.757 to the W score; the formula calls them W (68%).
- **very light (16 GeV), very narrow, pT spread over several particles** — 15.6% of jets, neuron 2.22. A mixture of gluons (37%), quarks (24%), W (19%) and Z (15%), 15.61% of jets: mass 16.0 GeV, width 0.0011, with 48.72% of the pT inside 0.025 and 44.99% at 0.025-0.05 from the axis. width < 0.00868 (+8.282), girth2 < 0.00668 (+3.713), girth2 < 0.0132 (+3.161) and centroid_offset < 0.0499 (+2.875) outweigh girth < 0.0872 (-3.83), e2_sq < 0.00639 (-2.769), lam1 < 0.00838 (-2.358), e2 < 0.0445 (-1.709) and width < 0.00356 (-1.229). The neuron sits at 2.224 (on for 98.3%) and adds +0.834 to the W score; the formula splits them g 53% / q 19% / W 17%.
- **light (39 GeV), narrow, pT spread over several particles** — 13.4% of jets, neuron 3.37. Mostly W (39%, with 25% Z and 17% gluons), 13.43% of jets: mass 38.8 GeV, width 0.0035, with 56.13% of the pT at 0.025-0.05 from the axis. width < 0.00868 (+5.629), centroid_offset < 0.0499 (96.5%, +2.683), girth2 < 0.0132 (+2.529), girth2 < 0.00668 (+2.098) and planar_flow < 0.253 (80.6%, +1.302) outweigh girth < 0.0872 (-2.514), e2_sq < 0.00639 (-1.667), lam1 < 0.00838 (-1.603) and e2 < 0.0445 (-1.202). The neuron sits at 3.37 (on for 93.3%) and adds +1.264 to the W score; the formula calls them W (57%), with g 21%.
- **medium-mass (62 GeV), average width, pT spread over several particles** — 12.9% of jets, neuron 3.13. Mostly Z (59%, with 20% tops), 12.92% of jets: mass 61.6 GeV, width 0.0082, with 62.32% of the pT at 0.05-0.1 from the axis. centroid_offset < 0.0499 (all, +3.445), planar_flow < 0.253 (90.5%, +1.659), girth2 < 0.0132 (97%, +1.325), width < 0.00868 (78.3%, +1.015) and max_dr < 0.222 (88.6%) outweigh centroid_offset < 0.0499 and log_sum_pt < 6.8 (93.7%, -0.853), girth > 0.0761 (80.2%, -0.759) and girth < 0.0872 (74.1%). The neuron sits at 3.132 (on for 87.3%) and adds +1.174 to the W score; the formula calls them Z (70%), with t 20%.
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 4.8% of jets, neuron 0.00. Mostly tops (89%), 4.8% of jets: mass 79.9 GeV, width 0.024, soft (sum pT 543 GeV), with almost no pT inside 0.025 and 43.9% at 0.1-0.15 from the axis. The compactness tests width < 0.00868 and girth2 < 0.0132 never pass, and girth > 0.0761 (-7.241), girth > 0.102 (-1.933) and centroid_offset > 0.0144 (92.1%, -1.754) outweigh girth > 0.0761 and n_pt_above_50 < 7 (93.2%, +3.927) and centroid_offset < 0.0499 (69.5%). The neuron is off for essentially all of them and adds nothing to the scores; the formula calls them t (98%).
- **medium-mass (56 GeV), wide, pT spread over several particles, low pT** — 4.7% of jets, neuron 0.01. Mostly tops (66%) mixed with 21% gluons, 4.74% of jets: mass 55.7 GeV, width 0.0125, soft (sum pT 572 GeV), with 56.74% of the pT at 0.05-0.1 from the axis and an off-centre pT centroid. centroid_offset > 0.0144 passes for all (-2.213) and girth > 0.0761 for 95% (-2.316); girth > 0.0761 and n_pt_above_50 < 7 (84.4%, +1.222), the plain centroid_offset term (+0.979) and planar_flow < 0.253 (53.3%) do not make up for it, and width < 0.00868 passes for only 15.6%. The neuron is on for 2.4% and adds almost nothing; the formula calls them t (90%).
- **medium-mass (79 GeV), very wide, pT spread over several particles, low pT** — 4.6% of jets, neuron 0.03. Mostly tops (77%, with 16% gluons), 4.63% of jets: mass 79.5 GeV, width 0.0181, soft (sum pT 604 GeV), with pT mostly at 0.05-0.15 from the axis but a centred centroid. Unlike group 5 the centroid is centred, so centroid_offset < 0.0499 passes for 99.1% (+2.74), with girth > 0.0761 and n_pt_above_50 < 7 (79.2%, +1.891) and planar_flow < 0.253 (74.9%, +1.274); but girth > 0.0761 (-4.719), planar_flow < 0.253 and width > 0.0061 (74.9%, -1.412), centroid_offset < 0.0499 and log_sum_pt < 6.8 (-1.045) and girth > 0.102 keep it below zero. The neuron is on for 3.6% and adds almost nothing; the formula calls them t (96%).
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 1.8% of jets, neuron 0.00. Mostly tops (63%) mixed with 25% gluons, 1.8% of jets: mass 91.2 GeV, width 0.0294, soft (sum pT 556 GeV), with the pT spread from 0.1 to 0.3 from the axis. These very wide jets are also flat (planar_flow < 0.253 passes for all, +1.848), which here triggers the penalties planar_flow < 0.253 and girth2 > 0.0132 (-4.603) and planar_flow < 0.253 and width > 0.0061 (-4.243) on top of girth > 0.0761 (-8.942) and girth > 0.102 (-2.662), against girth > 0.0761 and n_pt_above_50 < 7 (91.8%, +5.06). The neuron is off for all of them and adds nothing to the scores; the formula calls them t (88%).
- **medium-mass (76 GeV), very wide, pT spread over several particles, low pT** — 1.8% of jets, neuron 0.01. Mostly tops (82%, with 14% gluons), 1.75% of jets: mass 75.7 GeV, width 0.0318, very soft (sum pT 447 GeV), with 36.2% of the pT at 0.15-0.2 from the axis. girth > 0.0761 (-9.888) is almost cancelled by girth > 0.0761 and n_pt_above_50 < 7 (all, +9.354), since few particles here exceed 50 GeV; girth > 0.102 (-3.067) and centroid_offset > 0.0144 (92.5%, -2.047) then keep it below zero. The neuron is on for 0.8% and adds almost nothing; the formula calls them t (96%).

### neuron 13: compactness (narrower than a top) (major)

- **What it measures:** Driven mostly by girth < 0.148 (its dominant term), lam1 < 0.0164 and width < 0.0132, all pushing it up, i.e. by the jet being narrower than a typical top; small e2 (< 0.0503) and very small width (< 0.00752) pull it down a little. It falls with lam1, width and girth2 (rank correlations -0.76 each); quarks (5.50), W (5.05), Z (4.64) and gluons (4.62) all sit high, tops far lower (1.40, zero for 47.7% of them).
- *computed — its value:* largest for q (5.50), then W (5.05), then Z (4.64), then g (4.62), then t (1.40); it separates t jets from the rest best (AUC 0.10: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-49%); it also raises the W score (+9%) and the Z score (+9%). It does not (or hardly) enter the g or q scores.
- *computed — used by:* raises the score of W (+9%), Z (+9%); lowers the score of t (-49%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.42
if girth < 0.148: z += 44.71 × (0.148 − girth)
if e2 < 0.050: z += -87.97 × (0.050 − e2)
if lam1 < 0.016: z += 194 × (0.016 − lam1)
if width < 0.013: z += 182 × (0.013 − width)
if width < 0.0075: z += -316 × (0.0075 − width)
if centroid_offset < 0.038: z += 43.35 × (0.038 − centroid_offset)
if lam2 < 0.00031 and centroid_offset < 0.050: z += -125115 × (0.00031 − lam2) × (0.050 − centroid_offset)
if girth < 0.148 and log_sum_pt < 6.80: z += -45.34 × (0.148 − girth) × (6.80 − log_sum_pt)
if LHA > 0.093: z += -5.28 × (LHA − 0.093)
if lam1 < 0.0065: z += 252 × (0.0065 − lam1)
if lam1 < 0.016 and centroid_offset < 0.038: z += -2314 × (0.016 − lam1) × (0.038 − centroid_offset)
if girth < 0.148 and pt_7 < 38.53: z += -0.884 × (0.148 − girth) × (38.53 − pt_7)
if sum_pt_top5 > 658 and pt_7 < 43.50: z += 0.00058 × (sum_pt_top5 − 658) × (43.50 − pt_7)
if tau21 < 0.501 and max_dr > 0.016: z += -12.59 × (0.501 − tau21) × (max_dr − 0.016)
if lam1 < 0.016 and pt_6 < 56.53: z += -1.80 × (0.016 − lam1) × (56.53 − pt_6)
if lam2 < 0.00031: z += 1490 × (0.00031 − lam2)
if sum_pt > 988: z += -0.041 × (sum_pt − 988)
if sum_pt_top5 > 840: z += 0.020 × (sum_pt_top5 − 840)
if C2 > 0.067: z += -55.09 × (C2 − 0.067)
if width < 0.0036: z += -125 × (0.0036 − width)
if pt_7 < 25.58: z += -0.105 × (25.58 − pt_7)
if e2 < 0.050 and pt_dispersion > 0.397: z += 82.25 × (0.050 − e2) × (pt_dispersion − 0.397)
if sum_pt < 764: z += -0.0013 × (764 − sum_pt)
if z_7 < 0.028: z += -94.49 × (0.028 − z_7)
if mass > 53.33: z += -0.018 × (mass − 53.33)
if pt_6 < 31.91: z += -0.044 × (31.91 − pt_6)
if sum_pt_top5 > 658 and z_7 > 0.023: z += -0.268 × (sum_pt_top5 − 658) × (z_7 − 0.023)
if pt_5 < 24.58: z += -0.211 × (24.58 − pt_5)
if z_top5 > 0.877: z += -8.29 × (z_top5 − 0.877)
if z_6 < 0.067: z += 4.47 × (0.067 − z_6)
if sum_pt_top5 > 658: z += 0.0012 × (sum_pt_top5 − 658)
if sum_pt > 988 and D2 < 3.89: z += 0.0061 × (sum_pt − 988) × (3.89 − D2)
if e2 < 0.050 and n_pt_above_50 < 5.00: z += 3.07 × (0.050 − e2) × (5.00 − n_pt_above_50)
if sum_pt > 988 and n_pt_above_50 > 6.00: z += 0.014 × (sum_pt − 988) × (n_pt_above_50 − 6.00)
if girth > 0.102: z += 4.39 × (girth − 0.102)
if lam1 < 0.0065 and z_dr_0p05_0p1 > 0.164: z += 154 × (0.0065 − lam1) × (z_dr_0p05_0p1 − 0.164)
if girth > 0.102 and pt_balance01 < 0.476: z += -53.38 × (girth − 0.102) × (0.476 − pt_balance01)
if girth > 0.102 and pt_1 < 116: z += -0.140 × (girth − 0.102) × (116 − pt_1)
if width < 0.0075 and m012 > 16.90: z += -3.94 × (0.0075 − width) × (m012 − 16.90)
if sum_pt_top5 > 902 and D2 < 3.89: z += -0.0029 × (sum_pt_top5 − 902) × (3.89 − D2)
if sum_pt_top5 > 902: z += 0.0035 × (sum_pt_top5 − 902)
if sum_pt_top5 > 902 and n_pt_above_50 > 6.00: z += -0.013 × (sum_pt_top5 − 902) × (n_pt_above_50 − 6.00)
if sum_pt_top5 > 840 and n_pt_above_50 > 2.00: z += 0.00059 × (sum_pt_top5 − 840) × (n_pt_above_50 − 2.00)
if width < 0.0036 and n_pt_above_50 > 7.00: z += 95.58 × (0.0036 − width) × (n_pt_above_50 − 7.00)
if sum_pt_top5 < 531 and pt_5 < 29.88: z += 0.0014 × (531 − sum_pt_top5) × (29.88 − pt_5)
if z_top5_slots > 0.931: z += 9.68 × (z_top5_slots − 0.931)
if girth > 0.102 and pt_4 > 81.38: z += -4.81 × (girth − 0.102) × (pt_4 − 81.38)
if sum_pt_top5 > 658 and tau32 < 0.363: z += -0.023 × (sum_pt_top5 − 658) × (0.363 − tau32)
if sum_pt < 764 and z_4 < 0.037: z += 3.53 × (764 − sum_pt) × (0.037 − z_4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (56 GeV), average width, pT spread over several particles** — 23.8% of jets, neuron 3.96. Mostly Z (40%, with 30% W and 18% tops), 23.85% of jets: mass 56.3 GeV, width 0.0074, with 66.46% of the pT at 0.05-0.1 from the axis. The compactness tests girth < 0.148 (+3.11), lam1 < 0.0164 (+1.804), width < 0.0132 (99.1%, +1.068) and centroid_offset < 0.0378 (93.4%, +1.041) pass, against LHA > 0.0932 (all, -1.132), lam2 < 0.000306 and centroid_offset < 0.0499 (81.1%, -0.943), girth < 0.148 and log_sum_pt < 6.8 (97%, -0.934) and e2 < 0.0503 (95.6%, -0.91). The neuron sits at 3.961 (on for 98.7%), lowering the t score (-1.609) and raising the W and Z scores a little; the formula splits them Z 43% / W 37% / t 18%.
- **very light (10 GeV), very narrow, pT spread over several particles** — 17.2% of jets, neuron 6.56. A quark-gluon mixture (quarks 41%, gluons 37%), 17.18% of jets: mass 10.1 GeV, width 0.0004, somewhat hard (sum pT 813 GeV), with 85.88% of the pT inside 0.025 of the axis. All compactness bonuses pass at full size: girth < 0.148 (+5.994), lam1 < 0.0164 (+3.119), width < 0.0132 (+2.338), lam1 < 0.00651 (+1.553) and centroid_offset < 0.0378 (+1.264); the very-narrow penalties e2 < 0.0503 (-3.916), width < 0.00752 (-2.255) and lam2 < 0.000306 and centroid_offset < 0.0499 (-1.438) take back less. The neuron is high (6.562, on for all), lowering the t score (-2.666) and raising the W (+0.461) and Z scores; the formula splits them q 49% / g 38%.
- **light (41 GeV), narrow, pT spread over several particles** — 16.7% of jets, neuron 4.53. Mostly W (38%) with 27% Z, 15% gluons and 12% tops, 16.71% of jets: mass 41.0 GeV, width 0.0043, with 52.6% of the pT at 0.025-0.05 from the axis. girth < 0.148 (+4.182), lam1 < 0.0164 (+2.399), width < 0.0132 (+1.633) and centroid_offset < 0.0378 (91.1%) outweigh e2 < 0.0503 (-2.313), girth < 0.148 and log_sum_pt < 6.8 (96.6%, -1.178), width < 0.00752 (99.1%, -1.033) and LHA > 0.0932 (-0.841). The neuron sits at 4.534 (on for 99.5%), lowering the t score (-1.842) and raising the W and Z scores; the formula calls them W (54%).
- **medium-mass (79 GeV), very wide, pT spread over several particles, low pT** — 15.4% of jets, neuron 0.27. Mostly tops (80%, with 14% gluons), 15.41% of jets: mass 78.6 GeV, width 0.0224, soft (sum pT 559 GeV), with 34.71% of the pT at 0.1-0.15 from the axis. These jets fail most compactness tests (width < 0.0132 passes for 6.3%, lam1 < 0.0164 for 36.6%, girth < 0.148 for 68.7% at +0.807), and LHA > 0.0932 (-1.627), C2 > 0.0673 (50.9%, -0.896), tau21 < 0.501 and max_dr > 0.0156 (80.7%, -0.587) and mass > 53.3 (90.1%) push down. The neuron is on for 26.7% (mean 0.265) and lowers the t score only slightly (-0.108), so the lack of this neuron's usual push against t is what matters; the formula calls them t (96%).
- **very light (13 GeV), very narrow, pT spread over several particles, low pT** — 11.8% of jets, neuron 4.27. Mostly gluons (50%, with 20% quarks), 11.78% of jets: mass 12.7 GeV, width 0.0012, soft (sum pT 578 GeV, leading particle 153.57 GeV), with 47.03% of the pT inside 0.025 and 43.98% at 0.025-0.05 from the axis. girth < 0.148 (+5.353), lam1 < 0.0164 (+2.986), width < 0.0132 (+2.198) and lam1 < 0.00651 (+1.38) outweigh e2 < 0.0503 (-3.444), width < 0.00752 (-2.011) and, because they are soft, girth < 0.148 and log_sum_pt < 6.8 (all, -2.474). The neuron sits at 4.273 (on for 98.9%), lowering the t score (-1.736) and raising the W and Z scores; the formula calls them g (82%).
- **very light (14 GeV), very narrow, leading particle 46% of pT, high pT** — 6.7% of jets, neuron 5.96. Mostly quarks (63%, with 16% W and 11% Z), 6.69% of jets: mass 13.8 GeV, width 0.0005, hard (sum pT 911 GeV, leading particle 416.55 GeV), with 90.46% of the pT inside 0.025 of the axis. girth < 0.148 (+6.067), lam1 < 0.0164 (+3.095), width < 0.0132 (+2.315) and the hard-core test sum_pt_top5 > 658 and pt_7 < 43.5 (99.2%, +2.388) outweigh e2 < 0.0503 (-3.965), girth < 0.148 and pt_7 < 38.5 (-2.314), width < 0.00752 (-2.214) and lam2 < 0.000306 and centroid_offset < 0.0499 (99.1%, -1.5). The neuron is high (5.964, on for 98.1%), lowering the t score (-2.423) and raising the W and Z scores; the formula calls them q (84%).
- **medium-mass (64 GeV), average width, leading particle 49% of pT, high pT** — 4.4% of jets, neuron 4.68. An even Z/W mixture (Z 43.27%, W 42.89%), 4.41% of jets: mass 64.3 GeV, width 0.0053, hard (sum pT 908 GeV) with a dominant leading particle (448.99 GeV), 58.52% of the pT at 0.025-0.05 from the axis. girth < 0.148 (+4.138), sum_pt_top5 > 658 and pt_7 < 43.5 (99.8%, +2.241), lam1 < 0.0164 (+2.181), width < 0.0132 (+1.449) and centroid_offset < 0.0378 (+1.187) outweigh e2 < 0.0503 (-2.105), girth < 0.148 and pt_7 < 38.5 (-1.455) and lam2 < 0.000306 and centroid_offset < 0.0499 (95.1%, -1.22). The neuron sits at 4.684 (on for 98.5%), lowering the t score (-1.903) and raising the W and Z scores; the formula splits them W 55% / Z 40%.
- **very light (19 GeV), very narrow, leading particle 55% of pT, high pT** — 2.7% of jets, neuron 6.07. Mostly quarks (71%), 2.73% of jets: mass 18.7 GeV, width 0.0007, very hard (sum pT 1008 GeV) with a dominant leading particle (557.64 GeV), 92.14% of the pT inside 0.025 of the axis. The hard-core tests add a lot, sum_pt_top5 > 658 and pt_7 < 43.5 (+5.883) and sum_pt_top5 > 840 (99.7%, +2.452), on top of girth < 0.148 (+6.155), lam1 < 0.0164 (+3.058) and width < 0.0132 (+2.279); e2 < 0.0503 (-3.943), girth < 0.148 and pt_7 < 38.5 (-3.488), width < 0.00752 (-2.153) and z_7 < 0.0281 (-1.724) take back part. The neuron is high (6.069, on for 99.3%), lowering the t score (-2.466); the formula calls them q (87%).
- **light (27 GeV), very narrow, leading particle 42% of pT, high pT** — 1.0% of jets, neuron 7.63. Mostly gluons (50%, with 25% quarks), 1.01% of jets: mass 27.0 GeV, width 0.0013, very hard (sum pT 1164 GeV, leading particle 484.12 GeV), with 73.65% of the pT inside 0.025 of the axis. sum_pt > 988 passes for all (-7.15) but is more than offset by girth < 0.148 (99.8%, +5.709), sum_pt_top5 > 840 (98.7%, +3.552), lam1 < 0.0164 (+2.946), sum_pt_top5 > 658 and pt_7 < 43.5 (67.2%), width < 0.0132 (+2.178) and sum_pt > 988 and D2 < 3.89 (91.6%, +2.133), with e2 < 0.0503 (-3.587). The neuron reaches its highest value, 7.63 (on for 99.2%), lowering the t score (-3.1) and raising the W (+0.536) and Z scores; the formula calls them g (58%).
- **light (30 GeV), very narrow, leading particle 47% of pT, high pT** — 0.2% of jets, neuron 6.82. Mostly gluons (56%, with 24% quarks), 0.24% of jets: mass 29.9 GeV, width 0.0012, the hardest group (sum pT 1439 GeV, leading particle 682.63 GeV), with 77.9% of the pT inside 0.025 of the axis. sum_pt > 988 is huge here (-18.352) but is balanced by sum_pt_top5 > 840 (+8.88), girth < 0.148 (98.6%, +5.815), sum_pt > 988 and n_pt_above_50 > 6 (49.3%, +5.361), sum_pt > 988 and D2 < 3.89 (85.9%, +5.163) and sum_pt_top5 > 658 and pt_7 < 43.5 (55.6%, +4.0), less e2 < 0.0503 (-3.697) and sum_pt_top5 > 902 and n_pt_above_50 > 6 (-3.508). The neuron is high (6.822, on for 95.8%), lowering the t score (-2.772); the formula calls them g (77%).

### neuron 14: Z-sized width and mass (major)

- **What it measures:** Pushed down for the narrowest jets (width < 0.00752, its strongest term; also girth < 0.0872) but up for width < 0.00868 and girth2 < 0.0132, so it peaks for jets a bit wider than a typical W; it is also pushed down for small mass/pT (its square < 0.0117), favouring heavier jets; it rises with eccentricity and the number of particles above 10 GeV (rank correlations 0.351 each). Z jets sit highest (1.17), well above tops (0.32), gluons (0.16), W (0.15) and quarks (0.12).
- *computed — its value:* largest for Z (1.17), then t (0.32), then g (0.16), then W (0.15), then q (0.12); it separates Z jets from the rest best (AUC 0.78: large for Z)
- **How the class scores use it:** It is the W/Z separator: Z jets sit high and W jets low on it, so it raises the Z score (+6%) and lowers the W score (-8%). It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+6%); lowers the score of W (-8%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.979
if width < 0.0075: z += -1760 × (0.0075 − width)
if girth2 < 0.013: z += 637 × (0.013 − girth2)
if width < 0.0087: z += 1050 × (0.0087 − width)
if girth < 0.087: z += -85.57 × (0.087 − girth)
if mass_over_sum_pt_sq < 0.012: z += -412 × (0.012 − mass_over_sum_pt_sq)
if e2 < 0.038: z += 133 × (0.038 − e2)
if width < 0.0075 and n_dr_0p1_0p2 < 3.00: z += 166 × (0.0075 − width) × (3.00 − n_dr_0p1_0p2)
if max_dr < 0.177: z += -20.02 × (0.177 − max_dr)
if lam1 > 0.0025: z += 291 × (lam1 − 0.0025)
if e2_sq < 0.017: z += 101 × (0.017 − e2_sq)
if z_dr_0p05_0p1 < 0.588 and C2 < 0.067: z += -58.71 × (0.588 − z_dr_0p05_0p1) × (0.067 − C2)
if girth2 < 0.013 and n_dr_0p1_0p2 < 3.00: z += -46.84 × (0.013 − girth2) × (3.00 − n_dr_0p1_0p2)
if z_dr_0p05_0p1 < 0.588: z += 1.74 × (0.588 − z_dr_0p05_0p1)
if lam1 > 0.0042: z += -195 × (lam1 − 0.0042)
if e2 < 0.036: z += -39.64 × (0.036 − e2)
if width < 0.0087 and D2 < 1.00: z += -1487 × (0.0087 − width) × (1.00 − D2)
if lam1 > 0.012: z += -393 × (lam1 − 0.012)
if mass < 76.66: z += -0.012 × (76.66 − mass)
if lam1 > 0.006: z += 166 × (lam1 − 0.006)
if lam1 > 0.0054: z += -153 × (lam1 − 0.0054)
if n_dr_0p05_0p1 < 5.00: z += -0.128 × (5.00 − n_dr_0p05_0p1)
if z_dr_0p05_0p1 < 0.588 and n_dr_0p1_0p2 < 3.00: z += 0.387 × (0.588 − z_dr_0p05_0p1) × (3.00 − n_dr_0p1_0p2)
if girth2 < 0.013 and eccentricity > 0.970: z += 4904 × (0.013 − girth2) × (eccentricity − 0.970)
if width < 0.0075 and planar_flow < 0.112: z += -4437 × (0.0075 − width) × (0.112 − planar_flow)
if width < 0.0075 and D2 < 1.00: z += 1262 × (0.0075 − width) × (1.00 − D2)
if LHA < 0.303: z += -3.40 × (0.303 − LHA)
if D2 < 1.12: z += -1.08 × (1.12 − D2)
if planar_flow < 0.112 and centroid_offset > 0.0095: z += 893 × (0.112 − planar_flow) × (centroid_offset − 0.0095)
if girth2 < 0.013 and D2 < 1.00: z += 161 × (0.013 − girth2) × (1.00 − D2)
if girth2 < 0.0044 and n_dr_0p2_0p4 < 1.00: z += -94.65 × (0.0044 − girth2) × (1.00 − n_dr_0p2_0p4)
if planar_flow < 0.112 and centroid_offset > 0.018: z += -839 × (0.112 − planar_flow) × (centroid_offset − 0.018)
if planar_flow < 0.112 and max_dr < 0.160: z += -118 × (0.112 − planar_flow) × (0.160 − max_dr)
if D2 < 1.12 and centroid_offset < 0.031: z += 27.79 × (1.12 − D2) × (0.031 − centroid_offset)
if width < 0.0075 and log_sum_pt < 6.80: z += 159 × (0.0075 − width) × (6.80 − log_sum_pt)
if lam1 > 0.0084: z += -57.70 × (lam1 − 0.0084)
if D2 < 0.746: z += 1.05 × (0.746 − D2)
if z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 1.00: z += 4.77 × (z_dr_0p05_0p1 − 0.751) × (1.00 − n_dr_0p2_0p4)
if max_dr < 0.081: z += -6.39 × (0.081 − max_dr)
if centroid_offset > 0.050: z += -119 × (centroid_offset − 0.050)
if z_dr_0p05_0p1 > 0.751: z += -3.87 × (z_dr_0p05_0p1 − 0.751)
if planar_flow < 0.112: z += 2.54 × (0.112 − planar_flow)
if z_dr_0p05_0p1 < 0.588 and eccentricity > 0.970: z += 35.62 × (0.588 − z_dr_0p05_0p1) × (eccentricity − 0.970)
if girth2 < 0.0044 and D2 < 0.876: z += -5044 × (0.0044 − girth2) × (0.876 − D2)
if mass < 76.66 and D2 < 0.746: z += -0.048 × (76.66 − mass) × (0.746 − D2)
if lam1 > 0.0054 and max_dr < 0.160: z += 5461 × (lam1 − 0.0054) × (0.160 − max_dr)
if planar_flow < 0.112 and sum_pt < 740: z += -0.024 × (0.112 − planar_flow) × (740 − sum_pt)
if e2 < 0.038 and D2 < 1.00: z += 57.56 × (0.038 − e2) × (1.00 − D2)
if mass < 76.66 and n_dr_0p1_0p2 > 4.00: z += 0.013 × (76.66 − mass) × (n_dr_0p1_0p2 − 4.00)
if lam1 > 0.0073 and max_dr < 0.133: z += 22721 × (lam1 − 0.0073) × (0.133 − max_dr)
if planar_flow < 0.112 and sum_pt > 840: z += -0.024 × (0.112 − planar_flow) × (sum_pt − 840)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles** — 27.1% of jets, neuron 0.00. Mostly quarks (48%, with 34% gluons), 27.05% of jets: mass 7.9 GeV, width 0.0002, somewhat hard (sum pT 821 GeV), with 92.02% of the pT inside 0.025 of the axis. The narrowest-jet penalty width < 0.00752 (-12.845) together with girth < 0.0872 (-6.479), mass_over_sum_pt_sq < 0.0117 (-4.753) and max_dr < 0.177 (99.4%, -2.895) beats width < 0.00868 (+8.88), girth2 < 0.0132 (+8.289), e2 < 0.0385 (+4.475) and width < 0.00752 and n_dr_0p1_0p2 < 3 (+3.607). The neuron is off for all of them and adds nothing to the scores; the formula splits them q 54% / g 40%.
- **medium-mass (58 GeV), average width, pT spread over several particles** — 14.1% of jets, neuron 1.72. Mostly Z (60%, with 18% tops), 14.13% of jets: mass 58.0 GeV, width 0.0077, with 62.51% of the pT at 0.05-0.1 from the axis. Being a bit wider than a typical W, only 47.7% pay width < 0.00752 (-0.479), while girth2 < 0.0132 (+3.516), lam1 > 0.00246 (+1.451), width < 0.00868 (84.5%, +1.122) and e2_sq < 0.0172 (+1.001) all pass; mass_over_sum_pt_sq < 0.0117 (-1.811), max_dr < 0.177 (82.4%) and width < 0.00868 and D2 < 1 (70.9%, -0.656) take back part. The neuron sits at its highest, 1.72 (on for 94.6%), lowering the W score (-1.29) and raising the Z score (+0.645); the formula calls them Z (70%).
- **medium-mass (55 GeV), average width, pT spread over several particles** — 10.1% of jets, neuron 0.15. Mostly W (68%, with 19% Z), 10.12% of jets: mass 54.9 GeV, width 0.0056, with almost no pT inside 0.025 and 71.21% at 0.05-0.1 from the axis, two well-separated prongs. girth2 < 0.0132 (+4.864), width < 0.00868 (+3.232), width < 0.00752 and D2 < 1 (+1.386) and e2_sq < 0.0172 are cancelled by width < 0.00752 (-3.379), the clean-two-prong penalty width < 0.00868 and D2 < 1 (all, -2.636), mass_over_sum_pt_sq < 0.0117 (-2.568), max_dr < 0.177 and girth < 0.0872 (-1.386). The neuron is on for 25.2% (mean 0.154), lowering the W score only slightly; the formula calls them W (89%).
- **very light (19 GeV), very narrow, pT spread over several particles** — 10.0% of jets, neuron 0.01. A mixture of gluons (35%), quarks (26%), W (18%) and Z (16%), 9.98% of jets: mass 19.2 GeV, width 0.0014, with 38.94% of the pT inside 0.025 and 51.95% at 0.025-0.05 from the axis. width < 0.00752 (-10.795), girth < 0.0872 (-4.783) and mass_over_sum_pt_sq < 0.0117 (-4.448) outweigh width < 0.00868 (+7.657), girth2 < 0.0132 (+7.547), e2 < 0.0385 (+3.611) and width < 0.00752 and n_dr_0p1_0p2 < 3 (99.6%, +2.789). The neuron is on for 1.3% and adds almost nothing; the formula calls them g (49%).
- **light (49 GeV), average width, pT spread over several particles** — 9.6% of jets, neuron 0.76. An even W/Z mixture (W 37%, Z 36%, gluons and tops 11% each), 9.58% of jets: mass 48.5 GeV, width 0.0052, with 49.83% of the pT at 0.025-0.05 and 31.34% at 0.05-0.1 from the axis. girth2 < 0.0132 (+5.115), width < 0.00868 (+3.647), e2 < 0.0385 (94.4%, +1.442), e2_sq < 0.0172 (+1.259) and lam1 > 0.00246 (+0.727) outweigh width < 0.00752 (-4.074), mass_over_sum_pt_sq < 0.0117 (-2.862) and girth < 0.0872 (-2.401); the two-prong penalty width < 0.00868 and D2 < 1 passes for only 38.5%. The neuron is on for 64.5% (mean 0.757), lowering the W score (-0.568) and raising the Z score (+0.284); the formula splits them W 49% / Z 37%.
- **light (39 GeV), narrow, pT spread over several particles** — 9.5% of jets, neuron 0.27. Mostly W (39%, with 26% Z and 17% gluons), 9.5% of jets: mass 38.6 GeV, width 0.0034, with 59.79% of the pT at 0.025-0.05 from the axis. width < 0.00752 (-7.337), mass_over_sum_pt_sq < 0.0117 (-3.647) and girth < 0.0872 (-3.555) nearly cancel girth2 < 0.0132 (+6.296), width < 0.00868 (+5.593), e2 < 0.0385 (+2.632), width < 0.00752 and n_dr_0p1_0p2 < 3 (95.2%) and e2_sq < 0.0172 (+1.451). The neuron is on for 32.4% (mean 0.266), lowering the W score a little; the formula calls them W (60%).
- **medium-mass (68 GeV), very wide, pT spread over several particles, low pT** — 8.5% of jets, neuron 0.34. Mostly tops (75%, with 16% gluons), 8.48% of jets: mass 67.8 GeV, width 0.0153, soft (sum pT 586 GeV), with pT mostly at 0.05-0.15 from the axis. The width tests no longer pass; the value comes from a ladder of lam1 cuts, lam1 > 0.00246 (+3.11) and lam1 > 0.00595 (+1.193) against lam1 > 0.00418 (-1.75), lam1 > 0.00543 (-1.179) and lam1 > 0.012 (65.6%, -0.646). The neuron is on for 34.6% (mean 0.336), lowering the W score and raising the Z score a little; the formula calls them t (95%).
- **medium-mass (82 GeV), very wide, pT spread over several particles, low pT** — 6.6% of jets, neuron 0.00. Mostly tops (84%), 6.63% of jets: mass 82.3 GeV, width 0.0239, soft (sum pT 556 GeV), with the pT spread from 0.05 to 0.3 from the axis. All the lam1 steps pass, and the penalties lam1 > 0.012 (-3.503), lam1 > 0.00418 (-3.265), lam1 > 0.00543 (-2.364) and lam1 > 0.00838 (-0.724) outweigh lam1 > 0.00246 (+5.368) and lam1 > 0.00595 (+2.479). The neuron is off for essentially all of them and adds nothing to the scores; the formula calls them t (98%).
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.5% of jets, neuron 0.00. Mostly tops (72%, with 20% gluons), 2.49% of jets: mass 90.8 GeV, width 0.0335, soft (sum pT 518 GeV), with 29.78% of the pT at 0.15-0.2 and 30.17% at 0.2-0.3 from the axis. The same lam1 ladder at larger size: lam1 > 0.00246 (+8.396) and lam1 > 0.00595 (+4.203) against lam1 > 0.012 (-7.592), lam1 > 0.00418 (-5.295) and lam1 > 0.00543 (-3.953). The neuron is off for all of them and adds nothing to the scores; the formula calls them t (90%).
- **light (32 GeV), narrow, pT spread over several particles** — 2.0% of jets, neuron 0.00. A mixture of gluons (33%), quarks (25%), W (23%) and Z (12%), 2.03% of jets: mass 32.4 GeV, width 0.0022, with 54.29% of the pT at 0.025-0.05 and 29.82% at 0.05-0.1 from the axis. On top of width < 0.00752 (-9.332), mass_over_sum_pt_sq < 0.0117 (-3.965) and girth < 0.0872 (-3.769), the low-D2 tests width < 0.00868 and D2 < 1 (-4.109) and girth2 < 0.00437 and D2 < 0.876 (-3.047) pass for all, outweighing girth2 < 0.0132 (+7.018), width < 0.00868 (+6.784) and width < 0.00752 and D2 < 1 (+2.848). The neuron is off for all of them and adds nothing to the scores; the formula splits them g 48% / W 27% / q 24%.

### neuron 0: compact, massive, elongated two-prong jet (moderate)

- **What it measures:** Pushed up for compact jets (girth2 < 0.0132 and width < 0.00868 are its two strongest terms) and pushed down for small mass/pT (< 0.131), for the very narrowest jets (girth < 0.0761) and for light jets (mass < 21.8 GeV and < 64.6 GeV); it rises with eccentricity and mass and falls with planar flow and τ21 (rank correlations 0.576, 0.408, -0.576, -0.569). Z (1.66) and W (1.62) jets sit highest, tops lower (0.40), gluons (0.20) and quarks (0.17) lowest.
- *computed — its value:* largest for Z (1.66), then W (1.62), then t (0.40), then g (0.20), then q (0.17); it separates Z jets from the rest best (AUC 0.76: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, it raises the W score (+8%) and lowers the g score (-5%). It does not (or hardly) enter the q, Z or t scores; although Z jets sit highest on it, the Z score relies on the related neuron 7 instead.
- *computed — used by:* raises the score of W (+8%); lowers the score of g (-5%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.633
if girth2 < 0.013: z += 424 × (0.013 − girth2)
if width < 0.0087: z += 778 × (0.0087 − width)
if mass_over_sum_pt < 0.131: z += -20.65 × (0.131 − mass_over_sum_pt)
if girth < 0.076: z += -47.25 × (0.076 − girth)
if mass_over_sum_pt_sq < 0.0082: z += -276 × (0.0082 − mass_over_sum_pt_sq)
if lam1 < 0.0084: z += -246 × (0.0084 − lam1)
if girth < 0.087: z += -24.46 × (0.087 − girth)
if mass < 21.78: z += -0.195 × (21.78 − mass)
if girth2 < 0.019: z += 64.92 × (0.019 − girth2)
if width < 0.0044: z += -476 × (0.0044 − width)
if girth2_top3 < 0.0079: z += -156 × (0.0079 − girth2_top3)
if mass < 64.62: z += -0.025 × (64.62 − mass)
if lam1 < 0.0065: z += 235 × (0.0065 − lam1)
if lam1 < 0.0005: z += -6921 × (0.0005 − lam1)
if mass < 29.64: z += -0.077 × (29.64 − mass)
if lam1 < 0.0054: z += -239 × (0.0054 − lam1)
if girth2_top5 < 0.0083: z += -108 × (0.0083 − girth2_top5)
if mass < 56.92: z += -0.022 × (56.92 − mass)
if z_dr_0_0p05 > 0.848: z += 8.72 × (z_dr_0_0p05 − 0.848)
if mass < 29.64 and phi_1 > -0.059: z += -0.968 × (29.64 − mass) × (phi_1 − -0.059)
if centroid_offset < 0.031: z += 24.97 × (0.031 − centroid_offset)
if girth2_top3 < 0.004: z += 207 × (0.004 − girth2_top3)
if mass < 64.62 and pt_7 < 40.04: z += -0.0015 × (64.62 − mass) × (40.04 − pt_7)
if e2 < 0.020: z += 62.34 × (0.020 − e2)
if girth2 < 0.019 and eccentricity > 0.960: z += 1921 × (0.019 − girth2) × (eccentricity − 0.960)
if mass < 64.62 and centroid_offset > 0.013: z += 1.97 × (64.62 − mass) × (centroid_offset − 0.013)
if sum_pt > 902: z += -0.022 × (sum_pt − 902)
if sum_pt_top5 > 687: z += 0.0072 × (sum_pt_top5 − 687)
if log_sum_pt > 6.64: z += -2.70 × (log_sum_pt − 6.64)
if log_sum_pt > 6.38: z += 0.598 × (log_sum_pt − 6.38)
if sum_pt_top5 > 687 and z_7 > 0.023: z += -0.593 × (sum_pt_top5 − 687) × (z_7 − 0.023)
if log_sum_pt > 6.64 and dr_4 < 0.072: z += 44.68 × (log_sum_pt − 6.64) × (0.072 − dr_4)
if sum_pt > 902 and pt_7 < 25.58: z += 0.0013 × (sum_pt − 902) × (25.58 − pt_7)
if girth2 < 0.013 and D2 < 1.00: z += -76.22 × (0.013 − girth2) × (1.00 − D2)
if planar_flow < 0.148: z += 1.44 × (0.148 − planar_flow)
if mass < 56.92 and C2 > 0.024: z += 0.972 × (56.92 − mass) × (C2 − 0.024)
if planar_flow < 0.148 and centroid_offset < 0.050: z += -39.60 × (0.148 − planar_flow) × (0.050 − centroid_offset)
if mass < 29.64 and D2 < 0.876: z += -3.08 × (29.64 − mass) × (0.876 − D2)
if sum_pt > 902 and pt_7 > 29.04: z += -0.00067 × (sum_pt − 902) × (pt_7 − 29.04)
if planar_flow < 0.148 and max_dr < 0.112: z += -130 × (0.148 − planar_flow) × (0.112 − max_dr)
if girth2_top3 < 0.0079 and pt_6 < 35.28: z += -3.12 × (0.0079 − girth2_top3) × (35.28 − pt_6)
if lam1 < 0.0065 and D2 < 0.876: z += -611 × (0.0065 − lam1) × (0.876 − D2)
if planar_flow < 0.148 and sum_pt_top2 < 381: z += -0.014 × (0.148 − planar_flow) × (381 − sum_pt_top2)
if z_dr_0p05_0p1 > 0.846: z += 3.08 × (z_dr_0p05_0p1 − 0.846)
if width < 0.0044 and D2 < 0.746: z += -4251 × (0.0044 − width) × (0.746 − D2)
if girth2 < 0.019 and phi_0 > 0.021: z += 494 × (0.019 − girth2) × (phi_0 − 0.021)
if sum_pt > 902 and dr_2 < 0.038: z += -0.101 × (sum_pt − 902) × (0.038 − dr_2)
if girth2 < 0.013 and centroid_offset > 0.014: z += -708 × (0.013 − girth2) × (centroid_offset − 0.014)
if planar_flow < 0.013: z += 41.20 × (0.013 − planar_flow)
if girth2 < 0.019 and mass_top2 > 28.79: z += -2.74 × (0.019 − girth2) × (mass_top2 − 28.79)
if planar_flow < 0.148 and dr_2 < 0.028: z += -249 × (0.148 − planar_flow) × (0.028 − dr_2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (52 GeV), average width, pT spread over several particles** — 18.5% of jets, neuron 2.07. A W/Z mixture (W 52%, Z 29%, tops 10%), 18.46% of jets: mass 52.4 GeV, width 0.0055 and average pT, with the pT sitting 0.025-0.1 from the axis rather than in the core, as two close prongs would. The two compactness tests girth2 < 0.0132 (+3.267) and width < 0.00868 (+2.45) always pass, helped by girth2 < 0.0188 and eccentricity > 0.96 (81.5% pass); the lightness penalties mass_over_sum_pt < 0.131 (-1.217), mass_over_sum_pt_sq < 0.00817 and lam1 < 0.00838 take back much less than in the lighter groups, and mass < 21.8 almost never passes. The neuron sits at 2.07 and is on for 97.1%, which raises the W score (+0.712) and lowers the g score; the formula calls them W (68%).
- **medium-mass (76 GeV), very wide, pT spread over several particles, low pT** — 17.6% of jets, neuron 0.10. Mostly tops (78%, plus 15% gluons), 17.58% of jets: mass 76.4 GeV, width 0.0211 (over three times the average), softer than average (sum pT 564 GeV) and spread out, with most of the pT beyond 0.1 from the axis. They are too wide for width < 0.00868 (never passes) and mostly for girth2 < 0.0132 (16.5%), so the two big bonuses are missing; the leftover pieces, mass < 64.6 and centroid_offset > 0.0126 (27.2%, +0.247) and centroid_offset < 0.0312 (49.6%) against mass_over_sum_pt < 0.131 (-0.179), nearly cancel. The neuron stays at 0.101 (on for 16.4%) and hardly moves any score; the formula calls them t (96%) on the strength of other neurons.
- **very light (7 GeV), very narrow, pT spread over several particles** — 17.4% of jets, neuron 0.00. Mostly quarks (49%) with 37% gluons, 17.43% of jets: mass 6.7 GeV, width 0.0001, slightly harder than average, with 96.68% of the pT inside 0.025 of the axis. The compactness bonuses are at their largest (width < 0.00868 +6.632, girth2 < 0.0132 +5.546), but every lightness and narrowness penalty passes too: girth < 0.0761 (-3.128), mass < 21.8 (-2.946), lam1 < 0.000505 (-2.636), mass_over_sum_pt < 0.131 (-2.524) and more, and together they win. The neuron is off for all of them and adds nothing to any score; the formula splits them q 56% / g 42%.
- **medium-mass (59 GeV), average width, pT spread over several particles** — 13.6% of jets, neuron 1.96. Mostly Z (59%) with 18% tops and 11% W, 13.62% of jets: mass 58.9 GeV, width 0.0078, a little softer than average, with 64.3% of the pT at 0.05-0.1 from the axis, so the prongs sit further apart than in group 0. girth2 < 0.0132 (+2.322) always passes, but width < 0.00868 passes for only 83.9% and adds less (+0.799); girth2 < 0.0188 (+0.719), girth2 < 0.0188 and eccentricity > 0.96 (82.6%) and centroid_offset < 0.0312 (89.4%, +0.448) fill in, while the lightness penalties are small because the jets are heavy. The neuron sits at 1.963 (on for 94.9%), raising the W score and lowering the g score just as for group 0; the formula calls them Z (70%), so the W-Z split is left to other neurons.
- **light (40 GeV), narrow, pT spread over several particles** — 12.3% of jets, neuron 1.12. Mostly W (40%) with 25% Z, 16% gluons and 11% quarks, 12.34% of jets: mass 39.8 GeV, narrow (width 0.0034), a harder leading particle (266.86 GeV vs 240.22 on average), with 59.48% of the pT at 0.025-0.05 from the axis. Both compactness bonuses pass at full strength (girth2 < 0.0132 +4.175, width < 0.00868 +4.115), but these narrower, lighter jets also pay mass_over_sum_pt < 0.131 (-1.589), mass_over_sum_pt_sq < 0.00817 (-1.438), girth < 0.0761 (-1.396) and lam1 < 0.00838 (-1.267), all passing. The net is 1.124 (on for 73.8%), which raises the W score (+0.386) and lowers the g score; the formula calls them W (59%), with a notable g share (19%).
- **light (21 GeV), very narrow, pT spread over several particles** — 9.4% of jets, neuron 0.09. Mostly gluons (38%) with 35% quarks plus 14% W and 9% Z, 9.4% of jets: mass 20.9 GeV, width 0.0011, with 59.86% of the pT inside 0.025 and 32.77% at 0.025-0.05 from the axis. Large compactness bonuses (width < 0.00868 +5.909, girth2 < 0.0132 +5.152, lam1 < 0.00651 +1.297) are cancelled by penalties that all pass: girth < 0.0761 (-2.367), mass_over_sum_pt < 0.131 (-2.108), mass_over_sum_pt_sq < 0.00817 (-2.013), lam1 < 0.00838 (-1.821), width < 0.00437 (-1.567). Unlike group 2, mass < 21.8 passes for only 63.2% and lam1 < 0.000505 for 20.6%, which leaves the neuron just below zero; it is on for 9.6% (mean 0.094) and barely touches the scores, and the formula splits them g 50% / q 38%.
- **very light (7 GeV), very narrow, pT spread over several particles** — 5.8% of jets, neuron 0.00. A mixture (gluons 35%, Z 23%, W 21%, quarks 13%), 5.81% of jets: mass only 7.03 GeV yet width 0.0013, somewhat softer than average, with 57.58% of the pT at 0.025-0.05 from the axis instead of in the core. What marks this group is mass < 64.6 and centroid_offset > 0.0126 passing for 99.9% (+2.131) on top of the compactness bonuses (width < 0.00868 +5.744, girth2 < 0.0132 +5.06); but mass < 21.8 (-2.873), mass_over_sum_pt < 0.131 (-2.47), mass_over_sum_pt_sq < 0.00817 (-2.213), girth < 0.0761 (-2.036) and mass < 29.6 (-1.734) all pass and pull it under zero. The neuron is on for only 0.3% and adds nothing to the scores; the formula splits them g 52% / Z 28% / W 18%.
- **very light (10 GeV), very narrow, leading particle 49% of pT, high pT** — 4.1% of jets, neuron 0.02. Mostly quarks (68%, plus 18% gluons), 4.12% of jets: mass 9.5 GeV, width 0.0001, very hard (sum pT 1030 GeV, leading particle 503.69 GeV, about twice the average), with 97.6% of the pT inside 0.025 of the axis. As in group 2 the compactness bonuses (width < 0.00868 +6.64, girth2 < 0.0132 +5.55) are beaten by girth < 0.0761 (-3.279), lam1 < 0.000505 (97.3%, -2.797), mass_over_sum_pt < 0.131 and mass < 21.8 (96.1%); in addition sum_pt > 902 always passes (-2.792). The neuron is on for 1.7% (mean 0.023) and adds almost nothing; the formula calls them q (86%).
- **very light (19 GeV), very narrow, pT spread over several particles** — 0.7% of jets, neuron 0.00. A quark-gluon mixture (gluons 40%, quarks 34%, W and Z 10% each), 0.67% of jets: mass 19.4 GeV, width 0.0015, softer than average, with 50.4% of the pT at 0.025-0.05 from the axis. The single test mass < 29.6 and D2 < 0.876 passes for all of them and adds -7.073, which on top of the usual lightness penalties (mass_over_sum_pt < 0.131 -2.039, girth < 0.0761 -1.975, width < 0.00437 and D2 < 0.746 at 86.8%) outweighs the compactness bonuses. The neuron is off for all of them and adds nothing to the scores; the formula calls them g (66%).
- **light (28 GeV), very narrow, pT spread over several particles, high pT** — 0.6% of jets, neuron 0.01. Mostly gluons (64%, with 12% each of quarks and W), 0.56% of jets: mass 27.6 GeV, narrow (width 0.0011), very hard (sum pT 1276 GeV), with even the 8th particle at 55.23 GeV and 73.4% of the pT inside 0.025 of the axis. The high-pT tests decide it: sum_pt > 902 (-8.117), sum_pt > 902 and pt_7 > 29 (93.2%, -6.096) and sum_pt_top5 > 687 and z_7 > 0.0232 (89%, -3.719) far outweigh sum_pt_top5 > 687 (+2.814) and the compactness bonuses. The neuron is on for 0.3% and adds nothing to the scores; the formula calls them g (89%).

### neuron 3: radiation at wide angle (ΔR 0.2-0.4) (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00868 and girth2 < 0.0132 are its two strongest terms, both pushing down) and pushed up for e2 < 0.0445; it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.66 and 0.654). Tops sit clearly highest (2.14), then gluons (0.66) and quarks (0.34); Z jets are rarely on it (0.11) and W jets almost never (0.01, zero for 98.5% of them).
- *computed — its value:* largest for t (2.14), then g (0.66), then q (0.34), then Z (0.11), then W (0.01); it separates t jets from the rest best (AUC 0.78: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so it lowers the W score (-10%) and the Z score (-15%). It does not (or hardly) enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-10%), Z (-15%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = 2.89
if girth2 < 0.0087: z += -1229 × (0.0087 − girth2)
if girth2 < 0.013: z += -501 × (0.013 − girth2)
if e2 < 0.044: z += 153 × (0.044 − e2)
if girth2 < 0.0044: z += 734 × (0.0044 − girth2)
if mass_over_sum_pt > 0.068 and n_dr_0_0p05 < 5.00: z += 14.55 × (mass_over_sum_pt − 0.068) × (5.00 − n_dr_0_0p05)
if mass_over_sum_pt > 0.068: z += -56.86 × (mass_over_sum_pt − 0.068)
if lam1 < 0.0073: z += 237 × (0.0073 − lam1)
if e2 > 0.029: z += -56.96 × (e2 − 0.029)
if z_dr_0p1_0p2 < 0.468: z += 1.55 × (0.468 − z_dr_0p1_0p2)
if centroid_offset > 0.014: z += 60.75 × (centroid_offset − 0.014)
if mass > 36.23 and lam2 < 0.0011: z += 37.69 × (mass − 36.23) × (0.0011 − lam2)
if e2 < 0.044 and z_dr_0p05_0p1 < 0.964: z += -20.87 × (0.044 − e2) × (0.964 − z_dr_0p05_0p1)
if mass > 36.23: z += -0.025 × (mass − 36.23)
if width < 0.0067: z += 113 × (0.0067 − width)
if mass > 36.23 and eccentricity > 0.621: z += 0.076 × (mass − 36.23) × (eccentricity − 0.621)
if lam1 > 0.012: z += -233 × (lam1 − 0.012)
if mass_over_sum_pt > 0.090: z += 28.39 × (mass_over_sum_pt − 0.090)
if max_dr > 0.145: z += 8.60 × (max_dr − 0.145)
if width > 0.019: z += 295 × (width − 0.019)
if e2 < 0.038: z += 14.17 × (0.038 − e2)
if e2 > 0.029 and eccentricity > 0.960: z += -1466 × (e2 − 0.029) × (eccentricity − 0.960)
if LHA > 0.313 and eccentricity > 0.960: z += 834 × (LHA − 0.313) × (eccentricity − 0.960)
if mass > 69.61: z += -0.060 × (mass − 69.61)
if mass_over_sum_pt > 0.108: z += -21.07 × (mass_over_sum_pt − 0.108)
if e2 < 0.044 and z_dr_0p1_0p2 > 0: z += -201 × (0.044 − e2) × (z_dr_0p1_0p2 − 0)
if LHA > 0.313 and max_dr < 0.145: z += -2241 × (LHA − 0.313) × (0.145 − max_dr)
if LHA > 0.326: z += -7.03 × (LHA − 0.326)
if lam1 > 0.016: z += 122 × (lam1 − 0.016)
if max_dr > 0.145 and dr_3 < 0.046: z += -510 × (max_dr − 0.145) × (0.046 − dr_3)
if LHA > 0.313 and mass_top3 < 50.35: z += -0.262 × (LHA − 0.313) × (50.35 − mass_top3)
if mass_top5 > 53.61: z += -0.037 × (mass_top5 − 53.61)
if mass_over_sum_pt > 0.068 and dr_7 < 0.042: z += -7069 × (mass_over_sum_pt − 0.068) × (0.042 − dr_7)
if mass_over_sum_pt > 0.090 and pt_6 < 56.53: z += 0.458 × (mass_over_sum_pt − 0.090) × (56.53 − pt_6)
if mean_eta < -0.013: z += 28.23 × (-0.013 − mean_eta)
if LHA > 0.313 and planar_flow > 0.008: z += 10.17 × (LHA − 0.313) × (planar_flow − 0.008)
if LHA > 0.424: z += -38.34 × (LHA − 0.424)
if mean_eta < -0.013 and pt_4 < 68.12: z += -1.38 × (-0.013 − mean_eta) × (68.12 − pt_4)
if mass_over_sum_pt > 0.108 and dr_7 < 0.049: z += 12805 × (mass_over_sum_pt − 0.108) × (0.049 − dr_7)
if C2 > 0.095: z += -50.17 × (C2 − 0.095)
if mass > 36.23 and dr_6 < 0.046: z += -1.59 × (mass − 36.23) × (0.046 − dr_6)
if mass_over_sum_pt > 0.068 and tau32 < 0.519: z += -13.93 × (mass_over_sum_pt − 0.068) × (0.519 − tau32)
if width > 0.019 and z_3 > 0.090: z += -2111 × (width − 0.019) × (z_3 − 0.090)
if mass_over_sum_pt > 0.090 and max_dr < 0.145: z += 11460 × (mass_over_sum_pt − 0.090) × (0.145 − max_dr)
if centroid_offset > 0.014 and eta_0 < -0.040: z += -350 × (centroid_offset − 0.014) × (-0.040 − eta_0)
if LHA > 0.313: z += -2.25 × (LHA − 0.313)
if mass_over_sum_pt > 0.090 and pt_7 < 48.72: z += 0.251 × (mass_over_sum_pt − 0.090) × (48.72 − pt_7)
if mass > 69.61 and max_dr < 0.177: z += 2.10 × (mass − 69.61) × (0.177 − max_dr)
if width > 0.019 and pt_dispersion < 0.492: z += 368 × (width − 0.019) × (0.492 − pt_dispersion)
if centroid_offset > 0.014 and phi_7 > -0.042: z += 54.25 × (centroid_offset − 0.014) × (phi_7 − -0.042)
if lam1 > 0.016 and eccentricity > 0.960: z += -2519 × (lam1 − 0.016) × (eccentricity − 0.960)
if mass > 36.23 and pt_7 < 20.12: z += -0.0054 × (mass − 36.23) × (20.12 − pt_7)
if mean_eta < -0.013 and z_4 < 0.120: z += 343 × (-0.013 − mean_eta) × (0.120 − z_4)
if e2 > 0.029 and n_pt_above_50 > 3.00: z += -0.931 × (e2 − 0.029) × (n_pt_above_50 − 3.00)
if max_dr > 0.145 and eta_1 < -0.060: z += -56.61 × (max_dr − 0.145) × (-0.060 − eta_1)
if lam1 > 0.012 and pt_6 < 38.25: z += 0.289 × (lam1 − 0.012) × (38.25 − pt_6)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (9 GeV), very narrow, pT spread over several particles** — 31.7% of jets, neuron 0.01. Mostly quarks (45%, with 34% gluons), 31.7% of jets: mass 9.0 GeV, width 0.0003, with 85.56% of the pT inside 0.025 of the axis and almost nothing at wide angle. The narrowness tests girth2 < 0.00868 (-10.261) and girth2 < 0.0132 (-6.467) pass for all and outweigh the bonuses e2 < 0.0445 (+5.986), girth2 < 0.00437 (+2.967) and lam1 < 0.00733 (+1.665). The neuron is off for almost all of them (on for 1.2%) and leaves the scores untouched; the formula splits them q 51% / g 40%.
- **medium-mass (55 GeV), average width, pT spread over several particles** — 16.2% of jets, neuron 0.17. Mostly W (48%, with 34% Z and 10% tops), 16.16% of jets: mass 54.8 GeV, width 0.0061, with 60.74% of the pT at 0.05-0.1 from the axis and little beyond 0.1. Both girth2 tests still pass (girth2 < 0.0132 -3.577, girth2 < 0.00868 -3.172), and the bonuses e2 < 0.0445 (99%, +1.324), mass > 36.2 and lam2 < 0.00113 (91.9%, +0.732) and mass > 36.2 and eccentricity > 0.621 (93.3%) do not quite make up for them. The neuron is on for only 10.3% (mean 0.169), so it lowers the W and Z scores only slightly; the formula calls them W (61%).
- **light (45 GeV), narrow, pT spread over several particles** — 13.7% of jets, neuron 0.22. Mostly W (43%, with 28% Z and 12% gluons), 13.65% of jets: mass 45.4 GeV, width 0.0044, with 51.68% of the pT at 0.025-0.05 from the axis. Narrower than group 1, they pay more for girth2 < 0.00868 (-5.274) and girth2 < 0.0132 (-4.434), against e2 < 0.0445 (+3.051), lam1 < 0.00733 (+0.734) and z_dr_0p1_0p2 < 0.468 (+0.602). The neuron is on for 12.4% (mean 0.221) and lowers the W and Z scores only a little; the formula calls them W (61%).
- **light (28 GeV), narrow, pT spread over several particles** — 11.3% of jets, neuron 0.14. A mixture of gluons (30%), W (26%), quarks (19%) and Z (18%), 11.31% of jets: mass 28.0 GeV, width 0.0022, with 57.39% of the pT at 0.025-0.05 from the axis. As for the thinnest jets, girth2 < 0.00868 (-7.97) and girth2 < 0.0132 (-5.533) outweigh e2 < 0.0445 (+4.477), girth2 < 0.00437 (+1.599) and lam1 < 0.00733 (+1.247). The neuron is on for 10.9% (mean 0.14) and barely touches the scores; the formula splits them g 44% / W 34%.
- **medium-mass (60 GeV), wide, pT spread over several particles** — 10.8% of jets, neuron 1.03. Mostly Z (56%, with 25% tops), 10.8% of jets: mass 59.6 GeV, width 0.0085, with 62.77% of the pT at 0.05-0.1 and 20.28% at 0.1-0.15 from the axis. girth2 < 0.0132 still passes (99.8%, -2.391) but girth2 < 0.00868 only for 68.5% (-0.752); mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (91.8%, +1.187) and mass > 36.2 and lam2 < 0.00113 (+0.868) are cancelled by mass_over_sum_pt > 0.0681 (-1.174) and e2 > 0.0285 (-0.807). The result is 1.026 (on for 37.3%), which lowers the Z (-0.577) and W (-0.513) scores and raises the t score slightly; the formula calls them Z (67%), with t 28%.
- **medium-mass (66 GeV), very wide, pT spread over several particles, low pT** — 6.8% of jets, neuron 3.73. Mostly tops (76%, with 17% gluons), 6.82% of jets: mass 65.9 GeV, width 0.0157, soft (sum pT 567 GeV), with 40.78% of the pT at 0.05-0.1 and 31.5% at 0.1-0.15 from the axis. The girth2 penalties no longer pass, and mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (96.7%, +3.003, few particles near the axis), centroid_offset > 0.0144 (87.2%, +1.531), mass_over_sum_pt > 0.0904 and max_dr > 0.145 (89.2%) outweigh mass_over_sum_pt > 0.0681 (-2.741) and e2 > 0.0285 (-1.472). The neuron is high (3.725, on for 91.7%), strongly lowering the Z (-2.095) and W (-1.863) scores and raising the t score (+0.233); the formula calls them t (96%).
- **medium-mass (84 GeV), very wide, pT spread over several particles, low pT** — 4.7% of jets, neuron 1.80. Mostly tops (92%), 4.7% of jets: mass 83.8 GeV, width 0.0257 (four times the average), soft (sum pT 540 GeV), with the pT spread from 0.05 out to 0.3 and 34.18% at 0.1-0.15. The big pair mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+6.136) and mass_over_sum_pt > 0.0681 (-4.959) leave a gain, joined by width > 0.0188 (+2.04) and mass_over_sum_pt > 0.0904 (+1.844), but the broadness penalties e2 > 0.0285 (-3.014) and lam1 > 0.012 (-2.044) cut it back. The neuron ends at 1.801 (on for 72.4%), lower than for group 5, lowering the Z and W scores and raising the t score; the formula calls them t (99%).
- **medium-mass (86 GeV), very wide, pT spread over several particles** — 2.7% of jets, neuron 3.20. Mostly tops (71%) with 20% gluons, 2.68% of jets: mass 86.5 GeV, width 0.0206, with 44.57% of the pT at 0.1-0.15 from the axis. Besides the usual pair (+4.826 / -4.045), the elongation tests set this group apart: LHA > 0.313 and eccentricity > 0.96 (99.1%, +2.344), mass > 36.2 and lam2 < 0.00113 (+1.833) and mass > 36.2 and eccentricity > 0.621 (+1.419) outweigh e2 > 0.0285 (-2.204), lam1 > 0.012 (-1.961) and e2 > 0.0285 and eccentricity > 0.96 (-1.803). The neuron is high (3.203, on for 91.5%), lowering the Z (-1.802) and W (-1.602) scores and raising the t score; the formula calls them t (93%).
- **heavy (93 GeV), very wide, pT spread over several particles, low pT** — 1.9% of jets, neuron 2.09. Mostly tops (71%) with 21% gluons, 1.92% of jets: mass 93.2 GeV, width 0.0354 (over five times the average), soft (sum pT 514 GeV), with 32.66% of the pT at 0.2-0.3 and 31.65% at 0.15-0.2 from the axis. The largest amounts of this neuron appear here: mass_over_sum_pt > 0.0681 and n_dr_0_0p05 < 5 (+8.07), width > 0.0188 (+4.899), mass_over_sum_pt > 0.0904 (+2.6) and lam1 > 0.0164 (+1.887) against mass_over_sum_pt > 0.0681 (-6.473), lam1 > 0.012 (-4.638) and e2 > 0.0285 (-3.836). The net is 2.088 (on for 71.1%), lowering the Z and W scores and raising the t score; the formula calls them t (90%).
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 1.82. Mostly tops (91%), 0.27% of jets: mass 80.3 GeV, width 0.0239, with an odd pT profile: a quarter of it inside 0.025 of the axis, yet 22.86% at 0.2-0.3. A pair of tests on a central 8th particle decides it: mass_over_sum_pt > 0.108 and dr_7 < 0.0488 (+13.823) against mass_over_sum_pt > 0.0681 and dr_7 < 0.0422 (-11.612), with mass_over_sum_pt > 0.0681 (-4.599), e2 > 0.0285 (-2.715) and lam1 > 0.012 (-2.498) pulling down. The neuron is on for 48.1% (mean 1.816), lowering the Z and W scores and raising the t score; the formula calls them t (98%).

### neuron 4: clean two-prong shape (low τ21, D2) (moderate)

- **What it measures:** Pushed up for width > 0.00165 and small e2_sq (< 0.0172), its two strongest terms, and for τ21 < 0.238; pushed down for light jets (mass < 69.6 GeV), with a mass/pT window (down above 0.0904, back up above 0.108); it falls with τ21 and D2 and rises with the pT share at 0.05 ≤ ΔR < 0.1 (rank correlations -0.582, -0.525, 0.506). Z jets sit highest (3.46), then W (2.56), tops (2.11) and gluons (1.51); quarks sit lowest (0.79) and it is zero for 44.3% of them.
- *computed — its value:* largest for Z (3.46), then W (2.56), then t (2.11), then g (1.51), then q (0.79); it separates Z jets from the rest best (AUC 0.75: large for Z)
- **How the class scores use it:** Z jets sit highest on it, so it raises the Z score (+7%); quarks sit lowest, so it lowers the q score (-12%) and, a little, the g score (-3%). It also raises the t score (+7%), a secondary input next to neurons 13 and 10 that reflects tops sitting above quarks and gluons on it; it does not (or hardly) enter the W score.
- *computed — used by:* raises the score of Z (+7%), t (+7%); lowers the score of g (-3%), q (-12%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** 

```
z = -7.10
if width > 0.0017: z += 984 × (width − 0.0017)
if e2_sq < 0.017: z += 423 × (0.017 − e2_sq)
if width > 0.00032: z += -502 × (width − 0.00032)
if mass < 69.61: z += -0.071 × (69.61 − mass)
if mass_over_sum_pt > 0.108: z += 404 × (mass_over_sum_pt − 0.108)
if mass_over_sum_pt > 0.090: z += -227 × (mass_over_sum_pt − 0.090)
if e2_sq < 0.024: z += 103 × (0.024 − e2_sq)
if e2 > 0.0071: z += 73.24 × (e2 − 0.0071)
if tau21 < 0.238: z += 27.14 × (0.238 − tau21)
if mass < 56.92: z += 0.066 × (56.92 − mass)
if mass_over_sum_pt > 0.108 and D2 < 3.89: z += -55.40 × (mass_over_sum_pt − 0.108) × (3.89 − D2)
if girth2_top3 < 0.005: z += 353 × (0.005 − girth2_top3)
if max_dr > 0.103: z += 18.60 × (max_dr − 0.103)
if lam2 < 0.00031 and mass_top3 < 50.35: z += -109 × (0.00031 − lam2) × (50.35 − mass_top3)
if centroid_offset < 0.014 and z_dr_0p05_0p1 < 0.675: z += -272 × (0.014 − centroid_offset) × (0.675 − z_dr_0p05_0p1)
if tau21 < 0.238 and lam2 < 0.0011: z += -10138 × (0.238 − tau21) × (0.0011 − lam2)
if C2 > 0.015: z += -32.74 × (C2 − 0.015)
if lam2 < 0.0011: z += -546 × (0.0011 − lam2)
if girth2 > 0.013: z += -340 × (girth2 − 0.013)
if sum_pt < 764: z += 0.0049 × (764 − sum_pt)
if lam2 < 0.00031: z += -2129 × (0.00031 − lam2)
if C2 > 0.067: z += -142 × (C2 − 0.067)
if width > 0.00032 and C2 < 0.067: z += 2212 × (width − 0.00032) × (0.067 − C2)
if sum_pt < 764 and n_dr_0p2_0p4 < 1.00: z += -0.0052 × (764 − sum_pt) × (1.00 − n_dr_0p2_0p4)
if width > 0.00032 and z_dr_0p2_0p4 < 0.056: z += 1772 × (width − 0.00032) × (0.056 − z_dr_0p2_0p4)
if girth < 0.048: z += 24.45 × (0.048 − girth)
if mass < 69.61 and dr_3 < 0.104: z += 0.112 × (69.61 − mass) × (0.104 − dr_3)
if girth > 0.102: z += -44.16 × (girth − 0.102)
if tau21 < 0.238 and mass < 62.55: z += -0.447 × (0.238 − tau21) × (62.55 − mass)
if girth2_top2 < 0.0095: z += -30.05 × (0.0095 − girth2_top2)
if girth2_top2 < 0.0095 and centroid_offset > 0.016: z += 8792 × (0.0095 − girth2_top2) × (centroid_offset − 0.016)
if tau21 < 0.238 and pt_7 > 33.22: z += 0.515 × (0.238 − tau21) × (pt_7 − 33.22)
if max_dr > 0.103 and pt_7 > 37.16: z += -1.57 × (max_dr − 0.103) × (pt_7 − 37.16)
if max_dr > 0.198: z += -10.71 × (max_dr − 0.198)
if C2 > 0.015 and pt_7 > 38.53: z += 5.42 × (C2 − 0.015) × (pt_7 − 38.53)
if tau21 < 0.238 and e2_sq > 0.012: z += -1727 × (0.238 − tau21) × (e2_sq − 0.012)
if centroid_offset < 0.014: z += 27.08 × (0.014 − centroid_offset)
if e2 > 0.041: z += -19.88 × (e2 − 0.041)
if tau21 < 0.238 and dr_7 < 0.175: z += -21.18 × (0.238 − tau21) × (0.175 − dr_7)
if tau21 < 0.238 and dr_6 < 0.170: z += -19.76 × (0.238 − tau21) × (0.170 − dr_6)
if tau21 < 0.238 and planar_flow > 0.045: z += 29.36 × (0.238 − tau21) × (planar_flow − 0.045)
if sum_pt < 764 and z_dr_0_0p05 > 0.152: z += 0.0015 × (764 − sum_pt) × (z_dr_0_0p05 − 0.152)
if sum_pt_top5 < 431: z += -0.0027 × (431 − sum_pt_top5)
if mass < 69.61 and mean_eta2 > 0.0042: z += 5.29 × (69.61 − mass) × (mean_eta2 − 0.0042)
if tau21 < 0.238 and pt_2 > 73.69: z += 0.022 × (0.238 − tau21) × (pt_2 − 73.69)
if tau21 < 0.238 and mean_phi > 0.0044: z += -157 × (0.238 − tau21) × (mean_phi − 0.0044)
if tau21 < 0.238 and pt_7 < 23.22: z += -1.13 × (0.238 − tau21) × (23.22 − pt_7)
if girth2_top3 < 0.005 and mean_eta < -0.0094: z += 9393 × (0.005 − girth2_top3) × (-0.0094 − mean_eta)
if mass_over_sum_pt_sq < 0.0011: z += 87.38 × (0.0011 − mass_over_sum_pt_sq)
if tau21 < 0.238 and pt_6 < 62.25: z += -0.018 × (0.238 − tau21) × (62.25 − pt_6)
if lam2 < 0.0011 and n_dr_0p1_0p2 > 2.00: z += 56.69 × (0.0011 − lam2) × (n_dr_0p1_0p2 − 2.00)
if mass_over_sum_pt > 0.108 and n_dr_0p1_0p2 > 6.00: z += -8.64 × (mass_over_sum_pt − 0.108) × (n_dr_0p1_0p2 − 6.00)
if tau21 < 0.238 and mean_eta > 0.026: z += -262 × (0.238 − tau21) × (mean_eta − 0.026)
if e2 > 0.063: z += 3.96 × (e2 − 0.063)
if tau21 < 0.238 and phi_6 > 0.119: z += -32.74 × (0.238 − tau21) × (phi_6 − 0.119)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (10 GeV), very narrow, pT spread over several particles** — 36.5% of jets, neuron 0.89. Mostly quarks (41.5%) with 35.76% gluons, 36.54% of jets: mass 10.1 GeV, width 0.0005, 77.06% of the pT inside 0.025 of the axis. They pass the small-e2 tests e2_sq < 0.0172 (+7.139) and e2_sq < 0.0238 (+2.414), mass < 56.9 (+3.086) and girth2_top3 < 0.00501 (+1.622), but mass < 69.6 (-4.256) and the light-jet combinations lam2 < 0.000306 and mass_top3 < 50.4 (97.6%, -1.341) and centroid_offset < 0.0144 and z_dr_0p05_0p1 < 0.675 (72.3%, -1.192) take much of it back; width > 0.00165 and tau21 < 0.238 almost never pass. The neuron ends at 0.893 (on for 68.3%), which slightly lowers the q and g scores and raises the t and Z scores; the formula splits them q 45% / g 44%.
- **medium-mass (60 GeV), average width, pT spread over several particles** — 21.3% of jets, neuron 4.44. Mostly Z (45%, with 39% W and 11% tops), 21.35% of jets: mass 59.6 GeV, average width (0.0064), with 58.94% of the pT at 0.05-0.1 from the axis and almost none in the core. All three main bonuses pass for every jet: width > 0.00165 (+4.675), e2_sq < 0.0172 (+4.637) and the two-prong test tau21 < 0.238 (+4.253), plus e2 > 0.00708 (+2.129); width > 0.000319 (-3.058), tau21 < 0.238 and lam2 < 0.00113 (-1.713) and mass < 69.6 (83.4%) take back only part. The neuron sits at 4.441 (on for all), lowering the q (-0.416) and g scores and raising the t (+0.555) and Z (+0.347) scores; the formula splits them W 49% / Z 45%.
- **light (39 GeV), narrow, pT spread over several particles** — 15.3% of jets, neuron 1.72. Mostly W (38%) with 25% Z, 16% gluons and 13% quarks, 15.28% of jets: mass 39.2 GeV, narrow (width 0.0034), with 54.06% of the pT at 0.025-0.05 from the axis. e2_sq < 0.0172 (+6.0) and e2_sq < 0.0238 (+2.137) carry it, with width > 0.00165 (96.3%, +1.756), while tau21 < 0.238 passes for only 60.4% (+1.311); mass < 69.6 (-2.17) and width > 0.000319 (-1.562) pull back. The neuron sits at 1.716 (on for 89.8%), mildly lowering the q and g scores and raising the t and Z scores; the formula calls them W (55%).
- **light (47 GeV), average width, pT spread over several particles, low pT** — 9.0% of jets, neuron 3.14. A mixture of Z (36%), W (25%), tops (17%) and gluons (15%), 9.02% of jets: mass 47.1 GeV, width 0.0068, soft (sum pT 607 GeV), with 54.98% of the pT at 0.05-0.1 from the axis. width > 0.00165 (+5.063), e2_sq < 0.0172 (+4.663) and e2 > 0.00708 (+2.198) pass for nearly all, against width > 0.000319 (-3.257) and mass < 69.6 (-1.619); unlike group 1, tau21 < 0.238 passes for only 45.1%. The neuron sits at 3.144 (on for 97.5%), lowering the q and g scores and raising the t and Z scores; the formula splits them Z 40% / W 31% / t 20%.
- **medium-mass (61 GeV), wide, pT spread over several particles, low pT** — 4.7% of jets, neuron 4.03. Mostly tops (67%) with 21% gluons, 4.66% of jets: mass 60.7 GeV, width 0.0126 (about twice the average), soft (sum pT 589 GeV), with the pT mostly at 0.05-0.15 from the axis. The width tests grow with width: width > 0.00165 (+10.769) outruns width > 0.000319 (-6.171), and e2 > 0.00708 (+2.889), e2_sq < 0.0172 (+2.742) and tau21 < 0.238 (73.6%, +2.67) add more, while mass_over_sum_pt > 0.0904 (91.2%, -3.024) is the main brake. The neuron is high (4.031, on for 85.4%), lowering the q (-0.378) and g scores and raising the t (+0.504) and Z scores; the formula calls them t (93%).
- **medium-mass (73 GeV), very wide, pT spread over several particles, low pT** — 3.6% of jets, neuron 1.46. Mostly tops (78%, plus 15% gluons), 3.64% of jets: mass 73.0 GeV, width 0.0169, soft (sum pT 590 GeV), with the pT mostly at 0.05-0.15 from the axis. Here the large terms nearly cancel: width > 0.00165 (+15.029) and mass_over_sum_pt > 0.108 (+6.313) against width > 0.000319 (-8.347), mass_over_sum_pt > 0.0904 (-7.549) and mass_over_sum_pt > 0.108 and D2 < 3.89 (-2.507); tau21 < 0.238 (56.3%) and max_dr > 0.103 (+2.41) tip the balance. The neuron sits at 1.462 (on for 53.5%), raising the t score and lowering the q score a little; the formula calls them t (96%).
- **medium-mass (80 GeV), very wide, pT spread over several particles, low pT** — 3.3% of jets, neuron 0.70. Mostly tops (82%), 3.3% of jets: mass 79.8 GeV, width 0.0211, soft (sum pT 572 GeV), with 37.92% of the pT at 0.1-0.15 from the axis. The same large cancelling terms, now bigger: width > 0.00165 (+19.14) and mass_over_sum_pt > 0.108 (+12.787) against mass_over_sum_pt > 0.0904 (-11.202), width > 0.000319 (-10.447) and mass_over_sum_pt > 0.108 and D2 < 3.89 (-5.168), with girth2 > 0.0132 (-2.679) and C2 > 0.0673 (50.6%) tipping it down. The neuron falls to 0.7 (on for 34.6%) and has only a small effect on the scores; the formula calls them t (97%).
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 3.2% of jets, neuron 0.25. Mostly tops (88%), 3.18% of jets: mass 85.2 GeV, width 0.0258 (four times the average), soft (sum pT 546 GeV), with the pT spread between 0.05 and 0.3 from the axis. width > 0.00165 (+23.755) and mass_over_sum_pt > 0.108 (+19.338) are cancelled by mass_over_sum_pt > 0.0904 (-14.893), width > 0.000319 (-12.804) and mass_over_sum_pt > 0.108 and D2 < 3.89 (-7.633), and the broadness penalties girth2 > 0.0132 (-4.275) and C2 > 0.0673 (71.8%, -3.382) push it below zero for most. The neuron is on for only 17.6% (mean 0.253) and barely moves the scores; the formula calls them t (98%).
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.4% of jets, neuron 0.16. Mostly tops (84%), 2.39% of jets: mass 90.9 GeV, width 0.0314, soft (sum pT 526 GeV), with most of the pT beyond 0.1 from the axis. The same balance at larger size: width > 0.00165 (+29.236) and mass_over_sum_pt > 0.108 (+26.198) against mass_over_sum_pt > 0.0904 (-18.758), width > 0.000319 (-15.604) and mass_over_sum_pt > 0.108 and D2 < 3.89 (-10.872), with girth2 > 0.0132 (-6.17) and girth > 0.102 (-2.825) making the net negative for most. The neuron is on for 16.2% (mean 0.156) and barely moves the scores; the formula calls them t (95%).
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.6% of jets, neuron 0.13. Mostly tops (60%) mixed with 30% gluons, 0.63% of jets: mass 94.8 GeV, width 0.0413 (over six times the average), the softest group (sum pT 483 GeV), with 42.14% of the pT at 0.2-0.3 from the axis. The largest amounts of all cancel: width > 0.00165 (+39.029) and mass_over_sum_pt > 0.108 (+35.803) against mass_over_sum_pt > 0.0904 (-24.17), width > 0.000319 (-20.607), mass_over_sum_pt > 0.108 and D2 < 3.89 (-15.291) and girth2 > 0.0132 (-9.557). The neuron is on for 11.4% (mean 0.131) and barely moves the scores; the formula calls them t (86%), with g 14%.

### neuron 5: quark-likeness: pT in few particles (moderate)

- **What it measures:** Pushed up when the 8th-hardest particle carries little of the pT (z_7 < 0.0715, its strongest term), for small e2 (< 0.0356) and small angularity (LHA < 0.216), and pushed down for very hard but narrow jets (log of total pT > 6.57 with lam1 < 0.012); it rises with the summed pT of the 3 hardest particles and falls with z_7 (rank correlations 0.799 and -0.775). Quarks sit far highest (3.83); Z (1.63) and W (1.59) are in the middle, gluons (0.99) and tops (0.47) lowest.
- *computed — its value:* largest for q (3.83), then Z (1.63), then W (1.59), then g (0.99), then t (0.47); it separates q jets from the rest best (AUC 0.73: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops low, it lowers the g score (-14%) and the t score (-12%) and raises the q score (+5%). It does not (or hardly) enter the W or Z scores.
- *computed — used by:* raises the score of q (+5%); lowers the score of g (-14%), t (-12%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = 0.469
if z_7 < 0.071: z += 57.78 × (0.071 − z_7)
if log_sum_pt > 6.57 and lam1 < 0.012: z += -1175 × (log_sum_pt − 6.57) × (0.012 − lam1)
if e2 < 0.036: z += 64.75 × (0.036 − e2)
if LHA < 0.216: z += 18.54 × (0.216 − LHA)
if LHA < 0.216 and log_sum_pt < 6.80: z += -120 × (0.216 − LHA) × (6.80 − log_sum_pt)
if dr_0 < 0.022: z += -128 × (0.022 − dr_0)
if z_7 < 0.071 and lam2 < 0.0011: z += 20000 × (0.071 − z_7) × (0.0011 − lam2)
if mean_phi2 < 0.014: z += -37.79 × (0.014 − mean_phi2)
if sum_pt_top2 < 548 and girth2_top3 < 0.004: z += -1.51 × (548 − sum_pt_top2) × (0.004 − girth2_top3)
if width < 0.0026 and centroid_offset < 0.024: z += 31979 × (0.0026 − width) × (0.024 − centroid_offset)
if sum_pt > 869: z += -0.021 × (sum_pt − 869)
if mean_phi2 < 0.014 and max_pair_mass < 40.05: z += 1.08 × (0.014 − mean_phi2) × (40.05 − max_pair_mass)
if z_7 < 0.071 and centroid_offset < 0.031: z += -873 × (0.071 − z_7) × (0.031 − centroid_offset)
if log_sum_pt > 6.57 and dr_0 < 0.022: z += 545 × (log_sum_pt − 6.57) × (0.022 − dr_0)
if mass_over_sum_pt < 0.085 and max_pair_mass < 18.10: z += -0.736 × (0.085 − mass_over_sum_pt) × (18.10 − max_pair_mass)
if mass_over_sum_pt < 0.085: z += -10.34 × (0.085 − mass_over_sum_pt)
if z_7 < 0.071 and sum_pt < 788: z += -0.337 × (0.071 − z_7) × (788 − sum_pt)
if log_sum_pt > 6.57 and girth2_top2 < 0.0063: z += 689 × (log_sum_pt − 6.57) × (0.0063 − girth2_top2)
if width < 0.0026: z += 346 × (0.0026 − width)
if sum_pt_top2 < 548: z += -0.0012 × (548 − sum_pt_top2)
if dr_0 < 0.026: z += -44.64 × (0.026 − dr_0)
if z_dr_0p1_0p2 < 0.045: z += 9.09 × (0.045 − z_dr_0p1_0p2)
if sum_pt_top5 > 752: z += 0.0097 × (sum_pt_top5 − 752)
if z_7 < 0.049 and mass_top5 < 62.55: z += 0.634 × (0.049 − z_7) × (62.55 − mass_top5)
if e2 < 0.036 and lam2 < 7.3e-05: z += 335180 × (0.036 − e2) × (7.3e-05 − lam2)
if log_sum_pt > 6.57: z += 2.05 × (log_sum_pt − 6.57)
if sum_pt_top2 < 548 and dr_0 < 0.022: z += 0.391 × (548 − sum_pt_top2) × (0.022 − dr_0)
if z_7 < 0.071 and lam1 < 0.0015: z += -12026 × (0.071 − z_7) × (0.0015 − lam1)
if z_7 < 0.071 and mean_phi2 < 0.0043: z += 2022 × (0.071 − z_7) × (0.0043 − mean_phi2)
if z_7 < 0.049: z += -16.41 × (0.049 − z_7)
if e2_sq < 0.0053 and centroid_offset < 0.014: z += -8767 × (0.0053 − e2_sq) × (0.014 − centroid_offset)
if pt_5 < 24.58: z += 0.333 × (24.58 − pt_5)
if z_7 < 0.032 and pt_5 < 29.88: z += -8.44 × (0.032 − z_7) × (29.88 − pt_5)
if z_6 < 0.029: z += 91.40 × (0.029 − z_6)
if LHA < 0.155: z += -5.67 × (0.155 − LHA)
if sum_pt > 869 and centroid_offset < 0.013: z += 0.506 × (sum_pt − 869) × (0.013 − centroid_offset)
if log_sum_pt > 6.90 and dr_0 < 0.041: z += -523 × (log_sum_pt − 6.90) × (0.041 − dr_0)
if log_sum_pt > 6.57 and mean_phi2 < 0.00015: z += 13831 × (log_sum_pt − 6.57) × (0.00015 − mean_phi2)
if z_7 < 0.032: z += 27.85 × (0.032 − z_7)
if log_sum_pt > 6.90 and centroid_offset < 0.018: z += -821 × (log_sum_pt − 6.90) × (0.018 − centroid_offset)
if log_sum_pt > 6.57 and mean_eta2 < 9e-05: z += 20848 × (log_sum_pt − 6.57) × (9e-05 − mean_eta2)
if sum_pt > 869 and pt_5 < 33.03: z += 0.00037 × (sum_pt − 869) × (33.03 − pt_5)
if log_sum_pt > 6.90: z += 5.82 × (log_sum_pt − 6.90)
if LHA < 0.216 and n_dr_0p2_0p4 > 0: z += -10.81 × (0.216 − LHA) × (n_dr_0p2_0p4 − 0)
if z_7 < 0.032 and pt_5 > 33.03: z += 1.58 × (0.032 − z_7) × (pt_5 − 33.03)
if e2_sq < 0.0053 and n_pt_above_50 > 6.00: z += -19.74 × (0.0053 − e2_sq) × (n_pt_above_50 − 6.00)
if girth2 < 4.8e-05: z += 15913 × (4.8e-05 − girth2)
if LHA < 0.216 and mass_top3 > 3.56: z += -0.763 × (0.216 − LHA) × (mass_top3 − 3.56)
if z_6 < 0.029 and phi_0 > 0.015: z += -7390 × (0.029 − z_6) × (phi_0 − 0.015)
if log_sum_pt > 6.90 and planar_flow < 0.045: z += -464 × (log_sum_pt − 6.90) × (0.045 − planar_flow)
if log_sum_pt > 6.57 and centroid_offset > 0.018: z += -109 × (log_sum_pt − 6.57) × (centroid_offset − 0.018)
if pt_5 < 24.58 and mean_eta2 > 1.8e-05: z += -21.51 × (24.58 − pt_5) × (mean_eta2 − 1.8e-05)
if log_sum_pt > 6.90 and dr_4 > 0.030: z += -58.42 × (log_sum_pt − 6.90) × (dr_4 − 0.030)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (55 GeV), wide, pT spread over several particles, low pT** — 28.3% of jets, neuron 0.15. A mixture led by tops (37%) with Z (21%), W (20%) and gluons (16%), 28.32% of jets: mass 54.9 GeV, width 0.0119, soft (sum pT 579 GeV), with pT shared evenly (leading particle 139.75 GeV, 8th particle 41.86 GeV) and spread over 0.025-0.2 from the axis. With a hard 8th particle, z_7 < 0.0715 passes for only 48.8%, and no if-statement adds much: sum_pt_top2 < 548 (-0.368) and mean_phi2 < 0.0143 (88.6%, -0.338) against mean_phi2 < 0.0143 and max_pair_mass < 40 (87.1%, +0.256) and z_7 < 0.0715 (+0.198). The neuron stays near zero (0.154, on for 38.6%) and barely moves the scores; the formula splits them t 41% / W 26% / Z 20%.
- **medium-mass (60 GeV), wide, pT spread over several particles** — 20.3% of jets, neuron 0.87. A mixture led by tops (38%) with Z (24%) and W (21%), 20.3% of jets: mass 59.9 GeV, width 0.0104, somewhat soft (sum pT 663 GeV), 8th particle 31.1 GeV, pT mostly at 0.025-0.15 from the axis. z_7 < 0.0715 passes for all (+1.389), with z_7 < 0.0715 and lam2 < 0.00113 (82.8%, +0.38), but z_7 < 0.0715 and sum_pt < 788 (93.4%, -0.928), mean_phi2 < 0.0143 and z_7 < 0.0715 and centroid_offset < 0.0312 take most of it back. The neuron sits at 0.872 (on for 75%), mildly lowering the t and g scores; the formula splits them t 41% / W 28% / Z 25%.
- **medium-mass (52 GeV), average width, pT spread over several particles, high pT** — 13.3% of jets, neuron 2.76. An even W/Z mixture (W 40%, Z 38%), 13.29% of jets: mass 51.8 GeV, width 0.0045, hard (sum pT 843 GeV, leading particle 335.6 GeV), soft 8th particle (28.74 GeV), pT mostly at 0.025-0.1 from the axis. z_7 < 0.0715 (+2.16), z_7 < 0.0715 and lam2 < 0.00113 (+0.782) and e2 < 0.0356 (78.3%, +0.79) outweigh the hard-and-narrow penalty log_sum_pt > 6.57 and lam1 < 0.012 (99.4%, -1.41) and z_7 < 0.0715 and centroid_offset < 0.0312 (-0.571). The neuron sits at 2.761 (on for 97.5%), lowering the t (-0.69) and g (-0.518) scores and raising the q score a little; the formula splits them W 52% / Z 35%.
- **very light (13 GeV), very narrow, pT spread over several particles** — 10.3% of jets, neuron 0.75. Mostly gluons (41%, with 22% quarks, 16% W, 14% Z), 10.27% of jets: mass 13.3 GeV, width 0.0012, somewhat soft (sum pT 651 GeV, leading particle 178.54 GeV), with the pT within 0.05 of the axis. e2 < 0.0356 (+1.662), z_7 < 0.0715 (74.4%, +0.736) and LHA < 0.216 (86.9%, +0.607) are offset by sum_pt_top2 < 548 and girth2_top3 < 0.00395 (99.1%, -1.137), LHA < 0.216 and log_sum_pt < 6.8 (86.8%, -1.118) and mass_over_sum_pt < 0.0848 and max_pair_mass < 18.1 (-0.729). The neuron sits at 0.75 (on for 56.3%), mildly lowering the t and g scores; the formula calls them g (63%).
- **very light (8 GeV), very narrow, pT spread over several particles, high pT** — 8.2% of jets, neuron 3.99. Mostly quarks (55%, with 28% gluons), 8.24% of jets: mass 8.5 GeV, width 0.0002, hard (sum pT 847 GeV, leading particle 296.04 GeV), with 96.48% of the pT inside 0.025 of the axis. Nearly all the quark-like bonuses pass: LHA < 0.216 (+2.057), e2 < 0.0356 (+2.002), z_7 < 0.0715 (96.6%, +1.792), width < 0.00264 and centroid_offset < 0.0236 (+1.4) and log_sum_pt > 6.57 and dr_0 < 0.0216 (+1.228), against log_sum_pt > 6.57 and lam1 < 0.012 (-2.331) and dr_0 < 0.0216 (-1.715). The neuron is high (3.986, on for 99.7%), raising the q score (+0.187) and lowering the t (-0.997) and g (-0.747) scores; the formula calls them q (81%).
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.2% of jets, neuron 0.15. Mostly gluons (58%, with 30% quarks), 7.21% of jets: mass 7.9 GeV, width 0.0003, somewhat soft (sum pT 631 GeV) with pT shared among many particles (8th particle 39.76 GeV), 91.5% of it inside 0.025 of the axis. Because the jets are not hard, LHA < 0.216 and log_sum_pt < 6.8 passes for all (-3.97) and cancels LHA < 0.216 (+1.782) and e2 < 0.0356 (+1.864); sum_pt_top2 < 548 and girth2_top3 < 0.00395 (-1.535) and dr_0 < 0.0216 (95.6%, -1.479) add to the loss, and z_7 < 0.0715 passes for only 67.6%. The neuron stays near zero (0.147, on for 18%); the formula calls them g (90%).
- **very light (8 GeV), very narrow, leading particle 43% of pT, high pT** — 5.9% of jets, neuron 6.74. Mostly quarks (68%, with 17% gluons), 5.94% of jets: mass 8.3 GeV, width 0.0001, hard (sum pT 969 GeV, leading particle 416.16 GeV), soft tail (8th particle 24.24 GeV), 97.94% of the pT inside 0.025 of the axis. The bonuses log_sum_pt > 6.57 and dr_0 < 0.0216 (+2.72), z_7 < 0.0715 (+2.693), LHA < 0.216 (+2.427), e2 < 0.0356 (+2.066) and width < 0.00264 and centroid_offset < 0.0236 (+1.647) all pass, while LHA < 0.216 and log_sum_pt < 6.8 almost never does; log_sum_pt > 6.57 and lam1 < 0.012 (-4.224), dr_0 < 0.0216 (-2.116) and sum_pt > 869 (-2.114) take back less. The neuron is high (6.735), raising the q score (+0.316) and lowering the t (-1.684) and g (-1.263) scores; the formula calls them q (90%).
- **medium-mass (53 GeV), narrow, leading particle 47% of pT, high pT** — 4.0% of jets, neuron 3.07. A W/Z mixture (W 39%, Z 32%, gluons 14%, quarks 13%), 3.99% of jets: mass 52.9 GeV, width 0.0037, hard (sum pT 979 GeV) with a dominant leading particle (456.94 GeV, nearly twice the average), pT within 0.05 of the axis. z_7 < 0.0715 (+2.619), sum_pt_top5 > 752 (+1.243), e2 < 0.0356 (89.8%, +1.083) and log_sum_pt > 6.57 and girth2_top2 < 0.0063 (94.4%, +0.982) outweigh log_sum_pt > 6.57 and lam1 < 0.012 (-3.065) and sum_pt > 869 (-2.317). The neuron sits at 3.066 (on for 79.8%), lowering the t and g scores and raising the q score a little; the formula calls them W (53%).
- **very light (18 GeV), very narrow, leading particle 59% of pT, high pT** — 1.5% of jets, neuron 9.88. Mostly quarks (75%), 1.45% of jets: mass 18.0 GeV, width 0.0006, hard (sum pT 985 GeV) and carried by about four particles (leading 579.56 GeV, 6th only 13.38 GeV, 8th 8.02 GeV), 94.45% of the pT inside 0.025 of the axis. The few-particle tests decide it: pt_5 < 24.6 (+3.726) and z_7 < 0.0715 (+3.657) pass for all, partly cancelled by z_7 < 0.0324 and pt_5 < 29.9 (-3.443), and with log_sum_pt > 6.57 and dr_0 < 0.0216 (95.5%, +2.79) and LHA < 0.216 (+2.343) they beat log_sum_pt > 6.57 and lam1 < 0.012 (-4.27) and sum_pt > 869 (-2.49). The neuron reaches its highest value, 9.881, raising the q score (+0.463) and strongly lowering the t (-2.47) and g (-1.853) scores; the formula calls them q (92%).
- **very light (17 GeV), very narrow, leading particle 45% of pT, high pT** — 1.0% of jets, neuron 1.69. A gluon-quark mixture (gluons 48%, quarks 36%), 0.99% of jets: mass 17.0 GeV, width 0.0004, very hard (sum pT 1239 GeV, leading particle 555.38 GeV), with 90.24% of the pT inside 0.025 of the axis. The high-pT penalties are at their largest: sum_pt > 869 (-7.771), log_sum_pt > 6.57 and lam1 < 0.012 (-7.367), log_sum_pt > 6.9 and dr_0 < 0.0412 (98.5%, -3.65) and log_sum_pt > 6.9 and centroid_offset < 0.0184 (-2.513), against log_sum_pt > 6.57 and dr_0 < 0.0216 (90.9%, +4.093), sum_pt_top5 > 752 (+3.388), z_7 < 0.0715 (+2.465) and LHA < 0.216 (+2.135). The neuron is on for 49.3% (mean 1.688), lowering the t and g scores; the formula splits them g 59% / q 36%.

### neuron 15: slightly wider than a W (moderate)

- **What it measures:** Pushed up for width < 0.0132 but switched down for narrow jets (width < 0.00668), pushed up for small e2 (< 0.0411) and down for very small e2_sq (< 0.00817), so it responds to jets just wider than the typical W but not broad; it rises with the number of particles at 0.05 ≤ ΔR < 0.1 and above 10 GeV and falls with τ21 (rank correlations 0.365, 0.357, -0.341). Z jets sit highest (1.03), then tops (0.45), with gluons (0.21), quarks (0.14) and W (0.10) low.
- *computed — its value:* largest for Z (1.03), then t (0.45), then g (0.21), then q (0.14), then W (0.10); it separates Z jets from the rest best (AUC 0.74: large for Z)
- **How the class scores use it:** W jets sit lowest on it, so it is evidence against a W: it lowers the W score (-8%). It also lowers the Z score, but only slightly (-2%), even though Z jets sit highest; it does not (or hardly) enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-8%), Z (-2%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -2.06
if width < 0.013: z += 519 × (0.013 − width)
if width < 0.0067: z += -1249 × (0.0067 − width)
if e2 < 0.041: z += 123 × (0.041 − e2)
if lam1 < 0.0084: z += -498 × (0.0084 − lam1)
if lam1 < 0.0065: z += 719 × (0.0065 − lam1)
if mass_over_sum_pt_sq < 0.0072: z += 577 × (0.0072 − mass_over_sum_pt_sq)
if e2_sq < 0.0082: z += -446 × (0.0082 − e2_sq)
if e2 < 0.025: z += 245 × (0.025 − e2)
if e2_sq < 0.012: z += -231 × (0.012 − e2_sq)
if girth2 < 0.019: z += 124 × (0.019 − girth2)
if width < 0.0075: z += -453 × (0.0075 − width)
if lam1 < 0.012: z += -183 × (0.012 − lam1)
if girth > 0.034: z += 28.78 × (girth − 0.034)
if z_dr_0p1_0p2 < 0.328: z += 3.82 × (0.328 − z_dr_0p1_0p2)
if mass_over_sum_pt < 0.068: z += -37.41 × (0.068 − mass_over_sum_pt)
if mass < 36.23: z += -0.062 × (36.23 − mass)
if lam2 < 0.0011: z += -662 × (0.0011 − lam2)
if girth2_top3 < 0.0022: z += -703 × (0.0022 − girth2_top3)
if LHA > 0.347: z += -53.71 × (LHA − 0.347)
if e2 < 0.063: z += -10.66 × (0.063 − e2)
if tau21 < 0.238 and z_dr_0p2_0p4 < 0.206: z += 36.50 × (0.238 − tau21) × (0.206 − z_dr_0p2_0p4)
if width < 0.0061: z += 141 × (0.0061 − width)
if lam2 < 0.00031: z += -1704 × (0.00031 − lam2)
if tau21 < 0.238: z += 5.14 × (0.238 − tau21)
if tau21 < 0.238 and girth2_top5 > 0.0083: z += -2024 × (0.238 − tau21) × (girth2_top5 − 0.0083)
if mass > 80.40: z += -0.191 × (mass − 80.40)
if LHA > 0.177 and sum_pt_top3 > 353: z += 0.037 × (LHA − 0.177) × (sum_pt_top3 − 353)
if z_dr_0p05_0p1 > 0.751: z += -9.22 × (z_dr_0p05_0p1 − 0.751)
if LHA > 0.177: z += 2.38 × (LHA − 0.177)
if n_dr_0p1_0p2 < 1.00: z += 0.336 × (1.00 − n_dr_0p1_0p2)
if width < 0.0075 and e2 > 0.025: z += -44515 × (0.0075 − width) × (e2 − 0.025)
if girth2_top2 < 0.0076: z += 34.19 × (0.0076 − girth2_top2)
if tau21 < 0.238 and z_dr_0p05_0p1 < 0.588: z += -8.18 × (0.238 − tau21) × (0.588 − z_dr_0p05_0p1)
if width < 0.0061 and log_sum_pt > 6.90: z += -4771 × (0.0061 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 23.07 × (log_sum_pt − 6.90)
if n_dr_0_0p05 < 2.00: z += 0.109 × (2.00 − n_dr_0_0p05)
if tau21 < 0.238 and mass < 76.66: z += -0.068 × (0.238 − tau21) × (76.66 − mass)
if n_dr_0p05_0p1 > 5.00: z += 0.230 × (n_dr_0p05_0p1 − 5.00)
if D2 < 0.746: z += -0.631 × (0.746 − D2)
if tau21 < 0.238 and max_dr < 0.122: z += -111 × (0.238 − tau21) × (0.122 − max_dr)
if z_dr_0p1_0p2 < 0.328 and n_dr_0p2_0p4 > 0: z += -0.862 × (0.328 − z_dr_0p1_0p2) × (n_dr_0p2_0p4 − 0)
if mass > 80.40 and z_dr_0p2_0p4 < 0.206: z += 0.661 × (mass − 80.40) × (0.206 − z_dr_0p2_0p4)
if z_dr_0p05_0p1 > 0.751 and n_dr_0p2_0p4 < 2.00: z += 1.17 × (z_dr_0p05_0p1 − 0.751) × (2.00 − n_dr_0p2_0p4)
if planar_flow < 0.084: z += -2.16 × (0.084 − planar_flow)
if e2 < 0.032: z += -3.16 × (0.032 − e2)
if LHA > 0.177 and z_top5 > 0.865: z += -118 × (LHA − 0.177) × (z_top5 − 0.865)
if width < 0.0075 and planar_flow < 0.061: z += -1617 × (0.0075 − width) × (0.061 − planar_flow)
if LHA > 0.177 and dr_7 < 0.042: z += -339 × (LHA − 0.177) × (0.042 − dr_7)
if girth2 < 0.019 and mass_top2 > 22.84: z += -1.85 × (0.019 − girth2) × (mass_top2 − 22.84)
if lam1 < 0.0065 and pt1_dr01 > 1.22: z += 6.63 × (0.0065 − lam1) × (pt1_dr01 − 1.22)
if log_sum_pt > 6.90 and mean_phi > 0.017: z += 50423 × (log_sum_pt − 6.90) × (mean_phi − 0.017)
if D2 < 0.746 and pt_7 < 23.22: z += -0.166 × (0.746 − D2) × (23.22 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **very light (8 GeV), very narrow, pT spread over several particles** — 28.1% of jets, neuron 0.00. Mostly quarks (46%, with 32% gluons), 28.08% of jets: mass 7.9 GeV, width 0.0003, with 86.79% of the pT inside 0.025 of the axis. The narrow-jet switch width < 0.00668 (-7.979) with lam1 < 0.00838 (-4.044), e2_sq < 0.00817 (-3.59) and width < 0.00752 (-3.277) cancels width < 0.0132 (+6.713), e2 < 0.0245 (+4.85), lam1 < 0.00651 (+4.486), e2 < 0.0411 (+4.483) and mass_over_sum_pt_sq < 0.00718 (+4.074). The neuron is off for all of them and adds nothing to the scores; the formula splits them q 52% / g 38%.
- **medium-mass (59 GeV), average width, pT spread over several particles** — 17.6% of jets, neuron 1.56. Mostly Z (53%, with 19% W and 16% tops), 17.64% of jets: mass 58.7 GeV, width 0.0075, with 63.23% of the pT at 0.05-0.1 from the axis. They are wide enough to escape width < 0.00668 (27.8% pass) yet narrow enough for width < 0.0132 (+2.982), girth2 < 0.0188 (+1.408) and girth > 0.0336 (+1.31), and the two-prong tests tau21 < 0.238 and z_dr_0p2_0p4 < 0.206 (85%, +0.876) and tau21 < 0.238 (+0.62) add more; e2_sq < 0.0117 (-1.045), lam1 < 0.012 (-0.864) and lam2 < 0.00113 (95.7%) take back part. The neuron sits at its highest, 1.56 (on for 84.5%), lowering the W score (-1.073) and, less, the Z score (-0.244); the formula calls them Z (60%).
- **medium-mass (51 GeV), average width, pT spread over several particles** — 14.7% of jets, neuron 0.31. Mostly W (52%, with 26% Z), 14.66% of jets: mass 50.9 GeV, width 0.0053, with 50.74% of the pT at 0.05-0.1 and 33.76% at 0.025-0.05 from the axis. width < 0.0132 (+4.094), girth2 < 0.0188 (+1.674), mass_over_sum_pt_sq < 0.00718 (+1.273) and e2 < 0.0411 (98.2%) are cancelled by width < 0.00668 (99%, -1.683), lam1 < 0.00838 (-1.602), e2_sq < 0.0117 (-1.544), e2_sq < 0.00817 (-1.424) and lam1 < 0.012 (-1.251). The neuron is on for 28.9% (mean 0.307), lowering the W score only a little; the formula calls them W (70%).
- **light (40 GeV), narrow, pT spread over several particles** — 10.8% of jets, neuron 0.23. Mostly W (39%, with 25% Z and 17% gluons), 10.78% of jets: mass 40.3 GeV, width 0.0036, with 57.78% of the pT at 0.025-0.05 from the axis. width < 0.0132 (+4.997), e2 < 0.0411 (+2.452), mass_over_sum_pt_sq < 0.00718 (+2.352), lam1 < 0.00651 (+2.205) and girth2 < 0.0188 (+1.89) are cancelled by width < 0.00668 (-3.845), lam1 < 0.00838 (-2.461), e2_sq < 0.00817 (-2.259) and e2_sq < 0.0117 (-1.976). The neuron is on for 24.6% (mean 0.235), lowering the W score slightly; the formula calls them W (56%).
- **light (23 GeV), very narrow, pT spread over several particles** — 10.4% of jets, neuron 0.04. A mixture of gluons (37%), quarks (25%), W (19%) and Z (13%), 10.44% of jets: mass 23.1 GeV, width 0.0016, with 38.58% of the pT inside 0.025 and 48.58% at 0.025-0.05 from the axis. width < 0.00668 (-6.321), lam1 < 0.00838 (-3.429), e2_sq < 0.00817 (-3.136) and width < 0.00752 (-2.675) cancel width < 0.0132 (+6.025), lam1 < 0.00651 (+3.6), mass_over_sum_pt_sq < 0.00718 (+3.486), e2 < 0.0411 (+3.416) and e2 < 0.0245 (+2.73). The neuron is on for 5.9% and adds almost nothing; the formula calls them g (52%).
- **medium-mass (68 GeV), very wide, pT spread over several particles, low pT** — 9.6% of jets, neuron 0.40. Mostly tops (76%, with 16% gluons), 9.61% of jets: mass 67.7 GeV, width 0.0163, soft (sum pT 567 GeV), with 39.95% of the pT at 0.05-0.1 and 28.65% at 0.1-0.15 from the axis. width < 0.0132 passes for only 27.4%, so girth > 0.0336 (+2.237), girth2 < 0.0188 (75.7%), LHA > 0.177 and the partial two-prong tests carry what is left, against LHA > 0.347 (71.2%, -1.16), tau21 < 0.238 and girth2_top5 > 0.00833 (52.2%, -0.838) and mass > 80.4 (23%). The neuron is on for 36.3% (mean 0.398), lowering the W score a little; the formula calls them t (96%).
- **medium-mass (87 GeV), very wide, pT spread over several particles, low pT** — 5.3% of jets, neuron 0.00. Mostly tops (89%), 5.26% of jets: mass 86.7 GeV, width 0.028, soft (sum pT 544 GeV), with almost no pT inside 0.05 and 35.97% at 0.1-0.15 from the axis. LHA > 0.347 passes for 99.8% (-4.714) and mass > 80.4 for 63.5% (-2.1), outweighing girth > 0.0336 (+3.539) and LHA > 0.177 (+0.613); width < 0.0132 never passes. The neuron is off for essentially all of them and adds nothing to the scores; the formula calls them t (98%).
- **heavy (91 GeV), very wide, pT spread over several particles, low pT** — 2.4% of jets, neuron 0.01. Mostly tops (65%) mixed with 25% gluons, 2.44% of jets: mass 91.3 GeV, width 0.0263, with no pT inside 0.025 and 41.05% at 0.1-0.15 from the axis; tau21 < 0.238 passes for all, so they look two-pronged. The wide two-prong penalty tau21 < 0.238 and girth2_top5 > 0.00833 (99.9%, -5.876) joins LHA > 0.347 (99.6%, -4.7) and mass > 80.4 (67.5%, -2.834), outweighing girth > 0.0336 (+3.483) and tau21 < 0.238 (+0.857). The neuron is on for 0.3% and adds nothing to the scores; the formula calls them t (87%), with g 12%.
- **very light (15 GeV), very narrow, leading particle 45% of pT, high pT** — 1.1% of jets, neuron 0.04. A gluon-quark mixture (gluons 48%, quarks 38%), 1.08% of jets: mass 15.2 GeV, width 0.0003, very hard (sum pT 1225 GeV, leading particle 549.45 GeV), with 91.6% of the pT inside 0.025 of the axis. Besides the usual narrow-jet balance (width < 0.00668 -7.94 against width < 0.0132 +6.697), the hard-jet pair width < 0.0061 and log_sum_pt > 6.9 (-5.7) and log_sum_pt > 6.9 (+4.828) cancels, and e2 < 0.0245, e2 < 0.0411, lam1 < 0.00651 and mass_over_sum_pt_sq < 0.00718 (about +4 each) balance lam1 < 0.00838 (-4.024). The neuron is on for 2.6% and adds almost nothing; the formula splits them g 60% / q 37%.

### neuron 8: narrow core, nothing at wide angle (minor)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00502, its strongest term), more so with little pT at 0.2 ≤ ΔR < 0.4 (below 0.101), and pushed down for the very narrowest jets (girth < 0.0611) and for light, centred jets (mass < 29.6 GeV with centroid offset < 0.0236); it falls with max ΔR, centroid offset and width (rank correlations -0.52, -0.501, -0.496). Quarks (1.12) and gluons (1.07) sit highest, then W (0.53), with Z (0.24) and tops (0.14) lowest.
- *computed — its value:* largest for q (1.12), then g (1.07), then W (0.53), then Z (0.24), then t (0.14); it separates t jets from the rest best (AUC 0.28: small for t)
- **How the class scores use it:** A single narrow core is evidence against a two-prong W, so it lowers the W score (-5%); it raises the q score slightly (+2%), in line with quarks sitting highest, and the t score (+3%), a small correction since tops sit lowest on it. It does not (or hardly) enter the g or Z scores.
- *computed — used by:* raises the score of q (+2%), t (+3%); lowers the score of W (-5%); does not (or hardly) enter the score of g, Z (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.303
if width < 0.005: z += 1284 × (0.005 − width)
if girth < 0.061 and width < 0.005: z += -20337 × (0.061 − girth) × (0.005 − width)
if width < 0.005 and z_dr_0p2_0p4 < 0.101: z += 6564 × (0.005 − width) × (0.101 − z_dr_0p2_0p4)
if girth2 < 0.0067 and centroid_offset < 0.024: z += 25556 × (0.0067 − girth2) × (0.024 − centroid_offset)
if girth < 0.061: z += -47.87 × (0.061 − girth)
if mass < 29.64 and centroid_offset < 0.024: z += -6.56 × (29.64 − mass) × (0.024 − centroid_offset)
if LHA < 0.197 and lam2 < 0.00031: z += -78874 × (0.197 − LHA) × (0.00031 − lam2)
if max_dr < 0.177: z += 6.37 × (0.177 − max_dr)
if girth < 0.061 and lam2 < 0.00019: z += 151983 × (0.061 − girth) × (0.00019 − lam2)
if e2 < 0.025 and centroid_offset < 0.031: z += 2641 × (0.025 − e2) × (0.031 − centroid_offset)
if mass < 21.78: z += -0.087 × (21.78 − mass)
if LHA < 0.197: z += -15.39 × (0.197 − LHA)
if max_dr < 0.177 and lam2 < 0.00019: z += 35040 × (0.177 − max_dr) × (0.00019 − lam2)
if LHA < 0.197 and width < 0.00056: z += 30981 × (0.197 − LHA) × (0.00056 − width)
if mass < 15.45 and lam2 < 0.00031: z += -429 × (15.45 − mass) × (0.00031 − lam2)
if e2 < 0.017: z += 67.77 × (0.017 − e2)
if width < 0.005 and centroid_offset > 0.0068: z += -23972 × (0.005 − width) × (centroid_offset − 0.0068)
if z_dr_0_0p05 > 0.848 and lam2 < 0.00054: z += -9164 × (z_dr_0_0p05 − 0.848) × (0.00054 − lam2)
if pt_7 > 34.53: z += -0.053 × (pt_7 − 34.53)
if width < 0.005 and pt_7 < 48.72: z += -7.40 × (0.005 − width) × (48.72 − pt_7)
if mass < 29.64: z += -0.028 × (29.64 − mass)
if girth2 < 0.0067: z += -58.06 × (0.0067 − girth2)
if log_sum_pt > 6.70 and z_dr_0p2_0p4 < 0.206: z += -23.77 × (log_sum_pt − 6.70) × (0.206 − z_dr_0p2_0p4)
if girth2 < 0.0067 and tau21 < 0.501: z += -551 × (0.0067 − girth2) × (0.501 − tau21)
if e2 < 0.025 and sum_pt > 788: z += 0.292 × (0.025 − e2) × (sum_pt − 788)
if girth < 0.061 and centroid_offset > 0.0068: z += -1884 × (0.061 − girth) × (centroid_offset − 0.0068)
if mass < 21.78 and centroid_offset < 0.027: z += 2.05 × (21.78 − mass) × (0.027 − centroid_offset)
if girth2 < 0.0067 and planar_flow < 0.401: z += 381 × (0.0067 − girth2) × (0.401 − planar_flow)
if girth2 < 0.0067 and phi_0 > -0.040: z += 1226 × (0.0067 − girth2) × (phi_0 − -0.040)
if log_sum_pt > 6.70: z += -4.10 × (log_sum_pt − 6.70)
if girth2 < 0.0067 and centroid_offset > 0.018: z += -20309 × (0.0067 − girth2) × (centroid_offset − 0.018)
if log_sum_pt > 6.70 and pt_7 < 48.72: z += 0.151 × (log_sum_pt − 6.70) × (48.72 − pt_7)
if pt_7 > 34.53 and width < 0.013: z += 2.89 × (pt_7 − 34.53) × (0.013 − width)
if mass < 15.45: z += 0.045 × (15.45 − mass)
if girth2 < 0.0067 and n_dr_0p05_0p1 > 0: z += -42.38 × (0.0067 − girth2) × (n_dr_0p05_0p1 − 0)
if girth < 0.061 and lam1 > 0.00028: z += -9671 × (0.061 − girth) × (lam1 − 0.00028)
if log_sum_pt > 6.70 and girth2 < 0.019: z += -163 × (log_sum_pt − 6.70) × (0.019 − girth2)
if girth2 < 0.0067 and log_sum_pt > 6.67: z += -399 × (0.0067 − girth2) × (log_sum_pt − 6.67)
if log_sum_pt > 6.70 and centroid_offset < 0.024: z += 128 × (log_sum_pt − 6.70) × (0.024 − centroid_offset)
if width < 0.005 and mass_over_sum_pt_sq > 0.00012: z += -78985 × (0.005 − width) × (mass_over_sum_pt_sq − 0.00012)
if girth2 < 0.00096: z += -320 × (0.00096 − girth2)
if girth < 0.061 and z_dr_0p2_0p4 < 0.056: z += -63.09 × (0.061 − girth) × (0.056 − z_dr_0p2_0p4)
if log_sum_pt > 6.70 and mass_over_sum_pt_sq < 0.00024: z += 19499 × (log_sum_pt − 6.70) × (0.00024 − mass_over_sum_pt_sq)
if mass < 15.45 and z_7 < 0.059: z += -1.41 × (15.45 − mass) × (0.059 − z_7)
if girth2 < 0.0067 and girth2_top3 > 0.0029: z += 117475 × (0.0067 − girth2) × (girth2_top3 − 0.0029)
if e2 < 0.025: z += -5.11 × (0.025 − e2)
if LHA < 0.197 and z_6 > 0.022: z += 37.01 × (0.197 − LHA) × (z_6 − 0.022)
if LHA < 0.197 and mean_phi > -0.00086: z += 398 × (0.197 − LHA) × (mean_phi − -0.00086)
if mass < 8.38: z += 0.034 × (8.38 − mass)
if max_dr < 0.177 and pt1_over_pt0 < 0.465: z += 7.13 × (0.177 − max_dr) × (0.465 − pt1_over_pt0)
if mass < 21.78 and centroid_offset > 0.031: z += -2.24 × (21.78 − mass) × (centroid_offset − 0.031)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **medium-mass (63 GeV), wide, pT spread over several particles** — 48.8% of jets, neuron 0.14. A broad mixture of tops (37%), Z (27%), W (21%) and gluons (10%), 48.8% of jets: mass 62.8 GeV, width 0.0118, somewhat soft (sum pT 655 GeV), with the pT spread over 0.025-0.15 from the axis. They are too wide for the main test width < 0.00502 (7.2% pass), so only small terms act: pt_7 > 34.5 (53.3%, -0.227) against max_dr < 0.177 (59.9%, +0.199) and max_dr < 0.177 and lam2 < 0.000195 (45.2%, +0.124). The neuron stays near zero (0.139, on for 29.1%) and barely moves the scores; the formula splits them t 41% / Z 29% / W 26%.
- **light (44 GeV), narrow, pT spread over several particles** — 8.0% of jets, neuron 0.55. Mostly W (46%, with 26% Z and 13% gluons), 7.95% of jets: mass 43.8 GeV, width 0.004, with 55.39% of the pT at 0.025-0.05 from the axis. width < 0.00502 (99.7%, +1.32), width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+0.614) and girth2 < 0.00668 and centroid_offset < 0.0236 (75.2%, +0.537) are partly cancelled by girth < 0.0611 (86.2%, -0.453), girth2 < 0.00668 and tau21 < 0.501 (93.2%, -0.429) and girth < 0.0611 and lam1 > 0.000276 (-0.327). The neuron is on for 44.9% (mean 0.554), lowering the W score (-0.139) and raising the t score a little; the formula calls them W (66%).
- **very light (8 GeV), very narrow, pT spread over several particles** — 7.0% of jets, neuron 1.38. A gluon-quark mixture (gluons 45%, quarks 40%), 7.04% of jets: mass 8.0 GeV, width 0.0002, with 95.44% of the pT inside 0.025 of the axis. The narrowness bonuses are large: width < 0.00502 (+6.188), width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+3.193), girth2 < 0.00668 and centroid_offset < 0.0236 (+2.637) and girth < 0.0611 and lam2 < 0.000195 (+1.217); against them girth < 0.0611 and width < 0.00502 (-4.788), girth < 0.0611 (-2.338), mass < 29.6 and centroid_offset < 0.0236 (-2.216) and LHA < 0.197 and lam2 < 0.000306 (-1.683) all pass. The neuron sits at 1.382 (on for 90%), lowering the W score (-0.346) and raising the t (+0.259) and q scores; the formula splits them g 58% / q 42%.
- **very light (5 GeV), very narrow, pT spread over several particles** — 7.0% of jets, neuron 0.95. Mostly quarks (57%, with 32% gluons), 7.03% of jets: mass 5.5 GeV, width 0.0001, with 99.17% of the pT inside 0.025 of the axis. The same bonuses pass (width < 0.00502 +6.365, girth2 < 0.00668 and centroid_offset < 0.0236 +3.469, width < 0.00502 and z_dr_0p2_0p4 < 0.101 +3.285), but the very-narrow penalties are larger than in group 2: girth < 0.0611 and width < 0.00502 (-5.507), mass < 29.6 and centroid_offset < 0.0236 (-3.253), girth < 0.0611 (-2.615), LHA < 0.197 and lam2 < 0.000306 (-2.582) and LHA < 0.197 (-1.721). The neuron sits at 0.95 (on for 88.5%), lowering the W score and raising the t and q scores a little; the formula calls them q (70%).
- **light (35 GeV), narrow, pT spread over several particles** — 6.1% of jets, neuron 0.92. A mixture led by W (36%) with gluons (21%), Z (20%) and quarks (15%), 6.13% of jets: mass 34.7 GeV, width 0.0027, with 61.5% of the pT at 0.025-0.05 from the axis. width < 0.00502 (+3.0), width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+1.452) and girth2 < 0.00668 and centroid_offset < 0.0236 (73.3%, +0.681) are partly cancelled by girth < 0.0611 and width < 0.00502 (-0.919), girth < 0.0611 (-0.914), width < 0.00502 and centroid_offset > 0.00679 (90.9%, -0.671) and girth2 < 0.00668 and tau21 < 0.501 (90%, -0.515). The neuron is on for 52% (mean 0.92), lowering the W score (-0.23) and raising the t score (+0.172); the formula splits them W 55% / g 31%.
- **very light (8 GeV), very narrow, leading particle 46% of pT, high pT** — 5.6% of jets, neuron 0.65. Mostly quarks (70%, with 17% gluons), 5.62% of jets: mass 7.7 GeV, width 0.0001, hard (sum pT 1018 GeV, leading particle 469.74 GeV), with 98.53% of the pT inside 0.025 of the axis. As in group 3, width < 0.00502 (+6.348), girth2 < 0.00668 and centroid_offset < 0.0236 (+3.588) and width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+3.274) meet girth < 0.0611 and width < 0.00502 (-5.577), mass < 29.6 and centroid_offset < 0.0236 (99.9%, -3.062), LHA < 0.197 and lam2 < 0.000306 (-2.765) and girth < 0.0611 (-2.655). The neuron sits at 0.648 (on for 75.3%), lowering the W score and raising the t score slightly; the formula calls them q (88%).
- **very light (17 GeV), very narrow, pT spread over several particles** — 4.7% of jets, neuron 2.61. A quark-gluon mixture (quarks 44%, gluons 42%), 4.68% of jets: mass 17.4 GeV, width 0.0006, with 81.81% of the pT inside 0.025 of the axis. The narrowness bonuses pass in full (width < 0.00502 +5.679, width < 0.00502 and z_dr_0p2_0p4 < 0.101 +2.916, girth2 < 0.00668 and centroid_offset < 0.0236 +2.615, e2 < 0.0245 and centroid_offset < 0.0312 +0.942), while the heavier mass makes the lightness penalties smaller (mass < 29.6 and centroid_offset < 0.0236 -1.323, LHA < 0.197 and lam2 < 0.000306 -1.012) next to girth < 0.0611 and width < 0.00502 (-3.805). The neuron is high (2.608, on for 96.3%), lowering the W score (-0.652) and raising the t (+0.489) and q (+0.163) scores; the formula splits them q 51% / g 48%.
- **light (25 GeV), very narrow, pT spread over several particles** — 4.6% of jets, neuron 2.60. Mostly gluons (40%) with 31% quarks and 15% W, 4.57% of jets: mass 25.3 GeV, width 0.0014, with 49.28% of the pT inside 0.025 and 40.68% at 0.025-0.05 from the axis. width < 0.00502 (+4.644), width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+2.36), girth2 < 0.00668 and centroid_offset < 0.0236 (+1.639) and max_dr < 0.177 (87.8%, +0.537) outweigh girth < 0.0611 and width < 0.00502 (-2.273), girth < 0.0611 (-1.472) and girth2 < 0.00668 and tau21 < 0.501 (84.3%, -0.537). The neuron is high (2.598, on for 90.1%), lowering the W score and raising the t and q scores; the formula splits them g 54% / q 34%.
- **very light (7 GeV), very narrow, pT spread over several particles** — 4.4% of jets, neuron 0.15. A mixture of gluons (36%), W (25%), quarks (18%) and Z (17%), 4.4% of jets: mass 7.0 GeV, width 0.0005, with 72.48% of the pT inside 0.025 and 26.87% at 0.025-0.05 from the axis. width < 0.00502 (+5.768), width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+2.976) and max_dr < 0.177 (+0.901) are eaten up by girth < 0.0611 and width < 0.00502 (-3.609), girth < 0.0611 (-1.886), the off-centre tests width < 0.00502 and centroid_offset > 0.00679 (all, -1.397) and girth < 0.0611 and centroid_offset > 0.00679 (-0.951), and the light-mass tests mass < 21.8 (-1.292) and mass < 15.5 and lam2 < 0.000306 (98.2%, -0.981). The neuron is on for 23.8% (mean 0.15) and barely moves the scores; the formula splits them g 54% / W 31%.
- **very light (9 GeV), very narrow, pT spread over several particles** — 3.8% of jets, neuron 0.02. A mixture of gluons (33%), Z (26%), W (20%) and quarks (13%), 3.79% of jets: mass 9.4 GeV, width 0.0015, with only 11.41% of the pT inside 0.025 but 76.97% at 0.025-0.05 from the axis. width < 0.00502 (+4.531) and width < 0.00502 and z_dr_0p2_0p4 < 0.101 (+2.338) are cancelled by the off-centre tests width < 0.00502 and centroid_offset > 0.00679 (-2.204), girth2 < 0.00668 and centroid_offset > 0.0184 (-1.555) and girth < 0.0611 and centroid_offset > 0.00679 (-1.154), plus girth < 0.0611 and width < 0.00502 (-1.812) and mass < 21.8 (97.3%); the centred bonus girth2 < 0.00668 and centroid_offset < 0.0236 passes for only 2.6%. The neuron is on for just 4.2% and adds almost nothing; the formula splits them g 52% / Z 38%.

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs girth2 > 0.0188 (pushes up, its dominant term) and is pushed down for e2 > 0.0634 and mass/pT > 0.131; it follows the number of particles above 10 GeV (rank correlation 0.831). Only tops reach it with any frequency (non-zero for 18.9% of them), so tops sit highest (0.21), then gluons (0.06) and quarks (0.02), with W and Z at 0.00.
- *computed — its value:* largest for t (0.21), then g (0.06), then q (0.02), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.59: large for t)
- **How the class scores use it:** It does not (or hardly) enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** 

```
z = -0.795
if girth2 > 0.019: z += 261 × (girth2 − 0.019)
if e2 > 0.063: z += -50.63 × (e2 − 0.063)
if mass_over_sum_pt > 0.131: z += -31.68 × (mass_over_sum_pt − 0.131)
if girth2 > 0.019 and pt_7 > 15.55: z += 2.70 × (girth2 − 0.019) × (pt_7 − 15.55)
if mass > 91.19: z += 0.057 × (mass − 91.19)
if girth2 > 0.019 and lam2 > 0.00054: z += 11113 × (girth2 − 0.019) × (lam2 − 0.00054)
if n_dr_0p2_0p4 > 1.00: z += 0.192 × (n_dr_0p2_0p4 − 1.00)
if girth2_top2 > 0.014 and width > 0.013: z += -1267 × (girth2_top2 − 0.014) × (width − 0.013)
if girth2_top2 > 0.014 and z_6 < 0.089: z += -1155 × (girth2_top2 − 0.014) × (0.089 − z_6)
if n_dr_0p2_0p4 > 1.00 and lam2 < 0.0034: z += -55.60 × (n_dr_0p2_0p4 − 1.00) × (0.0034 − lam2)
if girth2_top2 > 0.014: z += 12.32 × (girth2_top2 − 0.014)
if girth2 > 0.019 and mass < 80.40: z += -3.88 × (girth2 − 0.019) × (80.40 − mass)
if mean_phi > 0.026: z += 8.32 × (mean_phi − 0.026)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **light (36 GeV), narrow, pT spread over several particles** — 91.3% of jets, neuron 0.00. Almost all jets (91.32%), with every class present in similar shares (W 22%, Z 22%, gluons 21%, quarks 21%, tops 14%): mass 35.9 GeV, width 0.0044, average pT and spread. The neuron's gate girth2 > 0.0188 passes for only 1.3% of these jets, and the other if-statements (n_dr_0p2_0p4 > 1, e2 > 0.0634, mass > 91.2) pass for a few percent at most, so every amount is close to zero. The neuron is off for essentially all of them and adds nothing to any score; the formula's decisions spread over all classes (W 26%, g 22%).
- **medium-mass (76 GeV), very wide, pT spread over several particles, low pT** — 2.3% of jets, neuron 0.08. Mostly tops (83%, with 12% gluons), 2.34% of jets: mass 76.4 GeV, width 0.0221, soft (sum pT 538 GeV), with 39.21% of the pT at 0.1-0.15 from the axis. girth2 > 0.0188 passes for all but adds only +0.864, and e2 > 0.0634 (82.3%, -0.417) and mass_over_sum_pt > 0.131 (92.6%, -0.373) take most of it back; girth2 > 0.0188 and pt_7 > 15.6 adds +0.172. The neuron is on for 17.8% (mean 0.077) and lowers the t score only slightly (-0.039); the formula calls them t (98%).
- **medium-mass (83 GeV), very wide, pT spread over several particles, low pT** — 2.2% of jets, neuron 0.30. Mostly tops (87%), 2.21% of jets: mass 82.7 GeV, width 0.0257, soft (sum pT 534 GeV), with the pT spread from 0.05 to 0.3 from the axis. girth2 > 0.0188 (+1.787), girth2 > 0.0188 and pt_7 > 15.6 (+0.361) and girth2 > 0.0188 and lam2 > 0.000537 (79.3%) against e2 > 0.0634 (96.9%, -0.835) and mass_over_sum_pt > 0.131 (99.2%, -0.753). The neuron is on for 42% (mean 0.297) and lowers the t score a little (-0.149); the formula calls them t (98%).
- **medium-mass (85 GeV), very wide, pT spread over several particles, low pT** — 1.7% of jets, neuron 0.60. Mostly tops (91%), 1.72% of jets: mass 85.3 GeV, width 0.0293, soft (sum pT 511 GeV), with 33.31% of the pT at 0.15-0.2 from the axis. girth2 > 0.0188 now adds +2.729, plus girth2 > 0.0188 and pt_7 > 15.6 (+0.568) and girth2 > 0.0188 and lam2 > 0.000537 (85.8%, +0.488), outweighing e2 > 0.0634 (-1.301) and mass_over_sum_pt > 0.131 (-1.137). The neuron is on for 69.5% (mean 0.602) and lowers the t score (-0.301); the formula calls them t (99%).
- **medium-mass (86 GeV), very wide, pT spread over several particles, low pT** — 0.8% of jets, neuron 0.90. Mostly tops (77%, with 17% gluons), 0.78% of jets: mass 86.2 GeV, width 0.0341, very soft (sum pT 482 GeV), with 33.51% of the pT at 0.2-0.3 from the axis. girth2 > 0.0188 (+3.983) and girth2 > 0.0188 and pt_7 > 15.6 (+0.74) outweigh e2 > 0.0634 (-1.562), mass_over_sum_pt > 0.131 (-1.511) and girth2_top2 > 0.014 and width > 0.0132 (95.1%, -0.563). The neuron is on for 78.8% (mean 0.902) and lowers the t score (-0.451); the formula calls them t (96%).
- **heavy (117 GeV), very wide, pT spread over several particles, high pT** — 0.5% of jets, neuron 0.83. Mostly tops (63%) mixed with 25% gluons, 0.5% of jets: the heaviest-but-one group (mass 117.1 GeV), width 0.0201, hard (sum pT 853 GeV, leading particle 323.71 GeV). Here the mass test mass > 91.2 passes for all (+1.475) and does most of the work, with girth2 > 0.0188 passing for 67.8% (+0.566); mass_over_sum_pt > 0.131 (79.7%, -0.328) and e2 > 0.0634 (61.5%, -0.27) take little back. The neuron is on for 83.7% (mean 0.83) and lowers the t score (-0.415); the formula calls them t (79%), with g 20%.
- **heavy (126 GeV), very wide, pT spread over several particles** — 0.4% of jets, neuron 2.14. Mostly tops (62%) mixed with 28% gluons, 0.41% of jets: the heaviest group (mass 126.4 GeV), wide (width 0.0307), average pT (sum pT 732 GeV), with most of the pT beyond 0.1 from the axis. Both gates pass for all: girth2 > 0.0188 (+3.104) and mass > 91.2 (+2.004), with girth2 > 0.0188 and pt_7 > 15.6 (99.2%, +0.68), against mass_over_sum_pt > 0.131 (-1.328), e2 > 0.0634 (-1.302) and girth2_top2 > 0.014 and z_6 < 0.0891 (88.7%, -0.511). The neuron sits at 2.143 (on for 96.4%) and lowers the t score (-1.072); the formula calls them t (74%), with g 23%.
- **medium-mass (88 GeV), very wide, pT spread over several particles, low pT** — 0.4% of jets, neuron 2.55. Mostly tops (93%), 0.37% of jets: mass 87.7 GeV, width 0.0354, very soft (sum pT 480 GeV), with no pT inside 0.05 and 37.22% at 0.15-0.2 from the axis. girth2 > 0.0188 (+4.329) is joined by girth2 > 0.0188 and lam2 > 0.000537, passing for all (+1.839), as the pT spreads in both directions, and by girth2 > 0.0188 and pt_7 > 15.6 (+0.829); e2 > 0.0634 (-1.946) and mass_over_sum_pt > 0.131 (-1.635) take back less. The neuron sits at 2.549 (on for 98.6%) and lowers the t score (-1.274); the formula calls them t (100%).
- **heavy (93 GeV), very wide, pT spread over several particles, low pT** — 0.3% of jets, neuron 2.35. A top-gluon mixture (tops 51%, gluons 35%), 0.27% of jets: mass 92.8 GeV, width 0.0428, very soft (sum pT 469 GeV), with 48.89% of the pT at 0.2-0.3 from the axis. girth2 > 0.0188 adds +6.24 and girth2 > 0.0188 and pt_7 > 15.6 +1.157, against mass_over_sum_pt > 0.131 (-2.108), e2 > 0.0634 (99.4%, -2.03) and girth2_top2 > 0.014 and width > 0.0132 (99.4%, -1.256). The neuron sits at 2.354 (on for 91.2%) and lowers the t score (-1.177); the formula calls them t (81%), with g 19%.
- **heavy (95 GeV), very wide, pT spread over several particles, low pT** — 0.1% of jets, neuron 6.86. Mostly tops (67%, with 24% gluons), 0.08% of jets: mass 94.7 GeV, the widest group (width 0.0517, eight times the average), very soft with pT shared evenly (leading particle 94.22 GeV), and 55.07% of the pT at 0.2-0.3 from the axis. girth2 > 0.0188 (+8.57) and girth2 > 0.0188 and lam2 > 0.000537 (+3.863) are the largest amounts of this neuron and outweigh e2 > 0.0634 (-2.914), mass_over_sum_pt > 0.131 (-2.744) and girth2_top2 > 0.014 and width > 0.0132 (95.6%, -1.789). The neuron reaches its highest value, 6.861 (on for 97.8%), and lowers the t score strongly (-3.431); the formula calls them t (87%).
