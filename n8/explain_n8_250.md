# What each part of the 250-term formula does (8 particles)

*the formula simplified by hand with the training data (main result)*. Validation accuracy 65.18%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

Every jet is placed on 16 scales (each a clipped sum of if-statements on jet-shape quantities), and each class score adds some scales and subtracts others. Tops are recognised by size: the t score subtracts compactness (neuron 13, on which tops sit far below all other types) and adds overall size (neuron 10: e2, width, mass), so it is essentially a jet-width measure. Quarks and gluons both sit high on the narrow-and-light scale (neuron 9), which feeds both of their scores; they are told apart by how the pT is shared among the particles: the g score adds gluon-likeness (neuron 2: even the 8th-hardest particle is hard) and subtracts quark-likeness (neuron 5: pT concentrated in the few hardest particles), while the q score subtracts overall size (neuron 10). W and Z jets sit high on scales for a compact, centred, elongated two-prong jet with mass/pT in the boson window (neurons 7 and 13 feed both boson scores, neurons 0 and 11 only the W score), and both boson scores subtract the broad-off-centre and wide-angle-radiation scales (neurons 6 and 3), on which tops and gluons sit higher. W and Z are split mainly by width: Z jets sit highest on an intermediate-width scale (neuron 14), which the Z score adds and the W score subtracts, and a 'slightly wider than a W' scale (neuron 15) takes a larger share out of the W score than out of the Z score, consistent with the heavier Z opening wider at the same pT.

## The 5 class scores

### score g: gluon: many hard particles, light jet

High for light, fairly narrow jets whose pT is spread over many hard particles; it averages 2.21 for g jets, 1.40 for q and 1.08 for t, and is near zero for W and Z (AUC 0.81 for gluons against the rest).

Adds gluon-likeness (neuron 2, +36%) and narrow-light-ness (neuron 9, +26%), plus smaller amounts of neurons 1 and 6; subtracts quark-likeness (neuron 5, -15%) and, less, the compact two-prong scale (neuron 0) and neuron 4.

*computed:* largest for g (2.21), then q (1.40), then t (1.08), then W (0.18), then Z (0.04); it separates g jets from the rest best (AUC 0.81: large for g)

### score q: quark: narrow, light, small jet

High for narrow, light jets that are small in e2, width and mass; it averages 2.22 for q jets and 1.49 for g, with tops lower (0.32) and W and Z near or below zero (AUC 0.83 for quarks). Gluons are its main confusion.

Mostly adds narrow-light-ness (neuron 9, +49%); subtracts overall size (neuron 10, -18%) and neuron 4 (-16%); adds a little of neuron 6 and of quark-likeness (neuron 5, +5%).

*computed:* largest for q (2.22), then g (1.49), then t (0.32), then W (-0.00), then Z (-0.16); it separates q jets from the rest best (AUC 0.83: large for q)

### score W: W: compact, centred two-prong jet

High for compact, centred, elongated two-prong jets with W-like mass/pT that are not wider than a W; it averages 2.55 for W jets and 0.51 for Z, is near zero for q, slightly negative for g and strongly negative for t (-2.64) (AUC 0.89).

Adds the compact-centred scale (neuron 11, +24%, its largest input), the two-prong scales (neurons 0 and 7) and compactness (neuron 13); subtracts broad off-centre spread (neuron 6), the 'wider than a W' scales (neurons 15 and 14), wide-angle radiation (neuron 3) and a little of neurons 8 and 9.

*computed:* largest for W (2.55), then Z (0.51), then q (0.03), then g (-0.40), then t (-2.64); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z: two-prong jet, wider than a W

High for two-prong jets in the W/Z mass-to-pT window with an intermediate, Z-sized width; it averages 2.31 for Z jets and 1.58 for W (W is its main confusion), is near zero for q and g, and strongly negative for t (-1.93) (AUC 0.85).

Adds the two-prong W/Z-window scale (neuron 7, +27%, its largest input), neuron 4, compactness (neuron 13), the intermediate-width scale (neuron 14) and a little of neuron 1; subtracts broad off-centre spread (neuron 6, -21%), wide-angle radiation (neuron 3) and a little of neurons 9 and 15.

*computed:* largest for Z (2.31), then W (1.58), then q (0.14), then g (-0.16), then t (-1.93); it separates Z jets from the rest best (AUC 0.85: large for Z)

### score t: top: wide, massive jet

High for wide, massive jets: it follows girth, width and e2 with rank correlations of about 0.90. It averages 2.90 for t jets, 0.38 for Z and 0.18 for W, is near zero for g and negative for q (-1.42) (AUC 0.91).

Subtracts compactness (neuron 13, -46%) and adds overall size (neuron 10, +26%); also subtracts quark-likeness (neuron 5) and adds smaller amounts of neurons 4 and 8.

*computed:* largest for t (2.90), then Z (0.38), then W (0.18), then g (-0.01), then q (-1.42); it separates t jets from the rest best (AUC 0.91: large for t)

## How the scores are assembled (from the neurons' regimes)

Each class score is a bias plus the neurons' values times fixed weights, and each neuron feeds only a few classes, so the formula works like a set of width- and spread-based switches. Narrow jets are sorted into quarks and gluons by neuron 9 (thin core, up to ≈11.7, adding +2.96 to q and +2.0 to g) and neuron 2 (many hard particles, adding up to +2.15 to g only), with neuron 5 (few hard particles) taking away from g (up to −1.59) and t. Jets in the intermediate spread window 0.00302 < girth2 ≤ 0.0118 feed W and Z through neuron 7 (+2.04 to Z, +0.95 to W at most), with neurons 11 and 0 adding to W (+1.6 on 41%, +1.07 on 24%) and neurons 14 and 15 subtracting from W for jets wider than a W (−1.97 and −1.83), which is how W and Z are told apart. Wide jets become tops: the t score starts at a bias of 1.34, gains with jet size from neuron 10 (+1.29 on 59% of jets, up to +5.12) and loses at least 2.07 from neuron 13 whenever width ≤ 0.00902. Broad-jet neurons 6 and 3 act as vetoes on W and Z (up to −3.03 and −3.06 on Z), so top-like spread cannot be mistaken for a boson.

**What goes into score g:** The g score starts from a bias of −0.44 and is built mainly from two neurons. The thin, centred-core regime of neuron 9 (girth2 ≤ 0.00125, centroid_offset ≤ 0.00919; 21% of jets) adds +2.0, and the 'many hard particles' neuron 2 adds +2.15 when all particles lie within max_dr ≤ 0.0838 and the 8th-hardest particle carries z_7 > 0.0649 (9.2%), +1.74 for 0.0408 < z_7 ≤ 0.0649 (13%), +1.19 for 0.0163 < z_7 ≤ 0.0408 (11%), and still +0.91 for wider jets with a hard 8th particle (25%, 45% tops); neuron 9 also adds +0.67 for wider but light jets (mass ≤ 38.7 GeV, 15%) and neuron 6 adds +0.885 for broad jets (width > 0.00902, 9.7%, 75% tops). It is lowered by the quark-like neuron 5 (−1.59 when z_7 ≤ 0.0304 and e2 ≤ 0.0104, 8.0% of jets with 66% quarks; −0.705 for the 18% with e2 ≤ 0.0232 and sum_pt > 717 GeV) and by the compact two-prong regime of neuron 0 (−0.486 on 24% of jets, mostly bosons). So a jet scores high as a gluon when it is narrow and light but spreads its pT over many hard particles rather than a few; the mean score is 2.21 for gluons, but 1.40 for quarks and 1.08 for tops, which also collect the wide-jet contributions.

**What goes into score q:** The q score has a tiny bias (0.031) and is dominated by neuron 9: +2.96 for very thin, centred jets (girth2 ≤ 0.00125, centroid_offset ≤ 0.00919; 21% of jets, 56% quarks), +2.31 when slightly off-axis (4.7%), +0.99 for wider light jets (mass ≤ 38.7 GeV, 15%) and +0.19 on the 50% of heavier jets; neuron 9's wide-top branch (lam2 > 0.00514, 3.2%) adds +2.07 and neuron 6 adds +1.01 for broad jets (9.7%, 75% tops). It is pulled down by jet size, neuron 10 (−0.429 on the 59% bulk with lam2 ≤ 0.00277 and lam1 > 0.00118, −1.71 and −1.16 for the largest jets with lam2 > 0.00277), and by neuron 4 (−0.353 on the 54% of jets with C2 ≤ 0.0505 and tau21 > 0.0968, −0.49 for the two-prong-like 19%, −0.522 at low pT). Physically, a high quark score means a thin single-core jet with small overall size; the mean score is 2.22 for quarks against 1.49 for gluons, so the quark/gluon split is only partly made here.

**What goes into score W:** The W score (bias −0.125) is raised most by neuron 11 for narrow (width ≤ 0.00853), not-too-light (mass > 13.2 GeV), centred jets (+1.6 on 41% of jets), by the elongated compact two-prong regime of neuron 0 (+1.07 on 24%), by neuron 7 in the W/Z spread window 0.00302 < girth2 ≤ 0.0118 (+0.95 on 19%, +0.577 on 24%), and by the narrow regimes of neuron 13 (+0.593 on 22%, +0.358 on 33%). It is cut down whenever the jet is wider than a W: neuron 15 subtracts −1.83 for width > 0.00632 with lam1 > 0.00699 (15%, Z and tops), neuron 14 −1.97 for the Z-sized regime (11%, 59% Z), neuron 6 −2.53 for broad jets (9.7%, 75% tops) and neuron 3 −1.65 for wide non-three-prong jets (6.7%, 63% tops); neuron 8 also subtracts −0.803 for very narrow jets with width ≤ 0.00268 (12.5%, mostly quarks). So jets end up with a high W score when they are compact, centred two-prong objects no wider than a W; the mean score is 2.55 for W jets, 0.51 for Z jets and -2.64 for tops.

**What goes into score Z:** The Z score (bias −0.094) is driven by neuron 7 in the spread window 0.00302 < girth2 ≤ 0.0118: +2.04 when e2_sq > 0.0059 (19% of jets, 50% Z) and +1.24 when e2_sq ≤ 0.0059 (24%, 46% W). Smaller additions come from neuron 4 (+0.294 on the 54% with C2 ≤ 0.0505, +0.408 on the two-prong-like 19%), from neuron 14's Z-sized regime (+0.983 on 11%, 59% Z) and from the narrow regimes of neuron 13 (+0.461 on 22%, +0.279 on 33%). The broad-jet vetoes lower it: neuron 6 (−3.03 for width > 0.00902 and girth > 0.106, 9.7%; −1.87 for girth ≤ 0.106, 5.1%; −0.115 even on the 71% narrow bulk) and neuron 3 (−1.85 on 6.7% and −3.06 on 3.0% of wide, top-like jets). The result is a high score for jets of intermediate spread with two-prong structure; the mean score is 2.31 for Z jets but also 1.58 for W jets, so the W/Z split relies mainly on the W score being cut for wider jets.

**What goes into score t:** The t score starts from the largest bias (1.34) and gets a broad lift from jet size, neuron 10: +1.29 on the 59% bulk, +3.47 for 0.00277 < lam2 ≤ 0.00618 (3.4%, 89% tops) and +5.12 for lam2 > 0.00618 (2.3%, 95% tops), plus +0.47 from neuron 4 on the 54% with C2 ≤ 0.0505 and +0.653 on the two-prong-like 19%. Against this, neuron 13 subtracts heavily from every narrow jet: −3.42 for width ≤ 0.00079 at high pT (22%), −2.63 at lower pT (6.1%), −2.52 and −2.07 for 0.00079 < width ≤ 0.00902 (20% and 33%), and neuron 5 subtracts −2.12 for jets with a nearly empty 8th particle (z_7 ≤ 0.0304, 8.0%) and −0.939 on 18% more. Physically, any jet narrower than width 0.00902 loses at least 2.07, so only wide, large jets keep the bias and the size bonus; the mean score is 2.90 for tops, 0.376 for Z jets and -1.42 for quarks.

## The 16 neurons (most important first)

### neuron 2: gluon-likeness: many hard particles (major)

- **What it measures:** Moved up when the 8th-hardest particle is still hard (pT_7 > 30.2 GeV, and more so above 54.2 GeV) and for total pT below 788 GeV, and down for large angularity (LHA > 0.116); overall it falls with jet mass and spread (rank correlation -0.675 with mass). Gluons sit highest (3.79), then quarks (2.72), with W, Z and top jets lower (1.71-2.07).
- *computed — its value:* largest for g (3.79), then q (2.72), then W (2.07), then Z (1.78), then t (1.71); it separates g jets from the rest best (AUC 0.79: large for g)
- **How the class scores use it:** Since gluons sit highest on it, the g score reads it as direct evidence for a gluon: it raises the g score, where it is the largest input (+36%). It does not enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+36%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** Highest (≈5.01) for jets whose particles all lie within max_dr ≤ 0.0838 and whose 8th-hardest particle still carries z_7 > 0.0649 of the pT (9.2% of jets, mostly gluons, 54%), ≈4.06 for 0.0408 < z_7 ≤ 0.0649 (13%, mostly gluons, 41%) and ≈2.78 for 0.0163 < z_7 ≤ 0.0408 (11%, mostly quarks, 55%); when the 8th particle is nearly empty (z_7 ≤ 0.0163) it drops to ≈0.584 (2.3%, 80% quarks). For wider jets (max_dr > 0.0838) it is ≈2.12 when z_7 > 0.0565 and lam1 > 0.00452 (25%, 45% tops) and ≈0.827 when z_7 ≤ 0.0565 and lam1 > 0.00427 (24%). It feeds only the g score, always upward (+2.15 down to +0.251), with the 13% regime (+1.74), the 25% wide regime (+0.91) and the 9.2% regime (+2.15) mattering most; regime_r2 0.699.

Regimes (a small tree on its quantities; R² 0.699):

- `` — 9.2% of jets, value 5.01 (3.88…6.00), formula right 57%
- `` — 13.3% of jets, value 4.06 (2.81…5.19), formula right 52%
- `` — 4.7% of jets, value 3.73 (2.50…5.06), formula right 53%
- `` — 11.0% of jets, value 2.78 (1.75…3.69), formula right 61%
- `` — 24.6% of jets, value 2.12 (1.06…3.25), formula right 72%
- `` — 11.2% of jets, value 1.80 (0.00…3.19), formula right 58%
- `` — 23.6% of jets, value 0.83 (0.00…1.69), formula right 76%
- `` — 2.3% of jets, value 0.58 (0.00…1.62), formula right 79%

```
z = 3.38
if pt_7 > 30.20: z += 0.173 × (pt_7 − 30.20)
if LHA > 0.116: z += -7.91 × (LHA − 0.116)
if pt_7 < 54.20: z += -0.050 × (54.20 − pt_7)
if sum_pt < 788: z += 0.0056 × (788 − sum_pt)
if z_7 > 0.045: z += -45.70 × (z_7 − 0.045)
if lam1 < 0.0053: z += 267 × (0.0053 − lam1)
if pt_7 > 29.60 and C2 < 0.051: z += -1.91 × (pt_7 − 29.60) × (0.051 − C2)
if log_sum_pt < 6.46: z += 4.86 × (6.46 − log_sum_pt)
if planar_flow < 0.654: z += -0.791 × (0.654 − planar_flow)
if pt_7 > 29.20 and max_dr > 0.075: z += -0.564 × (pt_7 − 29.20) × (max_dr − 0.075)
if log_sum_pt > 6.83: z += 11.40 × (log_sum_pt − 6.83)
if z_7 < 0.018: z += -303 × (0.018 − z_7)
if lam1 < 0.0037 and centroid_offset < 0.0058: z += -32100 × (0.0037 − lam1) × (0.0058 − centroid_offset)
if log_sum_pt > 6.89: z += -13.70 × (log_sum_pt − 6.89)
if girth2 < 4.9e-05: z += -54100 × (4.9e-05 − girth2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Heavy boson/top jets, soft tail** — 21.5% of jets, neuron 0.83, formula right for 72%. A heavy mixture of Z (32%), W (29%) and tops (27%), 21.5% of jets: mass 61.7 GeV, width 0.0084, a hard leading particle (291 GeV) but a soft 8th particle (27.6 GeV), with pT spread mainly at 0.025-0.1 from the axis. The penalties LHA > 0.116 (-1.42) and pt_7 < 54.2 (-1.34) always apply and planar_flow < 0.654 (-0.42) nearly always, while the hard-tail bonus pt_7 > 30.2 fires for only 37%; sum_pt < 788 (60%, +0.39) adds a little. The neuron sits at its lowest on-value here (0.833, on for 83.9%), giving the gluon score only a small push; the formula splits them W / Z evenly (35% each, 25% t) and is right 71.9% of the time.
- **Narrow quark jets, hard leader** — 20.3% of jets, neuron 2.84, formula right for 57%. Mostly quarks (45%) with 23% gluons and 16% W, 20.3% of jets: mass 14.2 GeV, width 0.0007, a hard leading particle (320 GeV), soft tail and 76% of the pT inside 0.025. lam1 < 0.0053 always passes (+1.24) and cancels the always-on pt_7 < 54.2 (-1.38); pt_7 > 30.2 fires for only 36% and LHA > 0.116 (60%) costs little, so the value rests near the base, 2.836 (on for 99.7%). That raises the gluon score by about 1.2 even for these quark-like jets; the formula calls them q (58%) and is right only 57.1% of the time.
- **Heavy boson jets, moderate tail** — 13.7% of jets, neuron 1.64, formula right for 72%. A W (33%) / Z (32%) / top (22%) mixture, 13.7% of jets: mass 58.1 GeV, width 0.0082, slightly below-average pT with a fairly hard 8th particle (39.3 GeV) and most pT at 0.05-0.1. pt_7 > 30.2 always passes (+1.57) but is taken back by LHA > 0.116 (-1.46), pt_7 < 54.2 (-0.75), z_7 > 0.045 (93%, -0.52), pt_7 > 29.6 and C2 < 0.0514 (88%, -0.48) and pt_7 > 29.2 and max_dr > 0.0755 (96%, -0.41). The neuron is on for 98.6% at 1.642, a moderate push to the gluon score; the formula calls them W (42%) and is right 71.5% of the time.
- **Narrow, evenly shared gluon jets** — 10.3% of jets, neuron 4.56, formula right for 53%. Mostly gluons (47%) with 30% quarks, 10.3% of jets: mass 10.6 GeV, width 0.0006, a soft leading particle (199 GeV) and hard 8th particle (43.7 GeV), with 74% of the pT inside 0.025. Both gluon signals fire: pt_7 > 30.2 (+2.34) and lam1 < 0.0053 (+1.28), while LHA > 0.116 hits only 66% of them (-0.27); pt_7 > 29.6 and C2 < 0.0514 (-1.10) and z_7 > 0.045 (-0.78) take some back. The value is high (4.563, always on), raising the gluon score by about 2; the formula calls them g (66%) but is right only 53.2% of the time, since many are quarks.
- **Very soft, wide top/gluon jets** — 10.1% of jets, neuron 3.08, formula right for 66%. Mostly tops (52%) with 30% gluons, 10.1% of jets: mass 46.3 GeV, width 0.0150, very soft (sum pT 447 GeV, leading particle 114 GeV) and spread, with 44% of the pT beyond 0.1. The low-pT tests sum_pt < 788 (+1.92) and log_sum_pt < 6.46 (+1.80) always pass and outweigh LHA > 0.116 (-1.76), pt_7 < 54.2 (-1.18) and z_7 > 0.045 (-1.15). The neuron is always on at 3.082, giving these top-rich jets a gluon-like push of about 1.3; the formula calls them t (59%) and is right 66% of the time, with many gluon-top confusions (32% decided g).
- **Soft, wide top jets, hard tail** — 8.0% of jets, neuron 2.90, formula right for 65%. Mostly tops (43%) mixed with gluons 23%, Z 15%, W 13%, 8.0% of jets: mass 49.1 GeV, width 0.0126, soft (sum pT 509 GeV) with pT evenly shared (8th particle 40.9 GeV) and 37% beyond 0.1. Three bonuses always fire, pt_7 > 30.2 (+1.85), sum_pt < 788 (+1.57) and log_sum_pt < 6.46 (+1.12), against LHA > 0.116 (-1.65), z_7 > 0.045 (-1.64), pt_7 < 54.2 (-0.67) and pt_7 > 29.2 and max_dr > 0.0755 (88%, -0.59). The neuron is always on at 2.903, pushing the gluon score up; the formula calls them t (46%) and is right 65.1% of the time.
- **Heavy boson jets, many hard particles** — 7.7% of jets, neuron 2.31, formula right for 72%. W (32%) and Z (32%) with tops 20% and gluons 13%, 7.7% of jets: mass 58.8 GeV, width 0.0086, a soft leading particle (168 GeV) but hard 8th particle (49.9 GeV), pT mostly at 0.05-0.1. pt_7 > 30.2 always passes and adds a lot (+3.41), but the heavy-jet penalties LHA > 0.116 (-1.50), z_7 > 0.045 (-1.29), pt_7 > 29.6 and C2 < 0.0514 (89%, -1.05) and pt_7 > 29.2 and max_dr > 0.0755 (96%, -0.82) cut it back. The neuron is on for 98.3% at 2.312, a sizeable gluon push to boson jets; the formula calls them W (41%) and is right 71.5% of the time.
- **Many-hard-particle gluon jets** — 4.3% of jets, neuron 4.88, formula right for 62%. Mostly gluons (47%) with quarks 19%, W 18%, Z 13%, 4.3% of jets: mass 22.5 GeV, width 0.0019, pT shared very evenly (8th particle 60.2 GeV, leading 189 GeV) with 59% inside 0.025. pt_7 > 30.2 gives its largest bonus here (+5.19) and pt_7 < 54.2 mostly fails (passes 19%), with lam1 < 0.0053 (86%, +1.01); pt_7 > 29.6 and C2 < 0.0514 (-2.39) and z_7 > 0.045 (-1.44) are the main brakes. The neuron reaches its highest value (4.881), raising the gluon score by about 2.1; the formula calls them g (64%) and is right 61.8% of the time.
- **Hard, single-particle quark jets** — 3.1% of jets, neuron 0.20, formula right for 77%. Mostly quarks (69%), 3.1% of jets: mass 21.2 GeV, hard (sum pT 983 GeV) with a 535 GeV leading particle and a very soft 8th particle (9.3 GeV), 88% of pT inside 0.025. z_7 < 0.0185 always passes (-2.72) and pt_7 < 54.2 (-2.26) with it, while pt_7 > 30.2 never does; lam1 < 0.0053 (95%, +1.19) and log_sum_pt > 6.83 (86%, +0.76) cannot compensate. The neuron is on for only 28.9% (mean 0.201) and barely moves the gluon score; the formula calls them q (82%) and is right 76.9% of the time.
- **Very hard narrow gluon/quark jets** — 1.0% of jets, neuron 2.82, formula right for 61%. Mostly gluons (45%) with 32% quarks, 1.0% of jets: mass 27.0 GeV, width 0.0012, very hard (sum pT 1226 GeV, leading particle 567 GeV) with 77% of pT inside 0.025. The two high-pT tests log_sum_pt > 6.83 (+3.14) and log_sum_pt > 6.89 (-2.96) both pass and nearly cancel; lam1 < 0.0053 (96%, +1.16) and pt_7 > 30.2 (53%, +1.07) balance pt_7 < 54.2 (-1.12). The value stays near the base (2.818, on for 91.1%), raising the gluon score; the formula calls them g (55%) and is right 61.1% of the time.

### neuron 6: broad, off-centre spread (major)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00862 and width < 0.0133 both push it down) and pushed up for lam1 < 0.00802 and e2 < 0.0508; it follows the offset of the pT centroid from the jet axis and the jet width (rank correlations 0.562 and 0.525). Tops sit far highest (4.60), then gluons (1.82) and quarks (0.84), with Z (0.58) and W (0.20) lowest.
- *computed — its value:* largest for t (4.60), then g (1.82), then q (0.84), then Z (0.58), then W (0.20); it separates t jets from the rest best (AUC 0.83: large for t)
- **How the class scores use it:** Boosted W and Z jets sit lowest on it, so the W and Z scores read a broad, off-centre spread as evidence against a boson: it lowers the W score (-12%) and the Z score (-21%). It raises the g and q scores (among non-top jets a broad spread is more QCD-like than boson-like), and it does not enter the t score although tops sit highest on it; the t score gets its width information from neurons 13 and 10.
- *computed — used by:* raises the score of g (+6%), q (+9%); lowers the score of W (-12%), Z (-21%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** Low (≈0.305, on for only 26% of them) for the 71% of jets that are narrow (width ≤ 0.00902), centred (centroid_offset ≤ 0.0316) and high-pT (log_sum_pt > 6.22). It is highest (≈8.09) for wide jets (width > 0.00902) with lam2 ≤ 0.00398 and girth > 0.106 (9.7% of jets, 75% tops), ≈4.99 for girth ≤ 0.106 (5.1%, 62% tops) and ≈4.36 for narrow but off-centre jets (centroid_offset > 0.0414, 2.0%, mostly gluons, 39%). It is a strong veto on W and Z (−2.53 on W and −3.03 on Z in the top regime, −0.115 on Z even in the narrow bulk) while adding to q (+1.01) and g (+0.885); regime_r2 0.705.

Regimes (a small tree on its quantities; R² 0.705):

- `` — 9.7% of jets, value 8.09 (4.25…12.25), formula right 77%
- `` — 5.1% of jets, value 4.99 (1.75…8.38), formula right 66%
- `` — 2.0% of jets, value 4.36 (1.50…7.89), formula right 47%
- `` — 2.2% of jets, value 3.91 (0.75…7.39), formula right 92%
- `` — 3.6% of jets, value 2.17 (0.00…4.38), formula right 46%
- `` — 4.0% of jets, value 1.91 (0.00…5.12), formula right 56%
- `` — 2.1% of jets, value 1.31 (0.00…4.00), formula right 95%
- `` — 71.3% of jets, value 0.30 (0.00…1.12), formula right 64%

```
z = 6.56
if girth2 < 0.0086: z += -1680 × (0.0086 − girth2)
if width < 0.013: z += -540 × (0.013 − width)
if lam1 < 0.008: z += 919 × (0.008 − lam1)
if mass_over_sum_pt > 0.0094: z += -53.20 × (mass_over_sum_pt − 0.0094)
if e2 < 0.051: z += 85.70 × (0.051 − e2)
if max_dr > 0.028: z += 14.70 × (max_dr − 0.028)
if girth > 0.087: z += 126 × (girth − 0.087)
if lam2 < 0.00052: z += -2400 × (0.00052 − lam2)
if e2_sq < 0.0033: z += 626 × (0.0033 − e2_sq)
if mass < 46.60 and z_dr_0p05_0p1 < 0.800: z += 0.058 × (46.60 − mass) × (0.800 − z_dr_0p05_0p1)
if centroid_offset > 0.0078 and lam2 < 0.0035: z += 19800 × (centroid_offset − 0.0078) × (0.0035 − lam2)
if log_sum_pt < 6.57 and pt_7 < 45.40: z += 0.343 × (6.57 − log_sum_pt) × (45.40 − pt_7)
if C2 > 0.010: z += -20.80 × (C2 − 0.010)
if mass_over_sum_pt > 0.017 and tau32 < 0.530: z += -68.70 × (mass_over_sum_pt − 0.017) × (0.530 − tau32)
if centroid_offset > 0.021: z += -59.40 × (centroid_offset − 0.021)
if lam2 > 0.0029: z += -1090 × (lam2 − 0.0029)
if C2 > 0.010 and pt_7 > 32.60: z += 2.56 × (C2 − 0.010) × (pt_7 − 32.60)
if log_sum_pt < 6.71 and z_7 < 0.049: z += 833 × (6.71 − log_sum_pt) × (0.049 − z_7)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.159: z += 766 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.159)
if log_sum_pt < 6.31 and z_7 < 0.071: z += 1090 × (6.31 − log_sum_pt) × (0.071 − z_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 28.1% of jets, neuron 0.25, formula right for 61%. Mostly quarks (47%) with 34% gluons, 28.1% of jets: mass 8.0 GeV, width 0.0002, 90% of pT inside 0.025. The narrow-jet switches girth2 < 0.00862 (-14.07) and width < 0.0133 (-7.05) are nearly balanced by lam1 < 0.00802 (+7.17), e2 < 0.0508 (+3.93), e2_sq < 0.00328 (+1.97) and mass < 46.6 and z_dr_0p05_0p1 < 0.8 (+1.77); the neuron is on for only 20% (mean 0.253) and barely moves the scores. The formula calls them q (54%) and is right 61.3% of the time.
- **Two-prong W/Z jets** — 14.1% of jets, neuron 0.55, formula right for 70%. Mostly W (42%) with 39% Z and 10% tops, 14.1% of jets: mass 55.9 GeV, width 0.0063, 62% of pT at 0.05-0.1. girth2 < 0.00862 (-3.84), width < 0.0133 (-3.76) and mass_over_sum_pt > 0.00941 (-3.63) against lam1 < 0.00802 (+1.70), max_dr > 0.0279 (+1.56) and e2 < 0.0508 (+1.23), with e2 < 0.0502 and z_dr_0p1_0p2 > 0.159 (50%) and centroid_offset > 0.00781 and lam2 < 0.00351 (62%) deciding who turns on (35.3%, mean 0.553). It slightly lowers the W and Z scores; the formula calls them W (50%) and is right 70.2% of the time.
- **Compact mid-mass W jets** — 12.6% of jets, neuron 0.45, formula right for 65%. Mostly W (50%) with 27% Z, 12.6% of jets: mass 48.4 GeV, width 0.0048, pT split between 0.025-0.05 and 0.05-0.1. girth2 < 0.00862 (-6.39), width < 0.0133 (-4.58) and mass_over_sum_pt > 0.00941 (-3.02) outweigh lam1 < 0.00802 (+3.10), e2 < 0.0508 (+1.96) and max_dr > 0.0279 (+1.77); centroid_offset > 0.00781 and lam2 < 0.00351 (79%, +0.60) turns some on (26.6%, mean 0.452). The small effect slightly lowers the W and Z scores; the formula calls them W (66%) and is right 65.1% of the time.
- **Wider Z and top jets** — 10.5% of jets, neuron 2.48, formula right for 72%. Mostly Z (51%) with 28% tops and 12% gluons, 10.5% of jets: mass 60.2 GeV, width 0.0088, 61% of pT at 0.05-0.1 and 24% beyond 0.1. These jets are broad enough that girth2 < 0.00862 passes for only 61% (-0.78) and lam1 < 0.00802 for 49%; mass_over_sum_pt > 0.00941 (-4.30) and width < 0.0133 (-2.45) are balanced by max_dr > 0.0279 (+1.93), e2 < 0.0508 (+0.73), centroid_offset > 0.00781 and lam2 < 0.00351 (+0.63) and girth > 0.0868 (44%). The neuron is on for 79.5% at 2.485, which lowers the W and Z scores and raises g and q, working against the Z jets here; the formula still calls them Z (61%) and is right 72.1% of the time.
- **Light, slightly spread gluon-rich jets** — 9.7% of jets, neuron 1.00, formula right for 49%. Mostly gluons (37%) with quarks 26%, W 16%, Z 14%, 9.7% of jets: mass 19.6 GeV, width 0.0014, 53% of pT at 0.025-0.05. girth2 < 0.00862 (-12.07) and width < 0.0133 (-6.41) are offset by lam1 < 0.00802 (+6.16), e2 < 0.0508 (+3.30), e2_sq < 0.00328 (+1.49), mass < 46.6 and z_dr_0p05_0p1 < 0.8 (99%, +1.09) and centroid_offset > 0.00781 and lam2 < 0.00351 (79%, +0.82), leaving it on for 46.7% (mean 0.998). It mildly raises the g and q scores and lowers W and Z; the formula calls them g (52%) and is right only 49.1% of the time.
- **Narrow mid-mass W-leaning mixture** — 9.2% of jets, neuron 0.54, formula right for 53%. Mostly W (38%) with Z 22%, gluons 18%, quarks 13%, 9.2% of jets: mass 37.5 GeV, width 0.0031, 57% of pT at 0.025-0.05. girth2 < 0.00862 (-9.22), width < 0.0133 (-5.49) and mass_over_sum_pt > 0.00941 (-2.19) against lam1 < 0.00802 (+4.63), e2 < 0.0508 (+2.73), max_dr > 0.0279 (+1.89) and centroid_offset > 0.00781 and lam2 < 0.00351 (87%, +0.78); on for 31.2% (mean 0.537). The formula calls them W (58%) and is right only 53.1% of the time.
- **Wide top jets** — 8.1% of jets, neuron 6.29, formula right for 78%. Mostly tops (77%) with 16% gluons, 8.1% of jets: mass 71.9 GeV, width 0.0170, 52% of pT beyond 0.1. The narrow-jet switches all fail, so the neuron is built from girth > 0.0868 (97%, +3.59), max_dr > 0.0279 (+2.96), log_sum_pt < 6.57 and pt_7 < 45.4 (77%, +1.02) and centroid_offset > 0.00781 and lam2 < 0.00351 (78%, +1.01) against mass_over_sum_pt > 0.00941 (-6.09), mass_over_sum_pt > 0.0171 and tau32 < 0.53 (59%, -1.20) and C2 > 0.0101 (-0.98). It is high (6.295, on for 98%), strongly lowering the W and Z scores and raising g and q; the formula calls them t (95%) and is right 78.5% of the time.
- **Heavy, very wide top jets** — 4.9% of jets, neuron 8.42, formula right for 80%. Mostly tops (77%) with 17% gluons, 4.9% of jets: mass 86.3 GeV, width 0.0281, 85% of pT beyond 0.1. girth > 0.0868 (+8.89) and max_dr > 0.0279 (+3.11) outweigh mass_over_sum_pt > 0.00941 (-8.05), mass_over_sum_pt > 0.0171 and tau32 < 0.53 (72%, -2.04), centroid_offset > 0.0212 (74%, -1.14) and C2 > 0.0101 (-1.07). The neuron is on for 99.8% at 8.421, strongly lowering W and Z and raising g and q; the formula calls them t (92%) and is right 80% of the time, with gluons the main misses.
- **Clean, fat three-prong tops** — 2.5% of jets, neuron 1.67, formula right for 95%. Mostly tops (95%), 2.5% of jets: mass 86.5 GeV, width 0.0284, 87% of pT beyond 0.1. Like group 7, girth > 0.0868 (+9.01) meets mass_over_sum_pt > 0.00941 (-8.16), but here lam2 > 0.0029 always passes (-6.10) and mass_over_sum_pt > 0.0171 and tau32 < 0.53 (98%, -3.75) and C2 > 0.0101 (-1.99) cut deeper, leaving 1.667 (on for 60.6%). The formula calls them t (99%) and is right 94.8% of the time, via other neurons.
- **Light but widely spread top/gluon jets** — 0.4% of jets, neuron 8.76, formula right for 63%. Mostly tops (65%) with 26% gluons, 0.4% of jets: mass only 45.4 GeV for a width of 0.0153, with 85% of the pT beyond 0.1. This small group is set apart by e2 < 0.0502 and z_dr_0p1_0p2 > 0.159 (+8.43), plus centroid_offset > 0.00781 and lam2 < 0.00351 (94%, +4.06) and girth > 0.0868 (+4.06), against mass_over_sum_pt > 0.00941 (-3.98) and centroid_offset > 0.0212 (-3.79). The neuron reaches its highest value (8.76), strongly lowering W and Z; the formula calls them t (90%) and is right only 63.1% of the time, many gluons being called tops.

### neuron 7: two-prong shape in W/Z window (major)

- **What it measures:** Pushed up for mass/pT above 0.0729 but down again above 0.0906 (a window where boosted W/Z jets sit), and down both for very narrow jets (width < 0.00564) and for girth2 > 0.00445; it rises with eccentricity and falls with planar flow and τ21 (rank correlations 0.562, -0.562, -0.525). Z jets sit highest (3.45), then W (2.28), with top (1.10), gluon (0.84) and quark (0.44) jets low.
- *computed — its value:* largest for Z (3.45), then W (2.28), then t (1.10), then g (0.84), then q (0.44); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** High values mark a boson, so it raises the W score and the Z score, and it is the largest input of the Z score (+27%). It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+9%), Z (+27%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Highest (≈4.34) for jets of intermediate spread (0.00302 < girth2 ≤ 0.0118) with e2_sq > 0.0059 (19% of jets, mostly Z, 50%) and ≈2.64 for the same spread with e2_sq ≤ 0.0059 (24%, mostly W, 46%). It is off above girth2 0.0139 (≈0.007; 14% of jets, 81% tops) and nearly off (≈0.09) for the 33% of very thin, centred, light jets (girth2 ≤ 0.00302, centroid_offset ≤ 0.0218, mass ≤ 35.8 GeV; mostly quarks, 46%). It raises only the Z (+2.04 and +1.24 in its two main regimes) and W (+0.95 and +0.577) scores, and the regimes describe it well (regime_r2 0.822).

Regimes (a small tree on its quantities; R² 0.822):

- `` — 18.6% of jets, value 4.34 (3.00…5.88), formula right 72%
- `` — 23.8% of jets, value 2.64 (1.38…3.88), formula right 64%
- `` — 2.0% of jets, value 2.62 (1.50…3.88), formula right 44%
- `` — 4.7% of jets, value 1.47 (0.62…2.25), formula right 44%
- `` — 2.0% of jets, value 1.11 (0.00…2.12), formula right 53%
- `` — 2.0% of jets, value 0.96 (0.00…2.00), formula right 71%
- `` — 32.9% of jets, value 0.09 (0.00…0.38), formula right 60%
- `` — 14.0% of jets, value 0.01 (0.00…0.00), formula right 83%

```
z = 4.74
if girth2 > 0.0044: z += -765 × (girth2 − 0.0044)
if mass_over_sum_pt > 0.073: z += 183 × (mass_over_sum_pt − 0.073)
if width < 0.0056: z += -1070 × (0.0056 − width)
if girth < 0.089: z += -40.90 × (0.089 − girth)
if mass_over_sum_pt > 0.091: z += -185 × (mass_over_sum_pt − 0.091)
if e2 < 0.037: z += 81.60 × (0.037 − e2)
if girth2 > 0.0044 and eccentricity > 0.945: z += 10600 × (girth2 − 0.0044) × (eccentricity − 0.945)
if planar_flow < 0.205 and width > 0.0077: z += -3780 × (0.205 − planar_flow) × (width − 0.0077)
if planar_flow < 0.184: z += -7.43 × (0.184 − planar_flow)
if e2 < 0.025: z += 56.50 × (0.025 − e2)
if e2_sq < 0.0011: z += 1320 × (0.0011 − e2_sq)
if planar_flow < 0.179 and sum_pt > 607: z += 0.041 × (0.179 − planar_flow) × (sum_pt − 607)
if pt_7 < 44.50 and planar_flow < 0.596: z += -0.095 × (44.50 − pt_7) × (0.596 − planar_flow)
if e2 < 0.051 and D2 < 1.10: z += 120 × (0.051 − e2) × (1.10 − D2)
if girth2 < 0.00071: z += -2290 × (0.00071 − girth2)
if centroid_offset < 0.020: z += -37.90 × (0.020 − centroid_offset)
if mass > 80.40: z += -0.206 × (mass − 80.40)
if lam1 < 0.008 and D2 < 1.06: z += -754 × (0.008 − lam1) × (1.06 − D2)
if mass > 80.40 and eccentricity > 0.928: z += -2.23 × (mass − 80.40) × (eccentricity − 0.928)
if centroid_offset < 0.023 and C2 > 0.025: z += 1100 × (0.023 − centroid_offset) × (C2 − 0.025)
if centroid_offset > 0.031 and pt_0 > 375: z += -8.88 × (centroid_offset − 0.031) × (pt_0 − 375)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 32.2% of jets, neuron 0.28, formula right for 60%. Mostly quarks (44%) with 34% gluons, 32.3% of jets: mass 9.0 GeV, width 0.0003, 84% of pT inside 0.025. width < 0.00564 (-5.66), girth < 0.089 (-3.06) and girth2 < 0.000708 (83%, -0.94) outweigh e2 < 0.0372 (+2.58), e2_sq < 0.00115 (+1.27) and e2 < 0.0253 (+1.12); the mass-window tests never pass, so the neuron is on for only 22.1% (mean 0.279) and hardly touches the W and Z scores. The formula calls them q (50%) and is right 59.9% of the time.
- **Two-prong W jets** — 22.5% of jets, neuron 3.17, formula right for 68%. Mostly W (49%) with 30% Z and 10% tops, 22.5% of jets: mass 51.9 GeV, width 0.0056, 54% of pT at 0.05-0.1. Many small terms: e2 < 0.0513 and D2 < 1.1 (77%, +0.80), planar_flow < 0.179 and sum_pt > 607 (70%, +0.73), girth2 > 0.0044 and eccentricity > 0.945 (79%, +0.47) and e2 < 0.0372 (71%, +0.43) against girth2 > 0.00445 (93%, -0.91), girth < 0.089 (-0.90), planar_flow < 0.184 (83%, -0.88) and lam1 < 0.00797 and D2 < 1.06 (75%, -0.68); mass_over_sum_pt > 0.0729 passes for half. The neuron is on for 99.7% at 3.174, raising the W and Z scores; the formula calls them W (62%) and is right 67.9% of the time.
- **Light narrow two-prong mixture** — 16.7% of jets, neuron 1.43, formula right for 52%. A mixture of W (31%), gluons (24%), Z (21%) and quarks (16%), 16.7% of jets: mass 33.7 GeV, width 0.0028, 57% of pT at 0.025-0.05. width < 0.00564 (-3.04) and girth < 0.089 (-1.90) against e2 < 0.0372 (+1.59), e2 < 0.0253 (88%, +0.45) and planar_flow < 0.179 and sum_pt > 607 (56%, +0.58); the mass window is never reached. On for 79.2% at 1.428, it raises the W and Z scores moderately; the formula calls them W (45%) and is right only 52.5% of the time, many gluons being pushed towards bosons.
- **Z jets in the mass window** — 12.0% of jets, neuron 4.58, formula right for 73%. Mostly Z (58%) with 23% tops, 12.0% of jets: mass 59.3 GeV, width 0.0083, 61% of pT at 0.05-0.1. mass_over_sum_pt > 0.0729 passes (99%, +2.73) while mass_over_sum_pt > 0.0906 mostly does not (26%), so the window bonus survives; with girth2 > 0.0044 and eccentricity > 0.945 (83%, +1.41) it beats girth2 > 0.00445 (-2.97) and planar_flow < 0.184 (82%, -0.89). The neuron peaks here (4.577), raising the Z score by about 2.1 and W by 1.0; the formula calls them Z (70%) and is right 72.6% of the time, with tops the main misses.
- **Moderate-mass wide top jets** — 4.4% of jets, neuron 0.80, formula right for 71%. Mostly tops (69%) with 20% gluons, 4.4% of jets: mass 65.4 GeV, width 0.0137, 46% of pT beyond 0.1. The window bonus mass_over_sum_pt > 0.0729 (+6.80) is cancelled by girth2 > 0.00445 (-7.06) and mass_over_sum_pt > 0.0906 (99%, -3.62); girth2 > 0.0044 and eccentricity > 0.945 (72%, +2.84) against planar_flow < 0.205 and width > 0.00768 (72%, -2.38) decides the rest. It is on for 52.9% (mean 0.795), a small push towards W and Z; the formula calls them t (91%) and is right 71.4% of the time.
- **Heavy, wide top jets** — 3.5% of jets, neuron 0.00, formula right for 94%. Mostly tops (94%), 3.5% of jets: mass 85.2 GeV, width 0.0264, 79% of pT beyond 0.1. Huge terms cancel: mass_over_sum_pt > 0.0729 (+15.52) against girth2 > 0.00445 (-16.79) and mass_over_sum_pt > 0.0906 (-12.41), plus mass > 80.4 (61%, -1.99), so the neuron is off for every jet and adds nothing to W or Z. The formula calls them t (99%) and is right 93.7% of the time.
- **Mid-heavy wide top jets** — 3.3% of jets, neuron 0.00, formula right for 87%. Mostly tops (86%), 3.3% of jets: mass 73.4 GeV, width 0.0192, 59% of pT beyond 0.1. mass_over_sum_pt > 0.0729 (+10.72) is beaten by girth2 > 0.00445 (-11.27) and mass_over_sum_pt > 0.0906 (-7.57), so the neuron is off (on for 0.6%) and gives no boson push. The formula calls them t (99%) and is right 86.9% of the time.
- **Heavy, elongated flat top/gluon jets** — 2.8% of jets, neuron 0.00, formula right for 73%. Mostly tops (72%) with 19% gluons, 2.8% of jets: mass 82.1 GeV, width 0.0207, 71% of pT beyond 0.1. Both girth2 > 0.0044 and eccentricity > 0.945 (+7.53) and planar_flow < 0.205 and width > 0.00768 (-7.89) always pass and cancel, on top of the big mass_over_sum_pt / girth2 cancellation; mass > 80.4 (52%, -1.62) and mass > 80.4 and eccentricity > 0.928 (52%, -1.11) keep the neuron off for every jet. The formula calls them t (94%) and is right 73.4% of the time, with gluons the main misses.
- **Widest, heaviest top jets** — 1.3% of jets, neuron 0.00, formula right for 85%. Mostly tops (85%), 1.3% of jets: mass 90.3 GeV, width 0.0357, soft (sum pT 494 GeV), 88% of pT beyond 0.1. The largest terms of the neuron meet: girth2 > 0.00445 (-23.91) and mass_over_sum_pt > 0.0906 (-17.14) against mass_over_sum_pt > 0.0729 (+20.19), with mass > 80.4 (69%, -2.69), so the neuron is off and adds nothing. The formula calls them t (97%) and is right 84.7% of the time.
- **Heavy elongated top/gluon jets** — 1.2% of jets, neuron 0.00, formula right for 70%. Mostly tops (60%) with 28% gluons and 12% quarks, 1.2% of jets: mass 96.1 GeV, width 0.0321, 85% of pT beyond 0.1. girth2 > 0.0044 and eccentricity > 0.945 (+13.04) is outweighed by planar_flow < 0.205 and width > 0.00768 (-15.13), and girth2 > 0.00445 (-21.18) plus mass_over_sum_pt > 0.0906 (-15.40) beat mass_over_sum_pt > 0.0729 (+18.48); mass > 80.4 (73%, -3.79) completes the veto, so the neuron is off. The formula calls them t (74%) and is right 70% of the time, often mistaking the gluons.

### neuron 9: narrow, light single-core jet (major)

- **What it measures:** Pushed up mainly for narrow jets (width < 0.00662, its strongest term), a small largest particle distance (max ΔR < 0.257) and light, centred jets (mass < 51.1 GeV with small centroid offset); it falls with width and mass (rank correlations -0.652 and -0.629). Quarks sit highest (9.07), then gluons (6.63), far above top (2.40), W (2.35) and Z (1.42) jets.
- *computed — its value:* largest for q (9.07), then g (6.63), then t (2.40), then W (2.35), then Z (1.42); it separates q jets from the rest best (AUC 0.79: large for q)
- **How the class scores use it:** High values mean a light-parton (QCD) jet, so it raises the q score (its largest input, +49%) and the g score (+26%), and lowers the W and Z scores slightly. It does not enter the t score.
- *computed — used by:* raises the score of g (+26%), q (+49%); lowers the score of W (-3%), Z (-5%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** Highest (≈11.7, typically 8.5 to 14.5) for very thin jets (girth2 ≤ 0.00125) with the pT centroid on the axis (centroid_offset ≤ 0.00919): 21% of jets, mostly quarks (56%); ≈9.09 when slightly off-axis (0.00919 < centroid_offset ≤ 0.0149; 4.7%, mostly gluons, 47%). A second high branch (≈8.17) is very wide jets with lam2 > 0.00514 (3.2%, 94% tops); for girth2 > 0.00125 and lam2 ≤ 0.00514 it is ≈3.9 if mass ≤ 38.7 GeV (15%) and low (≈0.75) above 38.7 GeV (50% of jets). It raises q (+2.96 in the top regime) and g (+2.0) and lowers W and Z (−0.364); regime_r2 0.754.

Regimes (a small tree on its quantities; R² 0.754):

- `` — 21.0% of jets, value 11.65 (8.50…14.50), formula right 66%
- `` — 4.7% of jets, value 9.09 (6.00…12.50), formula right 56%
- `` — 3.2% of jets, value 8.17 (4.00…13.00), formula right 94%
- `` — 2.0% of jets, value 6.46 (3.00…9.75), formula right 53%
- `` — 15.0% of jets, value 3.90 (0.00…9.00), formula right 47%
- `` — 4.2% of jets, value 2.63 (0.00…5.75), formula right 41%
- `` — 49.8% of jets, value 0.75 (0.00…2.50), formula right 72%

```
z = -2.22
if width < 0.0066: z += 1550 × (0.0066 − width)
if max_dr < 0.257: z += 14.70 × (0.257 − max_dr)
if mass < 51.10 and centroid_offset < 0.024: z += 5.63 × (51.10 − mass) × (0.024 − centroid_offset)
if width < 0.0068 and centroid_offset > 0.0029: z += -43400 × (0.0068 − width) × (centroid_offset − 0.0029)
if mass < 31.30: z += -0.136 × (31.30 − mass)
if centroid_offset < 0.017: z += 170 × (0.017 − centroid_offset)
if log_sum_pt > 6.36: z += -4.60 × (log_sum_pt − 6.36)
if mass < 49.90 and log_sum_pt < 6.79: z += 0.208 × (49.90 − mass) × (6.79 − log_sum_pt)
if centroid_offset < 0.019 and z_4 > 0.031: z += -1420 × (0.019 − centroid_offset) × (z_4 − 0.031)
if lam2 > 0.0014: z += 1500 × (lam2 − 0.0014)
if n_dr_0p2_0p4 > 1.00: z += 1.23 × (n_dr_0p2_0p4 − 1.00)
if girth2 > 0.018 and pt_7 < 29.80: z += 59.40 × (girth2 − 0.018) × (29.80 − pt_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Wide, heavy top-rich jets** — 20.5% of jets, neuron 0.51, formula right for 69%. Mostly tops (51%) with Z 20% and gluons 17%, 20.5% of jets: mass 63.3 GeV, width 0.0141, 41% of pT beyond 0.1. The narrow-core bonus width < 0.00662 passes for only 16%, so the value comes from max_dr < 0.257 (78%, +1.03) and n_dr_0p2_0p4 > 1 (24%, +0.50) against log_sum_pt > 6.36 (53%, -0.42). The neuron is small (0.506, on for 30.8%) with little effect on the scores; the formula calls them t (61%) and is right 68.7% of the time.
- **Heavy two-prong Z/W jets** — 16.7% of jets, neuron 0.54, formula right for 78%. Mostly Z (44%) with 39% W and 12% tops, 16.7% of jets: mass 62.9 GeV, width 0.0074, 58% of pT at 0.05-0.1. centroid_offset < 0.0174 (always, +1.84) and max_dr < 0.257 (97%, +1.82) plus width < 0.00662 for half (+0.58) are taken back by log_sum_pt > 6.36 (90%, -1.19) and centroid_offset < 0.019 and z_4 > 0.031 (98%, -0.96). The neuron is small (0.543, on for 41.8%); the formula splits them Z 46% / W 45% and is right 77.5% of the time.
- **Pencil-thin quark jets** — 16.5% of jets, neuron 12.26, formula right for 67%. Mostly quarks (62%) with 26% gluons, 16.5% of jets: mass 8.5 GeV, width 0.0001, harder than average (sum pT 903 GeV), 97% of pT inside 0.025. Every light-core bonus passes: width < 0.00662 (+10.04), mass < 51.1 and centroid_offset < 0.0241 (+4.99), max_dr < 0.257 (+3.26) and centroid_offset < 0.0174 (+2.40), with only mass < 31.3 (-3.10) and log_sum_pt > 6.36 (-2.01) against. The neuron peaks here (12.262), raising the quark score by about 3.1 and gluon by 2.1 and lowering W and Z; the formula calls them q (81%) and is right 67% of the time.
- **Mid-mass W jets** — 12.8% of jets, neuron 1.58, formula right for 62%. Mostly W (47%) with 28% Z, 12.8% of jets: mass 46.6 GeV, width 0.0046, pT split between 0.025-0.05 and 0.05-0.1. width < 0.00662 always passes (+3.19) with max_dr < 0.257 (94%, +1.57), against width < 0.00676 and centroid_offset > 0.00285 (98%, -1.34) and log_sum_pt > 6.36 (83%, -1.05). The neuron is on for 67.1% at 1.581, slightly raising g and q; the formula calls them W (65%) and is right 62.1% of the time.
- **Narrow mid-mass mixture** — 8.9% of jets, neuron 3.87, formula right for 51%. A mixture of W (34%), gluons (20%), quarks (20%) and Z (19%), 8.9% of jets: mass 35.5 GeV, width 0.0025, 54% of pT at 0.025-0.05. width < 0.00662 (+6.38) and max_dr < 0.257 (+1.62) beat width < 0.00676 and centroid_offset > 0.00285 (97%, -2.37) and log_sum_pt > 6.36 (90%, -1.33), with mass < 51.1 and centroid_offset < 0.0241 (78%) adding a little. At 3.867 (on for 86.1%) it clearly raises the g and q scores; the formula calls them W (52%) and is right only 50.7% of the time.
- **Soft narrow gluon jets** — 6.9% of jets, neuron 10.25, formula right for 61%. Mostly gluons (60%) with 29% quarks, 6.9% of jets: mass 11.1 GeV, width 0.0006, soft (sum pT 595 GeV, leading particle 155 GeV) with pT evenly shared, 78% inside 0.025. width < 0.00662 (+9.39), mass < 51.1 and centroid_offset < 0.0241 (+3.80), mass < 49.9 and log_sum_pt < 6.79 (+3.34) and max_dr < 0.257 (+3.17) all pass, against mass < 31.3 (-2.75). The neuron is high (10.25), raising the gluon and quark scores and lowering W and Z; the formula calls them g (84%) and is right 60.8% of the time, with many quarks misread as gluons.
- **Light off-centre gluon-led mixture** — 6.3% of jets, neuron 2.69, formula right for 45%. A mixture led by gluons (38%) with Z 21%, W 18%, quarks 13%, 6.3% of jets: mass 10.3 GeV, width 0.0016, 63% of pT at 0.025-0.05. width < 0.00662 (+7.82), mass < 49.9 and log_sum_pt < 6.79 (+3.16) and max_dr < 0.257 (+2.94) are cut back by width < 0.00676 and centroid_offset > 0.00285 (always, -6.23) and mass < 31.3 (-2.86), and centroid_offset < 0.0174 fails here. The neuron is on for 71.6% at 2.695, raising g and q; the formula calls them g (62%) but is right only 45.2% of the time, as many light W/Z jets are pulled towards gluons.
- **Light narrow gluon/quark/boson mix** — 6.2% of jets, neuron 5.83, formula right for 48%. A mixture led by gluons (36%) with quarks 28%, W 20%, Z 14%, 6.2% of jets: mass 10.3 GeV, width 0.0005, 79% of pT inside 0.025. width < 0.00662 (+9.46), max_dr < 0.257 (+3.14) and mass < 51.1 and centroid_offset < 0.0241 (97%, +1.88) outweigh width < 0.00676 and centroid_offset > 0.00285 (-3.50), mass < 31.3 (-2.86) and log_sum_pt > 6.36 (98%, -1.41). At 5.834 (on for 96%) it raises the g and q scores; the formula calls them g (48%) and is right only 48.4% of the time.
- **Wide three-prong tops** — 3.6% of jets, neuron 5.06, formula right for 91%. Mostly tops (91%), 3.6% of jets: mass 79.4 GeV, width 0.0237, 68% of pT beyond 0.1. None of the narrow-core bonuses fire; instead lam2 > 0.00136 always passes (+5.58), with n_dr_0p2_0p4 > 1 (55%, +1.21), so the neuron is on (99.5%, mean 5.057) and, oddly, raises the gluon and quark scores for these tops. The formula still calls them t (99%) and is right 90.8% of the time.
- **Widest top jets** — 1.5% of jets, neuron 9.89, formula right for 95%. Mostly tops (95%), 1.5% of jets: mass 85.7 GeV, width 0.0292, 88% of pT beyond 0.1. lam2 > 0.00136 (+12.66) alone makes the value large (9.889, on for 98.8%), helped by n_dr_0p2_0p4 > 1 (62%, +1.42); the neuron thus raises g and q scores for the fattest tops, which other neurons must overcome. The formula calls them t (99%) and is right 94.7% of the time.

### neuron 10: overall jet size (e2, width, mass) (major)

- **What it measures:** Pushed down for small e2 (< 0.0467, its strongest term) and up for mass > 9.7 GeV and for mass/pT < 0.0966; it follows e2, width and mass very closely (rank correlations 0.798, 0.794, 0.789). Tops sit highest (5.97), W (3.26) and Z (3.19) in the middle, gluons (2.01) and quarks (1.65) lowest.
- *computed — its value:* largest for t (5.97), then W (3.26), then Z (3.19), then g (2.01), then q (1.65); it separates t jets from the rest best (AUC 0.79: large for t)
- **How the class scores use it:** Large size marks a top, so it raises the t score (+26%); small size marks a quark, so the q score reads a large value as evidence against a quark and it lowers the q score (-18%). It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+26%); lowers the score of q (-18%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** Follows jet size: highest (≈13.7) for lam2 > 0.00618 (2.3% of jets, 95% tops), ≈9.24 for 0.00277 < lam2 ≤ 0.00618 (3.4%, 89% tops) and ≈7.18 for lam2 ≤ 0.00277 with lam1 > 0.00118 and C2 > 0.0831 (76% tops). The bulk (lam2 ≤ 0.00277, lam1 > 0.00118, C2 ≤ 0.0831; 59% of jets) sits at ≈3.43, and thin jets (lam1 ≤ 0.00118) are low: ≈1.96 for eccentricity > 0.936 and ≈0.906 for eccentricity ≤ 0.936 (25%, mostly quarks, 47%). It raises the t score (+1.29 in the bulk, up to +5.12) and lowers q (−0.429 in the bulk, up to −1.71); the regimes describe it well (regime_r2 0.829).

Regimes (a small tree on its quantities; R² 0.829):

- `` — 2.3% of jets, value 13.66 (10.46…16.94), formula right 95%
- `` — 3.4% of jets, value 9.24 (6.06…12.12), formula right 89%
- `` — 2.6% of jets, value 7.18 (3.62…10.68), formula right 80%
- `` — 59.4% of jets, value 3.43 (2.31…4.25), formula right 65%
- `` — 7.5% of jets, value 1.96 (1.19…2.75), formula right 52%
- `` — 24.8% of jets, value 0.91 (0.25…1.62), formula right 62%

```
z = 1.73
if e2 < 0.047: z += -70.60 × (0.047 − e2)
if mass_over_sum_pt < 0.097: z += 31.90 × (0.097 − mass_over_sum_pt)
if mass > 9.70: z += 0.024 × (mass − 9.70)
if girth2 < 0.0018: z += -1260 × (0.0018 − girth2)
if eccentricity > 0.901 and z_dr_0p2_0p4 < 0.054: z += 259 × (eccentricity − 0.901) × (0.054 − z_dr_0p2_0p4)
if n_dr_0p05_0p1 < 2.95: z += 0.324 × (2.95 − n_dr_0p05_0p1)
if lam1 < 0.004 and log_sum_pt > 6.69: z += 3650 × (0.004 − lam1) × (log_sum_pt − 6.69)
if LHA > 0.301 and tau21 < 0.699: z += -42.40 × (LHA − 0.301) × (0.699 − tau21)
if log_sum_pt > 6.69: z += -6.91 × (log_sum_pt − 6.69)
if e2 > 0.046: z += 61.30 × (e2 − 0.046)
if lam2 > 0.00023: z += 514 × (lam2 − 0.00023)
if lam2 > 8.4e-05 and tau21 < 0.518: z += 2350 × (lam2 − 8.4e-05) × (0.518 − tau21)
if tau32 < 0.271: z += 12.10 × (0.271 − tau32)
if lam1 < 0.004 and sum_pt > 989: z += -9.18 × (0.004 − lam1) × (sum_pt − 989)
if C2 > 0.060: z += 27.90 × (C2 − 0.060)
if log_sum_pt < 6.24: z += -3.17 × (6.24 − log_sum_pt)
if n_dr_0p2_0p4 > 1.56: z += 0.519 × (n_dr_0p2_0p4 − 1.56)
if tau32 < 0.290 and n_dr_0p2_0p4 < 2.00: z += -4.11 × (0.290 − tau32) × (2.00 − n_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Two-prong Z/W jets** — 23.7% of jets, neuron 3.62, formula right for 72%. Mostly Z (39%) with W 29% and tops 20%, 23.7% of jets: mass 58.0 GeV, width 0.0078, 68% of pT at 0.05-0.1. mass > 9.7 (+1.14) and eccentricity > 0.901 and z_dr_0p2_0p4 < 0.0539 (86%, +1.02) plus mass_over_sum_pt < 0.0966 (86%, +0.41) outweigh e2 < 0.0467 (84%, -0.42) and LHA > 0.301 and tau21 < 0.699 (69%, -0.38). The neuron is on (99.9%) at 3.621, lowering the quark score and raising the top score; the formula calls them Z (42%) and is right 72.5% of the time.
- **Mid-mass W/Z jets** — 19.3% of jets, neuron 3.26, formula right for 61%. Mostly W (38%) with Z 30%, gluons 13%, tops 12%, 19.3% of jets: mass 48.2 GeV, width 0.0049, 51% of pT at 0.025-0.05. e2 < 0.0467 always passes (-1.43) but is outweighed by mass_over_sum_pt < 0.0966 (+0.98), mass > 9.7 (+0.91), eccentricity > 0.901 and z_dr_0p2_0p4 < 0.0539 (73%, +0.74) and n_dr_0p05_0p1 < 2.95 (62%, +0.47). At 3.261 it lowers q and raises t; the formula calls them W (51%) and is right 61% of the time.
- **Narrow light gluon/quark jets** — 19.2% of jets, neuron 0.98, formula right for 55%. Mostly gluons (42%) with 37% quarks, 19.2% of jets: mass 8.2 GeV, width 0.0004, 83% of pT inside 0.025. e2 < 0.0467 (-2.88) and girth2 < 0.00182 (-1.84) are balanced by mass_over_sum_pt < 0.0966 (+2.71) and n_dr_0p05_0p1 < 2.95 (+0.91), and mass > 9.7 passes for only 30%, so the neuron sits low (0.982, on for 99.2%). The formula calls them g (52%) and is right only 55.2% of the time.
- **Light-to-mid mass mixture** — 12.4% of jets, neuron 2.71, formula right for 50%. A mixture of gluons (30%), W (22%), quarks (20%) and Z (19%), 12.4% of jets: mass 25.4 GeV, width 0.0022, 55% of pT at 0.025-0.05. e2 < 0.0467 (-2.32) against mass_over_sum_pt < 0.0966 (+1.94), with eccentricity > 0.901 and z_dr_0p2_0p4 < 0.0539 (75%, +0.65), n_dr_0p05_0p1 < 2.95 (80%, +0.59) and mass > 9.7 (89%, +0.38). At 2.708 it lowers q and raises t moderately; the formula calls them g (41%) and is right only 50.4% of the time.
- **Hard narrow quark jets** — 9.6% of jets, neuron 1.49, formula right for 70%. Mostly quarks (63%) with 18% gluons, 9.6% of jets: mass 9.4 GeV, width 0.0002, hard (sum pT 951 GeV, leading particle 407 GeV), 96% of pT inside 0.025. e2 < 0.0467 (-3.01), girth2 < 0.00182 (-2.05) and log_sum_pt > 6.69 (-1.15) are offset by mass_over_sum_pt < 0.0966 (+2.77), lam1 < 0.004 and log_sum_pt > 6.69 (+2.31) and n_dr_0p05_0p1 < 2.95 (+0.88). The value is low (1.494), only slightly lowering the quark score; the formula calls them q (83%) and is right 70% of the time.
- **Wide, heavy top/gluon jets** — 6.5% of jets, neuron 4.34, formula right for 75%. Mostly tops (72%) with 20% gluons, 6.5% of jets: mass 80.5 GeV, width 0.0227, 81% of pT beyond 0.1. Here e2 < 0.0467 fails and the size terms mass > 9.7 (+1.67), e2 > 0.0458 (96%, +1.50), n_dr_0p05_0p1 < 2.95 (76%) and lam2 > 8.42e-05 and tau21 < 0.518 (79%) add up, but LHA > 0.301 and tau21 < 0.699 always passes (-2.75), holding the value at 4.343. It raises the top score and lowers q; the formula calls them t (92%) and is right 74.7% of the time, the gluons being the main misses.
- **Clean three-prong top jets** — 4.3% of jets, neuron 8.07, formula right for 86%. Mostly tops (83%), 4.3% of jets: mass 75.4 GeV, width 0.0192, pT spread from 0.05 out beyond 0.1. Several size and substructure bonuses pass together: mass > 9.7 (+1.55), tau32 < 0.271 (83%, +1.38), e2 > 0.0458 (88%, +1.23), lam2 > 0.000227 (92%, +1.16) and C2 > 0.0596 (94%, +1.08), with only LHA > 0.301 and tau21 < 0.699 (79%, -0.54) against. The value is high (8.069), raising the top score by about 3.0; the formula calls them t (93%) and is right 85.9% of the time.
- **Large, clean three-prong tops** — 3.5% of jets, neuron 12.69, formula right for 94%. Mostly tops (94%), 3.5% of jets: mass 84.3 GeV, width 0.0277, 84% of pT beyond 0.1. lam2 > 0.000227 (+3.72), e2 > 0.0458 (+2.46), mass > 9.7 (+1.76), lam2 > 8.42e-05 and tau21 < 0.518 (70%, +1.44), tau32 < 0.271 (79%, +1.37) and C2 > 0.0596 (+1.12) all add, against LHA > 0.301 and tau21 < 0.699 (95%, -1.44). The neuron peaks here (12.69), raising the top score by about 4.8 and lowering q; the formula calls them t (99%) and is right 94.2% of the time.
- **Very hard narrow quark/gluon jets** — 1.3% of jets, neuron 0.05, formula right for 68%. An even quark (44%) / gluon (43%) mixture, 1.3% of jets: mass 13.5 GeV, width 0.0003, very hard (sum pT 1137 GeV, leading particle 505 GeV), 92% of pT inside 0.025. lam1 < 0.00395 and sum_pt > 989 always passes (-4.85), with e2 < 0.0467 (-2.97), log_sum_pt > 6.69 (-2.38) and girth2 < 0.00182 (97%, -1.95), outweighing lam1 < 0.004 and log_sum_pt > 6.69 (+4.64) and mass_over_sum_pt < 0.0966 (+2.71); the neuron is on for only 9% and adds almost nothing. The formula splits them g 51% / q 46% and is right 67.9% of the time.
- **Extremely hard narrow gluon jets** — 0.2% of jets, neuron 0.00, formula right for 71%. Mostly gluons (60%) with 28% quarks, 0.2% of jets: mass 14.1 GeV, width 0.0002, extremely hard (sum pT 1417 GeV, leading particle 673 GeV). lam1 < 0.00395 and sum_pt > 989 subtracts 14.66, more than lam1 < 0.004 and log_sum_pt > 6.69 adds (+7.77), and log_sum_pt > 6.69 (-3.87) and e2 < 0.0467 (-3.00) complete it, so the neuron is off and adds nothing. The formula calls them g (67%) and is right 70.9% of the time.

### neuron 13: compactness (girth below top size) (major)

- **What it measures:** Driven mostly by girth < 0.15 (its dominant term) and width < 0.0159, both pushing it up, i.e. by the jet being narrower than a typical top; very small e2 (< 0.0509) pulls it down a little. It falls with width, girth and LHA (rank correlations about -0.81 to -0.83); quarks (7.15), W (6.17), gluons (5.88) and Z (5.66) all sit high, tops far lower (2.01).
- *computed — its value:* largest for q (7.15), then W (6.17), then g (5.88), then Z (5.66), then t (2.01); it separates t jets from the rest best (AUC 0.09: small for t)
- **How the class scores use it:** Being narrower than a top is the main evidence against a top, so it lowers the t score, where it is the largest input (-46%); it also raises the W and Z scores. It does not enter the g or q scores.
- *computed — used by:* raises the score of W (+9%), Z (+10%); lowers the score of t (-46%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** Set by width: ≈8.43 for the thinnest jets (width ≤ 0.00079) at high pT (log_sum_pt > 6.54; 22% of jets, mostly quarks, 54%) and ≈6.47 at lower pT (6.1%, mostly gluons, 60%); ≈6.21 for 0.00079 < width ≤ 0.00902 with sum_pt > 768 GeV (20%, mostly W, 40%) and ≈5.09 with sum_pt ≤ 768 GeV (33%). Above width 0.00902 it falls step by step: ≈3.13 (up to 0.0118), ≈2.06 (up to 0.0148), ≈1.04 (up to 0.0198) and ≈0.215 above 0.0198 (8.9%, 83% tops). It is the main brake on the t score (−3.42, −2.63, −2.52 and −2.07 in the four narrow regimes, only −0.087 for the widest) and adds a little to W (up to +0.593) and Z (up to +0.461); regime_r2 0.812.

Regimes (a small tree on its quantities; R² 0.812):

- `` — 21.7% of jets, value 8.43 (6.75…9.88), formula right 62%
- `` — 6.1% of jets, value 6.47 (4.75…7.88), formula right 60%
- `` — 19.6% of jets, value 6.21 (4.88…7.75), formula right 71%
- `` — 33.4% of jets, value 5.09 (3.62…6.50), formula right 58%
- `` — 3.0% of jets, value 3.13 (1.38…4.62), formula right 60%
- `` — 2.8% of jets, value 2.06 (0.56…3.25), formula right 73%
- `` — 4.3% of jets, value 1.04 (0.00…2.00), formula right 80%
- `` — 8.9% of jets, value 0.21 (0.00…0.62), formula right 84%

```
z = 1.68
if girth < 0.150: z += 67.00 × (0.150 − girth)
if width < 0.016: z += 217 × (0.016 − width)
if e2 < 0.051: z += -88.60 × (0.051 − e2)
if girth < 0.156 and log_sum_pt < 6.91: z += -41.10 × (0.156 − girth) × (6.91 − log_sum_pt)
if girth < 0.148 and pt_7 < 39.50: z += -1.09 × (0.148 − girth) × (39.50 − pt_7)
if tau21 < 0.518 and max_dr > -0.038: z += -14.90 × (0.518 − tau21) × (max_dr − -0.038)
if sum_pt_top5 > 653 and pt_7 < 42.60: z += 0.00059 × (sum_pt_top5 − 653) × (42.60 − pt_7)
if pt_7 < 24.80: z += -0.246 × (24.80 − pt_7)
if pt_0 > 179: z += 0.0022 × (pt_0 − 179)
if C2 > 0.066: z += -52.40 × (C2 − 0.066)
if sum_pt > 998: z += -0.024 × (sum_pt − 998)
if sum_pt_top5 > 677 and z_7 > 0.029: z += -0.566 × (sum_pt_top5 − 677) × (z_7 − 0.029)
if sum_pt > 1070 and n_pt_above_50 > 6.01: z += 0.019 × (sum_pt − 1070) × (n_pt_above_50 − 6.01)
if sum_pt < 764 and z_4 < 0.038: z += 6.47 × (764 − sum_pt) × (0.038 − z_4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Two-prong Z/W jets** — 23.6% of jets, neuron 5.01, formula right for 71%. Mostly Z (41%) with W 33% and tops 15%, 23.6% of jets: mass 55.5 GeV, width 0.0070, 69% of pT at 0.05-0.1. girth < 0.15 (+4.87) and width < 0.0159 (+1.94) set the value, trimmed by girth < 0.156 and log_sum_pt < 6.91 (-1.27), e2 < 0.0509 (99%, -1.05) and tau21 < 0.518 and max_dr > -0.0382 (97%, -0.91). At 5.013 it lowers the top score by about 2 and raises W and Z; the formula calls them Z (45%) and is right 71.1% of the time.
- **Mid-mass W-leaning jets** — 17.0% of jets, neuron 5.40, formula right for 56%. Mostly W (36%) with Z 27%, gluons 16%, tops 11%, quarks 10%, 17.0% of jets: mass 40.5 GeV, width 0.0041, 58% of pT at 0.025-0.05. girth < 0.15 (+6.52) and width < 0.0159 (+2.56) against e2 < 0.0509 (-2.47), girth < 0.156 and log_sum_pt < 6.91 (99%, -1.59) and tau21 < 0.518 and max_dr > -0.0382 (92%, -0.82). At 5.403 it lowers the top score by about 2.2; the formula calls them W (51%) and is right only 55.6% of the time.
- **Narrow light quark/gluon jets** — 15.8% of jets, neuron 8.32, formula right for 54%. Mostly quarks (40%) with 36% gluons, 15.9% of jets: mass 9.6 GeV, width 0.0004, 85% of pT inside 0.025. girth < 0.15 (+9.08) and width < 0.0159 (+3.37) outweigh e2 < 0.0509 (-4.03) and girth < 0.156 and log_sum_pt < 6.91 (95%, -1.19), and tau21 < 0.518 mostly fails here, giving 8.32, the highest value of the neuron. It lowers the top score by about 3.4 and raises W and Z; the formula calls them q (49%) and is right only 53.9% of the time.
- **Soft, narrow gluon jets** — 11.3% of jets, neuron 5.92, formula right for 55%. Mostly gluons (53%) with 21% quarks, 11.3% of jets: mass 12.0 GeV, width 0.0010, soft (sum pT 575 GeV, leading particle 150 GeV). girth < 0.15 (+8.30) and width < 0.0159 (+3.23) against e2 < 0.0509 (-3.56) and, since these jets are soft, girth < 0.156 and log_sum_pt < 6.91 (always, -3.01). At 5.915 it lowers the top score; the formula calls them g (85%) and is right only 54.7% of the time, misreading many quarks and light bosons.
- **Wide top jets beyond girth cut** — 9.3% of jets, neuron 0.27, formula right for 84%. Mostly tops (83%) with 12% gluons, 9.3% of jets: mass 84.3 GeV, width 0.0265, 84% of pT beyond 0.1. These are the jets the neuron is built to exclude: width < 0.0159 never passes and girth < 0.15 only for 51% (+0.43), while C2 > 0.0664 (60%, -1.01) and tau21 < 0.518 and max_dr > -0.0382 (81%, -0.87) subtract. The neuron is small (0.268, on for 46.9%), so the top score is barely lowered; the formula calls them t (95%) and is right 84.1% of the time.
- **Moderately wide top jets** — 7.6% of jets, neuron 1.73, formula right for 75%. Mostly tops (73%) with 17% gluons, 7.6% of jets: mass 67.5 GeV, width 0.0152, pT split between 0.05-0.1 and beyond 0.1. girth < 0.15 passes (+2.77) but width < 0.0159 only for 62% (+0.38), and girth < 0.156 and log_sum_pt < 6.91 (-1.10), tau21 < 0.518 and max_dr > -0.0382 (88%, -1.05) and C2 > 0.0664 (33%, -0.55) pull down. At 1.734 (on for 89.1%) it lowers the top score only mildly; the formula calls them t (92%) and is right 75.3% of the time.
- **Hard-led narrow quark jets** — 7.3% of jets, neuron 7.89, formula right for 66%. Mostly quarks (59%) with W 16%, gluons 12% and Z 12%, 7.3% of jets: mass 14.1 GeV, width 0.0006, hard leading particle (404 GeV) and soft 8th particle (20.6 GeV), 89% of pT inside 0.025. girth < 0.15 (+9.14) and width < 0.0159 (+3.33) against e2 < 0.0509 (-4.03); the soft-tail pair girth < 0.148 and pt_7 < 39.5 (-2.75) and sum_pt_top5 > 653 and pt_7 < 42.6 (98%, +2.13) roughly cancel, and pt_7 < 24.8 (82%) takes another 1.11. At 7.895 it lowers the top score by about 3.2; the formula calls them q (76%) and is right 65.7% of the time.
- **Hard, compact W/Z jets** — 4.1% of jets, neuron 6.06, formula right for 75%. Z (43%) and W (41%) jets, 4.1% of jets: mass 64.4 GeV, width 0.0053, hard (sum pT 911 GeV, leading particle 461 GeV) with a soft tail, 60% of pT at 0.025-0.05. girth < 0.15 (+6.29), sum_pt_top5 > 653 and pt_7 < 42.6 (99%, +2.42) and width < 0.0159 (+2.30) beat e2 < 0.0509 (-2.18), girth < 0.148 and pt_7 < 39.5 (-1.95), tau21 < 0.518 and max_dr > -0.0382 (99%, -1.27) and pt_7 < 24.8 (87%, -1.27). At 6.061 it lowers the top score and raises W and Z; the formula splits them W 48% / Z 46% and is right 75.3% of the time.
- **Single-particle-led quark jets** — 3.5% of jets, neuron 7.71, formula right for 76%. Mostly quarks (71%), 3.5% of jets: mass 17.7 GeV, width 0.0006, hard (sum pT 1003 GeV) with a 545 GeV leading particle and a very soft 8th particle (10.4 GeV). girth < 0.15 (+9.35) and sum_pt_top5 > 653 and pt_7 < 42.6 (+5.74) with width < 0.0159 (+3.31) outweigh girth < 0.148 and pt_7 < 39.5 (-4.36), e2 < 0.0509 (-4.05) and pt_7 < 24.8 (-3.54). At 7.712 it lowers the top score by about 3.1; the formula calls them q (85%) and is right 75.8% of the time.
- **Very hard, many-particle gluon jets** — 0.4% of jets, neuron 5.96, formula right for 68%. Mostly gluons (66%), 0.4% of jets: mass 27.4 GeV, width 0.0011, very hard (sum pT 1314 GeV) with pT shared (8th particle 53.8 GeV). girth < 0.15 (+8.74) and width < 0.0159 (+3.21) are joined by the high-pT pair sum_pt > 998 (-7.61) and sum_pt > 1.07e+03 and n_pt_above_50 > 6.01 (75%, +5.20), plus e2 < 0.0509 (-3.72) and sum_pt_top5 > 677 and z_7 > 0.0289 (74%, -2.67). At 5.961 (on for 95.5%) it lowers the top score; the formula calls them g (87%) and is right 67.5% of the time.

### neuron 14: intermediate width (Z-sized spread) (major)

- **What it measures:** Pushed up for girth2 < 0.0131 and e2 < 0.0385 but down for the narrowest jets (width < 0.00741, girth < 0.0869), so it peaks for jets of intermediate width, a bit wider than a typical W; it rises with the number and pT share of particles at 0.1-0.4 in ΔR (rank correlations about 0.33-0.34). Z jets sit highest (1.50), well above tops (0.43), gluons (0.24), W (0.18) and quarks (0.16).
- *computed — its value:* largest for Z (1.50), then t (0.43), then g (0.24), then W (0.18), then q (0.16); it separates Z jets from the rest best (AUC 0.78: large for Z)
- **How the class scores use it:** It is the W/Z separator: Z jets sit high and W jets low on it, so it raises the Z score and lowers the W score. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+7%); lowers the score of W (-9%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Highest (≈2.62) for jets of intermediate spread (girth2 > 0.00562 but width ≤ 0.0111) with large lam1 (> 0.00691): 11% of jets, mostly Z (59%); ≈0.976 for the same with lam1 ≤ 0.00691 (12%, mostly W, 41%) and ≈1.1 for girth2 ≤ 0.00562 with max_dr > 0.159 and lam1 > 0.00345 (5.4%, mostly Z, 43%). It is off (≈0.01) for the 46% of compact jets with girth2 ≤ 0.00562 and max_dr ≤ 0.131, and ≈0.037 for width > 0.0132 (15%, 81% tops). It separates Z from W: +0.983 on Z and −1.97 on W in the top regime; regime_r2 0.628.

Regimes (a small tree on its quantities; R² 0.628):

- `` — 10.8% of jets, value 2.62 (1.00…4.19), formula right 73%
- `` — 5.4% of jets, value 1.10 (0.00…2.31), formula right 65%
- `` — 12.4% of jets, value 0.98 (0.00…2.62), formula right 68%
- `` — 2.0% of jets, value 0.48 (0.00…1.62), formula right 68%
- `` — 4.4% of jets, value 0.27 (0.00…1.06), formula right 63%
- `` — 4.0% of jets, value 0.26 (0.00…1.00), formula right 58%
- `` — 14.7% of jets, value 0.04 (0.00…0.00), formula right 82%
- `` — 46.3% of jets, value 0.01 (0.00…0.00), formula right 58%

```
z = -1.51
if girth2 < 0.013: z += 632 × (0.013 − girth2)
if width < 0.0074: z += -1040 × (0.0074 − width)
if girth < 0.087: z += -80.10 × (0.087 − girth)
if e2 < 0.038: z += 106 × (0.038 − e2)
if max_dr < 0.177: z += -20.10 × (0.177 − max_dr)
if z_dr_0p05_0p1 < 0.597 and C2 < 0.066: z += -77.00 × (0.597 − z_dr_0p05_0p1) × (0.066 − C2)
if girth2 < 0.0045 and n_dr_0p2_0p4 < 0.930: z += -486 × (0.0045 − girth2) × (0.930 − n_dr_0p2_0p4)
if girth2 < 0.013 and eccentricity > 0.971: z += 11600 × (0.013 − girth2) × (eccentricity − 0.971)
if z_dr_0p05_0p1 < 0.572: z += 1.85 × (0.572 − z_dr_0p05_0p1)
if mass > 5.61: z += 0.019 × (mass − 5.61)
if width < 0.0076 and planar_flow < 0.109: z += -6890 × (0.0076 − width) × (0.109 − planar_flow)
if width < 0.0086 and D2 < 1.04: z += -986 × (0.0086 − width) × (1.04 − D2)
if planar_flow < 0.110 and centroid_offset > 0.0095: z += 1270 × (0.110 − planar_flow) × (centroid_offset − 0.0095)
if width < 0.0079 and log_sum_pt < 6.82: z += 381 × (0.0079 − width) × (6.82 − log_sum_pt)
if planar_flow < 0.116 and max_dr < 0.164: z += -187 × (0.116 − planar_flow) × (0.164 − max_dr)
if lam1 > 0.0073: z += -101 × (lam1 − 0.0073)
if planar_flow < 0.116 and centroid_offset > 0.019: z += -1310 × (0.116 − planar_flow) × (centroid_offset − 0.019)
if D2 < 1.04 and centroid_offset < 0.031: z += 38.50 × (1.04 − D2) × (0.031 − centroid_offset)
if lam1 > 0.0055 and max_dr < 0.168: z += 6850 × (lam1 − 0.0055) × (0.168 − max_dr)
if mass < 75.30 and D2 < 0.865: z += -0.049 × (75.30 − mass) × (0.865 − D2)
if planar_flow < 0.106 and sum_pt < 740: z += -0.051 × (0.106 − planar_flow) × (740 − sum_pt)
if e2 < 0.038 and D2 < 0.961: z += 192 × (0.038 − e2) × (0.961 − D2)
if lam1 > 0.0073 and max_dr < 0.134: z += 43000 × (lam1 − 0.0073) × (0.134 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 26.6% of jets, neuron 0.00, formula right for 62%. Mostly quarks (47%) with 35% gluons, 26.6% of jets: mass 7.9 GeV, width 0.0002, 92% of pT inside 0.025. girth2 < 0.0131 (+8.14) and e2 < 0.0385 (+3.56) are outweighed by the narrow-jet penalties width < 0.00741 (-7.47), girth < 0.0869 (-6.04), max_dr < 0.177 (-2.92), z_dr_0p05_0p1 < 0.597 and C2 < 0.0662 (-2.58) and girth2 < 0.00453 and n_dr_0p2_0p4 < 0.93 (-1.95), so the neuron is off and adds nothing. The formula calls them q (55%) and is right 61.9% of the time.
- **Wide top jets** — 15.8% of jets, neuron 0.13, formula right for 81%. Mostly tops (80%) with 14% gluons, 15.8% of jets: mass 77.0 GeV, width 0.0212, 64% of pT beyond 0.1. Too wide for the main bonus girth2 < 0.0131 (passes 15%); mass > 5.61 (+1.33) and z_dr_0p05_0p1 < 0.572 (75%, +0.59) are cancelled by lam1 > 0.00732 (always, -1.13), so the neuron is on for only 17.9% (mean 0.13). The formula calls them t (95%) and is right 81.2% of the time.
- **Z-width two-prong jets** — 14.3% of jets, neuron 2.18, formula right for 70%. Mostly Z (57%) with tops 18% and W 12%, 14.3% of jets: mass 57.0 GeV, width 0.0076, 65% of pT at 0.05-0.1. girth2 < 0.0131 always passes (+3.47) while these jets are just wide enough that width < 0.00741 hits only 46% (-0.32) and girth < 0.0869 costs 0.58; with mass > 5.61 (+0.96) and girth2 < 0.0135 and eccentricity > 0.971 (69%, +0.94) against max_dr < 0.177 (86%, -0.92), the neuron is on for 92.9% at 2.184, its highest value. That lowers the W score by about 1.6 and raises Z; the formula calls them Z (68%) and is right 70% of the time.
- **Light narrow gluon-rich jets** — 9.2% of jets, neuron 0.00, formula right for 50%. Mostly gluons (38%) with quarks 25%, W 17%, Z 14%, 9.2% of jets: mass 21.6 GeV, width 0.0015, pT at 0-0.05 from the axis. girth2 < 0.0131 (+7.31) and e2 < 0.0385 (+2.62) lose to width < 0.00741 (-6.11), girth < 0.0869 (-4.34), max_dr < 0.177 (90%, -1.76), z_dr_0p05_0p1 < 0.597 and C2 < 0.0662 (96%, -1.49) and girth2 < 0.00453 and n_dr_0p2_0p4 < 0.93 (93%, -1.26), so the neuron is off (on for 1.3%). The formula calls them g (55%) and is right only 49.8% of the time.
- **Clean W two-prong jets** — 8.4% of jets, neuron 0.22, formula right for 75%. Mostly W (68%) with 20% Z, 8.4% of jets: mass 54.5 GeV, width 0.0056, 82% of pT at 0.05-0.1 from the axis. girth2 < 0.0131 (+4.74) and girth2 < 0.0135 and eccentricity > 0.971 (98%, +2.02) are taken back by width < 0.00741 (-1.89), width < 0.00857 and D2 < 1.04 (99%, -1.82), max_dr < 0.177 (-1.48), girth < 0.0869 (-1.20), width < 0.00758 and planar_flow < 0.109 (98%, -1.10) and planar_flow < 0.116 and max_dr < 0.164 (99%, -1.00); on for only 21.9% (mean 0.22), it spares the W score. The formula calls them W (87%) and is right 74.7% of the time.
- **Mid-mass W-led mixture** — 7.8% of jets, neuron 0.38, formula right for 52%. A mixture led by W (35%) with Z 24%, gluons 19%, tops 12%, 7.8% of jets: mass 39.1 GeV, width 0.0043, pT at 0.025-0.1. girth2 < 0.0131 (+5.55) and e2 < 0.0385 (97%, +1.24) against width < 0.00741 (-3.21), girth < 0.0869 (-2.45) and max_dr < 0.177 (77%, -0.92); on for 39.5% (mean 0.381), it slightly lowers W and raises Z. The formula calls them W (51%) and is right only 51.5% of the time.
- **Hard, compact, flat W jets** — 6.4% of jets, neuron 0.52, formula right for 61%. Mostly W (45%) with 30% Z, 6.4% of jets: mass 43.0 GeV, width 0.0034, hard leading particle (308 GeV), 62% of pT at 0.025-0.05. girth2 < 0.0131 (+6.14), girth2 < 0.0135 and eccentricity > 0.971 (+2.38) and e2 < 0.0385 (+2.29) are matched by width < 0.00741 (-4.18), girth < 0.0869 (-3.47), width < 0.00758 and planar_flow < 0.109 (-2.15) and z_dr_0p05_0p1 < 0.597 and C2 < 0.0662 (88%, -1.17); on for 47.8% (mean 0.52). The formula calls them W (65%) and is right 60.9% of the time.
- **Hard, compact W/Z jets** — 6.1% of jets, neuron 1.16, formula right for 72%. An even Z (43%) / W (42%) mix, 6.1% of jets: mass 58.5 GeV, width 0.0055, hard leading particle (306 GeV), 70% of pT at 0.025-0.05. girth2 < 0.0131 (+4.79), girth2 < 0.0135 and eccentricity > 0.971 (99%, +2.02), e2 < 0.0385 (99%, +1.11) and mass > 5.61 (+0.99) beat girth < 0.0869 (-2.28), width < 0.00741 (99%, -1.96), z_dr_0p05_0p1 < 0.597 and C2 < 0.0662 (95%, -1.67) and width < 0.00758 and planar_flow < 0.109 (98%, -1.15). On for 65.2% at 1.155, it lowers W and raises Z; the formula splits them W 51% / Z 46% and is right 72.4% of the time.
- **Light, flat, elongated mixed jets** — 3.2% of jets, neuron 0.00, formula right for 46%. A mixture of gluons (30%), quarks (26%), W (19%) and Z (18%), 3.2% of jets: mass 14.8 GeV, width 0.0012, 56% of pT at 0.025-0.05. Nearly every test fires: girth2 < 0.0131 (+7.50), e2 < 0.0385 (+3.22), girth2 < 0.0135 and eccentricity > 0.971 (+2.42) against width < 0.00741 (-6.43), girth < 0.0869 (-4.47), width < 0.00758 and planar_flow < 0.109 (-2.72), max_dr < 0.177 (98%, -2.31) and z_dr_0p05_0p1 < 0.597 and C2 < 0.0662 (97%, -2.15), leaving the neuron off. The formula leans g (32%) and is right only 45.9% of the time: a badly mixed group.
- **Flat, wide top/gluon jets** — 2.2% of jets, neuron 0.90, formula right for 63%. Mostly tops (56%) with gluons 27% and quarks 15%, 2.2% of jets: mass 64.7 GeV, width 0.0180, 59% of pT beyond 0.1. This group is marked by planar_flow < 0.11 and centroid_offset > 0.00946 (always, +5.01) against planar_flow < 0.116 and centroid_offset > 0.0186 (always, -4.46), plus mass > 5.61 (97%, +1.11), lam1 > 0.00732 (92%, -1.08) and girth2 < 0.0131 (35%); on for 33.2% (mean 0.895), it lowers W and raises Z a little. The formula calls them t (78%) and is right only 62.9% of the time, mistaking gluons and quarks.

### neuron 0: compact massive two-prong-ness (moderate)

- **What it measures:** Pushed down for light jets (mass < 22.5 GeV and < 29.3 GeV are its two strongest terms) and up for compact jets (girth2 < 0.0121); it rises with eccentricity and falls with τ21 and planar flow, so it measures how much the jet looks like a compact, elongated two-prong object with real mass. Z (2.28) and W (1.95) jets sit highest, tops in the middle (0.76), gluons (0.28) and quarks (0.22) lowest.
- *computed — its value:* largest for Z (2.28), then W (1.95), then t (0.76), then g (0.28), then q (0.22); it separates Z jets from the rest best (AUC 0.75: large for Z)
- **How the class scores use it:** Because W jets sit high on it and gluons low, the W score reads it as evidence for a W and the g score as evidence against a gluon: it raises the W score and lowers the g score. The q, Z and t scores do not use it (the Z score relies on the related neuron 7 instead).
- *computed — used by:* raises the score of W (+9%); lowers the score of g (-6%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** Highest (≈3.11, typically 1.62 to 4.5) for very elongated (eccentricity > 0.985), compact (girth2 ≤ 0.0107) jets with lam1 > 0.00344, which are 24% of jets and mostly Z (43%, with 39% W). It falls to ≈1.49 for less elongated jets (eccentricity ≤ 0.985) that still have lam1 > 0.00295 and width ≤ 0.0098 (15% of jets), to ≈0.35 once width > 0.0098 (13%, 82% tops), and it is off (≈0.003) for the 36% of jets with eccentricity ≤ 0.985, lam1 ≤ 0.00295 and mass ≤ 30.8 GeV (mostly quarks, 42%). The top regime is the one that matters, adding +1.07 to the W score and −0.486 to the g score (+0.049 to t); the regimes describe the neuron reasonably well (regime_r2 0.719).

Regimes (a small tree on its quantities; R² 0.719):

- `` — 24.4% of jets, value 3.11 (1.62…4.50), formula right 74%
- `` — 14.7% of jets, value 1.49 (0.00…2.62), formula right 57%
- `` — 2.0% of jets, value 1.01 (0.00…2.25), formula right 74%
- `` — 4.8% of jets, value 0.91 (0.00…2.62), formula right 51%
- `` — 12.7% of jets, value 0.35 (0.00…1.12), formula right 83%
- `` — 2.5% of jets, value 0.28 (0.00…1.00), formula right 49%
- `` — 2.9% of jets, value 0.25 (0.00…0.88), formula right 69%
- `` — 36.0% of jets, value 0.00 (0.00…0.00), formula right 59%

```
z = -0.480
if mass < 22.50: z += -0.543 × (22.50 − mass)
if mass < 29.30 and phi_1 > -0.077: z += -4.03 × (29.30 − mass) × (phi_1 − -0.077)
if girth2 < 0.012: z += 301 × (0.012 − girth2)
if girth < 0.077: z += -56.80 × (0.077 − girth)
if centroid_offset < 0.034: z += 71.00 × (0.034 − centroid_offset)
if width < 0.0045: z += -799 × (0.0045 − width)
if mass < 71.30: z += -0.026 × (71.30 − mass)
if z_dr_0_0p05 > 0.852: z += 15.40 × (z_dr_0_0p05 − 0.852)
if girth2 < 0.020 and eccentricity > 0.961: z += 3910 × (0.020 − girth2) × (eccentricity − 0.961)
if mass < 64.90 and pt_7 < 40.10: z += -0.0022 × (64.90 − mass) × (40.10 − pt_7)
if mass < 64.80 and centroid_offset > 0.011: z += 2.55 × (64.80 − mass) × (centroid_offset − 0.011)
if sum_pt > 895: z += -0.028 × (sum_pt − 895)
if sum_pt > 871 and pt_7 < 27.80: z += 0.0018 × (sum_pt − 871) × (27.80 − pt_7)
if planar_flow < 0.148 and sum_pt_top2 < 388: z += -0.052 × (0.148 − planar_flow) × (388 − sum_pt_top2)
if lam1 < 0.0065 and D2 < 0.886: z += -2060 × (0.0065 − lam1) × (0.886 − D2)
if sum_pt_top5 > 655 and z_7 > 0.027: z += -0.514 × (sum_pt_top5 − 655) × (z_7 − 0.027)
if mass < 30.00 and D2 < 0.910: z += -2.90 × (30.00 − mass) × (0.910 − D2)
if eccentricity > 0.997: z += 499 × (eccentricity − 0.997)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Compact two-prong W/Z jets** — 30.5% of jets, neuron 2.77, formula right for 72%. A 41% W / 41% Z mix with 10% tops (30.5% of jets): mass 55.6 GeV, ordinary width 0.0060 and average pT, with the pT sitting at moderate distance (0.025-0.1) from the axis rather than in the core, as for two resolved prongs. Nothing here is light enough to trip mass < 22.5, and the compactness tests girth2 < 0.0121 (+1.82) and centroid_offset < 0.0338 (+1.59) always pass, joined by girth2 < 0.0197 and eccentricity > 0.961 (88%, +1.41); only girth < 0.0775 and mass < 71.3 take a little back. The neuron is on for 96.9% of these jets at 2.767, its highest value, which lowers the gluon score and raises the W score; the formula splits them W 52% / Z 43% and is right 72% of the time, with W-Z swaps behind most misses.
- **Wide, massive top jets** — 18.4% of jets, neuron 0.43, formula right for 78%. Mostly tops (74%, plus 14% gluons), 18.4% of jets: mass 76.0 GeV, width 0.0201 (about three times average), soft (sum pT 577 GeV) and spread out, with 60% of the pT beyond 0.1 from the axis. These jets are too wide to pass girth2 < 0.0121 (only 19%) and too heavy for any of the mass cuts, so almost no if-statement fires strongly; centroid_offset < 0.0338 (61%, +0.62) is the main contributor. The neuron stays small (0.428, on for 46.4%) and barely touches the scores; the formula calls them t (88%) using other neurons and is right 78% of the time.
- **Pencil-thin light quark/gluon jets** — 17.5% of jets, neuron 0.00, formula right for 61%. Mostly quarks (46%) with 37% gluons, 17.5% of jets: mass only 5.9 GeV, width 0.0002, with 95% of the pT inside 0.025 of the axis. The two light-mass tests mass < 22.5 (-9.01) and mass < 29.3 and phi_1 > -0.0771 (-7.30) always pass and overwhelm the compactness bonuses (girth2 < 0.0121 +3.59, z_dr_0_0p05 > 0.852 +2.27), so the neuron is always at zero and adds nothing to any score. The formula splits them q 52% / g 42% and is right only 60.8% of the time: quark-gluon confusion is the main error here.
- **Narrow mid-mass mixed jets** — 10.8% of jets, neuron 0.71, formula right for 52%. A genuine mixture (W 29%, gluons 23%, quarks 23%, Z 18%), 10.8% of jets: mass 33.1 GeV, narrow (width 0.0023), a harder-than-average leading particle, and 87% of the pT within 0.05 of the axis. All compactness tests pass (girth2 < 0.0121 +2.95, centroid_offset < 0.0338 +1.38), but the jets are also narrow enough to be hit by girth < 0.0775 (-2.35) and width < 0.00449 (-1.75), and mass < 29.3 catches 42% of them; whether the neuron ends up on (40.2%, mean 0.709) depends on the partial tests z_dr_0_0p05 > 0.852 (73%) and girth2 < 0.0197 and eccentricity > 0.961 (57%). It lowers the gluon score and raises the W score moderately; the formula calls them W (42%) and is right only 52.4% of the time, a clear trouble spot.
- **Light narrow gluon/quark jets** — 8.1% of jets, neuron 0.00, formula right for 56%. Mostly gluons (40%) with 37% quarks, 8.1% of jets: mass 13.8 GeV, width 0.0007, 68% of the pT within 0.025 but a noticeable 27% at 0.025-0.05. Both light-mass tests pass for all jets (mass < 22.5 -4.72, mass < 29.3 and phi_1 > -0.0771 -4.73) and, together with girth < 0.0775 (-3.18) and width < 0.00449 (-3.03), cancel the compactness bonuses; the neuron is always zero and adds nothing. The formula calls them g (52%) and is right 56.2% of the time, again limited by quark-gluon confusion.
- **Soft, off-centre mid-mass mixture** — 5.9% of jets, neuron 1.46, formula right for 52%. A mixture of tops (31%), gluons (26%), Z (20%) and W (13%), 5.9% of jets: mass 35.8 GeV, width 0.0073, soft (sum pT 543 GeV, leading particle 153 GeV) with most pT at 0.05-0.1 from the axis. What sets this group apart is mass < 64.8 and centroid_offset > 0.0106, which always passes (+2.29), together with girth2 < 0.0121 (87%, +1.65); centroid_offset < 0.0338 passes for only 39%. The neuron is on for 83.8% at 1.464, lowering the gluon score and raising the W score; the formula leans t (34%) with Z and g close behind and is right only 51.6% of the time.
- **Very light but lopsided jets** — 3.3% of jets, neuron 0.01, formula right for 44%. A mixture led by gluons (35%) with Z 24%, W 18%, quarks 12%, tops 11%, 3.3% of jets: mass only 6.2 GeV yet width 0.0018, with 70% of the pT at 0.025-0.05 from the axis and a soft leading particle. The light-mass tests mass < 22.5 (-8.83) and mass < 29.3 and phi_1 > -0.0771 (-7.40) dominate; the large off-centre bonus mass < 64.8 and centroid_offset > 0.0106 (+4.07) and girth2 < 0.0121 (+3.11) cannot rescue them, so the neuron is essentially off (on for 0.5%). The formula splits them g 52% / Z 37% and is right only 44.4% of the time: many boson jets that lost their mass land here and are mistaken.
- **Hard, single-particle quark jets** — 3.1% of jets, neuron 0.00, formula right for 76%. Mostly quarks (67%, plus 19% gluons), 3.1% of jets: mass 8.1 GeV, very hard (sum pT 1062 GeV) with the leading particle carrying 515 GeV, a soft 8th particle and 98% of the pT inside 0.025. Both light-mass tests pass (-7.82, -6.56), and the high-pT test sum_pt > 895 (-4.60) plus mass < 64.9 and pt_7 < 40.1 (-2.81) push further down, only partly offset by sum_pt > 871 and pt_7 < 27.8 (81%, +3.20); the neuron is always zero and adds nothing. The formula calls them q (80%) and is right 75.9% of the time.
- **Very hard, massive mixed jets** — 1.6% of jets, neuron 1.06, formula right for 62%. An even mixture of W (28%), Z (24%), quarks (23%) and gluons (22%), 1.6% of jets: mass 54.0 GeV, sum pT 1092 GeV with a 538 GeV leading particle, and half the pT inside 0.025. sum_pt > 895 always passes (-5.45) and is partly undone by sum_pt > 871 and pt_7 < 27.8 (65%, +2.70); the compactness tests girth2 < 0.0121 (+2.74) and centroid_offset < 0.0338 (+1.88) fight girth < 0.0775 (-2.40) and width < 0.00449 (-1.58). The result is on for only 32.5% (mean 1.063), mildly lowering g and raising W; the formula leans W (31%) and is right 62.4% of the time in this hard-to-separate mix.
- **Light low-D2 gluon/quark jets** — 0.8% of jets, neuron 0.00, formula right for 44%. Mostly gluons (41%) with 33% quarks, 0.8% of jets: mass 19.4 GeV, width 0.0014, pT shared more evenly than average (8th particle 39.7 GeV). This small group is defined by mass < 30 and D2 < 0.91, which always passes and subtracts 7.38, backed by mass < 29.3 and phi_1 > -0.0771 (-3.14) and lam1 < 0.00645 and D2 < 0.886 (-2.48); the neuron is always zero and adds nothing. The formula splits them g 50% / q 40% and is right only 43.9% of the time.

### neuron 1: moderate width at high pT (moderate)

- **What it measures:** Pushed down for narrow jets (width < 0.009, its strongest term) and up for small e2_sq and for high total pT (log of total pT > 6.41); it follows width, girth and mass (rank correlations about +0.44 to +0.46). Z (0.75) and top (0.74) jets sit highest, gluons in the middle (0.53), W (0.31) and quarks (0.21) lowest; its clearest separation is that it is small for quarks.
- *computed — its value:* largest for Z (0.75), then t (0.74), then g (0.53), then W (0.31), then q (0.21); it separates q jets from the rest best (AUC 0.33: small for q)
- **How the class scores use it:** The g score uses it mainly to tell gluons from quarks, which sit lowest on it: it raises the g score and, only slightly, the Z score. It does not (or hardly) enter the q, W or t scores; the Z and top jets that also sit high on it are kept out of the g score by other scales.
- *computed — used by:* raises the score of g (+7%), Z (+2%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** Highest (≈2.56) for massive jets (mass > 61.8 GeV) whose 8th-hardest particle is still hard (pt_7 > 41.5 GeV) at high total pT (log_sum_pt > 6.5), but these are only 3.1% of jets (mostly Z, 44%); it is also sizeable (≈1.65) for lighter jets (mass ≤ 61.8 GeV) with an off-axis pT centroid (centroid_offset > 0.0333) and e2 ≤ 0.0396 (5.4%, gluon-rich at 32%), and ≈0.987 for massive jets with pt_7 ≤ 41.5 GeV and tau32 > 0.311 (12%). It is nearly off (≈0.119) for the 48% of light, centred jets (mass ≤ 61.8 GeV, centroid_offset ≤ 0.0162). It adds up to +0.998 to the g score and +0.319 to the Z score with a small −0.08 on W; the regimes describe it only roughly (regime_r2 0.424).

Regimes (a small tree on its quantities; R² 0.424):

- `` — 3.1% of jets, value 2.56 (1.00…4.50), formula right 78%
- `` — 5.3% of jets, value 1.65 (0.38…3.12), formula right 50%
- `` — 2.0% of jets, value 1.58 (0.25…2.75), formula right 79%
- `` — 12.2% of jets, value 0.99 (0.00…2.12), formula right 79%
- `` — 2.5% of jets, value 0.83 (0.00…1.62), formula right 72%
- `` — 21.3% of jets, value 0.46 (0.00…1.38), formula right 55%
- `` — 5.9% of jets, value 0.28 (0.00…1.00), formula right 91%
- `` — 47.7% of jets, value 0.12 (0.00…0.50), formula right 64%

```
z = 1.20
if width < 0.009: z += -657 × (0.009 − width)
if e2_sq < 0.0077: z += 470 × (0.0077 − e2_sq)
if log_sum_pt > 6.59 and lam2 < 0.0012: z += 10300 × (log_sum_pt − 6.59) × (0.0012 − lam2)
if log_sum_pt > 6.41: z += 4.58 × (log_sum_pt − 6.41)
if pt_7 > 30.40: z += 0.115 × (pt_7 − 30.40)
if pt_7 > 29.20 and mass < 97.00: z += -0.0015 × (pt_7 − 29.20) × (97.00 − mass)
if log_sum_pt > 6.40 and centroid_offset < 0.024: z += -235 × (log_sum_pt − 6.40) × (0.024 − centroid_offset)
if C2 < 0.049: z += -23.70 × (0.049 − C2)
if log_sum_pt > 6.33 and max_dr < 0.191: z += 19.00 × (log_sum_pt − 6.33) × (0.191 − max_dr)
if e2_sq < 0.0078 and planar_flow < 0.085: z += -8390 × (0.0078 − e2_sq) × (0.085 − planar_flow)
if log_sum_pt > 6.60 and girth2_top3 < 0.0063: z += -1090 × (log_sum_pt − 6.60) × (0.0063 − girth2_top3)
if z_7 < 0.056: z += -33.70 × (0.056 − z_7)
if max_dr < 0.045: z += -63.30 × (0.045 − max_dr)
if e2 < 0.038 and eccentricity > 0.979: z += 6000 × (0.038 − e2) × (eccentricity − 0.979)
if e2 > 0.031 and tau32 < 0.629: z += -82.70 × (e2 − 0.031) × (0.629 − tau32)
if lam1 < 0.0083 and centroid_offset > 0.021: z += 14300 × (0.0083 − lam1) × (centroid_offset − 0.021)
if log_sum_pt > 6.59 and n_pt_above_50 > 7.01: z += -7.05 × (log_sum_pt − 6.59) × (n_pt_above_50 − 7.01)
if pt_7 > 32.90 and centroid_offset > 0.014: z += 1.53 × (pt_7 − 32.90) × (centroid_offset − 0.014)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Wide, soft top jets** — 24.1% of jets, neuron 0.84, formula right for 76%. Mostly tops (64%) with 15% Z and 14% gluons, 24.1% of jets: mass 69.7 GeV, width 0.0175 (well above average), soft (sum pT 575 GeV) with 53% of the pT beyond 0.1 from the axis. These jets are too wide for the strong narrow-jet penalty width < 0.009 (passes 23%) and for e2_sq < 0.00775, so the neuron is built from the soft-particle tests pt_7 > 30.4 (72%, +0.71) minus pt_7 > 29.2 and mass < 97 (-0.29), with e2 > 0.031 and tau32 < 0.629 (71%, -0.71) and C2 < 0.0491 (-0.31) pulling down. It is on for 73.7% at 0.842, raising the gluon and Z scores; the formula calls them t (75%) and is right 76.3% of the time.
- **Moderate-pT two-prong W/Z jets** — 17.7% of jets, neuron 0.41, formula right for 63%. Mostly W (43%) with Z 30% and tops 12%, 17.7% of jets: mass 46.5 GeV, width 0.0056, slightly below-average pT with most pT at 0.05-0.1 from the axis. width < 0.009 always passes (-2.23) and e2_sq < 0.00775 only half compensates (+1.20); the rest comes from pt_7 > 30.4 (84%, +1.09) against pt_7 > 29.2 and mass < 97 (-0.79) and log_sum_pt > 6.41 (68%, +0.46). The neuron is on for 43.9% at 0.414, a modest push to the g and Z scores; the formula calls them W (59%) and is right 62.6% of the time, with W-Z confusion.
- **Light, fairly narrow gluon jets** — 13.4% of jets, neuron 0.21, formula right for 49%. Mostly gluons (41%) with quarks 24%, W 15%, Z 12%, 13.4% of jets: mass 17.0 GeV, width 0.0014, pT a bit below average and concentrated within 0.05 of the axis. The narrow-jet penalty width < 0.009 (-4.97) is only partly offset by e2_sq < 0.00775 (+3.18); what keeps some jets on is log_sum_pt > 6.41 (58%) and pt_7 > 30.4 (64%), while C2 < 0.0491 (-0.68) pulls down. The neuron is on for 20.3% at 0.212, a small push to g and Z; the formula calls them g (58%) and is right only 49.3% of the time, with quarks and light bosons mixed in.
- **High-pT narrow quark jets** — 9.3% of jets, neuron 0.13, formula right for 60%. Mostly quarks (55%, plus 20% gluons), 9.3% of jets: mass 10.5 GeV, width 0.0004, harder than average (sum pT 859 GeV, leading particle 335 GeV) with 90% of the pT inside 0.025. width < 0.009 (-5.68) against e2_sq < 0.00775 (+3.53) sets a negative base; the high-pT tests log_sum_pt > 6.59 and lam2 < 0.0012 (+1.97), log_sum_pt > 6.41 (+1.58) and log_sum_pt > 6.33 and max_dr < 0.191 (+1.19) are cancelled by log_sum_pt > 6.4 and centroid_offset < 0.0235 (-1.30), log_sum_pt > 6.6 and girth2_top3 < 0.0063 (-1.02) and max_dr < 0.0455 (73%, -1.00). The neuron is on for only 14.4% (mean 0.131) and matters little; the formula calls them q (78%) and is right 60.1% of the time.
- **Narrow, evenly shared gluon jets** — 8.2% of jets, neuron 0.11, formula right for 55%. Mostly gluons (52%) with 28% quarks, 8.3% of jets: mass 10.5 GeV, width 0.0006, a soft leading particle (168 GeV) but a hard 8th particle (48.0 GeV), i.e. pT shared among many particles. pt_7 > 30.4 (+2.03) and pt_7 > 29.2 and mass < 97 (-2.51) always pass together and cancel, as do width < 0.009 (-5.52) and e2_sq < 0.00775 (+3.47); max_dr < 0.0455 (77%, -0.98) keeps the total negative. The neuron is on for only 9% (mean 0.108) and adds little; the formula calls them g (78%) and is right 54.5% of the time.
- **High-pT heavy Z/W jets** — 8.2% of jets, neuron 1.22, formula right for 81%. Mostly Z (52%) with 35% W, 8.2% of jets: mass 70.7 GeV, width 0.0069, hard (sum pT 864 GeV) with most pT at 0.05-0.1 from the axis. The high-pT tests log_sum_pt > 6.59 and lam2 < 0.0012 (+1.97) and log_sum_pt > 6.41 (+1.60) always pass, and since these jets are not as narrow the penalty width < 0.009 (-1.57) is smaller than for light jets; centroid_offset < 0.0235 (-1.31), C2 < 0.0491 (-0.73) and z_7 < 0.0562 (-0.69) take some back. The neuron reaches its highest value here (1.218, on for 79.7%), raising the gluon and Z scores; the formula calls them Z (55%) and is right 80.8% of the time.
- **Very hard, pencil-thin quark jets** — 6.8% of jets, neuron 0.19, formula right for 72%. Mostly quarks (67%, plus 17% gluons), 6.8% of jets: mass 10.9 GeV, width 0.0002, very hard (sum pT 1011 GeV, leading particle 476 GeV) with 96% of the pT inside 0.025. Large high-pT bonuses (log_sum_pt > 6.59 and lam2 < 0.0012 +3.94, log_sum_pt > 6.41 +2.31, log_sum_pt > 6.33 and max_dr < 0.191 +1.55) are cancelled by the high-pT compactness penalties log_sum_pt > 6.4 and centroid_offset < 0.0235 (-2.44), log_sum_pt > 6.6 and girth2_top3 < 0.0063 (-2.13) and z_7 < 0.0562 (-1.21), on top of width < 0.009 (-5.77) versus e2_sq < 0.00775 (+3.55). The neuron is on for 16.8% (mean 0.19) and matters little; the formula calls them q (86%) and is right 72.2% of the time.
- **Hard, flat two-prong W jets** — 5.3% of jets, neuron 0.76, formula right for 69%. Mostly W (51%) with 31% Z, 5.3% of jets: mass 53.7 GeV, width 0.0038, hard (sum pT 916 GeV, leading particle 408 GeV) with 85% of the pT within 0.05. Besides the high-pT bonuses (+2.72, +1.86), this group is marked by e2 < 0.0384 and eccentricity > 0.979 (92%, +1.40) against e2_sq < 0.00778 and planar_flow < 0.0847 (93%, -1.77), i.e. elongated, flat jets; width < 0.009 (-3.44), centroid_offset < 0.0235 (-1.17), girth2_top3 < 0.0063 (-1.12) and z_7 < 0.0562 (-0.93) pull down. It is on for 68.5% at 0.764, raising the gluon and Z scores and lowering W slightly; the formula calls them W (66%) and is right 68.7% of the time.
- **Flat, elongated mid-mass mixture** — 5.0% of jets, neuron 0.56, formula right for 55%. Mostly W (36%) but mixed with Z 26%, gluons 15%, quarks 13% and tops 10%, 5.0% of jets: mass 30.3 GeV, width 0.0031, average pT with 61% of the pT at 0.025-0.05 from the axis. e2_sq < 0.00778 and planar_flow < 0.0847 (-2.62) and e2 < 0.0384 and eccentricity > 0.979 (+1.90) always pass together, alongside width < 0.009 (-3.88) versus e2_sq < 0.00775 (+2.60); lam1 < 0.0083 and centroid_offset > 0.021 (59%, +0.52) is most common here. The neuron is on for 47.6% at 0.558, a moderate push to g and Z; the formula calls them W (49%) and is right only 55% of the time.
- **Hard, many-particle narrow gluon jets** — 2.0% of jets, neuron 0.22, formula right for 66%. Mostly gluons (58%, plus 27% quarks), 2.0% of jets: mass 12.0 GeV, width 0.0004, hard (sum pT 962 GeV) with pT shared evenly (8th particle 59.8 GeV vs 34.66 on average) yet 85% within 0.025. Almost every test fires: pt_7 > 30.4 (+3.38) is outweighed by pt_7 > 29.2 and mass < 97 (-3.96), the high-pT bonuses (+3.26, +2.06, +1.65) by width < 0.009 (-5.64), centroid_offset < 0.0235 (-1.88), girth2_top3 < 0.0063 (-1.67), max_dr < 0.0455 (-1.49), and log_sum_pt > 6.59 and n_pt_above_50 > 7.01 (85%, -1.61) is unique to this group. The neuron is on for only 12.4% (mean 0.218); the formula calls them g (70%) and is right 66% of the time.

### neuron 3: radiation at wide angle (ΔR 0.2-0.4) (moderate)

- **What it measures:** Switched off for narrow jets (girth2 < 0.00905 and girth2 < 0.0126 are its two strongest terms, both pushing down) and pushed up for e2 < 0.0433; it follows the pT share and number of particles at 0.2 ≤ ΔR < 0.4 (rank correlations 0.604 and 0.602). Tops sit clearly highest (1.80), then gluons (0.61) and quarks (0.32); Z jets are rarely on it (0.15) and W jets almost never (0.01, zero for 98% of them).
- *computed — its value:* largest for t (1.80), then g (0.61), then q (0.32), then Z (0.15), then W (0.01); it separates t jets from the rest best (AUC 0.71: large for t)
- **How the class scores use it:** A clean two-prong boson has little pT at such wide angles, so the W and Z scores read this scale as evidence against a boson: it lowers the W score and the Z score. It does not (or hardly) enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-7%), Z (-11%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Off (≈0.031) for the 72% of jets that are narrow (girth2 ≤ 0.00873) with a centred pT (centroid_offset ≤ 0.0412) and max_dr ≤ 0.205. It is highest (≈5.45) for wide jets (girth2 > 0.00873) that are not clean three-prong (tau32 > 0.407) and off-centre (centroid_offset > 0.0436), 3.0% of jets with 70% tops, ≈3.29 for the same with centroid_offset ≤ 0.0436 (6.7%, 63% tops), and ≈2.26 for wide jets with tau32 ≤ 0.407 and e2 ≤ 0.0634. It acts as a veto on the boson scores: up to −3.06 on Z and −2.72 on W (−1.85 and −1.65 in the 6.7% regime) while adding only up to +0.34 to t; regime_r2 0.64.

Regimes (a small tree on its quantities; R² 0.64):

- `` — 3.0% of jets, value 5.45 (2.00…8.88), formula right 72%
- `` — 6.7% of jets, value 3.29 (0.12…6.12), formula right 70%
- `` — 2.0% of jets, value 2.43 (0.25…5.50), formula right 46%
- `` — 3.4% of jets, value 2.26 (0.00…5.50), formula right 74%
- `` — 6.0% of jets, value 0.50 (0.00…1.62), formula right 62%
- `` — 6.7% of jets, value 0.32 (0.00…1.25), formula right 90%
- `` — 72.2% of jets, value 0.03 (0.00…0.00), formula right 63%

```
z = 2.90
if girth2 < 0.0091: z += -833 × (0.0091 − girth2)
if e2 < 0.043: z += 179 × (0.043 − e2)
if girth2 < 0.013: z += -441 × (0.013 − girth2)
if e2 > 0.028: z += -126 × (e2 − 0.028)
if mass_over_sum_pt > 0.062 and n_dr_0_0p05 < 5.99: z += 10.80 × (mass_over_sum_pt − 0.062) × (5.99 − n_dr_0_0p05)
if centroid_offset > 0.010: z += 49.30 × (centroid_offset − 0.010)
if mass_over_sum_pt > 0.075 and tau32 < 0.529: z += -133 × (mass_over_sum_pt − 0.075) × (0.529 − tau32)
if LHA > 0.316 and eccentricity > 0.953: z += 1090 × (LHA − 0.316) × (eccentricity − 0.953)
if lam1 > 0.015 and eccentricity > 0.951: z += -12100 × (lam1 − 0.015) × (eccentricity − 0.951)
if e2 < 0.043 and z_dr_0p1_0p2 > 0.0006: z += -328 × (0.043 − e2) × (z_dr_0p1_0p2 − 0.0006)
if mass > 71.90: z += -0.069 × (mass − 71.90)
if LHA > 0.308 and max_dr < 0.156: z += -1610 × (LHA − 0.308) × (0.156 − max_dr)
if mass_over_sum_pt > 0.068 and dr_7 < 0.042: z += -7530 × (mass_over_sum_pt − 0.068) × (0.042 − dr_7)
if mass > 62.00 and max_dr < 0.176: z += 2.06 × (mass − 62.00) × (0.176 − max_dr)
if mass_over_sum_pt > 0.091 and max_dr < 0.148: z += 9260 × (mass_over_sum_pt − 0.091) × (0.148 − max_dr)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 31.7% of jets, neuron 0.01, formula right for 60%. Mostly quarks (45%) with 33% gluons, 31.7% of jets: mass 8.9 GeV, width 0.0003, 85% of the pT inside 0.025, no wide-angle radiation. The narrow-jet switches girth2 < 0.00905 (-7.25) and girth2 < 0.0126 (-5.40) always pass and beat e2 < 0.0433 (+6.82), so the neuron is essentially off (on for 1.3%) and leaves the scores alone. The formula calls them q (51%) and is right only 59.7% of the time, the usual quark-gluon confusion.
- **Two-prong W/Z, no wide radiation** — 15.5% of jets, neuron 0.14, formula right for 70%. Mostly W (48%) with 34% Z, 15.5% of jets: mass 54.8 GeV, width 0.0062, 65% of the pT at 0.05-0.1 from the axis and little beyond 0.2. girth2 < 0.00905 (-2.37) and girth2 < 0.0126 (-2.82) always pass, and e2 > 0.0278 (99%, -1.18) cancels the bonuses e2 < 0.0433 (97%, +1.10) and mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (95%, +0.73). The neuron is on for only 9.9% (mean 0.141), so it hardly lowers the W and Z scores; the formula calls them W (60%) and is right 70.4% of the time.
- **Compact mid-mass W/Z jets** — 14.2% of jets, neuron 0.39, formula right for 62%. Mostly W (40%) with Z 30%, gluons 12%, tops 11%, 14.2% of jets: mass 47.4 GeV, width 0.0048, half of the pT at 0.025-0.05 from the axis. girth2 < 0.00905 (99%, -3.57) and girth2 < 0.0126 (-3.45) against e2 < 0.0433 (+3.12) leave it near zero; centroid_offset > 0.0103 (72%, +0.47) turns some on, e2 < 0.0435 and z_dr_0p1_0p2 > 0.000595 (69%, -0.50) turns some off. On for 22.4% (mean 0.387), it slightly lowers the W and Z scores; the formula calls them W (52%) and is right 61.5% of the time.
- **Narrow mid-mass mixed jets** — 13.0% of jets, neuron 0.15, formula right for 52%. A mixture of gluons (29%), W (27%), Z (19%) and quarks (18%), 13.0% of jets: mass 30.3 GeV, width 0.0024, 57% of the pT at 0.025-0.05. Same balance as the lightest jets: girth2 < 0.00905 (-5.57) and girth2 < 0.0126 (-4.51) against e2 < 0.0433 (+4.93), with centroid_offset > 0.0103 (74%, +0.49) deciding the few that turn on (10%, mean 0.149). The formula leans g (39%) over W (36%) and is right only 52.5% of the time: a poorly separated group.
- **Wider Z and top jets** — 10.5% of jets, neuron 1.46, formula right for 72%. Mostly Z (50%) with tops 30% and gluons 11%, 10.5% of jets: mass 59.7 GeV, width 0.0091, 63% of the pT at 0.05-0.1 and 27% beyond 0.1. These jets sit on the edge of the girth2 cuts (girth2 < 0.00905 passes 64%, girth2 < 0.0126 93%), and e2 > 0.0278 (-1.98) roughly cancels mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (+1.60); LHA > 0.308 and max_dr < 0.156 (67%, -0.92) and centroid_offset > 0.0103 (59%, +0.59) tip them either way. On for 48.8% (mean 1.46), it lowers the W and Z scores and slightly raises the top score; the formula calls them Z (60%) and is right 72.5% of the time, with Z-top confusion.
- **Wide three-prong top jets** — 6.5% of jets, neuron 3.30, formula right for 79%. Mostly tops (78%, plus 15% gluons), 6.5% of jets: mass 69.2 GeV, width 0.0170, soft, with 54% of the pT beyond 0.1 from the axis. The girth2 switches never pass, so the neuron is built from mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (+3.60) and centroid_offset > 0.0103 (+1.30) against e2 > 0.0278 (-3.90) and mass_over_sum_pt > 0.0749 and tau32 < 0.529 (63%, -1.07). It is on for 86.2% at 3.3, clearly lowering the W and Z scores and raising the top score; the formula calls them t (95%) and is right 79% of the time.
- **Heavy, clean three-prong tops** — 5.0% of jets, neuron 0.30, formula right for 92%. Mostly tops (92%), 5.0% of jets: mass 87.2 GeV, width 0.0282 (over four times average), soft, with 81% of the pT beyond 0.1. The large pair e2 > 0.0278 (-7.49) and mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (+6.37) nearly cancel, and mass_over_sum_pt > 0.0749 and tau32 < 0.529 (97%, -4.04) plus mass > 71.9 (81%, -1.16) switch most of them off, so the clearest three-prong tops are not what this neuron carries (on for 19%, mean 0.3). The formula still calls them t (99%) through other neurons and is right 92% of the time.
- **Wide, elongated top/gluon jets** — 2.3% of jets, neuron 3.95, formula right for 72%. Mostly tops (70%) with 21% gluons, 2.3% of jets: mass 82.8 GeV, width 0.0214, 81% of the pT beyond 0.1. LHA > 0.316 and eccentricity > 0.953 (+3.59) always passes and lam1 > 0.0152 and eccentricity > 0.951 (99%, -2.57) only partly offsets it, while e2 > 0.0278 (-5.02) and mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (+4.94) cancel. The neuron reaches its highest value (3.954, on for 89.6%), strongly lowering the W and Z scores and raising the top score; the formula calls them t (93%) and is right 71.7% of the time, the misses being gluons pulled towards t.
- **Widest, heaviest top/gluon jets** — 1.0% of jets, neuron 0.71, formula right for 68%. Mostly tops (56%) with 30% gluons and 13% quarks, 1.0% of jets: mass 97.0 GeV, width 0.0326, 88% of the pT beyond 0.1. Very large terms fight: lam1 > 0.0152 and eccentricity > 0.951 (-8.24) and e2 > 0.0278 (-7.59) against mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99 (+6.94) and LHA > 0.316 and eccentricity > 0.953 (+5.64), with mass > 71.9 (-1.79) and the tau32 < 0.529 test (54%, -1.82) tipping most off (on for 33.1%, mean 0.711). The formula calls them t (71%) and is right 68.1% of the time, with many gluons mistaken for tops.
- **Tops with a central softest particle** — 0.3% of jets, neuron 0.00, formula right for 87%. Mostly tops (86%), 0.3% of jets: mass 77.8 GeV, width 0.0217, an unusual profile with 29% of the pT inside 0.025 yet 48% beyond 0.1. mass_over_sum_pt > 0.0682 and dr_7 < 0.0419 always passes and subtracts 12.05, which alone keeps the neuron at zero for every jet here, so it adds nothing to any score. The formula calls them t (95%) and is right 87.4% of the time.

### neuron 4: low C2, mass/pT below top (moderate)

- **What it measures:** Pushed up for mass/pT < 0.111 and width > 0.00219, and down for mass/pT > 0.0885, for very thin jets (lam2 < 0.000389) and for C2 > 0.00878; it falls with C2 and D2 (rank correlations -0.515 and -0.421). It is on for almost all g, q, W and Z jets (Z highest at 4.54, then g 4.18, W 4.12, q 3.69) but zero for 35% of tops, which sit lowest (2.95).
- *computed — its value:* largest for Z (4.54), then g (4.18), then W (4.12), then q (3.69), then t (2.95); it separates t jets from the rest best (AUC 0.34: small for t)
- **How the class scores use it:** It raises the Z and t scores and lowers the g and q scores, and does not enter the W score. The q score (its strongest use, -16%) reads a high value as less typical of quarks than of Z, g or W jets; the t score adds it although tops sit lowest on it, so that use is not 'top-likeness' but a small correction alongside the t score's much larger inputs from neurons 13 and 10.
- *computed — used by:* raises the score of Z (+11%), t (+10%); lowers the score of g (-4%), q (-16%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** On for essentially every jet with C2 ≤ 0.0505: ≈3.76 for the bulk (tau21 > 0.0968 and sum_pt_top5 > 425 GeV, 54% of jets), ≈5.22 for two-prong-like jets (tau21 ≤ 0.0968, planar_flow ≤ 0.0923; 19%, mostly Z, 37%), ≈5.57 at lower pT (sum_pt_top5 ≤ 425 GeV; 10%, mostly gluons, 42%) and highest ≈8.38 for tau21 ≤ 0.0968 with planar_flow > 0.0923 (2.0%, 71% tops). For C2 > 0.0505 and max_dr > 0.186 it falls to ≈1.37 (mass_over_sum_pt ≤ 0.11) and ≈0.345 (mass/pT > 0.11; 8.5%, 88% tops). It pushes t (+0.47 in the bulk, up to +1.05) and Z (+0.294 in the bulk, up to +0.655) up and q down (−0.353 in the bulk, up to −0.786), with small minus on g; the regimes describe it moderately (regime_r2 0.515).

Regimes (a small tree on its quantities; R² 0.515):

- `` — 2.0% of jets, value 8.38 (5.00…12.75), formula right 76%
- `` — 10.0% of jets, value 5.57 (3.50…7.50), formula right 57%
- `` — 19.0% of jets, value 5.22 (3.00…7.00), formula right 76%
- `` — 53.6% of jets, value 3.76 (2.00…4.75), formula right 60%
- `` — 2.2% of jets, value 3.25 (0.25…6.20), formula right 55%
- `` — 4.6% of jets, value 1.37 (0.00…3.75), formula right 57%
- `` — 8.5% of jets, value 0.34 (0.00…1.25), formula right 88%

```
z = 2.33
if mass_over_sum_pt < 0.111: z += 41.60 × (0.111 − mass_over_sum_pt)
if width > 0.0022: z += 393 × (width − 0.0022)
if lam2 < 0.00039: z += -6470 × (0.00039 − lam2)
if mass_over_sum_pt > 0.088: z += -174 × (mass_over_sum_pt − 0.088)
if tau21 < 0.279: z += 15.50 × (0.279 − tau21)
if C2 > 0.0088: z += -57.40 × (C2 − 0.0088)
if tau21 < 0.285 and mass < 66.20: z += -0.609 × (0.285 − tau21) × (66.20 − mass)
if sum_pt_top5 < 486: z += 0.017 × (486 − sum_pt_top5)
if tau21 < 0.246 and pt_7 > 34.20: z += 0.670 × (0.246 − tau21) × (pt_7 − 34.20)
if C2 > 0.016 and pt_7 > 39.70: z += 11.00 × (C2 − 0.016) × (pt_7 − 39.70)
if max_dr > 0.109 and pt_7 > 39.20: z += -2.35 × (max_dr − 0.109) × (pt_7 − 39.20)
if tau21 < 0.243 and planar_flow > 0.033: z += 54.60 × (0.243 − tau21) × (planar_flow − 0.033)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 35.1% of jets, neuron 4.12, formula right for 59%. Mostly quarks (42%) with 35% gluons, 35.1% of jets: mass 9.8 GeV, width 0.0005, 78% of the pT inside 0.025. mass_over_sum_pt < 0.111 always passes (+4.07) and lam2 < 0.000389 (99%, -2.24) is the main loss; C2 > 0.00878 hits about half (-0.27). The neuron is always on at 4.121, which lowers the gluon and quark scores and raises the Z and top scores, so these light jets are pushed away from their own classes by this neuron; the formula splits them q 46% / g 42% and is right only 58.7% of the time.
- **Clean two-prong Z/W jets** — 19.2% of jets, neuron 5.64, formula right for 78%. Mostly Z (48%) with 36% W and 11% tops, 19.2% of jets: mass 62.4 GeV, width 0.0067, 57% of the pT at 0.05-0.1 from the axis. tau21 < 0.279 always passes (+3.01), together with width > 0.00219 (+1.78) and mass_over_sum_pt < 0.111 (+1.28); lam2 < 0.000389 (-2.18) and tau21 < 0.285 and mass < 66.2 (70%, -0.68) take some back. The neuron reaches its highest value (5.642), lowering the g and q scores and raising the Z and top scores; the formula calls them Z (52%) and is right 77.9% of the time.
- **Mid-mass jets, weak two-prong shape** — 15.3% of jets, neuron 3.32, formula right for 57%. A mixture of Z (29%), W (28%), gluons (19%), tops (13%) and quarks (11%), 15.3% of jets: mass 41.8 GeV, width 0.0051, somewhat soft, pT spread from the core out to 0.1. mass_over_sum_pt < 0.111 (+1.92) and width > 0.00219 (90%, +1.16) push up, but C2 > 0.00878 always passes (-2.26) and tau21 < 0.279 fires for only 39%, with lam2 < 0.000389 (65%, -1.02). The neuron is on for 91.6% at 3.321; the formula leans W (36%) and is right only 56.6% of the time in this mixed group.
- **Light two-prong W-like jets** — 13.6% of jets, neuron 3.71, formula right for 56%. Mostly W (39%) with Z 22%, gluons 15%, tops 13%, quarks 12%, 13.6% of jets: mass 39.6 GeV, width 0.0041, 44% of the pT at 0.025-0.05. tau21 < 0.279 always passes (+2.22) but so does tau21 < 0.285 and mass < 66.2 (-2.29), cancelling it for jets below 66 GeV; mass_over_sum_pt < 0.111 (+2.18) against lam2 < 0.000389 (95%, -1.98) and C2 > 0.00878 (-0.73). The neuron is always on at 3.71, lowering g and q and raising Z and top; the formula calls them W (57%) and is right only 56.4% of the time.
- **Two-prong-looking top jets** — 4.0% of jets, neuron 5.11, formula right for 73%. Mostly tops (71%, plus 18% gluons), 4.0% of jets: mass 66.0 GeV, width 0.0138, half of the pT beyond 0.1. width > 0.00219 (+4.57) and tau21 < 0.279 (99%, +2.77) beat mass_over_sum_pt > 0.0885 (99%, -3.76), lam2 < 0.000389 (69%, -1.23) and C2 > 0.00878 (-1.03), so the neuron stays high (5.108, on for 99.4%) although these are tops. It raises the top and Z scores; the formula calls them t (92%) and is right 73.1% of the time.
- **Heavy, wide three-prong tops** — 3.8% of jets, neuron 0.14, formula right for 93%. Mostly tops (93%), 3.7% of jets: mass 82.9 GeV, width 0.0254, 74% of the pT beyond 0.1. mass_over_sum_pt > 0.0885 (-11.50) and C2 > 0.00878 (-5.00) outweigh width > 0.00219 (+9.14) and sum_pt_top5 < 486 (76%, +1.40), and tau21 < 0.279 passes for only 17%, so the neuron is on for just 9.3% (mean 0.145). This is the top group the neuron was meant to separate; the formula calls them t (99%) and is right 92.8% of the time.
- **Moderately heavy, wide tops** — 3.3% of jets, neuron 0.69, formula right for 80%. Mostly tops (79%) with 15% gluons, 3.3% of jets: mass 69.2 GeV, width 0.0170, 48% of the pT beyond 0.1. mass_over_sum_pt > 0.0885 (-5.97) and C2 > 0.00878 (-4.67) against width > 0.00219 (+5.80) and sum_pt_top5 < 486 (62%, +1.19); with tau21 < 0.279 at only 22% the neuron is on for 33.2% (mean 0.689), a small effect. The formula calls them t (95%) and is right 80.3% of the time.
- **Heavy tops with low tau21** — 2.8% of jets, neuron 3.38, formula right for 74%. Mostly tops (71%) with 20% gluons, 2.8% of jets: mass 83.5 GeV, width 0.0216, 78% of the pT beyond 0.1. Unlike other heavy tops, tau21 < 0.279 always passes (+3.09) and C2 > 0.00878 costs only 1.00, so width > 0.00219 (+7.65) nearly matches mass_over_sum_pt > 0.0885 (-9.22); the neuron stays on (94%, mean 3.382), raising the top and Z scores and lowering g and q. The formula calls them t (93%) and is right 73.6% of the time, with gluons the main misses.
- **Widest, softest top jets** — 2.4% of jets, neuron 0.65, formula right for 80%. Mostly tops (77%) with 16% gluons, 2.4% of jets: mass 91.2 GeV, width 0.0346, soft (sum pT 506 GeV) with 88% of the pT beyond 0.1. The biggest terms of the neuron meet here: mass_over_sum_pt > 0.0885 (-16.02) against width > 0.00219 (+12.73), with C2 > 0.00878 (-4.00) and sum_pt_top5 < 486 (84%, +1.79); the neuron is on for 27% (mean 0.645). The formula calls them t (89%) and is right 80.3% of the time.
- **Heavy tops, many hard particles** — 0.6% of jets, neuron 2.58, formula right for 84%. Mostly tops (81%) with 15% gluons, 0.6% of jets: mass 93.1 GeV, width 0.0228, with pT shared evenly (8th particle 51.7 GeV). Only this group passes C2 > 0.0157 and pt_7 > 39.7 (+10.44), which together with width > 0.00219 (+8.10) outweighs mass_over_sum_pt > 0.0885 (-10.03), C2 > 0.00878 (-5.10) and max_dr > 0.109 and pt_7 > 39.2 (-4.33); the neuron is on for 68.9% (mean 2.577), raising the top and Z scores. The formula calls them t (93%) and is right 84.3% of the time.

### neuron 5: quark-likeness: pT in few particles (moderate)

- **What it measures:** Pushed up when the 8th-hardest particle carries little of the pT (z_7 < 0.0683, its strongest term) and for small e2 (< 0.0374); it rises with the summed pT of the 2-5 hardest particles and falls with e2 and z_7 (rank correlations about 0.56-0.59 in size). Quarks sit far highest (4.89); Z (2.10), W (2.04) and gluons (1.89) are similar, and tops lowest (1.09).
- *computed — its value:* largest for q (4.89), then Z (2.10), then W (2.04), then g (1.89), then t (1.09); it separates q jets from the rest best (AUC 0.77: large for q)
- **How the class scores use it:** Because quarks sit highest on it and gluons and tops well below them, it raises the q score and lowers the g and t scores: the g score reads it as evidence against a gluon (-15%), the t score as evidence against a top (-13%). It does not (or hardly) enter the W or Z scores.
- *computed — used by:* raises the score of q (+5%); lowers the score of g (-15%), t (-13%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** Highest (≈8.48) when the 8th-hardest particle is nearly empty (z_7 ≤ 0.0304), e2 is tiny (≤ 0.0104) and log_sum_pt ≤ 6.91 (8.0% of jets, 66% quarks); ≈5.08 for the same at log_sum_pt > 6.91 and ≈4.94 when e2 > 0.0104 with centroid_offset > 0.0092 (mostly Z, 36%). With z_7 > 0.0304 it is ≈3.76 for small e2 (≤ 0.0232) and sum_pt > 717 GeV (18%), ≈1.6 for sum_pt ≤ 717 GeV (16%, mostly gluons, 46%), and lowest (≈0.999 and ≈0.381) for e2 > 0.0232 (29% and 19% of jets). It lowers the t score (up to −2.12, −0.939 in the 18% regime) and the g score (up to −1.59, −0.705 in the 18% regime) and adds a little to q (up to +0.398) and Z (up to +0.133); regime_r2 0.714.

Regimes (a small tree on its quantities; R² 0.714):

- `` — 8.0% of jets, value 8.48 (6.00…11.12), formula right 70%
- `` — 2.0% of jets, value 5.08 (0.00…9.75), formula right 72%
- `` — 4.1% of jets, value 4.94 (2.62…7.25), formula right 66%
- `` — 18.3% of jets, value 3.76 (1.38…6.12), formula right 56%
- `` — 3.3% of jets, value 2.59 (0.12…5.12), formula right 76%
- `` — 16.3% of jets, value 1.60 (0.00…3.62), formula right 52%
- `` — 29.0% of jets, value 1.00 (0.38…2.12), formula right 71%
- `` — 18.9% of jets, value 0.38 (0.00…0.88), formula right 73%

```
z = 0.465
if z_7 < 0.068: z += 107 × (0.068 − z_7)
if e2 < 0.037: z += 106 × (0.037 − e2)
if z_7 < 0.071 and centroid_offset < 0.030: z += -3370 × (0.071 − z_7) × (0.030 − centroid_offset)
if width < 0.0027 and centroid_offset < 0.023: z += 75200 × (0.0027 − width) × (0.023 − centroid_offset)
if log_sum_pt > 6.59 and lam1 < 0.012: z += -982 × (log_sum_pt − 6.59) × (0.012 − lam1)
if dr_0 < 0.024: z += -169 × (0.024 − dr_0)
if log_sum_pt > 6.60 and dr_0 < 0.022: z += 757 × (log_sum_pt − 6.60) × (0.022 − dr_0)
if sum_pt_top2 < 546 and girth2_top3 < 0.0038: z += -1.79 × (546 − sum_pt_top2) × (0.0038 − girth2_top3)
if z_7 < 0.033: z += 185 × (0.033 − z_7)
if LHA < 0.221 and log_sum_pt < 6.79: z += -90.90 × (0.221 − LHA) × (6.79 − log_sum_pt)
if e2 < 0.034 and lam2 < 7.1e-05: z += 685000 × (0.034 − e2) × (7.1e-05 − lam2)
if log_sum_pt > 6.89: z += -67.90 × (log_sum_pt − 6.89)
if z_7 < 0.068 and sum_pt < 803: z += -0.278 × (0.068 − z_7) × (803 − sum_pt)
if girth2 < 6.4e-05: z += 31500 × (6.4e-05 − girth2)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Soft, wide top-rich jets** — 30.1% of jets, neuron 0.59, formula right for 70%. Mostly tops (42%) mixed with Z 20%, W 19% and gluons 15%, 30.1% of jets: mass 58.0 GeV, width 0.0130, soft (sum pT 576 GeV, leading particle 143 GeV) with pT evenly shared and 38% beyond 0.1. The quark signals mostly fail here: z_7 < 0.0683 passes for only 46% (+0.37) and e2 < 0.0374 for 30%, and the other tests hardly fire. The neuron is small (0.59, on for 88.9%) and only slightly lowers the gluon and top scores; the formula calls them t (45%) and is right 70.2% of the time.
- **Heavy W/Z/top jets, softish tail** — 18.0% of jets, neuron 1.22, formula right for 74%. An even W (29%) / Z (31%) / top (28%) mixture, 18.0% of jets: mass 64.4 GeV, width 0.0092, a fairly hard leading particle (253 GeV) and a soft 8th particle (31.7 GeV). z_7 < 0.0683 always passes (+2.65) but z_7 < 0.0707 and centroid_offset < 0.0301 (86%, -1.40) and z_7 < 0.0684 and sum_pt < 803 (74%, -0.53) take half of it back, while e2 < 0.0374 passes for only 47%. The neuron sits at 1.217 (on for 82.2%), mildly lowering the gluon and top scores; the formula leans W (36%) and is right 73.7% of the time.
- **Light, evenly shared gluon-rich jets** — 11.2% of jets, neuron 1.94, formula right for 50%. Mostly gluons (39%) with W 20%, Z 17%, quarks 15% and tops 10%, 11.2% of jets: mass 19.6 GeV, width 0.0021, a soft leading particle (160 GeV) and hard 8th particle (41.5 GeV), 55% of pT at 0.025-0.05. e2 < 0.0374 always passes (+2.52) but z_7 < 0.0683 only for 59%, and sum_pt_top2 < 546 and girth2_top3 < 0.00383 (96%, -1.19) and LHA < 0.221 and log_sum_pt < 6.79 (61%, -0.59) pull down. The neuron is on for 89.4% at 1.938, lowering the gluon and top scores; the formula calls them g (62%) but is right only 50% of the time.
- **Light, hard-led W/Z/quark mixture** — 9.3% of jets, neuron 4.06, formula right for 50%. A mixture of W (29%), Z (27%), quarks (19%) and gluons (18%), 9.3% of jets: mass 27.6 GeV, width 0.0022, a hard leading particle (289 GeV) and soft tail, 55% of pT at 0.025-0.05. Both quark signals always pass, z_7 < 0.0683 (+3.21) and e2 < 0.0374 (+2.69), while z_7 < 0.0707 and centroid_offset < 0.0301 (85%, -1.09) and log_sum_pt > 6.59 and lam1 < 0.0124 (75%, -0.94) take some back. The neuron is high (4.063, on for 97.5%), clearly lowering the gluon and top scores and raising the quark score; the formula leans W (38%) and is right only 50.4% of the time: these light bosons look quark-like to this neuron.
- **Narrow gluon jets, pT shared** — 9.0% of jets, neuron 1.38, formula right for 58%. Mostly gluons (57%) with 32% quarks, 9.0% of jets: mass 8.4 GeV, width 0.0003, soft leading particle (173 GeV) and hard 8th particle (42.0 GeV), 91% of pT inside 0.025. e2 < 0.0374 (+3.24) and width < 0.00267 and centroid_offset < 0.0234 (+3.00) push up, but LHA < 0.221 and log_sum_pt < 6.79 (-2.62), dr_0 < 0.0238 (97%, -2.22) and sum_pt_top2 < 546 and girth2_top3 < 0.00383 (-1.66) push down, and z_7 < 0.0683 passes for only 62%. The neuron is on for 65.9% at 1.379, a mild push away from g and t; the formula calls them g (79%) and is right 58.5% of the time, misreading many quarks.
- **Hard, narrow quark jets** — 8.1% of jets, neuron 5.52, formula right for 62%. Mostly quarks (57%) with 28% gluons, 8.1% of jets: mass 8.1 GeV, width 0.0002, hard (sum pT 877 GeV, leading particle 313 GeV) with 97% of pT inside 0.025. Four bonuses almost always pass, width < 0.00267 and centroid_offset < 0.0234 (+3.59), e2 < 0.0374 (+3.49), z_7 < 0.0683 (+3.17) and log_sum_pt > 6.6 and dr_0 < 0.0225 (+2.05), against dr_0 < 0.0238 (-2.83), z_7 < 0.0707 and centroid_offset < 0.0301 (-2.74) and log_sum_pt > 6.59 and lam1 < 0.0124 (-2.22). The value is high (5.523), lowering the gluon and top scores and raising the quark score; the formula calls them q (86%) and is right 62% of the time.
- **Hard-led, soft-tail W/Z jets** — 7.2% of jets, neuron 4.18, formula right for 70%. Mostly W (39%) with 36% Z and 15% quarks, 7.2% of jets: mass 56.6 GeV, width 0.0046, a very hard leading particle (435 GeV) and soft 8th particle (19.7 GeV). The soft tail triggers z_7 < 0.0683 (+4.94) and z_7 < 0.0333 (99%, +2.07), with e2 < 0.0374 (84%, +1.54); z_7 < 0.0707 and centroid_offset < 0.0301 (-3.27) and log_sum_pt > 6.59 and lam1 < 0.0124 (97%, -1.61) pull back. The neuron is on for 95.1% at 4.178, so these boson jets get a quark-like push (g and t scores lowered, q raised); the formula calls them W (50%) and is right 70.2% of the time.
- **Hard single-particle quark jets** — 5.4% of jets, neuron 9.37, formula right for 78%. Mostly quarks (78%), 5.4% of jets: mass 10.1 GeV, width 0.0002, hard (sum pT 956 GeV) with a 469 GeV leading particle, a very soft 8th particle (14.7 GeV) and 98% of pT inside 0.025. Every quark signal passes: z_7 < 0.0683 (+5.65), width < 0.00267 and centroid_offset < 0.0234 (+3.88), e2 < 0.0374 (+3.57), log_sum_pt > 6.6 and dr_0 < 0.0225 (+3.56) and z_7 < 0.0333 (+3.30), outweighing z_7 < 0.0707 and centroid_offset < 0.0301 (-5.10), log_sum_pt > 6.59 and lam1 < 0.0124 (-3.26) and dr_0 < 0.0238 (-3.24). The neuron peaks here (9.369, always on), strongly lowering the gluon and top scores and raising the quark score; the formula calls them q (99%) and is right 78.4% of the time.
- **Very hard gluon/quark jets** — 1.4% of jets, neuron 0.95, formula right for 64%. Mostly gluons (42%) with 37% quarks, 1.4% of jets: mass 24.2 GeV, width 0.0011, very hard (sum pT 1129 GeV, leading particle 496 GeV), 80% of pT inside 0.025. The quark signals pass (z_7 < 0.0683 +4.18, e2 < 0.0374 +3.10, log_sum_pt > 6.6 and dr_0 < 0.0225 +3.97), but the very-high-pT cut log_sum_pt > 6.89 (-9.41) and log_sum_pt > 6.59 and lam1 < 0.0124 (-4.89) switch most of them off (on for 34.2%, mean 0.954). The formula splits them g 48% / q 34% and is right 64% of the time.
- **Extremely hard gluon jets** — 0.4% of jets, neuron 0.00, formula right for 66%. Mostly gluons (58%) with 23% quarks, 0.4% of jets: mass 27.2 GeV, width 0.0011, extremely hard (sum pT 1383 GeV, leading particle 644 GeV). log_sum_pt > 6.89 always passes and subtracts 22.89, the largest single term, with log_sum_pt > 6.59 and lam1 < 0.0124 (-7.20), so despite the quark signals the neuron is off for every jet and adds nothing. The formula calls them g (66%) and is right 66% of the time.

### neuron 8: narrow core, nothing at wide angle (moderate)

- **What it measures:** Pushed up mainly for narrow jets with little pT at 0.2 ≤ ΔR < 0.4 (width < 0.00506 together with z_dr_0p2_0p4 < 0.0996, its strongest term), and down for narrow jets whose pT centroid is off the axis and for very high total pT (log of total pT > 6.71); it falls with width and centroid offset (rank correlations -0.414 and -0.404). Quarks (1.48) and gluons (1.24) sit highest; W (0.45), Z (0.26) and tops (0.21) low.
- *computed — its value:* largest for q (1.48), then g (1.24), then W (0.45), then Z (0.26), then t (0.21); it separates q jets from the rest best (AUC 0.72: large for q)
- **How the class scores use it:** The W score reads a single narrow core as evidence against a W, so it lowers the W score (-4%); it also raises the t score slightly (+3%), a small correction since tops sit lowest on it. Although quarks and gluons sit highest, it does not (or hardly) enter the g, q or Z scores.
- *computed — used by:* raises the score of t (+3%); lowers the score of W (-4%); does not (or hardly) enter the score of g, q, Z (share of each class score’s average input)
- **Boundaries:** Highest (≈3.21) for very narrow jets (width ≤ 0.00268) with mass > 9.36 GeV and a centred pT (centroid_offset ≤ 0.0147), 12.5% of jets, mostly quarks (45%); ≈1.02 for the very narrow, lighter ones (mass ≤ 9.36 GeV, 17%, mostly quarks, 52%) and ≈1.06 for 0.00268 < width ≤ 0.00451 with centroid_offset ≤ 0.0117 (mostly W, 51%). It is nearly off (≈0.099) for the 47% of jets with width > 0.00451 and centroid_offset > 0.00247. Its main effect is −0.803 on the W score in the top regime, with +0.602 on t and +0.201 on q there; regime_r2 0.626.

Regimes (a small tree on its quantities; R² 0.626):

- `` — 12.5% of jets, value 3.21 (1.00…5.50), formula right 60%
- `` — 3.1% of jets, value 1.06 (0.00…2.88), formula right 55%
- `` — 16.9% of jets, value 1.02 (0.00…2.25), formula right 64%
- `` — 2.1% of jets, value 0.67 (0.00…1.38), formula right 81%
- `` — 5.8% of jets, value 0.59 (0.00…1.88), formula right 46%
- `` — 7.8% of jets, value 0.24 (0.00…0.88), formula right 57%
- `` — 47.2% of jets, value 0.10 (0.00…0.25), formula right 73%
- `` — 4.7% of jets, value 0.06 (0.00…0.12), formula right 44%

```
z = 0.104
if width < 0.0051 and z_dr_0p2_0p4 < 0.100: z += 12900 × (0.0051 − width) × (0.100 − z_dr_0p2_0p4)
if LHA < 0.198 and lam2 < 0.00032: z += -169000 × (0.198 − LHA) × (0.00032 − lam2)
if width < 0.0054 and centroid_offset > 0.006: z += -76700 × (0.0054 − width) × (centroid_offset − 0.006)
if girth2 < 0.0062 and planar_flow < 0.453: z += 1240 × (0.0062 − girth2) × (0.453 − planar_flow)
if log_sum_pt > 6.71: z += -11.60 × (log_sum_pt − 6.71)
if mass < 20.20 and centroid_offset > 0.016: z += -15.20 × (20.20 − mass) × (centroid_offset − 0.016)
if e2 < 0.022 and sum_pt > 830: z += 0.561 × (0.022 − e2) × (sum_pt − 830)
if girth < 0.060 and lam1 > 0.0012: z += -32700 × (0.060 − girth) × (lam1 − 0.0012)
if pt_7 > 37.10: z += -0.043 × (pt_7 − 37.10)
if centroid_offset < 0.0037: z += 477 × (0.0037 − centroid_offset)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Broad heavy top/boson jets** — 47.7% of jets, neuron 0.14, formula right for 74%. Mostly tops (37%) with Z 27%, W 21% and gluons 10%, the largest group at 47.7% of jets: mass 63.0 GeV, width 0.0119, 34% of pT beyond 0.1. These jets are too broad for the neuron's main term, width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (passes 7%), and almost no other test adds much, so the value is tiny (0.136, on for 56%) and the scores are essentially untouched. The formula calls them t (40%) with Z and W close behind, and is right 73.7% of the time using other neurons.
- **Pencil-thin quark/gluon jets** — 11.4% of jets, neuron 1.61, formula right for 61%. Mostly quarks (51%) with 37% gluons, 11.4% of jets: mass 7.5 GeV, width 0.0001, 97% of pT inside 0.025. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 always passes (+6.32) but LHA < 0.198 and lam2 < 0.000321 always passes too (-4.99), leaving 1.608 (on for 93.9%) with small help from girth2 < 0.00621 and planar_flow < 0.453 (32%) and centroid_offset < 0.00366 (45%). It lowers the W score and raises q and t a little; the formula calls them q (61%) and is right 61.2% of the time.
- **Compact W jets** — 9.5% of jets, neuron 0.53, formula right for 60%. Mostly W (44%) with 27% Z and 12% gluons, 9.5% of jets: mass 47.3 GeV, width 0.0040, hard leading particle (298 GeV), 55% of pT at 0.025-0.05. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (93%, +1.26) and girth2 < 0.00621 and planar_flow < 0.453 (92%, +0.96) are cancelled by girth < 0.0603 and lam1 > 0.00119 (90%, -1.05), width < 0.00544 and centroid_offset > 0.00598 (83%, -0.88) and log_sum_pt > 6.71 (42%, -0.57). On for 41% (mean 0.532), it slightly lowers the W score; the formula calls them W (61%) and is right 59.5% of the time.
- **Very hard, pencil-thin quark jets** — 6.9% of jets, neuron 0.79, formula right for 76%. Mostly quarks (69%) with 19% gluons, 6.9% of jets: mass 8.1 GeV, width 0.0001, very hard (sum pT 1007 GeV, leading particle 447 GeV), 99% of pT inside 0.025. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+6.37) is cancelled by LHA < 0.198 and lam2 < 0.000321 (-6.27), and the high-pT penalty log_sum_pt > 6.71 (-2.32) is offset by e2 < 0.0218 and sum_pt > 830 (+1.83) and centroid_offset < 0.00366 (82%, +0.84). The result is modest (0.793, on for 76.5%); the formula calls them q (86%) and is right 75.5% of the time.
- **Narrow gluon/quark jets with some mass** — 6.2% of jets, neuron 4.24, formula right for 56%. Mostly gluons (45%) with 38% quarks, 6.1% of jets: mass 21.5 GeV, width 0.0010, 65% of pT inside 0.025 and 28% at 0.025-0.05. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+5.17) and girth2 < 0.00621 and planar_flow < 0.453 (77%, +1.33) face a weaker veto than for the thinnest jets (LHA < 0.198 and lam2 < 0.000321, 82%, -1.36), so the neuron reaches its highest value here (4.239, on for 98.9%). It lowers the W score by about 1.1 and raises q and t; the formula splits them g 50% / q 46% and is right only 56.5% of the time.
- **Narrow off-centre mid-mass mixture** — 5.9% of jets, neuron 0.69, formula right for 51%. A mixture of W (33%), gluons (23%), Z (21%) and quarks (15%), 5.9% of jets: mass 31.0 GeV, width 0.0026, 62% of pT at 0.025-0.05. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+3.01) is cancelled by width < 0.00544 and centroid_offset > 0.00598 (always, -3.28); girth2 < 0.00621 and planar_flow < 0.453 (81%, +1.13) against girth < 0.0603 and lam1 > 0.00119 (91%, -0.52) decides the rest, giving 0.69 (on for 49.7%). The formula calls them W (51%) and is right only 50.9% of the time.
- **Light narrow gluon jets, off-centre** — 5.2% of jets, neuron 1.21, formula right for 53%. Mostly gluons (45%) with quarks 29% and W 15%, 5.2% of jets: mass 9.1 GeV, width 0.0004, 89% of pT inside 0.025. width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+5.98) beats width < 0.00544 and centroid_offset > 0.00598 (-3.05) and LHA < 0.198 and lam2 < 0.000321 (-2.52), with girth2 < 0.00621 and planar_flow < 0.453 (63%, +0.97). On for 85.2% at 1.214, it lowers W and raises q and t; the formula calls them g (63%) and is right only 53.2% of the time.
- **Light off-centre mixed jets** — 4.8% of jets, neuron 0.10, formula right for 43%. A mixture of gluons (31%), W (27%), Z (22%) and quarks (15%), 4.8% of jets: mass 11.4 GeV, width 0.0011, 52% of pT at 0.025-0.05. width < 0.00544 and centroid_offset > 0.00598 (-6.23) outweighs width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+5.14), helped by mass < 20.2 and centroid_offset > 0.0163 (89%, -1.05), so the neuron is mostly off (on for 12.6%). The formula calls them g (51%) and is right only 42.7% of the time: the many light W/Z jets here are misread.
- **Very light, lopsided jets** — 2.4% of jets, neuron 0.00, formula right for 44%. A mixture led by gluons (34%) with Z 28%, W 15%, quarks 11%, tops 11%, 2.4% of jets: mass only 6.4 GeV with width 0.0016 and 78% of pT at 0.025-0.05. width < 0.00544 and centroid_offset > 0.00598 (-8.92) and mass < 20.2 and centroid_offset > 0.0163 (-4.47) outweigh width < 0.00506 and z_dr_0p2_0p4 < 0.0996 (+4.38) and girth2 < 0.00621 and planar_flow < 0.453 (+1.85), so the neuron is off and adds nothing. The formula splits them g 51% / Z 44% and is right only 44.3% of the time.
- **Near-massless yet broad jets** — 0.2% of jets, neuron 0.00, formula right for 53%. Mostly gluons (51%) with 33% tops, 0.2% of jets: mass only 6.6 GeV but width 0.0063, with 87% of pT at 0.05-0.1 from the axis. mass < 20.2 and centroid_offset > 0.0163 always passes (-12.29), with width < 0.00544 and centroid_offset > 0.00598 (53%, -2.37), so the neuron is off and adds nothing. The formula calls them g (70%) and is right only 53.3% of the time, missing the tops.

### neuron 11: compact, centred, flat (non-top) jet (moderate)

- **What it measures:** Pushed up for width < 0.0087 and a pT centroid close to the axis (centroid offset < 0.0498), and down for girth < 0.0883; it falls with centroid offset and planar flow (rank correlations -0.387 and -0.364). W jets sit highest (4.30), then Z (3.17), gluons (2.34) and quarks (2.26); tops are lowest (0.85) and at zero for 68% of them.
- *computed — its value:* largest for W (4.30), then Z (3.17), then g (2.34), then q (2.26), then t (0.85); it separates t jets from the rest best (AUC 0.16: small for t)
- **How the class scores use it:** It is the largest input of the W score, which it raises (+24%), since W jets sit highest on it. It does not enter the g, q, Z or t scores; in particular the Z score, though Z jets sit second on it, does not use it.
- *computed — used by:* raises the score of W (+24%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** Highest (≈4.26) for narrow (width ≤ 0.00853), not-too-light (mass > 13.2 GeV), centred (centroid_offset ≤ 0.0212) jets, 41% of jets, mostly W (36%); ≈2.31 when off-centre (11%), ≈2.23 for light jets (mass ≤ 13.2 GeV, 17%, mostly gluons, 41%) and ≈1.49 for the thinnest (width ≤ 0.0001, 11%, mostly quarks, 62%). It drops to ≈1.12 for 0.00853 < width ≤ 0.00986 and is nearly off above width 0.00986 (≈0.347 and ≈0.027; the off-centre part is 14% of jets, 78% tops). It feeds only the W score (+1.6 in its main regime, down to +0.01); regime_r2 0.701.

Regimes (a small tree on its quantities; R² 0.701):

- `` — 41.3% of jets, value 4.26 (2.62…6.00), formula right 67%
- `` — 11.1% of jets, value 2.31 (0.00…3.81), formula right 53%
- `` — 16.8% of jets, value 2.23 (1.44…3.06), formula right 54%
- `` — 10.7% of jets, value 1.49 (1.00…1.94), formula right 68%
- `` — 2.2% of jets, value 1.12 (0.00…2.62), formula right 61%
- `` — 3.5% of jets, value 0.35 (0.00…1.38), formula right 76%
- `` — 14.4% of jets, value 0.03 (0.00…0.00), formula right 80%

```
z = -1.15
if width < 0.0087: z += 1070 × (0.0087 − width)
if centroid_offset < 0.050: z += 101 × (0.050 − centroid_offset)
if girth < 0.088: z += -84.90 × (0.088 − girth)
if planar_flow < 0.271: z += 8.98 × (0.271 − planar_flow)
if girth > 0.077: z += -81.50 × (girth − 0.077)
if sum_pt_top5 < 688: z += 0.0054 × (688 − sum_pt_top5)
if centroid_offset < 0.044 and log_sum_pt < 6.84: z += -101 × (0.044 − centroid_offset) × (6.84 − log_sum_pt)
if width < 0.0037: z += -514 × (0.0037 − width)
if centroid_offset > 0.014: z += -73.10 × (centroid_offset − 0.014)
if girth > 0.077 and n_pt_above_50 < 7.47: z += 12.60 × (girth − 0.077) × (7.47 − n_pt_above_50)
if planar_flow < 0.217 and width > 0.0053: z += -1310 × (0.217 − planar_flow) × (width − 0.0053)
if planar_flow < 0.281 and max_dr > 0.114: z += -57.90 × (0.281 − planar_flow) × (max_dr − 0.114)
if LHA < 0.157: z += -22.80 × (0.157 − LHA)
if C2 < 0.035: z += -19.10 × (0.035 − C2)
if planar_flow < 0.254 and mass < 66.70: z += -0.128 × (0.254 − planar_flow) × (66.70 − mass)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin quark jets** — 21.4% of jets, neuron 1.97, formula right for 65%. Mostly quarks (55%) with 32% gluons, 21.4% of jets: mass 8.0 GeV, width 0.0002, harder than average, 97% of pT inside 0.025. width < 0.0087 (+9.15) and centroid_offset < 0.0498 (+4.57) always pass but so do the extreme-narrowness penalties girth < 0.0883 (-6.75), width < 0.00365 (-1.80) and LHA < 0.157 (-1.36), with C2 < 0.0355 (97%, -0.48). The neuron is on (99.9%) at a modest 1.971, which raises the W score a little even for these quark jets; the formula calls them q (67%) and is right 65.2% of the time.
- **Two-prong W/Z jets** — 18.7% of jets, neuron 4.51, formula right for 71%. Mostly W (49%) with 34% Z, 18.7% of jets: mass 55.0 GeV, width 0.0059, 57% of pT at 0.05-0.1. centroid_offset < 0.0498 (+3.83), width < 0.0087 (+2.96) and planar_flow < 0.271 (91%, +1.81) add up, while girth < 0.0883 costs only 1.60 for these not-so-thin jets and centroid_offset < 0.0437 and log_sum_pt < 6.84 (95%) 0.83. The neuron peaks here (4.507), raising the W score by about 1.7; the formula calls them W (62%) and is right 70.7% of the time.
- **Light narrow gluon-rich jets** — 15.6% of jets, neuron 3.21, formula right for 49%. Mostly gluons (39%) with quarks 25%, W 18%, Z 14%, 15.6% of jets: mass 15.8 GeV, width 0.0010, pT split between the core (53%) and 0.025-0.05 (41%). width < 0.0087 (+8.21) and centroid_offset < 0.0498 (+3.34) against girth < 0.0883 (-5.18) and width < 0.00365 (-1.35), with planar_flow < 0.271 (59%) and sum_pt_top5 < 688 (76%) adding and centroid_offset < 0.0437 and log_sum_pt < 6.84 (94%, -0.86) subtracting. At 3.212 it raises the W score by about 1.2; the formula calls them g (54%) and is right only 49.4% of the time.
- **Compact W jets** — 11.5% of jets, neuron 4.01, formula right for 60%. Mostly W (46%) with Z 27% and gluons 11%, 11.5% of jets: mass 45.5 GeV, width 0.0037, hard leading particle (297 GeV), 63% of pT at 0.025-0.05. width < 0.0087 (+5.34), centroid_offset < 0.0498 (+3.63) and planar_flow < 0.271 (90%, +1.69) beat girth < 0.0883 (-3.55) and planar_flow < 0.281 and max_dr > 0.114 (72%, -0.82). At 4.013 (on for 99.2%) it raises the W score by about 1.5; the formula calls them W (65%) and is right 59.5% of the time.
- **Wider Z and top jets** — 11.3% of jets, neuron 2.18, formula right for 74%. Mostly Z (51%) with 31% tops and 10% gluons, 11.3% of jets: mass 64.3 GeV, width 0.0096, 61% of pT at 0.05-0.1 and 27% beyond 0.1. centroid_offset < 0.0498 (+3.75) and planar_flow < 0.271 (90%, +1.81) with sum_pt_top5 < 688 (84%, +0.86) push up; width < 0.0087 passes for only 59%, and girth > 0.0766 (90%, -1.00), centroid_offset < 0.0437 and log_sum_pt < 6.84 (-1.05) and planar_flow < 0.217 and width > 0.0053 (88%, -0.81) pull down. On for 78.6% at 2.184, it gives a moderate push towards W; the formula calls them Z (61%) and is right 74.5% of the time.
- **Off-centre wide top jets** — 5.7% of jets, neuron 0.03, formula right for 72%. Mostly tops (69%) with 20% gluons, 5.7% of jets: mass 57.6 GeV, width 0.0135, soft, with pT at 0.05-0.1 (52%) and beyond (39%). centroid_offset > 0.0144 always passes (-2.59) and girth > 0.0766 (95%, -2.16) joins it, while width < 0.0087 passes for only 15%; sum_pt_top5 < 688 (+1.30) and girth > 0.0771 and n_pt_above_50 < 7.47 (91%, +0.94) cannot compensate, so the neuron is off (on for 4.3%) and adds nothing. The formula calls them t (86%) and is right 71.7% of the time.
- **Soft off-centre mid-mass mixture** — 5.4% of jets, neuron 1.99, formula right for 48%. A mixture of gluons (28%), Z (26%), W (20%), tops (15%) and quarks (11%), 5.4% of jets: mass 26.4 GeV, width 0.0037, a soft leading particle (178 GeV), pT at 0.025-0.1. width < 0.0087 (+5.35), centroid_offset < 0.0498 (90%, +1.42), planar_flow < 0.271 (69%, +1.14) and sum_pt_top5 < 688 (95%, +1.11) outweigh girth < 0.0883 (-2.77) and centroid_offset > 0.0144 (always, -1.62). On for 88.6% at 1.989, it raises the W score; the formula leans g (35.5%) over Z (31%) and W (26%) and is right only 47.6% of the time.
- **Heavy wide top jets** — 3.9% of jets, neuron 0.08, formula right for 90%. Mostly tops (90%), 3.9% of jets: mass 85.4 GeV, width 0.0241, 73% of pT beyond 0.1. girth > 0.0766 always passes (-5.34) and centroid_offset < 0.0437 and log_sum_pt < 6.84 (99%, -1.18) with it, outweighing centroid_offset < 0.0498 (+2.83), girth > 0.0771 and n_pt_above_50 < 7.47 (96%, +2.19) and sum_pt_top5 < 688 (+1.36); the neuron is on for only 12.5% and adds almost nothing. The formula calls them t (99%) and is right 90.2% of the time.
- **Heavy, flat top/gluon jets** — 3.4% of jets, neuron 0.00, formula right for 71%. Mostly tops (67%) with 22% gluons, 3.4% of jets: mass 87.4 GeV, width 0.0250, 81% of pT beyond 0.1. girth > 0.0766 (-5.69) and planar_flow < 0.217 and width > 0.0053 (always, -4.54), plus planar_flow < 0.281 and max_dr > 0.114 (-1.58) and centroid_offset > 0.0144 (80%, -1.39), outweigh girth > 0.0771 and n_pt_above_50 < 7.47 (+2.46), planar_flow < 0.271 (+2.09) and centroid_offset < 0.0498 (+2.08), so the neuron is off. The formula calls them t (88%) and is right 71.4% of the time, with gluons the main errors.
- **Very wide, soft top jets** — 3.0% of jets, neuron 0.02, formula right for 86%. Mostly tops (86%), 3.0% of jets: mass 75.1 GeV, width 0.0287, soft (sum pT 480 GeV), 89% of pT beyond 0.1. girth > 0.0766 (-6.83) and centroid_offset > 0.0144 (99%, -2.95) outweigh girth > 0.0771 and n_pt_above_50 < 7.47 (+4.13) and sum_pt_top5 < 688 (+1.72), and centroid_offset < 0.0498 passes for only 48%, so the neuron is off (on for 2.5%). The formula calls them t (97%) and is right 85.9% of the time.

### neuron 15: slightly wider than a W (moderate)

- **What it measures:** Switched off for narrow jets (width < 0.00695, its strongest term) and pushed up for lam1 < 0.00679 and small e2 (< 0.041), so it responds to jets just wider than the typical W but not broad; it follows lam1 and width (rank correlations 0.44 and 0.438). Z jets sit highest (1.41), then tops (0.96), with gluons (0.37), quarks (0.24) and W (0.21) low.
- *computed — its value:* largest for Z (1.41), then t (0.96), then g (0.37), then q (0.24), then W (0.21); it separates Z jets from the rest best (AUC 0.73: large for Z)
- **How the class scores use it:** W jets sit lowest on it, so the W score reads it as evidence against a W and it lowers the W score (-11%); it also lowers the Z score, but with a smaller share (-4%), even though Z jets sit highest. It does not (or hardly) enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-11%), Z (-4%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Highest (≈2.66) for jets somewhat wider than a W (width > 0.00632) but not broad (LHA ≤ 0.371) with lam1 > 0.00699: 15% of jets, 40% Z and 40% tops; ≈1.5 for the same with lam1 ≤ 0.00699 (6.1%, mostly Z, 46%). It is off for narrow jets with width ≤ 0.00632 and lam1 ≤ 0.00191 (≈0.004; 37% of jets, mostly quarks, 42%), low (≈0.319) for narrow jets with e2 > 0.0215 (19%, mostly W, 56%) and off again for broad jets with LHA > 0.387 (≈0.026). It is a W veto: −1.83 on W and −0.416 on Z in the top regime, with +0.166 on q; regime_r2 0.653.

Regimes (a small tree on its quantities; R² 0.653):

- `` — 15.3% of jets, value 2.66 (1.12…4.12), formula right 74%
- `` — 6.1% of jets, value 1.50 (0.00…3.00), formula right 65%
- `` — 5.3% of jets, value 0.97 (0.00…2.50), formula right 60%
- `` — 2.1% of jets, value 0.68 (0.00…1.62), formula right 79%
- `` — 19.2% of jets, value 0.32 (0.00…1.25), formula right 68%
- `` — 5.5% of jets, value 0.17 (0.00…0.75), formula right 48%
- `` — 9.6% of jets, value 0.03 (0.00…0.00), formula right 83%
- `` — 36.8% of jets, value 0.00 (0.00…0.00), formula right 58%

```
z = 0.951
if width < 0.0069: z += -2070 × (0.0069 − width)
if lam1 < 0.0068: z += 1610 × (0.0068 − lam1)
if lam1 < 0.0081: z += -790 × (0.0081 − lam1)
if e2 < 0.041: z += 104 × (0.041 − e2)
if e2 < 0.024: z += 199 × (0.024 − e2)
if z_dr_0p1_0p2 < 0.342: z += 3.61 × (0.342 − z_dr_0p1_0p2)
if tau21 < 0.240 and z_dr_0p2_0p4 < 0.210: z += 62.20 × (0.240 − tau21) × (0.210 − z_dr_0p2_0p4)
if LHA > 0.342: z += -54.80 × (LHA − 0.342)
if mass < 37.40: z += -0.046 × (37.40 − mass)
if girth2_top3 < 0.002: z += -665 × (0.002 − girth2_top3)
if n_dr_0_0p05 < 2.12: z += 0.454 × (2.12 − n_dr_0_0p05)
if LHA > 0.186 and sum_pt_top3 > 338: z += 0.031 × (LHA − 0.186) × (sum_pt_top3 − 338)
if width < 0.0079 and e2 > 0.024: z += -35800 × (0.0079 − width) × (e2 − 0.024)
if D2 < 0.711: z += -1.98 × (0.711 − D2)
if width < 0.0087 and planar_flow < 0.075: z += -3420 × (0.0087 − width) × (0.075 − planar_flow)
if width < 0.0061 and log_sum_pt > 6.90: z += -7700 × (0.0061 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 37.60 × (log_sum_pt − 6.90)
if z_dr_0p05_0p1 > 0.757: z += -6.11 × (z_dr_0p05_0p1 − 0.757)
if tau21 < 0.260 and z_dr_0p05_0p1 < 0.513: z += -9.02 × (0.260 − tau21) × (0.513 − z_dr_0p05_0p1)
if mass > 80.40 and z_dr_0p2_0p4 < 0.236: z += -0.572 × (mass − 80.40) × (0.236 − z_dr_0p2_0p4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Narrow light quark/gluon jets** — 26.4% of jets, neuron 0.00, formula right for 61%. Mostly quarks (47%) with 33% gluons, 26.4% of jets: mass 7.9 GeV, width 0.0002, 91% of pT inside 0.025. width < 0.00695 (-13.88) and lam1 < 0.00813 (-6.25) outweigh lam1 < 0.00679 (+10.59), e2 < 0.0244 (+3.89), e2 < 0.041 (+3.76) and z_dr_0p1_0p2 < 0.342 (+1.23), with mass < 37.4 (-1.37) and girth2_top3 < 0.00198 (-1.20), so the neuron is off and adds nothing. The formula calls them q (55%) and is right 60.9% of the time.
- **Z-width two-prong jets** — 12.3% of jets, neuron 2.48, formula right for 76%. Mostly Z (53%) with tops 25% and W 13%, 12.3% of jets: mass 63.2 GeV, width 0.0085, 64% of pT at 0.05-0.1 and 28% beyond. These jets are wide enough to escape width < 0.00695 (passes 26%) and lam1 < 0.00679 (23%); the value comes from tau21 < 0.24 and z_dr_0p2_0p4 < 0.21 (always, +2.14), n_dr_0_0p05 < 2.12 (90%, +0.78) and LHA > 0.186 and sum_pt_top3 > 338 (78%, +0.47), against D2 < 0.711 (97%, -0.67) and lam1 < 0.00813 (66%, -0.54). The neuron peaks here (2.485, on for 96.4%), lowering the W score by about 1.7 and Z by less; the formula calls them Z (61%) and is right 76.5% of the time.
- **Two-prong W jets** — 11.9% of jets, neuron 0.58, formula right for 71%. Mostly W (53%) with 28% Z and 10% tops, 11.9% of jets: mass 53.4 GeV, width 0.0057, 57% of pT at 0.05-0.1. width < 0.00695 (99%, -2.53) and lam1 < 0.00813 (-2.03) against lam1 < 0.00679 (+1.98), tau21 < 0.24 and z_dr_0p2_0p4 < 0.21 (84%, +1.45), z_dr_0p1_0p2 < 0.342 (98%, +0.79) and e2 < 0.041 (94%, +0.76), with width < 0.00788 and e2 > 0.0243 (93%, -0.73). On for 48.5% (mean 0.584), it trims the W score; the formula calls them W (67%) and is right 70.9% of the time.
- **Wide top-rich jets** — 11.6% of jets, neuron 1.75, formula right for 69%. Mostly tops (52%) with Z 20% and gluons 17%, 11.6% of jets: mass 61.7 GeV, width 0.0126, pT spread from 0.025 out beyond 0.1. The narrow-jet tests rarely pass (width < 0.00695 11%, lam1 < 0.00679 16%), so the value rests on the base plus z_dr_0p1_0p2 < 0.342 (79%, +0.69) and n_dr_0_0p05 < 2.12 (71%, +0.52), with LHA > 0.342 (35%, -0.48) cutting some. On for 87.2% at 1.751, it lowers the W score by about 1.2; the formula calls them t (61%) and is right 69.3% of the time.
- **Compact mid-mass W jets** — 9.8% of jets, neuron 0.39, formula right for 62%. Mostly W (48%) with 27% Z, 9.8% of jets: mass 45.9 GeV, width 0.0044, half of the pT at 0.025-0.05. width < 0.00695 (-5.19) and lam1 < 0.00813 (-3.05) against lam1 < 0.00679 (+4.07), e2 < 0.041 (+1.58), tau21 < 0.24 and z_dr_0p2_0p4 < 0.21 (72%, +0.98) and z_dr_0p1_0p2 < 0.342 (+0.93); on for 34.1% (mean 0.389). The formula calls them W (65%) and is right 61.8% of the time.
- **Light narrow gluon-rich jets** — 9.5% of jets, neuron 0.03, formula right for 50%. Mostly gluons (37%) with quarks 26%, W 17%, Z 14%, 9.5% of jets: mass 19.6 GeV, width 0.0014, pT at 0-0.05. width < 0.00695 (-11.54) and lam1 < 0.00813 (-5.43) outweigh lam1 < 0.00679 (+8.91), e2 < 0.041 (+2.99), e2 < 0.0244 (+2.42) and z_dr_0p1_0p2 < 0.342 (+1.20), with mass < 37.4 (96%, -0.83), so the neuron is off (on for 6.9%). The formula calls them g (51%) and is right only 49.7% of the time.
- **Heavy, wide top jets** — 9.0% of jets, neuron 0.01, formula right for 83%. Mostly tops (81%) with 13% gluons, 9.0% of jets: mass 83.4 GeV, width 0.0262, 87% of pT beyond 0.1. LHA > 0.342 always passes (-4.84) and none of the narrow-jet bonuses apply, so n_dr_0_0p05 < 2.12 (99%, +0.92) cannot keep it on; the neuron is off (on for 2%) and adds nothing. The formula calls them t (95%) and is right 82.9% of the time.
- **Narrow mid-mass mixture** — 8.0% of jets, neuron 0.24, formula right for 52%. A mixture of W (35%), Z (22%), gluons (21%) and quarks (13%), 8.0% of jets: mass 35.2 GeV, width 0.0030, 58% of pT at 0.025-0.05. width < 0.00695 (-8.23) and lam1 < 0.00813 (-4.20) against lam1 < 0.00679 (+6.40), e2 < 0.041 (+2.35), e2 < 0.0244 (81%, +1.30) and z_dr_0p1_0p2 < 0.342 (+1.09); on for 27.8% (mean 0.239). The formula calls them W (53%) and is right only 51.6% of the time.
- **Very hard narrow quark/gluon jets** — 1.2% of jets, neuron 0.03, formula right for 68%. Quarks (46%) and gluons (41%), 1.2% of jets: mass 13.3 GeV, width 0.0003, very hard (sum pT 1127 GeV, leading particle 503 GeV), 93% of pT inside 0.025. Besides the narrow-jet balance (width < 0.00695 -13.78, lam1 < 0.00679 +10.50, lam1 < 0.00813 -6.21), the high-pT pair width < 0.00614 and log_sum_pt > 6.9 (-5.65) and log_sum_pt > 6.9 (+4.76) always pass, so the neuron is off (on for 2.6%). The formula splits them g 50% / q 46% and is right 68.2% of the time.
- **Extremely hard narrow gluon jets** — 0.3% of jets, neuron 0.15, formula right for 68%. Mostly gluons (60%) with 24% quarks, 0.3% of jets: mass 19.4 GeV, width 0.0004, extremely hard (sum pT 1389 GeV, leading particle 646 GeV). The high-pT pair width < 0.00614 and log_sum_pt > 6.9 (-14.46) and log_sum_pt > 6.9 (+12.45) nearly cancel, and width < 0.00695 (-13.51) beats lam1 < 0.00679 (+10.29), so the neuron is off (on for 7.8%). The formula calls them g (67%) and is right 67.9% of the time.

### neuron 12: very wide jet, many hard particles (minor)

- **What it measures:** Almost always zero: it needs girth2 > 0.0188 (pushes up) and is pushed down strongly for e2 > 0.0622; it follows the number of particles above 10 GeV (rank correlation 0.843). Only tops reach it with any frequency (non-zero for 16% of them), so tops sit highest (0.24), then gluons (0.09) and quarks (0.03), with W and Z at 0.00.
- *computed — its value:* largest for t (0.24), then g (0.09), then q (0.03), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.58: large for t)
- **How the class scores use it:** It does not (or hardly) enter any of the five class scores, so it has essentially no effect on the classification.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** Off (≈0.001) for 93.5% of jets (girth2 ≤ 0.0246 and mass ≤ 97.2 GeV). It turns on only for very wide jets (girth2 > 0.0305: ≈2.18, 2.1% of jets, 75% tops), for massive jets with mass > 97.2 GeV (≈0.894, 2.0%, 78% tops) and weakly for 0.0246 < girth2 ≤ 0.0305 (≈0.328, 88% tops). It only lowers the t score (−1.09, −0.447, −0.164 in those regimes), trimming the widest, heaviest tops; the regimes describe it poorly (regime_r2 0.429).

Regimes (a small tree on its quantities; R² 0.429):

- `` — 2.1% of jets, value 2.18 (0.00…4.88), formula right 79%
- `` — 2.0% of jets, value 0.89 (0.00…2.38), formula right 84%
- `` — 2.4% of jets, value 0.33 (0.00…1.12), formula right 88%
- `` — 93.5% of jets, value 0.00 (0.00…0.00), formula right 64%

```
z = -0.910
if e2 > 0.062: z += -124 × (e2 − 0.062)
if girth2 > 0.019: z += 220 × (girth2 − 0.019)
if girth2 > 0.020 and pt_7 > 16.10: z += 6.92 × (girth2 − 0.020) × (pt_7 − 16.10)
if mass > 91.20: z += 0.078 × (mass − 91.20)
if girth2 > 0.019 and lam2 > 0.00049: z += 15300 × (girth2 − 0.019) × (lam2 − 0.00049)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Nearly all ordinary jets** — 91.1% of jets, neuron 0.00, formula right for 64%. 91.1% of all jets with a class mix close to the overall sample (W 22%, Z 22%, gluons 21%, quarks 21%, tops 14%) and average mass (35.8 GeV), width and pT. Neither of the two main tests fires (e2 > 0.0622 and girth2 > 0.0188 each pass for about 1-2%), so the neuron is off (on for 0.3%) and plays no part for these jets. The formula's 63.5% accuracy on them comes entirely from other neurons.
- **Wide tops just over threshold** — 2.5% of jets, neuron 0.18, formula right for 82%. Mostly tops (82%) with 13% gluons, 2.5% of jets: mass 75.2 GeV, width 0.0217, soft, 74% of pT beyond 0.1. e2 > 0.0622 (96%, -1.11) slightly outweighs girth2 > 0.0188 (89%, +0.66) and girth2 > 0.0197 and pt_7 > 16.1 (79%, +0.28), so the neuron is on for only 17.4% (mean 0.176) and barely lowers the top score. The formula calls them t (97%) and is right 82.2% of the time.
- **Wide, heavy top jets** — 2.2% of jets, neuron 0.31, formula right for 87%. Mostly tops (87%), 2.2% of jets: mass 82.4 GeV, width 0.0255, 77% of pT beyond 0.1. e2 > 0.0622 (-2.23) is roughly balanced by girth2 > 0.0188 (+1.47), girth2 > 0.0197 and pt_7 > 16.1 (+0.75), girth2 > 0.0186 and lam2 > 0.000489 (79%) and mass > 91.2 (30%); on for 29.8% (mean 0.312), it slightly lowers the top score. The formula calls them t (97%) and is right 87.3% of the time.
- **Wider, heavy top jets** — 1.7% of jets, neuron 0.43, formula right for 92%. Mostly tops (92%), 1.7% of jets: mass 84.7 GeV, width 0.0288, 84% of pT beyond 0.1. e2 > 0.0622 (-3.37) against girth2 > 0.0188 (+2.19), girth2 > 0.0197 and pt_7 > 16.1 (+1.18) and girth2 > 0.0186 and lam2 > 0.000489 (89%, +0.71); on for 44.2% (mean 0.426), a small cut to the top score. The formula calls them t (99%) and is right 92% of the time.
- **Very wide, soft top jets** — 0.9% of jets, neuron 1.11, formula right for 81%. Mostly tops (79%) with 16% gluons, 0.9% of jets: mass 87.2 GeV, width 0.0340, soft (sum pT 488 GeV), 85% of pT beyond 0.1. girth2 > 0.0188 (+3.35) and girth2 > 0.0197 and pt_7 > 16.1 (+1.82) with girth2 > 0.0186 and lam2 > 0.000489 (76%, +0.50) now beat e2 > 0.0622 (-4.11); the neuron is on for 75% (mean 1.107) and lowers the top score. The formula calls them t (91%) and is right 81.1% of the time.
- **Very heavy, hard top/gluon jets** — 0.5% of jets, neuron 1.21, formula right for 75%. Mostly tops (62%) with gluons 25% and quarks 10%, 0.5% of jets: mass 116.4 GeV, well above the top peak, and hard (sum pT 851 GeV). mass > 91.2 always passes (+1.96) and is what turns these on, with girth2 > 0.0188 (65%) and e2 > 0.0622 (64%) partly cancelling; on for 86.3% (mean 1.21), it lowers the top score. The formula calls them t (75%) and is right 74.6% of the time, many gluons and quarks being called tops.
- **Widest clean top jets** — 0.4% of jets, neuron 2.70, formula right for 94%. Mostly tops (94%), 0.4% of jets: mass 87.7 GeV, width 0.0356, 93% of pT beyond 0.1. girth2 > 0.0188 (+3.69), girth2 > 0.0186 and lam2 > 0.000489 (+2.62) and girth2 > 0.0197 and pt_7 > 16.1 (+1.94) outweigh e2 > 0.0622 (-5.03), so the neuron is on (96.7%) at 2.702 and lowers the top score by about 1.4. The formula still calls them t (100%) and is right 93.8% of the time.
- **Very heavy, wide top/gluon jets** — 0.4% of jets, neuron 3.14, formula right for 78%. Mostly tops (59%) with gluons 30% and quarks 10%, 0.4% of jets: mass 128.0 GeV, width 0.0305, 88% of pT beyond 0.1. mass > 91.2 (+2.87), girth2 > 0.0188 (+2.57) and girth2 > 0.0197 and pt_7 > 16.1 (+1.56) beat e2 > 0.0622 (-3.34), giving 3.138 (on for 99.6%), which lowers the top score by about 1.6. The formula calls them t (69%) and is right 78.4% of the time.
- **Extremely wide top/gluon jets** — 0.2% of jets, neuron 4.03, formula right for 57%. Mostly tops (51%) with gluons 36% and quarks 12%, 0.2% of jets: mass 97.0 GeV, width 0.0445, soft, 91% of pT beyond 0.1. girth2 > 0.0188 (+5.64), girth2 > 0.0197 and pt_7 > 16.1 (+3.18), girth2 > 0.0186 and lam2 > 0.000489 (81%, +0.96) and mass > 91.2 (55%, +0.91) beat e2 > 0.0622 (-5.77), giving 4.031 and a cut of about 2 to the top score. The formula calls them t (74%) and is right only 57.4% of the time, the gluons being the problem.
- **Fattest jets of all** — 0.0% of jets, neuron 10.42, formula right for 71%. A tiny handful of jets (well under 0.1%), mostly tops (75%) with 21% gluons: mass 100.3 GeV, width 0.0516, soft, 97% of pT beyond 0.1. girth2 > 0.0186 and lam2 > 0.000489 (+7.46), girth2 > 0.0188 (+7.21) and girth2 > 0.0197 and pt_7 > 16.1 (+4.48) overwhelm e2 > 0.0622 (-7.72), giving the neuron's highest value (10.422), which lowers the top score by about 5.2. The formula calls them t (96%) and is right 70.8% of the time.
