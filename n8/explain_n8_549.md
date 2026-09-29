# What each part of the 549-term formula does (8 particles)

*the simplest formula at the network's accuracy (from the 931-term tuned formula)*. Validation accuracy 65.74%. Words written by an AI agent from the numbers (60,000 training jets) and checked against them; lines marked *computed* come straight from the numbers. Plots: the page, tab "What each part does".

## Summary

The formula reads each jet on 16 scales, most of them variations on how wide the jet is, how its pT is shared among the hardest particles, and whether it looks like an elongated two-prong object. Tops are picked out mainly because they sit far below every other type on the compactness scale (neuron 13), which the t score subtracts, helped by the broad, massive, spread-out scale (neuron 10); there is no explicit 173 GeV mass or three-prong test among its main inputs. Quarks and gluons both sit high on the narrow, centred-jet scale (neuron 9), which feeds both of their scores; gluons are then told from quarks by the gluon-likeness scale (neuron 2, light jet with a hard 8th particle), which only the g score adds, and the quark-likeness scale (neuron 5, pT in few particles), which the g score subtracts. W and Z jets sit high on the two-prong scales (neurons 0 and 7) and at the bottom of the wide-angle-radiation and broad off-centre scales (neurons 3 and 6), which both boson scores subtract. W is told from Z by the compact, centred scale (neuron 11, highest for W and used only by the W score) against the scales that are largest for Z (neurons 7, 14 and 15), which weigh toward Z and against W; this W/Z split and the q/g split stay the weakest, with the Z score averaging 1.58 on true W jets and the q score 1.52 on true gluons.

## The 5 class scores

### score g: Gluon-like: light, hard-tailed, one-prong jets

High for light, one-prong jets whose pT is spread over many fairly hard particles: mean 2.22 for gluons and 1.37 for quarks, lower for tops (0.88) and near zero for W (0.13) and Z (-0.03).

Adds the gluon-likeness scale (neuron 2, +0.299), the narrow-centred scale (9, +0.267), the not-narrow/hard-8th-particle scale (1, +0.117) and the broad off-centre scale (6, +0.057); subtracts the quark-likeness scale (5, -0.132), the low-C2 scale (4, -0.069) and the two-prong massive scale (0, -0.059).

*computed:* largest for g (2.22), then q (1.37), then t (0.88), then W (0.13), then Z (-0.03); it separates g jets from the rest best (AUC 0.82: large for g)

### score q: Quark-like: narrow, centred jets

High for narrow, centred jets: mean 2.26 for quarks but also 1.52 for gluons (its weakest separation), and near zero for tops (0.07), W (0.03) and Z (-0.17).

Mostly the narrow-centred scale (neuron 9, +0.486), plus the broad off-centre (6, +0.081), quark-likeness (5, +0.041) and narrowness (8, +0.031) scales; subtracts the low-C2 scale (4, -0.255) and the broad, massive scale (10, -0.091).

*computed:* largest for q (2.26), then g (1.52), then t (0.07), then W (0.03), then Z (-0.17); it separates q jets from the rest best (AUC 0.85: large for q)

### score W: W-like: compact, centred two-prong jets

High for compact, centred two-prong jets without wide-angle radiation: mean 2.59 for W and 0.52 for Z, near zero for quarks (-0.04), negative for gluons (-0.59) and strongly negative for tops (-3.84).

Adds the compact-centred scale (neuron 11, +0.258), the two-prong massive (0, +0.081), compactness (13, +0.08) and Z-likeness (7, +0.064) scales; subtracts wide-angle radiation (3, -0.147), broad off-centre (6, -0.103), the two Z-leaning scales 14 (-0.085) and 15 (-0.081), narrowness (8, -0.063) and narrow-centred (9, -0.03).

*computed:* largest for W (2.59), then Z (0.52), then q (-0.04), then g (-0.59), then t (-3.84); it separates W jets from the rest best (AUC 0.89: large for W)

### score Z: Z-like: two-prong, mass-window jets, not broad

High for elongated two-prong jets in the boson mass/pT window that lack wide-angle radiation: mean 2.39 for Z, but also 1.58 for W (the main confusion), low for quarks (0.11) and negative for gluons (-0.38) and tops (-3.21).

Adds the Z-likeness (neuron 7, +0.191), low-C2 (4, +0.15), compactness (13, +0.087), heavier-wider (14, +0.059) and not-narrow (1, +0.033) scales; subtracts wide-angle radiation (3, -0.231, its largest input), broad off-centre (6, -0.172), narrow-centred (9, -0.042) and the width band above W (15, -0.026).

*computed:* largest for Z (2.39), then W (1.58), then q (0.11), then g (-0.38), then t (-3.21); it separates Z jets from the rest best (AUC 0.84: large for Z)

### score t: Top-like: essentially a wide-jet meter

High for wide, massive jets spread out in both directions: mean 2.83 for tops, far above Z (0.34), W (0.17) and gluons (0.05), and negative for quarks (-1.24); it tracks e2, girth and width (+0.884, +0.884, +0.88).

Mostly minus the compactness scale (neuron 13, -0.481), plus the low-C2 (4, +0.179), broad-massive (10, +0.144) and narrowness (8, +0.049) scales; subtracts the quark-likeness scale (5, -0.114).

*computed:* largest for t (2.83), then Z (0.34), then W (0.17), then g (0.05), then q (-1.24); it separates t jets from the rest best (AUC 0.90: large for t)

## How the scores are assembled (from the neurons' regimes)

Each of the five scores is a bias plus what the 16 neurons add in the regime a jet falls into, and most neurons feed only one or two scores: neuron 2 feeds only g, neuron 11 only W, neuron 12 only t, and neuron 7 only W and Z. The light-jet scores share one driver, the narrowness neuron 9 (+3.08 to q and +2.08 to g for the thinnest centred jets, 19% of jets), and g is split from q mainly by neuron 2 (+1.71 for light jets below 41.3 GeV with a hard 8th particle), so q and g stay close (true quarks score 1.37 on g, true gluons 1.52 on q). The boson scores are built from two-prong, compact, centred regimes (neuron 11: +1.90 to W; neuron 7: +1.99 to Z; neuron 0: +1.01 to W) and vetoed by broadness: wide-angle radiation (neuron 3) takes -3.86 from W and -4.34 from Z on 12% of jets, and neuron 6 takes up to -2.50 / -3.00. W is separated from Z by neurons 14 and 15, which subtract -1.59 and -1.83 from W where the jet is somewhat wider or heavier. The top score starts high (bias +1.34) and is pulled down for every compact jet by neuron 13 (down to -3.46), with neuron 10 adding +4.07 for heavy three-prong jets, so in effect it measures how wide and massive the jet is.

**What goes into score g:** Starting from a bias of -0.44, the g score is raised mainly by the narrowness neuron 9 for the thinnest, centred jets (girth2 ≤ 0.00148, centroid ≤ 0.00767; 19% of jets, +2.08, or +1.69 when the centroid is at 0.00767-0.0149, 7%) and by neuron 2 for light jets below 41.3 GeV whose 8th particle is not soft (z_7 > 0.0376: +1.71 when centred, 22% of jets; +1.18 when off-centre, 12%; +0.90 for 0.0162 < z_7 ≤ 0.0376, 12%). It is lowered by neuron 5 where pT sits in few particles (LHA > 0.103, e2 ≤ 0.0247, top-5 pT sum > 618 GeV: -0.74 on 18% of jets; LHA ≤ 0.103 at high pT: -1.23 on 6.4%), by the flat two-prong regime of neuron 0 (-0.46 on 25%) and slightly by the bulk of neuron 4 (-0.18 on 49%). Physically, a high g score means a light, narrow, centred jet whose pT is shared over many particles (a hard 8th particle), and not a quark-like jet with a few dominant particles. It separates only moderately from quarks: the mean g score is 2.22 for true gluons but 1.37 for true quarks, versus 0.125 for W and -0.028 for Z.

**What goes into score q:** The q score (bias 0.031) is built almost entirely from neuron 9: the thinnest, best-centred jets (girth2 ≤ 0.00148, centroid ≤ 0.00767; 19% of jets) get +3.08, those with centroid 0.00767-0.0149 get +2.50 (7%), and light, slightly wider centred jets below 38.7 GeV get +1.62 (5%). Against this, neuron 4 subtracts from nearly every jet (-0.55 on the 49% bulk, -0.84 on the 20% low-C2 regime, -0.68 and -0.83 elsewhere), neuron 10 subtracts -0.17 on 62% of jets and -1.36 for heavy three-prong jets (3.5%), while broad jets with τ32 > 0.234 from neuron 6 receive a surprising +0.66 (8.5% of jets, 66% t). Physically, a high q score means a very thin, well-centred one-prong jet; the quark-specific neuron 5 adds only +0.46 at most, so q and g are poorly separated (mean q score 2.26 for true quarks and 1.52 for true gluons, near 0.029 for W and -0.173 for Z).

**What goes into score W:** The W score (bias -0.125) is raised by neuron 11 for compact, centred jets (girth2 ≤ 0.00839, centroid ≤ 0.0217): +1.90 when τ21 ≤ 0.194 (25% of jets, mostly W) and +1.39 when τ21 > 0.194 (39%); by the flat intermediate-width regime of neuron 0 (+1.01 on 25%), by neuron 7 for flat compact jets above 54.5 GeV (+0.93 on 15%) and by neuron 13 for the thinnest high-pT jets (+0.60 on 23%). It is lowered strongly by wide-angle radiation (neuron 3, λ1 > 0.00978, LHA > 0.334, eccentricity > 0.759: -3.86 on 12%), by the broad regimes of neuron 6 (-1.65 on 8.5%, -2.50 on 4.3%), by the Z-like neurons 14 (λ1 > 0.00663, girth2 ≤ 0.0111: -1.59 on 13%; -0.55 on 23%) and 15 (-1.83 on 10%), and by neuron 8 for very thin jets (width ≤ 0.00209: -0.85 on 19%). So a high W score means a compact, centred, two-prong jet that is neither too thin (one-prong) nor too broad or heavy (Z or top). The mean W score is 2.59 for true W, 0.518 for Z and -3.84 for tops.

**What goes into score Z:** The Z score (bias -0.094) is raised mainly by neuron 7: +1.99 for flat, not-wide jets above 54.5 GeV (15% of jets, mostly Z), +1.00 for less flat jets with width ≤ 0.0106 (15%) and +0.84 below 54.5 GeV (14%); further by neuron 4 (+0.46 on the 49% bulk, +0.70 on the 20% low-C2 regime), neuron 13 (+0.47 on the thinnest 23%) and neuron 14 for larger λ1 (+0.79 on 13%, mostly Z). It is pushed down hardest by wide-angle radiation from neuron 3 (-4.34 on 12% of jets, -2.90 on 3.5%) and by the broad regimes of neuron 6 (-1.98 on 8.5%, -3.00 on 4.3%, and -0.15 even on the 73% compact bulk). Physically, a high Z score means a flat two-prong jet in the heavier mass range that is not broad; since W jets share most of these features, the mean Z score is 2.39 for true Z but still 1.58 for W, and the W/Z split relies on the W penalties of neurons 14 and 15. Tops sit far below at -3.21.

**What goes into score t:** The t score starts from a large bias of +1.34 and is mainly a wide-jet meter: neuron 13 subtracts from every compact jet, -3.46 for λ1 ≤ 0.00091 at high pT (23% of jets), -2.77 (7%), -2.62 (10%), -2.26 (8.7%), -1.91 (15%) and -1.56 for λ1 > 0.00581 with width ≤ 0.0111 (20%), while broad jets (width > 0.0162) lose only -0.10. Positive contributions come from neuron 4 (+0.73 on the 49% bulk, +1.12 on the 20% low-C2 regime), neuron 10 (+0.50 on 62%, and +4.07 for heavy three-prong jets with C2 > 0.0771, mass > 66.1 GeV, λ2 > 0.00353, 3.5% of jets, 95% t) and neuron 8 for very thin jets (+0.64 on 19%); neuron 5 lowers it by -0.99 where pT sits in few particles (18%). Physically, jets keep the positive bias only if they are wide, and the heaviest three-prong ones get an extra boost. The mean t score is 2.83 for true tops, -1.24 for quarks and near zero for the others (g 0.047, W 0.172, Z 0.336).

## The 16 neurons (most important first)

### neuron 1: Not-narrow jet, hard 8th particle (major)

- **What it measures:** Rises for jets that are not very narrow (width < 0.0087 and max ΔR < 0.25 push it down) with small e2_sq (< 0.0081 pushes it up), high total pT (log total pT > 6.4 and > 6.6) and a hard particle 7, the 8th hardest (pT_7 > 34 GeV). Lowest for q (0.49), with Z (1.04), g (1.03) and t (0.97) about equal and W (0.68) in between.
- *computed — its value:* largest for Z (1.04), then g (1.03), then t (0.97), then W (0.68), then q (0.49); it separates q jets from the rest best (AUC 0.37: small for q)
- **How the class scores use it:** The g score adds it (+12%) and the Z score a little (+3%); since quarks sit lowest on this scale, its main use in the g score is to pull gluons ahead of quarks. It does not enter the q, W or t scores.
- *computed — used by:* raises the score of g (+12%), Z (+3%); does not (or hardly) enter the score of q, W, t (share of each class score’s average input)
- **Boundaries:** Highest (3.65) when the 8th particle is hard (pT_7 > 45 GeV) in a high-pT jet (log total pT > 6.72), only 3.3% of jets and mostly g (46%); 2.54 for pT_7 > 45 GeV with lower total pT and mass > 58 GeV (3.2%, 39% t), about 2.0 for 38.2 < pT_7 ≤ 45 GeV with an off-centre pT centroid (> 0.0307), and low (0.39) in the bulk of jets with pT_7 ≤ 45 GeV, a centred centroid (≤ 0.0307) and mass ≤ 60.8 GeV (54% of jets). Its adds are mainly to g (+1.43 in the top regime, +0.15 in the bulk) and Z (+0.46 at most), with a small minus on W (-0.11), but its regimes cover few jets so its average effect is small. The regimes describe it only roughly (regime_r2 0.386).

Regimes (a small tree on its quantities; R² 0.386):

- `` — 3.3% of jets, value 3.65 (1.62…6.38), formula right 66%
- `` — 3.2% of jets, value 2.54 (1.12…4.12), formula right 76%
- `` — 3.1% of jets, value 1.98 (0.62…3.50), formula right 66%
- `` — 11.0% of jets, value 1.14 (0.00…2.38), formula right 67%
- `` — 15.4% of jets, value 1.03 (0.00…2.38), formula right 82%
- `` — 10.0% of jets, value 0.87 (0.00…2.12), formula right 61%
- `` — 53.9% of jets, value 0.39 (0.00…1.25), formula right 61%

```
z = 1.28
if width < 0.0087: z += -409 × (0.0087 − width)
if e2_sq < 0.0081: z += 381 × (0.0081 − e2_sq)
if max_dr < 0.250: z += -7.08 × (0.250 − max_dr)
if z_7 < 0.054: z += -85.70 × (0.054 − z_7)
if log_sum_pt > 6.60: z += 10.10 × (log_sum_pt − 6.60)
if log_sum_pt > 6.40: z += 3.43 × (log_sum_pt − 6.40)
if pt_7 > 34.00: z += 0.136 × (pt_7 − 34.00)
if z_7 < 0.054 and girth2_top2 < 0.014: z += 4820 × (0.054 − z_7) × (0.014 − girth2_top2)
if log_sum_pt > 6.40 and centroid_offset < 0.027: z += -132 × (log_sum_pt − 6.40) × (0.027 − centroid_offset)
if C2 < 0.044 and tau21 < 0.220: z += -350 × (0.044 − C2) × (0.220 − tau21)
if log_sum_pt > 6.40 and max_dr < 0.190: z += 22.10 × (log_sum_pt − 6.40) × (0.190 − max_dr)
if LHA > 0.280: z += 13.30 × (LHA − 0.280)
if pt_7 > 34.00 and mass < 80.40: z += -0.0015 × (pt_7 − 34.00) × (80.40 − mass)
if log_sum_pt > 6.60 and girth2_top3 < 0.0068: z += -749 × (log_sum_pt − 6.60) × (0.0068 − girth2_top3)
if tau21 < 0.210 and lam2 < 0.00021: z += 41000 × (0.210 − tau21) × (0.00021 − lam2)
if e2_sq < 0.0081 and planar_flow < 0.086: z += -4230 × (0.0081 − e2_sq) × (0.086 − planar_flow)
if lam1 < 0.0085 and z_7 > 0.037: z += -3520 × (0.0085 − lam1) × (z_7 − 0.037)
if z_7 < 0.044: z += -39.80 × (0.044 − z_7)
if e2 < 0.037 and eccentricity > 0.980: z += 5240 × (0.037 − e2) × (eccentricity − 0.980)
if mass_top5 < 6.10: z += -0.181 × (6.10 − mass_top5)
if e2 > 0.029 and tau32 < 0.640: z += -66.20 × (e2 − 0.029) × (0.640 − tau32)
if e2 > 0.034: z += -20.80 × (e2 − 0.034)
if LHA < 0.330 and planar_flow < 0.082: z += -127 × (0.330 − LHA) × (0.082 − planar_flow)
if max_dr < 0.048: z += -21.50 × (0.048 − max_dr)
if lam1 < 0.0083 and centroid_offset > 0.020: z += 14100 × (0.0083 − lam1) × (centroid_offset − 0.020)
if log_sum_pt > 6.60 and D2 < 1.20: z += 7.52 × (log_sum_pt − 6.60) × (1.20 − D2)
if pt_7 > 35.00 and max_dr < 0.080: z += 1.23 × (pt_7 − 35.00) × (0.080 − max_dr)
if pt_7 > 54.00: z += -0.120 × (pt_7 − 54.00)
if z_7 < 0.042 and z_dr_0p05_0p1 > 0.200: z += -89.80 × (0.042 − z_7) × (z_dr_0p05_0p1 − 0.200)
if pt_7 > 54.00 and mass_top3 < 52.00: z += 0.0024 × (pt_7 − 54.00) × (52.00 − mass_top3)
if n_pt_above_50 > 6.00 and girth2_top2 < 0.00075: z += 483 × (n_pt_above_50 − 6.00) × (0.00075 − girth2_top2)
if pt_7 > 35.00 and centroid_offset > 0.016: z += 1.49 × (pt_7 − 35.00) × (centroid_offset − 0.016)
if mass < 46.00 and tau21 < 0.200: z += 0.515 × (46.00 − mass) × (0.200 − tau21)
if log_sum_pt > 6.60 and n_pt_above_50 > 7.00: z += -3.36 × (log_sum_pt − 6.60) × (n_pt_above_50 − 7.00)
if e2 > 0.025 and pt_1 < 81.00: z += -0.427 × (e2 − 0.025) × (81.00 − pt_1)
if n_pt_above_50 > 6.10 and tau32 < 0.160: z += -19.30 × (n_pt_above_50 − 6.10) × (0.160 − tau32)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Light soft-leading gluon-rich jets** — 16.0% of jets, neuron 0.34, formula right for 53%. Mostly gluons (49%, q 24%, some W/Z) with low mass (12.4 GeV), narrow (91% of pT within 0.05), a softer leading particle (167 GeV) and a slightly harder 8th particle than average. Being narrow they pass width < 0.0087 (-3.144), max_dr < 0.25 and lam1 < 0.0085 and z_7 > 0.037, which mostly cancel e2_sq < 0.0081 (+2.862); pt_7 > 34 passes for 0.763 but is itself cancelled by pt_7 > 34 and mass < 80.4. The neuron is weak (0.339, on for 0.294) and nudges the g and Z scores up; the formula calls them g but is right only 0.529 of the time, many being quarks.
- **Wide top/Z jets, high LHA** — 15.4% of jets, neuron 0.98, formula right for 70%. Mostly tops (49%) with Z (24%) and some gluons, mass 60.4 GeV, about twice the average width and 38% of pT beyond 0.1 from the axis. They pass LHA > 0.28 (+0.804) and e2 > 0.034 far more than other groups, and fail width < 0.0087 for most jets so avoid its big minus; C2 < 0.044 and tau21 < 0.22 (-0.762) and max_dr < 0.25 pull back while pt_7 > 34 and tau21 < 0.21 and lam2 < 0.00021 add. The neuron is on for 0.758 (mean 0.978), adding to g and Z; the formula calls them t and is right 0.698 of the time, with Z the main alternative.
- **Medium-mass W-leaning mixed jets** — 14.1% of jets, neuron 0.58, formula right for 57%. A mixture led by W (38%) with Z (25%), g and t, at average mass (40.2 GeV) and slightly narrower than average, pT split between the core and 0.05-0.1. The narrow-jet pair width < 0.0087 (-1.612) and e2_sq < 0.0081 (+1.507) cancel; about half pass each of pt_7 > 34 (+), z_7 < 0.054 (-) and e2_sq < 0.0081 and planar_flow < 0.086 (-), and lam1 < 0.0083 and centroid_offset > 0.02 passes more often than elsewhere. The result is middling (0.583, on for 0.554); the formula calls them W but is right only 0.566 of the time.
- **Light narrow harder quark jets** — 10.1% of jets, neuron 0.49, formula right for 55%. Mostly light quarks (47%, g 24%, W 15%) with low mass (11.5 GeV), extremely narrow (97% of pT within 0.05), somewhat above-average pT (820 GeV) and a soft 8th particle. The narrowness terms cancel (width < 0.0087 vs e2_sq < 0.0081), z_7 < 0.054 (-1.572) and max_dr < 0.25 subtract, and log_sum_pt > 6.6 and log_sum_pt > 6.4 plus z_7 < 0.054 and girth2_top2 < 0.014 add back, leaving a modest value (0.493, on for 0.484). The formula calls them q and is right 0.548 of the time, with g and W as frequent true alternatives.
- **Wide heavy three-prong top jets** — 8.8% of jets, neuron 0.90, formula right for 86%. Mostly tops (85%) with the largest mass (82.2 GeV), four times the average width and 76% of pT beyond 0.1 from the axis; soft leading particle (143 GeV). They never pass width < 0.0087 or e2_sq < 0.0081, so the big narrow-jet terms drop out; LHA > 0.28 (+1.845) is offset by e2 > 0.029 and tau32 < 0.64 (-1.408) and e2 > 0.034, with pt_7 > 34 adding for half. The neuron is on for 0.732 (mean 0.897), adding to g and Z; the formula calls them t and is right 0.861 of the time.
- **Two-prong bosons with hard 8th particle** — 8.7% of jets, neuron 1.68, formula right for 74%. A W/Z mixture (W 44%, Z 37%) of 56.4 GeV with a hard 8th particle (49.6 GeV vs 35 average) and pT concentrated 0.05-0.1 from the axis. All pass pt_7 > 34 (+2.12), and most pass tau21 < 0.21 and lam2 < 0.00021 (+0.804) and log_sum_pt > 6.4, against C2 < 0.044 and tau21 < 0.22 (-1.282), width < 0.0087 and pt_7 > 34 and mass < 80.4. The neuron is high (1.675, on for 0.833), adding to g and Z and slightly lowering W; the formula calls them W and is right 0.735 of the time.
- **High-pT one-particle quark jets** — 8.5% of jets, neuron 0.61, formula right for 72%. Mostly light quarks (67%) with high pT (989 GeV), a leading particle of 471 GeV, a soft 8th particle and almost no mass or width. Large terms balance: width < 0.0087 and z_7 < 0.054 (about -6.4) against e2_sq < 0.0081, log_sum_pt > 6.6 and z_7 < 0.054 and girth2_top2 < 0.014, with log_sum_pt > 6.6 and girth2_top3 < 0.0068 and z_7 < 0.044 always passing here and pulling down. The value is moderate (0.612, on for 0.498); the formula calls them q and is right 0.718 of the time.
- **High-pT two-prong Z/W jets** — 7.4% of jets, neuron 1.22, formula right for 81%. Boosted bosons (Z 46%, W 39%) with mass 70.4 GeV, above-average pT (866 GeV) and pT mostly at 0.05-0.1, i.e. two clear prongs. The high-pT tests log_sum_pt > 6.6 and log_sum_pt > 6.4, and the two-prong tests tau21 < 0.21 and lam2 < 0.00021 and log_sum_pt > 6.6 and D2 < 1.2 (passing far more than elsewhere), outweigh z_7 < 0.054 (-1.726) and C2 < 0.044 and tau21 < 0.22 (-1.513). The neuron is on for 0.717 (1.217), adding to g and Z; the formula splits them Z/W and is right 0.808 of the time.
- **High-pT bosons with dominant core** — 6.3% of jets, neuron 0.79, formula right for 67%. A W/Z mixture (W 42%, Z 36%, q 14%) at 51.4 GeV with high pT (883 GeV), a very hard leading particle (412 GeV) and 89% of pT within 0.05 of the axis. z_7 < 0.054 (-2.409) and width < 0.0087 subtract, while log_sum_pt > 6.6, z_7 < 0.054 and girth2_top2 < 0.014, e2_sq < 0.0081 and e2 < 0.037 and eccentricity > 0.98 (much more common here) add. The neuron is on for 0.764 (0.791); the formula calls them W but is right only 0.666 of the time, with the quark admixture a source of error.
- **Many-hard-particle light gluon jets** — 4.6% of jets, neuron 1.98, formula right for 60%. Mostly gluons (54%, q 28%) with low mass (10.5 GeV), very narrow, and pT shared evenly among many particles (8th particle 56.5 GeV, leading only 216 GeV). pt_7 > 34 (+3.056) and pt_7 > 35 and max_dr < 0.08 (+1.357, far more common here) plus log_sum_pt > 6.6 win over pt_7 > 34 and mass < 80.4 (-2.379), with the narrow-jet pair cancelling. This gives the neuron's highest value (1.977, on for 0.894), raising the g and Z scores; the formula calls them g and is right only 0.597 of the time.

### neuron 2: Gluon-likeness: light, hard-tailed jet (major)

- **What it measures:** Rises for compact jets (λ1 < 0.0057) below 69 GeV in a total-pT band (log total pT > 6.3, total pT < 790 GeV) whose particle 7, the 8th hardest, is hard (pT_7 > 31 GeV pushes it up, pT_7 < 54 GeV down); it is also pushed down by LHA > 0.1 and mass < 36 GeV, and overall falls with mass (-0.666) and rises with τ21 (+0.551). Largest for g (3.21), then q (2.41), lower for W (1.58), Z (1.31) and t (1.28).
- *computed — its value:* largest for g (3.21), then q (2.41), then W (1.58), then Z (1.31), then t (1.28); it separates g jets from the rest best (AUC 0.77: large for g)
- **How the class scores use it:** Only the g score uses it (+30%, the g score's largest input), because gluons sit highest on this scale. It does not enter the q, W, Z or t scores.
- *computed — used by:* raises the score of g (+30%); does not (or hardly) enter the score of q, W, Z, t (share of each class score’s average input)
- **Boundaries:** High (3.98, on for essentially all) for light jets (mass ≤ 41.3 GeV) whose 8th particle carries a sizeable pT share (z_7 > 0.0376) and whose pT centroid is centred (≤ 0.0189): 22% of jets, mostly g (46%); still 2.74 when the centroid is off-centre (12%) and 2.09 for 0.0162 < z_7 ≤ 0.0376 (12%, mostly q at 53%). It falls for heavier jets (mass > 41.3 GeV: 0.87 with z_7 ≤ 0.0615 and max pair mass ≤ 24.6 GeV, 0.38 with max pair mass > 24.6 GeV) and for light jets with a very soft 8th particle (z_7 ≤ 0.0162: 0.40, 78% q). It feeds only the g score, adding +1.71, +1.18 and +0.90 in the three light regimes; the regimes describe it well (regime_r2 0.72).

Regimes (a small tree on its quantities; R² 0.719):

- `` — 21.9% of jets, value 3.98 (2.75…5.12), formula right 56%
- `` — 12.4% of jets, value 2.74 (1.38…4.12), formula right 47%
- `` — 11.8% of jets, value 2.09 (0.94…3.06), formula right 61%
- `` — 6.8% of jets, value 2.07 (1.12…3.12), formula right 65%
- `` — 11.0% of jets, value 1.22 (0.12…2.31), formula right 80%
- `` — 17.2% of jets, value 0.86 (0.00…1.69), formula right 70%
- `` — 3.5% of jets, value 0.40 (0.00…1.25), formula right 78%
- `` — 15.4% of jets, value 0.38 (0.00…1.06), formula right 81%

```
z = 2.32
if LHA > 0.100: z += -7.11 × (LHA − 0.100)
if pt_7 < 54.00: z += -0.048 × (54.00 − pt_7)
if mass < 36.00: z += -0.077 × (36.00 − mass)
if lam1 < 0.0057: z += 286 × (0.0057 − lam1)
if log_sum_pt > 6.30: z += 2.40 × (log_sum_pt − 6.30)
if sum_pt < 790: z += 0.0056 × (790 − sum_pt)
if mass < 69.00: z += 0.020 × (69.00 − mass)
if mass_over_sum_pt_sq < 0.012: z += 77.20 × (0.012 − mass_over_sum_pt_sq)
if pt_7 > 31.00: z += 0.089 × (pt_7 − 31.00)
if planar_flow < 0.680: z += -1.30 × (0.680 − planar_flow)
if mass < 37.00 and lam2 < 0.0011: z += 48.80 × (37.00 − mass) × (0.0011 − lam2)
if pt_7 > 30.00 and C2 < 0.051: z += -1.87 × (pt_7 − 30.00) × (0.051 − C2)
if max_dr < 0.160: z += -5.05 × (0.160 − max_dr)
if mass < 69.00 and z_7 < 0.068: z += -0.454 × (69.00 − mass) × (0.068 − z_7)
if log_sum_pt < 6.50: z += 1.81 × (6.50 − log_sum_pt)
if lam1 < 0.0059 and max_dr > 0.078: z += -3030 × (0.0059 − lam1) × (max_dr − 0.078)
if pt_7 > 30.00 and max_dr > 0.092: z += -0.446 × (pt_7 − 30.00) × (max_dr − 0.092)
if mass_over_sum_pt < 0.0098: z += -186 × (0.0098 − mass_over_sum_pt)
if z_7 > 0.044 and dr_7 < 0.066: z += -483 × (z_7 − 0.044) × (0.066 − dr_7)
if sum_pt_top5 > 740 and mean_eta2 < 0.0015: z += -3.84 × (sum_pt_top5 − 740) × (0.0015 − mean_eta2)
if pt_6 < 53.00 and lam2 > -0.00063: z += 5.77 × (53.00 − pt_6) × (lam2 − -0.00063)
if mass < 68.00 and centroid_offset > 0.010: z += -0.406 × (68.00 − mass) × (centroid_offset − 0.010)
if mass < 8.50: z += 0.145 × (8.50 − mass)
if lam1 < 0.004 and centroid_offset < 0.0062: z += -35100 × (0.004 − lam1) × (0.0062 − centroid_offset)
if log_sum_pt < 6.50 and eccentricity > 0.790: z += 8.11 × (6.50 − log_sum_pt) × (eccentricity − 0.790)
if z_7 < 0.017: z += -246 × (0.017 − z_7)
if pt_7 > 30.00 and centroid_offset > 0.013: z += -1.28 × (pt_7 − 30.00) × (centroid_offset − 0.013)
if sum_pt < 790 and dr_7 < 0.064: z += 0.041 × (790 − sum_pt) × (0.064 − dr_7)
if mass < 71.00 and max_pair_mass > 13.00: z += -0.0017 × (71.00 − mass) × (max_pair_mass − 13.00)
if log_sum_pt > 6.80: z += 4.36 × (log_sum_pt − 6.80)
if log_sum_pt > 6.90: z += -7.73 × (log_sum_pt − 6.90)
if pt_7 < 14.00 and n_pt_above_10 < 7.90: z += 0.125 × (14.00 − pt_7) × (7.90 − n_pt_above_10)
if girth2_top2 < 3.7e-05: z += 10700 × (3.7e-05 − girth2_top2)
if log_sum_pt > 6.90 and n_pt_above_50 > 5.00: z += -2.16 × (log_sum_pt − 6.90) × (n_pt_above_50 − 5.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Heavy boosted boson/top jets** — 19.1% of jets, neuron 0.47, formula right for 76%. Heavy jets (65.3 GeV) that are a Z/W/t mixture (Z 38%, W 33%, t 20%), of average total pT but with a harder leading particle (297 GeV) and pT spread over 0.05-0.1. Being heavy they almost never pass mass < 36 or mass < 37 and lam2 < 0.0011, and most fail lam1 < 0.0057, so the gluon-like pluses are missing; LHA > 0.1, pt_7 < 54 and planar_flow < 0.68 pull down against log_sum_pt > 6.3 and mass_over_sum_pt_sq < 0.012. The neuron is low (0.471, the lowest of the groups that are usually on), adding little to the g score; the formula splits them Z/W and is right 0.759 of the time.
- **Light narrow hard-core quark jets** — 14.2% of jets, neuron 2.32, formula right for 63%. Mostly light quarks (57%, g 20%) with almost no mass (7.9 GeV) or width, above-average pT (878 GeV) carried by a hard leading particle (358 GeV). The light-mass tests all pass and nearly cancel: mass < 36 (-2.159) and mass < 69 and z_7 < 0.068 against lam1 < 0.0057, mass < 37 and lam2 < 0.0011, mass < 69 and log_sum_pt > 6.3; pt_7 < 54 and, for most, mass_over_sum_pt < 0.0098 subtract. The neuron is on (2.322), pushing the g score up, but the formula still calls them q and is right 0.63 of the time.
- **Wide low-pT top jets** — 13.8% of jets, neuron 1.78, formula right for 72%. Mostly tops (66%, g 17%) with mass 60.3 GeV, about three times the average width, 55% of pT beyond 0.1 from the axis, and the lowest total pT (489 GeV). They pass the low-pT tests sum_pt < 790 (+1.674), log_sum_pt < 6.5 and pt_6 < 53 and lam2 > -0.00063, which outweigh LHA > 0.1 (-1.943) and pt_7 < 54, while failing lam1 < 0.0057 and mostly log_sum_pt > 6.3. The neuron is fairly high (1.785), adding to the g score, yet the formula calls them t and is right 0.715 of the time.
- **Hard-tailed medium-pT top/W/Z mix** — 13.0% of jets, neuron 1.64, formula right for 67%. A mixture of tops (35%), Z (25%), W (23%) and gluons at 56.0 GeV, wider than average, with lower total pT (601 GeV) but a hard 8th particle (41.8 GeV). LHA > 0.1 (-1.556) is outweighed by sum_pt < 790 and pt_7 > 31 (+0.964, always passing), with pt_7 > 30 and C2 < 0.051 and pt_7 > 30 and max_dr > 0.092 taking some back. The neuron is on (1.635), adding to g; the formula leans to t over W and Z and is right only 0.673 of the time.
- **Narrow high-pT W-leaning jets** — 9.1% of jets, neuron 1.29, formula right for 56%. A W/Z/quark mixture (W 37%, Z 26%, q 20%) at average mass (40.0 GeV), narrow (89% of pT within 0.05), with high pT concentrated in the leading particle (354 GeV). lam1 < 0.0057, log_sum_pt > 6.3 and mass_over_sum_pt_sq < 0.012 add, but pt_7 < 54, LHA > 0.1, planar_flow < 0.68 and especially lam1 < 0.0059 and max_dr > 0.078 (which passes far more often here than elsewhere) subtract. The neuron is middling (1.291); the formula calls them W but is right only 0.561 of the time.
- **Light gluon jets with hard tail** — 8.8% of jets, neuron 3.93, formula right for 54%. Mostly gluons (46%, q 31%) with little mass (8.5 GeV), narrow, with pT shared among many particles (leading 203 GeV, 8th 43.0 GeV). mass < 36 (-2.118) and pt_7 > 30 and C2 < 0.051 are beaten by lam1 < 0.0057, mass < 37 and lam2 < 0.0011, mass < 69, pt_7 > 31 and mass_over_sum_pt_sq < 0.012, all passing for every jet. The neuron is high (3.929), strongly raising the g score; the formula calls them g but is right only 0.536 of the time, because many are quarks.
- **Two-prong bosons with hard tail** — 8.5% of jets, neuron 1.55, formula right for 72%. A W/Z mixture (W 39%, Z 34%) of 58.0 GeV, pT at 0.05-0.1 from the axis, and a very hard 8th particle (50.9 GeV). pt_7 > 31 (+1.767) and log_sum_pt > 6.3 are balanced by LHA > 0.1, pt_7 > 30 and C2 < 0.051 and planar_flow < 0.68, while the light-mass pluses (mass < 36 etc.) are absent. The neuron is moderate (1.551), adding to g; the formula calls them W and is right 0.717 of the time.
- **Light low-pT gluon jets** — 8.2% of jets, neuron 4.00, formula right for 53%. Mostly gluons (51%, q 20%, about 10% each W/Z/t), light (14.6 GeV), of low total pT (537 GeV) and somewhat broader than the pencil-thin groups. Every jet passes sum_pt < 790, mass < 69 and mass_over_sum_pt_sq < 0.012, nearly all pass lam1 < 0.0057 and mass < 37 and lam2 < 0.0011, and most pass log_sum_pt < 6.5, together outweighing mass < 36 and pt_7 < 54. The neuron is high (3.999), raising the g score; the formula calls them g but is right only 0.528 of the time, the quark and boson admixture likely being misread as g.
- **Light gluons, very hard 8th particle** — 3.7% of jets, neuron 4.05, formula right for 60%. Mostly gluons (56%, q 28%), nearly massless (8.1 GeV) and narrow, with the hardest 8th particle of all groups (57.8 GeV) and a soft leading particle (204 GeV). pt_7 > 31 (+2.388) is cancelled by pt_7 > 30 and C2 < 0.051, but because pt_7 < 54 mostly fails its penalty is gone, so the light-mass pluses (lam1 < 0.0057, mass < 37 and lam2 < 0.0011, mass < 69) win over mass < 36. This gives the neuron's highest value (4.047), strongly raising g; the formula calls them g and is right 0.605 of the time.
- **One-particle high-pT quark jets** — 1.6% of jets, neuron 0.21, formula right for 80%. Mostly light quarks (80%), with high pT (1015 GeV) nearly all in the leading particle (566 GeV) and a very soft 8th particle (7.1 GeV). z_7 < 0.017 (-2.458), which nearly never passes elsewhere, together with pt_7 < 54, mass < 36, mass < 69 and z_7 < 0.068 and sum_pt_top5 > 740 and mean_eta2 < 0.0015 outweigh lam1 < 0.0057, pt_7 < 14 and n_pt_above_10 < 7.9 and log_sum_pt > 6.3. The neuron is mostly off (on for 0.264), leaving the g score alone; the formula calls them q and is right 0.799 of the time.

### neuron 3: Wide-angle radiation (0.2-0.4) (major)

- **What it measures:** Rises with the pT carried at 0.2 ≤ ΔR < 0.4 from the axis (rank correlation +0.695) and with jet width: compact jets are cut off (girth2 < 0.0088 and < 0.013 push it down) while e2 < 0.043 and LHA > 0.32 push it up. Largest for t (4.72), then g (1.13) and q (0.54), and nearly zero for Z (0.17) and W (0.01).
- *computed — its value:* largest for t (4.72), then g (1.13), then q (0.54), then Z (0.17), then W (0.01); it separates t jets from the rest best (AUC 0.80: large for t)
- **How the class scores use it:** The Z score (-23%) and W score (-15%) subtract it: W and Z jets sit at the bottom of this scale, so radiation at wide angle argues against a boson. It does not enter the g, q or t scores, even though tops sit highest on it.
- *computed — used by:* lowers the score of W (-15%), Z (-23%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Off (0.078, on for 6%) for the 79% of jets that are compact (λ1 ≤ 0.00978, centroid offset ≤ 0.0443, width ≤ 0.00857); it switches on for broad jets with λ1 > 0.00978 and LHA > 0.334, reaching 7.72 when eccentricity > 0.759 (12% of jets, 74% t) and 5.16 when eccentricity ≤ 0.759 (3.5%, 94% t). In those broad regimes it acts as a strong veto on the boson scores, adding -3.86 to W and -4.34 to Z (and -2.58 / -2.90 in the rounder regime) against only +0.48 to t; it does not enter g or q. The regimes describe it moderately well (regime_r2 0.62).

Regimes (a small tree on its quantities; R² 0.62):

- `` — 11.7% of jets, value 7.72 (1.25…14.00), formula right 76%
- `` — 3.5% of jets, value 5.16 (0.00…12.25), formula right 94%
- `` — 2.0% of jets, value 3.75 (0.00…10.12), formula right 74%
- `` — 2.0% of jets, value 3.01 (0.00…8.00), formula right 57%
- `` — 2.3% of jets, value 1.57 (0.00…4.38), formula right 62%
- `` — 78.5% of jets, value 0.08 (0.00…0.00), formula right 63%

```
z = 1.38
if girth2 < 0.0088: z += -760 × (0.0088 − girth2)
if girth2 < 0.013: z += -372 × (0.013 − girth2)
if e2 < 0.043: z += 130 × (0.043 − e2)
if LHA > 0.320: z += 87.60 × (LHA − 0.320)
if e2 > 0.028: z += -89.10 × (e2 − 0.028)
if mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.90: z += 10.80 × (mass_over_sum_pt − 0.065) × (4.90 − n_dr_0_0p05)
if centroid_offset > 0.013: z += 82.80 × (centroid_offset − 0.013)
if mass > 37.00 and lam2 < 0.0013: z += 55.10 × (mass − 37.00) × (0.0013 − lam2)
if max_dr > 0.150: z += 24.50 × (max_dr − 0.150)
if width > 0.019 and pt_dispersion < 0.490: z += 6930 × (width − 0.019) × (0.490 − pt_dispersion)
if mass_over_sum_pt > 0.070 and tau32 < 0.520: z += -178 × (mass_over_sum_pt − 0.070) × (0.520 − tau32)
if lam1 > 0.016: z += 556 × (lam1 − 0.016)
if LHA > 0.310 and eccentricity > 0.960: z += 2050 × (LHA − 0.310) × (eccentricity − 0.960)
if e2 > 0.051: z += 117 × (e2 − 0.051)
if lam1 > 0.0086 and pt_7 < 38.00: z += 41.60 × (lam1 − 0.0086) × (38.00 − pt_7)
if LHA > 0.320 and planar_flow > 0.012: z += -77.90 × (LHA − 0.320) × (planar_flow − 0.012)
if lam1 > 0.012 and pt_6 < 38.00: z += 78.80 × (lam1 − 0.012) × (38.00 − pt_6)
if lam1 > 0.0028 and z_7 < 0.077: z += -4000 × (lam1 − 0.0028) × (0.077 − z_7)
if e2 > 0.027 and n_pt_above_50 > 3.00: z += -13.00 × (e2 − 0.027) × (n_pt_above_50 − 3.00)
if max_dr > 0.140 and phi_0 < -0.041: z += 704 × (max_dr − 0.140) × (-0.041 − phi_0)
if C2 > 0.094: z += -247 × (C2 − 0.094)
if LHA > 0.310 and max_dr < 0.150: z += -3470 × (LHA − 0.310) × (0.150 − max_dr)
if mass > 70.00: z += -0.085 × (mass − 70.00)
if mass_top5 > 53.00: z += -0.094 × (mass_top5 − 53.00)
if centroid_offset > 0.012 and mean_phi < -0.0092: z += -2390 × (centroid_offset − 0.012) × (-0.0092 − mean_phi)
if mass_over_sum_pt > 0.069 and dr_7 < 0.042: z += -10900 × (mass_over_sum_pt − 0.069) × (0.042 − dr_7)
if mean_eta < -0.016 and z_4 < 0.130: z += 1710 × (-0.016 − mean_eta) × (0.130 − z_4)
if mean_eta < -0.015 and pt_4 < 69.00: z += -3.14 × (-0.015 − mean_eta) × (69.00 − pt_4)
if max_dr > 0.160 and dr_1 < 0.047: z += -640 × (max_dr − 0.160) × (0.047 − dr_1)
if mass_top5 > 54.00 and eta_7 < 0.035: z += -0.763 × (mass_top5 − 54.00) × (0.035 − eta_7)
if centroid_offset > 0.015 and phi_7 > -0.030: z += 246 × (centroid_offset − 0.015) × (phi_7 − -0.030)
if lam1 > 0.018 and eccentricity > 0.960: z += 12900 × (lam1 − 0.018) × (eccentricity − 0.960)
if mass > 70.00 and mean_phi > 0.00098: z += -5.08 × (mass − 70.00) × (mean_phi − 0.00098)
if LHA > 0.420 and pt_7 > 38.00: z += -30.10 × (LHA − 0.420) × (pt_7 − 38.00)
if max_dr > 0.150 and dr_3 < 0.046: z += -507 × (max_dr − 0.150) × (0.046 − dr_3)
if mass_over_sum_pt > 0.110 and dr_7 < 0.049: z += 21900 × (mass_over_sum_pt − 0.110) × (0.049 − dr_7)
if mass > 70.00 and max_dr < 0.180: z += 3.89 × (mass − 70.00) × (0.180 − max_dr)
if LHA > 0.310 and pt_5 < 30.00: z += 16.00 × (LHA − 0.310) × (30.00 − pt_5)
if mass_over_sum_pt > 0.091 and max_dr < 0.150: z += 8590 × (mass_over_sum_pt − 0.091) × (0.150 − max_dr)
if mass > 38.00 and dr_6 < 0.046: z += -1.28 × (mass − 38.00) × (0.046 − dr_6)
if mass > 36.00 and pt_7 < 20.00: z += -0.0044 × (mass − 36.00) × (20.00 − pt_7)
if e2 < 0.038 and phi_0 > 0.080: z += -50100 × (0.038 − e2) × (phi_0 − 0.080)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 38.4% of jets, neuron 0.01, formula right for 59%. The largest group (38%), mostly light quarks (41%) with gluons (34%), almost massless (11.5 GeV) with 97% of pT within 0.05 of the axis. girth2 < 0.0088 (-6.245) and girth2 < 0.013 (-4.619) always pass and beat e2 < 0.043 (+4.7), so the neuron is off (on for 0.008) and adds nothing. The formula splits them between q and g and is right only 0.587 of the time.
- **Compact medium-mass W-leaning jets** — 22.7% of jets, neuron 0.27, formula right for 60%. W-rich (40%) mixture with Z (27%) and gluons, mass 44.4 GeV, narrower than average with 60% of pT within 0.05. Both girth2 < 0.0088 and girth2 < 0.013 pass (about -6.6 together) against e2 < 0.043 (+2.407), with only partial help from max_dr > 0.15, mass > 37 and lam2 < 0.0013 and centroid_offset > 0.013. The neuron is usually zero (on for 0.145, mean 0.266), only slightly lowering the W and Z scores; the formula calls them W and is right 0.603 of the time.
- **Two-prong Z/W jets** — 22.4% of jets, neuron 0.63, formula right for 73%. Mostly Z (45%) with W (29%) and t (16%), mass 58.3 GeV and 65% of pT at 0.05-0.1 from the axis, i.e. two prongs with little wide-angle radiation. girth2 < 0.013, girth2 < 0.0088 and e2 > 0.028 (which passes far more often than elsewhere) subtract, and mass > 37 and lam2 < 0.0013 and mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9 do not fully compensate. The neuron is on for only 0.211 of jets (mean 0.626), so it lowers the W and Z scores just a little; the formula calls them Z and is right 0.726 of the time.
- **Moderately wide top jets** — 5.6% of jets, neuron 5.83, formula right for 77%. Mostly tops (76%, g 17%), 64.2 GeV, about 2.4 times the average width with 43% of pT beyond 0.1 from the axis. They escape girth2 < 0.0088 and mostly girth2 < 0.013, and gain from LHA > 0.32 (+3.205), centroid_offset > 0.013, max_dr > 0.15 and mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9, against e2 > 0.028 and mass_over_sum_pt > 0.07 and tau32 < 0.52. The neuron is high (5.829), lowering the W and Z scores strongly and raising t; the formula calls them t and is right 0.769 of the time, the gluon admixture likely accounting for most errors.
- **Wide heavy top jets** — 3.9% of jets, neuron 6.22, formula right for 92%. Almost all tops (92%), 82.6 GeV, nearly four times the average width with 77% of pT beyond 0.1. Large pluses LHA > 0.32 (+8.273), mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9, e2 > 0.051 and width > 0.019 and pt_dispersion < 0.49 are partly cancelled by e2 > 0.028, mass_over_sum_pt > 0.07 and tau32 < 0.52 and LHA > 0.32 and planar_flow > 0.012. The neuron is high (6.218), lowering W and Z and raising t; the formula calls them t and is right 0.921 of the time.
- **Wide elongated top/gluon jets** — 3.3% of jets, neuron 8.22, formula right for 73%. Mostly tops (71%) with gluons (19%), 78.4 GeV, with 74% of pT beyond 0.1 and an elongated shape. Besides LHA > 0.32 (+6.423) they nearly all pass LHA > 0.31 and eccentricity > 0.96 (+5.27), which is rare elsewhere, plus mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9 and mass > 37 and lam2 < 0.0013, with e2 > 0.028 the main minus. This gives the neuron's highest value (8.218), pushing W and Z down and t up; the formula calls them t and is right 0.73 of the time, the gluon admixture likely being misread.
- **Very wide soft-leading top jets** — 1.8% of jets, neuron 7.13, formula right for 91%. Almost all tops (91%), 89.7 GeV, five times the average width with 89% of pT beyond 0.1 and a soft leading particle (118 GeV). Huge pluses LHA > 0.32 (+11.708), width > 0.019 and pt_dispersion < 0.49 (+9.816), lam1 > 0.016 and mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9 beat the minuses mass_over_sum_pt > 0.07 and tau32 < 0.52, LHA > 0.32 and planar_flow > 0.012 and e2 > 0.028. The neuron is high (7.129), lowering W and Z and raising t; the formula calls them t and is right 0.91 of the time.
- **Very wide heavy top/gluon mix** — 1.0% of jets, neuron 7.94, formula right for 67%. Tops (57%) mixed with many gluons (31%), the heaviest group (96.5 GeV), five times the average width with 89% of pT beyond 0.1. All the width tests fire: LHA > 0.32 (+11.363), LHA > 0.31 and eccentricity > 0.96, width > 0.019 and pt_dispersion < 0.49, lam1 > 0.016 and lam1 > 0.018 and eccentricity > 0.96 (which almost only passes here), against e2 > 0.028. The neuron is very high (7.938), pushing W and Z down and t up; the formula calls them t and is right only 0.667 of the time, since wide gluons look the same here.
- **Wide jets with soft tail** — 0.7% of jets, neuron 7.94, formula right for 71%. Tops (68%) with gluons (23%), 81.5 GeV, very wide (0.0336) and of low pT (475 GeV) with a soft 7th and 8th particle (8th at 25.0 GeV). The decisive tests are the rarely-passed lam1 > 0.012 and pt_6 < 38 (+15.15) and lam1 > 0.0086 and pt_7 < 38 (+11.585), on top of LHA > 0.32 and lam1 > 0.016, with e2 > 0.028 the main minus. The neuron is very high (7.942), lowering W and Z and raising t; the formula calls them t and is right 0.707 of the time.
- **Heavy core plus wide halo tops** — 0.2% of jets, neuron 5.80, formula right for 92%. A tiny group (0.24%) of almost all tops (92%), 81.3 GeV, with an unusual split: 43% of pT within 0.05 and 51% beyond 0.1. It is defined by a pair that only fires here: mass_over_sum_pt > 0.11 and dr_7 < 0.049 (+24.105) against mass_over_sum_pt > 0.069 and dr_7 < 0.042 (-18.27), leaving a net plus that, with lam1 > 0.016 and width > 0.019 and pt_dispersion < 0.49, outweighs e2 > 0.028 and mass_over_sum_pt > 0.07 and tau32 < 0.52. The neuron is high (5.798), lowering W and Z and raising t; the formula calls them t and is right 0.923 of the time.

### neuron 4: Low C2/D2, moderate width (major)

- **What it measures:** Rises for jets of moderate width (girth2 > 0.0034 and girth < 0.13 push it up) with small e2_sq (< 0.017, its largest term) and small τ21 (< 0.24); it falls as C2 and D2 grow (-0.484, -0.444), i.e. when energy is spread beyond two prongs. Almost every jet has a sizeable value: largest for Z (7.18), then g (6.69), W (6.57), q (5.83), and lowest for t (4.71).
- *computed — its value:* largest for Z (7.18), then g (6.69), then W (6.57), then q (5.83), then t (4.71); it separates t jets from the rest best (AUC 0.36: small for t)
- **How the class scores use it:** The Z (+15%) and t (+18%) scores add it and the q (-26%) and g (-7%) scores subtract it, so a high value mainly pulls jets away from q toward Z. Tops sit lowest on this scale, so its place in the t score acts as a counterweight against q rather than as a top signature; it does not enter the W score.
- *computed — used by:* raises the score of Z (+15%), t (+18%); lowers the score of g (-7%), q (-26%); does not (or hardly) enter the score of W (share of each class score’s average input)
- **Boundaries:** On for almost every jet except those with large C2: highest (8.98) for C2 ≤ 0.0224 with more than 21.9% of pT at ΔR 0.05-0.1 (z_dr_0p05_0p1 > 0.219; 20% of jets, mostly Z at 37%), 8.81 for low-pT jets (top-5 pT sum ≤ 419 GeV; 5.4%, 47% g), 5.84 in the bulk (C2 ≤ 0.0552, z_dr_0p05_0p1 ≤ 0.219, top-5 pT sum > 419 GeV; 49% of jets), and dropping to 3.50, 1.32 and then off (0.17) above C2 = 0.0808 (7.4% of jets, 84% t). Because it is large almost everywhere, its adds act like a broad offset: in the bulk +0.73 to t, +0.46 to Z, -0.55 to q and -0.18 to g, and in the low-C2 regime +1.12 to t, +0.70 to Z, -0.84 to q; it does not enter W. The regimes describe it moderately (regime_r2 0.56).

Regimes (a small tree on its quantities; R² 0.561):

- `` — 19.6% of jets, value 8.98 (6.50…11.50), formula right 73%
- `` — 5.4% of jets, value 8.81 (5.25…12.25), formula right 63%
- `` — 13.1% of jets, value 7.27 (4.50…10.75), formula right 62%
- `` — 48.6% of jets, value 5.84 (3.75…7.50), formula right 62%
- `` — 3.9% of jets, value 3.50 (0.00…7.75), formula right 61%
- `` — 2.0% of jets, value 1.32 (0.00…4.25), formula right 68%
- `` — 7.4% of jets, value 0.17 (0.00…0.00), formula right 87%

```
z = -6.77
if e2_sq < 0.017: z += 526 × (0.017 − e2_sq)
if girth < 0.130: z += 49.20 × (0.130 − girth)
if girth2 > 0.0034: z += 839 × (girth2 − 0.0034)
if tau21 < 0.240: z += 41.10 × (0.240 − tau21)
if C2 > 0.013: z += -83.00 × (C2 − 0.013)
if tau21 < 0.260 and lam2 < 0.0013: z += -16900 × (0.260 − tau21) × (0.0013 − lam2)
if e2 > 0.020: z += 87.90 × (e2 − 0.020)
if e2_sq < 0.003: z += -1110 × (0.003 − e2_sq)
if mass_over_sum_pt > 0.089: z += -137 × (mass_over_sum_pt − 0.089)
if mass < 44.00: z += 0.082 × (44.00 − mass)
if girth2 > 0.0081: z += -484 × (girth2 − 0.0081)
if lam2 < 0.00033: z += -4450 × (0.00033 − lam2)
if max_dr > 0.094: z += 18.40 × (max_dr − 0.094)
if sum_pt < 760: z += 0.0059 × (760 − sum_pt)
if mass_over_sum_pt > 0.110: z += 102 × (mass_over_sum_pt − 0.110)
if centroid_offset < 0.014: z += 96.80 × (0.014 − centroid_offset)
if centroid_offset < 0.013 and z_dr_0p05_0p1 < 0.650: z += -231 × (0.013 − centroid_offset) × (0.650 − z_dr_0p05_0p1)
if width > -0.00013 and C2 < 0.063: z += 2140 × (width − -0.00013) × (0.063 − C2)
if sum_pt_top5 < 430: z += 0.022 × (430 − sum_pt_top5)
if girth2_top2 < 0.0087 and centroid_offset > 0.016: z += 17100 × (0.0087 − girth2_top2) × (centroid_offset − 0.016)
if C2 > 0.067: z += -102 × (C2 − 0.067)
if max_dr > 0.200: z += -24.60 × (max_dr − 0.200)
if tau21 < 0.230 and mass < 65.00: z += -0.618 × (0.230 − tau21) × (65.00 − mass)
if e2 > 0.064: z += -165 × (e2 − 0.064)
if max_dr > 0.110 and eccentricity > 0.980: z += -952 × (max_dr − 0.110) × (eccentricity − 0.980)
if C2 > 0.015 and pt_7 > 39.00: z += 9.44 × (C2 − 0.015) × (pt_7 − 39.00)
if tau21 < 0.250 and e2_sq > 0.011: z += -2370 × (0.250 − tau21) × (e2_sq − 0.011)
if tau21 < 0.230 and sum_pt_top2 < 410: z += 0.043 × (0.230 − tau21) × (410 − sum_pt_top2)
if tau21 < 0.240 and pt_7 > 32.00: z += 0.476 × (0.240 − tau21) × (pt_7 − 32.00)
if C2 > 0.065 and pt_7 < 36.00: z += -9.53 × (C2 − 0.065) × (36.00 − pt_7)
if max_dr > 0.098 and pt_7 > 38.00: z += -1.64 × (max_dr − 0.098) × (pt_7 − 38.00)
if tau21 < 0.290 and pt_7 < 24.00: z += -1.15 × (0.290 − tau21) × (24.00 − pt_7)
if mass < 70.00 and mean_eta2 > 0.0044: z += -6.87 × (70.00 − mass) × (mean_eta2 − 0.0044)
if tau21 < 0.220 and mean_phi < -0.028: z += -416 × (0.220 − tau21) × (-0.028 − mean_phi)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 35.5% of jets, neuron 6.63, formula right for 59%. The largest group (35.5%), mostly light quarks (42%) with gluons (36%), nearly massless (9.6 GeV) with 96% of pT within 0.05 of the axis. e2_sq < 0.017 (+8.81), girth < 0.13 and mass < 44 always pass, while the only big minus, e2_sq < 0.003 (-3.051), also always passes here and nowhere else as often; girth2 > 0.0034 and e2 > 0.02 almost never pass. The neuron is on for every jet (6.633), which raises the Z and t scores and lowers q and g, the same as for most groups, so it does little to separate them; the formula splits them between q and g and is right only 0.592 of the time.
- **Clean two-prong Z/W jets** — 17.2% of jets, neuron 9.12, formula right for 77%. Mostly Z (46%) with W (35%) and some tops, mass 61.0 GeV with 65% of pT at 0.05-0.1 from the axis, i.e. two well-separated prongs. Every jet passes tau21 < 0.24 (+6.883) on top of e2_sq < 0.017, girth2 > 0.0034, girth < 0.13 and e2 > 0.02, while being too heavy for mass < 44 and too spread for e2_sq < 0.003; tau21 < 0.26 and lam2 < 0.0013 takes back 3.936. This gives the neuron's highest value (9.116), raising the Z and t scores and lowering q and g; the formula calls them Z and is right 0.772 of the time, W and Z being hard to tell apart here.
- **Medium-mass two-prong W jets** — 12.7% of jets, neuron 6.07, formula right for 63%. W-led (43%) with Z (28%), 46.7 GeV, a bit narrower than average with pT split between the core (59%) and 0.05-0.1. e2_sq < 0.017, tau21 < 0.24 and girth < 0.13 all pass, reduced by tau21 < 0.26 and lam2 < 0.0013, lam2 < 0.00033 and tau21 < 0.23 and mass < 65 (which passes far more often here). The neuron is well on (6.072), adding to Z and t and taking from q and g; the formula calls them W and is right 0.626 of the time.
- **Compact medium-mass mixed jets** — 11.7% of jets, neuron 4.66, formula right for 54%. A mixture (W 34%, Z 25%, g 18%, q 14%) at 38.3 GeV, fairly narrow (70% of pT within 0.05) and without a clean two-prong shape. They pass e2_sq < 0.017, girth < 0.13 and mostly max_dr > 0.094, but all pass C2 > 0.013 (-2.74) and most fail tau21 < 0.24, so the neuron is lower than for the clean boson groups (4.661, on for 0.9). It still pushes Z and t up and q and g down; the formula calls them W but is right only 0.542 of the time.
- **Moderately wide low-pT mixed jets** — 7.0% of jets, neuron 7.04, formula right for 61%. A real mixture (Z 33%, t 24%, g 18%, W 17%) of 48.8 GeV, somewhat wider than average, with low total pT (580 GeV) and a soft leading particle. Nearly all pass e2_sq < 0.017, girth2 > 0.0034 (+3.916), girth < 0.13, e2 > 0.02, max_dr > 0.094 and sum_pt < 760, with only C2 > 0.013 (-2.709) against; tau21 < 0.24 passes for under half. The neuron is high (7.044), raising Z and t and lowering q and g; the formula calls them Z but is right only 0.609 of the time.
- **Wide two-prong-looking top jets** — 4.2% of jets, neuron 7.16, formula right for 76%. Mostly tops (74%, g 16%), 70.6 GeV, about 2.4 times the average width with 58% of pT beyond 0.1. girth2 > 0.0034 (+9.971) and tau21 < 0.24 (+6.393) dominate, against mass_over_sum_pt > 0.089, girth2 > 0.0081, tau21 < 0.26 and lam2 < 0.0013 and tau21 < 0.25 and e2_sq > 0.011, the last two firing much more than elsewhere. The neuron stays high (7.161), raising Z and t and lowering q and g; the formula calls them t and is right 0.756 of the time.
- **Wide heavy three-prong tops** — 4.2% of jets, neuron 0.37, formula right for 93%. Almost all tops (93%), 84.8 GeV, four times the average width with 76% of pT beyond 0.1. They fail e2_sq < 0.017 and tau21 < 0.24, so the big plus girth2 > 0.0034 (+19.26), with e2 > 0.02 and mass_over_sum_pt > 0.11, is beaten by mass_over_sum_pt > 0.089 (-9.407), girth2 > 0.0081, C2 > 0.013 and e2 > 0.064. The neuron is mostly off (on for 0.157, mean 0.373), so it barely adds to the scores; the formula calls them t anyway and is right 0.93 of the time.
- **Wide top jets, high C2** — 3.9% of jets, neuron 0.87, formula right for 82%. Mostly tops (81%, g 14%), 71.5 GeV, 2.8 times the average width with half the pT beyond 0.1. girth2 > 0.0034 (+11.914), e2 > 0.02 and max_dr > 0.094 are outweighed by C2 > 0.013 (-6.433), mass_over_sum_pt > 0.089, girth2 > 0.0081 and, for most, C2 > 0.067, while tau21 < 0.24 mostly fails. The neuron is usually zero (on for 0.28, mean 0.867); the formula calls them t and is right 0.817 of the time.
- **Very wide jets with small tau21** — 2.1% of jets, neuron 4.74, formula right for 71%. Tops (67%) with gluons (23%), 87.7 GeV, four times the average width with 84% of pT beyond 0.1, but with a two-prong-like tau21. Unlike the other wide groups they all pass tau21 < 0.24 (+6.772), which with girth2 > 0.0034 (+18.281), e2 > 0.02 and mass_over_sum_pt > 0.11 beats mass_over_sum_pt > 0.089, girth2 > 0.0081 and tau21 < 0.25 and e2_sq > 0.011 (almost only passing here). The neuron is on (4.743), raising Z and t; the formula calls them t and is right 0.709 of the time, the gluon admixture likely accounting for most errors.
- **Widest soft-leading top jets** — 1.5% of jets, neuron 1.07, formula right for 80%. Tops (77%) with gluons (18%), 90.4 GeV, the widest group (0.0369) with 88.5% of pT beyond 0.1 and a soft leading particle (124 GeV). The huge plus girth2 > 0.0034 (+28.087), with mass_over_sum_pt > 0.11 and e2 > 0.02, is cancelled by girth2 > 0.0081, mass_over_sum_pt > 0.089, C2 > 0.013 and e2 > 0.064, and girth < 0.13 never passes. The neuron is usually zero (on for 0.337, mean 1.073); the formula calls them t and is right 0.795 of the time.

### neuron 9: Narrow, centred one-prong jet (major)

- **What it measures:** Rises for narrow jets (width < 0.0061 is its largest term; max ΔR < 0.23 and centroid offset < 0.018 also push it up); overall it falls with λ1 (-0.734), width and girth2 (-0.727). Largest for q (9.14) and g (7.10), far lower for W (2.50), t (1.77) and Z (1.45).
- *computed — its value:* largest for q (9.14), then g (7.10), then W (2.50), then t (1.77), then Z (1.45); it separates q jets from the rest best (AUC 0.80: large for q)
- **How the class scores use it:** The q (+49%) and g (+27%) scores are built mainly on it, since both light-parton types sit high; the Z (-4%) and W (-3%) scores subtract a little. It does not enter the t score, which uses the related compactness scale (neuron 13) instead.
- *computed — used by:* raises the score of g (+27%), q (+49%); lowers the score of W (-3%), Z (-4%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** Very high (12.1) for the thinnest, best-centred jets (girth2 ≤ 0.00148, centroid ≤ 0.00767), 19% of jets and mostly q (57%); 9.84 for centroid 0.00767-0.0149 (7%, 47% g) and 6.08 for 0.0149-0.0216, and 6.37 for slightly wider light jets (girth2 > 0.00148, mass ≤ 38.7 GeV, centred). It is low (0.73) for wider jets heavier than 38.7 GeV with e2 ≤ 0.0858 (50% of jets), except 4.64 for those with e2 > 0.0858 (3.4%, 86% t). It is the main source of both light-jet scores: +3.08 to q and +2.08 to g in the top regime (+2.50 / +1.69 in the next), with -0.38 to W and Z and nothing to t; the regimes describe it very well (regime_r2 0.86).

Regimes (a small tree on its quantities; R² 0.863):

- `` — 19.5% of jets, value 12.12 (10.50…13.75), formula right 66%
- `` — 7.0% of jets, value 9.84 (7.00…12.25), formula right 58%
- `` — 5.0% of jets, value 6.37 (2.25…10.75), formula right 48%
- `` — 3.6% of jets, value 6.08 (3.00…9.25), formula right 48%
- `` — 3.4% of jets, value 4.64 (0.25…9.25), formula right 87%
- `` — 3.6% of jets, value 2.36 (0.00…5.50), formula right 42%
- `` — 8.3% of jets, value 2.02 (0.00…4.75), formula right 48%
- `` — 49.6% of jets, value 0.73 (0.00…2.50), formula right 73%

```
z = -1.82
if width < 0.0061: z += 3030 × (0.0061 − width)
if mass_over_sum_pt < 0.076: z += -127 × (0.076 − mass_over_sum_pt)
if mass_over_sum_pt_sq < 0.0033: z += 1400 × (0.0033 − mass_over_sum_pt_sq)
if mass < 55.00 and centroid_offset < 0.026: z += 5.17 × (55.00 − mass) × (0.026 − centroid_offset)
if girth < 0.055: z += -92.30 × (0.055 − girth)
if mass < 43.00 and centroid_offset < 0.027: z += -6.37 × (43.00 − mass) × (0.027 − centroid_offset)
if max_dr < 0.230: z += 12.10 × (0.230 − max_dr)
if width < 0.0059 and centroid_offset > 0.0029: z += -48100 × (0.0059 − width) × (centroid_offset − 0.0029)
if centroid_offset < 0.018: z += 151 × (0.018 − centroid_offset)
if mass < 52.00 and log_sum_pt < 6.80: z += 0.190 × (52.00 − mass) × (6.80 − log_sum_pt)
if e2 < 0.017 and centroid_offset < 0.024: z += 11400 × (0.017 − e2) × (0.024 − centroid_offset)
if centroid_offset < 0.018 and z_4 > 0.043: z += -2590 × (0.018 − centroid_offset) × (z_4 − 0.043)
if e2 < 0.032 and dr01 < 0.056: z += 823 × (0.032 − e2) × (0.056 − dr01)
if mass_over_sum_pt < 0.072 and girth2_top2 < 0.00078: z += -35700 × (0.072 − mass_over_sum_pt) × (0.00078 − girth2_top2)
if log_sum_pt > 6.40: z += -2.06 × (log_sum_pt − 6.40)
if girth2 < 0.0046 and centroid_offset > 0.011: z += -60200 × (0.0046 − girth2) × (centroid_offset − 0.011)
if centroid_offset < 0.018 and pt_4 > 49.00: z += 3.08 × (0.018 − centroid_offset) × (pt_4 − 49.00)
if mass < 51.00 and planar_flow < 0.320: z += 0.219 × (51.00 − mass) × (0.320 − planar_flow)
if girth2 > 0.018: z += 326 × (girth2 − 0.018)
if centroid_offset < 0.018 and z_5 > 0.033: z += -1200 × (0.018 − centroid_offset) × (z_5 − 0.033)
if girth2_top2 < 0.00073 and z_7 > 0.023: z += -69100 × (0.00073 − girth2_top2) × (z_7 − 0.023)
if e2 < 0.019 and pt_7 < 53.00: z += -2.29 × (0.019 − e2) × (53.00 − pt_7)
if sum_pt > 850: z += -0.0091 × (sum_pt − 850)
if e2 < 0.032 and tau21 < 0.450: z += -265 × (0.032 − e2) × (0.450 − tau21)
if mean_phi < 0.00042: z += -34.10 × (0.00042 − mean_phi)
if n_dr_0p2_0p4 > 0.900: z += 1.09 × (n_dr_0p2_0p4 − 0.900)
if mass < 33.00 and planar_flow < 0.320: z += -0.335 × (33.00 − mass) × (0.320 − planar_flow)
if C2 > 0.051: z += 29.40 × (C2 − 0.051)
if girth2_top2 < 0.00075 and pt_7 > 33.00: z += 111 × (0.00075 − girth2_top2) × (pt_7 − 33.00)
if lam2 > 0.0017: z += 376 × (lam2 − 0.0017)
if C2 > 0.051 and pt_2 < 85.00: z += -1.03 × (C2 − 0.051) × (85.00 − pt_2)
if C2 > 0.048 and dr_5 < 0.190: z += -230 × (C2 − 0.048) × (0.190 − dr_5)
if girth2 > 0.018 and planar_flow < 0.360: z += -581 × (girth2 − 0.018) × (0.360 − planar_flow)
if log_sum_pt > 6.40 and mean_phi > 0.026: z += -2120 × (log_sum_pt − 6.40) × (mean_phi − 0.026)
if girth2 > 0.019 and mean_eta < 0.025: z += -2270 × (girth2 − 0.019) × (0.025 − mean_eta)
if width < 0.006 and C2 > 0.031: z += -9190 × (0.006 − width) × (C2 − 0.031)
if lam2 > 0.001 and mass_top2 > 16.00: z += 10.80 × (lam2 − 0.001) × (mass_top2 − 16.00)
if n_dr_0p2_0p4 > 0.870 and dr_6 > 0.220: z += 9.75 × (n_dr_0p2_0p4 − 0.870) × (dr_6 − 0.220)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Wide massive top-led jets** — 21.7% of jets, neuron 0.37, formula right for 70%. A top-led mixture (t 46%, Z 24.5%, g 15%) at 60.1 GeV, about twice the average width with 35% of pT beyond 0.1. Too wide for width < 0.0061 (passes 0.159), the neuron's main plus; what is left are small terms like max_dr < 0.23 (+0.694), mass < 52 and log_sum_pt < 6.8 and mean_phi < 0.00042 (-0.329). The neuron is usually zero (on for 0.263, mean 0.371), barely touching the scores; the formula calls them t and is right 0.699 of the time.
- **Centred two-prong Z/W jets** — 15.7% of jets, neuron 0.41, formula right for 77%. Z (42%) and W (38%) with some tops, 62.1 GeV, 58% of pT at 0.05-0.1 from the axis and a well-centred pT distribution. centroid_offset < 0.018 (+1.753) and max_dr < 0.23 pass for nearly all, but centroid_offset < 0.018 and z_4 > 0.043 (-1.338), centroid_offset < 0.018 and z_5 > 0.033 and log_sum_pt > 6.4 take most back, and width < 0.0061 mostly fails. The neuron is usually zero (on for 0.348, mean 0.412); the formula splits them evenly between W and Z and is right 0.771 of the time.
- **Pencil-thin high-pT quark jets** — 13.6% of jets, neuron 12.09, formula right for 69%. Mostly light quarks (64%, g 24%), essentially massless (6.9 GeV), all pT within 0.05, with above-average pT (905 GeV) and a hard leading particle. width < 0.0061 adds the most here (+18.244, more the narrower the jet), with mass < 55 and centroid_offset < 0.026, mass_over_sum_pt_sq < 0.0033 and e2 < 0.017 and centroid_offset < 0.024, against mass_over_sum_pt < 0.076 (-8.688), mass < 43 and centroid_offset < 0.027 and girth < 0.055. This is the neuron's highest value (12.086), strongly raising q and g and lowering W and Z; the formula calls them q and is right 0.686 of the time.
- **Medium-mass two-prong W jets** — 10.8% of jets, neuron 1.88, formula right for 63%. Mostly W (49%) with Z (27%), 46.9 GeV, of intermediate width (55% of pT within 0.05, 35% at 0.05-0.1). width < 0.0061 passes but adds only +4.769 because they are close to the cut; mass_over_sum_pt < 0.076, width < 0.0059 and centroid_offset > 0.0029 and e2 < 0.032 and tau21 < 0.45 subtract, while max_dr < 0.23, centroid_offset < 0.018 and mass < 52 and log_sum_pt < 6.8 add. The neuron is modest (1.881), nudging g and q up; the formula calls them W and is right 0.633 of the time.
- **Very narrow light gluon/quark jets** — 9.6% of jets, neuron 11.36, formula right for 62%. Gluons (49%) and quarks (38%), light (10.7 GeV), 99% of pT within 0.05, at average pT. width < 0.0061 (+17.49), mass_over_sum_pt_sq < 0.0033, mass < 55 and centroid_offset < 0.026, max_dr < 0.23 and e2 < 0.017 and centroid_offset < 0.024 outweigh mass_over_sum_pt < 0.076, mass < 43 and centroid_offset < 0.027, girth < 0.055 and girth2_top2 < 0.00073 and z_7 > 0.023. The neuron is very high (11.36), raising g and q; the formula calls them g and is right 0.616 of the time, quark-gluon confusion likely being the error.
- **Compact medium-mass W-led mixture** — 7.6% of jets, neuron 3.10, formula right for 52%. A W-led (39%) mixture with Z (22%), gluons (18%) and quarks (13%) at 36.5 GeV, narrow (77% of pT within 0.05). width < 0.0061 (+9.381) and several smaller pluses (mass_over_sum_pt_sq < 0.0033, mass < 52 and log_sum_pt < 6.8, max_dr < 0.23) beat mass_over_sum_pt < 0.076, width < 0.0059 and centroid_offset > 0.0029, girth2 < 0.0046 and centroid_offset > 0.011 and e2 < 0.032 and tau21 < 0.45. The neuron is on (3.099), raising g and q and slightly lowering W and Z; the formula calls them W but is right only 0.522 of the time.
- **Very wide top jets** — 6.1% of jets, neuron 3.59, formula right for 86%. Mostly tops (84%), 87.7 GeV, 4.6 times the average width with 81% of pT beyond 0.1. They never pass width < 0.0061, but a separate set switches the neuron on: girth2 > 0.018 (+3.762), n_dr_0p2_0p4 > 0.9, lam2 > 0.0017 and C2 > 0.051, partly cancelled by girth2 > 0.018 and planar_flow < 0.36, C2 > 0.051 and pt_2 < 85 and girth2 > 0.019 and mean_eta < 0.025. The neuron is on (3.587), raising g and q rather than t, yet the formula calls them t and is right 0.862 of the time.
- **Narrow light gluon/quark jets** — 5.3% of jets, neuron 8.70, formula right for 54%. Gluons (39%) and quarks (34%) with some W, 25.8 GeV, narrow (89% of pT within 0.05). width < 0.0061 (+14.153), mass_over_sum_pt_sq < 0.0033 and mass < 55 and centroid_offset < 0.026 beat mass_over_sum_pt < 0.076, girth < 0.055, mass < 43 and centroid_offset < 0.027, width < 0.0059 and centroid_offset > 0.0029 and e2 < 0.032 and tau21 < 0.45. The neuron is high (8.701), raising g and q; the formula calls them g but is right only 0.539 of the time.
- **Massless slightly off-centre mixture** — 4.9% of jets, neuron 5.62, formula right for 47%. A gluon-led (37%) mixture with W (23%), quarks (21%) and Z (16%), nearly massless (8.4 GeV) and narrow, with the pT centroid slightly off the axis. width < 0.0061 (+16.865), mass_over_sum_pt_sq < 0.0033 and max_dr < 0.23 outweigh mass_over_sum_pt < 0.076, width < 0.0059 and centroid_offset > 0.0029, girth < 0.055 and girth2 < 0.0046 and centroid_offset > 0.011 (passing for all here). The neuron is high (5.625), raising g and q even for the bosons; the formula calls them g but is right only 0.467 of the time.
- **Light off-centre mixed jets** — 4.6% of jets, neuron 1.94, formula right for 44%. A mixture (g 33%, Z 25%, W 20%, q 13%), nearly massless (9.7 GeV) with a softer leading particle (185 GeV) and a clearly off-centre pT centroid. No jet passes centroid_offset < 0.018, and the off-centre penalties width < 0.0059 and centroid_offset > 0.0029 (-6.218) and girth2 < 0.0046 and centroid_offset > 0.011, together with mass_over_sum_pt < 0.076 and mass < 33 and planar_flow < 0.32, eat most of width < 0.0061 (+13.892). The neuron is moderate (1.939, on for 0.649); the formula calls them g but is right only 0.435 of the time, many being low-mass bosons.

### neuron 10: Broad, massive, spread-out jet (major)

- **What it measures:** Rises for jets spread out in both directions (λ2 < 0.0034, its largest term, pushes it down; girth2 > 0.0085 up) with mass/pT < 0.1 and mass > 17 GeV; overall it grows with max ΔR (+0.603), mass (+0.556) and C2 (+0.543). Largest for t (4.14), then W (1.39), Z (1.21), g (0.97) and q (0.61).
- *computed — its value:* largest for t (4.14), then W (1.39), then Z (1.21), then g (0.97), then q (0.61); it separates q jets from the rest best (AUC 0.25: small for q)
- **How the class scores use it:** The t score adds it (+14%) and the q score subtracts it (-9%), since tops sit highest and quarks lowest on this scale. It does not enter the g, W or Z scores.
- *computed — used by:* raises the score of t (+14%); lowers the score of q (-9%); does not (or hardly) enter the score of g, W, Z (share of each class score’s average input)
- **Boundaries:** High for heavy three-prong jets: 10.9 for C2 > 0.0771, mass > 66.1 GeV and λ2 > 0.00353 (3.5% of jets, 95% t) and 7.80 with λ2 ≤ 0.00353 (2.2%, 89% t); 4.95 for low-C2 jets with λ2 > 0.00174 and 4.34 for C2 > 0.0771 below 66.1 GeV. In the bulk (C2 ≤ 0.0771, λ2 ≤ 0.00174, girth2 > 0.00082; 62% of jets) it is 1.32, and for the thinnest jets (girth2 ≤ 0.00082; 28%, mostly q at 47%) 0.34. It adds to t (+4.07 in the top regime, +0.50 in the bulk) and subtracts from q (-1.36 top, -0.17 bulk), not entering g, W or Z; the regimes describe it well (regime_r2 0.82).

Regimes (a small tree on its quantities; R² 0.818):

- `` — 3.5% of jets, value 10.86 (8.12…13.88), formula right 95%
- `` — 2.2% of jets, value 7.80 (4.69…11.00), formula right 89%
- `` — 2.1% of jets, value 4.95 (2.06…8.34), formula right 81%
- `` — 2.4% of jets, value 4.34 (1.31…7.56), formula right 70%
- `` — 61.8% of jets, value 1.32 (0.19…2.19), formula right 64%
- `` — 28.1% of jets, value 0.34 (0.00…0.81), formula right 62%

```
z = 2.32
if lam2 < 0.0034: z += -835 × (0.0034 − lam2)
if mass_over_sum_pt < 0.100: z += 41.30 × (0.100 − mass_over_sum_pt)
if girth2 > 0.0085: z += 494 × (girth2 − 0.0085)
if e2 < 0.037: z += -56.70 × (0.037 − e2)
if girth2 < 0.0017: z += -1460 × (0.0017 − girth2)
if lam1 > 0.0085: z += -347 × (lam1 − 0.0085)
if mass > 17.00: z += 0.024 × (mass − 17.00)
if girth2_top2 < 0.0038: z += 349 × (0.0038 − girth2_top2)
if lam1 < 0.0044: z += -271 × (0.0044 − lam1)
if n_dr_0p05_0p1 < 3.00: z += 0.183 × (3.00 − n_dr_0p05_0p1)
if e2 > 0.036: z += 45.90 × (e2 − 0.036)
if eccentricity > 0.890 and z_dr_0p2_0p4 < 0.044: z += 149 × (eccentricity − 0.890) × (0.044 − z_dr_0p2_0p4)
if log_sum_pt > 6.70: z += -8.24 × (log_sum_pt − 6.70)
if LHA > 0.300 and tau21 < 0.570: z += -40.70 × (LHA − 0.300) × (0.570 − tau21)
if girth2_top3 < 0.0016: z += 463 × (0.0016 − girth2_top3)
if eccentricity > 0.890 and mass_top2 < 37.00: z += -0.145 × (eccentricity − 0.890) × (37.00 − mass_top2)
if LHA > 0.330: z += -17.00 × (LHA − 0.330)
if log_sum_pt < 6.30: z += -5.54 × (6.30 − log_sum_pt)
if mass > 8.00 and n_dr_0p2_0p4 < 1.90: z += -0.0033 × (mass − 8.00) × (1.90 − n_dr_0p2_0p4)
if tau32 < 0.270: z += 9.95 × (0.270 − tau32)
if C2 > 0.055: z += 24.20 × (C2 − 0.055)
if lam1 < 0.0044 and log_sum_pt > 6.80: z += 2070 × (0.0044 − lam1) × (log_sum_pt − 6.80)
if z_7 > 0.062: z += -18.00 × (z_7 − 0.062)
if lam2 > 0.00048 and n_pt_above_50 < 8.10: z += -49.50 × (lam2 − 0.00048) × (8.10 − n_pt_above_50)
if lam2 > 0.00023 and tau21 < 0.510: z += 1370 × (lam2 − 0.00023) × (0.510 − tau21)
if n_dr_0p2_0p4 > 1.90: z += 0.756 × (n_dr_0p2_0p4 − 1.90)
if lam1 > 0.0083 and min_pair_mass < 2.80: z += -16.30 × (lam1 − 0.0083) × (2.80 − min_pair_mass)
if pt_7 > 46.00: z += 0.033 × (pt_7 − 46.00)
if tau32 < 0.270 and n_dr_0p2_0p4 < 2.00: z += -3.53 × (0.270 − tau32) × (2.00 − n_dr_0p2_0p4)
if centroid_offset > 0.038: z += 17.50 × (centroid_offset − 0.038)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Two-prong Z/W jets** — 27.0% of jets, neuron 1.25, formula right for 71%. The largest group (27%): Z (41%), W (33%) and some tops at 56.1 GeV, pT mostly 0.05-0.1 from the axis (two prongs). lam2 < 0.0034 (-2.661) passes for nearly all and is outweighed by mass > 17, mass_over_sum_pt < 0.1 and eccentricity > 0.89 and z_dr_0p2_0p4 < 0.044, with small minuses mass > 8 and n_dr_0p2_0p4 < 1.9 and eccentricity > 0.89 and mass_top2 < 37. The neuron is modest (1.246, on for 0.926), raising the t score and lowering q; the formula calls them Z and is right 0.713 of the time, mostly confusing W and Z.
- **Pencil-thin light quark/gluon jets** — 22.3% of jets, neuron 0.44, formula right for 59%. Mostly light quarks (43%) with gluons (37%), nearly massless (8.0 GeV) with 99% of pT within 0.05. mass_over_sum_pt < 0.1 (+3.686), girth2_top2 < 0.0038 and girth2_top3 < 0.0016 are outweighed by lam2 < 0.0034, girth2 < 0.0017, e2 < 0.037 and lam1 < 0.0044, and they are too light for mass > 17. The neuron is low (0.443, on for 0.752), with a small push up of t and down of q; the formula calls them q and is right only 0.59 of the time.
- **Medium-mass W-led jets** — 16.9% of jets, neuron 1.58, formula right for 60%. W-led (39%) mixture with Z (29%) and gluons (13%), 45.7 GeV, fairly narrow (71% of pT within 0.05). mass_over_sum_pt < 0.1, girth2_top2 < 0.0038 and mass > 17 add, while lam2 < 0.0034 (-2.715), e2 < 0.037 and eccentricity > 0.89 and mass_top2 < 37 subtract. The neuron is moderate (1.578), raising t and lowering q; the formula calls them W but is right only 0.598 of the time.
- **Light compact gluon-led mixture** — 10.9% of jets, neuron 1.29, formula right for 50%. A gluon-led (36%) mixture with quarks (23%), W (18%) and Z (16%), 20.1 GeV, narrow (84% of pT within 0.05). mass_over_sum_pt < 0.1 (+2.901), girth2_top2 < 0.0038 and n_dr_0p05_0p1 < 3 beat lam2 < 0.0034, e2 < 0.037, lam1 < 0.0044 and (for about half) girth2 < 0.0017. The neuron is on (1.292), raising t and lowering q; the formula calls them g but is right only 0.497 of the time, a badly mixed group.
- **High-pT pencil-thin quark jets** — 6.5% of jets, neuron 0.08, formula right for 71%. Mostly light quarks (60%, g 25%), nearly massless, at high pT (1031 GeV) with a 465 GeV leading particle. The narrow-jet minuses lam2 < 0.0034, girth2 < 0.0017 and e2 < 0.037, plus log_sum_pt > 6.7 (-1.924), outweigh mass_over_sum_pt < 0.1, girth2_top2 < 0.0038 and lam1 < 0.0044 and log_sum_pt > 6.8. The neuron is usually zero (on for 0.156); the formula calls them q and is right 0.712 of the time.
- **Moderately wide top jets** — 5.7% of jets, neuron 2.74, formula right for 75%. Mostly tops (74%, g 18%), 66.5 GeV, about twice the average width with 48% of pT beyond 0.1. Most fail mass_over_sum_pt < 0.1, but girth2 > 0.0085 (+3.007, always passing here), mass > 17 and e2 > 0.036 add, against lam2 < 0.0034, lam1 > 0.0085, LHA > 0.3 and tau21 < 0.57 and LHA > 0.33. The neuron is on (2.738), raising t and lowering q; the formula calls them t and is right 0.749 of the time, the gluons being errors.
- **Wide heavy top jets** — 3.4% of jets, neuron 2.27, formula right for 76%. Mostly tops (75%, g 18%), 79.2 GeV, over three times the average width with 73% of pT beyond 0.1. girth2 > 0.0085 (+6.057), mass > 17 and e2 > 0.036 add, but lam1 > 0.0085 (-4.047), lam2 < 0.0034 and the two LHA tests take much of it back. The neuron is on (2.267), raising t and lowering q; the formula calls them t and is right 0.762 of the time.
- **Very wide top jets** — 3.3% of jets, neuron 7.24, formula right for 85%. Mostly tops (84%), 88.3 GeV, 4.6 times the average width with 83% of pT beyond 0.1. girth2 > 0.0085 grows with width (+10.465 here) and, with e2 > 0.036 and mass > 17, beats lam1 > 0.0085 (-6.109) and the LHA tests, while lam2 < 0.0034 passes for only 0.596. The neuron is high (7.244), strongly raising t and lowering q; the formula calls them t and is right 0.852 of the time.
- **Clean three-prong top jets** — 3.0% of jets, neuron 9.22, formula right for 96%. Almost pure tops (96%), 81.5 GeV, wide (72% of pT beyond 0.1) with a clear three-prong structure. Unlike the other wide groups most escape lam2 < 0.0034 (passes 0.188), and they pass tau32 < 0.27 and C2 > 0.055 (rare elsewhere), which with girth2 > 0.0085, e2 > 0.036 and mass > 17 outweigh lam1 > 0.0085 and LHA > 0.33. This is the neuron's highest value (9.224), raising t most strongly; the formula calls them t and is right 0.958 of the time.
- **Widest top/gluon jets** — 0.9% of jets, neuron 7.72, formula right for 68%. Tops (61%) with many gluons (29%), 93.0 GeV, the widest group (0.0397) with 91% of pT beyond 0.1 and low pT. girth2 > 0.0085 (+15.432), e2 > 0.036, mass > 17 and n_dr_0p2_0p4 > 1.9 (almost only here) beat lam1 > 0.0085 (-9.63), LHA > 0.33 and LHA > 0.3 and tau21 < 0.57. The neuron is high (7.722), raising t; the formula calls them t but is right only 0.675 of the time, the wide gluons likely being misread as tops.

### neuron 11: Compact, centred, flat jet (major)

- **What it measures:** Rises for fairly compact jets (width < 0.0085) whose pT centroid sits near the axis (centroid offset < 0.04) and with low planar flow (< 0.27), while girth < 0.089 pulls it down; overall it falls with centroid offset (-0.602) and grows with total pT (+0.499). Largest for W (4.79), then Z (3.50), q (3.42) and g (2.89), and far lower for t (0.83).
- *computed — its value:* largest for W (4.79), then Z (3.50), then q (3.42), then g (2.89), then t (0.83); it separates t jets from the rest best (AUC 0.13: small for t)
- **How the class scores use it:** Only the W score uses it (+26%, the W score's largest input), because W jets sit highest; the sizeable values on Z, q and g jets are countered by other inputs of the W score (neurons 14 and 15 for Z-like jets, 8 and 9 for narrow jets). It does not enter the g, q, Z or t scores.
- *computed — used by:* raises the score of W (+26%); does not (or hardly) enter the score of g, q, Z, t (share of each class score’s average input)
- **Boundaries:** High for compact (girth2 ≤ 0.00839), centred (centroid ≤ 0.0217) jets: 5.07 when τ21 ≤ 0.194 (25% of jets, mostly W at 43%) and 3.70 when τ21 > 0.194 (39%, mostly q at 39%), falling to 2.59 for centroid 0.0217-0.0322 and 1.18 beyond 0.0322. It is off for broad jets (girth2 > 0.00936: 0.035 with ≤ 6.5 particles above 50 GeV, 16% of jets, 78% t; 0.38 otherwise). It feeds only the W score, adding +1.90 and +1.39 in the two big compact regimes (together about two thirds of jets); the regimes describe it fairly well (regime_r2 0.72).

Regimes (a small tree on its quantities; R² 0.721):

- `` — 25.2% of jets, value 5.07 (3.12…7.00), formula right 73%
- `` — 39.3% of jets, value 3.70 (2.69…4.88), formula right 61%
- `` — 9.9% of jets, value 2.59 (1.25…3.88), formula right 52%
- `` — 2.1% of jets, value 1.24 (0.00…3.00), formula right 66%
- `` — 5.1% of jets, value 1.18 (0.00…2.38), formula right 47%
- `` — 2.3% of jets, value 0.38 (0.00…0.83), formula right 76%
- `` — 16.3% of jets, value 0.04 (0.00…0.00), formula right 79%

```
z = -0.351
if width < 0.0085: z += 1320 × (0.0085 − width)
if centroid_offset < 0.040: z += 98.30 × (0.040 − centroid_offset)
if girth < 0.089: z += -61.90 × (0.089 − girth)
if sum_pt_top5 < 690: z += 0.009 × (690 − sum_pt_top5)
if planar_flow < 0.270: z += 9.88 × (0.270 − planar_flow)
if e2_sq < 0.0062: z += -393 × (0.0062 − e2_sq)
if girth > 0.076: z += -103 × (girth − 0.076)
if centroid_offset < 0.050 and log_sum_pt < 6.80: z += -133 × (0.050 − centroid_offset) × (6.80 − log_sum_pt)
if girth > 0.089 and n_pt_above_50 < 7.10: z += -37.20 × (girth − 0.089) × (7.10 − n_pt_above_50)
if girth > 0.077 and n_pt_above_50 < 7.00: z += 27.40 × (girth − 0.077) × (7.00 − n_pt_above_50)
if max_dr < 0.220: z += 6.36 × (0.220 − max_dr)
if centroid_offset > 0.014: z += -84.60 × (centroid_offset − 0.014)
if width < 0.0036: z += -425 × (0.0036 − width)
if C2 < 0.035: z += -27.10 × (0.035 − C2)
if planar_flow < 0.270 and mass < 71.00: z += -0.147 × (0.270 − planar_flow) × (71.00 − mass)
if planar_flow < 0.210 and width > 0.0056: z += -1310 × (0.210 − planar_flow) × (width − 0.0056)
if girth > 0.075 and pt_7 < 40.00: z += -4.29 × (girth − 0.075) × (40.00 − pt_7)
if n_dr_0p1_0p2 < 2.80: z += -0.160 × (2.80 − n_dr_0p1_0p2)
if log_sum_pt < 6.70 and dr_0 < 0.120: z += -27.80 × (6.70 − log_sum_pt) × (0.120 − dr_0)
if planar_flow < 0.230 and girth2 > 0.015: z += 3710 × (0.230 − planar_flow) × (girth2 − 0.015)
if girth2_top3 < 0.002: z += -399 × (0.002 − girth2_top3)
if girth < 0.021: z += -77.50 × (0.021 − girth)
if girth2 < 0.013 and mean_phi < -0.0014: z += -6780 × (0.013 − girth2) × (-0.0014 − mean_phi)
if e2_sq < 0.0063 and mean_phi < -1.8e-05: z += 14900 × (0.0063 − e2_sq) × (-1.8e-05 − mean_phi)
if planar_flow < 0.260 and max_dr > 0.110: z += -33.10 × (0.260 − planar_flow) × (max_dr − 0.110)
if girth2 < 0.014 and mean_phi > 0.0032: z += -6840 × (0.014 − girth2) × (mean_phi − 0.0032)
if pt_7 < 30.00 and n_dr_0p2_0p4 < 1.90: z += -0.041 × (30.00 − pt_7) × (1.90 − n_dr_0p2_0p4)
if e2_sq < 0.0062 and mean_phi > 0.0014: z += 15100 × (0.0062 − e2_sq) × (mean_phi − 0.0014)
if centroid_offset > 0.015 and tau21 < 0.110: z += 1650 × (centroid_offset − 0.015) × (0.110 − tau21)
if LHA < 0.160 and z_7 < 0.028: z += 1400 × (0.160 − LHA) × (0.028 − z_7)
if mass_top5 < 5.50: z += 0.103 × (5.50 − mass_top5)
if girth2 < 0.013 and mass_top5 > 49.00: z += 7.00 × (0.013 − girth2) × (mass_top5 − 49.00)
if m01 > 46.00 and n_pt_above_50 > 6.00: z += 0.931 × (m01 − 46.00) × (n_pt_above_50 − 6.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **High-pT pencil-thin quark jets** — 22.1% of jets, neuron 3.74, formula right for 64%. Mostly light quarks (55%, g 29%), nearly massless (9.3 GeV), all pT within 0.05, at above-average pT (881 GeV) with a 341 GeV leading particle. width < 0.0085 (+10.941, larger the narrower the jet), centroid_offset < 0.04 and max_dr < 0.22 add, against girth < 0.089, e2_sq < 0.0062, width < 0.0036 and girth < 0.021 (which passes almost only here). The neuron is on for all (3.739), which raises the W score even for these quark jets; the formula calls them q and is right 0.64 of the time.
- **Centred flat two-prong W/Z jets** — 20.0% of jets, neuron 4.87, formula right for 72%. Mostly W (49%) with Z (34%), 55.5 GeV, 53% of pT at 0.05-0.1 from the axis, centred with low planar flow. width < 0.0085, centroid_offset < 0.04, planar_flow < 0.27 and, for most, sum_pt_top5 < 690 add, with girth < 0.089, centroid_offset < 0.05 and log_sum_pt < 6.8 and planar_flow < 0.27 and mass < 71 subtracting less; width < 0.0036 never passes. This is the neuron's highest value (4.874), raising the W score; the formula calls them W and is right 0.716 of the time.
- **Light soft-leading gluon-led jets** — 15.1% of jets, neuron 3.30, formula right for 52%. A gluon-led (45%) mixture with quarks (22%), W (15%) and Z (13%), light (13.3 GeV), narrow, with low pT (643 GeV) and a soft leading particle (185 GeV). width < 0.0085 (+9.914), centroid_offset < 0.04 and sum_pt_top5 < 690 beat girth < 0.089, e2_sq < 0.0062, centroid_offset < 0.05 and log_sum_pt < 6.8 and width < 0.0036. The neuron is on (3.299), raising the W score; the formula calls them g but is right only 0.519 of the time.
- **Medium-mass narrowish W-led jets** — 14.9% of jets, neuron 3.67, formula right for 55%. A W-led (39%) mixture with Z (25%), gluons (16%) and quarks (11%), 38.7 GeV, fairly narrow (69% of pT within 0.05). width < 0.0085 (+6.566), centroid_offset < 0.04, planar_flow < 0.27 and sum_pt_top5 < 690 add; girth < 0.089, e2_sq < 0.0062 (always passing here), centroid_offset < 0.05 and log_sum_pt < 6.8 and planar_flow < 0.27 and mass < 71 subtract. The neuron is high (3.671), raising W; the formula calls them W but is right only 0.551 of the time.
- **Broader Z/top two-prong jets** — 13.4% of jets, neuron 1.77, formula right for 71%. Mostly Z (46%) with tops (32%) and gluons (13%), 60.3 GeV, somewhat wider than average (24% of pT beyond 0.1). centroid_offset < 0.04, planar_flow < 0.27 and sum_pt_top5 < 690 add, but only about half pass width < 0.0085, and girth > 0.076 (common here), centroid_offset < 0.05 and log_sum_pt < 6.8, centroid_offset > 0.014 and planar_flow < 0.21 and width > 0.0056 subtract. The neuron is middling (1.768, on for 0.631), raising W; the formula calls them Z and is right 0.708 of the time, the top admixture likely being the main confusion.
- **Wide top jets** — 7.8% of jets, neuron 0.03, formula right for 81%. Mostly tops (80%, g 14%), 74.9 GeV, about three times the average width with 61% of pT beyond 0.1. They fail width < 0.0085 and girth < 0.089, and girth > 0.076 (-4.901), girth > 0.089 and n_pt_above_50 < 7.1 and centroid_offset > 0.014 outweigh girth > 0.077 and n_pt_above_50 < 7, sum_pt_top5 < 690 and centroid_offset < 0.04, so the neuron is essentially off (on for 0.009). The formula calls them t and is right 0.812 of the time.
- **Very wide top jets** — 4.2% of jets, neuron 0.00, formula right for 87%. Mostly tops (87%), 76.7 GeV, nearly four times the average width with 82% of pT beyond 0.1. girth > 0.076 and girth > 0.089 and n_pt_above_50 < 7.1 (together about -14.5), girth > 0.075 and pt_7 < 40 and centroid_offset > 0.014 beat girth > 0.077 and n_pt_above_50 < 7 and sum_pt_top5 < 690, so the neuron is off for all. The formula calls them t and is right 0.866 of the time.
- **Very wide flat top/gluon jets** — 1.6% of jets, neuron 0.16, formula right for 70%. Tops (64%) and gluons (25%), heavy (92.4 GeV), very wide with 85% of pT beyond 0.1, but flat (low planar flow). All pass planar_flow < 0.23 and girth2 > 0.015 (+10.189, rare elsewhere) and planar_flow < 0.27, yet girth > 0.076, girth > 0.089 and n_pt_above_50 < 7.1 and planar_flow < 0.21 and width > 0.0056 cancel them, so the neuron is almost always zero (on for 0.066). The formula calls them t and is right 0.696 of the time, the gluon admixture likely accounting for most errors.
- **Widest low-pT top jets** — 1.0% of jets, neuron 0.00, formula right for 77%. Mostly tops (76%, g 18%), 72.5 GeV, extremely wide with 93% of pT beyond 0.1 and the lowest total pT (419 GeV). girth > 0.089 and n_pt_above_50 < 7.1 (-15.306), girth > 0.076, girth > 0.075 and pt_7 < 40 and centroid_offset > 0.014 overwhelm girth > 0.077 and n_pt_above_50 < 7 (+12.612) and sum_pt_top5 < 690, so the neuron is off for all. The formula calls them t and is right 0.772 of the time.

### neuron 13: Compactness (not a wide jet) (major)

- **What it measures:** Rises for any jet that is not wide: girth < 0.15 is by far its largest term, with λ1 < 0.015 and λ1 < 0.0062 adding; overall it falls with λ1 (-0.921), width and girth2 (-0.918) and mass/pT (-0.899). High for q (7.27), g (6.04), W (5.48) and Z (5.07), much lower for t (1.83).
- *computed — its value:* largest for q (7.27), then g (6.04), then W (5.48), then Z (5.07), then t (1.83); it separates t jets from the rest best (AUC 0.10: small for t)
- **How the class scores use it:** The t score subtracts it heavily (-48%, its largest input): tops sit far below every other type, so a compact jet is pushed away from top. The Z (+9%) and W (+8%) scores add it; it does not enter the g or q scores.
- *computed — used by:* raises the score of W (+8%), Z (+9%); lowers the score of t (-48%); does not (or hardly) enter the score of g, q (share of each class score’s average input)
- **Boundaries:** Large for any compact jet and falling step by step with λ1: 8.53 for λ1 ≤ 0.00091 with log total pT > 6.54 (23% of jets, mostly q at 53%), 6.83 at lower pT, 6.46 for λ1 0.00091-0.00251, 5.55 for 0.00251-0.00402, 4.69 for 0.00402-0.00581 and 3.84 for λ1 > 0.00581 (both with width ≤ 0.0111); it drops to 1.67 for width 0.0111-0.0162 and 0.25 for width > 0.0162 (12% of jets, 82% t). It is the main brake on the top score: -3.46 on t in the top regime and -1.56 to -2.77 in the other compact regimes, with small plus signs on W (+0.60 at most) and Z (+0.47). The regimes describe it very well (regime_r2 0.87).

Regimes (a small tree on its quantities; R² 0.87):

- `` — 22.9% of jets, value 8.53 (7.12…9.75), formula right 62%
- `` — 7.0% of jets, value 6.83 (5.50…7.88), formula right 60%
- `` — 9.9% of jets, value 6.46 (5.00…7.75), formula right 48%
- `` — 8.7% of jets, value 5.55 (4.38…6.75), formula right 55%
- `` — 14.8% of jets, value 4.69 (3.75…5.75), formula right 68%
- `` — 19.9% of jets, value 3.84 (2.88…4.75), formula right 72%
- `` — 4.7% of jets, value 1.67 (0.50…2.75), formula right 72%
- `` — 12.0% of jets, value 0.25 (0.00…0.88), formula right 84%

```
z = 0.817
if girth < 0.150: z += 64.50 × (0.150 − girth)
if lam1 < 0.015: z += 187 × (0.015 − lam1)
if width < 0.0076: z += -321 × (0.0076 − width)
if e2 < 0.049: z += -48.10 × (0.049 − e2)
if lam1 < 0.0062: z += 409 × (0.0062 − lam1)
if girth < 0.140 and log_sum_pt < 6.80: z += -42.80 × (0.140 − girth) × (6.80 − log_sum_pt)
if centroid_offset < 0.038: z += 31.60 × (0.038 − centroid_offset)
if lam1 < 0.017 and centroid_offset < 0.037: z += -2220 × (0.017 − lam1) × (0.037 − centroid_offset)
if girth < 0.150 and pt_7 < 39.00: z += -0.794 × (0.150 − girth) × (39.00 − pt_7)
if sum_pt_top5 > 640 and pt_7 < 45.00: z += 0.0005 × (sum_pt_top5 − 640) × (45.00 − pt_7)
if lam2 < 0.00031 and centroid_offset < 0.041: z += -77100 × (0.00031 − lam2) × (0.041 − centroid_offset)
if tau21 < 0.530 and max_dr > 0.013: z += -13.90 × (0.530 − tau21) × (max_dr − 0.013)
if tau21 < 0.500: z += -1.34 × (0.500 − tau21)
if z_7 < 0.027: z += -160 × (0.027 − z_7)
if pt_7 < 25.00: z += -0.139 × (25.00 − pt_7)
if sum_pt_top5 > 660 and z_7 > 0.023: z += -0.559 × (sum_pt_top5 − 660) × (z_7 − 0.023)
if C2 > 0.066: z += -52.50 × (C2 − 0.066)
if sum_pt_top5 > 830 and n_pt_above_50 > 0.880: z += 0.0038 × (sum_pt_top5 − 830) × (n_pt_above_50 − 0.880)
if e2 < 0.053 and pt_dispersion > 0.400: z += 65.00 × (0.053 − e2) × (pt_dispersion − 0.400)
if sum_pt > 980 and D2 < 4.20: z += -0.011 × (sum_pt − 980) × (4.20 − D2)
if sum_pt > 1000: z += -0.023 × (sum_pt − 1000)
if sum_pt_top5 > 900 and D2 < 4.30: z += 0.010 × (sum_pt_top5 − 900) × (4.30 − D2)
if sum_pt > 990 and n_pt_above_50 > 6.00: z += 0.028 × (sum_pt − 990) × (n_pt_above_50 − 6.00)
if width < 0.0074 and m012 > 16.00: z += -12.00 × (0.0074 − width) × (m012 − 16.00)
if lam1 < 0.007 and z_dr_0p05_0p1 > 0.180: z += 277 × (0.007 − lam1) × (z_dr_0p05_0p1 − 0.180)
if sum_pt_top5 > 900 and n_pt_above_50 > 6.10: z += 0.037 × (sum_pt_top5 − 900) × (n_pt_above_50 − 6.10)
if sum_pt_top5 < 520 and pt_5 < 30.00: z += 0.0015 × (520 − sum_pt_top5) × (30.00 − pt_5)
if sum_pt_top5 > 650 and tau32 < 0.370: z += -0.040 × (sum_pt_top5 − 650) × (0.370 − tau32)
if sum_pt < 760 and z_4 < 0.037: z += 7.44 × (760 − sum_pt) × (0.037 − z_4)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Two-prong Z/W/top jets** — 26.2% of jets, neuron 3.78, formula right for 72%. The largest group (26%): Z (39%), W (27%) and t (21%) at 57.1 GeV, pT mostly 0.05-0.1 from the axis, somewhat wider than average. girth < 0.15 (+4.526), lam1 < 0.015 and centroid_offset < 0.038 add, while girth < 0.14 and log_sum_pt < 6.8, tau21 < 0.53 and max_dr > 0.013, lam1 < 0.017 and centroid_offset < 0.037 and tau21 < 0.5 subtract; lam1 < 0.0062 mostly fails. The neuron is moderate (3.783), raising W and Z and lowering t; the formula calls them Z and is right 0.717 of the time.
- **Compact medium-mass W-led jets** — 19.7% of jets, neuron 5.14, formula right for 60%. W-led (41%) mixture with Z (28%) and gluons (13%), 43.9 GeV, fairly narrow (64% of pT within 0.05). girth < 0.15 (+6.211), lam1 < 0.015 and lam1 < 0.0062 add, against e2 < 0.049, width < 0.0076 and girth < 0.14 and log_sum_pt < 6.8. The neuron is high (5.136), raising W and Z and lowering t; the formula calls them W but is right only 0.602 of the time.
- **Light soft-leading gluon jets** — 16.5% of jets, neuron 6.83, formula right for 53%. Mostly gluons (48%) with quarks (23%) and some W/Z, light (12.6 GeV), narrow, low pT (626 GeV) spread over many particles (leading 170 GeV). girth < 0.15 (+8.028), lam1 < 0.015 and lam1 < 0.0062 add, while width < 0.0076, e2 < 0.049, girth < 0.14 and log_sum_pt < 6.8 and lam1 < 0.017 and centroid_offset < 0.037 subtract. The neuron is high (6.834), raising W and Z and lowering t; the formula calls them g but is right only 0.526 of the time.
- **Pencil-thin harder quark jets** — 15.9% of jets, neuron 8.48, formula right for 59%. Mostly light quarks (49%, g 29%), nearly massless (10.0 GeV), all pT within 0.05, above-average pT (861 GeV) with a 304 GeV leading particle. girth < 0.15 (+8.889), lam1 < 0.015, lam1 < 0.0062, centroid_offset < 0.038 and (for most) sum_pt_top5 > 640 and pt_7 < 45 add, while width < 0.0076, e2 < 0.049 and lam1 < 0.017 and centroid_offset < 0.037 subtract. The neuron is very high (8.476), strongly lowering t and raising W and Z; the formula calls them q and is right 0.593 of the time.
- **Wide top jets** — 14.1% of jets, neuron 0.43, formula right for 82%. Mostly tops (81%, g 14%), 79.4 GeV, 3.6 times the average width with 73% of pT beyond 0.1. Too wide for width < 0.0076 and lam1 < 0.0062 and mostly lam1 < 0.015, so only girth < 0.15 (passing 0.679) adds much, against C2 > 0.066, tau21 < 0.53 and max_dr > 0.013 and tau21 < 0.5. The neuron is low (0.426, on for 0.432), so it does not hold back the t score; the formula calls them t and is right 0.82 of the time.
- **One-particle high-pT quark jets** — 6.5% of jets, neuron 7.78, formula right for 74%. Mostly light quarks (62%) with W (17%) and Z (14%), 21.5 GeV, very narrow, high pT (969 GeV) with a 509 GeV leading particle and a very soft 8th particle (13.3 GeV). girth < 0.15 (+8.714) and sum_pt_top5 > 640 and pt_7 < 45 (+4.293), plus lam1 < 0.015 and lam1 < 0.0062, beat girth < 0.15 and pt_7 < 39, width < 0.0076 and the soft-tail tests z_7 < 0.027 and pt_7 < 25 (which pass almost only here). The neuron is high (7.778), lowering t and raising W and Z; the formula calls them q and is right 0.741 of the time.
- **Very high-pT light mixed jets** — 0.5% of jets, neuron 7.83, formula right for 54%. A tiny group (0.5%): gluons (38%), quarks (30%), Z (17%) and W (13%) at very high pT (1230 GeV) with a 625 GeV leading particle, 33.2 GeV and narrow. girth < 0.15 and the high-pT pluses sum_pt_top5 > 900 and D2 < 4.3, sum_pt_top5 > 640 and pt_7 < 45 and sum_pt_top5 > 830 and n_pt_above_50 > 0.88 outweigh sum_pt > 980 and D2 < 4.2 (-5.97) and sum_pt > 1e+03, all rarely passed elsewhere. The neuron is high (7.835); the formula calls them g but is right only 0.538 of the time, a hard mixed group.
- **Very high-pT many-particle gluon jets** — 0.4% of jets, neuron 7.41, formula right for 68%. A tiny group (0.4%), mostly gluons (64%), at very high pT (1158 GeV) with pT shared by many hard particles (8th at 57.3 GeV). sum_pt > 990 and n_pt_above_50 > 6 (+7.531, almost only here and in the next group), girth < 0.15 and sum_pt_top5 > 830 and n_pt_above_50 > 0.88 beat sum_pt > 980 and D2 < 4.2, sum_pt_top5 > 660 and z_7 > 0.023 and sum_pt > 1e+03. The neuron is high (7.408); the formula calls them g and is right 0.68 of the time.
- **Highest-pT many-particle gluon jets** — 0.1% of jets, neuron 9.01, formula right for 66%. The smallest group (0.12%), mostly gluons (66%), at the highest pT (1401 GeV) with many hard particles (8th at 60.1 GeV). Huge pluses sum_pt > 990 and n_pt_above_50 > 6 (+19.165) and sum_pt_top5 > 900 and n_pt_above_50 > 6.1 (+16.065), with sum_pt_top5 > 830 and n_pt_above_50 > 0.88, sum_pt_top5 > 900 and D2 < 4.3 and girth < 0.15, outweigh sum_pt > 980 and D2 < 4.2 and sum_pt > 1e+03. This is the neuron's highest value (9.007), lowering t and raising W and Z; the formula calls them g and is right 0.657 of the time.

### neuron 14: Heavier, wider two-prong (Z-like) (major)

- **What it measures:** Rises in an intermediate-width window (girth2 < 0.013 and λ1 > 0.0042 push it up; girth < 0.087 and width < 0.0061 push it down) and grows with mass (+0.437) and mass/pT (+0.417). Largest for Z (1.50), then t (0.53), and low for g (0.20), W (0.18) and q (0.15).
- *computed — its value:* largest for Z (1.50), then t (0.53), then g (0.20), then W (0.18), then q (0.15); it separates Z jets from the rest best (AUC 0.77: large for Z)
- **How the class scores use it:** The Z score adds it (+6%) and the W score subtracts it (-9%): Z jets sit high and W jets low, so it separates W from Z directly. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of Z (+6%); lowers the score of W (-9%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Highest (2.11) for jets with larger λ1 (> 0.00663) that are not broad (girth2 ≤ 0.0111), 13% of jets and mostly Z (59%); 0.735 for 0.00387 < λ1 ≤ 0.00663 (23%, mostly W at 49%), about 0.5-0.6 for broader jets with centred centroid or thin jets with max ΔR > 0.2, and off (0.003) for thin jets with max ΔR ≤ 0.153 (42% of jets). It separates Z from W: -1.59 on W and +0.79 on Z in the top regime, -0.55 / +0.28 in the next, and nothing to g, q or t. The regimes describe it only moderately (regime_r2 0.48).

Regimes (a small tree on its quantities; R² 0.478):

- `` — 12.6% of jets, value 2.11 (0.25…3.56), formula right 72%
- `` — 23.2% of jets, value 0.73 (0.00…2.44), formula right 69%
- `` — 2.8% of jets, value 0.60 (0.00…1.88), formula right 61%
- `` — 9.8% of jets, value 0.53 (0.00…1.25), formula right 81%
- `` — 2.7% of jets, value 0.20 (0.00…0.81), formula right 58%
- `` — 6.9% of jets, value 0.04 (0.00…0.00), formula right 80%
- `` — 42.0% of jets, value 0.00 (0.00…0.00), formula right 57%

```
z = 0.032
if girth2 < 0.013: z += 487 × (0.013 − girth2)
if lam1 > 0.0042: z += 1080 × (lam1 − 0.0042)
if girth < 0.087: z += -89.10 × (0.087 − girth)
if width < 0.0061: z += -1210 × (0.0061 − width)
if lam1 > 0.0025: z += -542 × (lam1 − 0.0025)
if e2 < 0.043: z += 119 × (0.043 − e2)
if mass_over_sum_pt_sq < 0.0081: z += -443 × (0.0081 − mass_over_sum_pt_sq)
if e2_sq < 0.0058: z += 700 × (0.0058 − e2_sq)
if lam1 > 0.0025 and D2 > 0.400: z += 760 × (lam1 − 0.0025) × (D2 − 0.400)
if lam1 > 0.0042 and D2 > 0.410: z += -857 × (lam1 − 0.0042) × (D2 − 0.410)
if lam1 > 0.0061: z += -478 × (lam1 − 0.0061)
if C2 < 0.067: z += -24.20 × (0.067 − C2)
if max_dr < 0.180: z += -12.50 × (0.180 − max_dr)
if girth2 < 0.013 and n_dr_0p1_0p2 < 3.00: z += -45.70 × (0.013 − girth2) × (3.00 − n_dr_0p1_0p2)
if width < 0.0077 and n_dr_0p1_0p2 < 3.00: z += 87.00 × (0.0077 − width) × (3.00 − n_dr_0p1_0p2)
if z_dr_0p05_0p1 < 0.580 and C2 < 0.068: z += -44.30 × (0.580 − z_dr_0p05_0p1) × (0.068 − C2)
if girth2 < 0.013 and eccentricity > 0.970: z += 11300 × (0.013 − girth2) × (eccentricity − 0.970)
if girth2 < 0.0043 and n_dr_0p2_0p4 < 1.10: z += -384 × (0.0043 − girth2) × (1.10 − n_dr_0p2_0p4)
if girth < 0.034: z += -93.40 × (0.034 − girth)
if z_dr_0p05_0p1 < 0.600: z += 1.38 × (0.600 − z_dr_0p05_0p1)
if width < 0.0076 and D2 < 1.00: z += -1950 × (0.0076 − width) × (1.00 − D2)
if width < 0.0075 and planar_flow < 0.110: z += -6520 × (0.0075 − width) × (0.110 − planar_flow)
if planar_flow < 0.110 and centroid_offset > 0.0094: z += 1340 × (0.110 − planar_flow) × (centroid_offset − 0.0094)
if n_dr_0p05_0p1 < 4.80: z += -0.122 × (4.80 − n_dr_0p05_0p1)
if z_dr_0p05_0p1 < 0.600 and n_dr_0p1_0p2 < 3.00: z += 0.317 × (0.600 − z_dr_0p05_0p1) × (3.00 − n_dr_0p1_0p2)
if planar_flow < 0.110 and centroid_offset > 0.018: z += -1670 × (0.110 − planar_flow) × (centroid_offset − 0.018)
if centroid_offset > 0.050: z += -296 × (centroid_offset − 0.050)
if D2 < 0.800: z += -2.01 × (0.800 − D2)
if e2 < 0.041 and D2 < 0.990: z += 304 × (0.041 − e2) × (0.990 − D2)
if planar_flow < 0.110 and max_dr < 0.160: z += -202 × (0.110 − planar_flow) × (0.160 − max_dr)
if centroid_offset > 0.031: z += -81.40 × (centroid_offset − 0.031)
if D2 < 1.20 and centroid_offset < 0.030: z += 41.50 × (1.20 − D2) × (0.030 − centroid_offset)
if tau21 < 0.140: z += 9.90 × (0.140 − tau21)
if lam1 > 0.0054 and max_dr < 0.160: z += 9530 × (lam1 − 0.0054) × (0.160 − max_dr)
if lam1 > 0.0025 and D2 > 1.60: z += -503 × (lam1 − 0.0025) × (D2 − 1.60)
if planar_flow < 0.110 and sum_pt < 750: z += -0.048 × (0.110 − planar_flow) × (750 − sum_pt)
if girth2 < 0.013 and mass_top3 > 24.00: z += 5.76 × (0.013 − girth2) × (mass_top3 − 24.00)
if sum_pt_top3 < 310: z += -0.0077 × (310 − sum_pt_top3)
if max_dr < 0.078: z += 6.00 × (0.078 − max_dr)
if z_dr_0p05_0p1 > 0.740 and n_dr_0p2_0p4 < 1.00: z += 2.71 × (z_dr_0p05_0p1 − 0.740) × (1.00 − n_dr_0p2_0p4)
if width < 0.0074 and mass_top3 > 24.00: z += -18.50 × (0.0074 − width) × (mass_top3 − 24.00)
if width < 0.006 and mean_phi < -0.026: z += 35800 × (0.006 − width) × (-0.026 − mean_phi)
if width < 0.006 and mean_eta < -0.027: z += 35400 × (0.006 − width) × (-0.027 − mean_eta)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 32.2% of jets, neuron 0.00, formula right for 61%. The largest group (32%), mostly light quarks (45%) with gluons (35%), nearly massless (9.3 GeV) with 99% of pT within 0.05. The narrow-jet minuses width < 0.0061 (-6.96), girth < 0.087, mass_over_sum_pt_sq < 0.0081, girth < 0.034 and girth2 < 0.0043 and n_dr_0p2_0p4 < 1.1 beat the pluses girth2 < 0.013, e2 < 0.043, e2_sq < 0.0058 and width < 0.0077 and n_dr_0p1_0p2 < 3, and lam1 > 0.0025 never passes, so the neuron is off and adds nothing. The formula calls them q and is right only 0.608 of the time.
- **Two-prong W-rich jets** — 21.0% of jets, neuron 0.71, formula right for 69%. Mostly W (50%) with Z (29%), 51.7 GeV, pT split between the core and 0.05-0.1. girth2 < 0.013 (+3.681), girth2 < 0.013 and eccentricity > 0.97, e2 < 0.043 and e2 < 0.041 and D2 < 0.99 add, while girth < 0.087, lam1 > 0.0025, mass_over_sum_pt_sq < 0.0081, width < 0.0076 and D2 < 1 and width < 0.0075 and planar_flow < 0.11 (both far more common here) subtract. The neuron is on for 0.434 (0.712), lowering the W score and raising Z, which works against these W jets; still the formula calls them W and is right 0.687 of the time.
- **Compact medium-mass mixed jets** — 16.0% of jets, neuron 0.25, formula right for 52%. A mixture (W 31%, g 23%, Z 22%, q 16%) at 32.4 GeV, narrow (79% of pT within 0.05). girth2 < 0.013 (+4.999), e2 < 0.043 and e2_sq < 0.0058 are cancelled by width < 0.0061, girth < 0.087, mass_over_sum_pt_sq < 0.0081 and girth2 < 0.0043 and n_dr_0p2_0p4 < 1.1, and lam1 > 0.0042 almost never passes. The neuron is usually zero (on for 0.207); the formula calls them W but is right only 0.522 of the time.
- **Wider two-prong Z jets** — 14.0% of jets, neuron 1.90, formula right for 73%. Mostly Z (56%) with tops (22%) and W (10%), 59.5 GeV, somewhat wider than average with 64% of pT at 0.05-0.1. Here the intermediate-width window opens: lam1 > 0.0042 (+3.964), girth2 < 0.013, girth2 < 0.013 and eccentricity > 0.97 and lam1 > 0.0054 and max_dr < 0.16 add, against lam1 > 0.0025, C2 < 0.067 and lam1 > 0.0061, while width < 0.0061 never passes. This is the neuron's highest value (1.896, on for 0.883), raising Z and lowering W; the formula calls them Z and is right 0.726 of the time.
- **Wide top jets** — 4.4% of jets, neuron 0.10, formula right for 77%. Mostly tops (75%, g 17%), 74.5 GeV, 2.7 times the average width with 66% of pT beyond 0.1. Too wide for girth2 < 0.013 and girth < 0.087, they are left with the lam1 ladder: lam1 > 0.0042 (+13.467) against lam1 > 0.0025 and lam1 > 0.0061, nearly cancelling, plus small terms like planar_flow < 0.11 and centroid_offset > 0.0094 and C2 < 0.067. The neuron is usually zero (on for 0.196); the formula calls them t and is right 0.766 of the time.
- **Wide heavy top jets** — 4.0% of jets, neuron 0.57, formula right for 90%. Almost all tops (91%), 82.8 GeV, nearly four times the average width with 75% of pT beyond 0.1. Two opposing pairs dominate, lam1 > 0.0042 (+17.249) against lam1 > 0.0025 and lam1 > 0.0061, and lam1 > 0.0025 and D2 > 0.4 against lam1 > 0.0042 and D2 > 0.41, with centroid_offset > 0.031 taking a little; the remainder is small (0.567, on for 0.669). It slightly lowers W and raises Z; the formula calls them t and is right 0.904 of the time.
- **Moderately wide top/gluon jets** — 3.8% of jets, neuron 0.59, formula right for 70%. Tops (67%) and gluons (22%), 63.2 GeV, 2.3 times the average width with 40% of pT beyond 0.1. The same lam1 and lam1-with-D2 pairs nearly cancel (lam1 > 0.0042 +7.956, lam1 > 0.0025 and D2 > 0.4 +7.684 against lam1 > 0.0042 and D2 > 0.41, lam1 > 0.0025 and lam1 > 0.0061), and centroid_offset > 0.05 and centroid_offset > 0.031 pull the rest down for off-centre jets. The neuron is on for about half (0.595); the formula calls them t and is right 0.705 of the time.
- **Very wide heavy top/gluon jets** — 1.9% of jets, neuron 0.23, formula right for 72%. Tops (66%) and gluons (24%), heavy (91.8 GeV), nearly five times the average width with 91% of pT beyond 0.1. lam1 > 0.0042 (+26.96) is nearly cancelled by lam1 > 0.0025 and lam1 > 0.0061, the D2 pair cancels, and planar_flow < 0.11 and centroid_offset > 0.0094 is offset by the same with centroid_offset > 0.018. The neuron is usually zero (on for 0.28); the formula calls them t and is right 0.72 of the time, the gluon admixture likely accounting for most errors.
- **Widest soft-leading top jets** — 1.4% of jets, neuron 0.44, formula right for 83%. Mostly tops (83%, g 14%), 86.2 GeV, over five times the average width, with a soft leading particle (124 GeV). The largest opposing terms of this neuron (lam1 > 0.0042 +28.23 and lam1 > 0.0025 and D2 > 0.4 against lam1 > 0.0025, lam1 > 0.0042 and D2 > 0.41 and lam1 > 0.0061) leave a small remainder that centroid_offset > 0.05 and centroid_offset > 0.031 often cancel. The neuron is on for half (0.442); the formula calls them t and is right 0.832 of the time.
- **Top jets with large D2** — 1.4% of jets, neuron 0.08, formula right for 84%. Mostly tops (84%, g 10%), 72.5 GeV, with pT spread evenly over core (27%), 0.05-0.1 (34%) and beyond 0.1 (39%). Besides the cancelling lam1 and lam1-with-D2 pairs, they almost all pass lam1 > 0.0025 and D2 > 1.6 (-3.393), which rarely fires elsewhere, and a third pass centroid_offset > 0.05, so the neuron is usually zero (on for 0.116). The formula calls them t and is right 0.844 of the time.

### neuron 0: Elongated two-prong massive jet (moderate)

- **What it measures:** Rises for jets that are compact but not the very narrowest (girth2 < 0.013 pushes it up, girth < 0.078 down) and is pushed down for light jets (mass < 59, < 30 and < 22 GeV); overall it follows an elongated two-prong shape (eccentricity +0.591, planar flow -0.591, τ21 -0.571). Largest for Z (2.15) and W (2.03), low for t (0.60), g (0.29) and q (0.22).
- *computed — its value:* largest for Z (2.15), then W (2.03), then t (0.60), then g (0.29), then q (0.22); it separates Z jets from the rest best (AUC 0.75: large for Z)
- **How the class scores use it:** The W score adds it (+8%) and the g score subtracts it (-6%): a high value marks a massive two-prong jet, which argues for a W and against a gluon. It does not enter the q, Z or t scores, even though Z jets sit highest on it.
- *computed — used by:* raises the score of W (+8%); lowers the score of g (-6%); does not (or hardly) enter the score of q, Z, t (share of each class score’s average input)
- **Boundaries:** High (mean 2.95, on for 96%) for flat, elongated jets (planar flow ≤ 0.0587) of intermediate width 0.00298 < width ≤ 0.00955, which are 25% of jets and mostly Z (43%, with 40% W); it is moderate (1.59) for less flat jets with λ1 > 0.00248 and mass/pT² ≤ 0.00846 (16%), and essentially off (0.003) for non-flat, very thin light jets (planar flow > 0.0587, λ1 ≤ 0.00248, mass ≤ 28.4 GeV; 35% of jets, mostly q at 42%) as well as low for wide flat jets (width > 0.0142: 0.24, 67% t). What matters is the top regime, where it adds +1.01 to the W score and -0.46 to the g score (+0.55 and -0.25 in the second regime); it gives nothing to q or Z and only a few hundredths to t. The regimes describe it fairly well (regime_r2 0.73).

Regimes (a small tree on its quantities; R² 0.73):

- `` — 25.1% of jets, value 2.95 (1.50…4.12), formula right 74%
- `` — 16.1% of jets, value 1.59 (0.00…2.75), formula right 57%
- `` — 2.0% of jets, value 0.86 (0.00…1.88), formula right 65%
- `` — 3.7% of jets, value 0.57 (0.00…2.38), formula right 50%
- `` — 2.1% of jets, value 0.29 (0.00…1.12), formula right 49%
- `` — 3.5% of jets, value 0.24 (0.00…0.88), formula right 71%
- `` — 12.6% of jets, value 0.13 (0.00…0.38), formula right 83%
- `` — 34.7% of jets, value 0.00 (0.00…0.00), formula right 60%

```
z = -0.511
if girth2 < 0.013: z += 299 × (0.013 − girth2)
if girth < 0.078: z += -75.60 × (0.078 − girth)
if width < 0.0089: z += 330 × (0.0089 − width)
if width < 0.0044: z += -815 × (0.0044 − width)
if mass < 59.00: z += -0.051 × (59.00 − mass)
if mass < 22.00: z += -0.252 × (22.00 − mass)
if lam1 < 0.0006: z += 9390 × (0.0006 − lam1)
if mass < 30.00: z += -0.132 × (30.00 − mass)
if lam1 < 0.0015: z += -1870 × (0.0015 − lam1)
if z_dr_0_0p05 > 0.850: z += 11.50 × (z_dr_0_0p05 − 0.850)
if girth2_top5 < 0.0085: z += -127 × (0.0085 − girth2_top5)
if centroid_offset < 0.033: z += 30.50 × (0.033 − centroid_offset)
if mass < 30.00 and phi_1 > -0.056: z += -1.13 × (30.00 − mass) × (phi_1 − -0.056)
if girth2 < 0.019 and eccentricity > 0.960: z += 2570 × (0.019 − girth2) × (eccentricity − 0.960)
if mass_over_sum_pt_sq < 0.0081 and n_pt_above_50 < 7.90: z += 37.60 × (0.0081 − mass_over_sum_pt_sq) × (7.90 − n_pt_above_50)
if sum_pt > 810: z += -0.013 × (sum_pt − 810)
if mass < 64.00 and pt_7 < 40.00: z += -0.0017 × (64.00 − mass) × (40.00 − pt_7)
if sum_pt_top5 > 700: z += 0.011 × (sum_pt_top5 − 700)
if tau32 > 0.440: z += -1.89 × (tau32 − 0.440)
if sum_pt > 900: z += -0.025 × (sum_pt − 900)
if n_dr_0_0p05 > 3.80: z += 0.172 × (n_dr_0_0p05 − 3.80)
if mass < 63.00 and centroid_offset > 0.012: z += 1.83 × (63.00 − mass) × (centroid_offset − 0.012)
if planar_flow < 0.140 and centroid_offset < 0.051: z += 135 × (0.140 − planar_flow) × (0.051 − centroid_offset)
if sum_pt > 900 and pt_7 < 26.00: z += 0.0024 × (sum_pt − 900) × (26.00 − pt_7)
if lam1 < 0.0065 and D2 < 0.880: z += -1970 × (0.0065 − lam1) × (0.880 − D2)
if mass < 56.00 and C2 > 0.024: z += 1.57 × (56.00 − mass) × (C2 − 0.024)
if planar_flow < 0.150 and sum_pt_top2 < 370: z += -0.038 × (0.150 − planar_flow) × (370 − sum_pt_top2)
if sum_pt > 900 and pt_7 > 26.00: z += -0.0009 × (sum_pt − 900) × (pt_7 − 26.00)
if D2 > 3.90: z += -0.707 × (D2 − 3.90)
if planar_flow < 0.180 and z_top5 < 0.820: z += 43.50 × (0.180 − planar_flow) × (0.820 − z_top5)
if girth2 < 0.019 and mass_top2 > 29.00: z += -8.40 × (0.019 − girth2) × (mass_top2 − 29.00)
if planar_flow < 0.013: z += 81.10 × (0.013 − planar_flow)
if girth < 0.087 and m01 > 29.00: z += 7.56 × (0.087 − girth) × (m01 − 29.00)
if lam1 < 0.0053 and mass_top3 > 15.00: z += -29.10 × (0.0053 − lam1) × (mass_top3 − 15.00)
if mass < 29.00 and D2 < 0.870: z += -1.28 × (29.00 − mass) × (0.870 − D2)
if planar_flow < 0.150 and dr_2 < 0.028: z += -380 × (0.150 − planar_flow) × (0.028 − dr_2)
if D2 > 3.90 and phi_2 < -0.010: z += -126 × (D2 − 3.90) × (-0.010 − phi_2)
if mass < 30.00 and dr_7 > 0.130: z += -4.44 × (30.00 − mass) × (dr_7 − 0.130)
if girth2_top3 < 0.0039 and m01 > 29.00: z += 632 × (0.0039 − girth2_top3) × (m01 − 29.00)
if z_dr_0p05_0p1 > 0.840 and dr_7 < 0.0073: z += -121000 × (z_dr_0p05_0p1 − 0.840) × (0.0073 − dr_7)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Two-prong W/Z boson jets** — 28.3% of jets, neuron 2.72, formula right for 72%. The largest group (28%), a W/Z mixture (Z 42%, W 38%, some t) with mass 54.5 GeV, well above the 40 GeV average, normal width and pT, and most pT sitting 0.05-0.1 from the axis, i.e. in two separated prongs. Almost every jet passes girth2 < 0.013 (+2.015) and width < 0.0089 (+0.871), and most pass girth2 < 0.019 and eccentricity > 0.96 and centroid_offset < 0.033, while the light-mass penalties (mass < 30, width < 0.0044) almost never fire; only girth < 0.078 and mass < 59 take a little back. The neuron sits at its highest value (2.718), which raises the W score and lowers the g score; the formula splits them between W and Z and is right 0.716 of the time, W and Z being hard to tell apart here.
- **Wide massive top-like jets** — 19.8% of jets, neuron 0.33, formula right for 77%. Mostly top jets (74%, plus some gluons) with mass 73.5 GeV, three times the average width and a softer leading particle; 60% of the pT lies beyond 0.1 from the axis, so the radiation is spread out. These jets are too wide to pass width < 0.0089, girth < 0.078 or (mostly) girth2 < 0.013, so the big positive terms are lost; what remains is small pluses such as centroid_offset < 0.033 and mass < 63 and centroid_offset > 0.012 against minuses such as tau32 > 0.44 and planar_flow < 0.15 and sum_pt_top2 < 370. The neuron is only weakly on (0.332, on for 0.372 of jets), so it barely touches the scores; the formula calls them t and is right 0.772 of the time.
- **Very light pencil-thin quark/gluon jets** — 16.3% of jets, neuron 0.00, formula right for 62%. Light-quark and gluon jets (q 47%, g 38%) with almost no mass (6.7 GeV) and essentially all pT within 0.05 of the axis. Every compactness and light-mass test passes: the pluses girth2 < 0.013, lam1 < 0.0006 and width < 0.0089 are outweighed by the minuses girth < 0.078 (-5.124), mass < 22, width < 0.0044 and mass < 30, so the sum is negative and the neuron is off for all of them and adds nothing to any score. The formula splits them between q and g and is right only 0.615 of the time: the quark-gluon confusion is decided elsewhere.
- **Compact medium-mass mixed jets** — 13.7% of jets, neuron 1.37, formula right for 53%. A mixture (W 34%, Z 23%, g 19%, q 14%) of jets at 36.9 GeV, a bit below average mass, narrow (78% of pT within 0.05) but not pencil-thin. The large terms nearly cancel: girth2 < 0.013 (+2.926) and width < 0.0089 against girth < 0.078 (-2.519), mass < 59 and, for most jets, width < 0.0044; what tips it positive are medium-mass tests such as mass < 63 and centroid_offset > 0.012, mass < 56 and C2 > 0.024 and n_dr_0_0p05 > 3.8, which pass far more often here than in other groups. The neuron is on for 0.66 of jets (mean 1.368), raising W and lowering g; the formula calls them W but is right only 0.531 of the time, a poorly separated group.
- **Light narrow gluon/quark jets** — 7.4% of jets, neuron 0.00, formula right for 55%. Gluon and light-quark jets (g 42%, q 36%, a few W/Z) of 17.7 GeV mass, narrow with 94% of pT within 0.05 of the axis, at average pT. As for the lightest group, the compactness pluses (girth2 < 0.013, width < 0.0089) are beaten by girth < 0.078, width < 0.0044, mass < 59, mass < 30 and lam1 < 0.0015, all of which almost always pass, so the neuron is off and contributes nothing. The formula calls them g and is right only 0.55 of the time, so these are often mistaken quarks (or light bosons).
- **Near-massless but spread-out jets** — 5.5% of jets, neuron 0.01, formula right for 46%. A real mixture (g 35%, Z 23%, W 21%, q 13%) with very low mass (6.9 GeV), a softer leading particle (192 GeV) and more pT at 0.05-0.1 than the pencil-thin groups. All three light-mass penalties mass < 22, mass < 30 and mass < 59 pass (together about -9.5) along with girth < 0.078 and width < 0.0044; mass < 63 and centroid_offset > 0.012 almost always passes (+2.083) but cannot compensate, so the neuron is essentially off. The formula calls them g but is right only 0.457 of the time, one of the worst groups here: bosons whose reconstructed mass came out tiny get lost.
- **High-pT pencil-like quark jets** — 4.8% of jets, neuron 0.00, formula right for 72%. Mostly light-quark jets (66%) with high total pT (1003 GeV vs 716 average), a leading particle carrying 458 GeV and almost no mass or width. Beyond the usual narrow-jet cancellation (girth < 0.078 -5.396 vs lam1 < 0.0006 +4.914), the high-pT tests sum_pt > 900 and sum_pt > 810 subtract about 5 while sum_pt_top5 > 700 adds back only 2.243, so the neuron is off. The formula calls them q and is right 0.724 of the time.
- **High-pT boosted W/Z jets** — 2.8% of jets, neuron 1.51, formula right for 67%. Boosted bosons (W 40%, Z 35%) at very high total pT (1006 GeV), mass 62.2 GeV and a hard leading particle (489 GeV), still fairly compact. The high-pT penalties sum_pt > 900 and sum_pt > 810 (about -5) are offset by sum_pt_top5 > 700 and, for 0.65 of jets, sum_pt > 900 and pt_7 < 26 (+1.208), while girth2 < 0.013 and width < 0.0089 give the usual boost. The neuron is on for about half (mean 1.507), raising W and lowering g; the formula calls them W and is right 0.673 of the time.
- **One-particle-dominated high-pT quark jets** — 0.8% of jets, neuron 0.11, formula right for 76%. A tiny group (0.8%), mostly light quarks (77%) at high pT (1081 GeV) where the leading particle carries 624 GeV and the 8th only 7.5 GeV. sum_pt > 900 and pt_7 < 26 adds a huge +7.403 but is cancelled by D2 > 3.9 (-4.699), sum_pt > 900, sum_pt > 810 and girth < 0.078, so the neuron is almost always off (on for 0.036). The formula calls them q and is right 0.755 of the time.
- **Very high-pT many-particle gluon jets** — 0.4% of jets, neuron 0.00, formula right for 68%. The smallest group (0.4%), mostly gluons (70%) at the highest pT (1308 GeV) with all eight particles hard (8th at 58.7 GeV), light and narrow. sum_pt > 900 and pt_7 > 26 (-11.108), sum_pt > 900 (-10.29) and sum_pt > 810 dwarf everything else, so the neuron is off. The formula calls them g and is right 0.683 of the time.

### neuron 5: Quark-likeness: pT in few particles (moderate)

- **What it measures:** Rises when particle 7 (the 8th hardest) is soft (pT_7 < 53 GeV) and the two hardest particles carry much of the pT (their sum < 540 GeV pushes it down), for narrow, centred, low-e2 jets (e2 < 0.034 pushes it up); overall it falls with e2 (-0.683) and mass/pT (-0.633). Largest for q (4.65), well above W (1.68), Z (1.64) and g (1.57), and lowest for t (0.50).
- *computed — its value:* largest for q (4.65), then W (1.68), then Z (1.64), then g (1.57), then t (0.50); it separates q jets from the rest best (AUC 0.78: large for q)
- **How the class scores use it:** The g (-13%) and t (-11%) scores subtract it and the q score adds a little (+4%): a high value is quark-like, which argues against a gluon and against a top (tops sit lowest). It does not enter the W or Z scores.
- *computed — used by:* raises the score of q (+4%); lowers the score of g (-13%), t (-11%); does not (or hardly) enter the score of W, Z (share of each class score’s average input)
- **Boundaries:** Highest (9.85) for jets with pT in few particles: low LHA (≤ 0.103), high total pT (log > 6.71) and a soft 8th particle (pT_7 ≤ 17.9 GeV), 3.2% of jets and 83% q; 6.56 for the same with pT_7 > 17.9 GeV (6.4%, 63% q) and 3.96 for LHA > 0.103 with e2 ≤ 0.0247 and top-5 pT sum > 618 GeV (18%). It is low (0.18) for LHA > 0.103, e2 > 0.0247 and top-2 pT sum ≤ 387 GeV (38% of jets, mostly t at 41%). Mostly it works by subtraction: -1.85 on g and -2.46 on t in the top regime (-0.74 and -0.99 in the 18% regime), with small plus signs on q (+0.46 at most) and Z (+0.15); the regimes describe it well (regime_r2 0.725).

Regimes (a small tree on its quantities; R² 0.725):

- `` — 3.2% of jets, value 9.85 (6.75…12.62), formula right 83%
- `` — 6.4% of jets, value 6.56 (3.50…9.38), formula right 68%
- `` — 18.0% of jets, value 3.96 (1.12…6.50), formula right 58%
- `` — 3.0% of jets, value 2.41 (0.00…4.75), formula right 57%
- `` — 19.3% of jets, value 1.31 (0.00…3.00), formula right 52%
- `` — 12.4% of jets, value 1.17 (0.00…3.00), formula right 77%
- `` — 37.8% of jets, value 0.18 (0.00…0.75), formula right 71%

```
z = 0.107
if pt_7 < 53.00: z += 0.074 × (53.00 − pt_7)
if sum_pt_top2 < 540: z += -0.0066 × (540 − sum_pt_top2)
if z_7 < 0.071 and centroid_offset < 0.031: z += -2300 × (0.071 − z_7) × (0.031 − centroid_offset)
if width < 0.0026 and centroid_offset < 0.025: z += 69700 × (0.0026 − width) × (0.025 − centroid_offset)
if dr_0 < 0.023: z += -216 × (0.023 − dr_0)
if e2 < 0.034: z += 65.70 × (0.034 − e2)
if mean_phi2 < 0.015 and max_pair_mass < 41.00: z += 1.83 × (0.015 − mean_phi2) × (41.00 − max_pair_mass)
if log_sum_pt > 6.60 and n_dr_0p2_0p4 < 0.990: z += -10.60 × (log_sum_pt − 6.60) × (0.990 − n_dr_0p2_0p4)
if log_sum_pt > 6.60: z += 7.85 × (log_sum_pt − 6.60)
if mass < 57.00 and n_dr_0p2_0p4 < 2.00: z += 0.012 × (57.00 − mass) × (2.00 − n_dr_0p2_0p4)
if log_sum_pt > 6.60 and dr_0 < 0.021: z += 792 × (log_sum_pt − 6.60) × (0.021 − dr_0)
if log_sum_pt > 6.80: z += -32.70 × (log_sum_pt − 6.80)
if z_7 < 0.070 and n_dr_0p2_0p4 < 1.00: z += 25.80 × (0.070 − z_7) × (1.00 − n_dr_0p2_0p4)
if log_sum_pt > 6.30 and centroid_offset > 0.00063: z += 137 × (log_sum_pt − 6.30) × (centroid_offset − 0.00063)
if sum_pt_top2 < 570 and girth2_top3 < 0.0036: z += -1.43 × (570 − sum_pt_top2) × (0.0036 − girth2_top3)
if z_7 < 0.073 and sum_pt < 790: z += -0.289 × (0.073 − z_7) × (790 − sum_pt)
if sum_pt > 870 and centroid_offset < 0.013: z += 1.93 × (sum_pt − 870) × (0.013 − centroid_offset)
if LHA < 0.220 and log_sum_pt < 6.80: z += -62.50 × (0.220 − LHA) × (6.80 − log_sum_pt)
if z_7 < 0.024: z += 248 × (0.024 − z_7)
if log_sum_pt > 6.90 and dr_0 < 0.043: z += -1630 × (log_sum_pt − 6.90) × (0.043 − dr_0)
if sum_pt_top2 < 550 and dr_0 < 0.022: z += 0.401 × (550 − sum_pt_top2) × (0.022 − dr_0)
if log_sum_pt > 6.60 and mean_phi2 < 0.00014: z += 29200 × (log_sum_pt − 6.60) × (0.00014 − mean_phi2)
if LHA < 0.210 and centroid_offset > 0.0028: z += -920 × (0.210 − LHA) × (centroid_offset − 0.0028)
if sum_pt_top5 > 730 and D2 < 1.60: z += -0.012 × (sum_pt_top5 − 730) × (1.60 − D2)
if sum_pt > 870: z += -0.0037 × (sum_pt − 870)
if girth2 < 5.2e-05: z += 55900 × (5.2e-05 − girth2)
if log_sum_pt > 6.60 and centroid_offset > 0.017: z += -799 × (log_sum_pt − 6.60) × (centroid_offset − 0.017)
if LHA < 0.150 and centroid_offset > 0.0034: z += -3370 × (0.150 − LHA) × (centroid_offset − 0.0034)
if mass_over_sum_pt < 0.081 and mass_top3 > 29.00: z += 6.77 × (0.081 − mass_over_sum_pt) × (mass_top3 − 29.00)
if log_sum_pt > 6.90 and pt_5 > 48.00: z += 0.534 × (log_sum_pt − 6.90) × (pt_5 − 48.00)
if LHA < 0.220 and n_dr_0p2_0p4 > -1.6e-05: z += -14.70 × (0.220 − LHA) × (n_dr_0p2_0p4 − -1.6e-05)
if LHA < 0.210 and mass_top3 > 3.50: z += -1.74 × (0.210 − LHA) × (mass_top3 − 3.50)
if LHA < 0.210 and planar_flow < 0.083: z += 397 × (0.210 − LHA) × (0.083 − planar_flow)
if log_sum_pt > 6.90 and dr_4 > 0.039: z += 201 × (log_sum_pt − 6.90) × (dr_4 − 0.039)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Soft-leading wide massive mixed jets** — 26.6% of jets, neuron 0.17, formula right for 69%. A top-led mixture (t 38%, Z 21%, W 20%, g 16%) at 55.8 GeV, about twice the average width, low total pT (578 GeV) shared evenly (leading particle only 136 GeV, 8th 42.0 GeV). sum_pt_top2 < 540 (-2.008) passes for all, and the small pluses pt_7 < 53 and mean_phi2 < 0.015 and max_pair_mass < 41 cannot beat it, while e2 < 0.034 mostly fails. The neuron is mostly off (on for 0.24, mean 0.17), so it barely touches the scores; the formula calls them t but is right 0.692 of the time.
- **Wide top-led jets, soft 8th particle** — 19.4% of jets, neuron 0.87, formula right for 69%. A top-led mixture (t 40%, Z 23%, W 20%) at 59.1 GeV, wider than average, with a harder leading particle (216 GeV) and a soft 8th particle. pt_7 < 53 (+1.639) always passes and outweighs sum_pt_top2 < 540, but z_7 < 0.073 and sum_pt < 790 (-0.906, far more common here) and z_7 < 0.071 and centroid_offset < 0.031 pull back. The neuron is on for about half (0.866), slightly lowering the t and g scores; the formula calls them t and is right 0.692 of the time.
- **High-pT two-prong W/Z jets** — 15.2% of jets, neuron 2.66, formula right for 71%. A W/Z mixture (W 39%, Z 38%) at 54.4 GeV, of high total pT (855 GeV) with a hard leading particle (352 GeV), fairly narrow. Being high-pT they pass pt_7 < 53, log_sum_pt > 6.6 and log_sum_pt > 6.3 and centroid_offset > 0.00063 and often escape sum_pt_top2 < 540, against z_7 < 0.071 and centroid_offset < 0.031 (-1.582) and log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99. The neuron is fairly high (2.66), raising the q score and lowering g and t; the formula calls them W and is right 0.707 of the time.
- **Light narrow gluon-led mixture** — 12.7% of jets, neuron 2.03, formula right for 49%. A gluon-led mixture (g 39%, q 20%, W 18%, Z 16%), light (14.8 GeV), narrow (90% of pT within 0.05), with a soft leading particle (179 GeV). Pluses e2 < 0.034, pt_7 < 53, mass < 57 and n_dr_0p2_0p4 < 2 and mean_phi2 < 0.015 and max_pair_mass < 41 beat sum_pt_top2 < 540, sum_pt_top2 < 570 and girth2_top3 < 0.0036 and LHA < 0.22 and log_sum_pt < 6.8. The neuron is on (2.03), raising q and lowering g and t; the formula calls them g but is right only 0.486 of the time, the worst of this neuron's groups.
- **Pencil-thin soft-leading gluon jets** — 9.0% of jets, neuron 1.28, formula right for 59%. Mostly gluons (53%, q 35%), nearly massless (8.5 GeV), 98% of pT within 0.05, with pT spread over many particles (leading 182 GeV, 8th 40.7 GeV). width < 0.0026 and centroid_offset < 0.025 (+2.992), e2 < 0.034 and sum_pt_top2 < 550 and dr_0 < 0.022 add, but dr_0 < 0.023 (-2.908), LHA < 0.22 and log_sum_pt < 6.8, sum_pt_top2 < 540 and sum_pt_top2 < 570 and girth2_top3 < 0.0036 subtract. The neuron is moderate (1.278, on for 0.666); the formula calls them g and is right 0.587 of the time.
- **Narrow harder quark jets** — 8.4% of jets, neuron 6.18, formula right for 65%. Mostly light quarks (61%, g 23%), nearly massless (8.6 GeV), extremely narrow, of above-average pT (876 GeV) with a hard leading particle (330 GeV). width < 0.0026 and centroid_offset < 0.025 (+3.444), e2 < 0.034, log_sum_pt > 6.6 and dr_0 < 0.021 (common only here and in the high-pT groups) and pt_7 < 53 beat dr_0 < 0.023 (-3.449), z_7 < 0.071 and centroid_offset < 0.031 and log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99. The neuron is high (6.181), raising q and lowering g and t; the formula calls them q and is right 0.652 of the time.
- **High-pT one-particle quark jets** — 4.6% of jets, neuron 9.18, formula right for 77%. Mostly light quarks (75%), nearly massless, of high pT (980 GeV) with the leading particle carrying 468 GeV and a soft 8th particle. Large pluses width < 0.0026 and centroid_offset < 0.025, log_sum_pt > 6.6 and dr_0 < 0.021, pt_7 < 53 and sum_pt > 870 and centroid_offset < 0.013 outweigh dr_0 < 0.023, z_7 < 0.071 and centroid_offset < 0.031, log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99 and log_sum_pt > 6.8. This is the neuron's highest value (9.184), strongly raising q and lowering g and t; the formula calls them q and is right 0.77 of the time.
- **Very high-pT W/Z jets** — 2.6% of jets, neuron 1.04, formula right for 67%. A W/Z mixture (W 35%, Z 34%, g 19%) at 62.2 GeV and very high pT (1015 GeV), leading particle 465 GeV. log_sum_pt > 6.8 (-3.932) always passes, joined by log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99, z_7 < 0.071 and centroid_offset < 0.031 and sum_pt_top5 > 730 and D2 < 1.6, outweighing log_sum_pt > 6.6, pt_7 < 53 and sum_pt > 870 and centroid_offset < 0.013. The neuron is usually zero (on for 0.385, mean 1.037); the formula calls them W and is right 0.672 of the time.
- **Very high-pT light quark/gluon jets** — 1.2% of jets, neuron 2.29, formula right for 67%. Quarks (46%) and gluons (41%) in nearly equal parts, light (14.3 GeV), very narrow, at very high pT (1134 GeV) with a 505 GeV leading particle. The very-high-pT penalties log_sum_pt > 6.8 (-7.588) and log_sum_pt > 6.9 and dr_0 < 0.043 (-7.333), plus log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99 and dr_0 < 0.023, are mostly offset by log_sum_pt > 6.6 and dr_0 < 0.021, sum_pt > 870 and centroid_offset < 0.013 and width < 0.0026 and centroid_offset < 0.025. The neuron is on for 0.567 (2.295); the formula divides them almost evenly between g and q and is right 0.669 of the time.
- **Highest-pT gluon-rich jets** — 0.3% of jets, neuron 0.04, formula right for 66%. A tiny group (0.3%), mostly gluons (59%, q 30%), at the highest pT (1391 GeV) with a 673 GeV leading particle, light and very narrow. log_sum_pt > 6.9 and dr_0 < 0.043 (-19.634) and log_sum_pt > 6.8 (-14.112) outweigh sum_pt > 870 and centroid_offset < 0.013 and log_sum_pt > 6.6 and dr_0 < 0.021, so the neuron is off (on for 0.029). The formula calls them g and is right 0.659 of the time.

### neuron 6: Broad jet with off-centre pT (moderate)

- **What it measures:** Rises for broad jets (compact ones are cut off by girth2 < 0.0086 and width < 0.013) and grows with the offset of the pT centroid from the axis (+0.492), falling at high total pT (-0.376). Largest for t (3.80), then g (1.86), q (0.92), Z (0.56) and W (0.23).
- *computed — its value:* largest for t (3.80), then g (1.86), then q (0.92), then Z (0.56), then W (0.23); it separates t jets from the rest best (AUC 0.78: large for t)
- **How the class scores use it:** The Z (-17%) and W (-10%) scores subtract it and the q (+8%) and g (+6%) scores add it: W and Z sit lowest on this scale, so a broad, lopsided jet moves the decision from the bosons toward quark or gluon. It does not enter the t score, even though tops sit highest on it.
- *computed — used by:* raises the score of g (+6%), q (+8%); lowers the score of W (-10%), Z (-17%); does not (or hardly) enter the score of t (share of each class score’s average input)
- **Boundaries:** Low (0.40, on for 32%) for the 73% of jets that are not broad and have a centred centroid (girth2 ≤ 0.00885, centroid offset ≤ 0.0342, log total pT > 6.19); high (8.01) for broad jets (girth2 > 0.00885) with τ32 > 0.234 and an off-centre centroid (> 0.0417; 4.3%, 71% t) and 5.28 with a centred one (8.5%, 66% t), but only 0.59 for broad jets with a clean three-prong τ32 ≤ 0.234 and λ2 > 0.00515 (2.4%, 97% t). In the broad regimes it subtracts strongly from the bosons (-2.50 W and -3.00 Z in the top regime, -1.65 / -1.98 in the next) and adds to g (+0.88) and q (+1.00); it does not enter t. The regimes describe it moderately well (regime_r2 0.63).

Regimes (a small tree on its quantities; R² 0.626):

- `` — 4.3% of jets, value 8.01 (3.00…12.75), formula right 73%
- `` — 8.5% of jets, value 5.28 (2.00…8.38), formula right 70%
- `` — 2.0% of jets, value 5.03 (1.38…9.75), formula right 48%
- `` — 4.2% of jets, value 3.53 (0.88…6.25), formula right 86%
- `` — 2.2% of jets, value 2.55 (0.00…5.00), formula right 47%
- `` — 3.5% of jets, value 2.15 (0.00…5.62), formula right 56%
- `` — 2.4% of jets, value 0.59 (0.00…2.00), formula right 97%
- `` — 72.9% of jets, value 0.40 (0.00…1.38), formula right 64%

```
z = 10.50
if girth2 < 0.0086: z += -1290 × (0.0086 − girth2)
if width < 0.013: z += -657 × (0.013 − width)
if mass_over_sum_pt > 0.0081: z += -80.70 × (mass_over_sum_pt − 0.0081)
if lam1 < 0.0078: z += 485 × (0.0078 − lam1)
if e2 < 0.051: z += 57.20 × (0.051 − e2)
if girth > 0.083: z += 100 × (girth − 0.083)
if girth2_top2 < 0.010: z += 106 × (0.010 − girth2_top2)
if girth2 < 0.0036: z += 573 × (0.0036 − girth2)
if mass < 50.00 and z_dr_0p05_0p1 < 0.760: z += 0.056 × (50.00 − mass) × (0.760 − z_dr_0p05_0p1)
if log_sum_pt < 6.70 and pt_7 < 46.00: z += 0.317 × (6.70 − log_sum_pt) × (46.00 − pt_7)
if lam2 < 0.00054 and z_dr_0p2_0p4 < 0.200: z += -8070 × (0.00054 − lam2) × (0.200 − z_dr_0p2_0p4)
if max_dr > 0.120: z += 14.10 × (max_dr − 0.120)
if mass < 50.00: z += -0.028 × (50.00 − mass)
if centroid_offset > 0.0076 and lam2 < 0.0035: z += 14800 × (centroid_offset − 0.0076) × (0.0035 − lam2)
if max_dr < 0.110: z += -13.80 × (0.110 − max_dr)
if C2 < 0.034: z += 27.20 × (0.034 − C2)
if mass_over_sum_pt > 0.0086 and pt_7 < 42.00: z += -0.709 × (mass_over_sum_pt − 0.0086) × (42.00 − pt_7)
if mass_over_sum_pt > 0.015 and tau32 < 0.520: z += -54.50 × (mass_over_sum_pt − 0.015) × (0.520 − tau32)
if girth2_top5 > 0.011: z += 178 × (girth2_top5 − 0.011)
if log_sum_pt < 6.80: z += -1.05 × (6.80 − log_sum_pt)
if centroid_offset > 0.0084 and C2 < 0.096: z += 477 × (centroid_offset − 0.0084) × (0.096 − C2)
if D2 < 1.70 and min_pair_mass < 2.40: z += -0.302 × (1.70 − D2) × (2.40 − min_pair_mass)
if e2 < 0.050 and z_dr_0p1_0p2 > 0.160: z += 1160 × (0.050 − e2) × (z_dr_0p1_0p2 − 0.160)
if C2 > 0.033: z += -23.90 × (C2 − 0.033)
if centroid_offset > 0.019: z += -38.10 × (centroid_offset − 0.019)
if D2 < 1.80 and pt_4 < 86.00: z += -0.011 × (1.80 − D2) × (86.00 − pt_4)
if girth2_top5 > 0.011 and pt_7 > 17.00: z += -6.12 × (girth2_top5 − 0.011) × (pt_7 − 17.00)
if lam2 > 0.0034: z += -1060 × (lam2 − 0.0034)
if C2 > 0.010 and pt_7 > 32.00: z += 1.88 × (C2 − 0.010) × (pt_7 − 32.00)
if planar_flow < 0.060: z += -10.50 × (0.060 − planar_flow)
if sum_pt > 990: z += 0.026 × (sum_pt − 990)
if log_sum_pt < 6.70 and z_7 < 0.049: z += 542 × (6.70 − log_sum_pt) × (0.049 − z_7)
if log_sum_pt < 6.70 and pt_6 < 37.00: z += 0.234 × (6.70 − log_sum_pt) × (37.00 − pt_6)
if centroid_offset > 0.0078 and mass_top3 < 3.50: z += -10.10 × (centroid_offset − 0.0078) × (3.50 − mass_top3)
if centroid_offset > 0.050: z += 106 × (centroid_offset − 0.050)
if log_sum_pt < 6.30 and z_7 < 0.071: z += 1010 × (6.30 − log_sum_pt) × (0.071 − z_7)
if sum_pt_top5 > 840: z += -0.0083 × (sum_pt_top5 − 840)
if log_sum_pt < 6.70 and mean_phi > 0.0098: z += -76.10 × (6.70 − log_sum_pt) × (mean_phi − 0.0098)
if log_sum_pt < 6.30 and pt_6 > 27.00: z += 0.248 × (6.30 − log_sum_pt) × (pt_6 − 27.00)
if girth2_top5 > 0.0082 and mean_eta > 0.014: z += -4370 × (girth2_top5 − 0.0082) × (mean_eta − 0.014)
if log_sum_pt < 6.70 and pt_dispersion > 0.400: z += -20.20 × (6.70 − log_sum_pt) × (pt_dispersion − 0.400)
if centroid_offset > 0.0079 and mean_eta2 < 0.0012: z += -24600 × (centroid_offset − 0.0079) × (0.0012 − mean_eta2)
if pt_6 < 25.00: z += 0.079 × (25.00 − pt_6)
if girth2_top2 < 0.0096 and mean_phi < -0.0091: z += 4430 × (0.0096 − girth2_top2) × (-0.0091 − mean_phi)
if mass < 49.00 and dr_7 > 0.150: z += 0.972 × (49.00 − mass) × (dr_7 − 0.150)
if girth2 < 0.0035 and dr_7 > 0.130: z += -7360 × (0.0035 − girth2) × (dr_7 − 0.130)
if centroid_offset > 0.019 and pt_5 > 59.00: z += 4.75 × (centroid_offset − 0.019) × (pt_5 − 59.00)
if width < 0.014 and mean_phi > 0.026: z += 7910 × (0.014 − width) × (mean_phi − 0.026)
if sum_pt > 970 and pt_6 > 36.00: z += -0.0002 × (sum_pt − 970) × (pt_6 − 36.00)
if width < 0.014 and mean_phi < -0.026: z += 5630 × (0.014 − width) × (-0.026 − mean_phi)
if centroid_offset > 0.019 and pt_5 < 25.00: z += 31.80 × (centroid_offset − 0.019) × (25.00 − pt_5)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 29.8% of jets, neuron 0.54, formula right for 61%. The largest group (30%), mostly light quarks (45%) with gluons (34%), nearly massless (8.2 GeV) with 99% of pT within 0.05 of the axis. The compactness cuts girth2 < 0.0086 (-10.725) and width < 0.013 (-8.353), plus mass < 50 and max_dr < 0.11, subtract heavily, only partly offset by lam1 < 0.0078, e2 < 0.051, girth2 < 0.0036 and mass < 50 and z_dr_0p05_0p1 < 0.76. The neuron is usually zero (on for 0.394, mean 0.541), so its small push up of g and q and down of W and Z hardly matters; the formula calls them q and is right only 0.611 of the time.
- **Two-prong W-rich jets** — 17.7% of jets, neuron 0.46, formula right for 71%. Mostly W (51%) with Z (31%), mass 54.0 GeV, pT concentrated at 0.05-0.1 from the axis (two prongs) with little beyond 0.1. mass_over_sum_pt > 0.0081 (-5.334), width < 0.013 and girth2 < 0.0086 all pass and outweigh lam1 < 0.0078, e2 < 0.051 and girth2_top2 < 0.01, while girth2 < 0.0036 never passes. The neuron is usually zero (on for 0.259, mean 0.458), only slightly lowering W and Z; the formula calls them W and is right 0.708 of the time.
- **Compact medium-mass W-led jets** — 13.0% of jets, neuron 0.53, formula right for 58%. W-led (44%) mixture with Z (26%) and gluons, 42.5 GeV, fairly narrow (66% of pT within 0.05). girth2 < 0.0086, width < 0.013 and mass_over_sum_pt > 0.0081 subtract about 16 in total, against lam1 < 0.0078, e2 < 0.051, girth2_top2 < 0.01, max_dr > 0.12 and centroid_offset > 0.0076 and lam2 < 0.0035. The neuron is usually zero (on for 0.285, mean 0.533); the formula calls them W but is right only 0.58 of the time.
- **Z jets with extra radiation** — 11.5% of jets, neuron 1.58, formula right for 72%. Mostly Z (62%) with tops (19%), 59.7 GeV, somewhat wider than average with pT at 0.05-0.1 and 23% beyond 0.1. mass_over_sum_pt > 0.0081 (-6.35) and width < 0.013 subtract, but girth2 < 0.0086 fails for some and e2 < 0.05 and z_dr_0p1_0p2 > 0.16 (common only here and in the off-centre group) plus log_sum_pt < 6.7 and pt_7 < 46 add. The neuron is on for 0.638 (1.576), which lowers the Z and W scores and raises g and q, working against the right answer, yet the formula still calls them Z and is right 0.723 of the time.
- **Light compact gluon-led mixture** — 10.5% of jets, neuron 0.94, formula right for 50%. A mixture led by gluons (35%) with quarks (25%), W (19%) and Z (14%), light (23.5 GeV) and narrow (85% of pT within 0.05). girth2 < 0.0086 (-8.827), width < 0.013, mass_over_sum_pt > 0.0081 and mass < 50 subtract; lam1 < 0.0078, e2 < 0.051, girth2 < 0.0036, girth2_top2 < 0.01 and mass < 50 and z_dr_0p05_0p1 < 0.76 add back. The neuron is on for 0.392 (0.94); the formula calls them g but is right only 0.504 of the time, a badly mixed group.
- **Moderately wide top jets** — 6.5% of jets, neuron 5.62, formula right for 73%. Mostly tops (71%, g 19%), 65.7 GeV, about twice the average width with 40% of pT beyond 0.1 and low pT (588 GeV). They escape girth2 < 0.0086 and lam1 < 0.0078 and mostly width < 0.013, and gain from girth > 0.083, max_dr > 0.12, log_sum_pt < 6.7 and pt_7 < 46 and centroid_offset > 0.0076 and lam2 < 0.0035, against mass_over_sum_pt > 0.0081 (-8.329). The neuron is high (5.619), strongly lowering W and Z and raising g and q; the formula calls them t and is right 0.728 of the time, the gluon admixture likely accounting for most errors.
- **Wide top jets** — 5.9% of jets, neuron 5.07, formula right for 83%. Mostly tops (82%), 79.6 GeV, over three times the average width with 71% of pT beyond 0.1. No compactness cut passes (width < 0.013 and e2 < 0.051 fail); girth > 0.083 (+5.053), girth2_top5 > 0.011, max_dr > 0.12 and log_sum_pt < 6.7 and pt_7 < 46 add, while mass_over_sum_pt > 0.0081 (-10.691), mass_over_sum_pt > 0.015 and tau32 < 0.52 and girth2_top5 > 0.011 and pt_7 > 17 subtract. The neuron is high (5.07), lowering W and Z; the formula calls them t and is right 0.829 of the time.
- **Very wide top/gluon jets** — 3.0% of jets, neuron 6.11, formula right for 79%. Mostly tops (75%, g 18%), 89.4 GeV, five times the average width with 87% of pT beyond 0.1 and low pT (523 GeV). girth > 0.083 (+8.435), girth2_top5 > 0.011 and log_sum_pt < 6.7 and pt_7 < 46 add more than mass_over_sum_pt > 0.0081 (-13.151), girth2_top5 > 0.011 and pt_7 > 17 and mass_over_sum_pt > 0.015 and tau32 < 0.52 take away, the amounts growing with the width. The neuron is high (6.109), lowering W and Z; the formula calls them t and is right 0.791 of the time.
- **Wide tops with large lam2** — 1.9% of jets, neuron 0.63, formula right for 95%. Almost all tops (95%), 86.9 GeV, very wide with 89% of pT beyond 0.1. They look like the other wide top groups except that all pass lam2 > 0.0034 (-6.159), which barely fires elsewhere, and nearly all pass mass_over_sum_pt > 0.015 and tau32 < 0.52; with mass_over_sum_pt > 0.0081 (-12.695) this beats girth > 0.083 and girth2_top5 > 0.011. The neuron is usually zero (on for 0.254, mean 0.626), but the formula still calls them t and is right 0.953 of the time.
- **Off-centre wide jets** — 0.4% of jets, neuron 7.61, formula right for 67%. A tiny group (0.4%) of tops (67%) and gluons (24%), lighter (44.5 GeV), wide with 87% of pT beyond 0.1 and a pT centroid far from the axis. All pass centroid_offset > 0.05, which is rare elsewhere, and e2 < 0.05 and z_dr_0p1_0p2 > 0.16 (+13.65), plus girth > 0.083 and centroid_offset > 0.0076 and lam2 < 0.0035, outweighing mass_over_sum_pt > 0.0081 and centroid_offset > 0.019. This is the neuron's highest value (7.614), lowering W and Z and raising g and q; the formula calls them t but is right only 0.667 of the time.

### neuron 7: Z-likeness: two-prong, mass window (moderate)

- **What it measures:** Rises for elongated two-prong jets (eccentricity +0.569, planar flow -0.569, τ21 -0.52) inside a mass/pT window (mass/pT > 0.072 and > 0.085 push it up, > 0.091 pushes it down) and of intermediate width (set by several girth2 cuts). Largest for Z (3.17), then W (1.87), with t (0.70), g (0.51) and q (0.26) low.
- *computed — its value:* largest for Z (3.17), then W (1.87), then t (0.70), then g (0.51), then q (0.26); it separates Z jets from the rest best (AUC 0.82: large for Z)
- **How the class scores use it:** The Z score adds it strongly (+19%) and the W score more weakly (+6%), so it marks a boson and leans the choice toward Z. It does not enter the g, q or t scores.
- *computed — used by:* raises the score of W (+6%), Z (+19%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** High (4.24) for flat (planar flow ≤ 0.0587), not-wide (girth2 ≤ 0.0109) jets with mass > 54.5 GeV, 15% of jets, mostly Z (51%); 2.14 for less flat jets with girth2 > 0.00328 and width ≤ 0.0106 (15%) and 1.79 for flat light ones with mass ≤ 54.5 GeV (14%, mostly W at 41%). It is off for thin centred jets (planar flow > 0.0587, girth2 ≤ 0.00328, centroid ≤ 0.0204: 0.071, 32% of jets, mostly q at 46%) and for broad jets (width > 0.0106: 0.052; girth2 > 0.0109 with mass > 75.4 GeV: 0.054). It feeds only the bosons, adding +1.99 to Z and +0.93 to W in the top regime (+1.00 / +0.47 and +0.84 / +0.39 in the next two); the regimes describe it fairly well (regime_r2 0.69).

Regimes (a small tree on its quantities; R² 0.694):

- `` — 15.1% of jets, value 4.24 (2.38…6.25), formula right 80%
- `` — 14.9% of jets, value 2.14 (0.23…3.88), formula right 58%
- `` — 14.4% of jets, value 1.79 (0.00…3.62), formula right 62%
- `` — 6.3% of jets, value 0.77 (0.00…1.75), formula right 44%
- `` — 2.1% of jets, value 0.66 (0.00…2.12), formula right 68%
- `` — 32.2% of jets, value 0.07 (0.00…0.00), formula right 61%
- `` — 2.7% of jets, value 0.05 (0.00…0.00), formula right 72%
- `` — 12.2% of jets, value 0.05 (0.00…0.00), formula right 84%

```
z = 9.24
if girth2 > 0.0075: z += -1410 × (girth2 − 0.0075)
if girth2 > 0.0015: z += -612 × (girth2 − 0.0015)
if girth2 > 0.0087: z += 1470 × (girth2 − 0.0087)
if lam1 < 0.0084: z += -610 × (0.0084 − lam1)
if max_dr < 0.160: z += -40.20 × (0.160 − max_dr)
if mass_over_sum_pt > 0.091: z += -280 × (mass_over_sum_pt − 0.091)
if girth < 0.088: z += -52.90 × (0.088 − girth)
if mass_over_sum_pt > 0.085: z += 191 × (mass_over_sum_pt − 0.085)
if width < 0.0055: z += -781 × (0.0055 − width)
if max_dr < 0.160 and z_dr_0p05_0p1 < 0.670: z += 56.90 × (0.160 − max_dr) × (0.670 − z_dr_0p05_0p1)
if girth2 > 0.0044: z += -457 × (girth2 − 0.0044)
if mass_over_sum_pt > 0.072: z += 97.80 × (mass_over_sum_pt − 0.072)
if centroid_offset < 0.039 and sum_pt > 560: z += 0.227 × (0.039 − centroid_offset) × (sum_pt − 560)
if width < 0.0056 and n_dr_0p2_0p4 < 0.960: z += -445 × (0.0056 − width) × (0.960 − n_dr_0p2_0p4)
if width < 0.0053 and n_dr_0p1_0p2 < 3.10: z += 128 × (0.0053 − width) × (3.10 − n_dr_0p1_0p2)
if mass < 30.00: z += 0.093 × (30.00 − mass)
if girth2 > 0.015: z += 554 × (girth2 − 0.015)
if girth2 > 0.004 and eccentricity > 0.950: z += 9130 × (girth2 − 0.004) × (eccentricity − 0.950)
if max_dr < 0.200 and z_dr_0p05_0p1 > 0.056: z += 37.60 × (0.200 − max_dr) × (z_dr_0p05_0p1 − 0.056)
if centroid_offset < 0.021: z += -70.80 × (0.021 − centroid_offset)
if pt_7 < 47.00 and planar_flow < 0.740: z += -0.097 × (47.00 − pt_7) × (0.740 − planar_flow)
if girth2_top2 < 0.0011 and n_dr_0p2_0p4 < 0.970: z += 1880 × (0.0011 − girth2_top2) × (0.970 − n_dr_0p2_0p4)
if girth2 > 0.0075 and log_sum_pt > 6.20: z += -1440 × (girth2 − 0.0075) × (log_sum_pt − 6.20)
if lam1 < 0.0081 and D2 < 1.10: z += -1250 × (0.0081 − lam1) × (1.10 − D2)
if e2_sq < 0.0012: z += 1250 × (0.0012 − e2_sq)
if max_dr < 0.160 and D2 < 1.20: z += 34.60 × (0.160 − max_dr) × (1.20 − D2)
if girth2_top2 < 0.0011: z += -1090 × (0.0011 − girth2_top2)
if planar_flow < 0.200 and D2 < 1.60: z += -4.77 × (0.200 − planar_flow) × (1.60 − D2)
if mass_over_sum_pt < 0.130 and z_dr_0p05_0p1 > 0.270: z += -34.70 × (0.130 − mass_over_sum_pt) × (z_dr_0p05_0p1 − 0.270)
if girth2_top2 < 0.001 and log_sum_pt > 6.30: z += -2170 × (0.001 − girth2_top2) × (log_sum_pt − 6.30)
if planar_flow < 0.200 and sum_pt > 620: z += 0.023 × (0.200 − planar_flow) × (sum_pt − 620)
if mass > 80.40: z += -0.189 × (mass − 80.40)
if e2 < 0.038 and D2 < 1.10: z += 325 × (0.038 − e2) × (1.10 − D2)
if width < 0.00052: z += -2300 × (0.00052 − width)
if planar_flow < 0.190 and n_dr_0p1_0p2 < 2.90: z += -1.17 × (0.190 − planar_flow) × (2.90 − n_dr_0p1_0p2)
if e2 < 0.025 and tau21 < 0.460: z += 338 × (0.025 − e2) × (0.460 − tau21)
if centroid_offset < 0.022 and C2 > 0.025: z += 2380 × (0.022 − centroid_offset) × (C2 − 0.025)
if z_dr_0p05_0p1 > 0.660: z += -2.07 × (z_dr_0p05_0p1 − 0.660)
if lam1 < 0.0087 and n_pt_above_50 < 4.10: z += -78.80 × (0.0087 − lam1) × (4.10 − n_pt_above_50)
if centroid_offset < 0.037 and C2 > 0.068: z += -1970 × (0.037 − centroid_offset) × (C2 − 0.068)
if e2 < 0.025 and D2 < 1.10: z += -509 × (0.025 − e2) × (1.10 − D2)
if mass > 80.40 and eccentricity > 0.920: z += -1.05 × (mass − 80.40) × (eccentricity − 0.920)
if centroid_offset > 0.031 and pt_2 > 64.00: z += -1.32 × (centroid_offset − 0.031) × (pt_2 − 64.00)
if width < 0.006 and mean_phi < -0.0044: z += 7290 × (0.006 − width) × (-0.0044 − mean_phi)
if mass > 80.40 and m012 < 5.40: z += -0.056 × (mass − 80.40) × (5.40 − m012)
if centroid_offset > 0.031 and pt_0 > 380: z += -10.90 × (centroid_offset − 0.031) × (pt_0 − 380)
if girth2_top3 < 0.005 and n_dr_0p2_0p4 > 0.880: z += 147 × (0.005 − girth2_top3) × (n_dr_0p2_0p4 − 0.880)
if e2 < 0.025 and phi_0 > 0.055: z += -38000 × (0.025 − e2) × (phi_0 − 0.055)
if girth2_top3 < 0.0059 and max_pair_mass > 46.00: z += 255 × (0.0059 − girth2_top3) × (max_pair_mass − 46.00)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 32.9% of jets, neuron 0.20, formula right for 60%. The largest group (33%), mostly light quarks (42%) with gluons (36%), nearly massless (9.0 GeV) with 98% of pT within 0.05. Every compactness test fires: max_dr < 0.16, lam1 < 0.0084, width < 0.0055, girth < 0.088 and width < 0.0056 and n_dr_0p2_0p4 < 0.96 subtract about 20, while max_dr < 0.16 and z_dr_0p05_0p1 < 0.67, mass < 30, girth2_top2 < 0.0011 and n_dr_0p2_0p4 < 0.97 and e2_sq < 0.0012 add back less. The neuron is usually zero (on for 0.191, mean 0.197), so it adds almost nothing to the W and Z scores; the formula calls them q and is right only 0.597 of the time.
- **Two-prong Z/W jets** — 28.5% of jets, neuron 3.30, formula right for 71%. Mostly Z (42%) with W (34%) and tops (13%), mass 54.7 GeV and 64% of pT at 0.05-0.1 from the axis: a clean two-prong shape of intermediate width. The two-prong tests max_dr < 0.2 and z_dr_0p05_0p1 > 0.056, centroid_offset < 0.039 and sum_pt > 560 and girth2 > 0.004 and eccentricity > 0.95 add, and the width cuts girth2 > 0.0015, girth2 > 0.0044, max_dr < 0.16, lam1 < 0.0084 and lam1 < 0.0081 and D2 < 1.1 subtract moderately, leaving the neuron's highest value (3.299, on for 0.958). This raises the Z score most and the W score too; the formula calls them Z and is right 0.708 of the time, W and Z being hard to tell apart here.
- **Compact medium-mass W-led mixture** — 20.2% of jets, neuron 1.25, formula right for 57%. A W-led (36%) mixture with Z (24%), gluons (17%) and quarks (15%), 38.6 GeV, narrow (79% of pT within 0.05). lam1 < 0.0084, girth < 0.088, width < 0.0055, girth2 > 0.0015, pt_7 < 47 and planar_flow < 0.74 and (for half) max_dr < 0.16 subtract, against centroid_offset < 0.039 and sum_pt > 560, width < 0.0053 and n_dr_0p1_0p2 < 3.1 and e2 < 0.025 and tau21 < 0.46; the mass window cut mass_over_sum_pt > 0.072 almost never passes. The neuron is middling (1.255, on for 0.664), adding to W and Z; the formula calls them W but is right only 0.571 of the time.
- **Moderately wide top/gluon jets** — 3.9% of jets, neuron 1.16, formula right for 65%. Mostly tops (63%) with gluons (22%), 58.6 GeV, nearly twice the average width, 37% of pT beyond 0.1. Here the width ladder girth2 > 0.0015, girth2 > 0.0044, girth2 > 0.0075 (together about -15) against girth2 > 0.0087 (+4.096) meets the mass window mass_over_sum_pt > 0.072 and > 0.085 (+) and > 0.091 (-), and the balance stays slightly positive (1.164, on for 0.594). That adds to the W and Z scores for these tops; the formula still calls them t and is right 0.648 of the time.
- **Wide top jets** — 3.6% of jets, neuron 0.08, formula right for 76%. Mostly tops (75%, g 17%), 69.3 GeV, 2.4 times the average width with 52% of pT beyond 0.1. The same ladder with bigger amounts (they grow with width): girth2 > 0.0075 (-11.238) against girth2 > 0.0087 (+9.952), and mass_over_sum_pt > 0.091 (-7.428) against mass_over_sum_pt > 0.085 and > 0.072, with girth2 > 0.0015 and > 0.0044 tipping it negative. The neuron is mostly zero (on for 0.094), leaving the W and Z scores alone; the formula calls them t and is right 0.763 of the time.
- **Wider heavy top jets** — 3.3% of jets, neuron 0.03, formula right for 82%. Mostly tops (81%), 77.2 GeV, three times the average width with 65% of pT beyond 0.1. All pass the whole girth2 ladder and mass window; the opposing pairs girth2 > 0.0075 / girth2 > 0.0087 and mass_over_sum_pt > 0.091 / > 0.085 nearly cancel, and girth2 > 0.0015 and girth2 > 0.0044 leave the total below zero, so the neuron is essentially off (on for 0.034). The formula calls them t and is right 0.821 of the time.
- **Very wide heavy top jets** — 3.1% of jets, neuron 0.02, formula right for 86%. Mostly tops (86%), 82.7 GeV, nearly four times the average width with 75% of pT beyond 0.1. Same cancellation with larger amounts (girth2 > 0.0075 at -23.081 against girth2 > 0.0087 at +22.299), and girth2 > 0.015 now adds but cannot overcome girth2 > 0.0015 and the mass-window minus, so the neuron is off (on for 0.026). The formula calls them t and is right 0.864 of the time.
- **Very wide heavy top jets** — 2.6% of jets, neuron 0.01, formula right for 88%. Mostly tops (87%), 88.8 GeV, 4.5 times the average width with 83% of pT beyond 0.1. The girth2 ladder and the mass_over_sum_pt window again cancel pairwise (about -29.69 against +29.19 for the first pair) with the leftover negative, so the neuron is off (on for 0.016). The formula calls them t and is right 0.884 of the time.
- **Extremely wide top/gluon jets** — 1.4% of jets, neuron 0.02, formula right for 81%. Mostly tops (77%, g 17%), 91.1 GeV, over five times the average width with 88% of pT beyond 0.1. Same pattern with even larger opposing terms (girth2 > 0.0075 and girth2 > 0.0087 each near 37.6 in size) and girth2 > 0.015 adding, but the total stays negative and the neuron is off (on for 0.029). The formula calls them t and is right 0.814 of the time.
- **Widest soft-leading top/gluon jets** — 0.3% of jets, neuron 0.02, formula right for 61%. A tiny group (0.3%) of tops (58%) and gluons (30%), 93.9 GeV, the widest jets (0.0449) with 92% of pT beyond 0.1 and a soft leading particle (113 GeV). The largest cancellation of all (girth2 > 0.0087 +53.247 against girth2 > 0.0075 -52.766, and mass_over_sum_pt > 0.091 against > 0.085) with girth2 > 0.0015 and > 0.0044 winning, so the neuron is off (on for 0.02). The formula calls them t but is right only 0.613 of the time, the gluon admixture likely being misread.

### neuron 8: Narrowness (small width) (moderate)

- **What it measures:** Rises for narrow jets: width < 0.0049 is its largest term, though the very narrowest (girth < 0.063 with width < 0.0055) are pulled back; overall it falls with width and girth2 (-0.686). Largest for q (2.96), then g (1.74), with W (0.55), Z (0.35) and t (0.15) low.
- *computed — its value:* largest for q (2.96), then g (1.74), then W (0.55), then Z (0.35), then t (0.15); it separates q jets from the rest best (AUC 0.82: large for q)
- **How the class scores use it:** The q score adds it (+3%) and the W score subtracts it (-6%), in line with quarks sitting high and W jets low. The t score also adds a little (+5%) although tops sit lowest, probably as a small correction for narrow jets; it does not enter the g or Z scores.
- *computed — used by:* raises the score of q (+3%), t (+5%); lowers the score of W (-6%); does not (or hardly) enter the score of g, Z (share of each class score’s average input)
- **Boundaries:** High only for the narrowest jets: 4.49 for width ≤ 0.00209, centroid ≤ 0.0124 and planar flow ≤ 0.285 (6.3% of jets, mostly q at 50%) and 3.40 with planar flow > 0.285 (19%, mostly q at 53%); 1.68 once the centroid moves to 0.0124-0.0169, and off (0.031) for jets with width > 0.00477 and girth2 > 0.00349 (47% of jets, mostly t at 38%). Its adds are -0.85 on W, +0.64 on t and +0.21 on q in the 19% regime (-1.12 / +0.84 / +0.28 in the narrowest one), so it mainly keeps very thin one-prong jets away from W; it does not enter g or Z. The regimes describe it well (regime_r2 0.815).

Regimes (a small tree on its quantities; R² 0.815):

- `` — 6.3% of jets, value 4.49 (2.50…6.12), formula right 60%
- `` — 19.4% of jets, value 3.40 (2.12…4.62), formula right 66%
- `` — 2.0% of jets, value 2.21 (0.00…4.50), formula right 46%
- `` — 3.5% of jets, value 1.68 (0.25…3.12), formula right 52%
- `` — 5.2% of jets, value 0.41 (0.00…1.50), formula right 52%
- `` — 8.5% of jets, value 0.33 (0.00…1.12), formula right 62%
- `` — 7.8% of jets, value 0.32 (0.00…1.12), formula right 44%
- `` — 47.3% of jets, value 0.03 (0.00…0.00), formula right 74%

```
z = -0.441
if width < 0.0049: z += 2510 × (0.0049 − width)
if girth < 0.063 and width < 0.0055: z += -31300 × (0.063 − girth) × (0.0055 − width)
if girth2 < 0.0067 and centroid_offset < 0.025: z += 24600 × (0.0067 − girth2) × (0.025 − centroid_offset)
if max_dr < 0.180 and lam2 < 0.00021: z += 85800 × (0.180 − max_dr) × (0.00021 − lam2)
if C2 < 0.027: z += -92.20 × (0.027 − C2)
if mass < 29.00 and centroid_offset < 0.024: z += -7.46 × (29.00 − mass) × (0.024 − centroid_offset)
if width < 0.0051 and centroid_offset > 0.0071: z += -60500 × (0.0051 − width) × (centroid_offset − 0.0071)
if LHA < 0.200 and width < 0.00067: z += 51300 × (0.200 − LHA) × (0.00067 − width)
if log_sum_pt > 6.70: z += -16.90 × (log_sum_pt − 6.70)
if z_dr_0_0p05 > 0.870 and lam2 < 0.00054: z += -18700 × (z_dr_0_0p05 − 0.870) × (0.00054 − lam2)
if dr_0 < 0.016: z += -152 × (0.016 − dr_0)
if girth < 0.057 and log_sum_pt > 6.70: z += 219 × (0.057 − girth) × (log_sum_pt − 6.70)
if width < 0.0048 and mass_over_sum_pt_sq > 0.00045: z += -402000 × (0.0048 − width) × (mass_over_sum_pt_sq − 0.00045)
if girth2 < 0.0065 and planar_flow < 0.380: z += 523 × (0.0065 − girth2) × (0.380 − planar_flow)
if pt_7 > 34.00: z += -0.035 × (pt_7 − 34.00)
if log_sum_pt > 6.70 and pt_7 < 50.00: z += 0.155 × (log_sum_pt − 6.70) × (50.00 − pt_7)
if mass < 22.00 and centroid_offset > 0.031: z += -28.00 × (22.00 − mass) × (centroid_offset − 0.031)
if centroid_offset < 0.0034: z += 433 × (0.0034 − centroid_offset)
if LHA < 0.200 and mean_phi > -0.00075: z += -912 × (0.200 − LHA) × (mean_phi − -0.00075)
if mass < 22.00 and max_pair_mass > 13.00: z += 61.50 × (22.00 − mass) × (max_pair_mass − 13.00)
if girth2_top5 < 0.00022 and n_dr_0p2_0p4 > 0.029: z += 23500 × (0.00022 − girth2_top5) × (n_dr_0p2_0p4 − 0.029)
if LHA < 0.190 and girth2 > 0.0075: z += 178000 × (0.190 − LHA) × (girth2 − 0.0075)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Massive wide boson/top jets** — 47.5% of jets, neuron 0.06, formula right for 73%. Nearly half of all jets (47%): a top-led mixture (t 38%, Z 26%, W 21%, g 10%) at 60.8 GeV, about twice the average width with 34% of pT beyond 0.1. They are too wide for width < 0.0049 (passes for 0.104), the neuron's main plus, so only small terms remain, such as C2 < 0.027 (-0.464) and max_dr < 0.18 and lam2 < 0.00021 (+0.337) for about half. The neuron is usually zero (on for 0.108, mean 0.06) and leaves the scores alone; the formula calls them t and is right 0.726 of the time, the separation happening in other neurons.
- **Pencil-thin massless quark jets** — 10.6% of jets, neuron 3.37, formula right for 63%. Mostly light quarks (52%) with gluons (37%), essentially massless (6.4 GeV) with all pT within 0.05 of the axis. width < 0.0049 adds the most here (+12.033, it grows the narrower the jet), helped by girth2 < 0.0067 and centroid_offset < 0.025, LHA < 0.2 and width < 0.00067 and max_dr < 0.18 and lam2 < 0.00021, while girth < 0.063 and width < 0.0055 (-9.243), mass < 29 and centroid_offset < 0.024, C2 < 0.027 and dr_0 < 0.016 pull back. The neuron is on for all (3.365), raising the q and t scores and lowering W; the formula calls them q and is right 0.626 of the time, the gluon admixture likely accounting for most errors.
- **Very narrow light gluon/quark jets** — 9.6% of jets, neuron 2.81, formula right for 57%. Gluons (44%) and quarks (35%) with some W, light (11.8 GeV) and very narrow (98% of pT within 0.05). width < 0.0049 (+11.148) against girth < 0.063 and width < 0.0055 (-7.101), with smaller pluses (girth2 < 0.0067 and centroid_offset < 0.025, max_dr < 0.18 and lam2 < 0.00021) and minuses (mass < 29 and centroid_offset < 0.024, C2 < 0.027, width < 0.0051 and centroid_offset > 0.0071). The neuron is high (2.812), raising q and t and lowering W; the formula calls them g but is right only 0.57 of the time.
- **Medium-mass narrowish W-led jets** — 8.7% of jets, neuron 0.72, formula right for 53%. A W-led (39%) mixture with Z (23%), gluons (18%) and quarks (11%) at 37.6 GeV, fairly narrow (74% of pT within 0.05). Being just narrow enough, width < 0.0049 adds only +4.17, and width < 0.0051 and centroid_offset > 0.0071, width < 0.0048 and mass_over_sum_pt_sq > 0.00045 (almost only here and in the next-lightest mixed group) and girth < 0.063 and width < 0.0055 take most of it back. The neuron is on for about half (0.716); the formula calls them W but is right only 0.532 of the time.
- **High-pT pencil-thin quark jets** — 6.8% of jets, neuron 3.74, formula right for 74%. Mostly light quarks (67%), essentially massless, at high pT (1012 GeV) with a 454 GeV leading particle. width < 0.0049 (+12.065), girth2 < 0.0067 and centroid_offset < 0.025, LHA < 0.2 and width < 0.00067 and girth < 0.057 and log_sum_pt > 6.7 add, while girth < 0.063 and width < 0.0055, log_sum_pt > 6.7 and mass < 29 and centroid_offset < 0.024 subtract. This is the neuron's highest value (3.741), raising q and t and lowering W; the formula calls them q and is right 0.744 of the time.
- **Light off-centre mixed jets** — 5.8% of jets, neuron 0.16, formula right for 43%. A mixture (g 31%, Z 25%, W 24%, q 13%), nearly massless (9.2 GeV) but with the pT centroid off the axis, and a softer leading particle (204 GeV). width < 0.0049 (+9.452) is cancelled by width < 0.0051 and centroid_offset > 0.0071 (-5.031, passing for all here) and girth < 0.063 and width < 0.0055, with C2 < 0.027 and z_dr_0_0p05 > 0.87 and lam2 < 0.00054 pushing it below zero. The neuron is usually zero (on for 0.174); the formula calls them g but is right only 0.428 of the time, the worst group of this neuron: low-mass bosons likely get misread.
- **Narrow light gluon-led mixture** — 5.7% of jets, neuron 2.76, formula right for 52%. A gluon-led (35%) mixture with quarks (29%) and W (20%), 28.0 GeV, narrow (88% of pT within 0.05). width < 0.0049 (+8.197) and girth2 < 0.0067 and centroid_offset < 0.025 outweigh girth < 0.063 and width < 0.0055, width < 0.0048 and mass_over_sum_pt_sq > 0.00045 and width < 0.0051 and centroid_offset > 0.0071. The neuron is high (2.76), raising q and t and lowering W; the formula calls them g but is right only 0.524 of the time.
- **High-pT two-prong Z/W jets** — 5.0% of jets, neuron 0.01, formula right for 78%. Z (45%) and W (41%) at 73.0 GeV, high pT (943 GeV) with a 412 GeV leading particle, moderately narrow. Most fail width < 0.0049, and log_sum_pt > 6.7 (-2.476, always passing) plus C2 < 0.027 beat log_sum_pt > 6.7 and pt_7 < 50 and max_dr < 0.18 and lam2 < 0.00021, so the neuron is essentially off (on for 0.02). The formula splits them Z/W and is right 0.783 of the time.
- **Massless jets with off-axis pT** — 0.6% of jets, neuron 0.00, formula right for 54%. A tiny group (0.6%) of gluons (52%), tops (26%) and quarks (13%) with almost no mass (6.6 GeV) yet 75% of pT at 0.05-0.1 from the axis and a soft leading particle (146 GeV): an odd, off-centre configuration. mass < 22 and centroid_offset > 0.031 (-13.26), which fires for all of them and almost never elsewhere, overwhelms width < 0.0049, so the neuron is off for every jet. The formula calls them g and is right only 0.544 of the time.

### neuron 15: Width band above W (Z-like) (moderate)

- **What it measures:** Rises in a width band (width < 0.0067 pushes it down, width < 0.013 up; λ1 < 0.0067 up, λ1 < 0.0083 down) with small e2 (< 0.024 and < 0.041 push it up); it grows with the number of particles at 0.05 ≤ ΔR < 0.1 (+0.384). Largest for Z (1.33), then t (0.64), low for g (0.29), q (0.19) and W (0.17).
- *computed — its value:* largest for Z (1.33), then t (0.64), then g (0.29), then q (0.19), then W (0.17); it separates Z jets from the rest best (AUC 0.74: large for Z)
- **How the class scores use it:** The W score subtracts it (-8%) and the Z score more weakly (-3%); since W jets sit lowest and Z jets highest, the net effect moves jets from W toward Z. It does not enter the g, q or t scores.
- *computed — used by:* lowers the score of W (-8%), Z (-3%); does not (or hardly) enter the score of g, q, t (share of each class score’s average input)
- **Boundaries:** Highest (2.66) for jets with more than 50.8% of pT at ΔR 0.05-0.1 (z_dr_0p05_0p1 > 0.508) and width 0.00667-0.0116, 10% of jets and mostly Z (59%); 1.48 for narrower such jets with e2 ≤ 0.0254, 0.81 for wider ones (width > 0.0116), 0.54 for z_dr_0p05_0p1 ≤ 0.508 with λ1 > 0.00191 and LHA ≤ 0.372 (25%), and off (0.004) for thin low-LHA jets (λ1 ≤ 0.00191, LHA ≤ 0.212; 35% of jets). It subtracts from both bosons but much more from W: -1.83 on W and -0.42 on Z in the top regime (-0.37 / -0.085 in the 25% regime), with +0.17 at most on q. The regimes describe it moderately (regime_r2 0.57).

Regimes (a small tree on its quantities; R² 0.574):

- `` — 10.5% of jets, value 2.66 (1.00…4.25), formula right 73%
- `` — 2.0% of jets, value 1.48 (0.00…3.38), formula right 51%
- `` — 4.1% of jets, value 0.81 (0.00…2.00), formula right 79%
- `` — 24.8% of jets, value 0.54 (0.00…1.75), formula right 63%
- `` — 11.7% of jets, value 0.34 (0.00…1.25), formula right 70%
- `` — 2.0% of jets, value 0.20 (0.00…0.75), formula right 48%
- `` — 10.0% of jets, value 0.04 (0.00…0.00), formula right 82%
- `` — 34.8% of jets, value 0.00 (0.00…0.00), formula right 60%

```
z = -1.86
if width < 0.0067: z += -1390 × (0.0067 − width)
if lam1 < 0.0067: z += 933 × (0.0067 − lam1)
if lam1 < 0.0083: z += -654 × (0.0083 − lam1)
if width < 0.013: z += 315 × (0.013 − width)
if e2 < 0.024: z += 244 × (0.024 − e2)
if e2 < 0.041: z += 64.20 × (0.041 − e2)
if girth > 0.032: z += 33.70 × (girth − 0.032)
if z_dr_0p1_0p2 < 0.320: z += 3.68 × (0.320 − z_dr_0p1_0p2)
if girth2_top2 < 0.0038: z += -436 × (0.0038 − girth2_top2)
if tau21 < 0.230 and z_dr_0p2_0p4 < 0.220: z += 69.70 × (0.230 − tau21) × (0.220 − z_dr_0p2_0p4)
if lam2 < 0.00031: z += -3280 × (0.00031 − lam2)
if mass_over_sum_pt < 0.068: z += -27.10 × (0.068 − mass_over_sum_pt)
if girth2_top2 < 0.0074: z += 89.90 × (0.0074 − girth2_top2)
if LHA > 0.340: z += -35.40 × (LHA − 0.340)
if LHA > 0.180 and sum_pt_top3 > 350: z += 0.049 × (LHA − 0.180) × (sum_pt_top3 − 350)
if width < 0.0076 and e2 > 0.024: z += -66400 × (0.0076 − width) × (e2 − 0.024)
if width < 0.0061 and log_sum_pt > 6.90: z += -10600 × (0.0061 − width) × (log_sum_pt − 6.90)
if log_sum_pt > 6.90: z += 48.60 × (log_sum_pt − 6.90)
if tau21 < 0.270 and girth2_top5 > 0.0084: z += -1160 × (0.270 − tau21) × (girth2_top5 − 0.0084)
if z_dr_0p05_0p1 > 0.750: z += -5.80 × (z_dr_0p05_0p1 − 0.750)
if LHA > 0.310 and pt_dispersion < 0.450: z += -163 × (LHA − 0.310) × (0.450 − pt_dispersion)
if lam1 < 0.0082 and D2 < 0.760: z += -852 × (0.0082 − lam1) × (0.760 − D2)
if tau21 < 0.230 and z_dr_0p05_0p1 < 0.630: z += -7.98 × (0.230 − tau21) × (0.630 − z_dr_0p05_0p1)
if lam1 < 0.0065 and pt1_dr01 > 1.30: z += 33.00 × (0.0065 − lam1) × (pt1_dr01 − 1.30)
if n_dr_0_0p05 < 1.80 and pt_dispersion < 0.430: z += 5.81 × (1.80 − n_dr_0_0p05) × (0.430 − pt_dispersion)
if width < 0.0086 and planar_flow < 0.064: z += -2630 × (0.0086 − width) × (0.064 − planar_flow)
if lam1 < 0.0083 and m01 > 17.00: z += -24.50 × (0.0083 − lam1) × (m01 − 17.00)
if mass > 80.40 and z_dr_0p2_0p4 < 0.220: z += -0.665 × (mass − 80.40) × (0.220 − z_dr_0p2_0p4)
if log_sum_pt > 6.90 and mean_phi > 0.017: z += 17900 × (log_sum_pt − 6.90) × (mean_phi − 0.017)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **Pencil-thin light quark/gluon jets** — 28.0% of jets, neuron 0.01, formula right for 61%. The largest group (28%), mostly light quarks (46%) with gluons (33%), nearly massless (8.1 GeV) with 99% of pT within 0.05. width < 0.0067 (-8.907), lam1 < 0.0083, girth2_top2 < 0.0038 and mass_over_sum_pt < 0.068 outweigh lam1 < 0.0067, e2 < 0.024, width < 0.013 and e2 < 0.041, and they almost never pass girth > 0.032, so the neuron is off (on for 0.017). The formula calls them q and is right only 0.606 of the time.
- **Two-prong W jets** — 14.2% of jets, neuron 0.36, formula right for 69%. Mostly W (54%) with Z (26%), 51.0 GeV, pT split between the core (36%) and 0.05-0.1 (52%). width < 0.013, lam1 < 0.0067, girth > 0.032 and tau21 < 0.23 and z_dr_0p2_0p4 < 0.22 add, but lam1 < 0.0083, width < 0.0067 and width < 0.0076 and e2 > 0.024 (far more common here) subtract, and e2 < 0.024 mostly fails. The neuron is usually zero (on for 0.321, mean 0.357); the formula calls them W and is right 0.687 of the time.
- **Clean two-prong Z jets** — 13.4% of jets, neuron 2.15, formula right for 76%. Mostly Z (57%) with W (21%) and tops (15%), 62.3 GeV, 66% of pT at 0.05-0.1 from the axis: two clear prongs slightly wider than the W groups. All pass tau21 < 0.23 and z_dr_0p2_0p4 < 0.22 (+2.276), and nearly all width < 0.013 and girth > 0.032, while only a third reach the width < 0.0067 penalty; lam2 < 0.00031, lam1 < 0.0083 and width < 0.0076 and e2 > 0.024 take a little back. This is the neuron's highest value (2.149, on for 0.9), lowering the W score much more than Z (and nudging q up), which helps separate Z from W; the formula calls them Z and is right 0.765 of the time.
- **Moderately wide top-led jets** — 11.0% of jets, neuron 1.11, formula right for 69%. Mostly tops (55%) with gluons (18%) and Z (17%), 60.5 GeV, about twice the average width with 30% of pT beyond 0.1 and lower pT (584 GeV). They escape the narrow-jet penalties (width < 0.0067 and lam1 < 0.0083 mostly fail) and gain from girth > 0.032 (+2.08), width < 0.013 for most, z_dr_0p1_0p2 < 0.32 and, for about half, tau21 < 0.23 and z_dr_0p2_0p4 < 0.22, against LHA > 0.34 and lam2 < 0.00031. The neuron is on for 0.776 (1.111), lowering W; the formula calls them t and is right 0.691 of the time.
- **Compact medium-mass W-led jets** — 10.9% of jets, neuron 0.36, formula right for 56%. A W-led (40%) mixture with Z (25%) and gluons (16%), 40.5 GeV, fairly narrow (70% of pT within 0.05). width < 0.0067 (-4.182) and lam1 < 0.0083 roughly cancel lam1 < 0.0067 and width < 0.013, and the rest (e2 < 0.041, z_dr_0p1_0p2 < 0.32, e2 < 0.024, girth > 0.032 against mass_over_sum_pt < 0.068) leaves the neuron on for only 0.355 (0.36). The formula calls them W but is right only 0.563 of the time.
- **Light compact gluon-led mixture** — 10.4% of jets, neuron 0.15, formula right for 52%. A gluon-led (36%) mixture with quarks (24%), W (20%) and Z (14%), 23.4 GeV, narrow (88% of pT within 0.05). width < 0.0067 (-6.973), lam1 < 0.0083, girth2_top2 < 0.0038 and mass_over_sum_pt < 0.068 outweigh lam1 < 0.0067, width < 0.013, e2 < 0.024 and e2 < 0.041, so the neuron is usually zero (on for 0.173). The formula calls them g but is right only 0.518 of the time.
- **Wide top jets** — 6.6% of jets, neuron 0.05, formula right for 88%. Mostly tops (87%), 81.2 GeV, four times the average width with 81% of pT beyond 0.1. They fail width < 0.013 and lam1 < 0.0083, and girth > 0.032 (+4.004) is cancelled by LHA > 0.34 (-3.03) and LHA > 0.31 and pt_dispersion < 0.45, so the neuron is almost always zero (on for 0.07). The formula calls them t and is right 0.876 of the time.
- **Wide tops with small tau21** — 3.8% of jets, neuron 0.02, formula right for 74%. Tops (70%) and gluons (20%), 84.1 GeV, wide with 87% of pT beyond 0.1. girth > 0.032 and, for most, tau21 < 0.23 and z_dr_0p2_0p4 < 0.22 add, but tau21 < 0.27 and girth2_top5 > 0.0084 (-3.273, rare elsewhere), LHA > 0.34, LHA > 0.31 and pt_dispersion < 0.45 and tau21 < 0.23 and z_dr_0p05_0p1 < 0.63 take it back, so the neuron is almost always zero (on for 0.028). The formula calls them t and is right 0.741 of the time.
- **High-pT narrow quark/gluon jets** — 1.3% of jets, neuron 0.18, formula right for 66%. Quarks (43%) and gluons (40%) at high pT (1128 GeV) with a 506 GeV leading particle, light (16.9 GeV) and very narrow. width < 0.0061 and log_sum_pt > 6.9 (-7.38) and log_sum_pt > 6.9 (+6.167), which fire almost only in this and the next group, roughly cancel, and the usual narrow-jet balance (width < 0.0067 and lam1 < 0.0083 against lam1 < 0.0067, e2 < 0.024 and width < 0.013) leaves the neuron usually zero (on for 0.088). Although quarks are the largest class, the formula calls them g and is right 0.658 of the time.
- **Highest-pT narrow gluon jets** — 0.3% of jets, neuron 0.13, formula right for 63%. A tiny group (0.3%), mostly gluons (59%, q 26%), at the highest pT (1384 GeV) with a 642 GeV leading particle, light and very narrow. width < 0.0061 and log_sum_pt > 6.9 (-19.606) outweighs log_sum_pt > 6.9 (+15.908), and the narrow-jet terms cancel as for other light jets, so the neuron is usually zero (on for 0.065). The formula calls them g and is right 0.633 of the time.

### neuron 12: Very wide, busy jet (minor)

- **What it measures:** Switched on only for very wide jets (girth2 > 0.019, more so with pT_7 > 15 GeV, a little more for mass > 91.2 GeV) and cut off when e2 > 0.063; it tracks the number of particles above 10 GeV (+0.849). Small for all types: t (0.23), g (0.10), q (0.03), W (0.00), Z (0.00).
- *computed — its value:* largest for t (0.23), then g (0.10), then q (0.03), then W (0.00), then Z (0.00); it separates t jets from the rest best (AUC 0.57: large for t)
- **How the class scores use it:** It does not enter any class score appreciably (at most a -0.009 share, in the t score), so it has almost no effect on the decision.
- *computed — used by:* ; does not (or hardly) enter the score of g, q, W, Z, t (share of each class score’s average input)
- **Boundaries:** Off (0.001) for 94% of jets (girth2 ≤ 0.0251 and mass ≤ 97.3 GeV); it switches on only for the very widest jets (girth2 > 0.0305: 2.22, 2.1% of jets, 75% t) and for heavy jets above 97.3 GeV (1.08, 2%, 78% t), with 0.29 for 0.0251 < girth2 ≤ 0.0305. It enters only the t score, subtracting -1.11 and -0.54 in those two regimes, so it trims the top score for extreme, busy jets; its effect on the average is tiny. The regimes describe it only roughly (regime_r2 0.43).

Regimes (a small tree on its quantities; R² 0.432):

- `` — 2.1% of jets, value 2.22 (0.00…5.00), formula right 79%
- `` — 2.0% of jets, value 1.08 (0.00…2.62), formula right 84%
- `` — 2.2% of jets, value 0.28 (0.00…1.12), formula right 88%
- `` — 93.8% of jets, value 0.00 (0.00…0.00), formula right 65%

```
z = -1.38
if e2 > 0.063: z += -119 × (e2 − 0.063)
if girth2 > 0.019: z += 232 × (girth2 − 0.019)
if girth2 > 0.019 and pt_7 > 15.00: z += 6.87 × (girth2 − 0.019) × (pt_7 − 15.00)
if mass > 91.20: z += 0.099 × (mass − 91.20)
if girth2 > 0.015 and lam2 > 5.6e-06: z += 8450 × (girth2 − 0.015) × (lam2 − 5.6e-06)
h = max(0, z)
```

Groups of jets (every jet in one group; formed by how all its if-statements add up):

- **All ordinary-width jets** — 91.2% of jets, neuron 0.00, formula right for 64%. Almost all jets (91%), an even mix of every class (g, q, W and Z about 21-22% each, t 14%) at 35.8 GeV and roughly average width and pT. None of this neuron's few if-statements (girth2 > 0.019, girth2 > 0.019 and pt_7 > 15, girth2 > 0.015 and lam2 > 5.6e-06, mass > 91.2, e2 > 0.063) pass for more than about 5% of them, so the neuron is off (on for 0.001) and adds nothing. The formula's accuracy on them (0.64) is decided entirely by other neurons.
- **Wide tops, e2 cut dominates** — 2.1% of jets, neuron 0.16, formula right for 90%. Almost all tops (90%), 81.4 GeV, about four times the average width with 78% of pT beyond 0.1. They pass girth2 > 0.019 (+1.604), girth2 > 0.019 and pt_7 > 15 and girth2 > 0.015 and lam2 > 5.6e-06, but also e2 > 0.063 (-2.35), which roughly cancels them, so the neuron is on for only 0.225 (mean 0.164) and slightly lowers the t score. The formula calls them t and is right 0.895 of the time.
- **Moderately wide tops, neuron off** — 2.0% of jets, neuron 0.00, formula right for 84%. Mostly tops (84%, g 12%), 75.5 GeV, over three times the average width. e2 > 0.063 (-1.244) always passes while girth2 > 0.019 and its companions pass for 0.876 but add less because the jets are only just over the cut, so the neuron is off (on for 0.009). The formula calls them t and is right 0.836 of the time.
- **Very wide tops, half on** — 1.7% of jets, neuron 0.53, formula right for 89%. Mostly tops (88%), 83.3 GeV, nearly five times the average width with 86% of pT beyond 0.1. girth2 > 0.019 (+2.708), girth2 > 0.019 and pt_7 > 15 and girth2 > 0.015 and lam2 > 5.6e-06 all pass and together roughly match e2 > 0.063 (-3.449), so the neuron is on for about half (0.52, mean 0.53), mildly lowering the t score. The formula calls them t and is right 0.889 of the time.
- **Wide jets escaping the e2 cut** — 0.8% of jets, neuron 0.81, formula right for 76%. Tops (75%) and gluons (19%), 76.4 GeV, four times the average width. All pass girth2 > 0.019 and girth2 > 0.019 and pt_7 > 15, and a quarter escape e2 > 0.063, so the neuron is on for 0.727 (mean 0.809), lowering the t score a little. The formula calls them t and is right 0.763 of the time, the gluon admixture likely accounting for most errors.
- **Very wide soft-leading tops** — 0.7% of jets, neuron 2.20, formula right for 80%. Tops (78%) and gluons (17%), 88.5 GeV, nearly six times the average width with 90% of pT beyond 0.1 and a soft leading particle (114 GeV). The width terms girth2 > 0.019 (+4.256), girth2 > 0.019 and pt_7 > 15 and girth2 > 0.015 and lam2 > 5.6e-06 grow with width and beat e2 > 0.063 (-4.583), with mass > 91.2 helping for about half. The neuron is on (2.197), lowering the t score; the formula calls them t and is right 0.803 of the time.
- **Very heavy wide top jets** — 0.7% of jets, neuron 2.73, formula right for 81%. Tops (75%) and gluons (17%), the heavy side (116.8 GeV), very wide, at near-average pT (682 GeV). mass > 91.2 (+2.544), which rarely passes elsewhere, joins girth2 > 0.019 and girth2 > 0.019 and pt_7 > 15 in beating e2 > 0.063 (-3.182), so the neuron is on for all (2.728), lowering the t score. The formula calls them t and is right 0.814 of the time.
- **Heavy high-pT top/gluon jets** — 0.6% of jets, neuron 1.38, formula right for 79%. Tops (65%) and gluons (24%), heavy (115.9 GeV), less wide than the other groups here (0.0205) but with high pT (838 GeV) and a 315 GeV leading particle. mass > 91.2 (+2.457) always passes, while girth2 > 0.019 and e2 > 0.063 each pass for about two thirds and roughly cancel, so the neuron is on for 0.879 (1.382), lowering t. The formula calls them t and is right 0.794 of the time.
- **Widest diffuse jets** — 0.1% of jets, neuron 5.79, formula right for 66%. A tiny group (0.14%) of tops (65%) and gluons (28%), 91.1 GeV, the widest jets (0.0486) with 94% of pT beyond 0.1 and a very soft leading particle (97 GeV). The width terms are largest here (girth2 > 0.019 +6.862, girth2 > 0.019 and pt_7 > 15, girth2 > 0.015 and lam2 > 5.6e-06) and beat e2 > 0.063 (-6.425), so the neuron is high (5.794), lowering the t score strongly. The formula calls them t but is right only 0.659 of the time.
- **Very heavy wide gluon jets** — 0.1% of jets, neuron 10.25, formula right for 67%. A handful of jets (0.05%), mostly gluons (73%, q 17%), with the largest mass (174.2 GeV), very wide, high pT (952 GeV) and many hard particles (8th at 47.6 GeV). mass > 91.2 (+8.246) plus girth2 > 0.019 and girth2 > 0.019 and pt_7 > 15 far outweigh e2 > 0.063, giving the neuron's highest value (10.254), which strongly lowers the t score and so keeps these top-like heavy jets from being called tops. The formula calls them g and is right 0.667 of the time.
