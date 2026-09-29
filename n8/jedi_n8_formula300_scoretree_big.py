"""JEDI-linear jet tagger, 8 particles, 3 features: the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. scores():     5 class scores, each a plain sum of the formula's own terms of the quantities
                 (coef * max(0, Q.x - t): only counts when x > t;  coef * max(0, t - Q.x): only when x < t).
                 Coefficients fitted on the entire training set (595,000 jets) to reproduce the formula's
                 class probabilities.
3. decide():     one tree; each test is "quantity > threshold" or "score_c - score_d > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 65.34% (the formula: 65.49%); same class as the formula for 95.56% of jets.  882 leaves, depth 21.
"""

import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
def quantities(pt, eta, phi):
    pt, eta, phi = [float(x) for x in pt], [float(x) for x in eta], [float(x) for x in phi]
    n = len(pt)
    P = range(n)
    real = [i for i in P if pt[i] > 0]
    tot = sum(pt)
    z = [x / tot for x in pt]
    zs = sorted(z, reverse=True)
    dr = [math.hypot(eta[i], phi[i]) for i in P]

    def dist2(i, j):
        return (eta[i] - eta[j]) ** 2 + (phi[i] - phi[j]) ** 2

    def mass_of(k):
        E = sum(pt[i] * math.cosh(eta[i]) for i in range(k))
        px = sum(pt[i] * math.cos(phi[i]) for i in range(k))
        py = sum(pt[i] * math.sin(phi[i]) for i in range(k))
        pz = sum(pt[i] * math.sinh(eta[i]) for i in range(k))
        return math.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0))

    def pair_mass(i, j):
        return math.sqrt(max(2 * pt[i] * pt[j] * (math.cosh(eta[i] - eta[j]) - math.cos(phi[i] - phi[j])), 0.0))

    def tau(k):
        axes = [(eta[max(P, key=lambda i: pt[i])], phi[max(P, key=lambda i: pt[i])])]
        for _ in range(1, k):
            far = max(P, key=lambda i: pt[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes))
            axes.append((eta[far], phi[far]))
        for _ in range(6):
            nearest = [min(range(k), key=lambda j: math.hypot(eta[i] - axes[j][0], phi[i] - axes[j][1])) for i in P]
            for j in range(k):
                w = sum(pt[i] for i in P if nearest[i] == j)
                if w > 0:
                    axes[j] = (sum(pt[i] * eta[i] for i in P if nearest[i] == j) / w, sum(pt[i] * phi[i] for i in P if nearest[i] == j) / w)
        return sum(z[i] * min(math.hypot(eta[i] - a, phi[i] - b) for a, b in axes) for i in P) / 0.8

    ta = sum(z[i] * eta[i] ** 2 for i in P)
    tb = sum(z[i] * eta[i] * phi[i] for i in P)
    tc = sum(z[i] * phi[i] ** 2 for i in P)
    disc = math.sqrt(max((ta - tc) ** 2 / 4 + tb ** 2, 0.0))
    lam1, lam2 = (ta + tc) / 2 + disc, max((ta + tc) / 2 - disc, 0.0)

    hard = sorted(P, key=lambda i: -pt[i])[:24]
    R = {(i, j): math.sqrt(dist2(i, j)) for i in hard for j in hard}
    e2 = sum(z[i] * z[j] * R[i, j] for i in hard for j in hard if i < j)
    e3 = sum(z[i] * z[j] * z[k] * R[i, j] * R[i, k] * R[j, k] for i in hard for j in hard for k in hard if i < j < k)

    return SimpleNamespace(
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def score_g(Q):
    return (6.334
        + 0.02478 * max(0.0, Q.D2 - 3.7)
        - 61.46 * max(0.0, 0.0252 - Q.e2)
        - 32.4 * max(0.0, Q.eccentricity - 0.997)
        + 31.74 * max(0.0, 0.0768 - Q.girth)
        - 161.4 * max(0.0, 0.0128 - Q.girth2)
        + 391.3 * max(0.0, 0.000706 - Q.lam1)
        - 50.42 * max(0.0, 0.00142 - Q.lam1)
        + 0.02072 * max(0.0, 29.3 - Q.mass)
        - 0.0003473 * max(0.0, 59.8 - Q.mass)
        + 0.007762 * max(0.0, Q.sum_pt - 816.0)
        - 0.0002339 * max(0.0, Q.sum_pt - 890.0)
        - 8.001e-05 * max(0.0, Q.sum_pt_top5 - 699.0)
        - 141.4 * max(0.0, 0.00459 - Q.width)
        - 178.9 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2)
        + 0.01157 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184)
        - 0.5041 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133)
        + 9.35e-05 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7)
        + 0.1049 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset)
        - 6.665e-05 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9)
        + 4.427e-05 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7)
        + 1.01 * max(0.0, Q.LHA - 0.282)
        - 6.138 * max(0.0, Q.e2 - 0.0316)
        - 64.32 * max(0.0, 0.00749 - Q.e2)
        + 0.9103 * max(0.0, Q.log_sum_pt - 6.4)
        - 2.283 * max(0.0, 0.252 - Q.max_dr)
        + 0.02903 * max(0.0, Q.pt_7 - 33.5)
        - 18.31 * max(0.0, 0.0515 - Q.z_7)
        - 38.32 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21)
        - 112.2 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 6743.0 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201)
        - 971.9 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272)
        + 2.639 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2)
        - 60.38 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset)
        + 2303.0 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2)
        - 5.908 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr)
        + 0.7566 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188)
        + 8505.0 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2)
        - 346.7 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow)
        - 12.36 * Q.LHA
        - 17010.0 * max(0.0, 4.88e-05 - Q.girth2)
        + 218.8 * max(0.0, 0.00591 - Q.lam1)
        - 6.811 * max(0.0, Q.log_sum_pt - 6.93)
        + 1.809 * max(0.0, 6.43 - Q.log_sum_pt)
        - 0.04216 * max(0.0, 36.7 - Q.mass)
        - 0.6967 * max(0.0, 0.748 - Q.planar_flow)
        + 0.02546 * max(0.0, Q.pt_7 - 31.0)
        - 0.188 * max(0.0, 54.0 - Q.pt_7)
        + 0.0104 * max(0.0, 794.0 - Q.sum_pt)
        - 0.001182 * max(0.0, 714.0 - Q.sum_pt_top5)
        - 23170.0 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset)
        - 734.9 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819)
        + 31.91 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2)
        - 0.0005594 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4)
        - 0.1742 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7)
        - 0.7777 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2)
        - 0.4502 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132)
        - 0.1497 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111)
        - 5301.0 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width)
        - 3.597 * max(0.0, Q.C2 - 0.0894)
        + 0.9783 * max(0.0, Q.LHA - 0.326)
        + 11.04 * max(0.0, Q.centroid_offset - 0.0106)
        - 291.0 * max(0.0, Q.centroid_offset - 0.0495)
        + 8.778 * max(0.0, Q.e2 - 0.0263)
        - 11.15 * max(0.0, 0.0397 - Q.e2)
        - 5.947 * max(0.0, 0.121 - Q.girth)
        + 264.9 * max(0.0, 0.00863 - Q.girth2)
        + 0.004845 * max(0.0, Q.mass - 75.6)
        + 3.849 * max(0.0, Q.max_dr - 0.122)
        + 73.27 * max(0.0, Q.width - 0.0185)
        - 162.9 * max(0.0, 0.00539 - Q.width)
        + 64.62 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957)
        + 95.08 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr)
        - 1.24 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227)
        + 0.6182 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5)
        - 0.1561 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68)
        + 605.1 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957)
        - 0.04158 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7)
        - 567.5 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7)
        - 0.009786 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705)
        + 0.3299 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr)
        + 317.7 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr)
        + 12.53 * max(0.0, Q.C2 - 0.0604)
        - 9.309 * max(0.0, 0.0602 - Q.C2)
        - 2.067 * max(0.0, 0.0429 - Q.centroid_offset)
        + 1.076 * max(0.0, Q.e2 - 0.0328)
        + 13.63 * max(0.0, Q.girth - 0.0651)
        - 0.000832 * max(0.0, 47.9 - Q.mass)
        + 129.3 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 2.118 * max(0.0, Q.mass_over_sum_pt - 0.109)
        + 0.0008468 * max(0.0, 762.0 - Q.sum_pt)
        - 0.003527 * max(0.0, 439.0 - Q.sum_pt_top5)
        - 1.562 * max(0.0, 0.238 - Q.tau21)
        - 266.9 * max(0.0, Q.width - 0.000872)
        - 0.02514 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6)
        + 0.1014 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7)
        + 132.1 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982)
        + 0.04596 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass)
        - 1.466 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357)
        + 0.02787 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7)
        + 37.73 * max(0.0, 0.0405 - Q.e2)
        - 7715.0 * max(0.0, 5.65e-05 - Q.girth2)
        + 0.1599 * max(0.0, 54.4 - Q.pt_7)
        + 0.009833 * max(0.0, Q.sum_pt - 935.0)
        + 0.001319 * max(0.0, 721.0 - Q.sum_pt_top5)
        - 52.76 * max(0.0, 0.0243 - Q.z_7)
        + 657.5 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367)
        - 0.1944 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt)
        + 46.42 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241)
        + 202.1 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151)
        - 0.06079 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset)
        - 10640.0 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset)
        + 533.1 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset)
        + 0.05046 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        - 2.638 * max(0.0, Q.C2 - 0.0419)
        + 8.415 * max(0.0, Q.LHA - 0.313)
        - 37.62 * max(0.0, Q.centroid_offset - 0.0525)
        - 35.06 * max(0.0, 0.0508 - Q.e2)
        + 149.6 * max(0.0, 0.00371 - Q.girth2)
        - 783.9 * max(0.0, 0.00868 - Q.girth2)
        + 40.57 * max(0.0, 0.00819 - Q.lam1)
        - 191.7 * max(0.0, Q.lam2 - 0.000765)
        + 0.7583 * max(0.0, 6.33 - Q.log_sum_pt)
        - 5.34 * max(0.0, 6.72 - Q.log_sum_pt)
        + 3.724 * max(0.0, 0.132 - Q.mass_over_sum_pt)
        + 1.747 * max(0.0, Q.max_dr - 0.0866)
        - 3.586 * max(0.0, 0.0452 - Q.max_dr)
        - 3.802 * max(0.0, 0.0747 - Q.planar_flow)
        - 0.001444 * max(0.0, Q.sum_pt - 997.0)
        + 125.9 * max(0.0, 0.0135 - Q.width)
        + 0.1912 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0)
        - 22.99 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2)
        + 113.6 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        + 0.05665 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7)
        - 34.15 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7)
        + 45.71 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7)
        - 0.0989 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7)
        - 14.82 * max(0.0, 0.236 - Q.LHA)
        + 11.07 * max(0.0, Q.centroid_offset - 0.0284)
        - 19.94 * max(0.0, 0.0204 - Q.centroid_offset)
        - 14.29 * max(0.0, 0.0384 - Q.e2)
        + 141.6 * max(0.0, 0.0501 - Q.e2)
        - 6.2 * max(0.0, Q.eccentricity - 0.955)
        - 83.4 * max(0.0, Q.girth2 - 0.00741)
        + 512.0 * max(0.0, Q.girth2 - 0.00874)
        + 59.32 * max(0.0, 0.000555 - Q.girth2)
        + 339.7 * max(0.0, 0.00845 - Q.lam1)
        - 0.03246 * max(0.0, Q.mass - 80.4)
        - 0.02207 * max(0.0, 27.6 - Q.mass)
        + 8.834 * max(0.0, Q.mass_over_sum_pt - 0.0849)
        - 142.0 * max(0.0, Q.mass_over_sum_pt - 0.0905)
        + 81.18 * max(0.0, 0.00524 - Q.width)
        - 61.08 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213)
        + 241.1 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065)
        + 0.04428 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0)
        + 21.47 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2)
        + 57.61 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21)
        - 636.3 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942)
        - 72.6 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13)
        - 32.26 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2)
        + 0.5133 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50)
        + 0.09161 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934)
        - 2.06 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2)
        - 0.004153 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0)
        + 0.01253 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow)
        - 5.706 * max(0.0, 0.0262 - Q.C2)
        + 19.46 * max(0.0, 0.234 - Q.LHA)
        - 2.944 * max(0.0, 0.00353 - Q.centroid_offset)
        + 0.0007742 * max(0.0, 24.1 - Q.mass)
        + 1.782 * max(0.0, 0.188 - Q.max_dr)
        + 223.3 * max(0.0, 0.0055 - Q.width)
        - 70.65 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66)
        + 41.68 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow)
        + 25.73 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2)
        - 0.07754 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7)
        + 0.5988 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794)
        - 0.01658 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 1987.0 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2)
        - 5560.0 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627)
        - 26.18 * max(0.0, 0.0383 - Q.girth)
        + 54.23 * max(0.0, Q.girth2 - 0.0241)
        + 78.45 * max(0.0, Q.lam2 - 0.00101)
        + 0.5416 * max(0.0, Q.log_sum_pt - 6.37)
        + 0.03933 * max(0.0, Q.mass - 40.0)
        + 4.146 * max(0.0, 0.194 - Q.max_dr)
        - 0.0005897 * max(0.0, Q.sum_pt - 863.0)
        - 0.001197 * max(0.0, Q.sum_pt_top5 - 446.0)
        + 363.1 * max(0.0, 0.00617 - Q.width)
        + 3719.0 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset)
        - 284.1 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907)
        - 0.1394 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7)
        - 90.49 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21)
        - 8296.0 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012)
        - 2.626 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset)
        + 1.783 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset)
        - 18.99 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1)
        + 0.05704 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 0.02399 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow)
        - 1027.0 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308)
        - 7652.0 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326)
        + 1.935 * max(0.0, Q.C2 - 0.0492)
        - 2.873 * max(0.0, Q.LHA - 0.24)
        - 13.74 * max(0.0, 0.0634 - Q.girth)
        - 10.13 * max(0.0, Q.girth2 - 0.00809)
        - 196.2 * max(0.0, 0.00157 - Q.girth2)
        - 1.011 * max(0.0, 0.00516 - Q.lam1)
        + 18.44 * max(0.0, 0.00362 - Q.lam2)
        - 0.3218 * max(0.0, 6.28 - Q.log_sum_pt)
        - 0.01979 * max(0.0, Q.mass - 39.2)
        - 4.784 * max(0.0, Q.max_dr - 0.113)
        - 0.008572 * max(0.0, Q.sum_pt - 792.0)
        + 0.3227 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21)
        - 45.7 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21)
        - 4.638 * max(0.0, Q.centroid_offset - 0.0143)
        - 0.5127 * max(0.0, 0.0501 - Q.centroid_offset)
        + 1.469 * max(0.0, Q.girth - 0.0709)
        - 6.742 * max(0.0, 0.0204 - Q.girth)
        + 4.563 * max(0.0, 0.124 - Q.girth)
        + 0.02081 * max(0.0, Q.mass - 91.2)
        + 14.59 * max(0.0, Q.mass_over_sum_pt - 0.0933)
        + 0.03743 * max(0.0, 0.289 - Q.planar_flow)
        - 0.003549 * max(0.0, 674.0 - Q.sum_pt_top5)
        - 227.5 * max(0.0, 0.0034 - Q.width)
        - 210.7 * max(0.0, 0.00909 - Q.width)
        + 10.37 * max(0.0, 0.0424 - Q.z_7)
        + 21.1 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7)
        - 19.34 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt)
        + 5.516 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21)
        + 0.00831 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7)
        - 268.4 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161)
        - 0.006966 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass)
        + 1.499 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121)
        + 138.5 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586)
        + 49.68 * max(0.0, Q.e2 - 0.0629)
        - 21.22 * max(0.0, Q.girth2 - 0.0193)
        - 3903.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06)
        - 1.406 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4)
        + 11.38 * max(0.0, Q.C2 - 0.0657)
        + 14.82 * max(0.0, 0.0387 - Q.centroid_offset)
        - 23.0 * max(0.0, 0.0489 - Q.e2)
        + 0.3551 * max(0.0, 0.143 - Q.girth)
        + 148.0 * max(0.0, 0.00647 - Q.lam1)
        - 24.6 * max(0.0, 0.0157 - Q.lam1)
        - 413.9 * max(0.0, 0.00741 - Q.width)
        + 12.62 * max(0.0, 0.0286 - Q.z_7)
        + 5.169 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt)
        - 0.1583 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
        + 332.4 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset)
        - 5010.0 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset)
        - 7.467e-05 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2)
        - 0.0007542 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08)
        + 8.143e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7)
        + 0.06998 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023)
        + 2.083 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292)
        - 11.09 * max(0.0, Q.C2 - 0.0359)
        - 15.71 * max(0.0, Q.C2 - 0.0669)
        + 12.25 * max(0.0, 0.0356 - Q.C2)
        - 8.883 * max(0.0, Q.centroid_offset - 0.0296)
        + 320.6 * max(0.0, Q.centroid_offset - 0.0496)
        - 14.15 * max(0.0, 0.0437 - Q.e2)
        - 7.082 * max(0.0, 0.0332 - Q.girth)
        - 4.723 * max(0.0, 0.0879 - Q.girth)
        + 313.5 * max(0.0, 0.00457 - Q.girth2)
        + 201.7 * max(0.0, 0.0137 - Q.girth2)
        - 79.83 * max(0.0, Q.lam1 - 0.00419)
        - 147.0 * max(0.0, Q.lam1 - 0.00616)
        + 0.002811 * max(0.0, 457.0 - Q.sum_pt_top5)
        + 0.9216 * max(0.0, 0.136 - Q.tau21)
        + 361.9 * max(0.0, 0.00735 - Q.width)
        - 17.73 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset)
        - 85.0 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2)
        - 700.7 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968)
        - 34.89 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419)
        + 10.71 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65)
        + 92.88 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395)
        - 51.22 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378)
        - 2139.0 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr)
        + 0.003134 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2)
        - 151.4 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913)
        + 167.4 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184)
        + 28.63 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        - 0.0002714 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt)
        + 456.9 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 480.3 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow)
        - 0.2172 * max(0.0, Q.LHA - 0.343)
        - 9.476 * max(0.0, 0.305 - Q.LHA)
        + 24.02 * max(0.0, Q.e2 - 0.0244)
        - 86.32 * max(0.0, Q.e2 - 0.0497)
        - 45.22 * max(0.0, 0.0627 - Q.e2)
        + 8.073 * max(0.0, Q.girth - 0.0329)
        - 378.4 * max(0.0, 0.00855 - Q.lam1)
        - 64.76 * max(0.0, 0.000335 - Q.lam2)
        + 44.38 * max(0.0, 0.00323 - Q.lam2)
        - 4.727 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.03088 * max(0.0, 80.4 - Q.mass)
        + 0.1764 * max(0.0, 0.25 - Q.tau21)
        - 307.8 * max(0.0, 0.0136 - Q.width)
        + 9565.0 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244)
        + 1308.0 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 86.57 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow)
    )


def score_q(Q):
    return (3.909
        - 0.01067 * max(0.0, Q.D2 - 3.7)
        - 60.41 * max(0.0, 0.0252 - Q.e2)
        + 3.526 * max(0.0, Q.eccentricity - 0.997)
        + 16.92 * max(0.0, 0.0768 - Q.girth)
        - 73.88 * max(0.0, 0.0128 - Q.girth2)
        + 1049.0 * max(0.0, 0.000706 - Q.lam1)
        + 27.56 * max(0.0, 0.00142 - Q.lam1)
        + 0.01173 * max(0.0, 29.3 - Q.mass)
        - 0.01403 * max(0.0, 59.8 - Q.mass)
        + 0.002456 * max(0.0, Q.sum_pt - 816.0)
        - 0.0004533 * max(0.0, Q.sum_pt - 890.0)
        - 0.0008382 * max(0.0, Q.sum_pt_top5 - 699.0)
        - 324.8 * max(0.0, 0.00459 - Q.width)
        - 163.0 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2)
        + 0.2462 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184)
        - 0.4221 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133)
        - 0.0001068 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7)
        - 25.65 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset)
        + 1.487e-05 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9)
        - 4.879e-05 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7)
        - 1.63 * max(0.0, Q.LHA - 0.282)
        + 13.44 * max(0.0, Q.e2 - 0.0316)
        - 31.19 * max(0.0, 0.00749 - Q.e2)
        + 0.4893 * max(0.0, Q.log_sum_pt - 6.4)
        - 0.7839 * max(0.0, 0.252 - Q.max_dr)
        - 0.007162 * max(0.0, Q.pt_7 - 33.5)
        + 2.816 * max(0.0, 0.0515 - Q.z_7)
        + 6.522 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21)
        - 781.1 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 3583.0 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201)
        - 226.8 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272)
        + 0.1765 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2)
        + 65.85 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset)
        - 428.9 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2)
        - 5.0 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr)
        - 0.1308 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188)
        - 1497.0 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2)
        + 810.0 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow)
        - 10.17 * Q.LHA
        - 3939.0 * max(0.0, 4.88e-05 - Q.girth2)
        + 236.2 * max(0.0, 0.00591 - Q.lam1)
        - 0.2201 * max(0.0, Q.log_sum_pt - 6.93)
        - 1.017 * max(0.0, 6.43 - Q.log_sum_pt)
        + 0.001769 * max(0.0, 36.7 - Q.mass)
        - 0.2126 * max(0.0, 0.748 - Q.planar_flow)
        - 0.01818 * max(0.0, Q.pt_7 - 31.0)
        - 0.1113 * max(0.0, 54.0 - Q.pt_7)
        + 0.00413 * max(0.0, 794.0 - Q.sum_pt)
        - 0.0006274 * max(0.0, 714.0 - Q.sum_pt_top5)
        - 8903.0 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset)
        - 265.1 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819)
        + 3.219 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2)
        + 5.502e-05 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4)
        - 0.008568 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7)
        + 0.4753 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2)
        - 0.1228 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132)
        - 0.02836 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111)
        + 148.3 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width)
        - 8.114 * max(0.0, Q.C2 - 0.0894)
        - 1.272 * max(0.0, Q.LHA - 0.326)
        + 12.35 * max(0.0, Q.centroid_offset - 0.0106)
        - 191.2 * max(0.0, Q.centroid_offset - 0.0495)
        + 17.98 * max(0.0, Q.e2 - 0.0263)
        - 11.71 * max(0.0, 0.0397 - Q.e2)
        + 1.868 * max(0.0, 0.121 - Q.girth)
        + 325.0 * max(0.0, 0.00863 - Q.girth2)
        + 0.0007604 * max(0.0, Q.mass - 75.6)
        + 0.9568 * max(0.0, Q.max_dr - 0.122)
        + 84.13 * max(0.0, Q.width - 0.0185)
        - 189.3 * max(0.0, 0.00539 - Q.width)
        + 5.163 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957)
        + 181.8 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr)
        - 0.8747 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227)
        + 0.5183 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5)
        - 0.1928 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68)
        + 1942.0 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957)
        - 0.5218 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7)
        - 324.4 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7)
        - 0.00637 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705)
        + 0.1728 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr)
        - 120.8 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr)
        + 4.856 * max(0.0, Q.C2 - 0.0604)
        + 1.806 * max(0.0, 0.0602 - Q.C2)
        - 4.423 * max(0.0, 0.0429 - Q.centroid_offset)
        - 4.685 * max(0.0, Q.e2 - 0.0328)
        + 13.94 * max(0.0, Q.girth - 0.0651)
        - 0.01005 * max(0.0, 47.9 - Q.mass)
        + 90.66 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        - 4.137 * max(0.0, Q.mass_over_sum_pt - 0.109)
        + 0.0007141 * max(0.0, 762.0 - Q.sum_pt)
        - 0.006896 * max(0.0, 439.0 - Q.sum_pt_top5)
        - 3.657 * max(0.0, 0.238 - Q.tau21)
        - 616.5 * max(0.0, Q.width - 0.000872)
        - 0.2167 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6)
        - 0.04204 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7)
        + 106.3 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982)
        + 0.06716 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass)
        - 4.458 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357)
        + 0.1134 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7)
        + 31.24 * max(0.0, 0.0405 - Q.e2)
        + 7908.0 * max(0.0, 5.65e-05 - Q.girth2)
        + 0.1131 * max(0.0, 54.4 - Q.pt_7)
        + 6.951e-05 * max(0.0, Q.sum_pt - 935.0)
        + 0.0008286 * max(0.0, 721.0 - Q.sum_pt_top5)
        + 8.769 * max(0.0, 0.0243 - Q.z_7)
        - 46.56 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367)
        - 21.14 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt)
        + 61.25 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241)
        + 107.9 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151)
        + 0.02652 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset)
        + 8235.0 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset)
        - 57.91 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset)
        + 0.02172 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        - 2.81 * max(0.0, Q.C2 - 0.0419)
        + 5.737 * max(0.0, Q.LHA - 0.313)
        - 22.94 * max(0.0, Q.centroid_offset - 0.0525)
        - 10.62 * max(0.0, 0.0508 - Q.e2)
        + 349.1 * max(0.0, 0.00371 - Q.girth2)
        - 921.2 * max(0.0, 0.00868 - Q.girth2)
        + 22.61 * max(0.0, 0.00819 - Q.lam1)
        - 173.5 * max(0.0, Q.lam2 - 0.000765)
        + 2.25 * max(0.0, 6.33 - Q.log_sum_pt)
        - 2.566 * max(0.0, 6.72 - Q.log_sum_pt)
        + 10.34 * max(0.0, 0.132 - Q.mass_over_sum_pt)
        + 3.306 * max(0.0, Q.max_dr - 0.0866)
        - 4.083 * max(0.0, 0.0452 - Q.max_dr)
        - 1.676 * max(0.0, 0.0747 - Q.planar_flow)
        + 0.008903 * max(0.0, Q.sum_pt - 997.0)
        + 159.3 * max(0.0, 0.0135 - Q.width)
        + 0.5307 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0)
        - 25.69 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2)
        + 111.8 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        + 0.04472 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7)
        + 9.307 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7)
        + 50.07 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7)
        - 0.1003 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7)
        - 31.88 * max(0.0, 0.236 - Q.LHA)
        - 0.9965 * max(0.0, Q.centroid_offset - 0.0284)
        - 2.71 * max(0.0, 0.0204 - Q.centroid_offset)
        - 1.992 * max(0.0, 0.0384 - Q.e2)
        + 139.1 * max(0.0, 0.0501 - Q.e2)
        - 3.468 * max(0.0, Q.eccentricity - 0.955)
        + 254.2 * max(0.0, Q.girth2 - 0.00741)
        + 490.8 * max(0.0, Q.girth2 - 0.00874)
        + 116.8 * max(0.0, 0.000555 - Q.girth2)
        + 342.0 * max(0.0, 0.00845 - Q.lam1)
        - 0.03071 * max(0.0, Q.mass - 80.4)
        - 0.02138 * max(0.0, 27.6 - Q.mass)
        + 7.677 * max(0.0, Q.mass_over_sum_pt - 0.0849)
        - 84.48 * max(0.0, Q.mass_over_sum_pt - 0.0905)
        + 108.4 * max(0.0, 0.00524 - Q.width)
        - 184.9 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213)
        + 204.4 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065)
        - 0.005461 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0)
        + 15.87 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2)
        + 27.32 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21)
        - 212.5 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942)
        - 35.1 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13)
        - 73.19 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2)
        - 1.013 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50)
        - 0.003478 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934)
        + 0.5543 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2)
        + 5.721e-05 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0)
        + 0.009776 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow)
        - 8.095 * max(0.0, 0.0262 - Q.C2)
        + 32.65 * max(0.0, 0.234 - Q.LHA)
        + 15.34 * max(0.0, 0.00353 - Q.centroid_offset)
        + 0.01248 * max(0.0, 24.1 - Q.mass)
        + 4.56 * max(0.0, 0.188 - Q.max_dr)
        + 304.3 * max(0.0, 0.0055 - Q.width)
        - 22.61 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66)
        + 77.71 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow)
        - 70.77 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2)
        + 0.02359 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7)
        + 0.4834 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794)
        - 0.04686 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 5277.0 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2)
        - 7684.0 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627)
        - 48.68 * max(0.0, 0.0383 - Q.girth)
        + 73.46 * max(0.0, Q.girth2 - 0.0241)
        + 44.48 * max(0.0, Q.lam2 - 0.00101)
        - 0.4432 * max(0.0, Q.log_sum_pt - 6.37)
        + 0.04246 * max(0.0, Q.mass - 40.0)
        + 0.88 * max(0.0, 0.194 - Q.max_dr)
        - 0.0008653 * max(0.0, Q.sum_pt - 863.0)
        + 0.002506 * max(0.0, Q.sum_pt_top5 - 446.0)
        + 491.4 * max(0.0, 0.00617 - Q.width)
        + 4793.0 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset)
        - 337.4 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907)
        - 0.4396 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7)
        - 77.27 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21)
        - 13040.0 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012)
        - 3.587 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset)
        + 2.581 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset)
        - 37.0 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1)
        + 0.05029 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 0.02729 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow)
        - 2110.0 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308)
        - 8657.0 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326)
        - 5.447 * max(0.0, Q.C2 - 0.0492)
        + 1.131 * max(0.0, Q.LHA - 0.24)
        - 17.87 * max(0.0, 0.0634 - Q.girth)
        - 6.724 * max(0.0, Q.girth2 - 0.00809)
        - 26.04 * max(0.0, 0.00157 - Q.girth2)
        + 69.79 * max(0.0, 0.00516 - Q.lam1)
        + 168.5 * max(0.0, 0.00362 - Q.lam2)
        - 0.3438 * max(0.0, 6.28 - Q.log_sum_pt)
        - 0.02637 * max(0.0, Q.mass - 39.2)
        - 0.801 * max(0.0, Q.max_dr - 0.113)
        - 0.004814 * max(0.0, Q.sum_pt - 792.0)
        + 6.71 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21)
        - 275.3 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21)
        - 4.482 * max(0.0, Q.centroid_offset - 0.0143)
        + 3.418 * max(0.0, 0.0501 - Q.centroid_offset)
        + 1.055 * max(0.0, Q.girth - 0.0709)
        - 1.942 * max(0.0, 0.0204 - Q.girth)
        - 0.5184 * max(0.0, 0.124 - Q.girth)
        + 0.01008 * max(0.0, Q.mass - 91.2)
        + 15.23 * max(0.0, Q.mass_over_sum_pt - 0.0933)
        + 0.05001 * max(0.0, 0.289 - Q.planar_flow)
        + 0.0006877 * max(0.0, 674.0 - Q.sum_pt_top5)
        - 340.0 * max(0.0, 0.0034 - Q.width)
        - 174.6 * max(0.0, 0.00909 - Q.width)
        + 6.222 * max(0.0, 0.0424 - Q.z_7)
        + 73.7 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7)
        - 29.81 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt)
        - 21.53 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21)
        + 0.1495 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7)
        - 521.8 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161)
        - 0.007563 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass)
        + 0.5987 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121)
        + 50.01 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586)
        + 56.81 * max(0.0, Q.e2 - 0.0629)
        - 36.91 * max(0.0, Q.girth2 - 0.0193)
        - 3866.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06)
        - 0.8384 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4)
        + 35.46 * max(0.0, Q.C2 - 0.0657)
        + 20.33 * max(0.0, 0.0387 - Q.centroid_offset)
        - 21.41 * max(0.0, 0.0489 - Q.e2)
        - 0.9176 * max(0.0, 0.143 - Q.girth)
        + 42.18 * max(0.0, 0.00647 - Q.lam1)
        - 31.92 * max(0.0, 0.0157 - Q.lam1)
        - 886.1 * max(0.0, 0.00741 - Q.width)
        - 2.38 * max(0.0, 0.0286 - Q.z_7)
        + 8.077 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt)
        - 0.04979 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
        - 299.5 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset)
        + 11680.0 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset)
        - 0.0002258 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2)
        - 0.0003784 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08)
        + 1.434e-06 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7)
        - 0.003772 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023)
        + 1.694 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292)
        + 2.261 * max(0.0, Q.C2 - 0.0359)
        - 39.33 * max(0.0, Q.C2 - 0.0669)
        - 0.3591 * max(0.0, 0.0356 - Q.C2)
        - 2.38 * max(0.0, Q.centroid_offset - 0.0296)
        + 212.8 * max(0.0, Q.centroid_offset - 0.0496)
        - 8.894 * max(0.0, 0.0437 - Q.e2)
        - 5.852 * max(0.0, 0.0332 - Q.girth)
        + 2.634 * max(0.0, 0.0879 - Q.girth)
        + 430.5 * max(0.0, 0.00457 - Q.girth2)
        + 264.5 * max(0.0, 0.0137 - Q.girth2)
        - 39.17 * max(0.0, Q.lam1 - 0.00419)
        - 235.6 * max(0.0, Q.lam1 - 0.00616)
        + 0.002225 * max(0.0, 457.0 - Q.sum_pt_top5)
        + 1.089 * max(0.0, 0.136 - Q.tau21)
        + 486.2 * max(0.0, 0.00735 - Q.width)
        - 16.6 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset)
        - 60.59 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2)
        - 153.8 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968)
        + 12.3 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419)
        + 20.09 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65)
        + 21.23 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395)
        - 39.08 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378)
        - 1950.0 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr)
        - 0.001987 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2)
        - 53.25 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913)
        + 38.52 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184)
        - 0.2227 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        - 0.002909 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt)
        + 360.6 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 396.3 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow)
        - 1.016 * max(0.0, Q.LHA - 0.343)
        - 7.308 * max(0.0, 0.305 - Q.LHA)
        + 20.9 * max(0.0, Q.e2 - 0.0244)
        - 104.9 * max(0.0, Q.e2 - 0.0497)
        - 50.29 * max(0.0, 0.0627 - Q.e2)
        + 18.38 * max(0.0, Q.girth - 0.0329)
        - 347.2 * max(0.0, 0.00855 - Q.lam1)
        - 160.7 * max(0.0, 0.000335 - Q.lam2)
        - 12.88 * max(0.0, 0.00323 - Q.lam2)
        - 5.681 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.03125 * max(0.0, 80.4 - Q.mass)
        + 1.157 * max(0.0, 0.25 - Q.tau21)
        - 438.7 * max(0.0, 0.0136 - Q.width)
        + 14330.0 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244)
        - 412.9 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 114.6 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow)
    )


def score_W(Q):
    return (-20.75
        - 0.1156 * max(0.0, Q.D2 - 3.7)
        + 185.4 * max(0.0, 0.0252 - Q.e2)
        + 118.4 * max(0.0, Q.eccentricity - 0.997)
        - 112.0 * max(0.0, 0.0768 - Q.girth)
        + 367.7 * max(0.0, 0.0128 - Q.girth2)
        + 311.6 * max(0.0, 0.000706 - Q.lam1)
        + 706.2 * max(0.0, 0.00142 - Q.lam1)
        - 0.04578 * max(0.0, 29.3 - Q.mass)
        - 0.01344 * max(0.0, 59.8 - Q.mass)
        - 0.001609 * max(0.0, Q.sum_pt - 816.0)
        - 0.003957 * max(0.0, Q.sum_pt - 890.0)
        + 0.003113 * max(0.0, Q.sum_pt_top5 - 699.0)
        - 1750.0 * max(0.0, 0.00459 - Q.width)
        - 1118.0 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2)
        + 0.5452 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184)
        + 0.06371 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133)
        - 0.0004475 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7)
        + 95.01 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset)
        + 0.0001076 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9)
        + 4.921e-05 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7)
        + 19.57 * max(0.0, Q.LHA - 0.282)
        - 113.3 * max(0.0, Q.e2 - 0.0316)
        + 19.55 * max(0.0, 0.00749 - Q.e2)
        + 0.6164 * max(0.0, Q.log_sum_pt - 6.4)
        + 2.97 * max(0.0, 0.252 - Q.max_dr)
        - 0.005519 * max(0.0, Q.pt_7 - 33.5)
        - 8.176 * max(0.0, 0.0515 - Q.z_7)
        + 126.5 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21)
        + 258.8 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 19390.0 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201)
        - 658.8 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272)
        + 1.804 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2)
        - 54.96 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset)
        - 150.8 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2)
        - 2.943 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr)
        - 0.7728 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188)
        + 13260.0 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2)
        + 688.2 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow)
        + 36.4 * Q.LHA
        + 2389.0 * max(0.0, 4.88e-05 - Q.girth2)
        - 474.5 * max(0.0, 0.00591 - Q.lam1)
        + 8.54 * max(0.0, Q.log_sum_pt - 6.93)
        + 0.8186 * max(0.0, 6.43 - Q.log_sum_pt)
        + 0.006813 * max(0.0, 36.7 - Q.mass)
        - 0.2317 * max(0.0, 0.748 - Q.planar_flow)
        - 0.002616 * max(0.0, Q.pt_7 - 31.0)
        + 0.3993 * max(0.0, 54.0 - Q.pt_7)
        - 0.003451 * max(0.0, 794.0 - Q.sum_pt)
        + 0.002499 * max(0.0, 714.0 - Q.sum_pt_top5)
        - 18630.0 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset)
        + 2071.0 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819)
        - 28.42 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2)
        - 9.142e-05 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4)
        - 0.124 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7)
        - 0.1688 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2)
        + 0.2799 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132)
        + 0.03441 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111)
        - 2435.0 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width)
        + 14.7 * max(0.0, Q.C2 - 0.0894)
        - 11.1 * max(0.0, Q.LHA - 0.326)
        - 29.45 * max(0.0, Q.centroid_offset - 0.0106)
        + 2015.0 * max(0.0, Q.centroid_offset - 0.0495)
        - 85.93 * max(0.0, Q.e2 - 0.0263)
        + 39.31 * max(0.0, 0.0397 - Q.e2)
        + 217.5 * max(0.0, 0.121 - Q.girth)
        - 854.8 * max(0.0, 0.00863 - Q.girth2)
        + 0.06657 * max(0.0, Q.mass - 75.6)
        - 9.042 * max(0.0, Q.max_dr - 0.122)
        - 787.7 * max(0.0, Q.width - 0.0185)
        + 370.0 * max(0.0, 0.00539 - Q.width)
        + 266.9 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957)
        + 1630.0 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr)
        + 46.09 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227)
        + 0.116 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5)
        - 0.01821 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68)
        + 4028.0 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957)
        + 4.466 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7)
        - 1419.0 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7)
        - 0.01441 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705)
        - 2.316 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr)
        - 2534.0 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr)
        + 105.8 * max(0.0, Q.C2 - 0.0604)
        - 108.5 * max(0.0, 0.0602 - Q.C2)
        - 4.929 * max(0.0, 0.0429 - Q.centroid_offset)
        - 15.39 * max(0.0, Q.e2 - 0.0328)
        - 55.45 * max(0.0, Q.girth - 0.0651)
        + 0.03459 * max(0.0, 47.9 - Q.mass)
        - 1286.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 8.915 * max(0.0, Q.mass_over_sum_pt - 0.109)
        - 0.0001076 * max(0.0, 762.0 - Q.sum_pt)
        + 0.003799 * max(0.0, 439.0 - Q.sum_pt_top5)
        - 3.106 * max(0.0, 0.238 - Q.tau21)
        - 412.7 * max(0.0, Q.width - 0.000872)
        - 0.7981 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6)
        - 1.001 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7)
        - 192.1 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982)
        + 0.01616 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass)
        + 1.022 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357)
        + 0.05197 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7)
        - 154.1 * max(0.0, 0.0405 - Q.e2)
        - 1792.0 * max(0.0, 5.65e-05 - Q.girth2)
        - 0.4058 * max(0.0, 54.4 - Q.pt_7)
        + 6.632e-05 * max(0.0, Q.sum_pt - 935.0)
        - 0.004182 * max(0.0, 721.0 - Q.sum_pt_top5)
        + 28.22 * max(0.0, 0.0243 - Q.z_7)
        + 171.6 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367)
        - 0.5105 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt)
        - 21.67 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241)
        - 150.2 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151)
        + 0.01826 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset)
        + 24150.0 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset)
        + 320.9 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset)
        - 0.01804 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 3.055 * max(0.0, Q.C2 - 0.0419)
        - 88.08 * max(0.0, Q.LHA - 0.313)
        + 116.2 * max(0.0, Q.centroid_offset - 0.0525)
        + 63.02 * max(0.0, 0.0508 - Q.e2)
        - 814.7 * max(0.0, 0.00371 - Q.girth2)
        + 1746.0 * max(0.0, 0.00868 - Q.girth2)
        - 461.4 * max(0.0, 0.00819 - Q.lam1)
        + 494.5 * max(0.0, Q.lam2 - 0.000765)
        - 0.4961 * max(0.0, 6.33 - Q.log_sum_pt)
        - 1.719 * max(0.0, 6.72 - Q.log_sum_pt)
        + 22.96 * max(0.0, 0.132 - Q.mass_over_sum_pt)
        - 0.4042 * max(0.0, Q.max_dr - 0.0866)
        - 7.254 * max(0.0, 0.0452 - Q.max_dr)
        + 0.407 * max(0.0, 0.0747 - Q.planar_flow)
        - 0.01478 * max(0.0, Q.sum_pt - 997.0)
        + 648.2 * max(0.0, 0.0135 - Q.width)
        + 0.113 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0)
        - 92.51 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2)
        - 0.7349 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        - 0.08752 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7)
        - 207.8 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7)
        - 123.2 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7)
        - 0.08462 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7)
        - 37.93 * max(0.0, 0.236 - Q.LHA)
        - 51.13 * max(0.0, Q.centroid_offset - 0.0284)
        + 72.06 * max(0.0, 0.0204 - Q.centroid_offset)
        + 26.48 * max(0.0, 0.0384 - Q.e2)
        - 1055.0 * max(0.0, 0.0501 - Q.e2)
        - 21.35 * max(0.0, Q.eccentricity - 0.955)
        + 275.1 * max(0.0, Q.girth2 - 0.00741)
        + 224.0 * max(0.0, Q.girth2 - 0.00874)
        - 1612.0 * max(0.0, 0.000555 - Q.girth2)
        - 517.1 * max(0.0, 0.00845 - Q.lam1)
        + 0.05465 * max(0.0, Q.mass - 80.4)
        + 0.0573 * max(0.0, 27.6 - Q.mass)
        - 15.53 * max(0.0, Q.mass_over_sum_pt - 0.0849)
        + 1383.0 * max(0.0, Q.mass_over_sum_pt - 0.0905)
        - 921.9 * max(0.0, 0.00524 - Q.width)
        + 895.7 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213)
        - 1121.0 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065)
        + 0.02587 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0)
        - 23.6 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2)
        - 10.35 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21)
        + 6543.0 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942)
        + 201.5 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13)
        - 176.0 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2)
        - 4.494 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50)
        - 0.4718 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934)
        + 6.221 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2)
        + 0.005081 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0)
        + 0.005822 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow)
        - 6.63 * max(0.0, 0.0262 - Q.C2)
        + 25.78 * max(0.0, 0.234 - Q.LHA)
        - 103.7 * max(0.0, 0.00353 - Q.centroid_offset)
        + 0.0454 * max(0.0, 24.1 - Q.mass)
        + 3.947 * max(0.0, 0.188 - Q.max_dr)
        - 1055.0 * max(0.0, 0.0055 - Q.width)
        - 49.12 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66)
        - 141.8 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow)
        + 2.563 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2)
        - 0.003619 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7)
        - 2.158 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794)
        + 0.01404 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 613.7 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2)
        + 26710.0 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627)
        + 5.874 * max(0.0, 0.0383 - Q.girth)
        - 12.31 * max(0.0, Q.girth2 - 0.0241)
        - 402.0 * max(0.0, Q.lam2 - 0.00101)
        - 0.8303 * max(0.0, Q.log_sum_pt - 6.37)
        - 0.08678 * max(0.0, Q.mass - 40.0)
        - 2.96 * max(0.0, 0.194 - Q.max_dr)
        + 0.001675 * max(0.0, Q.sum_pt - 863.0)
        - 0.001381 * max(0.0, Q.sum_pt_top5 - 446.0)
        - 1277.0 * max(0.0, 0.00617 - Q.width)
        - 813.6 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset)
        + 307.7 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907)
        - 0.2989 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7)
        + 89.11 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21)
        + 22790.0 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012)
        + 6.378 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset)
        - 5.528 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset)
        + 11.27 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1)
        + 0.01746 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 0.07379 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow)
        + 352.8 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308)
        + 6611.0 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326)
        - 10.87 * max(0.0, Q.C2 - 0.0492)
        + 18.39 * max(0.0, Q.LHA - 0.24)
        + 22.02 * max(0.0, 0.0634 - Q.girth)
        + 307.0 * max(0.0, Q.girth2 - 0.00809)
        - 211.1 * max(0.0, 0.00157 - Q.girth2)
        + 284.8 * max(0.0, 0.00516 - Q.lam1)
        - 42.08 * max(0.0, 0.00362 - Q.lam2)
        - 0.4304 * max(0.0, 6.28 - Q.log_sum_pt)
        + 0.03483 * max(0.0, Q.mass - 39.2)
        + 2.117 * max(0.0, Q.max_dr - 0.113)
        + 0.006428 * max(0.0, Q.sum_pt - 792.0)
        + 2.086 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21)
        + 23.45 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21)
        - 26.11 * max(0.0, Q.centroid_offset - 0.0143)
        + 36.26 * max(0.0, 0.0501 - Q.centroid_offset)
        - 36.11 * max(0.0, Q.girth - 0.0709)
        + 8.93 * max(0.0, 0.0204 - Q.girth)
        - 184.9 * max(0.0, 0.124 - Q.girth)
        - 0.02677 * max(0.0, Q.mass - 91.2)
        - 38.4 * max(0.0, Q.mass_over_sum_pt - 0.0933)
        + 1.034 * max(0.0, 0.289 - Q.planar_flow)
        + 0.003519 * max(0.0, 674.0 - Q.sum_pt_top5)
        + 251.3 * max(0.0, 0.0034 - Q.width)
        - 96.29 * max(0.0, 0.00909 - Q.width)
        - 18.76 * max(0.0, 0.0424 - Q.z_7)
        + 406.5 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7)
        - 38.66 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt)
        + 247.9 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21)
        - 1.138 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7)
        - 582.4 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161)
        + 0.03463 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass)
        - 3.363 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121)
        - 2646.0 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586)
        - 976.4 * max(0.0, Q.e2 - 0.0629)
        + 240.7 * max(0.0, Q.girth2 - 0.0193)
        - 3021.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06)
        - 0.7778 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4)
        - 2.106 * max(0.0, Q.C2 - 0.0657)
        - 95.1 * max(0.0, 0.0387 - Q.centroid_offset)
        - 17.32 * max(0.0, 0.0489 - Q.e2)
        + 20.16 * max(0.0, 0.143 - Q.girth)
        - 337.2 * max(0.0, 0.00647 - Q.lam1)
        + 320.7 * max(0.0, 0.0157 - Q.lam1)
        + 5089.0 * max(0.0, 0.00741 - Q.width)
        - 25.64 * max(0.0, 0.0286 - Q.z_7)
        + 0.2561 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt)
        + 0.223 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
        + 4777.0 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset)
        + 11260.0 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset)
        + 5.834e-05 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2)
        + 0.0005369 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08)
        + 3.106e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7)
        - 0.1054 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023)
        - 3.811 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292)
        - 119.3 * max(0.0, Q.C2 - 0.0359)
        + 69.98 * max(0.0, Q.C2 - 0.0669)
        + 106.8 * max(0.0, 0.0356 - Q.C2)
        + 58.55 * max(0.0, Q.centroid_offset - 0.0296)
        - 2041.0 * max(0.0, Q.centroid_offset - 0.0496)
        - 79.94 * max(0.0, 0.0437 - Q.e2)
        + 1.796 * max(0.0, 0.0332 - Q.girth)
        + 109.8 * max(0.0, 0.0879 - Q.girth)
        + 1239.0 * max(0.0, 0.00457 - Q.girth2)
        - 3080.0 * max(0.0, 0.0137 - Q.girth2)
        - 727.5 * max(0.0, Q.lam1 - 0.00419)
        + 955.8 * max(0.0, Q.lam1 - 0.00616)
        - 0.0004387 * max(0.0, 457.0 - Q.sum_pt_top5)
        - 7.002 * max(0.0, 0.136 - Q.tau21)
        - 3735.0 * max(0.0, 0.00735 - Q.width)
        - 39.1 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset)
        + 15.86 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2)
        - 337.3 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968)
        - 401.9 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419)
        + 126.3 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65)
        + 323.5 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395)
        + 3.278 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378)
        - 10570.0 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr)
        + 0.02856 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2)
        - 291.3 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913)
        + 347.2 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184)
        - 39.77 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 0.01067 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt)
        + 613.5 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 527.6 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow)
        + 51.49 * max(0.0, Q.LHA - 0.343)
        + 32.31 * max(0.0, 0.305 - Q.LHA)
        + 189.1 * max(0.0, Q.e2 - 0.0244)
        + 949.6 * max(0.0, Q.e2 - 0.0497)
        + 881.6 * max(0.0, 0.0627 - Q.e2)
        - 11.32 * max(0.0, Q.girth - 0.0329)
        + 767.4 * max(0.0, 0.00855 - Q.lam1)
        + 1001.0 * max(0.0, 0.000335 - Q.lam2)
        + 441.5 * max(0.0, 0.00323 - Q.lam2)
        - 6.564 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.03133 * max(0.0, 80.4 - Q.mass)
        - 1.274 * max(0.0, 0.25 - Q.tau21)
        + 2039.0 * max(0.0, 0.0136 - Q.width)
        - 103600.0 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244)
        + 1822.0 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 719.2 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow)
    )


def score_Z(Q):
    return (-25.84
        - 0.00444 * max(0.0, Q.D2 - 3.7)
        - 111.9 * max(0.0, 0.0252 - Q.e2)
        + 16.71 * max(0.0, Q.eccentricity - 0.997)
        + 15.35 * max(0.0, 0.0768 - Q.girth)
        + 1517.0 * max(0.0, 0.0128 - Q.girth2)
        + 913.1 * max(0.0, 0.000706 - Q.lam1)
        - 202.2 * max(0.0, 0.00142 - Q.lam1)
        + 0.03431 * max(0.0, 29.3 - Q.mass)
        + 0.007364 * max(0.0, 59.8 - Q.mass)
        - 0.01011 * max(0.0, Q.sum_pt - 816.0)
        + 0.00256 * max(0.0, Q.sum_pt - 890.0)
        - 0.0004517 * max(0.0, Q.sum_pt_top5 - 699.0)
        + 154.6 * max(0.0, 0.00459 - Q.width)
        + 1029.0 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2)
        - 0.309 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184)
        + 0.7353 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133)
        - 0.0005129 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7)
        + 0.517 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset)
        + 1.667e-05 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9)
        - 0.000124 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7)
        + 11.2 * max(0.0, Q.LHA - 0.282)
        - 38.36 * max(0.0, Q.e2 - 0.0316)
        + 4.918 * max(0.0, 0.00749 - Q.e2)
        + 1.633 * max(0.0, Q.log_sum_pt - 6.4)
        - 2.515 * max(0.0, 0.252 - Q.max_dr)
        - 0.002022 * max(0.0, Q.pt_7 - 33.5)
        - 12.69 * max(0.0, 0.0515 - Q.z_7)
        - 3.119 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21)
        + 449.5 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 653.9 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201)
        - 923.4 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272)
        + 1.057 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2)
        - 60.72 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset)
        + 2713.0 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2)
        - 5.071 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr)
        - 0.315 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188)
        + 2560.0 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2)
        + 450.9 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow)
        - 15.04 * Q.LHA
        + 4870.0 * max(0.0, 4.88e-05 - Q.girth2)
        + 550.0 * max(0.0, 0.00591 - Q.lam1)
        - 0.7887 * max(0.0, Q.log_sum_pt - 6.93)
        + 0.194 * max(0.0, 6.43 - Q.log_sum_pt)
        + 0.03482 * max(0.0, 36.7 - Q.mass)
        + 0.3103 * max(0.0, 0.748 - Q.planar_flow)
        + 0.00963 * max(0.0, Q.pt_7 - 31.0)
        + 0.372 * max(0.0, 54.0 - Q.pt_7)
        - 0.00403 * max(0.0, 794.0 - Q.sum_pt)
        + 0.001374 * max(0.0, 714.0 - Q.sum_pt_top5)
        - 3357.0 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset)
        + 76.39 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819)
        - 8.465 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2)
        - 0.0001268 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4)
        + 0.1684 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7)
        - 0.2969 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2)
        + 0.3418 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132)
        + 0.002154 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111)
        + 8331.0 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width)
        + 27.02 * max(0.0, Q.C2 - 0.0894)
        - 16.16 * max(0.0, Q.LHA - 0.326)
        - 28.52 * max(0.0, Q.centroid_offset - 0.0106)
        + 2089.0 * max(0.0, Q.centroid_offset - 0.0495)
        - 37.67 * max(0.0, Q.e2 - 0.0263)
        - 39.17 * max(0.0, 0.0397 - Q.e2)
        + 267.7 * max(0.0, 0.121 - Q.girth)
        - 455.4 * max(0.0, 0.00863 - Q.girth2)
        + 0.04704 * max(0.0, Q.mass - 75.6)
        - 7.763 * max(0.0, Q.max_dr - 0.122)
        - 679.1 * max(0.0, Q.width - 0.0185)
        + 94.86 * max(0.0, 0.00539 - Q.width)
        - 357.6 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957)
        + 1014.0 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr)
        + 7.507 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227)
        + 0.7193 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5)
        + 2.885 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68)
        + 7602.0 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957)
        - 5.276 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7)
        + 4928.0 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7)
        - 0.05187 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705)
        - 1.149 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr)
        - 4421.0 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr)
        + 32.15 * max(0.0, Q.C2 - 0.0604)
        - 26.83 * max(0.0, 0.0602 - Q.C2)
        - 15.43 * max(0.0, 0.0429 - Q.centroid_offset)
        - 50.89 * max(0.0, Q.e2 - 0.0328)
        + 4.443 * max(0.0, Q.girth - 0.0651)
        + 0.02879 * max(0.0, 47.9 - Q.mass)
        - 103.1 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        - 2.459 * max(0.0, Q.mass_over_sum_pt - 0.109)
        - 0.00562 * max(0.0, 762.0 - Q.sum_pt)
        + 0.006806 * max(0.0, 439.0 - Q.sum_pt_top5)
        + 3.805 * max(0.0, 0.238 - Q.tau21)
        - 530.0 * max(0.0, Q.width - 0.000872)
        - 0.2762 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6)
        + 0.5644 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7)
        - 46.77 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982)
        - 0.01387 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass)
        + 2.393 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357)
        - 0.01659 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7)
        + 27.56 * max(0.0, 0.0405 - Q.e2)
        - 5784.0 * max(0.0, 5.65e-05 - Q.girth2)
        - 0.376 * max(0.0, 54.4 - Q.pt_7)
        - 0.002607 * max(0.0, Q.sum_pt - 935.0)
        - 0.002086 * max(0.0, 721.0 - Q.sum_pt_top5)
        + 14.07 * max(0.0, 0.0243 - Q.z_7)
        + 352.2 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367)
        + 6.018 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt)
        - 55.31 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241)
        - 109.7 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151)
        + 0.1048 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset)
        - 7122.0 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset)
        + 146.1 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset)
        - 0.02873 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 8.664 * max(0.0, Q.C2 - 0.0419)
        - 21.4 * max(0.0, Q.LHA - 0.313)
        + 222.0 * max(0.0, Q.centroid_offset - 0.0525)
        + 295.5 * max(0.0, 0.0508 - Q.e2)
        - 347.5 * max(0.0, 0.00371 - Q.girth2)
        + 3407.0 * max(0.0, 0.00868 - Q.girth2)
        - 801.1 * max(0.0, 0.00819 - Q.lam1)
        + 395.6 * max(0.0, Q.lam2 - 0.000765)
        - 0.3497 * max(0.0, 6.33 - Q.log_sum_pt)
        + 7.694 * max(0.0, 6.72 - Q.log_sum_pt)
        + 13.73 * max(0.0, 0.132 - Q.mass_over_sum_pt)
        - 10.4 * max(0.0, Q.max_dr - 0.0866)
        + 9.172 * max(0.0, 0.0452 - Q.max_dr)
        - 1.895 * max(0.0, 0.0747 - Q.planar_flow)
        + 0.0007764 * max(0.0, Q.sum_pt - 997.0)
        - 1756.0 * max(0.0, 0.0135 - Q.width)
        - 0.2759 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0)
        + 144.6 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2)
        - 54.6 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        - 0.09366 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7)
        - 203.4 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7)
        - 186.3 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7)
        - 0.08889 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7)
        - 12.37 * max(0.0, 0.236 - Q.LHA)
        - 62.78 * max(0.0, Q.centroid_offset - 0.0284)
        - 14.62 * max(0.0, 0.0204 - Q.centroid_offset)
        + 34.85 * max(0.0, 0.0384 - Q.e2)
        - 1630.0 * max(0.0, 0.0501 - Q.e2)
        - 20.31 * max(0.0, Q.eccentricity - 0.955)
        + 3058.0 * max(0.0, Q.girth2 - 0.00741)
        - 1910.0 * max(0.0, Q.girth2 - 0.00874)
        - 1069.0 * max(0.0, 0.000555 - Q.girth2)
        - 450.0 * max(0.0, 0.00845 - Q.lam1)
        + 0.06628 * max(0.0, Q.mass - 80.4)
        + 0.001748 * max(0.0, 27.6 - Q.mass)
        + 135.0 * max(0.0, Q.mass_over_sum_pt - 0.0849)
        + 37.53 * max(0.0, Q.mass_over_sum_pt - 0.0905)
        - 372.3 * max(0.0, 0.00524 - Q.width)
        + 258.0 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213)
        - 874.8 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065)
        - 0.03618 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0)
        + 79.71 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2)
        + 10.18 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21)
        + 6256.0 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942)
        - 189.6 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13)
        - 453.0 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2)
        - 14.72 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50)
        - 1.258 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934)
        + 4.677 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2)
        + 0.01008 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0)
        - 0.01614 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow)
        + 27.52 * max(0.0, 0.0262 - Q.C2)
        - 2.567 * max(0.0, 0.234 - Q.LHA)
        - 4.859 * max(0.0, 0.00353 - Q.centroid_offset)
        - 0.001947 * max(0.0, 24.1 - Q.mass)
        - 2.421 * max(0.0, 0.188 - Q.max_dr)
        - 134.0 * max(0.0, 0.0055 - Q.width)
        - 51.1 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66)
        + 59.07 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow)
        + 168.4 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2)
        - 0.05561 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7)
        - 0.1785 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794)
        - 0.0002393 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 8007.0 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2)
        + 787.5 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627)
        + 12.68 * max(0.0, 0.0383 - Q.girth)
        - 125.7 * max(0.0, Q.girth2 - 0.0241)
        - 593.0 * max(0.0, Q.lam2 - 0.00101)
        - 0.7275 * max(0.0, Q.log_sum_pt - 6.37)
        + 0.01188 * max(0.0, Q.mass - 40.0)
        + 2.631 * max(0.0, 0.194 - Q.max_dr)
        - 0.001043 * max(0.0, Q.sum_pt - 863.0)
        - 0.000463 * max(0.0, Q.sum_pt_top5 - 446.0)
        - 522.5 * max(0.0, 0.00617 - Q.width)
        + 313.7 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset)
        - 191.0 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907)
        - 0.6459 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7)
        + 50.48 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21)
        + 11250.0 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012)
        + 3.497 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset)
        - 1.297 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset)
        - 11.11 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1)
        - 0.06931 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 0.02614 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow)
        - 2059.0 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308)
        + 1900.0 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326)
        - 0.2129 * max(0.0, Q.C2 - 0.0492)
        + 8.069 * max(0.0, Q.LHA - 0.24)
        - 35.49 * max(0.0, 0.0634 - Q.girth)
        + 188.3 * max(0.0, Q.girth2 - 0.00809)
        + 168.4 * max(0.0, 0.00157 - Q.girth2)
        - 57.58 * max(0.0, 0.00516 - Q.lam1)
        + 5.843 * max(0.0, 0.00362 - Q.lam2)
        + 0.7686 * max(0.0, 6.28 - Q.log_sum_pt)
        - 0.04599 * max(0.0, Q.mass - 39.2)
        + 11.21 * max(0.0, Q.max_dr - 0.113)
        + 0.009233 * max(0.0, Q.sum_pt - 792.0)
        - 22.23 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21)
        + 46.89 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21)
        - 0.189 * max(0.0, Q.centroid_offset - 0.0143)
        + 10.57 * max(0.0, 0.0501 - Q.centroid_offset)
        + 1.485 * max(0.0, Q.girth - 0.0709)
        + 29.1 * max(0.0, 0.0204 - Q.girth)
        - 202.5 * max(0.0, 0.124 - Q.girth)
        - 0.009056 * max(0.0, Q.mass - 91.2)
        - 163.9 * max(0.0, Q.mass_over_sum_pt - 0.0933)
        - 2.711 * max(0.0, 0.289 - Q.planar_flow)
        + 0.0005287 * max(0.0, 674.0 - Q.sum_pt_top5)
        + 416.1 * max(0.0, 0.0034 - Q.width)
        + 1196.0 * max(0.0, 0.00909 - Q.width)
        - 3.782 * max(0.0, 0.0424 - Q.z_7)
        + 80.15 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7)
        + 26.84 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt)
        - 36.7 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21)
        + 0.2512 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7)
        - 2374.0 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161)
        + 0.07351 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass)
        + 6.189 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121)
        - 504.1 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586)
        - 1565.0 * max(0.0, Q.e2 - 0.0629)
        + 261.1 * max(0.0, Q.girth2 - 0.0193)
        - 1236.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06)
        + 4.717 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4)
        - 65.98 * max(0.0, Q.C2 - 0.0657)
        + 1.485 * max(0.0, 0.0387 - Q.centroid_offset)
        - 207.5 * max(0.0, 0.0489 - Q.e2)
        + 31.08 * max(0.0, 0.143 - Q.girth)
        - 46.7 * max(0.0, 0.00647 - Q.lam1)
        + 451.2 * max(0.0, 0.0157 - Q.lam1)
        - 649.7 * max(0.0, 0.00741 - Q.width)
        - 38.22 * max(0.0, 0.0286 - Q.z_7)
        - 5.536 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt)
        + 0.2419 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
        - 1032.0 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset)
        + 220.4 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset)
        + 0.0004792 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2)
        + 0.0001374 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08)
        + 4.924e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7)
        - 0.06098 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023)
        - 0.3736 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292)
        - 1.361 * max(0.0, Q.C2 - 0.0359)
        + 46.82 * max(0.0, Q.C2 - 0.0669)
        + 1.84 * max(0.0, 0.0356 - Q.C2)
        + 14.8 * max(0.0, Q.centroid_offset - 0.0296)
        - 2231.0 * max(0.0, Q.centroid_offset - 0.0496)
        + 130.9 * max(0.0, 0.0437 - Q.e2)
        - 21.4 * max(0.0, 0.0332 - Q.girth)
        - 34.92 * max(0.0, 0.0879 - Q.girth)
        - 261.6 * max(0.0, 0.00457 - Q.girth2)
        - 3393.0 * max(0.0, 0.0137 - Q.girth2)
        + 318.8 * max(0.0, Q.lam1 - 0.00419)
        - 615.1 * max(0.0, Q.lam1 - 0.00616)
        - 0.005553 * max(0.0, 457.0 - Q.sum_pt_top5)
        + 0.249 * max(0.0, 0.136 - Q.tau21)
        - 3514.0 * max(0.0, 0.00735 - Q.width)
        + 1.432 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset)
        - 40.27 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2)
        + 4954.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968)
        + 170.0 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419)
        - 115.4 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65)
        - 401.3 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395)
        + 225.7 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378)
        + 9130.0 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr)
        - 0.03822 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2)
        + 429.9 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913)
        - 577.2 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184)
        - 24.68 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        - 0.01259 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt)
        - 355.1 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 2329.0 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow)
        + 29.2 * max(0.0, Q.LHA - 0.343)
        - 8.939 * max(0.0, 0.305 - Q.LHA)
        + 272.2 * max(0.0, Q.e2 - 0.0244)
        + 1365.0 * max(0.0, Q.e2 - 0.0497)
        + 1494.0 * max(0.0, 0.0627 - Q.e2)
        + 40.65 * max(0.0, Q.girth - 0.0329)
        - 12.94 * max(0.0, 0.00855 - Q.lam1)
        + 107.5 * max(0.0, 0.000335 - Q.lam2)
        - 361.5 * max(0.0, 0.00323 - Q.lam2)
        - 10.23 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.1166 * max(0.0, 80.4 - Q.mass)
        - 3.713 * max(0.0, 0.25 - Q.tau21)
        + 4579.0 * max(0.0, 0.0136 - Q.width)
        - 40570.0 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244)
        + 1360.0 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 439.7 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow)
    )


def score_t(Q):
    return (14.3
        + 0.04527 * max(0.0, Q.D2 - 3.7)
        - 2.678 * max(0.0, 0.0252 - Q.e2)
        + 12.53 * max(0.0, Q.eccentricity - 0.997)
        + 5.528 * max(0.0, 0.0768 - Q.girth)
        - 100.5 * max(0.0, 0.0128 - Q.girth2)
        + 490.9 * max(0.0, 0.000706 - Q.lam1)
        + 131.2 * max(0.0, 0.00142 - Q.lam1)
        + 0.00459 * max(0.0, 29.3 - Q.mass)
        + 0.02407 * max(0.0, 59.8 - Q.mass)
        + 0.001521 * max(0.0, Q.sum_pt - 816.0)
        - 0.001026 * max(0.0, Q.sum_pt - 890.0)
        - 0.002378 * max(0.0, Q.sum_pt_top5 - 699.0)
        + 600.0 * max(0.0, 0.00459 - Q.width)
        - 265.3 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2)
        - 0.4617 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184)
        - 0.4549 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133)
        + 0.0001048 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7)
        + 33.07 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset)
        - 0.0001277 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9)
        + 0.000174 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7)
        - 3.395 * max(0.0, Q.LHA - 0.282)
        + 5.518 * max(0.0, Q.e2 - 0.0316)
        + 6.78 * max(0.0, 0.00749 - Q.e2)
        + 0.7722 * max(0.0, Q.log_sum_pt - 6.4)
        + 0.4548 * max(0.0, 0.252 - Q.max_dr)
        - 0.01375 * max(0.0, Q.pt_7 - 33.5)
        - 7.723 * max(0.0, 0.0515 - Q.z_7)
        + 49.7 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21)
        - 1024.0 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 2173.0 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201)
        + 557.4 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272)
        + 0.1878 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2)
        - 33.57 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset)
        + 629.1 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2)
        + 4.948 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr)
        + 0.3683 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188)
        - 6968.0 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2)
        + 435.9 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow)
        + 3.896 * Q.LHA
        + 11170.0 * max(0.0, 4.88e-05 - Q.girth2)
        + 36.72 * max(0.0, 0.00591 - Q.lam1)
        - 6.787 * max(0.0, Q.log_sum_pt - 6.93)
        - 1.587 * max(0.0, 6.43 - Q.log_sum_pt)
        + 0.004428 * max(0.0, 36.7 - Q.mass)
        - 0.009461 * max(0.0, 0.748 - Q.planar_flow)
        + 0.01986 * max(0.0, Q.pt_7 - 31.0)
        - 0.09036 * max(0.0, 54.0 - Q.pt_7)
        + 0.00229 * max(0.0, 794.0 - Q.sum_pt)
        + 0.008325 * max(0.0, 714.0 - Q.sum_pt_top5)
        + 12030.0 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset)
        + 267.0 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819)
        - 11.04 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2)
        + 1.491e-05 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4)
        - 0.002272 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7)
        - 0.3509 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2)
        - 0.3251 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132)
        - 0.008684 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111)
        - 259.4 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width)
        + 2.414 * max(0.0, Q.C2 - 0.0894)
        + 5.091 * max(0.0, Q.LHA - 0.326)
        - 17.84 * max(0.0, Q.centroid_offset - 0.0106)
        - 499.0 * max(0.0, Q.centroid_offset - 0.0495)
        - 12.36 * max(0.0, Q.e2 - 0.0263)
        + 15.51 * max(0.0, 0.0397 - Q.e2)
        - 35.52 * max(0.0, 0.121 - Q.girth)
        - 300.3 * max(0.0, 0.00863 - Q.girth2)
        + 0.001453 * max(0.0, Q.mass - 75.6)
        + 2.128 * max(0.0, Q.max_dr - 0.122)
        + 93.15 * max(0.0, Q.width - 0.0185)
        + 29.22 * max(0.0, 0.00539 - Q.width)
        + 7.742 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957)
        - 91.82 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr)
        - 3.783 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227)
        - 0.1045 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5)
        - 0.4031 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68)
        - 624.5 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957)
        - 0.5401 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7)
        - 249.7 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7)
        - 0.04096 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705)
        + 0.1096 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr)
        + 659.0 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr)
        - 44.49 * max(0.0, Q.C2 - 0.0604)
        + 35.48 * max(0.0, 0.0602 - Q.C2)
        - 5.284 * max(0.0, 0.0429 - Q.centroid_offset)
        + 3.54 * max(0.0, Q.e2 - 0.0328)
        - 1.427 * max(0.0, Q.girth - 0.0651)
        + 0.009778 * max(0.0, 47.9 - Q.mass)
        + 77.96 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 14.5 * max(0.0, Q.mass_over_sum_pt - 0.109)
        - 0.001734 * max(0.0, 762.0 - Q.sum_pt)
        + 0.002879 * max(0.0, 439.0 - Q.sum_pt_top5)
        + 3.463 * max(0.0, 0.238 - Q.tau21)
        - 299.7 * max(0.0, Q.width - 0.000872)
        + 0.2176 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6)
        - 0.2444 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7)
        - 71.04 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982)
        - 0.0601 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass)
        + 4.668 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357)
        - 0.1011 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7)
        - 6.965 * max(0.0, 0.0405 - Q.e2)
        - 23010.0 * max(0.0, 5.65e-05 - Q.girth2)
        + 0.07555 * max(0.0, 54.4 - Q.pt_7)
        + 0.009779 * max(0.0, Q.sum_pt - 935.0)
        - 0.007443 * max(0.0, 721.0 - Q.sum_pt_top5)
        - 51.93 * max(0.0, 0.0243 - Q.z_7)
        + 551.1 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367)
        + 21.82 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt)
        + 3.959 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241)
        + 201.5 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151)
        - 0.2088 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset)
        - 7828.0 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset)
        + 681.3 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset)
        + 0.00336 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        - 1.44 * max(0.0, Q.C2 - 0.0419)
        + 6.041 * max(0.0, Q.LHA - 0.313)
        - 38.96 * max(0.0, Q.centroid_offset - 0.0525)
        - 175.2 * max(0.0, 0.0508 - Q.e2)
        + 60.14 * max(0.0, 0.00371 - Q.girth2)
        + 130.1 * max(0.0, 0.00868 - Q.girth2)
        + 107.9 * max(0.0, 0.00819 - Q.lam1)
        - 129.6 * max(0.0, Q.lam2 - 0.000765)
        - 2.156 * max(0.0, 6.33 - Q.log_sum_pt)
        + 0.07038 * max(0.0, 6.72 - Q.log_sum_pt)
        - 6.006 * max(0.0, 0.132 - Q.mass_over_sum_pt)
        - 2.724 * max(0.0, Q.max_dr - 0.0866)
        + 3.038 * max(0.0, 0.0452 - Q.max_dr)
        - 2.943 * max(0.0, 0.0747 - Q.planar_flow)
        - 0.002 * max(0.0, Q.sum_pt - 997.0)
        - 60.61 * max(0.0, 0.0135 - Q.width)
        - 0.1198 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0)
        - 11.31 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2)
        + 117.8 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        + 0.01224 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7)
        + 23.7 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7)
        + 33.97 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7)
        + 0.00203 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7)
        + 4.124 * max(0.0, 0.236 - Q.LHA)
        - 0.03468 * max(0.0, Q.centroid_offset - 0.0284)
        - 1.014 * max(0.0, 0.0204 - Q.centroid_offset)
        - 4.194 * max(0.0, 0.0384 - Q.e2)
        + 728.1 * max(0.0, 0.0501 - Q.e2)
        - 9.158 * max(0.0, Q.eccentricity - 0.955)
        + 298.4 * max(0.0, Q.girth2 - 0.00741)
        + 118.6 * max(0.0, Q.girth2 - 0.00874)
        + 693.6 * max(0.0, 0.000555 - Q.girth2)
        + 100.5 * max(0.0, 0.00845 - Q.lam1)
        + 0.01002 * max(0.0, Q.mass - 80.4)
        - 0.02657 * max(0.0, 27.6 - Q.mass)
        - 13.82 * max(0.0, Q.mass_over_sum_pt - 0.0849)
        - 118.1 * max(0.0, Q.mass_over_sum_pt - 0.0905)
        + 20.24 * max(0.0, 0.00524 - Q.width)
        - 141.4 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213)
        + 450.5 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065)
        + 0.03971 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0)
        - 5.87 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2)
        - 38.66 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21)
        - 279.7 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942)
        - 120.0 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13)
        + 20.52 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2)
        + 1.824 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50)
        - 0.02723 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934)
        - 0.8777 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2)
        + 0.006812 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0)
        + 0.01111 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow)
        - 7.314 * max(0.0, 0.0262 - Q.C2)
        - 3.848 * max(0.0, 0.234 - Q.LHA)
        + 24.98 * max(0.0, 0.00353 - Q.centroid_offset)
        - 0.005137 * max(0.0, 24.1 - Q.mass)
        + 0.1083 * max(0.0, 0.188 - Q.max_dr)
        + 58.28 * max(0.0, 0.0055 - Q.width)
        + 31.64 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66)
        + 45.13 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow)
        - 60.69 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2)
        - 0.05977 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7)
        + 0.6725 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794)
        + 0.08021 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 10020.0 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2)
        - 2296.0 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627)
        - 0.3921 * max(0.0, 0.0383 - Q.girth)
        - 113.1 * max(0.0, Q.girth2 - 0.0241)
        + 271.4 * max(0.0, Q.lam2 - 0.00101)
        - 0.2825 * max(0.0, Q.log_sum_pt - 6.37)
        + 0.02494 * max(0.0, Q.mass - 40.0)
        - 1.018 * max(0.0, 0.194 - Q.max_dr)
        - 0.0001008 * max(0.0, Q.sum_pt - 863.0)
        + 0.0004369 * max(0.0, Q.sum_pt_top5 - 446.0)
        + 73.54 * max(0.0, 0.00617 - Q.width)
        + 1359.0 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset)
        - 248.0 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907)
        + 0.04214 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7)
        + 17.53 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21)
        + 4304.0 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012)
        - 0.1152 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset)
        + 0.05443 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset)
        - 15.95 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1)
        + 0.07598 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 0.008347 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow)
        + 1968.0 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308)
        - 4698.0 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326)
        + 15.94 * max(0.0, Q.C2 - 0.0492)
        - 8.151 * max(0.0, Q.LHA - 0.24)
        + 13.66 * max(0.0, 0.0634 - Q.girth)
        + 60.33 * max(0.0, Q.girth2 - 0.00809)
        - 786.1 * max(0.0, 0.00157 - Q.girth2)
        - 134.6 * max(0.0, 0.00516 - Q.lam1)
        - 294.5 * max(0.0, 0.00362 - Q.lam2)
        - 0.5432 * max(0.0, 6.28 - Q.log_sum_pt)
        - 0.00295 * max(0.0, Q.mass - 39.2)
        - 3.022 * max(0.0, Q.max_dr - 0.113)
        - 0.002941 * max(0.0, Q.sum_pt - 792.0)
        - 21.66 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21)
        + 634.2 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21)
        + 11.38 * max(0.0, Q.centroid_offset - 0.0143)
        - 18.18 * max(0.0, 0.0501 - Q.centroid_offset)
        + 7.225 * max(0.0, Q.girth - 0.0709)
        - 18.26 * max(0.0, 0.0204 - Q.girth)
        + 26.56 * max(0.0, 0.124 - Q.girth)
        - 0.03917 * max(0.0, Q.mass - 91.2)
        + 8.085 * max(0.0, Q.mass_over_sum_pt - 0.0933)
        - 2.227 * max(0.0, 0.289 - Q.planar_flow)
        + 0.0008904 * max(0.0, 674.0 - Q.sum_pt_top5)
        - 82.61 * max(0.0, 0.0034 - Q.width)
        - 60.75 * max(0.0, 0.00909 - Q.width)
        - 2.151 * max(0.0, 0.0424 - Q.z_7)
        - 351.8 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7)
        - 11.11 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt)
        - 18.72 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21)
        + 0.1271 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7)
        - 58.35 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161)
        + 0.05576 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass)
        + 0.02822 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121)
        + 190.9 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586)
        + 785.6 * max(0.0, Q.e2 - 0.0629)
        - 68.16 * max(0.0, Q.girth2 - 0.0193)
        - 7684.0 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06)
        - 3.303 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4)
        - 30.68 * max(0.0, Q.C2 - 0.0657)
        - 7.775 * max(0.0, 0.0387 - Q.centroid_offset)
        + 189.0 * max(0.0, 0.0489 - Q.e2)
        - 12.09 * max(0.0, 0.143 - Q.girth)
        - 137.1 * max(0.0, 0.00647 - Q.lam1)
        - 112.3 * max(0.0, 0.0157 - Q.lam1)
        - 152.1 * max(0.0, 0.00741 - Q.width)
        + 77.49 * max(0.0, 0.0286 - Q.z_7)
        + 11.31 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt)
        + 0.3159 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7)
        + 1358.0 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset)
        + 42690.0 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset)
        + 0.0003927 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2)
        - 0.003736 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08)
        - 0.0001244 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7)
        + 0.2183 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023)
        + 5.723 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292)
        + 33.64 * max(0.0, Q.C2 - 0.0359)
        + 42.79 * max(0.0, Q.C2 - 0.0669)
        - 38.24 * max(0.0, 0.0356 - Q.C2)
        - 5.814 * max(0.0, Q.centroid_offset - 0.0296)
        + 535.0 * max(0.0, Q.centroid_offset - 0.0496)
        - 6.629 * max(0.0, 0.0437 - Q.e2)
        - 5.273 * max(0.0, 0.0332 - Q.girth)
        - 7.236 * max(0.0, 0.0879 - Q.girth)
        - 648.3 * max(0.0, 0.00457 - Q.girth2)
        + 330.5 * max(0.0, 0.0137 - Q.girth2)
        + 120.3 * max(0.0, Q.lam1 - 0.00419)
        - 82.22 * max(0.0, Q.lam1 - 0.00616)
        - 0.00253 * max(0.0, 457.0 - Q.sum_pt_top5)
        + 1.161 * max(0.0, 0.136 - Q.tau21)
        - 41.6 * max(0.0, 0.00735 - Q.width)
        - 9.638 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset)
        - 16.14 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2)
        - 448.9 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968)
        + 10.15 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419)
        + 22.23 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65)
        + 6.885 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395)
        - 1.975 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378)
        - 438.0 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr)
        + 0.01655 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2)
        + 85.74 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913)
        - 70.49 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184)
        + 2.07 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 0.003557 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt)
        + 146.1 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 153.4 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow)
        - 6.141 * max(0.0, Q.LHA - 0.343)
        - 0.2525 * max(0.0, 0.305 - Q.LHA)
        - 0.852 * max(0.0, Q.e2 - 0.0244)
        - 727.9 * max(0.0, Q.e2 - 0.0497)
        - 735.1 * max(0.0, 0.0627 - Q.e2)
        - 8.207 * max(0.0, Q.girth - 0.0329)
        - 196.8 * max(0.0, 0.00855 - Q.lam1)
        - 351.3 * max(0.0, 0.000335 - Q.lam2)
        + 102.9 * max(0.0, 0.00323 - Q.lam2)
        - 0.03703 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.002116 * max(0.0, 80.4 - Q.mass)
        - 0.3154 * max(0.0, 0.25 - Q.tau21)
        - 219.2 * max(0.0, 0.0136 - Q.width)
        + 13360.0 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244)
        + 1717.0 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 844.2 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.girth2 > 0.009197166189551353:
        if s['g'] - s['t'] > -0.2617592513561249:
            if s['g'] - s['t'] > 0.0590989850461483:
                if s['g'] - s['q'] > -0.0750000812113285:
                    if s['g'] - s['Z'] > 0.2508794441819191:
                        if s['g'] - s['t'] > 0.38286687433719635:
                            return 'g'   # 93% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -5.440409183502197:
                                if s['g'] - s['t'] > 0.2625720649957657:
                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2021653801202774:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.135965824127197:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.25800736248493195:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.0203025341033936:
                                        if Q.girth2 > 0.02334404829889536:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0727127268910408:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 101.66497039794922:
                                                    return 't'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.06705150380730629:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 5.756829261779785:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > 3.506490468978882:
                            return 'g'   # 45% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 96% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['q'] > -0.2680085152387619:
                        if Q.log_sum_pt > 6.638956308364868:
                            return 'q'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 60% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 89% of the training jets here get this class from the formula
            else:
                if s['g'] - s['q'] > -0.06741268560290337:
                    if s['Z'] - s['t'] > -0.11830485239624977:
                        return 'Z'   # 87% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.09283626452088356:
                            if Q.C2 > 0.061866628006100655:
                                if Q.mass_over_sum_pt > 0.098692387342453:
                                    if s['g'] - s['q'] > 2.030332922935486:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 29.5078125:
                                        if Q.lam2 > 0.001849996391683817:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 420.1328125:
                                    if s['Z'] - s['t'] > -10.781240463256836:
                                        if Q.sum_pt > 780.875:
                                            if s['g'] - s['Z'] > 5.185987949371338:
                                                return 't'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 47.489219665527344:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.23496223986148834:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.15129657834768295:
                                        return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.04124322347342968:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.07730357348918915:
                                if Q.pt_7 > 32.609375:
                                    if Q.lam1 > 0.011209431570023298:
                                        if s['g'] - s['Z'] > 5.397094964981079:
                                            return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.011449805926531553:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.023058135993778706:
                                    if Q.log_sum_pt > 6.031323671340942:
                                        if Q.D2 > 0.3527231067419052:
                                            if Q.centroid_offset > 0.03164690267294645:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_pair_mass > 14.611237525939941:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 336.953125:
                                        if Q.mass > 95.38383865356445:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.08549447730183601:
                                                if s['g'] - s['Z'] > 4.491364002227783:
                                                    if Q.max_dr > 0.27870818972587585:
                                                        if Q.max_pair_mass > 1.475390911102295:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 52.072092056274414:
                                                            return 't'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.03496182709932327:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.010045251809060574:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.988654375076294:
                                                    if Q.C2 > 0.044519586488604546:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.2253475859761238:
                                            if s['q'] - s['Z'] > 7.3643012046813965:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 26.40625:
                                                    if Q.pt_7 > 35.390625:
                                                        if Q.lam2 > 0.0007215386140160263:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                else:
                    return 'q'   # 78% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.1989518105983734:
                if s['Z'] - s['t'] > 0.12970490008592606:
                    if s['W'] - s['t'] > -4.738190650939941:
                        return 'Z'   # 96% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > 0.40052708983421326:
                            return 'Z'   # 100% of the training jets here get this class from the formula
                        else:
                            return 't'   # 50% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > -4.293712139129639:
                        if Q.planar_flow > 0.23356449604034424:
                            return 't'   # 63% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.3324090242385864:
                                if Q.lam1 > 0.009460155386477709:
                                    if Q.e2 > 0.05086346156895161:
                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 93% of the training jets here get this class from the formula
                    else:
                        return 't'   # 77% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.2871916741132736:
                    if s['q'] - s['t'] > -0.08419431746006012:
                        if Q.girth2 > 0.011343421880155802:
                            if Q.LHA > 0.44350990653038025:
                                return 'q'   # 91% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > 0.2497832179069519:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.06298992037773132:
                                        return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.749208211898804:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 70% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.LHA > 0.43374669551849365:
                            if Q.pt_7 > 32.578125:
                                return 't'   # 37% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.07958134636282921:
                                return 't'   # 66% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 76% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.6034312546253204:
                        if s['q'] - s['Z'] > 5.714873790740967:
                            if Q.C2 > 0.07021014392375946:
                                if Q.pt_7 > 29.2890625:
                                    if Q.D2 > 1.1654086709022522:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 46.081077575683594:
                                        if Q.eccentricity > 0.9223827719688416:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 328.546875:
                                    if Q.girth > 0.10494691133499146:
                                        if Q.max_dr > 0.23073766380548477:
                                            if s['g'] - s['q'] > 0.014751605223864317:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > 7.771298885345459:
                                                if Q.eccentricity > 0.9939142167568207:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.17137642949819565:
                                            if Q.girth > 0.0998329371213913:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 85% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.3766099214553833:
                                if Q.pt_7 > 27.078125:
                                    if s['g'] - s['Z'] > 0.30418966710567474:
                                        if Q.centroid_offset > 0.05759195797145367:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.26973408460617065:
                                        return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.06396293267607689:
                                    return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.09902337193489075:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > -3.6604775190353394:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 83% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.5722130537033081:
                            if s['q'] - s['W'] > 0.9386627674102783:
                                if Q.planar_flow > 0.07962847501039505:
                                    if Q.eccentricity > 0.9250422418117523:
                                        if Q.e2 > 0.048877425491809845:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.01073992159217596:
                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > -0.6370660364627838:
                                if Q.girth > 0.1008351780474186:
                                    if s['g'] - s['Z'] > 4.912320137023926:
                                        if Q.sum_pt_top5 > 607.640625:
                                            if s['g'] - s['W'] > 9.360849380493164:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.1423315480351448:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 97% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 91% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.9088332653045654:
                                    if Q.sum_pt > 297.1171875:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04656355828046799:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > -0.4104505032300949:
                                        return 'W'   # 48% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 100% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.18530333042144775:
            if s['g'] - s['q'] > 0.033340251073241234:
                if s['g'] - s['t'] > 0.003015218419022858:
                    if s['g'] - s['W'] > 0.06541060656309128:
                        if s['g'] - s['q'] > 0.11147293075919151:
                            if s['g'] - s['t'] > 0.29362934827804565:
                                if s['g'] - s['W'] > 0.31332580745220184:
                                    if s['g'] - s['q'] > 0.1602935865521431:
                                        if Q.pt_7 > 13.0:
                                            if s['g'] - s['W'] > 10.362672328948975:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 32.63522911071777:
                                                    if s['g'] - s['W'] > 0.696587324142456:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.032870834693312645:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['Z'] > 1.2240887880325317:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 37.28125:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 0.7012664079666138:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.15449582040309906:
                                            if Q.log_sum_pt > 6.449284791946411:
                                                if Q.sum_pt_top5 > 537.546875:
                                                    if Q.mass > 26.024495124816895:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.18973758816719055:
                                                            return 'q'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.06598629429936409:
                                                        if Q.pt_7 > 35.15625:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 5.24697208404541:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.023320217616856098:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.17365770041942596:
                                        return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.10052981227636337:
                                            if s['g'] - s['Z'] > 0.5899008512496948:
                                                if Q.girth2 > 0.0033514206297695637:
                                                    if Q.z_7 > 0.058050304651260376:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 5.000232340535149e-05:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.0835394635796547:
                                    return 't'   # 47% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 0.6076339185237885:
                                        if Q.D2 > 1.6403024196624756:
                                            if Q.z_7 > 0.06447828561067581:
                                                if Q.centroid_offset > 0.05138624832034111:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['Z'] > -3.3241357803344727:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 363.359375:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.007624851306900382:
                                                if Q.sum_pt > 419.21875:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 33.414031982421875:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.z_7 > 0.04776947200298309:
                                if Q.sum_pt > 620.328125:
                                    if Q.centroid_offset > 0.0033102036686614156:
                                        if Q.planar_flow > 0.06105818971991539:
                                            if Q.log_sum_pt > 6.5650670528411865:
                                                if Q.pt_7 > 42.46875:
                                                    if Q.mass > 6.46497106552124:
                                                        if Q.log_sum_pt > 6.6626136302948:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.006664115469902754:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['q'] > 0.05536757409572601:
                                                            if Q.log_sum_pt > 6.627317667007446:
                                                                if s['Z'] - s['t'] > 2.299502372741699:
                                                                    if Q.e2 > 0.0027772708563134074:
                                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.0716763474047184:
                                                        if s['W'] - s['Z'] > -0.059752846136689186:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.09336400032043457:
                                                            if Q.max_dr > 0.05279206112027168:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.006274574436247349:
                                                    if Q.max_dr > 0.0761098712682724:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 632.609375:
                                                            return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 2.8218599557876587:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 531.765625:
                                                if Q.log_sum_pt > 6.822793006896973:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.04230549931526184:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0020801156060770154:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.11030837148427963:
                                                    if s['q'] - s['W'] > 2.3926066160202026:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.01866560149937868:
                                        if Q.planar_flow > 0.06809847429394722:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 1.4109578728675842:
                                    if s['W'] - s['t'] > -2.0776236057281494:
                                        if Q.z_7 > 0.02142875548452139:
                                            if Q.z_7 > 0.04517990164458752:
                                                if Q.centroid_offset > 0.008608575910329819:
                                                    if Q.lam2 > 2.4559573830629233e-05:
                                                        if s['g'] - s['q'] > 0.08193737268447876:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.16901283711194992:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 869.53125:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.6555280983448029:
                                            if s['g'] - s['q'] > 0.04492833465337753:
                                                if Q.log_sum_pt > 6.458068609237671:
                                                    if Q.LHA > 0.1801827922463417:
                                                        if Q.mass > 25.12407398223877:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 2.0744056701660156:
                                                            if Q.sum_pt > 838.828125:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 11.779612064361572:
                                                return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.19595422595739365:
                            if Q.mass > 33.71452331542969:
                                if Q.girth > 0.051309214904904366:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.14027226716279984:
                                        return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.9768152236938477:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.6807028949260712:
                                    if Q.max_dr > 0.16063254326581955:
                                        return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.0337523240596056:
                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.9604735970497131:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 29.714153289794922:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.05552361719310284:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4879138469696045:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 0.09737134724855423:
                                        if Q.e2 > 0.003654091851785779:
                                            if s['g'] - s['Z'] > 0.46540121734142303:
                                                if Q.eccentricity > 0.8265347480773926:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.12061475589871407:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 33.2598762512207:
                                if s['g'] - s['t'] > 1.292481243610382:
                                    if Q.C2 > 0.03262944892048836:
                                        return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.0882137082517147:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.686089962720871:
                                    if s['g'] - s['q'] > 0.8075485825538635:
                                        if Q.girth2 > 0.004074774449691176:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 28.63732147216797:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 88% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.23165678977966309:
                        if s['W'] - s['t'] > 0.2060590460896492:
                            return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.z_7 > 0.0670614093542099:
                                return 'g'   # 41% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 76% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.30857519805431366:
                            if Q.z_dr_0p1_0p2 > 0.18831666558980942:
                                if Q.mass_over_sum_pt > 0.0852297842502594:
                                    return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 26.625:
                                    if s['g'] - s['t'] > -0.1442306712269783:
                                        if Q.mass > 53.87432098388672:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 7.995654344558716:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.022591357119381428:
                                                    if Q.e2 > 0.034458549693226814:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 505.859375:
                                        return 't'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 477.546875:
                                return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.057697610929608345:
                                    return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.5952581167221069:
                                        if s['q'] - s['W'] > 4.751234769821167:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 80% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > -0.02652248553931713:
                    if s['q'] - s['t'] > -0.0008856177737470716:
                        if s['g'] - s['q'] > -0.06430241465568542:
                            if Q.pt_7 > 35.359375:
                                if Q.eccentricity > 0.9905053377151489:
                                    if Q.sum_pt > 738.25:
                                        return 'q'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > -0.001043887052219361:
                                        if Q.centroid_offset > 0.0030985987978056073:
                                            if Q.z_7 > 0.04194701090455055:
                                                if Q.max_dr > 0.06693264842033386:
                                                    if Q.sum_pt > 775.71875:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if s['Z'] - s['t'] > 2.344399571418762:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.0890285074710846:
                                                    if Q.log_sum_pt > 6.615374565124512:
                                                        if Q.width > 0.00011428538709878922:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 633.65625:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 4.074429750442505:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.406106233596802:
                                            if Q.sum_pt > 1054.953125:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 9.859870433807373:
                                                    if Q.log_sum_pt > 6.651733160018921:
                                                        if Q.centroid_offset > 0.004434495931491256:
                                                            if Q.log_sum_pt > 6.734534502029419:
                                                                return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.007325998041778803:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.567836284637451:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.004042332526296377:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.03947780467569828:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 63% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.012967209797352552:
                                    if Q.sum_pt > 630.0234375:
                                        if Q.sum_pt_top5 > 768.640625:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11896103993058205:
                                                if Q.e2 > 0.014184732921421528:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 6.131764888763428:
                                                    if s['g'] - s['t'] > 0.7266537249088287:
                                                        if s['g'] - s['q'] > -0.003293877001851797:
                                                            if Q.z_7 > 0.04333481937646866:
                                                                return 'q'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.014824233949184418:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['W'] > 0.21684762835502625:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 9.7384508990217e-05:
                                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 1.2230717539787292:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 1.380026060360251e-05:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.009805041830986738:
                                            if s['W'] - s['Z'] > 0.4867893010377884:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.19158847630023956:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 38% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05427619442343712:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 0.0011744697258109227:
                                                    return 'g'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.642374753952026:
                                        if Q.z_7 > 0.0390995591878891:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > 1.7360094785690308:
                                                if Q.sum_pt > 1209.1796875:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 938.0:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.011429014150053263:
                                            if s['g'] - s['q'] > -0.004438355448655784:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.041991058737039566:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.21292444318532944:
                                if s['g'] - s['q'] > -0.1489449366927147:
                                    if Q.sum_pt_top5 > 952.5625:
                                        if Q.sum_pt_top5 > 1130.40625:
                                            return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9901085197925568:
                                            if Q.max_dr > 0.07414573058485985:
                                                if s['W'] - s['Z'] > 0.5760534107685089:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 33.609375:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.000151242216816172:
                                                    if s['q'] - s['t'] > 0.5013381987810135:
                                                        if Q.sum_pt_top5 > 784.265625:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 8.308286851388402e-06:
                                                                return 'q'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.635915040969849:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.004012031247839332:
                                                            if s['g'] - s['q'] > -0.09171128645539284:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 0.32658813893795013:
                                        if s['q'] - s['W'] > 0.5159193575382233:
                                            if Q.sum_pt > 1396.21875:
                                                if s['q'] - s['Z'] > 2.750488758087158:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.002845590002834797:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.054691024124622345:
                                                    if Q.lam1 > 0.001666595519054681:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > 0.7563762366771698:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 42.43354797363281:
                                            return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.022234086878597736:
                                                return 't'   # 41% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.432563066482544:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 1.0440113544464111:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.04400346428155899:
                                    if s['q'] - s['Z'] > 0.767114669084549:
                                        return 'q'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > 0.7017346620559692:
                                        if s['q'] - s['Z'] > 0.3297937363386154:
                                            if Q.sum_pt > 900.1328125:
                                                if s['g'] - s['q'] > -0.34583038091659546:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['W'] > 0.12160495668649673:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9366087019443512:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02225838601589203:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.16296962648630142:
                                                        if s['g'] - s['W'] > -0.9637495875358582:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00015898122364887968:
                                                            if s['W'] - s['Z'] > 0.5312397181987762:
                                                                return 'q'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_pair_mass > 2.4741262197494507:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 1.3635094546771143e-05:
                                                    if Q.log_sum_pt > 6.790330410003662:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.24470586329698563:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 47% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.19041290134191513:
                            if Q.mass_over_sum_pt > 0.0818958692252636:
                                return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -2.966715931892395:
                                    return 't'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.03798227943480015:
                                        return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.28606365621089935:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                        else:
                            return 't'   # 94% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.3935696929693222:
                        if Q.C2 > 0.040369968861341476:
                            if s['q'] - s['Z'] > 0.6567349135875702:
                                if Q.e2 > 0.011761077214032412:
                                    return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.025815804488956928:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 91% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > 0.9278695583343506:
                                if Q.planar_flow > 0.167819544672966:
                                    if s['q'] - s['Z'] > 0.39972952008247375:
                                        if Q.eccentricity > 0.8065727651119232:
                                            if Q.log_sum_pt > 6.794708490371704:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.161154642701149:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.151003859937191:
                                            if Q.centroid_offset > 0.020182163454592228:
                                                return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.5539283454418182:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.12089546397328377:
                                                    if s['g'] - s['q'] > -1.092879056930542:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 77% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['t'] > -0.06524127535521984:
                            if s['q'] - s['W'] > -0.6495448052883148:
                                if s['q'] - s['t'] > 1.1533547043800354:
                                    if Q.max_dr > 0.13784946501255035:
                                        return 'W'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > 0.7524299919605255:
                                            return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 98% of the training jets here get this class from the formula
                        else:
                            return 't'   # 76% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.04878599941730499:
                if s['g'] - s['W'] > 0.01618628203868866:
                    if s['g'] - s['W'] > 0.21395431458950043:
                        if s['g'] - s['t'] > -0.08235682174563408:
                            if s['g'] - s['Z'] > 0.649870753288269:
                                if Q.mass > 50.108022689819336:
                                    if s['g'] - s['Z'] > 1.4377657175064087:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -2.727085828781128:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 28.11927318572998:
                                        if s['g'] - s['Z'] > 0.7992858290672302:
                                            if s['g'] - s['t'] > 2.183339834213257:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 34.80418395996094:
                                                    if s['g'] - s['W'] > 0.6871399581432343:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 51.421875:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.15200550854206085:
                                    if s['g'] - s['q'] > 1.208725392818451:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.46125921607017517:
                                            if s['g'] - s['W'] > 0.33307021856307983:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 585.84375:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9153952300548553:
                                                        if Q.log_sum_pt > 6.495477914810181:
                                                            if s['g'] - s['t'] > 2.653312921524048:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.n_pt_above_50 > 5.5:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > 0.05836205370724201:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.4911091923713684:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 726.84375:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 621.765625:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.020053057000041008:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 80% of the training jets here get this class from the formula
                        else:
                            return 't'   # 89% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 0.502947062253952:
                            if s['g'] - s['t'] > 0.08728914707899094:
                                if Q.mass > 32.421932220458984:
                                    if s['g'] - s['t'] > 2.575908660888672:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 56% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 1.0538830161094666:
                                        if Q.girth > 0.05338112823665142:
                                            if Q.lam1 > 0.003022161894477904:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.17829854041337967:
                                            if s['g'] - s['Z'] > 0.5948907136917114:
                                                if Q.max_dr > 0.029032031074166298:
                                                    if Q.planar_flow > 0.6263967752456665:
                                                        if Q.D2 > 1.7839996218681335:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > 0.12247653678059578:
                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 616.828125:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                return 't'   # 52% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['q'] > 1.182551383972168:
                                if Q.mass > 25.571269035339355:
                                    if s['Z'] - s['t'] > -0.006857765605673194:
                                        if Q.LHA > 0.24879102408885956:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.01519074384123087:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.36628463864326477:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 1.4110556244850159:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.026915723457932472:
                                    if Q.LHA > 0.2195109724998474:
                                        return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.17611627280712128:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.09674692526459694:
                                                if s['g'] - s['Z'] > 0.09200100228190422:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.19433102011680603:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.019612998701632023:
                                        if s['q'] - s['t'] > -0.7293947637081146:
                                            if s['g'] - s['Z'] > 0.3835330903530121:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.01874243374913931:
                                                    if Q.pt_7 > 34.84375:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > 0.09383166953921318:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 57% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.16678138822317123:
                        if s['g'] - s['W'] > -0.2985376566648483:
                            if s['g'] - s['q'] > 1.0752930045127869:
                                if Q.LHA > 0.24285904318094254:
                                    if s['W'] - s['t'] > -0.03379435744136572:
                                        if Q.z_dr_0p1_0p2 > 0.21240603923797607:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.028041623532772064:
                                                if Q.sum_pt > 398.5:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > 0.5763267874717712:
                                                    if Q.z_7 > 0.0702344998717308:
                                                        if Q.max_dr > 0.08792848885059357:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.5158210098743439:
                                        if Q.mass > 34.20643997192383:
                                            if Q.max_dr > 0.16082142293453217:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -0.650636225938797:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.05163612402975559:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 47.203125:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.026599744334816933:
                                            if s['g'] - s['q'] > 1.3293471932411194:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0010055426973849535:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.45105113089084625:
                                    if s['q'] - s['t'] > -0.9878169596195221:
                                        if Q.lam1 > 0.00048305092786904424:
                                            if Q.width > 0.002681807498447597:
                                                if Q.z_7 > 0.07467691972851753:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.713258683681488:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 35.84375:
                                                        if Q.log_sum_pt > 6.631878137588501:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.06002223305404186:
                                                                return 'g'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 73% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.3505493253469467:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.0451949629932642:
                                if s['W'] - s['Z'] > 0.356755331158638:
                                    if s['g'] - s['W'] > -0.5736817717552185:
                                        if s['g'] - s['Z'] > 0.9175727963447571:
                                            if Q.centroid_offset > 0.015761242248117924:
                                                if Q.LHA > 0.23688892275094986:
                                                    return 'W'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 1.1862569451332092:
                                                if Q.mass_over_sum_pt > 0.047886911779642105:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 0.3530934453010559:
                                                        if Q.e2 > 0.017726109363138676:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.25284819304943085:
                                                    if Q.girth2 > 0.002901937812566757:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.019512105733156204:
                                                            if Q.mass_over_sum_pt > 0.045919882133603096:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1075.4921875:
                                            if s['q'] - s['Z'] > -2.1535422801971436:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.024947253055870533:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 0.4960751235485077:
                                                if s['g'] - s['W'] > -0.9252465963363647:
                                                    if s['g'] - s['Z'] > 0.8769378364086151:
                                                        if Q.centroid_offset > 0.016289460472762585:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['Z'] > 0.5970439612865448:
                                                        return 'W'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.07841900736093521:
                                                            if Q.LHA > 0.18822933733463287:
                                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.0668147802352905:
                                                    if Q.z_7 > 0.07597947865724564:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1045.83203125:
                                        if Q.tau21 > 0.18791325390338898:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -3.0073916912078857:
                                                if Q.girth > 0.04381086118519306:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.17435244470834732:
                                            if Q.max_dr > 0.28149253129959106:
                                                if Q.LHA > 0.1858067587018013:
                                                    if Q.D2 > 4.720690011978149:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > 0.22419053316116333:
                                                    if Q.lam2 > 0.00026859977515414357:
                                                        if Q.tau21 > 0.1771509274840355:
                                                            if Q.lam1 > 0.005594947841018438:
                                                                if Q.D2 > 0.9050803184509277:
                                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.026785860769450665:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['Z'] > -0.7007799446582794:
                                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 3.7446894793902175e-06:
                                                        if Q.D2 > 4.631124496459961:
                                                            if Q.girth > 0.03406944498419762:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > -1.5760830640792847:
                                                                if Q.mass_over_sum_pt > 0.07241344079375267:
                                                                    if Q.C2 > 0.03520067222416401:
                                                                        if s['g'] - s['W'] > -2.179661989212036:
                                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 680.65625:
                                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.24315254390239716:
                                    if Q.centroid_offset > 0.013204785995185375:
                                        if Q.z_dr_0p1_0p2 > 0.21792352199554443:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.06860103830695152:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > -1.1761738061904907:
                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.LHA > 0.18245602399110794:
                            if s['W'] - s['Z'] > 0.051393626257777214:
                                if s['W'] - s['t'] > -0.11350298672914505:
                                    if Q.z_7 > 0.02521111909300089:
                                        if s['g'] - s['t'] > 2.144441246986389:
                                            if s['q'] - s['Z'] > -1.004179060459137:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.011458339635282755:
                                                    if s['g'] - s['t'] > 2.439116954803467:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > 0.06375687196850777:
                                                if s['q'] - s['Z'] > -3.310081720352173:
                                                    if Q.e2 > 0.012263946700841188:
                                                        if Q.eccentricity > 0.8830162584781647:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_pair_mass > 2.291175603866577:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['t'] > 0.6314431726932526:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['W'] > -1.1457584500312805:
                                                            if s['g'] - s['t'] > 1.45077383518219:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.005004511214792728:
                                                                    if Q.lam2 > 0.0001258300617337227:
                                                                        return 'Z'   # 44% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 5.0552662287373096e-05:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.0028181009693071246:
                                                                    if Q.C2 > 0.041506171226501465:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['Z'] > 0.0983348898589611:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.10861120745539665:
                                                    if s['W'] - s['Z'] > 0.13201207667589188:
                                                        if Q.tau21 > 0.3891858458518982:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.005513546988368034:
                                                            if Q.D2 > 0.8739675879478455:
                                                                if s['g'] - s['Z'] > -1.780631184577942:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.9913247525691986:
                                                                    if Q.centroid_offset > 0.003524819272570312:
                                                                        if Q.lam1 > 0.006807749858126044:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            if s['g'] - s['Z'] > -3.1930394172668457:
                                                                                if s['W'] - s['Z'] > 0.09161665290594101:
                                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 0.766207754611969:
                                                                if s['q'] - s['t'] > 0.6706714034080505:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.025185699574649334:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.032610608264803886:
                                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.C2 > 0.031040429137647152:
                                                                                if Q.e2 > 0.0221668416634202:
                                                                                    if Q.sum_pt_top5 > 608.125:
                                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.C2 > 0.023224695585668087:
                                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.07799942791461945:
                                                        if Q.C2 > 0.01718926802277565:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0020049097947776318:
                                            if Q.D2 > 4.3255295753479:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > -1.3611990213394165:
                                                    if Q.girth > 0.0406586192548275:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 2.963531017303467:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.03313986398279667:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.0018151375115849078:
                                    if Q.max_dr > 0.14998450130224228:
                                        if Q.planar_flow > 0.09676310420036316:
                                            if Q.e2 > 0.016910351812839508:
                                                if Q.lam2 > 0.00033535834518261254:
                                                    if s['g'] - s['t'] > 0.27255189418792725:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.002165351528674364:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > -1.2674521207809448:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 735.390625:
                                                    if s['g'] - s['t'] > -1.0437846183776855:
                                                        if Q.e2 > 0.012049244251102209:
                                                            if Q.mass > 46.61056137084961:
                                                                if Q.max_pair_mass > 1.4948359727859497:
                                                                    if s['g'] - s['q'] > -1.1986960768699646:
                                                                        if s['W'] - s['t'] > 2.643666625022888:
                                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.022038581781089306:
                                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.002173332031816244:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.005267318105325103:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.00843567680567503:
                                            if Q.z_dr_0p1_0p2 > 0.055131712928414345:
                                                if Q.lam1 > 0.005491319578140974:
                                                    if Q.width > 0.006898917490616441:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 7.443381764460355e-05:
                                                            if Q.D2 > 0.7907927334308624:
                                                                if s['g'] - s['Z'] > -1.1127036809921265:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.28960123658180237:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.07512111961841583:
                                                                    if s['W'] - s['Z'] > 0.00912953820079565:
                                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.006697997683659196:
                                                                if Q.sum_pt > 735.546875:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.020847821608185768:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.log_sum_pt > 6.820860385894775:
                                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.83108115196228:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.8513200283050537:
                                                            if s['g'] - s['t'] > 0.3569728881120682:
                                                                if Q.girth2 > 0.002828290918841958:
                                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.620635032653809:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.log_sum_pt > 6.404555082321167:
                                                                        if Q.mass > 42.35992240905762:
                                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02726675570011139:
                                                    if s['g'] - s['t'] > 2.2254377603530884:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.08067991584539413:
                                                            if Q.tau21 > 0.18226203322410583:
                                                                if Q.e2 > 0.014634219463914633:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau21 > 0.33282534778118134:
                                                                        if Q.eccentricity > 0.9090605974197388:
                                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -0.8814332485198975:
                                                if Q.girth > 0.03112979047000408:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if s['Z'] - s['t'] > 2.5984301567077637:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.028391952626407146:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 65% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > -1.0843443274497986:
                                if s['q'] - s['W'] > 0.06273444183170795:
                                    return 'q'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.02248226758092642:
                                    if s['g'] - s['t'] > 2.6603527069091797:
                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 82% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > -0.010175571776926517:
                    if s['g'] - s['t'] > 0.0206719059497118:
                        if s['g'] - s['Z'] > 0.22168569266796112:
                            if s['g'] - s['Z'] > 0.51680988073349:
                                if s['g'] - s['t'] > 0.2674921154975891:
                                    if Q.D2 > 0.48287734389305115:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > 1.8676839470863342:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9884056150913239:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.9801725447177887:
                                    if s['q'] - s['W'] > 0.35475559532642365:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.036482442170381546:
                                            if Q.e2 > 0.020072123035788536:
                                                if s['W'] - s['Z'] > -0.4744156450033188:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 1.505529761314392:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 0.36612236499786377:
                                                        if s['q'] - s['W'] > -0.10290106013417244:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 787.6875:
                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 25.47726345062256:
                                        if s['q'] - s['W'] > 0.6864488124847412:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.12384838238358498:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.278674840927124:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.09974043071269989:
                                if s['g'] - s['Z'] > 0.07744241505861282:
                                    if Q.mass_over_sum_pt > 0.025244570337235928:
                                        if s['g'] - s['W'] > 0.45366938412189484:
                                            if s['g'] - s['t'] > 0.13437829166650772:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.033448582515120506:
                                            if Q.sum_pt_top5 > 448.8125:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 405.65625:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 32.390625:
                                        if Q.pt_7 > 50.765625:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 8.464843273162842:
                                                if s['g'] - s['W'] > 0.22577019780874252:
                                                    if Q.sum_pt_top5 > 507.828125:
                                                        if Q.eccentricity > 0.9164234101772308:
                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.3140576034784317:
                                            return 'W'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > -1.1588045358657837:
                                    if Q.centroid_offset > 0.0325651653110981:
                                        if Q.e2 > 0.013802758418023586:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 0.33829137682914734:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.09747092798352242:
                                            return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.11884213611483574:
                                                if Q.centroid_offset > 0.026076968759298325:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 0.07543215528130531:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 655.046875:
                                            if Q.centroid_offset > 0.014487081672996283:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 60% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.1874193772673607:
                            if Q.sum_pt > 567.75:
                                if Q.max_dr > 0.23063786327838898:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 49% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.8336440920829773:
                                    if Q.log_sum_pt > 6.186832427978516:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.08112657070159912:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            return 't'   # 97% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.07243525981903076:
                        if s['W'] - s['Z'] > -0.2531536966562271:
                            if Q.LHA > 0.18046343326568604:
                                if Q.sum_pt_top5 > 420.609375:
                                    if s['q'] - s['Z'] > -0.7561671137809753:
                                        if Q.mass > 29.934667587280273:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.027374540455639362:
                                                if s['q'] - s['W'] > -0.02506269793957472:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0033996697748079896:
                                                        if s['Z'] - s['t'] > 1.975784182548523:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 2.8536983728408813:
                                                                if Q.centroid_offset > 0.031009788624942303:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 782.9609375:
                                                    if Q.centroid_offset > 0.025284675881266594:
                                                        if s['W'] - s['t'] > 2.6014366149902344:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_pair_mass > 7.0896689891815186:
                                            if Q.z_dr_0p1_0p2 > 0.060875946655869484:
                                                if Q.mass_over_sum_pt > 0.07364705950021744:
                                                    if Q.z_dr_0p1_0p2 > 0.2991398870944977:
                                                        if Q.centroid_offset > 0.0063357551116496325:
                                                            if Q.LHA > 0.30396999418735504:
                                                                if Q.z_dr_0p1_0p2 > 0.33821330964565277:
                                                                    if s['q'] - s['t'] > -2.300359845161438:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006863145157694817:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 3.1767311156727374e-05:
                                                                if Q.sum_pt > 695.7890625:
                                                                    if s['g'] - s['Z'] > -2.819478154182434:
                                                                        if s['Z'] - s['t'] > 1.670314610004425:
                                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.0716802105307579:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > -0.11530221626162529:
                                                            if s['g'] - s['t'] > 0.029608899727463722:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1027.84765625:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.606058835983276:
                                                                    if s['g'] - s['Z'] > -3.407417416572571:
                                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 818.6640625:
                                                    if Q.lam1 > 0.003970115212723613:
                                                        if Q.mass_over_sum_pt > 0.07304857671260834:
                                                            if s['g'] - s['t'] > -1.495649516582489:
                                                                if s['g'] - s['W'] > -3.3162606954574585:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.4210241287946701:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if s['Z'] - s['t'] > 0.45129503309726715:
                                                        if Q.centroid_offset > 0.006992470705881715:
                                                            if Q.centroid_offset > 0.03315889649093151:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.006824157200753689:
                                                                return 'W'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -0.13072405755519867:
                                                if Q.mass > 9.663259983062744:
                                                    if Q.max_dr > 0.14680831879377365:
                                                        if Q.lam2 > 8.377547055715695e-05:
                                                            if Q.eccentricity > 0.9001788198947906:
                                                                if Q.log_sum_pt > 6.576424598693848:
                                                                    if Q.girth > 0.04245533607900143:
                                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 673.71875:
                                                                if Q.max_dr > 0.20644459128379822:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['q'] - s['t'] > 0.6543128192424774:
                                                                        if Q.e2 > 0.014781243167817593:
                                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > -1.8612375259399414:
                                                            if Q.mass > 42.819440841674805:
                                                                if Q.z_dr_0p1_0p2 > 0.18553874641656876:
                                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['q'] - s['t'] > -0.7114353477954865:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 0.7338355481624603:
                                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.10866736993193626:
                                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.016002334654331207:
                                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.027415361255407333:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.265129417181015:
                                                    if Q.max_dr > 0.10321034491062164:
                                                        if s['g'] - s['Z'] > -3.0996843576431274:
                                                            if Q.sum_pt > 693.5390625:
                                                                if Q.C2 > 0.020160225220024586:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.059799665585160255:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.008982316125184298:
                                                        if s['g'] - s['t'] > 0.6750843822956085:
                                                            if s['q'] - s['Z'] > -1.0803046822547913:
                                                                if Q.tau21 > 0.6287627816200256:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.planar_flow > 0.25653018057346344:
                                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > -0.9581175446510315:
                                        if Q.sum_pt_top5 > 379.09375:
                                            if s['g'] - s['W'] > -0.36101095378398895:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.2307731807231903:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.8120259940624237:
                                            if Q.sum_pt_top5 > 365.359375:
                                                if s['W'] - s['Z'] > -0.10585254058241844:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.01757551822811365:
                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['q'] > 0.43148307502269745:
                                    if Q.mass > 7.846825361251831:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.11328848823904991:
                                            if s['g'] - s['q'] > 0.7021167278289795:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -1.0137640237808228:
                                        if Q.girth > 0.02730832900851965:
                                            return 'W'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > 0.06141149252653122:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.1769547313451767:
                                                    if Q.log_sum_pt > 6.721218109130859:
                                                        if Q.mass_over_sum_pt > 0.007774864789098501:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.17201115936040878:
                                            if Q.sum_pt > 914.8828125:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > -0.4145319014787674:
                                if s['q'] - s['W'] > 0.47255177795886993:
                                    if s['q'] - s['Z'] > -0.17609160393476486:
                                        if Q.z_dr_0p1_0p2 > 0.03854451701045036:
                                            if Q.max_dr > 0.16578059643507004:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.022862371057271957:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.1311599463224411:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -2.844177484512329:
                                                if Q.max_pair_mass > 1.0988101363182068:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.02870251052081585:
                                        if Q.pt_7 > 24.203125:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -0.21002031862735748:
                                                if Q.sum_pt_top5 > 671.6796875:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.4561001658439636:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 62% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.33744537830352783:
                                    if s['g'] - s['Z'] > -0.33357618749141693:
                                        if s['g'] - s['W'] > 1.0260867476463318:
                                            if Q.log_sum_pt > 6.488467454910278:
                                                if Q.D2 > 0.5718351602554321:
                                                    if s['W'] - s['t'] > -0.21847063302993774:
                                                        if s['g'] - s['q'] > 1.3269705176353455:
                                                            if Q.eccentricity > 0.9872713983058929:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.03560681454837322:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 5.953526973724365:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 17.945018768310547:
                                                if s['W'] - s['Z'] > -0.4661150127649307:
                                                    if Q.LHA > 0.25789758563041687:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.294462502002716:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.001999506726861:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > -0.20974790304899216:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.5176551342010498:
                                                    if Q.C2 > 0.005259721307083964:
                                                        if s['g'] - s['t'] > 2.053423047065735:
                                                            if s['g'] - s['q'] > 1.8269001245498657:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.007774852914735675:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.5293352007865906:
                                            if Q.LHA > 0.17844177037477493:
                                                if Q.sum_pt_top5 > 394.515625:
                                                    if Q.max_pair_mass > 8.860328674316406:
                                                        if Q.centroid_offset > 0.018803227692842484:
                                                            if Q.e2 > 0.037492359057068825:
                                                                if Q.girth > 0.07820180431008339:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.03803661838173866:
                                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p1_0p2 > 0.06712707132101059:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.017259737476706505:
                                                                        if s['Z'] - s['t'] > 1.873242437839508:
                                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.girth2 > 0.003180554253049195:
                                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.007040947675704956:
                                                                if Q.centroid_offset > 0.00853881984949112:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -0.6348473429679871:
                                                            if s['W'] - s['t'] > 2.0723636150360107:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.006764269899576902:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 2.3778737783432007:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.028436514548957348:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 3.440426826477051:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 3.561716914176941:
                                                if s['g'] - s['t'] > 0.6148157715797424:
                                                    if Q.sum_pt > 1006.34375:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if s['Z'] - s['t'] > 0.6776107251644135:
                                                        if s['q'] - s['Z'] > -0.8444299399852753:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['Z'] > -1.0073450803756714:
                                                                if s['g'] - s['W'] > 5.41080904006958:
                                                                    return 'g'   # 44% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 7.4823918112088e-05:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.25447241961956024:
                                                                if Q.pt_7 > 30.6171875:
                                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.14523658901453018:
                                                    if s['q'] - s['Z'] > -0.6619211137294769:
                                                        if Q.D2 > 3.580392360687256:
                                                            if Q.centroid_offset > 0.024162321351468563:
                                                                if Q.tau21 > 0.29175877571105957:
                                                                    if Q.max_dr > 0.22879870980978012:
                                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['W'] > 2.8963286876678467:
                                        if Q.max_dr > 0.134804405272007:
                                            if Q.girth > 0.057000527158379555:
                                                if s['g'] - s['t'] > -0.26659607887268066:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00030281553335953504:
                                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 0.20863719284534454:
                                            if Q.girth > 0.0488218292593956:
                                                if Q.max_pair_mass > 20.561667442321777:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.4285687357187271:
                                                        return 'Z'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['t'] > -3.5454742908477783:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9796154797077179:
                                                                return 't'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 41.15821647644043:
                                                if Q.LHA > 0.3054163157939911:
                                                    if Q.lam2 > 3.1652531106374227e-05:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > 1.1044092774391174:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.20790203660726547:
                                                        if Q.mass_over_sum_pt > 0.08082447573542595:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > 0.5464074313640594:
                                                    if Q.tau21 > 0.22272000461816788:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9457746148109436:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.1942535787820816:
                            if Q.width > 0.008844251278787851:
                                if Q.log_sum_pt > 6.461992263793945:
                                    if Q.pt_7 > 34.03125:
                                        return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > -0.062302179634571075:
                                    if s['W'] - s['t'] > -0.3525448143482208:
                                        if Q.eccentricity > 0.9762446284294128:
                                            return 'W'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.023744293488562107:
                                            if Q.eccentricity > 0.9896151423454285:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.03823903948068619:
                                                    if Q.LHA > 0.3074214607477188:
                                                        if Q.z_dr_0p1_0p2 > 0.2668079435825348:
                                                            return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 42% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 32.109375:
                                                if s['W'] - s['Z'] > -4.4204018115997314:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.029016917571425438:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > -0.5883713662624359:
                                        if s['q'] - s['Z'] > -0.20329993218183517:
                                            return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.23694246262311935:
                                                if Q.LHA > 0.3094521313905716:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.1602933630347252:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -0.4098541885614395:
                                if Q.width > 0.009012872353196144:
                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0015412381035275757:
                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 90% of the training jets here get this class from the formula
                            else:
                                return 't'   # 97% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
