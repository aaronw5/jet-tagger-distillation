"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 65.18% (the formula: 65.33%); same class as the formula for 93.29% of jets.  46 leaves, depth 9.
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
        max_dr=max(dr[i] for i in real),
        pt_0=pt[0],
        pt_7=pt[7],
        z_4=z[4],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_1=phi[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def score_g(Q):
    return (15.96
        + 5.799 * max(0.0, 0.0338 - Q.centroid_offset)
        + 60.69 * max(0.0, 0.0775 - Q.girth)
        - 132.9 * max(0.0, 0.0121 - Q.girth2)
        - 0.05479 * max(0.0, 22.5 - Q.mass)
        - 0.05875 * max(0.0, 71.3 - Q.mass)
        - 204.6 * max(0.0, 0.00449 - Q.width)
        - 718.2 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 234.0 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 1.393 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.0002699 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 5.1 * max(0.0, 0.0491 - Q.C2)
        - 611.3 * max(0.0, 0.00775 - Q.e2_sq)
        + 4.68 * max(0.0, Q.log_sum_pt - 6.41)
        + 0.1093 * max(0.0, Q.pt_7 - 30.4)
        + 506.0 * max(0.0, 0.009 - Q.width)
        + 80.12 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 18.59 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 928.6 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 8886.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 47.85 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        + 2713.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 0.0003745 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        - 11.92 * max(0.0, Q.LHA - 0.116)
        + 87.85 * max(0.0, 0.0053 - Q.lam1)
        + 25.71 * max(0.0, Q.log_sum_pt - 6.89)
        + 0.1655 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.02061 * max(0.0, Q.pt_7 - 30.2)
        - 0.03159 * max(0.0, 54.2 - Q.pt_7)
        + 0.00201 * max(0.0, 788.0 - Q.sum_pt)
        - 24.57 * max(0.0, Q.z_7 - 0.045)
        - 22830.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.5041 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        - 0.1347 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        + 3.218 * max(0.0, Q.centroid_offset - 0.0103)
        + 5.405 * max(0.0, Q.e2 - 0.0278)
        + 0.796 * max(0.0, 0.0433 - Q.e2)
        - 800.8 * max(0.0, 0.00905 - Q.girth2)
        + 49.43 * max(0.0, 0.0126 - Q.girth2)
        + 0.08145 * max(0.0, Q.mass - 71.9)
        + 26.77 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 105.3 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 583.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        + 108.3 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        + 0.6207 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 1.221 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        + 28.53 * max(0.0, Q.C2 - 0.00878)
        - 7.733 * max(0.0, 0.000389 - Q.lam2)
        + 126.9 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 9.681 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        - 0.00254 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 0.7372 * max(0.0, 0.279 - Q.tau21)
        - 147.1 * max(0.0, Q.width - 0.00219)
        + 12.99 * max(0.0, 0.0374 - Q.e2)
        - 14.5 * max(0.0, 0.0333 - Q.z_7)
        - 12.14 * max(0.0, 0.0683 - Q.z_7)
        - 17.14 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        - 32.04 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 156.1 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        - 29810.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 910.1 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        - 25.47 * max(0.0, Q.C2 - 0.0101)
        + 0.3122 * max(0.0, Q.centroid_offset - 0.0212)
        + 148.3 * max(0.0, 0.0508 - Q.e2)
        + 275.4 * max(0.0, 0.00328 - Q.e2_sq)
        + 73.49 * max(0.0, Q.girth - 0.0868)
        - 761.4 * max(0.0, 0.00862 - Q.girth2)
        - 50.89 * max(0.0, 0.00802 - Q.lam1)
        - 2.788 * max(0.0, Q.lam2 - 0.0029)
        - 73.94 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        + 0.2909 * max(0.0, Q.max_dr - 0.0279)
        - 28.25 * max(0.0, 0.0133 - Q.width)
        + 0.09389 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 1267.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 0.06203 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 98.04 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 164.8 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 5.056 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 24.98 * max(0.0, 0.0201 - Q.centroid_offset)
        + 30.75 * max(0.0, 0.0253 - Q.e2)
        - 22.1 * max(0.0, 0.0372 - Q.e2)
        - 404.5 * max(0.0, 0.00115 - Q.e2_sq)
        + 8.099 * max(0.0, 0.089 - Q.girth)
        + 208.1 * max(0.0, Q.girth2 - 0.00445)
        - 618.4 * max(0.0, 0.000708 - Q.girth2)
        - 9.932e-05 * max(0.0, Q.mass - 80.4)
        - 33.25 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        - 10.33 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 426.0 * max(0.0, 0.00564 - Q.width)
        + 0.8396 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 35.68 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 1527.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        - 89.23 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        - 1.833e-05 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        - 12.42 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.3681 * max(0.0, Q.log_sum_pt - 6.71)
        + 11950.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        + 0.07197 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 136.0 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 1.448 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 18850.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 2334.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 7.353 * max(0.0, 0.0174 - Q.centroid_offset)
        + 267.1 * max(0.0, Q.lam2 - 0.00136)
        - 2.281 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.009116 * max(0.0, 31.3 - Q.mass)
        + 0.3269 * max(0.0, 0.257 - Q.max_dr)
        + 248.6 * max(0.0, 0.00662 - Q.width)
        + 0.6265 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.07108 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 1.855 * max(0.0, Q.C2 - 0.0596)
        - 8.065 * max(0.0, Q.e2 - 0.0458)
        - 6.477 * max(0.0, 0.0467 - Q.e2)
        - 28.36 * max(0.0, 0.00182 - Q.girth2)
        - 189.4 * max(0.0, Q.lam2 - 0.000227)
        + 2.226 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.04712 * max(0.0, Q.mass - 9.7)
        + 11.09 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 2.507 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        - 56.56 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 1775.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        + 1.803 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        - 29.67 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 2.916 * max(0.0, 0.0355 - Q.C2)
        - 0.5924 * max(0.0, Q.centroid_offset - 0.0144)
        - 8.344 * max(0.0, 0.0498 - Q.centroid_offset)
        - 46.43 * max(0.0, Q.girth - 0.0766)
        - 9.887 * max(0.0, 0.0883 - Q.girth)
        - 0.549 * max(0.0, 0.271 - Q.planar_flow)
        + 0.00181 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 288.1 * max(0.0, 0.00365 - Q.width)
        + 366.8 * max(0.0, 0.0087 - Q.width)
        + 0.04591 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 2.739 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 355.5 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 15.11 * max(0.0, Q.e2 - 0.0622)
        + 2.346 * max(0.0, Q.girth2 - 0.0188)
        - 0.003663 * max(0.0, Q.mass - 91.2)
        - 5467.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        + 0.1311 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 3.941 * max(0.0, Q.C2 - 0.0664)
        - 133.5 * max(0.0, 0.0509 - Q.e2)
        - 2.48 * max(0.0, 0.15 - Q.girth)
        - 0.05781 * max(0.0, 24.8 - Q.pt_7)
        - 0.01993 * max(0.0, Q.sum_pt - 998.0)
        - 10.01 * max(0.0, 0.0159 - Q.width)
        + 2.638 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        - 0.29 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 5.702e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 1.88 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        + 5.364 * max(0.0, 0.0385 - Q.e2)
        - 58.72 * max(0.0, 0.0869 - Q.girth)
        - 44.47 * max(0.0, 0.0131 - Q.girth2)
        - 65.41 * max(0.0, Q.lam1 - 0.00732)
        - 0.05176 * max(0.0, Q.mass - 5.61)
        + 2.592 * max(0.0, 0.177 - Q.max_dr)
        - 59.37 * max(0.0, 0.00741 - Q.width)
        + 16.52 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 154.7 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 24.28 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 1124.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 3689.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        - 1.085 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        + 2.294 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        + 409.4 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 103.8 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        + 817.8 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        + 0.4986 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        - 1.546 * max(0.0, Q.LHA - 0.342)
        - 84.05 * max(0.0, 0.0244 - Q.e2)
        + 0.2503 * max(0.0, 0.041 - Q.e2)
        + 162.9 * max(0.0, 0.00679 - Q.lam1)
        + 181.8 * max(0.0, 0.00813 - Q.lam1)
        - 4.267 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.01566 * max(0.0, 37.4 - Q.mass)
        - 120.9 * max(0.0, 0.00695 - Q.width)
        + 0.01048 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 0.5978 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 0.3148 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 15460.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        - 487.2 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 216.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_q(Q):
    return (3.054
        + 4.967 * max(0.0, 0.0338 - Q.centroid_offset)
        + 28.95 * max(0.0, 0.0775 - Q.girth)
        - 135.3 * max(0.0, 0.0121 - Q.girth2)
        - 0.02163 * max(0.0, 22.5 - Q.mass)
        + 0.02254 * max(0.0, 71.3 - Q.mass)
        + 137.5 * max(0.0, 0.00449 - Q.width)
        - 317.6 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 222.5 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 1.306 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.0001282 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 3.469 * max(0.0, 0.0491 - Q.C2)
        - 188.2 * max(0.0, 0.00775 - Q.e2_sq)
        + 0.3041 * max(0.0, Q.log_sum_pt - 6.41)
        + 0.1879 * max(0.0, Q.pt_7 - 30.4)
        + 864.7 * max(0.0, 0.009 - Q.width)
        - 389.5 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 9.003 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        + 1643.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 4974.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        + 143.8 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        - 980.7 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 0.0002161 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        + 0.2967 * max(0.0, Q.LHA - 0.116)
        + 111.7 * max(0.0, 0.0053 - Q.lam1)
        + 2.882 * max(0.0, Q.log_sum_pt - 6.89)
        - 0.4984 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.1979 * max(0.0, Q.pt_7 - 30.2)
        + 0.0003545 * max(0.0, 54.2 - Q.pt_7)
        - 0.0002339 * max(0.0, 788.0 - Q.sum_pt)
        + 2.925 * max(0.0, Q.z_7 - 0.045)
        - 15260.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        + 0.254 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        - 0.02038 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 15.57 * max(0.0, Q.centroid_offset - 0.0103)
        + 9.81 * max(0.0, Q.e2 - 0.0278)
        + 10.57 * max(0.0, 0.0433 - Q.e2)
        - 914.7 * max(0.0, 0.00905 - Q.girth2)
        + 89.96 * max(0.0, 0.0126 - Q.girth2)
        + 0.001855 * max(0.0, Q.mass - 71.9)
        + 6.624 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 99.51 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 512.2 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 349.5 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        + 0.9674 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        - 13.89 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        + 30.65 * max(0.0, Q.C2 - 0.00878)
        + 452.6 * max(0.0, 0.000389 - Q.lam2)
        + 32.34 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 5.27 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        - 0.002408 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 0.9834 * max(0.0, 0.279 - Q.tau21)
        + 60.73 * max(0.0, Q.width - 0.00219)
        + 0.9064 * max(0.0, 0.0374 - Q.e2)
        + 9.875 * max(0.0, 0.0333 - Q.z_7)
        + 8.131 * max(0.0, 0.0683 - Q.z_7)
        - 41.2 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        + 50.92 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        + 0.3358 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        - 21330.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        - 105.3 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        - 26.65 * max(0.0, Q.C2 - 0.0101)
        + 15.88 * max(0.0, Q.centroid_offset - 0.0212)
        - 17.57 * max(0.0, 0.0508 - Q.e2)
        + 177.8 * max(0.0, 0.00328 - Q.e2_sq)
        + 30.43 * max(0.0, Q.girth - 0.0868)
        - 640.9 * max(0.0, 0.00862 - Q.girth2)
        - 420.0 * max(0.0, 0.00802 - Q.lam1)
        + 23.74 * max(0.0, Q.lam2 - 0.0029)
        - 29.99 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        + 0.704 * max(0.0, Q.max_dr - 0.0279)
        - 40.14 * max(0.0, 0.0133 - Q.width)
        + 0.2215 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 1422.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 0.0439 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 88.79 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 86.27 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.1919 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 57.7 * max(0.0, 0.0201 - Q.centroid_offset)
        + 70.65 * max(0.0, 0.0253 - Q.e2)
        + 15.77 * max(0.0, 0.0372 - Q.e2)
        - 187.6 * max(0.0, 0.00115 - Q.e2_sq)
        + 14.68 * max(0.0, 0.089 - Q.girth)
        - 168.2 * max(0.0, Q.girth2 - 0.00445)
        - 1283.0 * max(0.0, 0.000708 - Q.girth2)
        + 0.004783 * max(0.0, Q.mass - 80.4)
        - 16.46 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 23.64 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 379.0 * max(0.0, 0.00564 - Q.width)
        + 0.8546 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 32.98 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 254.6 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        + 293.1 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        + 0.00172 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 571.5 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        - 0.1773 * max(0.0, Q.log_sum_pt - 6.71)
        + 1476.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        + 0.009671 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        + 106.9 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 1.347 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 16700.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 1092.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 30.19 * max(0.0, 0.0174 - Q.centroid_offset)
        + 463.6 * max(0.0, Q.lam2 - 0.00136)
        - 1.093 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.02296 * max(0.0, 31.3 - Q.mass)
        - 0.06656 * max(0.0, 0.257 - Q.max_dr)
        + 182.5 * max(0.0, 0.00662 - Q.width)
        + 2.261 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.06281 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 2.015 * max(0.0, Q.C2 - 0.0596)
        - 18.72 * max(0.0, Q.e2 - 0.0458)
        + 12.01 * max(0.0, 0.0467 - Q.e2)
        + 154.4 * max(0.0, 0.00182 - Q.girth2)
        - 222.6 * max(0.0, Q.lam2 - 0.000227)
        - 0.459 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.02972 * max(0.0, Q.mass - 9.7)
        - 3.626 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        + 3.124 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        - 91.73 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 1069.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 1.049 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        - 461.3 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 3.097 * max(0.0, 0.0355 - Q.C2)
        + 3.07 * max(0.0, Q.centroid_offset - 0.0144)
        - 4.575 * max(0.0, 0.0498 - Q.centroid_offset)
        - 19.46 * max(0.0, Q.girth - 0.0766)
        - 4.676 * max(0.0, 0.0883 - Q.girth)
        + 0.8067 * max(0.0, 0.271 - Q.planar_flow)
        + 0.0004241 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 51.9 * max(0.0, 0.00365 - Q.width)
        + 222.9 * max(0.0, 0.0087 - Q.width)
        + 0.006858 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 4.075 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 628.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 14.0 * max(0.0, Q.e2 - 0.0622)
        + 31.45 * max(0.0, Q.girth2 - 0.0188)
        - 0.0001613 * max(0.0, Q.mass - 91.2)
        - 6780.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 1.255 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 4.789 * max(0.0, Q.C2 - 0.0664)
        + 28.9 * max(0.0, 0.0509 - Q.e2)
        + 5.639 * max(0.0, 0.15 - Q.girth)
        + 0.01646 * max(0.0, 24.8 - Q.pt_7)
        + 0.005118 * max(0.0, Q.sum_pt - 998.0)
        + 12.21 * max(0.0, 0.0159 - Q.width)
        - 5.872 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        - 0.2085 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 2.008e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 2.864 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 11.35 * max(0.0, 0.0385 - Q.e2)
        - 22.52 * max(0.0, 0.0869 - Q.girth)
        - 80.12 * max(0.0, 0.0131 - Q.girth2)
        + 115.0 * max(0.0, Q.lam1 - 0.00732)
        + 0.01164 * max(0.0, Q.mass - 5.61)
        + 4.541 * max(0.0, 0.177 - Q.max_dr)
        - 201.1 * max(0.0, 0.00741 - Q.width)
        + 20.74 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 753.3 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 102.0 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 1379.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 5346.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 121.9 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        - 139.3 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        - 10.59 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 129.9 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 466.6 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 3.949 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        - 4.125 * max(0.0, Q.LHA - 0.342)
        - 119.4 * max(0.0, 0.0244 - Q.e2)
        + 1.345 * max(0.0, 0.041 - Q.e2)
        + 71.18 * max(0.0, 0.00679 - Q.lam1)
        + 398.1 * max(0.0, 0.00813 - Q.lam1)
        - 8.57 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.01189 * max(0.0, 37.4 - Q.mass)
        - 118.7 * max(0.0, 0.00695 - Q.width)
        + 0.0116 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 1.271 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        + 0.3391 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 14560.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 949.1 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 862.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_W(Q):
    return (-11.13
        - 35.76 * max(0.0, 0.0338 - Q.centroid_offset)
        - 293.9 * max(0.0, 0.0775 - Q.girth)
        + 98.8 * max(0.0, 0.0121 - Q.girth2)
        + 0.05263 * max(0.0, 22.5 - Q.mass)
        - 0.02255 * max(0.0, 71.3 - Q.mass)
        - 1468.0 * max(0.0, 0.00449 - Q.width)
        - 254.9 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 984.3 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        + 1.508 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        - 0.0005041 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 6.643 * max(0.0, 0.0491 - Q.C2)
        + 805.3 * max(0.0, 0.00775 - Q.e2_sq)
        - 3.377 * max(0.0, Q.log_sum_pt - 6.41)
        - 0.1862 * max(0.0, Q.pt_7 - 30.4)
        - 1555.0 * max(0.0, 0.009 - Q.width)
        - 778.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 28.48 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 3244.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        - 25240.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 51.92 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        + 389.1 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        + 3.082e-05 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        + 7.745 * max(0.0, Q.LHA - 0.116)
        + 86.26 * max(0.0, 0.0053 - Q.lam1)
        - 0.6355 * max(0.0, Q.log_sum_pt - 6.89)
        + 2.882 * max(0.0, 6.46 - Q.log_sum_pt)
        + 0.2021 * max(0.0, Q.pt_7 - 30.2)
        + 0.004388 * max(0.0, 54.2 - Q.pt_7)
        + 0.002745 * max(0.0, 788.0 - Q.sum_pt)
        - 6.976 * max(0.0, Q.z_7 - 0.045)
        - 51240.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.4125 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        + 0.1863 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 32.62 * max(0.0, Q.centroid_offset - 0.0103)
        + 6.733 * max(0.0, Q.e2 - 0.0278)
        - 119.3 * max(0.0, 0.0433 - Q.e2)
        + 2258.0 * max(0.0, 0.00905 - Q.girth2)
        + 66.6 * max(0.0, 0.0126 - Q.girth2)
        - 0.05089 * max(0.0, Q.mass - 71.9)
        - 198.0 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 134.3 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        + 2718.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 6198.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        - 6.487 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 38.84 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 91.01 * max(0.0, Q.C2 - 0.00878)
        + 781.0 * max(0.0, 0.000389 - Q.lam2)
        - 138.9 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 21.98 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.0009664 * max(0.0, 486.0 - Q.sum_pt_top5)
        + 2.49 * max(0.0, 0.279 - Q.tau21)
        - 529.9 * max(0.0, Q.width - 0.00219)
        + 110.4 * max(0.0, 0.0374 - Q.e2)
        - 6.987 * max(0.0, 0.0333 - Q.z_7)
        - 16.48 * max(0.0, 0.0683 - Q.z_7)
        + 21.45 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        - 28.4 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 177.6 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 53520.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 388.2 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 74.04 * max(0.0, Q.C2 - 0.0101)
        + 30.1 * max(0.0, Q.centroid_offset - 0.0212)
        - 174.4 * max(0.0, 0.0508 - Q.e2)
        - 866.8 * max(0.0, 0.00328 - Q.e2_sq)
        - 272.8 * max(0.0, Q.girth - 0.0868)
        + 3638.0 * max(0.0, 0.00862 - Q.girth2)
        + 4850.0 * max(0.0, 0.00802 - Q.lam1)
        + 354.9 * max(0.0, Q.lam2 - 0.0029)
        + 48.64 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        - 4.79 * max(0.0, Q.max_dr - 0.0279)
        + 393.3 * max(0.0, 0.0133 - Q.width)
        - 1.073 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        - 11400.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        - 0.1323 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        - 275.5 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        - 219.2 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        + 10.25 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        + 89.88 * max(0.0, 0.0201 - Q.centroid_offset)
        - 326.1 * max(0.0, 0.0253 - Q.e2)
        - 105.0 * max(0.0, 0.0372 - Q.e2)
        + 240.3 * max(0.0, 0.00115 - Q.e2_sq)
        + 68.53 * max(0.0, 0.089 - Q.girth)
        + 1127.0 * max(0.0, Q.girth2 - 0.00445)
        + 830.6 * max(0.0, 0.000708 - Q.girth2)
        + 0.1308 * max(0.0, Q.mass - 80.4)
        + 37.32 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 62.05 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        - 1988.0 * max(0.0, 0.00564 - Q.width)
        - 0.8711 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        + 51.46 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 1286.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        - 2320.0 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        + 0.005485 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 1483.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.4549 * max(0.0, Q.log_sum_pt - 6.71)
        + 21800.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        - 0.1114 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 260.3 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        - 1.623 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        + 46110.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        - 5164.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        - 12.11 * max(0.0, 0.0174 - Q.centroid_offset)
        - 226.0 * max(0.0, Q.lam2 - 0.00136)
        + 1.412 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.012 * max(0.0, 31.3 - Q.mass)
        + 4.798 * max(0.0, 0.257 - Q.max_dr)
        - 1125.0 * max(0.0, 0.00662 - Q.width)
        - 2.242 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        - 0.02261 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 12.71 * max(0.0, Q.C2 - 0.0596)
        + 74.76 * max(0.0, Q.e2 - 0.0458)
        - 53.22 * max(0.0, 0.0467 - Q.e2)
        + 546.1 * max(0.0, 0.00182 - Q.girth2)
        - 249.1 * max(0.0, Q.lam2 - 0.000227)
        - 2.663 * max(0.0, Q.log_sum_pt - 6.69)
        + 0.01283 * max(0.0, Q.mass - 9.7)
        - 54.07 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 5.41 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 32.11 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        + 1359.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 3.658 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        + 908.7 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        - 26.27 * max(0.0, 0.0355 - Q.C2)
        - 39.26 * max(0.0, Q.centroid_offset - 0.0144)
        + 8.245 * max(0.0, 0.0498 - Q.centroid_offset)
        + 190.7 * max(0.0, Q.girth - 0.0766)
        - 36.9 * max(0.0, 0.0883 - Q.girth)
        + 2.21 * max(0.0, 0.271 - Q.planar_flow)
        + 0.001178 * max(0.0, 688.0 - Q.sum_pt_top5)
        + 85.89 * max(0.0, 0.00365 - Q.width)
        - 2850.0 * max(0.0, 0.0087 - Q.width)
        - 0.05137 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 7.791 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 1124.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        - 134.6 * max(0.0, Q.e2 - 0.0622)
        + 109.8 * max(0.0, Q.girth2 - 0.0188)
        - 0.05683 * max(0.0, Q.mass - 91.2)
        - 6403.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 4.835 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        + 3.654 * max(0.0, Q.C2 - 0.0664)
        + 157.3 * max(0.0, 0.0509 - Q.e2)
        + 2.832 * max(0.0, 0.15 - Q.girth)
        - 0.01627 * max(0.0, 24.8 - Q.pt_7)
        + 0.009386 * max(0.0, Q.sum_pt - 998.0)
        + 42.76 * max(0.0, 0.0159 - Q.width)
        - 21.73 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        + 0.24 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 4.559e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        - 5.867 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 57.24 * max(0.0, 0.0385 - Q.e2)
        + 269.5 * max(0.0, 0.0869 - Q.girth)
        - 333.8 * max(0.0, 0.0131 - Q.girth2)
        - 419.3 * max(0.0, Q.lam1 - 0.00732)
        + 0.01218 * max(0.0, Q.mass - 5.61)
        - 3.199 * max(0.0, 0.177 - Q.max_dr)
        + 1920.0 * max(0.0, 0.00741 - Q.width)
        - 45.7 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 1120.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        - 163.6 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        + 8961.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        - 42320.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        - 151.1 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        + 274.9 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        + 2385.0 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        - 453.5 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        + 921.9 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 11.78 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        + 33.49 * max(0.0, Q.LHA - 0.342)
        + 652.9 * max(0.0, 0.0244 - Q.e2)
        - 113.2 * max(0.0, 0.041 - Q.e2)
        - 1277.0 * max(0.0, 0.00679 - Q.lam1)
        - 3571.0 * max(0.0, 0.00813 - Q.lam1)
        - 28.2 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.02811 * max(0.0, 37.4 - Q.mass)
        + 1300.0 * max(0.0, 0.00695 - Q.width)
        - 0.01331 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 6.962 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 26.0 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 90320.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 6393.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 3985.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_Z(Q):
    return (-6.111
        - 44.84 * max(0.0, 0.0338 - Q.centroid_offset)
        + 44.09 * max(0.0, 0.0775 - Q.girth)
        + 395.5 * max(0.0, 0.0121 - Q.girth2)
        + 0.01359 * max(0.0, 22.5 - Q.mass)
        - 0.01387 * max(0.0, 71.3 - Q.mass)
        - 471.1 * max(0.0, 0.00449 - Q.width)
        - 1192.0 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        + 684.1 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        + 0.9707 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.000248 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 1.078 * max(0.0, 0.0491 - Q.C2)
        + 129.8 * max(0.0, 0.00775 - Q.e2_sq)
        - 1.748 * max(0.0, Q.log_sum_pt - 6.41)
        - 0.3135 * max(0.0, Q.pt_7 - 30.4)
        - 1355.0 * max(0.0, 0.009 - Q.width)
        + 1190.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 27.26 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 2872.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 1015.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 144.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        + 196.1 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 0.0002567 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        + 6.413 * max(0.0, Q.LHA - 0.116)
        + 226.4 * max(0.0, 0.0053 - Q.lam1)
        + 1.286 * max(0.0, Q.log_sum_pt - 6.89)
        + 1.889 * max(0.0, 6.46 - Q.log_sum_pt)
        + 0.3356 * max(0.0, Q.pt_7 - 30.2)
        - 0.009249 * max(0.0, 54.2 - Q.pt_7)
        - 0.002555 * max(0.0, 788.0 - Q.sum_pt)
        - 2.417 * max(0.0, Q.z_7 - 0.045)
        - 21040.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.1844 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        + 0.04643 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 34.28 * max(0.0, Q.centroid_offset - 0.0103)
        - 81.67 * max(0.0, Q.e2 - 0.0278)
        - 64.55 * max(0.0, 0.0433 - Q.e2)
        + 1741.0 * max(0.0, 0.00905 - Q.girth2)
        + 76.06 * max(0.0, 0.0126 - Q.girth2)
        + 0.02546 * max(0.0, Q.mass - 71.9)
        - 521.8 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 130.5 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        + 8458.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 4081.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        - 3.508 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 44.68 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 25.29 * max(0.0, Q.C2 - 0.00878)
        - 467.7 * max(0.0, 0.000389 - Q.lam2)
        + 77.79 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 22.93 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.001592 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 0.4776 * max(0.0, 0.279 - Q.tau21)
        + 459.2 * max(0.0, Q.width - 0.00219)
        + 298.1 * max(0.0, 0.0374 - Q.e2)
        - 3.734 * max(0.0, 0.0333 - Q.z_7)
        - 9.754 * max(0.0, 0.0683 - Q.z_7)
        + 5.693 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        + 12.95 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 72.03 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 23330.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 418.3 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 30.49 * max(0.0, Q.C2 - 0.0101)
        - 7.622 * max(0.0, Q.centroid_offset - 0.0212)
        - 520.8 * max(0.0, 0.0508 - Q.e2)
        - 895.6 * max(0.0, 0.00328 - Q.e2_sq)
        - 23.01 * max(0.0, Q.girth - 0.0868)
        + 2501.0 * max(0.0, 0.00862 - Q.girth2)
        + 965.7 * max(0.0, 0.00802 - Q.lam1)
        + 39.36 * max(0.0, Q.lam2 - 0.0029)
        - 1.812 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        - 7.404 * max(0.0, Q.max_dr - 0.0279)
        + 606.3 * max(0.0, 0.0133 - Q.width)
        - 0.8892 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        - 2158.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        - 0.137 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        - 288.8 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        - 227.7 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        + 15.26 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        + 0.722 * max(0.0, 0.0201 - Q.centroid_offset)
        - 61.53 * max(0.0, 0.0253 - Q.e2)
        - 310.4 * max(0.0, 0.0372 - Q.e2)
        + 509.7 * max(0.0, 0.00115 - Q.e2_sq)
        - 73.08 * max(0.0, 0.089 - Q.girth)
        + 233.8 * max(0.0, Q.girth2 - 0.00445)
        - 512.9 * max(0.0, 0.000708 - Q.girth2)
        - 0.1245 * max(0.0, Q.mass - 80.4)
        + 57.07 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        - 142.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        - 979.6 * max(0.0, 0.00564 - Q.width)
        - 1.549 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        + 63.12 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 1514.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        - 103.6 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        + 0.01651 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        - 2619.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        - 1.978 * max(0.0, Q.log_sum_pt - 6.71)
        + 3181.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        - 0.1207 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        + 34.26 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        - 1.86 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        + 19340.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        - 4844.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 10.03 * max(0.0, 0.0174 - Q.centroid_offset)
        - 95.69 * max(0.0, Q.lam2 - 0.00136)
        + 0.879 * max(0.0, Q.log_sum_pt - 6.36)
        + 0.003842 * max(0.0, 31.3 - Q.mass)
        - 0.5846 * max(0.0, 0.257 - Q.max_dr)
        - 306.4 * max(0.0, 0.00662 - Q.width)
        + 0.9512 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        - 0.0002206 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 18.63 * max(0.0, Q.C2 - 0.0596)
        + 152.5 * max(0.0, Q.e2 - 0.0458)
        - 82.98 * max(0.0, 0.0467 - Q.e2)
        - 60.09 * max(0.0, 0.00182 - Q.girth2)
        - 249.8 * max(0.0, Q.lam2 - 0.000227)
        + 4.114 * max(0.0, Q.log_sum_pt - 6.69)
        + 0.0008751 * max(0.0, Q.mass - 9.7)
        - 46.17 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        + 1.574 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 2.115 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        + 637.2 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 1.804 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        - 69.89 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 7.346 * max(0.0, 0.0355 - Q.C2)
        + 2.409 * max(0.0, Q.centroid_offset - 0.0144)
        + 33.67 * max(0.0, 0.0498 - Q.centroid_offset)
        - 25.16 * max(0.0, Q.girth - 0.0766)
        + 70.81 * max(0.0, 0.0883 - Q.girth)
        - 1.154 * max(0.0, 0.271 - Q.planar_flow)
        - 0.0001757 * max(0.0, 688.0 - Q.sum_pt_top5)
        + 661.6 * max(0.0, 0.00365 - Q.width)
        - 1683.0 * max(0.0, 0.0087 - Q.width)
        - 0.002355 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        + 13.34 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        + 1266.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        - 147.1 * max(0.0, Q.e2 - 0.0622)
        + 160.9 * max(0.0, Q.girth2 - 0.0188)
        + 0.1004 * max(0.0, Q.mass - 91.2)
        - 4798.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 3.266 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 13.19 * max(0.0, Q.C2 - 0.0664)
        + 438.4 * max(0.0, 0.0509 - Q.e2)
        + 26.83 * max(0.0, 0.15 - Q.girth)
        - 0.01594 * max(0.0, 24.8 - Q.pt_7)
        + 0.006938 * max(0.0, Q.sum_pt - 998.0)
        + 124.4 * max(0.0, 0.0159 - Q.width)
        - 15.79 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        - 0.05323 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 2.66e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 0.8453 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 3.372 * max(0.0, 0.0385 - Q.e2)
        - 60.56 * max(0.0, 0.0869 - Q.girth)
        - 189.3 * max(0.0, 0.0131 - Q.girth2)
        - 585.6 * max(0.0, Q.lam1 - 0.00732)
        + 0.01092 * max(0.0, Q.mass - 5.61)
        - 5.607 * max(0.0, 0.177 - Q.max_dr)
        + 61.59 * max(0.0, 0.00741 - Q.width)
        + 8.295 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 4219.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 24.92 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        + 789.9 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 34900.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 415.8 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        - 429.9 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        - 536.6 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 166.7 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 1281.0 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 18.66 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        + 27.9 * max(0.0, Q.LHA - 0.342)
        + 281.0 * max(0.0, 0.0244 - Q.e2)
        + 21.98 * max(0.0, 0.041 - Q.e2)
        + 12.32 * max(0.0, 0.00679 - Q.lam1)
        - 1034.0 * max(0.0, 0.00813 - Q.lam1)
        - 15.55 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.01956 * max(0.0, 37.4 - Q.mass)
        - 187.4 * max(0.0, 0.00695 - Q.width)
        - 0.004378 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 2.637 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 7.379 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 57060.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 2679.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 952.4 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_t(Q):
    return (7.988
        + 12.84 * max(0.0, 0.0338 - Q.centroid_offset)
        - 8.679 * max(0.0, 0.0775 - Q.girth)
        + 80.45 * max(0.0, 0.0121 - Q.girth2)
        + 0.01628 * max(0.0, 22.5 - Q.mass)
        - 0.04245 * max(0.0, 71.3 - Q.mass)
        - 31.55 * max(0.0, 0.00449 - Q.width)
        - 348.6 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 89.53 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 0.6322 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.0003208 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 0.1119 * max(0.0, 0.0491 - Q.C2)
        - 475.9 * max(0.0, 0.00775 - Q.e2_sq)
        + 2.994 * max(0.0, Q.log_sum_pt - 6.41)
        - 0.09188 * max(0.0, Q.pt_7 - 30.4)
        - 70.75 * max(0.0, 0.009 - Q.width)
        - 1297.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 21.32 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        + 1843.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 4204.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 7.53 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        - 99.84 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        + 1.797e-05 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        - 13.3 * max(0.0, Q.LHA - 0.116)
        + 54.13 * max(0.0, 0.0053 - Q.lam1)
        - 1.169 * max(0.0, Q.log_sum_pt - 6.89)
        - 0.7269 * max(0.0, 6.46 - Q.log_sum_pt)
        + 0.1021 * max(0.0, Q.pt_7 - 30.2)
        - 0.006539 * max(0.0, 54.2 - Q.pt_7)
        + 0.003679 * max(0.0, 788.0 - Q.sum_pt)
        - 14.75 * max(0.0, Q.z_7 - 0.045)
        + 24230.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        + 0.1122 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        - 0.116 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        + 8.174 * max(0.0, Q.centroid_offset - 0.0103)
        + 9.894 * max(0.0, Q.e2 - 0.0278)
        + 13.84 * max(0.0, 0.0433 - Q.e2)
        + 58.56 * max(0.0, 0.00905 - Q.girth2)
        - 123.8 * max(0.0, 0.0126 - Q.girth2)
        + 0.0305 * max(0.0, Q.mass - 71.9)
        + 26.46 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 116.1 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 1009.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        + 790.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        - 0.4314 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 37.28 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 37.91 * max(0.0, Q.C2 - 0.00878)
        - 675.3 * max(0.0, 0.000389 - Q.lam2)
        + 34.35 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 11.61 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.0002564 * max(0.0, 486.0 - Q.sum_pt_top5)
        + 1.099 * max(0.0, 0.279 - Q.tau21)
        + 31.82 * max(0.0, Q.width - 0.00219)
        - 75.5 * max(0.0, 0.0374 - Q.e2)
        - 56.65 * max(0.0, 0.0333 - Q.z_7)
        - 13.58 * max(0.0, 0.0683 - Q.z_7)
        + 6.583 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        - 130.9 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        + 82.45 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 6050.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 748.0 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 27.3 * max(0.0, Q.C2 - 0.0101)
        + 1.614 * max(0.0, Q.centroid_offset - 0.0212)
        + 654.9 * max(0.0, 0.0508 - Q.e2)
        + 115.7 * max(0.0, 0.00328 - Q.e2_sq)
        + 9.771 * max(0.0, Q.girth - 0.0868)
        - 394.5 * max(0.0, 0.00862 - Q.girth2)
        + 45.18 * max(0.0, 0.00802 - Q.lam1)
        - 16.83 * max(0.0, Q.lam2 - 0.0029)
        - 15.79 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        - 1.277 * max(0.0, Q.max_dr - 0.0279)
        - 94.95 * max(0.0, 0.0133 - Q.width)
        + 0.4224 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 2330.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        - 0.02534 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 32.09 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 49.65 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 1.583 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 10.91 * max(0.0, 0.0201 - Q.centroid_offset)
        + 2.923 * max(0.0, 0.0253 - Q.e2)
        + 66.47 * max(0.0, 0.0372 - Q.e2)
        - 260.9 * max(0.0, 0.00115 - Q.e2_sq)
        + 2.744 * max(0.0, 0.089 - Q.girth)
        + 19.13 * max(0.0, Q.girth2 - 0.00445)
        - 231.0 * max(0.0, 0.000708 - Q.girth2)
        + 0.01922 * max(0.0, Q.mass - 80.4)
        - 17.81 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 2.03 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 114.3 * max(0.0, 0.00564 - Q.width)
        + 0.7438 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 25.96 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 1035.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        + 219.3 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        - 0.0002983 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 179.1 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        - 1.908 * max(0.0, Q.log_sum_pt - 6.71)
        - 32980.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        + 0.352 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        + 79.16 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 0.5195 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 8173.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 2684.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 8.16 * max(0.0, 0.0174 - Q.centroid_offset)
        + 0.8527 * max(0.0, Q.lam2 - 0.00136)
        - 2.298 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.004291 * max(0.0, 31.3 - Q.mass)
        - 3.63 * max(0.0, 0.257 - Q.max_dr)
        - 15.44 * max(0.0, 0.00662 - Q.width)
        - 0.1055 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.02443 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 11.77 * max(0.0, Q.C2 - 0.0596)
        - 13.02 * max(0.0, Q.e2 - 0.0458)
        - 6.747 * max(0.0, 0.0467 - Q.e2)
        - 600.9 * max(0.0, 0.00182 - Q.girth2)
        + 133.1 * max(0.0, Q.lam2 - 0.000227)
        - 1.687 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.02315 * max(0.0, Q.mass - 9.7)
        + 17.76 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 15.38 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 44.31 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 493.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        + 3.235 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        + 1068.0 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 1.722 * max(0.0, 0.0355 - Q.C2)
        - 0.07593 * max(0.0, Q.centroid_offset - 0.0144)
        - 3.851 * max(0.0, 0.0498 - Q.centroid_offset)
        + 14.89 * max(0.0, Q.girth - 0.0766)
        - 15.03 * max(0.0, 0.0883 - Q.girth)
        - 0.299 * max(0.0, 0.271 - Q.planar_flow)
        + 0.0003749 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 109.7 * max(0.0, 0.00365 - Q.width)
        + 299.6 * max(0.0, 0.0087 - Q.width)
        + 0.02438 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        + 2.734 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 324.7 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 63.97 * max(0.0, Q.e2 - 0.0622)
        - 30.97 * max(0.0, Q.girth2 - 0.0188)
        - 0.04814 * max(0.0, Q.mass - 91.2)
        - 10020.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 3.012 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        + 23.67 * max(0.0, Q.C2 - 0.0664)
        - 636.4 * max(0.0, 0.0509 - Q.e2)
        - 26.14 * max(0.0, 0.15 - Q.girth)
        + 0.07701 * max(0.0, 24.8 - Q.pt_7)
        - 0.03046 * max(0.0, Q.sum_pt - 998.0)
        - 75.68 * max(0.0, 0.0159 - Q.width)
        + 13.41 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        + 0.266 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        - 0.0001561 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 3.364 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        + 10.98 * max(0.0, 0.0385 - Q.e2)
        - 3.671 * max(0.0, 0.0869 - Q.girth)
        + 133.2 * max(0.0, 0.0131 - Q.girth2)
        - 105.2 * max(0.0, Q.lam1 - 0.00732)
        + 0.005931 * max(0.0, Q.mass - 5.61)
        + 1.769 * max(0.0, 0.177 - Q.max_dr)
        + 48.18 * max(0.0, 0.00741 - Q.width)
        + 18.72 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 366.2 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        - 39.83 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 710.7 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 735.2 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        - 24.26 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        - 5.465 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        - 93.79 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 69.95 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 339.5 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        + 7.121 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        + 1.536 * max(0.0, Q.LHA - 0.342)
        - 32.83 * max(0.0, 0.0244 - Q.e2)
        - 2.784 * max(0.0, 0.041 - Q.e2)
        + 93.67 * max(0.0, 0.00679 - Q.lam1)
        - 53.32 * max(0.0, 0.00813 - Q.lam1)
        + 46.46 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.00867 * max(0.0, 37.4 - Q.mass)
        - 3.12 * max(0.0, 0.00695 - Q.width)
        - 0.002375 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 0.5799 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        + 1.521 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 15580.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        - 2814.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 338.4 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.girth2 > 0.009295286610722542:
        if s['g'] - s['t'] > -0.08964680135250092:
            if s['g'] - s['q'] > -0.0537016075104475:
                return 'g'   # 77% of the training jets here get this class from the formula
            else:
                return 'q'   # 80% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.2419518604874611:
                return 'Z'   # 74% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.12398064136505127:
                    return 'q'   # 66% of the training jets here get this class from the formula
                else:
                    return 't'   # 98% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.1164613738656044:
            if s['g'] - s['q'] > -0.05220819637179375:
                if s['g'] - s['t'] > 0.03420554660260677:
                    if s['g'] - s['W'] > 0.059497758746147156:
                        if s['g'] - s['q'] > 0.1409664750099182:
                            return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.dr_0 > 0.009837700054049492:
                                if s['g'] - s['q'] > 0.03589777089655399:
                                    return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 3.6280333006288856e-05:
                                    return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 56% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 1.1003970503807068:
                            if s['g'] - s['W'] > -0.4614105373620987:
                                return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 81% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 82% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.11614278331398964:
                        return 'W'   # 88% of the training jets here get this class from the formula
                    else:
                        return 't'   # 84% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.15105580538511276:
                    if s['q'] - s['t'] > 0.040695758536458015:
                        if s['g'] - s['q'] > -0.2828112244606018:
                            if Q.dr_0 > 0.007867147680372:
                                return 'q'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 4.00435492338147e-05:
                                    return 'g'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 74% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.5541094243526459:
                                return 'q'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2399943396449089:
                                    return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                    else:
                        return 't'   # 81% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['Z'] > 1.0270942449569702:
                        if Q.width > 0.002148618921637535:
                            return 'W'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 79% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 86% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > 0.004576867446303368:
                if s['g'] - s['W'] > 0.05527365207672119:
                    if s['g'] - s['W'] > 0.3420807719230652:
                        if Q.mass > 52.60424995422363:
                            return 'W'   # 62% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 95% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 0.6153872907161713:
                            return 'g'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 47% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.2571515887975693:
                        if s['g'] - s['W'] > -0.24283742159605026:
                            return 'W'   # 68% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.09521989524364471:
                                return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                return 't'   # 57% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 68% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.021743876859545708:
                    if s['g'] - s['t'] > 0.02180575579404831:
                        if s['g'] - s['Z'] > 0.3463212698698044:
                            return 'g'   # 93% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > 0.02669314481317997:
                                return 'Z'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 66% of the training jets here get this class from the formula
                    else:
                        return 't'   # 88% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.057620819658041:
                        if s['W'] - s['Z'] > -0.26930001378059387:
                            if Q.z_dr_0p05_0p1 > 0.7761788368225098:
                                if Q.mass > 61.90439987182617:
                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 66% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > -0.124050572514534:
                                    if s['q'] - s['Z'] > -0.6830261647701263:
                                        return 'W'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 78% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.4906582087278366:
                                if s['g'] - s['q'] > 2.1219593286514282:
                                    return 'g'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 97% of the training jets here get this class from the formula
                    else:
                        return 't'   # 80% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
