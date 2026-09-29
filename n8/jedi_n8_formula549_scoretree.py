"""JEDI-linear jet tagger, 8 particles, 3 features: the simplest formula at the network's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 65.35% (the formula: 65.80%); same class as the formula for 91.76% of jets.  44 leaves, depth 9.
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_7=z[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_2=phi[2],
        phi_7=phi[7],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def score_g(Q):
    return (15.53
        + 24.27 * max(0.0, 0.033 - Q.centroid_offset)
        + 14.74 * max(0.0, 0.078 - Q.girth)
        - 49.85 * max(0.0, 0.013 - Q.girth2)
        + 761.9 * max(0.0, 0.0015 - Q.lam1)
        - 0.009618 * max(0.0, 22.0 - Q.mass)
        + 0.01107 * max(0.0, 30.0 - Q.mass)
        + 0.03608 * max(0.0, 59.0 - Q.mass)
        + 0.001756 * max(0.0, Q.sum_pt - 810.0)
        - 0.009455 * max(0.0, Q.sum_pt - 900.0)
        + 174.0 * max(0.0, 0.0044 - Q.width)
        - 495.4 * max(0.0, 0.0089 - Q.width)
        - 156.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        - 1.434 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        - 1.856 * max(0.0, Q.LHA - 0.28)
        - 257.3 * max(0.0, 0.0081 - Q.e2_sq)
        + 3.573 * max(0.0, Q.log_sum_pt - 6.4)
        - 1.507 * max(0.0, 0.25 - Q.max_dr)
        + 0.05306 * max(0.0, Q.pt_7 - 34.0)
        - 50.31 * max(0.0, 0.0087 - Q.width)
        - 14.69 * max(0.0, 0.054 - Q.z_7)
        - 68.57 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        + 18.05 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 9.258 * max(0.0, Q.LHA - 0.1)
        + 9.642 * max(0.0, 0.0057 - Q.lam1)
        + 8.409 * max(0.0, Q.log_sum_pt - 6.3)
        + 8.338 * max(0.0, Q.log_sum_pt - 6.8)
        - 7.998 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.05157 * max(0.0, 36.0 - Q.mass)
        - 0.04196 * max(0.0, 69.0 - Q.mass)
        - 243.1 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 3.42 * max(0.0, 0.16 - Q.max_dr)
        - 0.02605 * max(0.0, Q.pt_7 - 31.0)
        + 0.05109 * max(0.0, 54.0 - Q.pt_7)
        + 0.01274 * max(0.0, 790.0 - Q.sum_pt)
        + 0.5195 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        + 35.24 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        + 0.03937 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        - 2.532 * max(0.0, Q.LHA - 0.32)
        - 42.89 * max(0.0, Q.centroid_offset - 0.013)
        - 7.822 * max(0.0, Q.e2 - 0.028)
        + 5.728 * max(0.0, Q.e2 - 0.051)
        - 20.34 * max(0.0, 0.043 - Q.e2)
        + 459.0 * max(0.0, 0.0088 - Q.girth2)
        - 116.2 * max(0.0, Q.lam1 - 0.016)
        + 0.0374 * max(0.0, Q.mass - 70.0)
        - 0.1277 * max(0.0, Q.max_dr - 0.15)
        + 98.59 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        - 676.3 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 8.775 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.9742 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        - 14.26 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        - 24.14 * max(0.0, Q.C2 - 0.067)
        - 16.86 * max(0.0, Q.e2 - 0.02)
        + 62.71 * max(0.0, Q.e2 - 0.064)
        + 243.9 * max(0.0, 0.003 - Q.e2_sq)
        + 23.64 * max(0.0, 0.017 - Q.e2_sq)
        - 7.31 * max(0.0, 0.13 - Q.girth)
        - 213.6 * max(0.0, Q.girth2 - 0.0034)
        + 109.6 * max(0.0, Q.girth2 - 0.0081)
        - 1096.0 * max(0.0, 0.00033 - Q.lam2)
        + 0.001405 * max(0.0, 44.0 - Q.mass)
        + 32.27 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 52.42 * max(0.0, Q.mass_over_sum_pt - 0.11)
        - 4.757 * max(0.0, Q.max_dr - 0.094)
        + 5.694 * max(0.0, Q.max_dr - 0.2)
        - 0.3253 * max(0.0, 0.24 - Q.tau21)
        + 2114.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        - 5.738 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        - 140.3 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 191.3 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 28.97 * max(0.0, 0.034 - Q.e2)
        - 0.09444 * max(0.0, 53.0 - Q.pt_7)
        + 435.7 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        - 0.007759 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 258.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 523.9 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 3.178 * max(0.0, Q.C2 - 0.033)
        + 17.19 * max(0.0, Q.centroid_offset - 0.019)
        - 3.735 * max(0.0, Q.centroid_offset - 0.05)
        + 1.87 * max(0.0, 0.051 - Q.e2)
        + 20.9 * max(0.0, Q.girth - 0.083)
        + 40.65 * max(0.0, 0.0036 - Q.girth2)
        - 92.69 * max(0.0, 0.0086 - Q.girth2)
        - 6.574 * max(0.0, Q.girth2_top5 - 0.011)
        + 26.12 * max(0.0, 0.0078 - Q.lam1)
        + 618.3 * max(0.0, Q.lam2 - 0.0034)
        + 1.435 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.0116 * max(0.0, 50.0 - Q.mass)
        - 94.96 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        + 3.727 * max(0.0, 0.11 - Q.max_dr)
        + 0.01516 * max(0.0, Q.sum_pt - 990.0)
        - 49.85 * max(0.0, 0.013 - Q.width)
        + 1360.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 0.3095 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        + 0.1121 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 0.004879 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        - 6.098 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        - 12.39 * max(0.0, 0.021 - Q.centroid_offset)
        + 0.6542 * max(0.0, 0.088 - Q.girth)
        - 333.3 * max(0.0, Q.girth2 - 0.0015)
        + 39.89 * max(0.0, Q.girth2 - 0.0044)
        + 96.31 * max(0.0, Q.girth2 - 0.0075)
        - 207.4 * max(0.0, Q.girth2 - 0.0087)
        + 23.17 * max(0.0, Q.girth2 - 0.015)
        + 8.315 * max(0.0, 0.0011 - Q.girth2_top2)
        - 289.2 * max(0.0, 0.0084 - Q.lam1)
        + 0.00459 * max(0.0, Q.mass - 80.4)
        - 22.53 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 12.39 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 64.79 * max(0.0, Q.mass_over_sum_pt - 0.091)
        + 202.9 * max(0.0, 0.0055 - Q.width)
        - 0.1508 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 77.26 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 48.72 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        - 129.0 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 7.405 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        + 0.1225 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 2.028 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 26.35 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 238.4 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        + 0.589 * max(0.0, Q.log_sum_pt - 6.7)
        + 311.5 * max(0.0, 0.0049 - Q.width)
        + 10770.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 216.3 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 9276.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 8699.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        - 2.569 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 25320.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 5.153 * max(0.0, Q.C2 - 0.051)
        - 4.868 * max(0.0, 0.018 - Q.centroid_offset)
        - 2.589 * max(0.0, 0.055 - Q.girth)
        - 19.01 * max(0.0, Q.girth2 - 0.018)
        - 18.51 * max(0.0, Q.lam2 - 0.0017)
        - 45.77 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 137.6 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 1.761 * max(0.0, 0.23 - Q.max_dr)
        + 58.18 * max(0.0, 0.0061 - Q.width)
        + 160.3 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 581.6 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 1.155 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        + 0.4561 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.07313 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.01208 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 517.6 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 4906.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 2.356 * max(0.0, Q.C2 - 0.055)
        - 0.3816 * max(0.0, Q.LHA - 0.33)
        + 22.43 * max(0.0, Q.centroid_offset - 0.038)
        + 8.242 * max(0.0, Q.e2 - 0.036)
        - 0.7895 * max(0.0, 0.037 - Q.e2)
        - 117.0 * max(0.0, Q.girth2 - 0.0085)
        - 367.5 * max(0.0, 0.0017 - Q.girth2)
        + 264.6 * max(0.0, Q.lam1 - 0.0085)
        - 166.8 * max(0.0, 0.0044 - Q.lam1)
        - 606.9 * max(0.0, 0.0034 - Q.lam2)
        - 7.76 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.03738 * max(0.0, Q.mass - 17.0)
        - 1.072 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 677.9 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 2.706 * max(0.0, 0.035 - Q.C2)
        + 31.1 * max(0.0, Q.centroid_offset - 0.014)
        + 17.98 * max(0.0, 0.04 - Q.centroid_offset)
        + 76.9 * max(0.0, 0.0062 - Q.e2_sq)
        + 9.984 * max(0.0, Q.girth - 0.076)
        + 1.243 * max(0.0, 0.089 - Q.girth)
        - 1.507 * max(0.0, 0.22 - Q.max_dr)
        - 0.002513 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 40.65 * max(0.0, 0.0036 - Q.width)
        + 38.97 * max(0.0, 0.0085 - Q.width)
        - 32.37 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 2.341 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 2.507 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 16.4 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        - 78.3 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        - 55.71 * max(0.0, Q.e2 - 0.063)
        + 50.13 * max(0.0, Q.girth2 - 0.019)
        + 0.005341 * max(0.0, Q.mass - 91.2)
        - 928.2 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        + 36.24 * max(0.0, Q.C2 - 0.066)
        - 24.52 * max(0.0, 0.038 - Q.centroid_offset)
        + 4.031 * max(0.0, 0.049 - Q.e2)
        - 0.5183 * max(0.0, 0.15 - Q.girth)
        - 114.1 * max(0.0, 0.0062 - Q.lam1)
        + 89.51 * max(0.0, 0.015 - Q.lam1)
        - 0.01167 * max(0.0, Q.sum_pt - 1000.0)
        - 228.6 * max(0.0, 0.0076 - Q.width)
        + 15.19 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.1842 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        - 42.92 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        - 4088.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        - 3.113e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 6.335 * max(0.0, 0.067 - Q.C2)
        - 20.78 * max(0.0, Q.centroid_offset - 0.031)
        + 226.6 * max(0.0, 0.0058 - Q.e2_sq)
        + 17.79 * max(0.0, 0.034 - Q.girth)
        - 13.83 * max(0.0, 0.087 - Q.girth)
        - 74.9 * max(0.0, Q.lam1 - 0.0025)
        + 297.4 * max(0.0, Q.lam1 - 0.0042)
        + 150.6 * max(0.0, Q.lam1 - 0.0061)
        - 234.9 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        - 7.179 * max(0.0, 0.078 - Q.max_dr)
        + 3.005 * max(0.0, 0.18 - Q.max_dr)
        + 7.476 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        - 1068.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 8.785 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 221.6 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 30.44 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        - 25.72 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 3416.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 359.3 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 25.46 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 1.055 * max(0.0, Q.LHA - 0.34)
        - 5.791 * max(0.0, 0.024 - Q.e2)
        + 12.84 * max(0.0, 0.041 - Q.e2)
        + 5.692 * max(0.0, Q.girth - 0.032)
        - 7.894 * max(0.0, 0.0067 - Q.lam1)
        + 92.66 * max(0.0, 0.0083 - Q.lam1)
        + 1336.0 * max(0.0, 0.00031 - Q.lam2)
        - 2.108 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 103.8 * max(0.0, 0.0067 - Q.width)
        + 3.267 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 4126.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 1048.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_q(Q):
    return (0.6109
        + 39.1 * max(0.0, 0.033 - Q.centroid_offset)
        + 17.95 * max(0.0, 0.078 - Q.girth)
        - 41.23 * max(0.0, 0.013 - Q.girth2)
        + 184.8 * max(0.0, 0.0015 - Q.lam1)
        + 0.02668 * max(0.0, 22.0 - Q.mass)
        + 0.03549 * max(0.0, 30.0 - Q.mass)
        + 0.0095 * max(0.0, 59.0 - Q.mass)
        - 0.03511 * max(0.0, Q.sum_pt - 810.0)
        + 0.00818 * max(0.0, Q.sum_pt - 900.0)
        + 201.1 * max(0.0, 0.0044 - Q.width)
        - 587.5 * max(0.0, 0.0089 - Q.width)
        - 312.8 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        - 1.059 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        - 2.199 * max(0.0, Q.LHA - 0.28)
        + 587.5 * max(0.0, 0.0081 - Q.e2_sq)
        - 0.492 * max(0.0, Q.log_sum_pt - 6.4)
        - 2.373 * max(0.0, 0.25 - Q.max_dr)
        - 0.008169 * max(0.0, Q.pt_7 - 34.0)
        - 35.5 * max(0.0, 0.0087 - Q.width)
        + 15.8 * max(0.0, 0.054 - Q.z_7)
        - 15.83 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        + 174.6 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        + 5.646 * max(0.0, Q.LHA - 0.1)
        + 13.89 * max(0.0, 0.0057 - Q.lam1)
        + 2.421 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.02605 * max(0.0, Q.log_sum_pt - 6.8)
        - 32.37 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.01071 * max(0.0, 36.0 - Q.mass)
        + 0.01879 * max(0.0, 69.0 - Q.mass)
        - 32.59 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 4.695 * max(0.0, 0.16 - Q.max_dr)
        + 0.017 * max(0.0, Q.pt_7 - 31.0)
        + 0.1006 * max(0.0, 54.0 - Q.pt_7)
        + 0.005092 * max(0.0, 790.0 - Q.sum_pt)
        + 0.2391 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        + 4.672 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 0.1137 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 0.7984 * max(0.0, Q.LHA - 0.32)
        + 17.33 * max(0.0, Q.centroid_offset - 0.013)
        + 9.221 * max(0.0, Q.e2 - 0.028)
        - 5.757 * max(0.0, Q.e2 - 0.051)
        - 11.41 * max(0.0, 0.043 - Q.e2)
        + 737.1 * max(0.0, 0.0088 - Q.girth2)
        - 158.1 * max(0.0, Q.lam1 - 0.016)
        - 0.003254 * max(0.0, Q.mass - 70.0)
        - 0.7206 * max(0.0, Q.max_dr - 0.15)
        + 49.37 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        - 88.11 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 6.705 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.8976 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        - 6.955 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        - 40.0 * max(0.0, Q.C2 - 0.067)
        - 12.14 * max(0.0, Q.e2 - 0.02)
        + 70.44 * max(0.0, Q.e2 - 0.064)
        + 76.42 * max(0.0, 0.003 - Q.e2_sq)
        + 39.02 * max(0.0, 0.017 - Q.e2_sq)
        + 0.4506 * max(0.0, 0.13 - Q.girth)
        + 49.16 * max(0.0, Q.girth2 - 0.0034)
        + 167.7 * max(0.0, Q.girth2 - 0.0081)
        - 305.4 * max(0.0, 0.00033 - Q.lam2)
        + 0.002664 * max(0.0, 44.0 - Q.mass)
        + 20.25 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 1.525 * max(0.0, Q.mass_over_sum_pt - 0.11)
        - 5.59 * max(0.0, Q.max_dr - 0.094)
        + 5.673 * max(0.0, Q.max_dr - 0.2)
        - 1.628 * max(0.0, 0.24 - Q.tau21)
        + 1125.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        + 64.89 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 138.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 452.8 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 4.008 * max(0.0, 0.034 - Q.e2)
        - 0.09579 * max(0.0, 53.0 - Q.pt_7)
        - 0.9199 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        - 0.004511 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 16730.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        - 86.31 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 17.82 * max(0.0, Q.C2 - 0.033)
        + 19.19 * max(0.0, Q.centroid_offset - 0.019)
        - 5.134 * max(0.0, Q.centroid_offset - 0.05)
        + 24.68 * max(0.0, 0.051 - Q.e2)
        - 0.2792 * max(0.0, Q.girth - 0.083)
        + 82.05 * max(0.0, 0.0036 - Q.girth2)
        - 193.4 * max(0.0, 0.0086 - Q.girth2)
        - 14.36 * max(0.0, Q.girth2_top5 - 0.011)
        - 18.54 * max(0.0, 0.0078 - Q.lam1)
        + 168.3 * max(0.0, Q.lam2 - 0.0034)
        - 2.479 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.009335 * max(0.0, 50.0 - Q.mass)
        - 5.552 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        + 0.1061 * max(0.0, 0.11 - Q.max_dr)
        + 0.02411 * max(0.0, Q.sum_pt - 990.0)
        - 41.23 * max(0.0, 0.013 - Q.width)
        + 3068.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        + 0.03236 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        + 0.05532 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 0.007082 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        - 3.963 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        - 30.5 * max(0.0, 0.021 - Q.centroid_offset)
        - 25.52 * max(0.0, 0.088 - Q.girth)
        - 218.9 * max(0.0, Q.girth2 - 0.0015)
        - 45.79 * max(0.0, Q.girth2 - 0.0044)
        + 238.2 * max(0.0, Q.girth2 - 0.0075)
        - 276.9 * max(0.0, Q.girth2 - 0.0087)
        + 15.41 * max(0.0, Q.girth2 - 0.015)
        - 271.3 * max(0.0, 0.0011 - Q.girth2_top2)
        - 254.3 * max(0.0, 0.0084 - Q.lam1)
        + 0.003395 * max(0.0, Q.mass - 80.4)
        - 3.132 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 12.67 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 33.48 * max(0.0, Q.mass_over_sum_pt - 0.091)
        + 282.7 * max(0.0, 0.0055 - Q.width)
        - 0.1926 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 74.29 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        + 212.3 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        - 160.4 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 5.602 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        - 3.256 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 1.295 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 23.92 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 17.34 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        + 25.89 * max(0.0, Q.log_sum_pt - 6.7)
        + 176.8 * max(0.0, 0.0049 - Q.width)
        + 7153.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        + 37.68 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 13860.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 7084.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        - 3.852 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 10250.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 3.345 * max(0.0, Q.C2 - 0.051)
        - 5.127 * max(0.0, 0.018 - Q.centroid_offset)
        - 11.06 * max(0.0, 0.055 - Q.girth)
        - 45.86 * max(0.0, Q.girth2 - 0.018)
        + 57.29 * max(0.0, Q.lam2 - 0.0017)
        - 70.45 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 236.4 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        + 2.325 * max(0.0, 0.23 - Q.max_dr)
        + 452.8 * max(0.0, 0.0061 - Q.width)
        + 4485.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        - 8546.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 0.6481 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        + 1.273 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.0612 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.006335 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 14460.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        - 4614.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        - 3.176 * max(0.0, Q.C2 - 0.055)
        + 0.4192 * max(0.0, Q.LHA - 0.33)
        + 18.94 * max(0.0, Q.centroid_offset - 0.038)
        - 2.539 * max(0.0, Q.e2 - 0.036)
        + 2.883 * max(0.0, 0.037 - Q.e2)
        - 151.1 * max(0.0, Q.girth2 - 0.0085)
        + 153.3 * max(0.0, 0.0017 - Q.girth2)
        + 322.1 * max(0.0, Q.lam1 - 0.0085)
        - 71.04 * max(0.0, 0.0044 - Q.lam1)
        - 88.56 * max(0.0, 0.0034 - Q.lam2)
        - 0.7977 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.007369 * max(0.0, Q.mass - 17.0)
        - 0.1584 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        - 1274.0 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        + 9.732 * max(0.0, 0.035 - Q.C2)
        - 9.042 * max(0.0, Q.centroid_offset - 0.014)
        + 15.35 * max(0.0, 0.04 - Q.centroid_offset)
        + 35.57 * max(0.0, 0.0062 - Q.e2_sq)
        - 2.587 * max(0.0, Q.girth - 0.076)
        + 12.21 * max(0.0, 0.089 - Q.girth)
        - 4.445 * max(0.0, 0.22 - Q.max_dr)
        - 0.002485 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 82.05 * max(0.0, 0.0036 - Q.width)
        + 89.92 * max(0.0, 0.0085 - Q.width)
        - 37.71 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.461 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.4962 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 85.38 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        - 68.41 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        - 57.84 * max(0.0, Q.e2 - 0.063)
        + 57.15 * max(0.0, Q.girth2 - 0.019)
        + 0.001772 * max(0.0, Q.mass - 91.2)
        - 1147.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        + 64.72 * max(0.0, Q.C2 - 0.066)
        - 1.782 * max(0.0, 0.038 - Q.centroid_offset)
        - 9.9 * max(0.0, 0.049 - Q.e2)
        + 4.326 * max(0.0, 0.15 - Q.girth)
        - 54.33 * max(0.0, 0.0062 - Q.lam1)
        + 145.6 * max(0.0, 0.015 - Q.lam1)
        + 0.01284 * max(0.0, Q.sum_pt - 1000.0)
        - 384.8 * max(0.0, 0.0076 - Q.width)
        - 0.4227 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.09325 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        - 807.3 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 20180.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        + 2.83e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 24.9 * max(0.0, 0.067 - Q.C2)
        - 40.2 * max(0.0, Q.centroid_offset - 0.031)
        + 409.2 * max(0.0, 0.0058 - Q.e2_sq)
        + 10.5 * max(0.0, 0.034 - Q.girth)
        + 16.63 * max(0.0, 0.087 - Q.girth)
        - 117.8 * max(0.0, Q.lam1 - 0.0025)
        + 123.2 * max(0.0, Q.lam1 - 0.0042)
        + 9.871 * max(0.0, Q.lam1 - 0.0061)
        - 805.9 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 0.09637 * max(0.0, 0.078 - Q.max_dr)
        + 3.038 * max(0.0, 0.18 - Q.max_dr)
        - 2.584 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        + 65.28 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 8.436 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 36.74 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 63.97 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        - 50.4 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 2928.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 437.0 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 28.33 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 2.395 * max(0.0, Q.LHA - 0.34)
        - 10.71 * max(0.0, 0.024 - Q.e2)
        + 13.41 * max(0.0, 0.041 - Q.e2)
        + 3.0 * max(0.0, Q.girth - 0.032)
        + 20.87 * max(0.0, 0.0067 - Q.lam1)
        + 28.98 * max(0.0, 0.0083 - Q.lam1)
        + 647.1 * max(0.0, 0.00031 - Q.lam2)
        + 2.335 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 145.2 * max(0.0, 0.0067 - Q.width)
        + 4.789 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 9227.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 272.6 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_W(Q):
    return (-15.19
        - 72.34 * max(0.0, 0.033 - Q.centroid_offset)
        - 40.55 * max(0.0, 0.078 - Q.girth)
        + 40.88 * max(0.0, 0.013 - Q.girth2)
        + 662.9 * max(0.0, 0.0015 - Q.lam1)
        + 0.03032 * max(0.0, 22.0 - Q.mass)
        - 0.04295 * max(0.0, 30.0 - Q.mass)
        - 0.02309 * max(0.0, 59.0 - Q.mass)
        + 0.05173 * max(0.0, Q.sum_pt - 810.0)
        + 0.01157 * max(0.0, Q.sum_pt - 900.0)
        - 1667.0 * max(0.0, 0.0044 - Q.width)
        - 761.8 * max(0.0, 0.0089 - Q.width)
        - 1142.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        + 3.217 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        + 16.45 * max(0.0, Q.LHA - 0.28)
        + 2884.0 * max(0.0, 0.0081 - Q.e2_sq)
        - 0.764 * max(0.0, Q.log_sum_pt - 6.4)
        + 3.951 * max(0.0, 0.25 - Q.max_dr)
        + 0.02411 * max(0.0, Q.pt_7 - 34.0)
        - 1683.0 * max(0.0, 0.0087 - Q.width)
        - 36.79 * max(0.0, 0.054 - Q.z_7)
        + 142.9 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 39.94 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        + 9.216 * max(0.0, Q.LHA - 0.1)
        + 101.4 * max(0.0, 0.0057 - Q.lam1)
        - 11.45 * max(0.0, Q.log_sum_pt - 6.3)
        - 13.81 * max(0.0, Q.log_sum_pt - 6.8)
        + 44.64 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.02631 * max(0.0, 36.0 - Q.mass)
        + 0.03348 * max(0.0, 69.0 - Q.mass)
        - 866.5 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        - 2.199 * max(0.0, 0.16 - Q.max_dr)
        - 0.04195 * max(0.0, Q.pt_7 - 31.0)
        - 0.1807 * max(0.0, 54.0 - Q.pt_7)
        - 0.005876 * max(0.0, 790.0 - Q.sum_pt)
        - 2.057 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        - 27.58 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 0.2892 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        - 31.61 * max(0.0, Q.LHA - 0.32)
        - 154.9 * max(0.0, Q.centroid_offset - 0.013)
        - 29.89 * max(0.0, Q.e2 - 0.028)
        - 37.34 * max(0.0, Q.e2 - 0.051)
        - 17.29 * max(0.0, 0.043 - Q.e2)
        + 1130.0 * max(0.0, 0.0088 - Q.girth2)
        - 414.5 * max(0.0, Q.lam1 - 0.016)
        - 0.1185 * max(0.0, Q.mass - 70.0)
        - 8.016 * max(0.0, Q.max_dr - 0.15)
        - 270.1 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        - 1384.0 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        + 9.638 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        - 11.9 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        + 7.654 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 193.3 * max(0.0, Q.C2 - 0.067)
        + 76.31 * max(0.0, Q.e2 - 0.02)
        - 426.0 * max(0.0, Q.e2 - 0.064)
        - 1046.0 * max(0.0, 0.003 - Q.e2_sq)
        - 43.76 * max(0.0, 0.017 - Q.e2_sq)
        + 27.37 * max(0.0, 0.13 - Q.girth)
        + 328.9 * max(0.0, Q.girth2 - 0.0034)
        - 395.8 * max(0.0, Q.girth2 - 0.0081)
        + 5565.0 * max(0.0, 0.00033 - Q.lam2)
        - 0.04267 * max(0.0, 44.0 - Q.mass)
        - 123.8 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 255.1 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 6.605 * max(0.0, Q.max_dr - 0.094)
        - 8.484 * max(0.0, Q.max_dr - 0.2)
        - 4.42 * max(0.0, 0.24 - Q.tau21)
        - 10180.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        + 867.7 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 3723.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        - 1476.0 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 59.08 * max(0.0, 0.034 - Q.e2)
        + 0.1924 * max(0.0, 53.0 - Q.pt_7)
        - 297.7 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        + 0.007222 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 26400.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 378.1 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 14.96 * max(0.0, Q.C2 - 0.033)
        - 99.42 * max(0.0, Q.centroid_offset - 0.019)
        + 45.22 * max(0.0, Q.centroid_offset - 0.05)
        - 39.86 * max(0.0, 0.051 - Q.e2)
        - 66.29 * max(0.0, Q.girth - 0.083)
        - 274.8 * max(0.0, 0.0036 - Q.girth2)
        + 1130.0 * max(0.0, 0.0086 - Q.girth2)
        - 1.082 * max(0.0, Q.girth2_top5 - 0.011)
        - 252.0 * max(0.0, 0.0078 - Q.lam1)
        - 1121.0 * max(0.0, Q.lam2 - 0.0034)
        - 2.228 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.008303 * max(0.0, 50.0 - Q.mass)
        + 25.37 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 15.89 * max(0.0, 0.11 - Q.max_dr)
        - 0.05468 * max(0.0, Q.sum_pt - 990.0)
        + 40.88 * max(0.0, 0.013 - Q.width)
        - 7405.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 0.9066 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 0.1552 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 0.006962 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        + 20.16 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        + 92.92 * max(0.0, 0.021 - Q.centroid_offset)
        + 23.09 * max(0.0, 0.088 - Q.girth)
        - 242.8 * max(0.0, Q.girth2 - 0.0015)
        + 1180.0 * max(0.0, Q.girth2 - 0.0044)
        - 901.1 * max(0.0, Q.girth2 - 0.0075)
        + 1057.0 * max(0.0, Q.girth2 - 0.0087)
        - 357.2 * max(0.0, Q.girth2 - 0.015)
        + 803.7 * max(0.0, 0.0011 - Q.girth2_top2)
        + 772.2 * max(0.0, 0.0084 - Q.lam1)
        + 0.1115 * max(0.0, Q.mass - 80.4)
        + 51.1 * max(0.0, Q.mass_over_sum_pt - 0.072)
        + 31.78 * max(0.0, Q.mass_over_sum_pt - 0.085)
        - 290.4 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 1840.0 * max(0.0, 0.0055 - Q.width)
        + 0.1923 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        + 659.8 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 1474.0 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        - 207.0 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 23.12 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        - 0.7006 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 14.86 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        + 238.6 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 45.69 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        - 38.97 * max(0.0, Q.log_sum_pt - 6.7)
        - 956.5 * max(0.0, 0.0049 - Q.width)
        - 16430.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 67.59 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        + 3108.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        - 31480.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 4.254 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        + 11050.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 23.76 * max(0.0, Q.C2 - 0.051)
        + 65.32 * max(0.0, 0.018 - Q.centroid_offset)
        + 31.44 * max(0.0, 0.055 - Q.girth)
        - 465.8 * max(0.0, Q.girth2 - 0.018)
        - 11.22 * max(0.0, Q.lam2 - 0.0017)
        + 37.79 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 166.5 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 2.999 * max(0.0, 0.23 - Q.max_dr)
        - 57.22 * max(0.0, 0.0061 - Q.width)
        - 3450.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 12180.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        + 3.637 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        - 3.796 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        - 0.06036 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.03263 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 17520.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 14340.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 21.5 * max(0.0, Q.C2 - 0.055)
        - 17.62 * max(0.0, Q.LHA - 0.33)
        - 30.58 * max(0.0, Q.centroid_offset - 0.038)
        - 36.34 * max(0.0, Q.e2 - 0.036)
        + 55.78 * max(0.0, 0.037 - Q.e2)
        + 1542.0 * max(0.0, Q.girth2 - 0.0085)
        + 280.3 * max(0.0, 0.0017 - Q.girth2)
        - 640.3 * max(0.0, Q.lam1 - 0.0085)
        + 625.2 * max(0.0, 0.0044 - Q.lam1)
        + 1346.0 * max(0.0, 0.0034 - Q.lam2)
        + 9.656 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.03102 * max(0.0, Q.mass - 17.0)
        - 41.0 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 1066.0 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 1.209 * max(0.0, 0.035 - Q.C2)
        + 96.45 * max(0.0, Q.centroid_offset - 0.014)
        - 18.33 * max(0.0, 0.04 - Q.centroid_offset)
        - 782.1 * max(0.0, 0.0062 - Q.e2_sq)
        - 95.43 * max(0.0, Q.girth - 0.076)
        - 69.65 * max(0.0, 0.089 - Q.girth)
        + 11.35 * max(0.0, 0.22 - Q.max_dr)
        + 0.003196 * max(0.0, 690.0 - Q.sum_pt_top5)
        - 274.8 * max(0.0, 0.0036 - Q.width)
        - 1196.0 * max(0.0, 0.0085 - Q.width)
        + 10.71 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 6.074 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 5.514 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 660.1 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        - 51.84 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        + 406.9 * max(0.0, Q.e2 - 0.063)
        + 180.0 * max(0.0, Q.girth2 - 0.019)
        - 0.06534 * max(0.0, Q.mass - 91.2)
        - 10580.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        - 156.6 * max(0.0, Q.C2 - 0.066)
        - 92.3 * max(0.0, 0.038 - Q.centroid_offset)
        - 27.87 * max(0.0, 0.049 - Q.e2)
        + 9.422 * max(0.0, 0.15 - Q.girth)
        - 74.06 * max(0.0, 0.0062 - Q.lam1)
        + 658.2 * max(0.0, 0.015 - Q.lam1)
        - 0.01147 * max(0.0, Q.sum_pt - 1000.0)
        + 2931.0 * max(0.0, 0.0076 - Q.width)
        - 39.4 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.01512 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        + 8011.0 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 790.4 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        + 7.984e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 18.64 * max(0.0, 0.067 - Q.C2)
        + 99.55 * max(0.0, Q.centroid_offset - 0.031)
        - 488.2 * max(0.0, 0.0058 - Q.e2_sq)
        - 51.23 * max(0.0, 0.034 - Q.girth)
        + 129.4 * max(0.0, 0.087 - Q.girth)
        + 466.0 * max(0.0, Q.lam1 - 0.0025)
        - 860.8 * max(0.0, Q.lam1 - 0.0042)
        + 11.5 * max(0.0, Q.lam1 - 0.0061)
        + 67.79 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 18.75 * max(0.0, 0.078 - Q.max_dr)
        - 2.551 * max(0.0, 0.18 - Q.max_dr)
        - 26.45 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        + 1927.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 53.5 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 25.09 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        - 375.5 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        + 297.5 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        + 8998.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 954.2 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 216.5 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 91.88 * max(0.0, Q.LHA - 0.34)
        + 259.0 * max(0.0, 0.024 - Q.e2)
        - 172.3 * max(0.0, 0.041 - Q.e2)
        + 48.39 * max(0.0, Q.girth - 0.032)
        - 900.2 * max(0.0, 0.0067 - Q.lam1)
        + 269.5 * max(0.0, 0.0083 - Q.lam1)
        - 4106.0 * max(0.0, 0.00031 - Q.lam2)
        - 0.885 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        + 1307.0 * max(0.0, 0.0067 - Q.width)
        - 23.77 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 93150.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        + 4077.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_Z(Q):
    return (-7.884
        + 17.4 * max(0.0, 0.033 - Q.centroid_offset)
        + 54.54 * max(0.0, 0.078 - Q.girth)
        + 809.6 * max(0.0, 0.013 - Q.girth2)
        - 116.0 * max(0.0, 0.0015 - Q.lam1)
        - 0.008464 * max(0.0, 22.0 - Q.mass)
        + 0.0457 * max(0.0, 30.0 - Q.mass)
        - 0.01081 * max(0.0, 59.0 - Q.mass)
        + 0.09652 * max(0.0, Q.sum_pt - 810.0)
        - 0.001384 * max(0.0, Q.sum_pt - 900.0)
        - 1698.0 * max(0.0, 0.0044 - Q.width)
        + 5271.0 * max(0.0, 0.0089 - Q.width)
        + 1024.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        + 1.439 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        - 6.191 * max(0.0, Q.LHA - 0.28)
        - 748.3 * max(0.0, 0.0081 - Q.e2_sq)
        + 1.397 * max(0.0, Q.log_sum_pt - 6.4)
        + 0.8625 * max(0.0, 0.25 - Q.max_dr)
        + 0.007464 * max(0.0, Q.pt_7 - 34.0)
        - 774.8 * max(0.0, 0.0087 - Q.width)
        - 42.9 * max(0.0, 0.054 - Q.z_7)
        + 108.4 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 199.5 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 7.694 * max(0.0, Q.LHA - 0.1)
        + 212.0 * max(0.0, 0.0057 - Q.lam1)
        - 10.81 * max(0.0, Q.log_sum_pt - 6.3)
        - 10.62 * max(0.0, Q.log_sum_pt - 6.8)
        + 84.87 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.03858 * max(0.0, 36.0 - Q.mass)
        - 0.1109 * max(0.0, 69.0 - Q.mass)
        - 565.2 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        - 18.31 * max(0.0, 0.16 - Q.max_dr)
        - 0.01902 * max(0.0, Q.pt_7 - 31.0)
        - 0.1506 * max(0.0, 54.0 - Q.pt_7)
        - 0.009837 * max(0.0, 790.0 - Q.sum_pt)
        - 0.2259 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        - 12.87 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        + 0.04668 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 25.09 * max(0.0, Q.LHA - 0.32)
        - 87.63 * max(0.0, Q.centroid_offset - 0.013)
        - 112.9 * max(0.0, Q.e2 - 0.028)
        + 64.35 * max(0.0, Q.e2 - 0.051)
        + 57.95 * max(0.0, 0.043 - Q.e2)
        - 5530.0 * max(0.0, 0.0088 - Q.girth2)
        - 399.6 * max(0.0, Q.lam1 - 0.016)
        + 0.09375 * max(0.0, Q.mass - 70.0)
        - 3.332 * max(0.0, Q.max_dr - 0.15)
        - 400.4 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 680.1 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 0.937 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        - 0.7673 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        + 55.07 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 253.6 * max(0.0, Q.C2 - 0.067)
        + 27.36 * max(0.0, Q.e2 - 0.02)
        - 626.0 * max(0.0, Q.e2 - 0.064)
        - 649.7 * max(0.0, 0.003 - Q.e2_sq)
        + 21.26 * max(0.0, 0.017 - Q.e2_sq)
        + 45.87 * max(0.0, 0.13 - Q.girth)
        + 936.8 * max(0.0, Q.girth2 - 0.0034)
        - 565.0 * max(0.0, Q.girth2 - 0.0081)
        - 2902.0 * max(0.0, 0.00033 - Q.lam2)
        - 0.0212 * max(0.0, 44.0 - Q.mass)
        - 40.28 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 107.2 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 13.69 * max(0.0, Q.max_dr - 0.094)
        - 16.09 * max(0.0, Q.max_dr - 0.2)
        - 2.209 * max(0.0, 0.24 - Q.tau21)
        - 1670.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        + 853.4 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 2474.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        - 2978.0 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 92.78 * max(0.0, 0.034 - Q.e2)
        + 0.1389 * max(0.0, 53.0 - Q.pt_7)
        - 219.1 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        + 0.01063 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 7661.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 238.7 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        + 33.81 * max(0.0, Q.C2 - 0.033)
        - 33.49 * max(0.0, Q.centroid_offset - 0.019)
        + 88.72 * max(0.0, Q.centroid_offset - 0.05)
        - 118.7 * max(0.0, 0.051 - Q.e2)
        - 46.72 * max(0.0, Q.girth - 0.083)
        + 5.795 * max(0.0, 0.0036 - Q.girth2)
        + 807.8 * max(0.0, 0.0086 - Q.girth2)
        + 27.95 * max(0.0, Q.girth2_top5 - 0.011)
        - 453.3 * max(0.0, 0.0078 - Q.lam1)
        - 120.7 * max(0.0, Q.lam2 - 0.0034)
        + 3.924 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.02677 * max(0.0, 50.0 - Q.mass)
        + 36.51 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 9.588 * max(0.0, 0.11 - Q.max_dr)
        - 0.0823 * max(0.0, Q.sum_pt - 990.0)
        + 809.6 * max(0.0, 0.013 - Q.width)
        + 3814.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        + 0.1074 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 0.1699 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        - 0.03023 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        + 7.967 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        + 23.02 * max(0.0, 0.021 - Q.centroid_offset)
        + 78.25 * max(0.0, 0.088 - Q.girth)
        - 939.8 * max(0.0, Q.girth2 - 0.0015)
        + 1065.0 * max(0.0, Q.girth2 - 0.0044)
        - 2763.0 * max(0.0, Q.girth2 - 0.0075)
        + 2043.0 * max(0.0, Q.girth2 - 0.0087)
        - 401.3 * max(0.0, Q.girth2 - 0.015)
        - 316.2 * max(0.0, 0.0011 - Q.girth2_top2)
        - 39.46 * max(0.0, 0.0084 - Q.lam1)
        - 0.1089 * max(0.0, Q.mass - 80.4)
        + 25.15 * max(0.0, Q.mass_over_sum_pt - 0.072)
        + 171.0 * max(0.0, Q.mass_over_sum_pt - 0.085)
        - 431.6 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 65.56 * max(0.0, 0.0055 - Q.width)
        + 0.273 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        + 71.31 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        + 301.9 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 467.9 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 30.73 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        + 35.27 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 31.98 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 63.73 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        - 559.3 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        - 72.86 * max(0.0, Q.log_sum_pt - 6.7)
        + 445.6 * max(0.0, 0.0049 - Q.width)
        + 1632.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 34.04 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 7347.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        - 1786.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        - 2.273 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 16170.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 12.52 * max(0.0, Q.C2 - 0.051)
        + 15.62 * max(0.0, 0.018 - Q.centroid_offset)
        - 3.064 * max(0.0, 0.055 - Q.girth)
        - 265.3 * max(0.0, Q.girth2 - 0.018)
        - 93.13 * max(0.0, Q.lam2 - 0.0017)
        + 61.14 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        - 200.1 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 13.14 * max(0.0, 0.23 - Q.max_dr)
        - 1128.0 * max(0.0, 0.0061 - Q.width)
        - 94.71 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 21190.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        + 3.851 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        - 0.9124 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        - 0.08209 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.02748 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 4614.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 23280.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 33.93 * max(0.0, Q.C2 - 0.055)
        - 17.9 * max(0.0, Q.LHA - 0.33)
        - 10.11 * max(0.0, Q.centroid_offset - 0.038)
        - 8.069 * max(0.0, Q.e2 - 0.036)
        + 45.55 * max(0.0, 0.037 - Q.e2)
        + 1733.0 * max(0.0, Q.girth2 - 0.0085)
        + 1276.0 * max(0.0, 0.0017 - Q.girth2)
        - 514.1 * max(0.0, Q.lam1 - 0.0085)
        + 404.7 * max(0.0, 0.0044 - Q.lam1)
        - 92.29 * max(0.0, 0.0034 - Q.lam2)
        + 7.487 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.0068 * max(0.0, Q.mass - 17.0)
        - 23.95 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 486.0 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 27.3 * max(0.0, 0.035 - Q.C2)
        + 54.16 * max(0.0, Q.centroid_offset - 0.014)
        - 70.97 * max(0.0, 0.04 - Q.centroid_offset)
        + 444.7 * max(0.0, 0.0062 - Q.e2_sq)
        - 15.56 * max(0.0, Q.girth - 0.076)
        - 223.4 * max(0.0, 0.089 - Q.girth)
        + 27.92 * max(0.0, 0.22 - Q.max_dr)
        - 0.0002925 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 5.795 * max(0.0, 0.0036 - Q.width)
        - 1091.0 * max(0.0, 0.0085 - Q.width)
        + 31.42 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 2.584 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 3.545 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 946.0 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        + 897.0 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        + 580.3 * max(0.0, Q.e2 - 0.063)
        + 110.3 * max(0.0, Q.girth2 - 0.019)
        + 0.04586 * max(0.0, Q.mass - 91.2)
        - 5763.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        - 301.3 * max(0.0, Q.C2 - 0.066)
        + 6.026 * max(0.0, 0.038 - Q.centroid_offset)
        - 61.36 * max(0.0, 0.049 - Q.e2)
        + 16.6 * max(0.0, 0.15 - Q.girth)
        - 320.3 * max(0.0, 0.0062 - Q.lam1)
        + 594.9 * max(0.0, 0.015 - Q.lam1)
        - 0.01824 * max(0.0, Q.sum_pt - 1000.0)
        + 2465.0 * max(0.0, 0.0076 - Q.width)
        - 17.83 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.2267 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        + 618.7 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        - 22720.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        - 5.574e-07 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        + 36.72 * max(0.0, 0.067 - Q.C2)
        - 49.02 * max(0.0, Q.centroid_offset - 0.031)
        - 344.4 * max(0.0, 0.0058 - Q.e2_sq)
        + 29.24 * max(0.0, 0.034 - Q.girth)
        - 5.464 * max(0.0, 0.087 - Q.girth)
        + 205.0 * max(0.0, Q.lam1 - 0.0025)
        - 341.9 * max(0.0, Q.lam1 - 0.0042)
        + 526.5 * max(0.0, Q.lam1 - 0.0061)
        + 1876.0 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 1.76 * max(0.0, 0.078 - Q.max_dr)
        - 10.82 * max(0.0, 0.18 - Q.max_dr)
        - 22.21 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        + 674.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 39.46 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 567.5 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        - 6.173 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        - 96.52 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        + 6705.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        - 1209.0 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 154.9 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 36.48 * max(0.0, Q.LHA - 0.34)
        + 182.5 * max(0.0, 0.024 - Q.e2)
        + 18.57 * max(0.0, 0.041 - Q.e2)
        + 3.791 * max(0.0, Q.girth - 0.032)
        + 114.1 * max(0.0, 0.0067 - Q.lam1)
        - 445.1 * max(0.0, 0.0083 - Q.lam1)
        + 3965.0 * max(0.0, 0.00031 - Q.lam2)
        - 29.41 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 675.1 * max(0.0, 0.0067 - Q.width)
        - 17.13 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 44010.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        + 2304.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_t(Q):
    return (0.7458
        + 7.467 * max(0.0, 0.033 - Q.centroid_offset)
        - 5.479 * max(0.0, 0.078 - Q.girth)
        - 35.7 * max(0.0, 0.013 - Q.girth2)
        - 5.046 * max(0.0, 0.0015 - Q.lam1)
        + 0.002956 * max(0.0, 22.0 - Q.mass)
        - 0.01231 * max(0.0, 30.0 - Q.mass)
        + 0.01532 * max(0.0, 59.0 - Q.mass)
        - 0.04453 * max(0.0, Q.sum_pt - 810.0)
        - 0.02642 * max(0.0, Q.sum_pt - 900.0)
        + 323.9 * max(0.0, 0.0044 - Q.width)
        - 35.77 * max(0.0, 0.0089 - Q.width)
        - 548.9 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        - 0.7888 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        + 2.824 * max(0.0, Q.LHA - 0.28)
        - 1947.0 * max(0.0, 0.0081 - Q.e2_sq)
        + 0.2555 * max(0.0, Q.log_sum_pt - 6.4)
        + 3.305 * max(0.0, 0.25 - Q.max_dr)
        - 0.02083 * max(0.0, Q.pt_7 - 34.0)
        + 186.8 * max(0.0, 0.0087 - Q.width)
        - 12.59 * max(0.0, 0.054 - Q.z_7)
        - 38.03 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 72.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 12.79 * max(0.0, Q.LHA - 0.1)
        - 26.62 * max(0.0, 0.0057 - Q.lam1)
        + 10.43 * max(0.0, Q.log_sum_pt - 6.3)
        + 26.24 * max(0.0, Q.log_sum_pt - 6.8)
        - 42.34 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.009804 * max(0.0, 36.0 - Q.mass)
        - 0.008335 * max(0.0, 69.0 - Q.mass)
        - 56.01 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 3.698 * max(0.0, 0.16 - Q.max_dr)
        + 0.03977 * max(0.0, Q.pt_7 - 31.0)
        + 0.01224 * max(0.0, 54.0 - Q.pt_7)
        + 0.008019 * max(0.0, 790.0 - Q.sum_pt)
        + 0.3217 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        + 4.816 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 0.03826 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 0.3949 * max(0.0, Q.LHA - 0.32)
        - 12.82 * max(0.0, Q.centroid_offset - 0.013)
        + 7.607 * max(0.0, Q.e2 - 0.028)
        - 18.77 * max(0.0, Q.e2 - 0.051)
        + 0.0814 * max(0.0, 0.043 - Q.e2)
        + 81.86 * max(0.0, 0.0088 - Q.girth2)
        - 6.771 * max(0.0, Q.lam1 - 0.016)
        + 0.003814 * max(0.0, Q.mass - 70.0)
        - 4.029 * max(0.0, Q.max_dr - 0.15)
        + 87.34 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        - 80.61 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 23.81 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.2287 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        + 14.9 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 13.94 * max(0.0, Q.C2 - 0.067)
        + 13.64 * max(0.0, Q.e2 - 0.02)
        + 204.0 * max(0.0, Q.e2 - 0.064)
        - 24.6 * max(0.0, 0.003 - Q.e2_sq)
        + 79.58 * max(0.0, 0.017 - Q.e2_sq)
        + 0.4466 * max(0.0, 0.13 - Q.girth)
        + 52.79 * max(0.0, Q.girth2 - 0.0034)
        + 27.29 * max(0.0, Q.girth2 - 0.0081)
        - 3240.0 * max(0.0, 0.00033 - Q.lam2)
        + 0.01339 * max(0.0, 44.0 - Q.mass)
        + 10.47 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 22.08 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 0.09967 * max(0.0, Q.max_dr - 0.094)
        + 3.943 * max(0.0, Q.max_dr - 0.2)
        + 7.086 * max(0.0, 0.24 - Q.tau21)
        + 4275.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        - 424.8 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        - 4502.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 458.4 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 5.05 * max(0.0, 0.034 - Q.e2)
        - 0.002777 * max(0.0, 53.0 - Q.pt_7)
        + 404.6 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        - 0.001634 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 13190.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 642.7 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 14.76 * max(0.0, Q.C2 - 0.033)
        + 6.293 * max(0.0, Q.centroid_offset - 0.019)
        + 6.215 * max(0.0, Q.centroid_offset - 0.05)
        + 16.25 * max(0.0, 0.051 - Q.e2)
        + 2.171 * max(0.0, Q.girth - 0.083)
        + 22.63 * max(0.0, 0.0036 - Q.girth2)
        + 209.9 * max(0.0, 0.0086 - Q.girth2)
        + 18.1 * max(0.0, Q.girth2_top5 - 0.011)
        + 33.88 * max(0.0, 0.0078 - Q.lam1)
        - 107.7 * max(0.0, Q.lam2 - 0.0034)
        + 3.234 * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.009019 * max(0.0, 50.0 - Q.mass)
        - 38.06 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 0.1135 * max(0.0, 0.11 - Q.max_dr)
        + 0.06798 * max(0.0, Q.sum_pt - 990.0)
        - 35.7 * max(0.0, 0.013 - Q.width)
        - 1189.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 1.929 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        + 0.01466 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 0.01171 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        - 1.68 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        - 5.514 * max(0.0, 0.021 - Q.centroid_offset)
        - 0.6732 * max(0.0, 0.088 - Q.girth)
        + 138.6 * max(0.0, Q.girth2 - 0.0015)
        - 100.6 * max(0.0, Q.girth2 - 0.0044)
        + 288.6 * max(0.0, Q.girth2 - 0.0075)
        - 255.6 * max(0.0, Q.girth2 - 0.0087)
        + 63.53 * max(0.0, Q.girth2 - 0.015)
        + 409.5 * max(0.0, 0.0011 - Q.girth2_top2)
        + 305.1 * max(0.0, 0.0084 - Q.lam1)
        + 0.0249 * max(0.0, Q.mass - 80.4)
        - 0.7852 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 24.12 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 68.36 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 157.8 * max(0.0, 0.0055 - Q.width)
        - 0.0004851 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 131.7 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 173.4 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        - 187.8 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 12.98 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        - 0.555 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 3.058 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        + 1.8 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 225.6 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        + 33.87 * max(0.0, Q.log_sum_pt - 6.7)
        + 265.8 * max(0.0, 0.0049 - Q.width)
        + 2646.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 188.6 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        + 1269.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 12640.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 0.9427 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 9929.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        + 5.284 * max(0.0, Q.C2 - 0.051)
        + 10.06 * max(0.0, 0.018 - Q.centroid_offset)
        - 7.89 * max(0.0, 0.055 - Q.girth)
        + 182.7 * max(0.0, Q.girth2 - 0.018)
        + 303.2 * max(0.0, Q.lam2 - 0.0017)
        - 61.25 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 17.93 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 8.333 * max(0.0, 0.23 - Q.max_dr)
        - 130.4 * max(0.0, 0.0061 - Q.width)
        - 1720.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 713.8 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 0.5908 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        - 0.8354 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.07815 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.03039 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 2982.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 4671.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 6.523 * max(0.0, Q.C2 - 0.055)
        - 6.027 * max(0.0, Q.LHA - 0.33)
        + 18.72 * max(0.0, Q.centroid_offset - 0.038)
        + 22.71 * max(0.0, Q.e2 - 0.036)
        - 30.18 * max(0.0, 0.037 - Q.e2)
        - 261.3 * max(0.0, Q.girth2 - 0.0085)
        - 476.0 * max(0.0, 0.0017 - Q.girth2)
        - 392.3 * max(0.0, Q.lam1 - 0.0085)
        - 245.4 * max(0.0, 0.0044 - Q.lam1)
        - 220.3 * max(0.0, 0.0034 - Q.lam2)
        - 10.88 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.008558 * max(0.0, Q.mass - 17.0)
        + 14.94 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 2644.0 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        + 11.3 * max(0.0, 0.035 - Q.C2)
        + 10.02 * max(0.0, Q.centroid_offset - 0.014)
        + 1.091 * max(0.0, 0.04 - Q.centroid_offset)
        - 49.0 * max(0.0, 0.0062 - Q.e2_sq)
        - 6.645 * max(0.0, Q.girth - 0.076)
        + 18.05 * max(0.0, 0.089 - Q.girth)
        + 0.7434 * max(0.0, 0.22 - Q.max_dr)
        + 0.001937 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 22.63 * max(0.0, 0.0036 - Q.width)
        + 178.7 * max(0.0, 0.0085 - Q.width)
        - 27.31 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.5376 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.247 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 81.01 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        + 99.81 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        - 176.9 * max(0.0, Q.e2 - 0.063)
        - 280.1 * max(0.0, Q.girth2 - 0.019)
        - 0.04489 * max(0.0, Q.mass - 91.2)
        - 6604.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        + 3.413 * max(0.0, Q.C2 - 0.066)
        - 23.17 * max(0.0, 0.038 - Q.centroid_offset)
        + 32.62 * max(0.0, 0.049 - Q.e2)
        - 25.42 * max(0.0, 0.15 - Q.girth)
        - 101.0 * max(0.0, 0.0062 - Q.lam1)
        - 151.8 * max(0.0, 0.015 - Q.lam1)
        - 0.01966 * max(0.0, Q.sum_pt - 1000.0)
        - 180.2 * max(0.0, 0.0076 - Q.width)
        + 19.66 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.3862 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        + 1590.0 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 42510.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        - 0.0002316 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        + 8.374 * max(0.0, 0.067 - Q.C2)
        - 9.661 * max(0.0, Q.centroid_offset - 0.031)
        + 430.5 * max(0.0, 0.0058 - Q.e2_sq)
        - 26.84 * max(0.0, 0.034 - Q.girth)
        - 15.81 * max(0.0, 0.087 - Q.girth)
        + 26.68 * max(0.0, Q.lam1 - 0.0025)
        + 330.1 * max(0.0, Q.lam1 - 0.0042)
        + 60.37 * max(0.0, Q.lam1 - 0.0061)
        + 1459.0 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        - 0.4029 * max(0.0, 0.078 - Q.max_dr)
        - 1.883 * max(0.0, 0.18 - Q.max_dr)
        + 14.18 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        - 506.3 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 3.934 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 407.1 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 63.01 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        - 49.53 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 1615.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 583.6 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 4.837 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 6.411 * max(0.0, Q.LHA - 0.34)
        + 0.8065 * max(0.0, 0.024 - Q.e2)
        - 3.827 * max(0.0, 0.041 - Q.e2)
        + 16.64 * max(0.0, Q.girth - 0.032)
        - 41.04 * max(0.0, 0.0067 - Q.lam1)
        - 75.97 * max(0.0, 0.0083 - Q.lam1)
        + 2345.0 * max(0.0, 0.00031 - Q.lam2)
        - 18.56 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 56.86 * max(0.0, 0.0067 - Q.width)
        + 6.013 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 5557.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 1107.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.width > 0.009139599744230509:
        if s['g'] - s['t'] > -0.2338188737630844:
            if s['g'] - s['t'] > 0.23750445246696472:
                return 'g'   # 84% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.08271557837724686:
                    if s['g'] - s['q'] > -0.047294216230511665:
                        if Q.pt_7 > 36.953125:
                            return 'g'   # 65% of the training jets here get this class from the formula
                        else:
                            return 't'   # 56% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 70% of the training jets here get this class from the formula
                else:
                    return 'Z'   # 70% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.2741070091724396:
                if s['W'] - s['t'] > -3.7809600830078125:
                    return 'Z'   # 90% of the training jets here get this class from the formula
                else:
                    return 't'   # 50% of the training jets here get this class from the formula
            else:
                if s['g'] - s['t'] > -0.6848350167274475:
                    if s['q'] - s['t'] > -0.13231747597455978:
                        return 'q'   # 59% of the training jets here get this class from the formula
                    else:
                        return 't'   # 79% of the training jets here get this class from the formula
                else:
                    return 't'   # 99% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.19611406326293945:
            if s['g'] - s['q'] > -0.03193817473948002:
                if s['g'] - s['t'] > -0.06135934963822365:
                    if s['g'] - s['W'] > 0.10827063024044037:
                        return 'g'   # 93% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.31159399449825287:
                            return 'g'   # 50% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 84% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > 0.012020778842270374:
                        return 'W'   # 92% of the training jets here get this class from the formula
                    else:
                        return 't'   # 83% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.08383183926343918:
                    if s['g'] - s['q'] > -0.25008220970630646:
                        if s['q'] - s['t'] > -0.007417362183332443:
                            if s['g'] - s['q'] > -0.10939201712608337:
                                return 'q'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.031208575703203678:
                                    return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 55% of the training jets here get this class from the formula
                        else:
                            return 't'   # 84% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > 0.0004665457963710651:
                            return 'q'   # 98% of the training jets here get this class from the formula
                        else:
                            return 't'   # 76% of the training jets here get this class from the formula
                else:
                    return 'W'   # 68% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.023965108208358288:
                if s['g'] - s['W'] > -0.026752823032438755:
                    if s['g'] - s['W'] > 0.24000369757413864:
                        if Q.mass > 29.851950645446777:
                            if s['g'] - s['t'] > 2.304056406021118:
                                return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 50% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 0.5733291208744049:
                            return 'g'   # 67% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 49% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.2864387333393097:
                        if s['g'] - s['W'] > -0.6049911379814148:
                            if s['W'] - s['t'] > 0.11120917275547981:
                                if s['g'] - s['Z'] > 0.31317582726478577:
                                    if Q.mass > 28.520703315734863:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 49% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                return 't'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 98% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['Z'] > -1.010757327079773:
                            return 'W'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.012250874191522598:
                                return 'W'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 52% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > -0.05626073107123375:
                    if s['g'] - s['t'] > -0.010046693496406078:
                        return 'g'   # 73% of the training jets here get this class from the formula
                    else:
                        return 't'   # 81% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.17773982137441635:
                        if s['W'] - s['Z'] > -0.36161506175994873:
                            if s['q'] - s['Z'] > -0.7992958426475525:
                                return 'W'   # 48% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 68% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.947811484336853:
                                if s['q'] - s['Z'] > -0.3195638507604599:
                                    return 'q'   # 49% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 3.4539737701416016:
                                        return 'g'   # 42% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 97% of the training jets here get this class from the formula
                    else:
                        return 't'   # 68% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
