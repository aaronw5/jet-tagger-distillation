"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 64.65% (the formula: 64.64%); same class as the formula for 95.84% of jets.  901 leaves, depth 19.
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
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
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
    return (6.016
        + 17.91 * max(0.0, 0.033 - Q.centroid_offset)
        - 9.567 * max(0.0, 0.013 - Q.girth2)
        + 0.04325 * max(0.0, 29.0 - Q.mass)
        + 0.01948 * max(0.0, 74.0 - Q.mass)
        + 0.003055 * max(0.0, Q.sum_pt - 880.0)
        + 303.0 * max(0.0, 0.005 - Q.width)
        - 2103.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 266.5 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 1.336 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.0001834 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
        + 8.088e-05 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        + 0.07175 * max(0.0, Q.pt_7 - 32.0)
        - 61.75 * max(0.0, 0.01 - Q.width)
        + 6533.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 59.24 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 1099.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 0.0005205 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        - 10.79 * max(0.0, Q.LHA - 0.13)
        + 42.55 * max(0.0, 0.005 - Q.lam1)
        + 1.646 * max(0.0, 6.4 - Q.log_sum_pt)
        + 0.0569 * max(0.0, Q.pt_7 - 28.0)
        + 0.003701 * max(0.0, 780.0 - Q.sum_pt)
        - 19.92 * max(0.0, Q.z_7 - 0.047)
        - 112.6 * max(0.0, 0.018 - Q.z_7)
        - 0.7346 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
        - 0.2497 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        + 2.858 * max(0.0, Q.LHA - 0.27)
        - 17.4 * max(0.0, Q.centroid_offset - 0.013)
        + 1.373 * max(0.0, 0.047 - Q.e2)
        - 61.75 * max(0.0, 0.01 - Q.girth2)
        - 13.79 * max(0.0, Q.lam1 - 0.016)
        - 6.211 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        - 7.565 * max(0.0, Q.C2 - 0.063)
        + 683.5 * max(0.0, 0.00042 - Q.lam2)
        + 0.003526 * max(0.0, 55.0 - Q.mass)
        - 13.08 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 0.3706 * max(0.0, 0.27 - Q.tau21)
        + 57.0 * max(0.0, Q.width - 0.0016)
        + 0.02003 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
        + 0.005505 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        + 7.947 * max(0.0, 0.04 - Q.e2)
        + 20.95 * max(0.0, Q.log_sum_pt - 6.9)
        - 27.34 * max(0.0, 0.035 - Q.z_7)
        - 19.34 * max(0.0, 0.066 - Q.z_7)
        - 10.51 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        - 102300.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
        - 32080.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 673.4 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 23.82 * max(0.0, Q.girth - 0.09)
        + 136.1 * max(0.0, Q.lam2 - 0.0027)
        - 292.5 * max(0.0, 0.00066 - Q.lam2)
        - 44.15 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        + 2.194 * Q.max_dr
        - 168.2 * max(0.0, 0.012 - Q.width)
        + 769.3 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
        + 55.52 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
        + 0.01536 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 88.38 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 122.4 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.007496 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 3.257 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 16.24 * max(0.0, 0.038 - Q.e2)
        - 431.5 * max(0.0, 0.0011 - Q.e2_sq)
        + 18.99 * max(0.0, 0.087 - Q.girth)
        + 41.55 * max(0.0, Q.girth2 - 0.0046)
        + 0.008359 * max(0.0, Q.mass - 80.4)
        + 6.607 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 14.96 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 1.241 * max(0.0, 0.19 - Q.planar_flow)
        - 588.6 * max(0.0, 0.00069 - Q.width)
        + 187.4 * max(0.0, 0.0056 - Q.width)
        + 0.9954 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
        - 47.89 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 0.006334 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
        + 58.25 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 0.001735 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        - 16.03 * max(0.0, 0.063 - Q.girth)
        - 56.95 * max(0.0, 0.0049 - Q.width)
        + 12180.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        - 77.69 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
        + 1.186 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 9299.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        - 4.423 * max(0.0, 0.019 - Q.centroid_offset)
        + 218.0 * max(0.0, Q.lam2 - 0.0015)
        - 0.5464 * max(0.0, Q.log_sum_pt - 6.3)
        - 0.06421 * max(0.0, 25.0 - Q.mass)
        + 0.1155 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
        - 342.1 * max(0.0, 0.00029 - Q.width)
        + 340.3 * max(0.0, 0.0065 - Q.width)
        + 0.6344 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
        + 0.07214 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 2001.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
        - 8876.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 8.459 * max(0.0, Q.C2 - 0.059)
        + 4.191 * max(0.0, Q.e2 - 0.037)
        - 7.494 * max(0.0, 0.037 - Q.e2)
        - 183.8 * max(0.0, 0.0018 - Q.girth2)
        - 227.9 * max(0.0, Q.lam2 - 0.00016)
        - 0.03971 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.05654 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
        + 0.6244 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
        + 12.01 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
        - 28.07 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        + 0.9754 * max(0.0, 0.15 - Q.LHA)
        - 28.69 * max(0.0, 0.05 - Q.centroid_offset)
        + 3.405 * max(0.0, Q.girth - 0.074)
        - 15.76 * max(0.0, 0.088 - Q.girth)
        - 166.2 * max(0.0, 0.0038 - Q.width)
        - 157.9 * max(0.0, 0.0088 - Q.width)
        - 101.8 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 7.384 * max(0.0, Q.e2 - 0.062)
        + 0.001941 * max(0.0, Q.mass - 91.2)
        - 4.23 * max(0.0, Q.width - 0.018)
        - 7180.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 1.533 * max(0.0, Q.C2 - 0.065)
        + 4.481 * max(0.0, 0.14 - Q.girth)
        - 0.01687 * max(0.0, Q.sum_pt - 980.0)
        - 25.09 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.2568 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 1.881e-06 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 0.2858 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        - 2.221 * max(0.0, 0.091 - Q.girth)
        - 45.36 * max(0.0, 0.014 - Q.girth2)
        + 3.425 * max(0.0, Q.lam1 - 0.0075)
        - 0.005936 * max(0.0, 80.4 - Q.mass)
        - 0.4223 * max(0.0, 0.18 - Q.max_dr)
        + 1.708 * max(0.0, 0.0075 - Q.width)
        - 0.04219 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
        + 1428.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 389.8 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
        + 33.53 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 23.37 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 30.02 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 193.3 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 112.1 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        + 247.6 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
        + 0.3863 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        - 3.668 * max(0.0, Q.LHA - 0.34)
        - 17.52 * max(0.0, 0.041 - Q.e2)
        + 136.5 * max(0.0, 0.0065 - Q.lam1)
        + 64.69 * max(0.0, 0.0081 - Q.lam1)
        - 113.9 * max(0.0, 0.0069 - Q.width)
        + 0.5014 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
        - 4.419 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 279.3 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_q(Q):
    return (7.077
        + 15.86 * max(0.0, 0.033 - Q.centroid_offset)
        + 3.527 * max(0.0, 0.013 - Q.girth2)
        + 0.007491 * max(0.0, 29.0 - Q.mass)
        + 0.01389 * max(0.0, 74.0 - Q.mass)
        - 0.0001114 * max(0.0, Q.sum_pt - 880.0)
        + 375.5 * max(0.0, 0.005 - Q.width)
        - 2182.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 146.4 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 1.315 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.0002305 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
        - 5.1e-06 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        + 0.03472 * max(0.0, Q.pt_7 - 32.0)
        - 9.534 * max(0.0, 0.01 - Q.width)
        + 4312.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        + 107.4 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        - 807.7 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 0.0001458 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        - 10.08 * max(0.0, Q.LHA - 0.13)
        - 62.78 * max(0.0, 0.005 - Q.lam1)
        + 0.2032 * max(0.0, 6.4 - Q.log_sum_pt)
        - 0.01598 * max(0.0, Q.pt_7 - 28.0)
        - 0.001645 * max(0.0, 780.0 - Q.sum_pt)
        - 7.029 * max(0.0, Q.z_7 - 0.047)
        + 10.18 * max(0.0, 0.018 - Q.z_7)
        - 0.07078 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
        - 0.1089 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        - 3.016 * max(0.0, Q.LHA - 0.27)
        - 18.15 * max(0.0, Q.centroid_offset - 0.013)
        + 4.291 * max(0.0, 0.047 - Q.e2)
        - 9.534 * max(0.0, 0.01 - Q.girth2)
        - 52.68 * max(0.0, Q.lam1 - 0.016)
        - 7.262 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 8.152 * max(0.0, Q.C2 - 0.063)
        + 1356.0 * max(0.0, 0.00042 - Q.lam2)
        - 0.003154 * max(0.0, 55.0 - Q.mass)
        - 4.093 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 2.14 * max(0.0, 0.27 - Q.tau21)
        - 108.4 * max(0.0, Q.width - 0.0016)
        - 0.1119 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
        + 0.05521 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        - 3.843 * max(0.0, 0.04 - Q.e2)
        - 4.355 * max(0.0, Q.log_sum_pt - 6.9)
        + 28.8 * max(0.0, 0.035 - Q.z_7)
        + 4.135 * max(0.0, 0.066 - Q.z_7)
        - 44.54 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 49280.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
        - 16710.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        - 103.6 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 39.87 * max(0.0, Q.girth - 0.09)
        + 268.3 * max(0.0, Q.lam2 - 0.0027)
        - 467.4 * max(0.0, 0.00066 - Q.lam2)
        - 34.83 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        + 2.431 * Q.max_dr
        - 193.3 * max(0.0, 0.012 - Q.width)
        - 1686.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
        + 72.15 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
        + 0.009784 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 90.1 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 113.2 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.01102 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 3.395 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 15.35 * max(0.0, 0.038 - Q.e2)
        - 579.9 * max(0.0, 0.0011 - Q.e2_sq)
        + 16.19 * max(0.0, 0.087 - Q.girth)
        + 78.38 * max(0.0, Q.girth2 - 0.0046)
        - 0.00101 * max(0.0, Q.mass - 80.4)
        - 7.308 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 17.99 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 1.093 * max(0.0, 0.19 - Q.planar_flow)
        - 469.3 * max(0.0, 0.00069 - Q.width)
        + 233.4 * max(0.0, 0.0056 - Q.width)
        + 0.948 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
        - 65.56 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 0.003338 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
        + 48.74 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 0.005843 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        - 12.11 * max(0.0, 0.063 - Q.girth)
        - 157.1 * max(0.0, 0.0049 - Q.width)
        - 1019.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 9.557 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
        + 1.385 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 11730.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 6.524 * max(0.0, 0.019 - Q.centroid_offset)
        + 243.0 * max(0.0, Q.lam2 - 0.0015)
        - 1.696 * max(0.0, Q.log_sum_pt - 6.3)
        - 0.03883 * max(0.0, 25.0 - Q.mass)
        + 0.07893 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
        + 33.95 * max(0.0, 0.00029 - Q.width)
        + 325.0 * max(0.0, 0.0065 - Q.width)
        + 3.368 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
        + 0.0683 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 3099.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
        - 10940.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 5.341 * max(0.0, Q.C2 - 0.059)
        - 10.83 * max(0.0, Q.e2 - 0.037)
        + 8.861 * max(0.0, 0.037 - Q.e2)
        + 214.1 * max(0.0, 0.0018 - Q.girth2)
        - 258.4 * max(0.0, Q.lam2 - 0.00016)
        + 1.244 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.438 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
        + 9.108 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
        - 30.53 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
        - 474.8 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        + 2.995 * max(0.0, 0.15 - Q.LHA)
        - 28.82 * max(0.0, 0.05 - Q.centroid_offset)
        + 8.189 * max(0.0, Q.girth - 0.074)
        - 20.82 * max(0.0, 0.088 - Q.girth)
        - 16.54 * max(0.0, 0.0038 - Q.width)
        - 157.4 * max(0.0, 0.0088 - Q.width)
        - 117.3 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 7.15 * max(0.0, Q.e2 - 0.062)
        - 0.001091 * max(0.0, Q.mass - 91.2)
        + 39.06 * max(0.0, Q.width - 0.018)
        - 9344.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 17.78 * max(0.0, Q.C2 - 0.065)
        + 2.03 * max(0.0, 0.14 - Q.girth)
        + 0.00521 * max(0.0, Q.sum_pt - 980.0)
        - 17.77 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.2886 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 2.206e-06 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 0.4383 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        - 7.923 * max(0.0, 0.091 - Q.girth)
        - 46.86 * max(0.0, 0.014 - Q.girth2)
        + 123.4 * max(0.0, Q.lam1 - 0.0075)
        + 0.005581 * max(0.0, 80.4 - Q.mass)
        + 0.3883 * max(0.0, 0.18 - Q.max_dr)
        - 50.22 * max(0.0, 0.0075 - Q.width)
        - 0.04087 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
        + 3065.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 283.9 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
        + 140.2 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        - 71.38 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 9.54 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 142.7 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 217.6 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        - 179.8 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
        + 0.5023 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        - 5.506 * max(0.0, Q.LHA - 0.34)
        + 9.85 * max(0.0, 0.041 - Q.e2)
        + 156.7 * max(0.0, 0.0065 - Q.lam1)
        - 82.89 * max(0.0, 0.0081 - Q.lam1)
        - 183.5 * max(0.0, 0.0069 - Q.width)
        + 0.3556 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
        - 4.079 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        - 609.2 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_W(Q):
    return (-10.81
        - 73.56 * max(0.0, 0.033 - Q.centroid_offset)
        - 98.85 * max(0.0, 0.013 - Q.girth2)
        - 0.02696 * max(0.0, 29.0 - Q.mass)
        - 0.05609 * max(0.0, 74.0 - Q.mass)
        - 0.00457 * max(0.0, Q.sum_pt - 880.0)
        - 1011.0 * max(0.0, 0.005 - Q.width)
        + 2144.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 1602.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        + 2.02 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        - 0.0005107 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
        + 5.043e-05 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        - 0.03066 * max(0.0, Q.pt_7 - 32.0)
        + 402.2 * max(0.0, 0.01 - Q.width)
        - 11270.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 67.84 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 500.1 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 6.325e-05 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        - 2.853 * max(0.0, Q.LHA - 0.13)
        + 165.9 * max(0.0, 0.005 - Q.lam1)
        + 0.6172 * max(0.0, 6.4 - Q.log_sum_pt)
        + 0.03204 * max(0.0, Q.pt_7 - 28.0)
        - 0.001253 * max(0.0, 780.0 - Q.sum_pt)
        + 5.505 * max(0.0, Q.z_7 - 0.047)
        + 6.797 * max(0.0, 0.018 - Q.z_7)
        - 0.09859 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
        + 0.1505 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        + 34.35 * max(0.0, Q.LHA - 0.27)
        + 11.37 * max(0.0, Q.centroid_offset - 0.013)
        - 90.51 * max(0.0, 0.047 - Q.e2)
        + 402.2 * max(0.0, 0.01 - Q.girth2)
        + 131.9 * max(0.0, Q.lam1 - 0.016)
        + 43.89 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 129.8 * max(0.0, Q.C2 - 0.063)
        + 248.9 * max(0.0, 0.00042 - Q.lam2)
        - 0.03193 * max(0.0, 55.0 - Q.mass)
        + 16.21 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 0.407 * max(0.0, 0.27 - Q.tau21)
        + 441.4 * max(0.0, Q.width - 0.0016)
        - 0.4573 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
        + 0.09805 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        + 139.9 * max(0.0, 0.04 - Q.e2)
        - 18.02 * max(0.0, Q.log_sum_pt - 6.9)
        - 22.85 * max(0.0, 0.035 - Q.z_7)
        - 9.594 * max(0.0, 0.066 - Q.z_7)
        + 20.2 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 368100.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
        + 9185.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 386.5 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        - 116.5 * max(0.0, Q.girth - 0.09)
        + 280.0 * max(0.0, Q.lam2 - 0.0027)
        + 344.6 * max(0.0, 0.00066 - Q.lam2)
        + 67.01 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 9.94 * Q.max_dr
        + 395.7 * max(0.0, 0.012 - Q.width)
        - 3711.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
        - 39.92 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
        - 0.0876 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        - 300.6 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        - 245.3 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        - 0.01434 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        + 11.65 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 67.17 * max(0.0, 0.038 - Q.e2)
        + 314.6 * max(0.0, 0.0011 - Q.e2_sq)
        - 69.33 * max(0.0, 0.087 - Q.girth)
        - 312.6 * max(0.0, Q.girth2 - 0.0046)
        + 0.03283 * max(0.0, Q.mass - 80.4)
        + 44.54 * max(0.0, Q.mass_over_sum_pt - 0.073)
        - 71.27 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 2.063 * max(0.0, 0.19 - Q.planar_flow)
        + 456.5 * max(0.0, 0.00069 - Q.width)
        - 649.1 * max(0.0, 0.0056 - Q.width)
        - 0.9153 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
        + 323.1 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 0.003327 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
        + 143.9 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 0.00191 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        - 37.55 * max(0.0, 0.063 - Q.girth)
        + 676.0 * max(0.0, 0.0049 - Q.width)
        + 19840.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        - 420.5 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
        - 1.968 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        + 8810.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 76.21 * max(0.0, 0.019 - Q.centroid_offset)
        - 147.4 * max(0.0, Q.lam2 - 0.0015)
        - 2.525 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.06944 * max(0.0, 25.0 - Q.mass)
        + 0.007917 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
        + 666.4 * max(0.0, 0.00029 - Q.width)
        - 679.3 * max(0.0, 0.0065 - Q.width)
        + 9.79 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
        - 0.02899 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 2721.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
        + 21240.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        - 59.04 * max(0.0, Q.C2 - 0.059)
        - 17.1 * max(0.0, Q.e2 - 0.037)
        + 137.7 * max(0.0, 0.037 - Q.e2)
        + 98.71 * max(0.0, 0.0018 - Q.girth2)
        + 86.36 * max(0.0, Q.lam2 - 0.00016)
        + 2.083 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.2695 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
        - 14.71 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
        + 53.12 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
        + 577.9 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        - 4.67 * max(0.0, 0.15 - Q.LHA)
        + 74.73 * max(0.0, 0.05 - Q.centroid_offset)
        - 39.98 * max(0.0, Q.girth - 0.074)
        - 68.07 * max(0.0, 0.088 - Q.girth)
        - 340.5 * max(0.0, 0.0038 - Q.width)
        + 434.2 * max(0.0, 0.0088 - Q.width)
        - 39.19 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 57.85 * max(0.0, Q.e2 - 0.062)
        - 0.01563 * max(0.0, Q.mass - 91.2)
        + 3.967 * max(0.0, Q.width - 0.018)
        - 4354.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 75.89 * max(0.0, Q.C2 - 0.065)
        - 18.3 * max(0.0, 0.14 - Q.girth)
        + 0.0008334 * max(0.0, Q.sum_pt - 980.0)
        + 15.59 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        + 0.286 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        + 4.497e-05 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 1.37 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        + 158.4 * max(0.0, 0.091 - Q.girth)
        - 24.17 * max(0.0, 0.014 - Q.girth2)
        - 125.1 * max(0.0, Q.lam1 - 0.0075)
        + 0.02058 * max(0.0, 80.4 - Q.mass)
        - 5.704 * max(0.0, 0.18 - Q.max_dr)
        + 1067.0 * max(0.0, 0.0075 - Q.width)
        - 0.0336 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
        - 1644.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 3686.0 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
        - 326.7 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 106.6 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 12.18 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 550.7 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 568.6 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        + 261.9 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
        - 3.177 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        + 26.51 * max(0.0, Q.LHA - 0.34)
        - 203.2 * max(0.0, 0.041 - Q.e2)
        - 1671.0 * max(0.0, 0.0065 - Q.lam1)
        + 1033.0 * max(0.0, 0.0081 - Q.lam1)
        + 1306.0 * max(0.0, 0.0069 - Q.width)
        + 6.39 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
        - 15.5 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 3895.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_Z(Q):
    return (-10.32
        - 36.91 * max(0.0, 0.033 - Q.centroid_offset)
        - 32.94 * max(0.0, 0.013 - Q.girth2)
        + 0.04739 * max(0.0, 29.0 - Q.mass)
        - 0.03917 * max(0.0, 74.0 - Q.mass)
        + 0.00137 * max(0.0, Q.sum_pt - 880.0)
        + 1122.0 * max(0.0, 0.005 - Q.width)
        - 4943.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 427.4 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        + 0.4932 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        - 6.885e-05 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
        - 4.871e-05 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        - 0.003173 * max(0.0, Q.pt_7 - 32.0)
        + 282.3 * max(0.0, 0.01 - Q.width)
        + 1417.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 97.84 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 737.8 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 0.0001485 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        + 5.897 * max(0.0, Q.LHA - 0.13)
        + 77.48 * max(0.0, 0.005 - Q.lam1)
        + 1.013 * max(0.0, 6.4 - Q.log_sum_pt)
        + 0.003451 * max(0.0, Q.pt_7 - 28.0)
        + 0.0005218 * max(0.0, 780.0 - Q.sum_pt)
        + 3.564 * max(0.0, Q.z_7 - 0.047)
        - 4.509 * max(0.0, 0.018 - Q.z_7)
        + 0.206 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
        + 0.08894 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        - 7.562 * max(0.0, Q.LHA - 0.27)
        + 0.5698 * max(0.0, Q.centroid_offset - 0.013)
        - 90.86 * max(0.0, 0.047 - Q.e2)
        + 282.3 * max(0.0, 0.01 - Q.girth2)
        + 126.6 * max(0.0, Q.lam1 - 0.016)
        + 55.65 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 4.878 * max(0.0, Q.C2 - 0.063)
        - 1096.0 * max(0.0, 0.00042 - Q.lam2)
        + 0.02337 * max(0.0, 55.0 - Q.mass)
        + 85.35 * max(0.0, Q.mass_over_sum_pt - 0.086)
        + 0.4957 * max(0.0, 0.27 - Q.tau21)
        + 33.7 * max(0.0, Q.width - 0.0016)
        + 0.1106 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
        - 0.0239 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        - 66.91 * max(0.0, 0.04 - Q.e2)
        - 8.352 * max(0.0, Q.log_sum_pt - 6.9)
        - 9.069 * max(0.0, 0.035 - Q.z_7)
        - 2.439 * max(0.0, 0.066 - Q.z_7)
        + 7.572 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 64630.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
        + 23850.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 146.3 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        - 79.37 * max(0.0, Q.girth - 0.09)
        + 50.12 * max(0.0, Q.lam2 - 0.0027)
        + 741.7 * max(0.0, 0.00066 - Q.lam2)
        + 61.1 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 6.97 * Q.max_dr
        + 610.3 * max(0.0, 0.012 - Q.width)
        - 3637.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
        - 89.0 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
        - 0.1083 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        - 314.4 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        - 282.7 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        - 0.003759 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        + 13.2 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        + 96.37 * max(0.0, 0.038 - Q.e2)
        + 1183.0 * max(0.0, 0.0011 - Q.e2_sq)
        - 14.2 * max(0.0, 0.087 - Q.girth)
        + 159.6 * max(0.0, Q.girth2 - 0.0046)
        - 0.07694 * max(0.0, Q.mass - 80.4)
        + 68.1 * max(0.0, Q.mass_over_sum_pt - 0.073)
        - 180.5 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 2.858 * max(0.0, 0.19 - Q.planar_flow)
        - 687.9 * max(0.0, 0.00069 - Q.width)
        - 628.4 * max(0.0, 0.0056 - Q.width)
        - 1.681 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
        + 6857.0 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 0.01278 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
        - 1939.0 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 0.02476 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        + 20.96 * max(0.0, 0.063 - Q.girth)
        - 1351.0 * max(0.0, 0.0049 - Q.width)
        - 3560.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 48.05 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
        - 1.884 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 11120.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 1.145 * max(0.0, 0.019 - Q.centroid_offset)
        + 50.89 * max(0.0, Q.lam2 - 0.0015)
        - 0.6633 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.009887 * max(0.0, 25.0 - Q.mass)
        - 0.005401 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
        + 366.3 * max(0.0, 0.00029 - Q.width)
        - 242.1 * max(0.0, 0.0065 - Q.width)
        + 11.21 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
        - 0.03498 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 858.9 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
        + 16400.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 21.73 * max(0.0, Q.C2 - 0.059)
        + 2.625 * max(0.0, Q.e2 - 0.037)
        - 26.39 * max(0.0, 0.037 - Q.e2)
        + 190.7 * max(0.0, 0.0018 - Q.girth2)
        - 14.06 * max(0.0, Q.lam2 - 0.00016)
        + 0.5037 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.1864 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
        - 6.634 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
        - 32.73 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
        - 18.72 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        - 1.323 * max(0.0, 0.15 - Q.LHA)
        + 33.81 * max(0.0, 0.05 - Q.centroid_offset)
        + 15.61 * max(0.0, Q.girth - 0.074)
        - 12.31 * max(0.0, 0.088 - Q.girth)
        + 130.0 * max(0.0, 0.0038 - Q.width)
        + 274.7 * max(0.0, 0.0088 - Q.width)
        + 26.0 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 92.66 * max(0.0, Q.e2 - 0.062)
        + 0.0917 * max(0.0, Q.mass - 91.2)
        + 162.5 * max(0.0, Q.width - 0.018)
        - 7835.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 34.13 * max(0.0, Q.C2 - 0.065)
        + 18.87 * max(0.0, 0.14 - Q.girth)
        - 0.000803 * max(0.0, Q.sum_pt - 980.0)
        - 1.441 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        + 0.03164 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        + 3.013e-05 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 0.2232 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        - 18.31 * max(0.0, 0.091 - Q.girth)
        + 406.5 * max(0.0, 0.014 - Q.girth2)
        - 315.6 * max(0.0, Q.lam1 - 0.0075)
        - 0.01743 * max(0.0, 80.4 - Q.mass)
        - 8.164 * max(0.0, 0.18 - Q.max_dr)
        - 630.8 * max(0.0, 0.0075 - Q.width)
        + 0.05962 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
        + 10520.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 5814.0 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
        + 364.1 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        - 509.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        - 45.84 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        - 197.7 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 13.1 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        - 1837.0 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
        - 11.03 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        + 19.97 * max(0.0, Q.LHA - 0.34)
        + 106.3 * max(0.0, 0.041 - Q.e2)
        - 318.8 * max(0.0, 0.0065 - Q.lam1)
        + 161.9 * max(0.0, 0.0081 - Q.lam1)
        + 291.6 * max(0.0, 0.0069 - Q.width)
        + 0.3321 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
        + 2.856 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 1591.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_t(Q):
    return (0.9866
        + 16.74 * max(0.0, 0.033 - Q.centroid_offset)
        - 65.33 * max(0.0, 0.013 - Q.girth2)
        - 0.01549 * max(0.0, 29.0 - Q.mass)
        - 0.005158 * max(0.0, 74.0 - Q.mass)
        + 0.002933 * max(0.0, Q.sum_pt - 880.0)
        + 164.7 * max(0.0, 0.005 - Q.width)
        - 756.8 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 80.68 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 0.5548 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.00015 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
        + 0.0001004 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        + 0.0007656 * max(0.0, Q.pt_7 - 32.0)
        - 23.42 * max(0.0, 0.01 - Q.width)
        + 2391.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 14.31 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        - 551.5 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        + 9.621e-05 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        - 3.483 * max(0.0, Q.LHA - 0.13)
        + 72.31 * max(0.0, 0.005 - Q.lam1)
        + 0.02359 * max(0.0, 6.4 - Q.log_sum_pt)
        - 0.003972 * max(0.0, Q.pt_7 - 28.0)
        + 0.0008682 * max(0.0, 780.0 - Q.sum_pt)
        + 4.4 * max(0.0, Q.z_7 - 0.047)
        - 3.552 * max(0.0, 0.018 - Q.z_7)
        - 0.06482 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
        + 0.004828 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        - 4.276 * max(0.0, Q.LHA - 0.27)
        + 2.905 * max(0.0, Q.centroid_offset - 0.013)
        + 23.04 * max(0.0, 0.047 - Q.e2)
        - 23.42 * max(0.0, 0.01 - Q.girth2)
        - 39.65 * max(0.0, Q.lam1 - 0.016)
        + 6.88 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        - 3.52 * max(0.0, Q.C2 - 0.063)
        - 800.5 * max(0.0, 0.00042 - Q.lam2)
        + 0.008654 * max(0.0, 55.0 - Q.mass)
        - 46.49 * max(0.0, Q.mass_over_sum_pt - 0.086)
        + 3.754 * max(0.0, 0.27 - Q.tau21)
        - 167.4 * max(0.0, Q.width - 0.0016)
        + 0.1702 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
        - 0.08646 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        - 20.44 * max(0.0, 0.04 - Q.e2)
        + 32.68 * max(0.0, Q.log_sum_pt - 6.9)
        - 36.61 * max(0.0, 0.035 - Q.z_7)
        - 22.69 * max(0.0, 0.066 - Q.z_7)
        + 15.76 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        - 117300.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
        + 12680.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 729.7 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 19.99 * max(0.0, Q.girth - 0.09)
        + 13.17 * max(0.0, Q.lam2 - 0.0027)
        + 10.53 * max(0.0, 0.00066 - Q.lam2)
        + 18.56 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        + 0.4758 * Q.max_dr
        + 4.248 * max(0.0, 0.012 - Q.width)
        + 2196.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
        + 85.09 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
        + 0.007649 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 64.39 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 47.74 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.001427 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 3.933 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        + 6.384 * max(0.0, 0.038 - Q.e2)
        + 37.18 * max(0.0, 0.0011 - Q.e2_sq)
        + 5.084 * max(0.0, 0.087 - Q.girth)
        + 131.9 * max(0.0, Q.girth2 - 0.0046)
        + 0.009117 * max(0.0, Q.mass - 80.4)
        + 4.176 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 20.88 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 0.182 * max(0.0, 0.19 - Q.planar_flow)
        - 188.7 * max(0.0, 0.00069 - Q.width)
        - 0.6456 * max(0.0, 0.0056 - Q.width)
        + 0.8778 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
        + 154.6 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 0.001089 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
        - 11.66 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 0.009119 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        - 2.571 * max(0.0, 0.063 - Q.girth)
        - 172.0 * max(0.0, 0.0049 - Q.width)
        - 40600.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 61.65 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
        + 0.1816 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 3685.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        - 6.816 * max(0.0, 0.019 - Q.centroid_offset)
        + 4.025 * max(0.0, Q.lam2 - 0.0015)
        + 1.298 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.02256 * max(0.0, 25.0 - Q.mass)
        + 0.2226 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
        + 287.2 * max(0.0, 0.00029 - Q.width)
        + 35.43 * max(0.0, 0.0065 - Q.width)
        - 1.195 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
        + 0.056 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 678.5 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
        - 2413.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 6.327 * max(0.0, Q.C2 - 0.059)
        + 23.7 * max(0.0, Q.e2 - 0.037)
        - 13.86 * max(0.0, 0.037 - Q.e2)
        - 450.3 * max(0.0, 0.0018 - Q.girth2)
        + 225.6 * max(0.0, Q.lam2 - 0.00016)
        - 3.652 * max(0.0, 6.3 - Q.log_sum_pt)
        + 1.042 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
        - 15.34 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
        + 109.6 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
        + 931.4 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        + 6.06 * max(0.0, 0.15 - Q.LHA)
        - 7.103 * max(0.0, 0.05 - Q.centroid_offset)
        + 2.591 * max(0.0, Q.girth - 0.074)
        - 8.707 * max(0.0, 0.088 - Q.girth)
        - 28.24 * max(0.0, 0.0038 - Q.width)
        - 85.77 * max(0.0, 0.0088 - Q.width)
        - 37.92 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 49.19 * max(0.0, Q.e2 - 0.062)
        - 0.04832 * max(0.0, Q.mass - 91.2)
        - 114.7 * max(0.0, Q.width - 0.018)
        - 11030.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        + 27.56 * max(0.0, Q.C2 - 0.065)
        - 24.63 * max(0.0, 0.14 - Q.girth)
        - 0.02124 * max(0.0, Q.sum_pt - 980.0)
        - 1.521 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        + 0.4849 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 0.0001828 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        + 8.29 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        - 10.08 * max(0.0, 0.091 - Q.girth)
        + 59.25 * max(0.0, 0.014 - Q.girth2)
        + 56.4 * max(0.0, Q.lam1 - 0.0075)
        + 0.02031 * max(0.0, 80.4 - Q.mass)
        + 1.199 * max(0.0, 0.18 - Q.max_dr)
        - 15.18 * max(0.0, 0.0075 - Q.width)
        - 0.1343 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
        + 766.3 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 111.7 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
        - 30.41 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 58.47 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 4.07 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
        + 80.79 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 114.7 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        + 68.31 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
        + 0.7922 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        - 0.5201 * max(0.0, Q.LHA - 0.34)
        + 10.07 * max(0.0, 0.041 - Q.e2)
        - 6.33 * max(0.0, 0.0065 - Q.lam1)
        + 2.524 * max(0.0, 0.0081 - Q.lam1)
        + 25.86 * max(0.0, 0.0069 - Q.width)
        - 0.2436 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
        - 2.351 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 514.1 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['W'] - s['t'] > -3.7964658737182617:
        if s['q'] - s['Z'] > 0.2755417674779892:
            if s['g'] - s['q'] > -0.01865352876484394:
                if s['g'] - s['W'] > 0.10182897374033928:
                    if s['g'] - s['t'] > -0.1738579049706459:
                        if s['g'] - s['q'] > 0.10682982206344604:
                            if Q.z_7 > 0.015656876377761364:
                                if s['g'] - s['W'] > 0.35275110602378845:
                                    if s['g'] - s['q'] > 0.17716870456933975:
                                        if s['g'] - s['t'] > -0.005794801516458392:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 29.2734375:
                                                if Q.max_dr > 0.17794394493103027:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.004937478573992848:
                                            if Q.log_sum_pt > 6.82664680480957:
                                                if Q.sum_pt > 990.0078125:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 24.2109375:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009003427810966969:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.552106618881226:
                                                if Q.log_sum_pt > 6.593536853790283:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0022721632849425077:
                                                        if Q.girth2 > 0.0001324503537034616:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > 0.1381344199180603:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.0975954532623291:
                                                    if Q.mass > 8.389962673187256:
                                                        if Q.sum_pt_top5 > 455.5:
                                                            if s['g'] - s['q'] > 0.13776962459087372:
                                                                if Q.centroid_offset > 0.003981504589319229:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 19.035673141479492:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0032070981105789542:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.04620402492582798:
                                        if s['g'] - s['Z'] > 1.0990195274353027:
                                            if Q.mass_over_sum_pt > 0.06049175001680851:
                                                if Q.D2 > 1.0624821186065674:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.21527913957834244:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.018526473082602024:
                                                    if s['q'] - s['W'] > -0.21930845826864243:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 2.531075358390808:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.019246895797550678:
                                                if s['g'] - s['Z'] > 0.6980211138725281:
                                                    if Q.LHA > 0.18943990021944046:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > 1.3667917847633362:
                                    return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 56% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.007028921507298946:
                                if Q.log_sum_pt > 6.793343544006348:
                                    if Q.log_sum_pt > 6.901100397109985:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.6722461581230164:
                                        if s['q'] - s['Z'] > 2.287893056869507:
                                            if Q.LHA > 0.1318579688668251:
                                                if Q.e2 > 0.02077472396194935:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 0.004339598352089524:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007753997575491667:
                                                    if Q.log_sum_pt > 6.681853532791138:
                                                        if Q.planar_flow > 0.20839455723762512:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 0.3165893852710724:
                                            if Q.z_7 > 0.03748298995196819:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.04208240285515785:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.627483129501343:
                                    if Q.sum_pt > 914.90625:
                                        if Q.sum_pt > 984.09375:
                                            if Q.z_7 > 0.017306839115917683:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 2.6689258813858032:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.641202449798584:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 4.31196141242981:
                                                if s['g'] - s['q'] > 0.04401994310319424:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.0019997863564640284:
                                        if s['g'] - s['q'] > 0.06403876841068268:
                                            if Q.centroid_offset > 0.005198847968131304:
                                                if Q.sum_pt > 689.421875:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.3743336945772171:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 745.859375:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.022265903651714325:
                                                        if Q.log_sum_pt > 6.5386598110198975:
                                                            if s['W'] - s['t'] > 1.7932992577552795:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if s['Z'] - s['t'] > -0.6596368551254272:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0029420825885608792:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 1.3917041420936584:
                                                if s['g'] - s['q'] > 0.03513113409280777:
                                                    if Q.sum_pt_top5 > 551.578125:
                                                        if Q.LHA > 0.10650953277945518:
                                                            if Q.log_sum_pt > 6.577860593795776:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 9.33369779586792:
                                            if Q.sum_pt > 643.34375:
                                                if Q.log_sum_pt > 6.590235233306885:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 16.456817626953125:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > 2.0312435626983643:
                                                if Q.C2 > 0.008520237635821104:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.4232007712125778:
                            if Q.tau32 > 0.3881451338529587:
                                if s['g'] - s['Z'] > 2.27314031124115:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.39458249509334564:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                return 't'   # 90% of the training jets here get this class from the formula
                        else:
                            return 't'   # 99% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['W'] > -0.3582760691642761:
                        if s['g'] - s['Z'] > 1.0449076294898987:
                            if Q.width > 0.0035795990843325853:
                                if s['g'] - s['t'] > -0.26840172708034515:
                                    if Q.max_dr > 0.08148464560508728:
                                        if Q.centroid_offset > 0.016901780851185322:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 88% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 1.32059645652771:
                                    if Q.max_dr > 0.09826997295022011:
                                        if Q.pt_7 > 48.78125:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00020904345728922635:
                                        if Q.LHA > 0.21107187122106552:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1147693581879139:
                                            if Q.planar_flow > 0.09226306527853012:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 36.234375:
                                if Q.mass > 34.12864303588867:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > 2.7681444883346558:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -0.18721126019954681:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 62% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > 0.0611904114484787:
                                    if Q.LHA > 0.21985632181167603:
                                        return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 713.6953125:
                                            return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.011102923192083836:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 60% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['t'] > 0.03726762533187866:
                            if s['g'] - s['W'] > -0.6619204580783844:
                                if s['g'] - s['Z'] > 0.9679013192653656:
                                    if Q.girth2 > 0.00339632376562804:
                                        return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.18391569703817368:
                                            if s['q'] - s['Z'] > 0.5243104696273804:
                                                if s['g'] - s['q'] > 0.3429102152585983:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.01617681421339512:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 93% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            return 't'   # 82% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.2637290507555008:
                    if s['g'] - s['q'] > -0.09803441911935806:
                        if Q.log_sum_pt > 6.901548385620117:
                            if Q.z_7 > 0.016270224004983902:
                                return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.012094365432858467:
                                if Q.sum_pt > 805.765625:
                                    if Q.pt_7 > 33.109375:
                                        return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 40% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.024404974654316902:
                                        if Q.z_7 > 0.03686579689383507:
                                            return 't'   # 39% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > -0.06546885147690773:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.04132585972547531:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 617.375:
                                    if Q.mass_over_sum_pt > 0.010964853689074516:
                                        if Q.sum_pt > 898.078125:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.06516432389616966:
                                                if s['g'] - s['W'] > 1.5294021964073181:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 803.734375:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 2.298215627670288:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 6.285428753471933e-05:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.06569939106702805:
                                            if Q.log_sum_pt > 6.722253084182739:
                                                if Q.sum_pt > 912.4375:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 3.28860533045372e-05:
                                                if s['g'] - s['q'] > -0.04974670521914959:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > -0.046405989676713943:
                                                    if Q.sum_pt_top5 > 705.125:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 832.296875:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['Z'] > 1.7948682308197021:
                                                                return 'q'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.0020737143931910396:
                                        if s['q'] - s['W'] > 1.5050881505012512:
                                            if Q.log_sum_pt > 6.659754514694214:
                                                if s['g'] - s['q'] > -0.0387257169932127:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.0554056242108345:
                                                if Q.D2 > 1.1746038794517517:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.08007804676890373:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 8.809406757354736:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.604346036911011:
                                                if s['g'] - s['W'] > 1.9805539846420288:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 39.697731018066406:
                            if s['q'] - s['t'] > 0.3814720958471298:
                                if s['q'] - s['W'] > 1.2217952013015747:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.253810852766037:
                                        if Q.LHA > 0.12503038346767426:
                                            if s['W'] - s['t'] > 1.084808886051178:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2437506914138794:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 2.823507384164259e-05:
                                                if Q.sum_pt > 1003.392578125:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0021483462769538164:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -0.21020061522722244:
                                    if Q.sum_pt_top5 > 697.4375:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.991500377655029:
                                if s['g'] - s['q'] > -0.5884493291378021:
                                    if Q.sum_pt_top5 > 1110.03125:
                                        return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 1.5626528859138489:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.3987182229757309:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > -0.8067536652088165:
                                        if Q.max_dr > 0.021531441248953342:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > 3.6193350553512573:
                                                if Q.centroid_offset > 0.0010538292699493468:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 100% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -0.08314025402069092:
                                    if s['q'] - s['Z'] > 0.6602312922477722:
                                        if Q.log_sum_pt > 6.6235878467559814:
                                            if s['g'] - s['q'] > -0.17629405856132507:
                                                if Q.sum_pt > 1007.4375:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 7.31214604456909e-05:
                                                        if s['q'] - s['Z'] > 2.257683753967285:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > 0.8708655834197998:
                                                    if Q.log_sum_pt > 6.951732397079468:
                                                        if s['g'] - s['q'] > -0.40854473412036896:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > -0.5168715417385101:
                                                                if Q.centroid_offset > 0.003332086489535868:
                                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0012836733949370682:
                                                        if Q.lam2 > 5.393726496549789e-05:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.5704557299613953:
                                                            if Q.width > 0.00038605110603384674:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.007132156984880567:
                                                if Q.centroid_offset > 0.0019061132334172726:
                                                    if s['q'] - s['W'] > 0.40245501697063446:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 673.953125:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 35% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 8.089913368225098:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 728.1875:
                                                            if Q.pt_7 > 38.828125:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.019118073396384716:
                                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0021985460771247745:
                                                    if Q.log_sum_pt > 6.571561098098755:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.003212842857465148:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 2.3533434867858887:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 729.53125:
                                                        if s['W'] - s['Z'] > -0.050975725054740906:
                                                            if Q.centroid_offset > 0.001916632812935859:
                                                                return 'q'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 37% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.015330298338085413:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.4018712490797043:
                                            if Q.C2 > 0.03394652158021927:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 3.0542585849761963:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > 0.4515448659658432:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.00446813297457993:
                                                        if Q.lam2 > 5.9068919654237106e-05:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['Z'] > 0.9267567694187164:
                        if Q.width > 0.002489391015842557:
                            if s['W'] - s['t'] > -0.1638856679201126:
                                if s['q'] - s['Z'] > 1.6330047845840454:
                                    if s['Z'] - s['t'] > -1.1631152629852295:
                                        return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.015517865307629108:
                                if Q.girth2 > 0.0014878867659717798:
                                    return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 4.365430140751414e-05:
                                        return 'W'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.22947604209184647:
                                    return 'W'   # 77% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 1.3027785420417786:
                                        if s['g'] - s['q'] > -0.08348310738801956:
                                            return 'q'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00019621593673946336:
                                                if Q.centroid_offset > 0.010877433232963085:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > 1.226474940776825:
                                            if s['g'] - s['q'] > -0.16256865113973618:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > 0.07950608059763908:
                            if Q.max_dr > 0.24212562292814255:
                                return 'W'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.018437865190207958:
                                    return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.2917037159204483:
                                        if s['W'] - s['Z'] > 0.6151149868965149:
                                            return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.8575255274772644:
                                                if s['q'] - s['W'] > 0.13920139521360397:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.04210079833865166:
                                            return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.width > 0.002207112149335444:
                                if s['W'] - s['t'] > -0.14415432512760162:
                                    return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.015220357105135918:
                                    if Q.planar_flow > 0.25535865128040314:
                                        return 'W'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.016941212117671967:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2_sq > 0.0012558051967062056:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 9.034606773639098e-05:
                                                    if Q.lam1 > 0.0011525824666023254:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 0.721196323633194:
                                        if Q.sum_pt > 818.2734375:
                                            if Q.LHA > 0.15671805292367935:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.1939697489142418:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.001726847025565803:
                                                if Q.girth > 0.025578641332685947:
                                                    if s['q'] - s['t'] > 1.674648404121399:
                                                        if s['q'] - s['Z'] > 0.5141471028327942:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.23342394828796387:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.03833678737282753:
                if s['g'] - s['W'] > 0.17036886513233185:
                    if s['g'] - s['Z'] > 0.6344333589076996:
                        if s['g'] - s['t'] > -0.11798443272709846:
                            if s['g'] - s['W'] > 0.4298621267080307:
                                if Q.D2 > 0.7348257303237915:
                                    if s['Z'] - s['t'] > 2.601183295249939:
                                        if s['g'] - s['W'] > 0.6113587617874146:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02022731676697731:
                                                if Q.pt_7 > 49.671875:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.9986023306846619:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.22115660458803177:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 2.504437208175659:
                                    if Q.centroid_offset > 0.018514161929488182:
                                        if s['g'] - s['Z'] > 0.7716341316699982:
                                            if s['Z'] - s['t'] > 2.7286239862442017:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.7982713878154755:
                                        if s['g'] - s['q'] > 1.1937179565429688:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.02257565502077341:
                                                if s['g'] - s['W'] > 0.2614527493715286:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.016854635439813137:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 7.084390745148994e-05:
                                            if Q.max_dr > 0.07821669802069664:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                        else:
                            return 't'   # 87% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top5 > 572.21875:
                            if Q.sum_pt > 981.203125:
                                return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.487448051571846:
                                    if s['W'] - s['t'] > 2.5373703241348267:
                                        if Q.girth2 > 0.0003927635698346421:
                                            if Q.lam1 > 0.0006117938319221139:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 658.0:
                                                return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > 0.31049440801143646:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 60% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -1.0120068192481995:
                                        if Q.sum_pt > 699.65625:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 38% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > -0.3439011722803116:
                                if Q.D2 > 0.7467233538627625:
                                    if s['g'] - s['W'] > 0.3112923353910446:
                                        if Q.centroid_offset > 0.039009423926472664:
                                            return 'W'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.00572377978824079:
                                            if Q.centroid_offset > 0.03689023479819298:
                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 0.9819343984127045:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 72% of the training jets here get this class from the formula
                            else:
                                return 't'   # 78% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.1767563670873642:
                        if s['g'] - s['W'] > -0.24279454350471497:
                            if s['g'] - s['q'] > 1.0520787239074707:
                                if s['g'] - s['Z'] > 0.9148944020271301:
                                    if Q.e2_sq > 0.003315152949653566:
                                        if Q.centroid_offset > 0.015387403778731823:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.0908760353922844:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 1.603143334388733:
                                        if s['g'] - s['W'] > -0.045428114011883736:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 892.375:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 26.763049125671387:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0250529944896698:
                                            if Q.LHA > 0.20992814004421234:
                                                if Q.log_sum_pt > 6.3689374923706055:
                                                    if Q.mass > 17.710275650024414:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 24.13597011566162:
                                                        if Q.C2 > 0.04352873936295509:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 346.171875:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if s['Z'] - s['t'] > 2.0119134187698364:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.878793001174927:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if s['Z'] - s['t'] > -0.16494249552488327:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.19912602752447128:
                                    if s['g'] - s['Z'] > 0.6798660755157471:
                                        if s['g'] - s['Z'] > 1.0124496221542358:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.03652854263782501:
                                                if s['g'] - s['W'] > 0.11906731873750687:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 727.890625:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.918431997299194:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 693.59375:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.19761501252651215:
                                                    if s['g'] - s['W'] > -0.004353032214567065:
                                                        if Q.sum_pt > 572.421875:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.4402309656143188:
                                                                return 'g'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.18331778794527054:
                                                        if Q.log_sum_pt > 6.506472587585449:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 93% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.15866537392139435:
                                if s['W'] - s['Z'] > 0.3590271770954132:
                                    if s['g'] - s['Z'] > 0.898365318775177:
                                        if Q.LHA > 0.2035617008805275:
                                            if Q.log_sum_pt > 6.778855085372925:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.017135399393737316:
                                                    if Q.planar_flow > 0.47435738146305084:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > -0.4259497821331024:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 33.25420379638672:
                                                if Q.planar_flow > 0.10254593938589096:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -0.6175082325935364:
                                            if Q.log_sum_pt > 6.814131259918213:
                                                if s['q'] - s['t'] > 1.6011318564414978:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > 2.066915273666382:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03055759984999895:
                                                    if Q.LHA > 0.21170533448457718:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 39% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > 0.47484883666038513:
                                                if s['W'] - s['t'] > 0.38114024698734283:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.2637655735015869:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 75.35129165649414:
                                                    if Q.width > 0.005884301150217652:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.16500592976808548:
                                        if Q.girth2 > 0.0034189592115581036:
                                            if Q.centroid_offset > 0.015718940645456314:
                                                if Q.lam1 > 0.0038970039458945394:
                                                    if Q.lam2 > 0.00013076110190013424:
                                                        if Q.D2 > 2.0730572938919067:
                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.029042408801615238:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.16918523609638214:
                                                                if s['W'] - s['t'] > 1.9671297073364258:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.022882044315338135:
                                                        if Q.max_dr > 0.17747624963521957:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1801086813211441:
                                                            if Q.girth > 0.045925868675112724:
                                                                if Q.lam2 > 8.353182056453079e-05:
                                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if s['q'] - s['t'] > -0.3016601651906967:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.04698743671178818:
                                                    if s['g'] - s['t'] > 0.05837186984717846:
                                                        if Q.eccentricity > 0.9863003492355347:
                                                            if Q.girth2 > 0.004068077774718404:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.0879785604774952:
                                                            if Q.pt_7 > 17.4921875:
                                                                if Q.lam1 > 0.003759061684831977:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 994.3671875:
                                            if s['q'] - s['t'] > -1.363108515739441:
                                                if Q.eccentricity > 0.9745284616947174:
                                                    if Q.e2 > 0.023236527107656002:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.028488523326814175:
                                                if Q.max_dr > 0.05874948389828205:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -1.4178138375282288:
                                                        if Q.C2 > 0.011780120432376862:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.12050729990005493:
                                    if Q.z_dr_0p05_0p1 > 0.05776963196694851:
                                        if s['q'] - s['t'] > -1.8000919222831726:
                                            if s['g'] - s['q'] > 0.7061577439308167:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 61% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 86% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.06269155815243721:
                            if s['W'] - s['t'] > 0.04824674315750599:
                                if Q.max_dr > 0.16253487765789032:
                                    if Q.width > 0.0031521024648100138:
                                        if Q.centroid_offset > 0.015136554837226868:
                                            if Q.girth2 > 0.0035936516942456365:
                                                if Q.eccentricity > 0.9726542532444:
                                                    if Q.LHA > 0.22335103899240494:
                                                        if Q.max_dr > 0.16951961815357208:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['Z'] > -2.5360283851623535:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 2.235801577568054:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.0982366800308228:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.2440309301018715:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.04099791310727596:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.04245717450976372:
                                                    if Q.LHA > 0.21922902762889862:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['q'] > -0.38831909000873566:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 0.026550812646746635:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.049997176975011826:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 59.302446365356445:
                                                if Q.C2 > 0.04277248866856098:
                                                    if s['W'] - s['Z'] > 0.11929795145988464:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > -2.0009725689888:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 0.1414036750793457:
                                                    if Q.C2 > 0.0481959767639637:
                                                        if Q.planar_flow > 0.055402571335434914:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.06022433377802372:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.03692180849611759:
                                                        if Q.e2 > 0.029753725044429302:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['t'] > -1.3787579536437988:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['t'] > -0.14701232314109802:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.04614304564893246:
                                                return 'Z'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02696342673152685:
                                        if Q.max_dr > 0.07825448364019394:
                                            if s['g'] - s['Z'] > 0.1027999222278595:
                                                return 'W'   # 49% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.019582362845540047:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00012975965364603326:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > -1.0786033868789673:
                                                if Q.planar_flow > 0.03477292321622372:
                                                    if Q.log_sum_pt > 6.478365659713745:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 26.29261016845703:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > -0.018085980787873268:
                                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -1.3545197248458862:
                                                            if Q.eccentricity > 0.9657204151153564:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 5.253797553450568e-06:
                                            if Q.centroid_offset > 0.008594107814133167:
                                                if Q.log_sum_pt > 6.957183837890625:
                                                    return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['W'] > -0.9942258596420288:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2_sq > 0.00018959544831886888:
                                                            if s['g'] - s['t'] > -1.1531612873077393:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if s['q'] - s['t'] > -1.64344984292984:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.1154857836663723:
                                                                        if Q.z_dr_0p05_0p1 > 0.19613178819417953:
                                                                            if Q.max_dr > 0.1282401904463768:
                                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['Z'] > -1.260152816772461:
                                                                if Q.eccentricity > 0.9828457534313202:
                                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.08026392012834549:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.5717560052871704:
                                                        if Q.max_dr > 0.11873610690236092:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 71.22565841674805:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['Z'] - s['t'] > 2.63531756401062:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > -0.28883880376815796:
                                    return 'W'   # 45% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.2621563822031021:
                                if s['g'] - s['t'] > -1.3902496099472046:
                                    if s['Z'] - s['t'] > 0.05864446796476841:
                                        if Q.sum_pt > 993.78125:
                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 29.564186096191406:
                                                if Q.max_dr > 0.16604331135749817:
                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > -1.7964152693748474:
                                                        if Q.max_dr > 0.14845959842205048:
                                                            if s['q'] - s['W'] > -1.9094334244728088:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.10537170991301537:
                                                            if Q.z_dr_0p05_0p1 > 0.550215482711792:
                                                                if s['g'] - s['q'] > 1.035780668258667:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 50% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > 0.0038927207933738828:
                                        if Q.max_dr > 0.10610569640994072:
                                            if s['q'] - s['t'] > -1.5960319638252258:
                                                if Q.max_dr > 0.16220049560070038:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.39904893934726715:
                                                        if Q.max_dr > 0.13292551040649414:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.5101360082626343:
                                                    if Q.max_dr > 0.12048646435141563:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 32.53125:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0348032396286726:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > -0.8859379589557648:
                                    if Q.eccentricity > 0.985616534948349:
                                        if Q.lam1 > 0.003336543682962656:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.06568575277924538:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -0.4510984271764755:
                                                    if Q.lam1 > 0.000863761903019622:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9893631637096405:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -0.68125981092453:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['t'] > 0.06456085108220577:
                                            return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if s['Z'] - s['t'] > 0.07038848102092743:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.06734319403767586:
                                        if Q.girth2 > 0.003358620684593916:
                                            if Q.max_dr > 0.15804693847894669:
                                                if Q.centroid_offset > 0.011125791817903519:
                                                    if Q.tau21 > 0.09528013318777084:
                                                        if s['g'] - s['t'] > -0.13906437903642654:
                                                            if Q.C2 > 0.03811338543891907:
                                                                if Q.lam2 > 0.0002660078462213278:
                                                                    if Q.D2 > 2.2675124406814575:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.22269205003976822:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.015403228346258402:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 867.578125:
                                                        if Q.C2 > 0.044146375730633736:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.19958177208900452:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1795213371515274:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.031633540987968445:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.7505864799022675:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.04052239470183849:
                                                if Q.max_dr > 0.17520925402641296:
                                                    if Q.LHA > 0.22786254435777664:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.03677411191165447:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.12636161595582962:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -1.435871422290802:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 0.8573807775974274:
                                                    return 'Z'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 0.2645065635442734:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04600055143237114:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02319545578211546:
                                            if Q.planar_flow > 0.060152556747198105:
                                                if s['q'] - s['Z'] > -1.0910701751708984:
                                                    if Q.D2 > 1.629733681678772:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9835891723632812:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.12079774588346481:
                    if s['g'] - s['t'] > -0.0011291475966572762:
                        if s['g'] - s['Z'] > 0.3888169080018997:
                            if s['g'] - s['t'] > 0.14615892618894577:
                                if s['g'] - s['Z'] > 0.6026102900505066:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 1.7217158079147339:
                                        if s['q'] - s['W'] > -0.7320866286754608:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.5355566442012787:
                                            if Q.centroid_offset > 0.035524824634194374:
                                                if s['W'] - s['t'] > -0.10890181735157967:
                                                    if Q.centroid_offset > 0.04298205301165581:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.01562982937321067:
                                                        if Q.log_sum_pt > 6.300585031509399:
                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > -0.6643056571483612:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 21.875595092773438:
                                    if s['q'] - s['W'] > 2.093517780303955:
                                        return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00014606349577661604:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 100% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > 1.4080457091331482:
                                if s['q'] - s['Z'] > -1.0457238554954529:
                                    if Q.C2 > 0.010838910937309265:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.289005383849144:
                                            if Q.sum_pt_top5 > 576.640625:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 68% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.021863672882318497:
                                    if Q.centroid_offset > 0.033651724457740784:
                                        if Q.max_dr > 0.10195978730916977:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.533334493637085:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.24647019058465958:
                                                    if Q.D2 > 1.639205813407898:
                                                        if s['W'] - s['t'] > 0.32091543078422546:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.24224106967449188:
                                            if Q.D2 > 0.6573048830032349:
                                                if Q.sum_pt > 693.484375:
                                                    if Q.max_dr > 0.04169490374624729:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > 0.484808012843132:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 44.96875:
                                                if Q.e2 > 0.00749882822856307:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 0.3963640630245209:
                                                    if Q.D2 > 1.8297352194786072:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.28672245144844055:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.1240963339805603:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.009705619420856237:
                                        if Q.sum_pt > 481.296875:
                                            if Q.mass > 24.77659320831299:
                                                if s['W'] - s['Z'] > -0.665604293346405:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6170291900634766:
                                                        return 'g'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.06596669554710388:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 38.12313461303711:
                                                return 'Z'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.e2_sq > 0.0009285556152462959:
                            if s['g'] - s['t'] > -0.18432259559631348:
                                return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 842.140625:
                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 97% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 87% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.06761506199836731:
                        if s['W'] - s['Z'] > -0.2685080021619797:
                            if Q.LHA > 0.25966599583625793:
                                if s['g'] - s['t'] > -1.0333108305931091:
                                    if Q.max_dr > 0.1603684350848198:
                                        if Q.e2 > 0.034122249111533165:
                                            if Q.LHA > 0.278792142868042:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.16605877131223679:
                                            if Q.sum_pt > 651.0625:
                                                if Q.mass > 72.11335372924805:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.15064917504787445:
                                                        if s['q'] - s['Z'] > -2.8081506490707397:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.2806273102760315:
                                                            if Q.max_dr > 0.12127896770834923:
                                                                if Q.LHA > 0.2940097600221634:
                                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1193847581744194:
                                                    if Q.LHA > 0.28965531289577484:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.027146821841597557:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 614.453125:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.10139264166355133:
                                                        if Q.planar_flow > 0.24129211157560349:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 509.609375:
                                                if Q.centroid_offset > 0.019043908454477787:
                                                    if Q.max_dr > 0.14454898983240128:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10760045796632767:
                                                        if Q.LHA > 0.2901061177253723:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > -0.2020658180117607:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.11545852199196815:
                                                    if Q.z_dr_0p05_0p1 > 0.5561702847480774:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13703925907611847:
                                                            if Q.e2 > 0.028281821869313717:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 5.145606701262295e-05:
                                                        if Q.z_dr_0p05_0p1 > 0.5953544676303864:
                                                            if Q.tau21 > 0.22237654030323029:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.10133268684148788:
                                        if s['W'] - s['Z'] > -0.10780385881662369:
                                            if s['q'] - s['t'] > -1.4818168878555298:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.10674953833222389:
                                                    if Q.z_dr_0p05_0p1 > 0.448223277926445:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.03525305911898613:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.07019027322530746:
                                                if Q.z_dr_0p05_0p1 > 0.47599610686302185:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.03468947298824787:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.017714088782668114:
                                                    if Q.max_dr > 0.14933834969997406:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > -1.9513543248176575:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9761695861816406:
                                            if s['W'] - s['Z'] > -0.15260348469018936:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.0973762609064579:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > -1.8416677713394165:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 100% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.8852770626544952:
                                    if Q.planar_flow > 0.06919309496879578:
                                        if Q.girth > 0.02714292611926794:
                                            if Q.girth2 > 0.0031523763900622725:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.029839995317161083:
                                                    if Q.D2 > 2.3939855098724365:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -0.4780232757329941:
                                                            return 'W'   # 47% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -0.6524316370487213:
                                                        if s['g'] - s['Z'] > -0.6127824783325195:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > -0.14561713486909866:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.011989965103566647:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.04699381999671459:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.152283675968647:
                                                if s['q'] - s['Z'] > -0.8095999658107758:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > 2.937796115875244:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 36% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > 1.452700436115265:
                                            if Q.D2 > 5.252820253372192:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > -0.07103052362799644:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -0.5296865701675415:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.006375353317707777:
                                                if Q.max_dr > 0.1567729264497757:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -0.4764809161424637:
                                                    if s['q'] - s['Z'] > -0.0842764861881733:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > 1.093785047531128:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 14.518572807312012:
                                        if Q.girth2 > 0.002983230398967862:
                                            if Q.max_dr > 0.1557973474264145:
                                                if Q.centroid_offset > 0.012613192666321993:
                                                    if Q.width > 0.003339587477967143:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.03954487666487694:
                                                            if Q.C2 > 0.03996146284043789:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.038231831043958664:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > -0.26708246767520905:
                                                            if Q.sum_pt > 945.34375:
                                                                if Q.z_dr_0p2_0p4 > 0.026478368788957596:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if s['W'] - s['Z'] > -0.10106129199266434:
                                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.17148347944021225:
                                                                if Q.LHA > 0.2327985242009163:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.034296989440918:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > -0.20885083824396133:
                                                    if s['Z'] - s['t'] > 1.5129145383834839:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.06031550467014313:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.038630517199635506:
                                                if Q.D2 > 0.6886676251888275:
                                                    if Q.max_dr > 0.0966075100004673:
                                                        if Q.max_dr > 0.17000453174114227:
                                                            if Q.girth > 0.040809642523527145:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 22.2578125:
                                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 4.719933531305287e-05:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.26023751497268677:
                                                    if s['q'] - s['t'] > 1.4575363993644714:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.05344313755631447:
                                                        if Q.centroid_offset > 0.027951596304774284:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['W'] > -0.46760718524456024:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.0337105356156826:
                                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.23296864330768585:
                                                            if Q.girth > 0.03341752476990223:
                                                                if Q.pt_7 > 23.15625:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00013763974857283756:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02408561483025551:
                                            if s['g'] - s['q'] > 2.0078293085098267:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > -1.0214159488677979:
                                                if Q.eccentricity > 0.979388564825058:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.007674029096961021:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.5064099729061127:
                                if Q.centroid_offset > 0.051349176093935966:
                                    if s['W'] - s['Z'] > -1.5531235933303833:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 2.167577862739563:
                                        if s['Z'] - s['t'] > 1.1914032101631165:
                                            if Q.z_dr_0p05_0p1 > 0.5343450307846069:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.03292000666260719:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04059807024896145:
                                            if Q.e2 > 0.006707916967570782:
                                                if Q.z_dr_0p05_0p1 > 0.25236572325229645:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0020683412440121174:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.004582476336508989:
                                                    if s['q'] - s['t'] > -0.14953535050153732:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 29.1171875:
                                                if Q.D2 > 0.6232326328754425:
                                                    if s['q'] - s['t'] > -0.08925987780094147:
                                                        if Q.log_sum_pt > 6.925236225128174:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > -0.11080687493085861:
                                                            if Q.e2 > 0.0036116804694756866:
                                                                if Q.max_dr > 0.10392643138766289:
                                                                    if Q.centroid_offset > 0.01924626249819994:
                                                                        if Q.z_dr_0p05_0p1 > 0.461354598402977:
                                                                            if s['g'] - s['Z'] > 0.017015055753290653:
                                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.04719538986682892:
                                                                if s['Z'] - s['t'] > 0.24280018359422684:
                                                                    if s['W'] - s['Z'] > -0.5229886174201965:
                                                                        if Q.z_dr_0p1_0p2 > 0.14397016167640686:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.007649356033653021:
                                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.14961912482976913:
                                                                            return 't'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > 0.2169213369488716:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 0.025972760282456875:
                                                    if Q.D2 > 2.7310948371887207:
                                                        if Q.sum_pt_top5 > 613.1875:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['t'] > 0.18854406476020813:
                                                                return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 32% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['W'] > 0.34015002846717834:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 33% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > -0.44510598480701447:
                                    if Q.LHA > 0.2600205987691879:
                                        if s['g'] - s['t'] > -0.7728597223758698:
                                            if Q.eccentricity > 0.9899959862232208:
                                                if Q.mass > 66.74995040893555:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13723506033420563:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.004941119579598308:
                                                            if Q.max_dr > 0.12020351737737656:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.11704879999160767:
                                                    if Q.LHA > 0.29067768156528473:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.18243297189474106:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > -1.465985119342804:
                                                        if Q.centroid_offset > 0.030034516006708145:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.0925341434776783:
                                                if Q.max_dr > 0.10112891718745232:
                                                    if s['q'] - s['t'] > -2.443232536315918:
                                                        if Q.e2_sq > 0.00461732386611402:
                                                            if Q.z_dr_0p05_0p1 > 0.33635464310646057:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.03473825752735138:
                                                                    if Q.LHA > 0.279847115278244:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.31600813567638397:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 6.838285844423808e-05:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > -0.5541587471961975:
                                            if s['g'] - s['t'] > 1.0286066830158234:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03406381234526634:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.2718176990747452:
                                                    if Q.lam1 > 0.003389306366443634:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 880.5859375:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 6.733515024185181:
                                                            if Q.lam1 > 0.0021616668673232198:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > 0.2518061399459839:
                                        if Q.centroid_offset > 0.053093595430254936:
                                            if s['g'] - s['W'] > 1.1932750344276428:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -0.7367745041847229:
                                                if Q.pt_7 > 22.7890625:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.029958883300423622:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0027587009826675057:
                                                    if s['W'] - s['Z'] > -0.6431060135364532:
                                                        if Q.max_dr > 0.12704607844352722:
                                                            if Q.e2 > 0.03407170996069908:
                                                                if Q.LHA > 0.27688321471214294:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.1437772512435913:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.28072308003902435:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.022534526884555817:
                                                                    if s['q'] - s['t'] > -1.1834216117858887:
                                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.9840021133422852:
                                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 0.3591621220111847:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.2741706371307373:
                                                                return 't'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 2.4112519025802612:
                                                        if s['g'] - s['W'] > 0.2223806232213974:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.5644839704036713:
                                                            if s['g'] - s['W'] > 0.10529770329594612:
                                                                if Q.sum_pt_top5 > 605.953125:
                                                                    if Q.centroid_offset > 0.04202929325401783:
                                                                        if Q.z_dr_0p05_0p1 > 0.5951957106590271:
                                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.05483083985745907:
                                            if Q.sum_pt > 526.4140625:
                                                if s['Z'] - s['t'] > 0.12926189601421356:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6437283754348755:
                                                        if Q.max_dr > 0.11740132048726082:
                                                            if Q.e2_sq > 0.006795764900743961:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.091223806142807:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.2942262142896652:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['t'] > -1.9557079076766968:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.03907141089439392:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.2580048441886902:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.18589642643928528:
                            if s['Z'] - s['t'] > -0.01267073955386877:
                                if s['W'] - s['t'] > -1.0568702816963196:
                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.e2_sq > 0.009019992779940367:
                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.22604206204414368:
                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -2.0147645473480225:
                                                if Q.max_dr > 0.11714726313948631:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.10014265775680542:
                                    if Q.e2_sq > 0.008854246232658625:
                                        if Q.tau21 > 0.169514998793602:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.7242770195007324:
                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_0 > 428.5:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > -0.31969335675239563:
                                                    return 't'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.034908194094896317:
                                                        if Q.girth > 0.07328140363097191:
                                                            if Q.lam1 > 0.00637223944067955:
                                                                return 't'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.3191012889146805:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.020649366080760956:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 32% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1114.0078125:
                                return 'Z'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.08683173730969429:
                                    return 't'   # 95% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 37% of the training jets here get this class from the formula
    else:
        if s['g'] - s['t'] > -0.018340615555644035:
            if s['g'] - s['t'] > 0.22807782143354416:
                if s['g'] - s['q'] > 0.05873514339327812:
                    if s['g'] - s['Z'] > 0.4186347723007202:
                        if s['Z'] - s['t'] > -7.4734787940979:
                            return 'g'   # 96% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > 0.8197393715381622:
                                return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 285.375:
                                    if Q.sum_pt_top5 > 404.84375:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 8.16905927658081:
                                            if Q.LHA > 0.34974658489227295:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.07144704088568687:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.09660867229104042:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > -0.894331306219101:
                            if Q.tau32 > 0.46387356519699097:
                                return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 91% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['q'] > -0.23623505979776382:
                        if Q.mass > 116.97656631469727:
                            return 'g'   # 61% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 56% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 84% of the training jets here get this class from the formula
            else:
                if s['g'] - s['q'] > -0.046011921018362045:
                    if s['g'] - s['Z'] > -0.3463671952486038:
                        if s['g'] - s['t'] > 0.11137091740965843:
                            if s['q'] - s['Z'] > 7.797520875930786:
                                if Q.z_7 > 0.06435386091470718:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 895.5625:
                                    return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0002568649797467515:
                                        if s['W'] - s['Z'] > -3.669652581214905:
                                            if s['q'] - s['W'] > 6.780165433883667:
                                                return 't'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.29191572964191437:
                                                if Q.max_dr > 0.1623298078775406:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.027802973054349422:
                                if s['q'] - s['Z'] > 8.75200366973877:
                                    return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.596463203430176:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -4.010431528091431:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 33.296875:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.16928310692310333:
                                    if s['W'] - s['Z'] > 0.7683823704719543:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.42529889941215515:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 58% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 6.3052685260772705:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 512.140625:
                        return 'q'   # 82% of the training jets here get this class from the formula
                    else:
                        return 't'   # 62% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > 0.09425321593880653:
                if s['Z'] - s['t'] > 0.4009745419025421:
                    if s['q'] - s['Z'] > -0.7326478064060211:
                        return 'q'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 93.33248138427734:
                            return 'Z'   # 46% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > 0.5967465341091156:
                                return 'Z'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.027842726558446884:
                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.e2_sq > 0.005721465218812227:
                                        return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 88% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.03791722282767296:
                        if s['Z'] - s['t'] > 0.18240711092948914:
                            return 'Z'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt > 0.09325329214334488:
                                return 'Z'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.12840263545513153:
                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 67% of the training jets here get this class from the formula
                    else:
                        if Q.e2_sq > 0.005765512818470597:
                            if Q.tau21 > 0.32123246788978577:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p2_0p4 > 0.11247017234563828:
                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 81% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 60% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.22369925677776337:
                    if s['q'] - s['t'] > -0.03588626906275749:
                        if Q.centroid_offset > 0.016633734107017517:
                            if s['q'] - s['t'] > 0.23665709793567657:
                                return 'q'   # 95% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 8.282254219055176:
                                    return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 692.875:
                                if s['q'] - s['t'] > 0.26368097960948944:
                                    return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 88% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 0.13880394399166107:
                            if Q.lam1 > 0.02376452274620533:
                                return 'q'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.007877558469772339:
                                    return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 50% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 78% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.2823885381221771:
                        if Q.pt_7 > 26.5234375:
                            if Q.lam1 > 0.030120261013507843:
                                if Q.centroid_offset > 0.06414608657360077:
                                    if Q.tau32 > 0.5072360634803772:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 86.6141586303711:
                                        if Q.max_dr > 0.24693495780229568:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.11254659295082092:
                                    if Q.sum_pt_top5 > 256.890625:
                                        if Q.mass > 51.31826972961426:
                                            if Q.lam2 > 0.0008212229877244681:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.7090043723583221:
                                                    if Q.C2 > 0.06161598488688469:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.08280472457408905:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 37.43135452270508:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.32266999781131744:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.017367269843816757:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 603.109375:
                                        return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.056880950927734375:
                                            if Q.z_dr_0p1_0p2 > 0.2545033246278763:
                                                if Q.e2 > 0.050084253773093224:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.08029394596815109:
                                                if Q.planar_flow > 0.4534817785024643:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p1_0p2 > 0.7041775584220886:
                                return 't'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.181002855300903:
                                    return 't'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.33840321004390717:
                                        if Q.pt_0 > 65.625:
                                            return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.17721201479434967:
                            if Q.tau32 > 0.38075074553489685:
                                if s['Z'] - s['t'] > 0.02370583824813366:
                                    if Q.centroid_offset > 0.014316588640213013:
                                        if Q.log_sum_pt > 6.403975009918213:
                                            if s['g'] - s['t'] > -1.5051482319831848:
                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.04012482427060604:
                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 62% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > -0.4945147782564163:
                                if Q.lam1 > 0.02618939708918333:
                                    if Q.max_dr > 0.25101786851882935:
                                        return 'q'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 4.07715368270874:
                                        return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 91% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.5158316791057587:
                                    if Q.pt_7 > 25.3984375:
                                        if Q.girth2 > 0.03754151239991188:
                                            if Q.centroid_offset > 0.0635228306055069:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 452.3046875:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.26186713576316833:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 100% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
