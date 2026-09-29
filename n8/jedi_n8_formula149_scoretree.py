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

Test set (50,000 jets): accuracy 64.33% (the formula: 64.64%); same class as the formula for 93.18% of jets.  36 leaves, depth 8.
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
    return (6.62
        + 21.66 * max(0.0, 0.033 - Q.centroid_offset)
        - 9.997 * max(0.0, 0.013 - Q.girth2)
        + 0.05459 * max(0.0, 29.0 - Q.mass)
        + 0.00965 * max(0.0, 74.0 - Q.mass)
        + 0.004829 * max(0.0, Q.sum_pt - 880.0)
        + 450.2 * max(0.0, 0.005 - Q.width)
        - 1985.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 246.4 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 1.172 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.03812 * max(0.0, Q.pt_7 - 32.0)
        - 55.51 * max(0.0, 0.01 - Q.width)
        + 6487.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 59.69 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 1526.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 8.379 * max(0.0, Q.LHA - 0.13)
        + 119.8 * max(0.0, 0.005 - Q.lam1)
        + 0.00282 * max(0.0, 780.0 - Q.sum_pt)
        + 4.423 * max(0.0, Q.LHA - 0.27)
        - 17.18 * max(0.0, Q.centroid_offset - 0.013)
        - 0.4003 * max(0.0, 0.047 - Q.e2)
        - 55.51 * max(0.0, 0.01 - Q.girth2)
        - 18.65 * max(0.0, Q.lam1 - 0.016)
        - 7.378 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        - 13.8 * max(0.0, Q.C2 - 0.063)
        + 0.01265 * max(0.0, 55.0 - Q.mass)
        - 4.122 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 1.566 * max(0.0, 0.27 - Q.tau21)
        + 181.0 * max(0.0, Q.width - 0.0016)
        + 2.889 * max(0.0, 0.04 - Q.e2)
        + 22.19 * max(0.0, Q.log_sum_pt - 6.9)
        - 47.91 * max(0.0, 0.035 - Q.z_7)
        - 15.41 * max(0.0, 0.066 - Q.z_7)
        - 6.011 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        - 33480.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 664.7 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 13.07 * max(0.0, Q.girth - 0.09)
        - 55.16 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 0.4379 * Q.max_dr
        - 168.4 * max(0.0, 0.012 - Q.width)
        + 0.1027 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 0.006176 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 2.259 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 18.91 * max(0.0, 0.038 - Q.e2)
        - 400.8 * max(0.0, 0.0011 - Q.e2_sq)
        + 29.33 * max(0.0, 0.087 - Q.girth)
        + 228.5 * max(0.0, Q.girth2 - 0.0046)
        + 0.04609 * max(0.0, Q.mass - 80.4)
        + 2.687 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 7.58 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 627.4 * max(0.0, 0.00069 - Q.width)
        + 222.1 * max(0.0, 0.0056 - Q.width)
        - 1188.0 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 353.0 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 25.45 * max(0.0, 0.063 - Q.girth)
        - 413.4 * max(0.0, 0.0049 - Q.width)
        + 11580.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 1.432 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 7349.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        - 5.174 * max(0.0, 0.019 - Q.centroid_offset)
        + 354.6 * max(0.0, Q.lam2 - 0.0015)
        - 2.242 * max(0.0, Q.log_sum_pt - 6.3)
        - 0.0816 * max(0.0, 25.0 - Q.mass)
        + 298.6 * max(0.0, 0.0065 - Q.width)
        + 0.02644 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 10820.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 0.1638 * max(0.0, Q.C2 - 0.059)
        + 4.591 * max(0.0, 0.037 - Q.e2)
        - 271.0 * max(0.0, 0.0018 - Q.girth2)
        - 526.4 * max(0.0, Q.lam2 - 0.00016)
        + 1.988 * max(0.0, 6.3 - Q.log_sum_pt)
        - 32.38 * max(0.0, 0.05 - Q.centroid_offset)
        + 3.979 * max(0.0, Q.girth - 0.074)
        - 39.76 * max(0.0, 0.088 - Q.girth)
        - 151.6 * max(0.0, 0.0038 - Q.width)
        - 200.1 * max(0.0, 0.0088 - Q.width)
        - 66.24 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 9.56 * max(0.0, Q.e2 - 0.062)
        - 0.01696 * max(0.0, Q.mass - 91.2)
        + 1.995 * max(0.0, Q.width - 0.018)
        - 6363.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        + 12.18 * max(0.0, Q.C2 - 0.065)
        + 6.029 * max(0.0, 0.14 - Q.girth)
        - 0.01849 * max(0.0, Q.sum_pt - 980.0)
        - 18.05 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.1148 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 5.465e-05 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        + 22.45 * max(0.0, 0.091 - Q.girth)
        - 35.36 * max(0.0, 0.014 - Q.girth2)
        - 284.3 * max(0.0, Q.lam1 - 0.0075)
        - 0.03549 * max(0.0, 80.4 - Q.mass)
        + 1.036 * max(0.0, 0.18 - Q.max_dr)
        + 162.3 * max(0.0, 0.0075 - Q.width)
        + 2330.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 19.66 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 29.67 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 190.3 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 72.68 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        - 4.971 * max(0.0, Q.LHA - 0.34)
        - 21.0 * max(0.0, 0.041 - Q.e2)
        + 230.4 * max(0.0, 0.0065 - Q.lam1)
        + 211.7 * max(0.0, 0.0081 - Q.lam1)
        - 128.6 * max(0.0, 0.0069 - Q.width)
        + 443.2 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_q(Q):
    return (6.739
        + 16.42 * max(0.0, 0.033 - Q.centroid_offset)
        - 13.14 * max(0.0, 0.013 - Q.girth2)
        + 0.0323 * max(0.0, 29.0 - Q.mass)
        + 0.01035 * max(0.0, 74.0 - Q.mass)
        + 0.001825 * max(0.0, Q.sum_pt - 880.0)
        + 557.4 * max(0.0, 0.005 - Q.width)
        - 1698.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 83.35 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 1.24 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.006645 * max(0.0, Q.pt_7 - 32.0)
        - 21.41 * max(0.0, 0.01 - Q.width)
        + 3535.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        + 114.1 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        - 992.6 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 6.375 * max(0.0, Q.LHA - 0.13)
        - 171.9 * max(0.0, 0.005 - Q.lam1)
        - 0.002671 * max(0.0, 780.0 - Q.sum_pt)
        + 5.587 * max(0.0, Q.LHA - 0.27)
        - 22.34 * max(0.0, Q.centroid_offset - 0.013)
        + 33.24 * max(0.0, 0.047 - Q.e2)
        - 21.41 * max(0.0, 0.01 - Q.girth2)
        - 35.99 * max(0.0, Q.lam1 - 0.016)
        - 9.452 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 18.74 * max(0.0, Q.C2 - 0.063)
        + 0.008539 * max(0.0, 55.0 - Q.mass)
        - 9.138 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 1.435 * max(0.0, 0.27 - Q.tau21)
        - 236.3 * max(0.0, Q.width - 0.0016)
        - 3.985 * max(0.0, 0.04 - Q.e2)
        - 4.3 * max(0.0, Q.log_sum_pt - 6.9)
        + 7.021 * max(0.0, 0.035 - Q.z_7)
        + 16.55 * max(0.0, 0.066 - Q.z_7)
        - 42.68 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        - 16730.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        - 207.3 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 15.26 * max(0.0, Q.girth - 0.09)
        - 40.72 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        + 0.5338 * Q.max_dr
        - 184.4 * max(0.0, 0.012 - Q.width)
        + 0.08196 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 0.003873 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 1.606 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 2.797 * max(0.0, 0.038 - Q.e2)
        - 451.0 * max(0.0, 0.0011 - Q.e2_sq)
        - 16.21 * max(0.0, 0.087 - Q.girth)
        + 135.1 * max(0.0, Q.girth2 - 0.0046)
        + 0.01657 * max(0.0, Q.mass - 80.4)
        - 6.113 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 17.19 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 454.1 * max(0.0, 0.00069 - Q.width)
        + 288.2 * max(0.0, 0.0056 - Q.width)
        - 445.1 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 169.5 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 35.69 * max(0.0, 0.063 - Q.girth)
        - 252.0 * max(0.0, 0.0049 - Q.width)
        + 6867.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 1.399 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 8433.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 16.0 * max(0.0, 0.019 - Q.centroid_offset)
        + 383.2 * max(0.0, Q.lam2 - 0.0015)
        - 2.959 * max(0.0, Q.log_sum_pt - 6.3)
        - 0.06619 * max(0.0, 25.0 - Q.mass)
        + 396.6 * max(0.0, 0.0065 - Q.width)
        + 0.0373 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 13680.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        - 24.91 * max(0.0, Q.C2 - 0.059)
        - 9.539 * max(0.0, 0.037 - Q.e2)
        + 347.9 * max(0.0, 0.0018 - Q.girth2)
        + 7.149 * max(0.0, Q.lam2 - 0.00016)
        + 1.957 * max(0.0, 6.3 - Q.log_sum_pt)
        - 32.9 * max(0.0, 0.05 - Q.centroid_offset)
        + 10.78 * max(0.0, Q.girth - 0.074)
        + 5.907 * max(0.0, 0.088 - Q.girth)
        - 31.44 * max(0.0, 0.0038 - Q.width)
        - 87.83 * max(0.0, 0.0088 - Q.width)
        - 64.67 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 7.785 * max(0.0, Q.e2 - 0.062)
        - 0.009025 * max(0.0, Q.mass - 91.2)
        + 3.134 * max(0.0, Q.width - 0.018)
        - 8005.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        + 0.9379 * max(0.0, Q.C2 - 0.065)
        + 8.159 * max(0.0, 0.14 - Q.girth)
        + 0.004261 * max(0.0, Q.sum_pt - 980.0)
        - 13.04 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.0895 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 1.687e-05 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        + 19.95 * max(0.0, 0.091 - Q.girth)
        - 7.478 * max(0.0, 0.014 - Q.girth2)
        + 272.5 * max(0.0, Q.lam1 - 0.0075)
        - 0.008667 * max(0.0, 80.4 - Q.mass)
        + 2.062 * max(0.0, 0.18 - Q.max_dr)
        - 75.62 * max(0.0, 0.0075 - Q.width)
        + 2347.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 164.3 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        - 132.7 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 26.9 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 222.6 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        - 8.606 * max(0.0, Q.LHA - 0.34)
        - 9.486 * max(0.0, 0.041 - Q.e2)
        - 7.58 * max(0.0, 0.0065 - Q.lam1)
        - 251.3 * max(0.0, 0.0081 - Q.lam1)
        - 108.0 * max(0.0, 0.0069 - Q.width)
        - 531.8 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_W(Q):
    return (-11.87
        - 78.49 * max(0.0, 0.033 - Q.centroid_offset)
        - 29.49 * max(0.0, 0.013 - Q.girth2)
        - 0.06337 * max(0.0, 29.0 - Q.mass)
        - 0.03769 * max(0.0, 74.0 - Q.mass)
        - 0.006039 * max(0.0, Q.sum_pt - 880.0)
        - 1048.0 * max(0.0, 0.005 - Q.width)
        + 770.3 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 1575.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        + 1.865 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        - 0.007478 * max(0.0, Q.pt_7 - 32.0)
        + 398.9 * max(0.0, 0.01 - Q.width)
        - 12600.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 80.42 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 1406.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        + 7.999 * max(0.0, Q.LHA - 0.13)
        + 150.4 * max(0.0, 0.005 - Q.lam1)
        + 0.001073 * max(0.0, 780.0 - Q.sum_pt)
        + 5.036 * max(0.0, Q.LHA - 0.27)
        + 3.753 * max(0.0, Q.centroid_offset - 0.013)
        - 101.3 * max(0.0, 0.047 - Q.e2)
        + 398.9 * max(0.0, 0.01 - Q.girth2)
        + 141.4 * max(0.0, Q.lam1 - 0.016)
        + 42.0 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 157.1 * max(0.0, Q.C2 - 0.063)
        - 0.03552 * max(0.0, 55.0 - Q.mass)
        + 0.0006909 * max(0.0, Q.mass_over_sum_pt - 0.086)
        - 0.09039 * max(0.0, 0.27 - Q.tau21)
        + 707.2 * max(0.0, Q.width - 0.0016)
        + 152.3 * max(0.0, 0.04 - Q.e2)
        - 14.86 * max(0.0, Q.log_sum_pt - 6.9)
        + 18.5 * max(0.0, 0.035 - Q.z_7)
        - 22.59 * max(0.0, 0.066 - Q.z_7)
        + 23.83 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 13390.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 546.2 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        - 94.04 * max(0.0, Q.girth - 0.09)
        + 49.41 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 8.494 * Q.max_dr
        + 362.7 * max(0.0, 0.012 - Q.width)
        - 0.1402 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        - 0.009754 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        + 13.94 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 101.9 * max(0.0, 0.038 - Q.e2)
        - 114.5 * max(0.0, 0.0011 - Q.e2_sq)
        - 4.105 * max(0.0, 0.087 - Q.girth)
        - 232.7 * max(0.0, Q.girth2 - 0.0046)
        + 0.02324 * max(0.0, Q.mass - 80.4)
        + 31.07 * max(0.0, Q.mass_over_sum_pt - 0.073)
        - 48.95 * max(0.0, Q.mass_over_sum_pt - 0.09)
        + 941.7 * max(0.0, 0.00069 - Q.width)
        - 737.2 * max(0.0, 0.0056 - Q.width)
        - 43.16 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        + 371.3 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 9.635 * max(0.0, 0.063 - Q.girth)
        + 473.4 * max(0.0, 0.0049 - Q.width)
        + 10650.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        - 2.136 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        + 10780.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 77.2 * max(0.0, 0.019 - Q.centroid_offset)
        - 4.687 * max(0.0, Q.lam2 - 0.0015)
        - 2.778 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.1035 * max(0.0, 25.0 - Q.mass)
        - 869.0 * max(0.0, 0.0065 - Q.width)
        - 0.003883 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 21170.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        - 34.96 * max(0.0, Q.C2 - 0.059)
        + 187.0 * max(0.0, 0.037 - Q.e2)
        - 123.6 * max(0.0, 0.0018 - Q.girth2)
        - 198.1 * max(0.0, Q.lam2 - 0.00016)
        + 2.936 * max(0.0, 6.3 - Q.log_sum_pt)
        + 82.03 * max(0.0, 0.05 - Q.centroid_offset)
        - 37.27 * max(0.0, Q.girth - 0.074)
        - 151.1 * max(0.0, 0.088 - Q.girth)
        - 341.2 * max(0.0, 0.0038 - Q.width)
        + 117.5 * max(0.0, 0.0088 - Q.width)
        - 110.3 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 65.95 * max(0.0, Q.e2 - 0.062)
        + 0.002827 * max(0.0, Q.mass - 91.2)
        + 1.881 * max(0.0, Q.width - 0.018)
        - 2144.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 109.8 * max(0.0, Q.C2 - 0.065)
        - 23.26 * max(0.0, 0.14 - Q.girth)
        + 0.000907 * max(0.0, Q.sum_pt - 980.0)
        - 7.087 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.3195 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        + 0.0001164 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        + 188.3 * max(0.0, 0.091 - Q.girth)
        - 103.0 * max(0.0, 0.014 - Q.girth2)
        - 480.4 * max(0.0, Q.lam1 - 0.0075)
        + 0.01769 * max(0.0, 80.4 - Q.mass)
        - 7.078 * max(0.0, 0.18 - Q.max_dr)
        + 1208.0 * max(0.0, 0.0075 - Q.width)
        + 807.9 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 423.5 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 271.6 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 531.2 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 549.1 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        + 33.78 * max(0.0, Q.LHA - 0.34)
        - 220.2 * max(0.0, 0.041 - Q.e2)
        - 1624.0 * max(0.0, 0.0065 - Q.lam1)
        + 1430.0 * max(0.0, 0.0081 - Q.lam1)
        + 1362.0 * max(0.0, 0.0069 - Q.width)
        + 3196.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_Z(Q):
    return (-12.09
        - 48.41 * max(0.0, 0.033 - Q.centroid_offset)
        + 17.74 * max(0.0, 0.013 - Q.girth2)
        + 0.03969 * max(0.0, 29.0 - Q.mass)
        - 0.04765 * max(0.0, 74.0 - Q.mass)
        - 0.001645 * max(0.0, Q.sum_pt - 880.0)
        + 1273.0 * max(0.0, 0.005 - Q.width)
        + 1281.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 453.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        + 0.5696 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        - 0.003508 * max(0.0, Q.pt_7 - 32.0)
        + 300.3 * max(0.0, 0.01 - Q.width)
        + 872.2 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 119.5 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        + 2760.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        + 8.357 * max(0.0, Q.LHA - 0.13)
        + 227.5 * max(0.0, 0.005 - Q.lam1)
        + 0.001651 * max(0.0, 780.0 - Q.sum_pt)
        - 3.746 * max(0.0, Q.LHA - 0.27)
        - 11.2 * max(0.0, Q.centroid_offset - 0.013)
        - 120.9 * max(0.0, 0.047 - Q.e2)
        + 300.3 * max(0.0, 0.01 - Q.girth2)
        + 123.9 * max(0.0, Q.lam1 - 0.016)
        + 45.71 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        - 29.69 * max(0.0, Q.C2 - 0.063)
        + 0.01739 * max(0.0, 55.0 - Q.mass)
        + 98.33 * max(0.0, Q.mass_over_sum_pt - 0.086)
        + 0.1487 * max(0.0, 0.27 - Q.tau21)
        + 289.7 * max(0.0, Q.width - 0.0016)
        - 57.36 * max(0.0, 0.04 - Q.e2)
        - 8.371 * max(0.0, Q.log_sum_pt - 6.9)
        + 11.6 * max(0.0, 0.035 - Q.z_7)
        - 27.39 * max(0.0, 0.066 - Q.z_7)
        + 5.952 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 25040.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 269.7 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        - 85.5 * max(0.0, Q.girth - 0.09)
        + 46.55 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 5.184 * Q.max_dr
        + 603.0 * max(0.0, 0.012 - Q.width)
        - 0.1796 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        - 0.01246 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        + 13.2 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        + 91.52 * max(0.0, 0.038 - Q.e2)
        + 1020.0 * max(0.0, 0.0011 - Q.e2_sq)
        + 35.9 * max(0.0, 0.087 - Q.girth)
        + 251.5 * max(0.0, Q.girth2 - 0.0046)
        - 0.07972 * max(0.0, Q.mass - 80.4)
        + 51.54 * max(0.0, Q.mass_over_sum_pt - 0.073)
        - 198.9 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 644.8 * max(0.0, 0.00069 - Q.width)
        - 661.9 * max(0.0, 0.0056 - Q.width)
        + 5619.0 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        - 1764.0 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 29.52 * max(0.0, 0.063 - Q.girth)
        - 1559.0 * max(0.0, 0.0049 - Q.width)
        - 3382.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        - 2.287 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 17670.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        + 0.589 * max(0.0, 0.019 - Q.centroid_offset)
        - 100.0 * max(0.0, Q.lam2 - 0.0015)
        - 0.2036 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.01991 * max(0.0, 25.0 - Q.mass)
        - 380.8 * max(0.0, 0.0065 - Q.width)
        - 0.02436 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 22840.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        + 59.47 * max(0.0, Q.C2 - 0.059)
        - 35.4 * max(0.0, 0.037 - Q.e2)
        - 56.58 * max(0.0, 0.0018 - Q.girth2)
        + 23.45 * max(0.0, Q.lam2 - 0.00016)
        + 1.452 * max(0.0, 6.3 - Q.log_sum_pt)
        + 48.51 * max(0.0, 0.05 - Q.centroid_offset)
        + 18.96 * max(0.0, Q.girth - 0.074)
        - 18.67 * max(0.0, 0.088 - Q.girth)
        + 108.9 * max(0.0, 0.0038 - Q.width)
        + 314.9 * max(0.0, 0.0088 - Q.width)
        + 111.2 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        - 79.41 * max(0.0, Q.e2 - 0.062)
        + 0.09598 * max(0.0, Q.mass - 91.2)
        + 95.28 * max(0.0, Q.width - 0.018)
        - 7169.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 32.99 * max(0.0, Q.C2 - 0.065)
        + 13.99 * max(0.0, 0.14 - Q.girth)
        + 0.003017 * max(0.0, Q.sum_pt - 980.0)
        - 3.037 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        - 0.1591 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        + 4.093e-05 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 48.4 * max(0.0, 0.091 - Q.girth)
        + 483.4 * max(0.0, 0.014 - Q.girth2)
        - 512.9 * max(0.0, Q.lam1 - 0.0075)
        - 0.01191 * max(0.0, 80.4 - Q.mass)
        - 8.65 * max(0.0, 0.18 - Q.max_dr)
        - 682.8 * max(0.0, 0.0075 - Q.width)
        - 1741.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 254.6 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        - 477.3 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        - 201.4 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 39.67 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        + 14.71 * max(0.0, Q.LHA - 0.34)
        + 138.8 * max(0.0, 0.041 - Q.e2)
        + 30.08 * max(0.0, 0.0065 - Q.lam1)
        + 11.14 * max(0.0, 0.0081 - Q.lam1)
        + 186.5 * max(0.0, 0.0069 - Q.width)
        + 862.6 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def score_t(Q):
    return (4.481
        + 18.98 * max(0.0, 0.033 - Q.centroid_offset)
        - 76.11 * max(0.0, 0.013 - Q.girth2)
        - 0.009263 * max(0.0, 29.0 - Q.mass)
        - 0.00396 * max(0.0, 74.0 - Q.mass)
        + 0.004226 * max(0.0, Q.sum_pt - 880.0)
        - 106.7 * max(0.0, 0.005 - Q.width)
        - 3228.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 231.1 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
        - 0.5367 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
        + 0.00203 * max(0.0, Q.pt_7 - 32.0)
        - 28.01 * max(0.0, 0.01 - Q.width)
        + 2613.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
        - 22.67 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
        - 620.1 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
        - 8.869 * max(0.0, Q.LHA - 0.13)
        + 0.1395 * max(0.0, 0.005 - Q.lam1)
        + 0.0002203 * max(0.0, 780.0 - Q.sum_pt)
        - 20.22 * max(0.0, Q.LHA - 0.27)
        + 5.958 * max(0.0, Q.centroid_offset - 0.013)
        + 7.572 * max(0.0, 0.047 - Q.e2)
        - 28.01 * max(0.0, 0.01 - Q.girth2)
        - 55.78 * max(0.0, Q.lam1 - 0.016)
        + 22.94 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        + 112.0 * max(0.0, Q.C2 - 0.063)
        - 0.003971 * max(0.0, 55.0 - Q.mass)
        - 58.28 * max(0.0, Q.mass_over_sum_pt - 0.086)
        + 3.219 * max(0.0, 0.27 - Q.tau21)
        - 217.1 * max(0.0, Q.width - 0.0016)
        - 22.57 * max(0.0, 0.04 - Q.e2)
        + 33.1 * max(0.0, Q.log_sum_pt - 6.9)
        - 46.84 * max(0.0, 0.035 - Q.z_7)
        - 21.96 * max(0.0, 0.066 - Q.z_7)
        + 15.04 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 10120.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
        + 756.8 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        + 56.0 * max(0.0, Q.girth - 0.09)
        + 27.29 * max(0.0, Q.mass_over_sum_pt - 0.0089)
        - 0.2588 * Q.max_dr
        + 10.02 * max(0.0, 0.012 - Q.width)
        - 0.004005 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
        + 0.02035 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
        - 12.63 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        - 1.392 * max(0.0, 0.038 - Q.e2)
        - 143.0 * max(0.0, 0.0011 - Q.e2_sq)
        + 3.158 * max(0.0, 0.087 - Q.girth)
        + 31.36 * max(0.0, Q.girth2 - 0.0046)
        + 0.01072 * max(0.0, Q.mass - 80.4)
        + 9.199 * max(0.0, Q.mass_over_sum_pt - 0.073)
        + 25.22 * max(0.0, Q.mass_over_sum_pt - 0.09)
        - 422.1 * max(0.0, 0.00069 - Q.width)
        + 17.19 * max(0.0, 0.0056 - Q.width)
        - 103.7 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
        - 5.351 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 21.93 * max(0.0, 0.063 - Q.girth)
        + 142.9 * max(0.0, 0.0049 - Q.width)
        - 33050.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
        + 0.8639 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
        - 2184.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        - 14.94 * max(0.0, 0.019 - Q.centroid_offset)
        + 73.81 * max(0.0, Q.lam2 - 0.0015)
        + 0.6801 * max(0.0, Q.log_sum_pt - 6.3)
        + 0.01324 * max(0.0, 25.0 - Q.mass)
        + 184.0 * max(0.0, 0.0065 - Q.width)
        + 0.06198 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 4926.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        - 49.51 * max(0.0, Q.C2 - 0.059)
        + 15.04 * max(0.0, 0.037 - Q.e2)
        - 428.8 * max(0.0, 0.0018 - Q.girth2)
        + 416.4 * max(0.0, Q.lam2 - 0.00016)
        - 2.973 * max(0.0, 6.3 - Q.log_sum_pt)
        - 8.255 * max(0.0, 0.05 - Q.centroid_offset)
        - 7.317 * max(0.0, Q.girth - 0.074)
        - 17.7 * max(0.0, 0.088 - Q.girth)
        - 69.88 * max(0.0, 0.0038 - Q.width)
        - 56.23 * max(0.0, 0.0088 - Q.width)
        - 100.3 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        + 59.05 * max(0.0, Q.e2 - 0.062)
        - 0.05141 * max(0.0, Q.mass - 91.2)
        - 88.66 * max(0.0, Q.width - 0.018)
        - 11650.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        - 44.93 * max(0.0, Q.C2 - 0.065)
        - 28.0 * max(0.0, 0.14 - Q.girth)
        - 0.02162 * max(0.0, Q.sum_pt - 980.0)
        - 5.069 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
        + 0.6769 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
        - 0.0001364 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
        - 29.05 * max(0.0, 0.091 - Q.girth)
        + 7.899 * max(0.0, 0.014 - Q.girth2)
        + 244.1 * max(0.0, Q.lam1 - 0.0075)
        + 0.02138 * max(0.0, 80.4 - Q.mass)
        + 2.091 * max(0.0, 0.18 - Q.max_dr)
        - 59.02 * max(0.0, 0.0075 - Q.width)
        + 4642.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 162.2 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
        + 169.4 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 104.6 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 119.3 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
        - 3.179 * max(0.0, Q.LHA - 0.34)
        + 18.8 * max(0.0, 0.041 - Q.e2)
        - 31.89 * max(0.0, 0.0065 - Q.lam1)
        - 114.0 * max(0.0, 0.0081 - Q.lam1)
        - 59.8 * max(0.0, 0.0069 - Q.width)
        + 513.8 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['W'] - s['t'] > -3.823155641555786:
        if s['q'] - s['Z'] > 0.27294622361660004:
            if s['g'] - s['q'] > -0.0029658502899110317:
                if s['g'] - s['W'] > 0.07860340550541878:
                    if s['g'] - s['t'] > -0.17496978491544724:
                        return 'g'   # 95% of the training jets here get this class from the formula
                    else:
                        return 't'   # 86% of the training jets here get this class from the formula
                else:
                    return 'W'   # 66% of the training jets here get this class from the formula
            else:
                if s['q'] - s['Z'] > 0.9457018673419952:
                    if s['g'] - s['q'] > -0.15522217005491257:
                        if Q.sum_pt > 1007.4375:
                            return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 4.054451346746646e-05:
                                if Q.planar_flow > 0.5113564133644104:
                                    return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 81% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 39.29115867614746:
                            return 'W'   # 45% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 98% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 0.22986458986997604:
                        if Q.max_dr > 0.24262042343616486:
                            return 'W'   # 76% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 67% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 76% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.02809105906635523:
                if s['g'] - s['W'] > 0.10180466249585152:
                    if s['g'] - s['Z'] > 0.6102837324142456:
                        return 'g'   # 94% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > 1.7121744751930237:
                            return 'W'   # 70% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 68% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.23255885392427444:
                        if s['g'] - s['W'] > -0.35228610038757324:
                            if s['g'] - s['q'] > 1.2271281480789185:
                                return 'g'   # 60% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 78% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.26662035286426544:
                                return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                return 't'   # 51% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.16951961815357208:
                            if Q.girth2 > 0.003359551541507244:
                                return 'Z'   # 60% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 82% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 76% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.17597804963588715:
                    if s['g'] - s['t'] > -0.015729310922324657:
                        return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        return 't'   # 74% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.11883841827511787:
                        if s['W'] - s['Z'] > -0.28178954124450684:
                            if Q.LHA > 0.2618902176618576:
                                if Q.max_dr > 0.11223690584301949:
                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 67% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.5774629712104797:
                                    return 'W'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.04064381122589111:
                                if Q.mass > 22.453892707824707:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 96% of the training jets here get this class from the formula
                    else:
                        return 't'   # 65% of the training jets here get this class from the formula
    else:
        if s['g'] - s['t'] > -0.09877188131213188:
            return 'g'   # 77% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.02337406761944294:
                return 'Z'   # 82% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.4017889052629471:
                    return 'q'   # 52% of the training jets here get this class from the formula
                else:
                    return 't'   # 98% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
