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

Test set (50,000 jets): accuracy 65.50% (the formula: 65.80%); same class as the formula for 95.12% of jets.  200 leaves, depth 15.
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
    return (2.564
        - 0.0109 * max(0.0, Q.D2 - 3.9)
        + 2.913 * max(0.0, 0.033 - Q.centroid_offset)
        + 17.6 * max(0.0, 0.078 - Q.girth)
        - 46.59 * max(0.0, 0.013 - Q.girth2)
        + 3.223 * max(0.0, 0.0085 - Q.girth2_top5)
        - 123.6 * max(0.0, 0.0006 - Q.lam1)
        - 98.23 * max(0.0, 0.0015 - Q.lam1)
        - 0.004917 * max(0.0, 22.0 - Q.mass)
        + 0.02609 * max(0.0, 30.0 - Q.mass)
        + 0.01046 * max(0.0, 59.0 - Q.mass)
        - 0.04861 * max(0.0, Q.n_dr_0_0p05 - 3.8)
        - 6.661 * max(0.0, 0.013 - Q.planar_flow)
        - 0.01256 * max(0.0, Q.sum_pt - 810.0)
        + 0.0001022 * max(0.0, Q.sum_pt - 900.0)
        - 0.0002741 * max(0.0, Q.sum_pt_top5 - 700.0)
        + 0.03491 * max(0.0, Q.tau32 - 0.44)
        + 280.3 * max(0.0, 0.0044 - Q.width)
        - 343.1 * max(0.0, 0.0089 - Q.width)
        + 0.3316 * max(0.0, Q.z_dr_0_0p05 - 0.85)
        + 2.108 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2)
        - 0.6423 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0)
        - 300.3 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96)
        + 0.5621 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0)
        + 1.982 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0)
        - 269.2 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        + 0.9905 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0)
        - 0.2078 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024)
        - 0.02188 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2)
        - 0.7903 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        + 0.1984 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13)
        - 0.02482 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056)
        + 0.0005677 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7)
        + 2.192 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50)
        - 47.84 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset)
        + 39.11 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2)
        - 0.001916 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2)
        + 0.1803 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5)
        + 0.0002275 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7)
        - 5.842e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0)
        + 519.3 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7)
        + 1.126 * max(0.0, Q.LHA - 0.28)
        - 6.669 * max(0.0, Q.e2 - 0.034)
        - 294.9 * max(0.0, 0.0081 - Q.e2_sq)
        + 1.352 * max(0.0, Q.log_sum_pt - 6.4)
        + 2.19 * max(0.0, Q.log_sum_pt - 6.6)
        - 0.04891 * max(0.0, 6.1 - Q.mass_top5)
        - 1.087 * max(0.0, 0.048 - Q.max_dr)
        - 2.819 * max(0.0, 0.25 - Q.max_dr)
        + 0.06361 * max(0.0, Q.pt_7 - 34.0)
        - 0.0558 * max(0.0, Q.pt_7 - 54.0)
        + 10.22 * max(0.0, 0.0087 - Q.width)
        - 9.845 * max(0.0, 0.044 - Q.z_7)
        - 26.38 * max(0.0, 0.054 - Q.z_7)
        - 140.7 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 42.0 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow)
        + 387.0 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98)
        - 0.09811 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1)
        - 21.37 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32)
        - 820.5 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow)
        + 4940.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02)
        - 70.74 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037)
        + 2.655 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2)
        - 13.77 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 121.5 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3)
        - 0.2572 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr)
        - 0.9607 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 0.167 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21)
        + 86.78 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2)
        - 6.299 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32)
        + 0.9708 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016)
        - 0.0007197 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass)
        + 0.001031 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3)
        + 0.2989 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr)
        + 13860.0 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2)
        + 1227.0 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2)
        - 34.48 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2)
        - 2.033 * max(0.0, Q.LHA - 0.1)
        + 3191.0 * max(0.0, 3.7e-05 - Q.girth2_top2)
        + 133.8 * max(0.0, 0.0057 - Q.lam1)
        + 5.873 * max(0.0, Q.log_sum_pt - 6.3)
        + 7.057 * max(0.0, Q.log_sum_pt - 6.8)
        - 18.06 * max(0.0, Q.log_sum_pt - 6.9)
        + 1.306 * max(0.0, 6.5 - Q.log_sum_pt)
        + 0.03061 * max(0.0, 8.5 - Q.mass)
        - 0.02655 * max(0.0, 36.0 - Q.mass)
        + 0.02434 * max(0.0, 69.0 - Q.mass)
        - 68.92 * max(0.0, 0.0098 - Q.mass_over_sum_pt)
        - 78.44 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        - 1.373 * max(0.0, 0.16 - Q.max_dr)
        - 0.6409 * max(0.0, 0.68 - Q.planar_flow)
        + 0.02304 * max(0.0, Q.pt_7 - 31.0)
        + 0.0142 * max(0.0, 54.0 - Q.pt_7)
        + 0.004381 * max(0.0, 790.0 - Q.sum_pt)
        - 47.21 * max(0.0, 0.017 - Q.z_7)
        - 17080.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset)
        - 569.3 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078)
        + 4.135 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79)
        - 2.18 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0)
        - 0.01972 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        + 12.63 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 0.0005411 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 0.3462 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 1.603 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063)
        - 0.4929 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2)
        - 0.6632 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013)
        - 0.07457 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092)
        + 0.02584 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10)
        + 0.02036 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7)
        - 0.9671 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2)
        - 206.6 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7)
        - 6.656 * max(0.0, Q.C2 - 0.094)
        + 0.8215 * max(0.0, Q.LHA - 0.32)
        + 10.16 * max(0.0, Q.centroid_offset - 0.013)
        + 6.52 * max(0.0, Q.e2 - 0.028)
        + 2.641 * max(0.0, Q.e2 - 0.051)
        - 22.25 * max(0.0, 0.043 - Q.e2)
        + 361.4 * max(0.0, 0.0088 - Q.girth2)
        - 46.83 * max(0.0, Q.lam1 - 0.016)
        - 0.01065 * max(0.0, Q.mass - 70.0)
        - 0.002165 * max(0.0, Q.mass_top5 - 53.0)
        + 0.504 * max(0.0, Q.max_dr - 0.15)
        + 52.54 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 131.9 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr)
        - 3.392 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012)
        + 0.03677 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5)
        + 0.4125 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0)
        - 20.57 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi)
        + 2.41 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03)
        - 0.06999 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 1308.0 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08)
        - 667.2 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96)
        + 0.2184 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6)
        - 0.4159 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7)
        - 602.5 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 0.03188 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6)
        + 4.933 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.2094 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr)
        - 0.03722 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098)
        - 0.0007538 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7)
        + 55.38 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7)
        + 221.6 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7)
        + 104.1 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr)
        + 0.2882 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        - 2.921 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 0.001169 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7)
        - 30.08 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1)
        - 1.65 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3)
        + 4.001 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0)
        - 0.07382 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4)
        + 43.27 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4)
        - 36.28 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion)
        + 4.092 * max(0.0, Q.C2 - 0.013)
        - 21.71 * max(0.0, Q.C2 - 0.067)
        - 12.56 * max(0.0, 0.014 - Q.centroid_offset)
        - 4.384 * max(0.0, Q.e2 - 0.02)
        + 52.35 * max(0.0, Q.e2 - 0.064)
        + 67.77 * max(0.0, 0.003 - Q.e2_sq)
        + 6.105 * max(0.0, 0.017 - Q.e2_sq)
        - 5.316 * max(0.0, 0.13 - Q.girth)
        - 5.98 * max(0.0, Q.girth2 - 0.0034)
        + 79.1 * max(0.0, Q.girth2 - 0.0081)
        + 116.3 * max(0.0, 0.00033 - Q.lam2)
        + 0.008941 * max(0.0, 44.0 - Q.mass)
        + 12.19 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 19.4 * max(0.0, Q.mass_over_sum_pt - 0.11)
        - 3.547 * max(0.0, Q.max_dr - 0.094)
        + 3.113 * max(0.0, Q.max_dr - 0.2)
        + 0.0002085 * max(0.0, 760.0 - Q.sum_pt)
        + 0.00106 * max(0.0, 430.0 - Q.sum_pt_top5)
        - 0.4332 * max(0.0, 0.24 - Q.tau21)
        - 0.2444 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0)
        + 0.2703 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7)
        + 1.166 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1)
        + 263.1 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        + 0.2905 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044)
        + 91.62 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98)
        + 0.001478 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0)
        + 10.76 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 75.33 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 0.03925 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass)
        + 44.31 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi)
        - 0.03111 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0)
        + 0.2018 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7)
        - 0.004304 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2)
        + 232.2 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        + 43.32 * max(0.0, 0.023 - Q.dr_0)
        + 8.234 * max(0.0, 0.034 - Q.e2)
        - 7207.0 * max(0.0, 5.2e-05 - Q.girth2)
        - 0.03611 * max(0.0, 53.0 - Q.pt_7)
        - 0.0001043 * max(0.0, Q.sum_pt - 870.0)
        + 0.001872 * max(0.0, 540.0 - Q.sum_pt_top2)
        - 61.67 * max(0.0, 0.024 - Q.z_7)
        + 478.8 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034)
        + 33.12 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028)
        + 4.734 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.3602 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5)
        + 3.831 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05)
        - 38.83 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow)
        - 4.242 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063)
        + 187.5 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017)
        - 172.1 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0)
        + 413.2 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        - 4.162 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039)
        - 3241.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2)
        + 0.7696 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4)
        - 0.004502 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0)
        - 0.002424 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 1.018 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0)
        - 0.1429 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass)
        - 0.08445 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset)
        - 0.1142 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0)
        + 0.1776 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3)
        + 0.0002435 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2)
        - 3924.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 355.4 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 3.749 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 0.06267 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt)
        - 9.182 * max(0.0, Q.C2 - 0.033)
        + 3.246 * max(0.0, 0.034 - Q.C2)
        - 17.19 * max(0.0, Q.centroid_offset - 0.019)
        - 1.662 * max(0.0, Q.centroid_offset - 0.05)
        + 17.97 * max(0.0, 0.051 - Q.e2)
        + 7.446 * max(0.0, Q.girth - 0.083)
        + 24.21 * max(0.0, 0.0036 - Q.girth2)
        - 263.3 * max(0.0, 0.0086 - Q.girth2)
        + 13.17 * max(0.0, 0.01 - Q.girth2_top2)
        + 20.35 * max(0.0, Q.girth2_top5 - 0.011)
        - 26.34 * max(0.0, 0.0078 - Q.lam1)
        + 101.2 * max(0.0, Q.lam2 - 0.0034)
        - 0.3574 * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.01496 * max(0.0, 50.0 - Q.mass)
        - 21.27 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        + 0.5567 * max(0.0, Q.max_dr - 0.12)
        + 0.9099 * max(0.0, 0.11 - Q.max_dr)
        - 1.311 * max(0.0, 0.06 - Q.planar_flow)
        + 0.0116 * max(0.0, 25.0 - Q.pt_6)
        + 0.02231 * max(0.0, Q.sum_pt - 990.0)
        - 0.001704 * max(0.0, Q.sum_pt_top5 - 840.0)
        - 46.6 * max(0.0, 0.013 - Q.width)
        + 0.2259 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0)
        - 0.0216 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass)
        - 0.0006864 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4)
        - 4.531 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2)
        + 1140.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 0.7095 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3)
        - 2492.0 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2)
        + 2.372 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5)
        + 0.576 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0)
        + 32.17 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16)
        - 525.5 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13)
        + 830.5 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi)
        - 163.5 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014)
        - 0.8604 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 460.4 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        - 3.387 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098)
        - 0.003291 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0)
        + 0.01137 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6)
        - 0.01917 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        - 0.9324 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4)
        - 10.13 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 46.66 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.04413 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15)
        + 0.007874 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        + 0.231 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7)
        - 4.851 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        - 9.597e-06 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0)
        + 1231.0 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 22.53 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026)
        + 4.583 * max(0.0, 0.021 - Q.centroid_offset)
        - 50.27 * max(0.0, 0.0012 - Q.e2_sq)
        - 10.69 * max(0.0, 0.088 - Q.girth)
        + 146.6 * max(0.0, Q.girth2 - 0.0015)
        + 48.68 * max(0.0, Q.girth2 - 0.0044)
        - 1.083 * max(0.0, Q.girth2 - 0.0075)
        - 221.7 * max(0.0, Q.girth2 - 0.0087)
        + 14.06 * max(0.0, Q.girth2 - 0.015)
        + 66.52 * max(0.0, 0.0011 - Q.girth2_top2)
        - 88.82 * max(0.0, 0.0084 - Q.lam1)
        + 0.01318 * max(0.0, Q.mass - 80.4)
        - 3.274 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 10.14 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 37.97 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 6.843 * max(0.0, 0.00052 - Q.width)
        + 141.1 * max(0.0, 0.0055 - Q.width)
        - 0.2308 * max(0.0, Q.z_dr_0p05_0p1 - 0.66)
        - 147.3 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025)
        + 376.9 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068)
        + 0.06806 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0)
        + 0.1707 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0)
        - 0.07252 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        + 9.379 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 30.79 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 731.1 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055)
        + 15.87 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21)
        + 33.95 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95)
        - 45.23 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 870.6 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3)
        - 10.6 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 40.86 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0)
        - 36.19 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88)
        + 50.81 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 3.47 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50)
        + 0.01433 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92)
        - 0.0002794 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012)
        - 0.8914 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        + 0.9201 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2)
        + 2.828 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 1.938 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        + 0.3233 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2)
        + 0.1391 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2)
        - 0.003572 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0)
        + 0.005498 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow)
        - 1606.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi)
        - 19.07 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 147.3 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        + 4.575 * max(0.0, 0.027 - Q.C2)
        + 15.83 * max(0.0, 0.0034 - Q.centroid_offset)
        + 0.2647 * max(0.0, 0.016 - Q.dr_0)
        + 11.44 * max(0.0, Q.log_sum_pt - 6.7)
        + 152.3 * max(0.0, 0.0049 - Q.width)
        - 3522.0 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075)
        + 11.22 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075)
        + 6123.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 37.17 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 7495.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 2254.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 44.55 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow)
        + 110.1 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029)
        + 0.006686 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7)
        + 1.085 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031)
        - 2.593 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 0.008926 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 1533.0 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2)
        - 5070.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 4150.0 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045)
        - 1946.0 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2)
        + 4.351 * max(0.0, Q.C2 - 0.051)
        + 9.382 * max(0.0, 0.018 - Q.centroid_offset)
        - 12.87 * max(0.0, 0.055 - Q.girth)
        + 34.52 * max(0.0, Q.girth2 - 0.018)
        - 108.1 * max(0.0, Q.lam2 - 0.0017)
        - 29.61 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 209.5 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        + 1.959 * max(0.0, 0.23 - Q.max_dr)
        - 2.814 * max(0.0, 0.00042 - Q.mean_phi)
        + 0.1417 * max(0.0, Q.n_dr_0p2_0p4 - 0.9)
        - 0.0003254 * max(0.0, Q.sum_pt - 850.0)
        + 398.5 * max(0.0, 0.0061 - Q.width)
        - 18.79 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5)
        - 0.05193 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2)
        + 0.2174 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0)
        - 124.9 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043)
        - 181.2 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033)
        + 2866.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 110.4 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01)
        - 0.01566 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7)
        - 68.8 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21)
        - 8397.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 218.7 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta)
        - 224.7 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow)
        + 24.61 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0)
        - 13370.0 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023)
        + 0.9186 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0)
        - 11.89 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026)
        - 0.9064 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        + 0.7055 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.08084 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.07069 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 0.04848 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 1122.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 0.9181 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22)
        + 404.1 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031)
        - 10530.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        - 0.6903 * max(0.0, Q.C2 - 0.055)
        + 0.7845 * max(0.0, Q.LHA - 0.33)
        + 10.82 * max(0.0, Q.centroid_offset - 0.038)
        - 4.43 * max(0.0, Q.e2 - 0.036)
        + 1.869 * max(0.0, 0.037 - Q.e2)
        - 75.12 * max(0.0, Q.girth2 - 0.0085)
        - 59.6 * max(0.0, 0.0017 - Q.girth2)
        + 9.848 * max(0.0, 0.0038 - Q.girth2_top2)
        + 31.5 * max(0.0, 0.0016 - Q.girth2_top3)
        + 65.19 * max(0.0, Q.lam1 - 0.0085)
        - 89.44 * max(0.0, 0.0044 - Q.lam1)
        - 43.31 * max(0.0, 0.0034 - Q.lam2)
        - 3.673 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.02895 * max(0.0, Q.mass - 17.0)
        + 0.7077 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 0.01313 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 0.02333 * max(0.0, Q.n_dr_0p2_0p4 - 1.9)
        + 0.008042 * max(0.0, Q.pt_7 - 46.0)
        + 0.6125 * max(0.0, 0.27 - Q.tau32)
        - 7.032 * max(0.0, Q.z_7 - 0.062)
        + 1.113 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21)
        + 0.001874 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2)
        - 22.27 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4)
        - 69.81 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 0.7682 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass)
        + 0.4432 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50)
        - 124.8 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21)
        + 0.001244 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 0.08864 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 3.203 * max(0.0, 0.035 - Q.C2)
        + 6.388 * max(0.0, Q.centroid_offset - 0.014)
        + 11.46 * max(0.0, 0.04 - Q.centroid_offset)
        - 8.78 * max(0.0, 0.0062 - Q.e2_sq)
        + 12.94 * max(0.0, Q.girth - 0.076)
        + 0.6806 * max(0.0, 0.021 - Q.girth)
        - 0.5472 * max(0.0, 0.089 - Q.girth)
        - 25.07 * max(0.0, 0.002 - Q.girth2_top3)
        + 0.00956 * max(0.0, 5.5 - Q.mass_top5)
        - 2.656 * max(0.0, 0.22 - Q.max_dr)
        - 0.004344 * max(0.0, 2.8 - Q.n_dr_0p1_0p2)
        + 0.01599 * max(0.0, 0.27 - Q.planar_flow)
        - 0.000861 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 24.21 * max(0.0, 0.0036 - Q.width)
        + 156.2 * max(0.0, 0.0085 - Q.width)
        - 1.907 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7)
        - 14.66 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 7.962 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21)
        - 10.31 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014)
        + 113.7 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi)
        - 0.8353 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.937 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 0.404 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7)
        + 1.001 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0)
        + 178.2 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi)
        + 82.3 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032)
        + 0.09594 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0)
        + 0.0004815 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 298.6 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        - 0.001759 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass)
        + 1.314 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11)
        - 188.3 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        + 0.006021 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 45.35 * max(0.0, Q.e2 - 0.063)
        + 74.48 * max(0.0, Q.girth2 - 0.019)
        + 0.01373 * max(0.0, Q.mass - 91.2)
        - 3059.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        - 0.5738 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0)
        + 22.81 * max(0.0, Q.C2 - 0.066)
        - 8.837 * max(0.0, 0.038 - Q.centroid_offset)
        - 11.62 * max(0.0, 0.049 - Q.e2)
        + 2.538 * max(0.0, 0.15 - Q.girth)
        + 80.88 * max(0.0, 0.0062 - Q.lam1)
        + 6.763 * max(0.0, 0.015 - Q.lam1)
        - 0.01729 * max(0.0, 25.0 - Q.pt_7)
        - 0.008209 * max(0.0, Q.sum_pt - 1000.0)
        - 0.05552 * max(0.0, 0.5 - Q.tau21)
        - 148.4 * max(0.0, 0.0076 - Q.width)
        + 16.69 * max(0.0, 0.027 - Q.z_7)
        + 2.077 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4)
        + 6.763 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.2178 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        - 89.98 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 60.34 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18)
        + 13570.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        + 0.0005158 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2)
        - 0.0005227 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.07482 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4)
        - 0.0002527 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2)
        + 3.248e-05 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88)
        + 0.001842 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1)
        - 0.0001885 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5)
        - 8.634e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 0.003486 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32)
        + 0.02453 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023)
        + 2.155 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013)
        + 1.148 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0)
        - 8.477 * max(0.0, 0.067 - Q.C2)
        + 0.01089 * max(0.0, 0.8 - Q.D2)
        + 1.588 * max(0.0, Q.centroid_offset - 0.031)
        + 86.62 * max(0.0, 0.0058 - Q.e2_sq)
        + 8.175 * max(0.0, 0.034 - Q.girth)
        + 6.132 * max(0.0, 0.087 - Q.girth)
        - 39.52 * max(0.0, Q.lam1 - 0.0025)
        + 52.61 * max(0.0, Q.lam1 - 0.0042)
        - 148.4 * max(0.0, Q.lam1 - 0.0061)
        + 122.1 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        - 3.671 * max(0.0, 0.078 - Q.max_dr)
        + 2.543 * max(0.0, 0.18 - Q.max_dr)
        + 0.01242 * max(0.0, 4.8 - Q.n_dr_0p05_0p1)
        - 0.001058 * max(0.0, 310.0 - Q.sum_pt_top3)
        + 0.5572 * max(0.0, 0.14 - Q.tau21)
        - 0.1381 * max(0.0, 0.6 - Q.z_dr_0p05_0p1)
        - 7.679 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        + 6.703 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2)
        - 501.9 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 0.8266 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0)
        - 6.071 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 114.7 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        - 3.387 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        + 18.0 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6)
        + 21.93 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 2939.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        - 124.5 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094)
        + 133.6 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 15.03 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr)
        + 0.00351 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt)
        + 194.4 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 1.654 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0)
        - 1556.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta)
        - 585.5 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        + 27.44 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 891.3 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow)
        + 6.552 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2)
        - 0.06581 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.6858 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 2.371 * max(0.0, Q.LHA - 0.34)
        - 27.9 * max(0.0, 0.024 - Q.e2)
        + 17.23 * max(0.0, 0.041 - Q.e2)
        - 2.98 * max(0.0, Q.girth - 0.032)
        - 4.239 * max(0.0, 0.0074 - Q.girth2_top2)
        + 37.75 * max(0.0, 0.0067 - Q.lam1)
        - 17.99 * max(0.0, 0.0083 - Q.lam1)
        - 363.3 * max(0.0, 0.00031 - Q.lam2)
        - 7.286 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        + 4.189 * max(0.0, 0.0067 - Q.width)
        - 0.6701 * max(0.0, Q.z_dr_0p05_0p1 - 0.75)
        - 0.2451 * max(0.0, 0.32 - Q.z_dr_0p1_0p2)
        - 8.333 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion)
        + 0.0003688 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0)
        + 138.4 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2)
        + 1.97 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0)
        - 0.658 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3)
        + 748.6 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017)
        - 0.01601 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 0.06336 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion)
        + 15.92 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084)
        - 0.2583 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1)
        - 0.3403 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 7354.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 918.1 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 620.3 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow)
    )


def score_q(Q):
    return (3.477
        - 0.01997 * max(0.0, Q.D2 - 3.9)
        + 12.67 * max(0.0, 0.033 - Q.centroid_offset)
        + 5.807 * max(0.0, 0.078 - Q.girth)
        - 36.32 * max(0.0, 0.013 - Q.girth2)
        - 4.547 * max(0.0, 0.0085 - Q.girth2_top5)
        - 376.9 * max(0.0, 0.0006 - Q.lam1)
        + 58.66 * max(0.0, 0.0015 - Q.lam1)
        + 0.01433 * max(0.0, 22.0 - Q.mass)
        + 0.02937 * max(0.0, 30.0 - Q.mass)
        - 0.006772 * max(0.0, 59.0 - Q.mass)
        - 0.01662 * max(0.0, Q.n_dr_0_0p05 - 3.8)
        - 1.747 * max(0.0, 0.013 - Q.planar_flow)
        - 0.01943 * max(0.0, Q.sum_pt - 810.0)
        + 0.01094 * max(0.0, Q.sum_pt - 900.0)
        - 5.356e-05 * max(0.0, Q.sum_pt_top5 - 700.0)
        + 0.007224 * max(0.0, Q.tau32 - 0.44)
        + 146.8 * max(0.0, 0.0044 - Q.width)
        - 526.1 * max(0.0, 0.0089 - Q.width)
        + 0.9092 * max(0.0, Q.z_dr_0_0p05 - 0.85)
        + 0.1415 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2)
        + 0.2509 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0)
        - 220.9 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96)
        - 0.3388 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0)
        - 27.42 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0)
        - 284.3 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        + 1.71 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0)
        + 0.1576 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024)
        + 0.03617 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2)
        - 0.6569 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        + 0.04833 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13)
        - 0.03041 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056)
        + 0.0002173 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7)
        + 2.596 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50)
        - 55.55 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset)
        + 13.46 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2)
        - 0.001025 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2)
        - 0.4052 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5)
        - 6.213e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7)
        + 1.64e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0)
        + 207.1 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7)
        - 0.3921 * max(0.0, Q.LHA - 0.28)
        - 4.253 * max(0.0, Q.e2 - 0.034)
        - 174.1 * max(0.0, 0.0081 - Q.e2_sq)
        + 0.3507 * max(0.0, Q.log_sum_pt - 6.4)
        + 0.1905 * max(0.0, Q.log_sum_pt - 6.6)
        - 0.01139 * max(0.0, 6.1 - Q.mass_top5)
        + 1.919 * max(0.0, 0.048 - Q.max_dr)
        - 1.568 * max(0.0, 0.25 - Q.max_dr)
        + 0.009437 * max(0.0, Q.pt_7 - 34.0)
        - 0.0279 * max(0.0, Q.pt_7 - 54.0)
        - 107.8 * max(0.0, 0.0087 - Q.width)
        + 0.2726 * max(0.0, 0.044 - Q.z_7)
        - 3.11 * max(0.0, 0.054 - Q.z_7)
        - 40.58 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 15.15 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow)
        - 503.5 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98)
        + 0.04599 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1)
        + 4.368 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32)
        + 48.21 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow)
        + 1467.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02)
        + 34.41 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037)
        + 0.4294 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2)
        + 89.07 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 36.65 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3)
        - 4.009 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr)
        - 0.1877 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 0.05102 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21)
        - 6.592 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2)
        + 1.515 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32)
        - 0.06223 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016)
        - 2.466e-05 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass)
        + 3.234e-05 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3)
        + 0.1134 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr)
        + 1600.0 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2)
        + 189.4 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2)
        - 3.352 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2)
        - 1.225 * max(0.0, Q.LHA - 0.1)
        - 850.9 * max(0.0, 3.7e-05 - Q.girth2_top2)
        + 71.7 * max(0.0, 0.0057 - Q.lam1)
        + 0.5909 * max(0.0, Q.log_sum_pt - 6.3)
        - 5.334 * max(0.0, Q.log_sum_pt - 6.8)
        - 11.7 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.305 * max(0.0, 6.5 - Q.log_sum_pt)
        - 0.0134 * max(0.0, 8.5 - Q.mass)
        - 0.004207 * max(0.0, 36.0 - Q.mass)
        + 0.0126 * max(0.0, 69.0 - Q.mass)
        + 5.201 * max(0.0, 0.0098 - Q.mass_over_sum_pt)
        - 32.73 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 1.188 * max(0.0, 0.16 - Q.max_dr)
        - 0.08278 * max(0.0, 0.68 - Q.planar_flow)
        - 0.009005 * max(0.0, Q.pt_7 - 31.0)
        + 0.03327 * max(0.0, 54.0 - Q.pt_7)
        + 0.002798 * max(0.0, 790.0 - Q.sum_pt)
        + 3.55 * max(0.0, 0.017 - Q.z_7)
        - 5940.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset)
        - 449.1 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078)
        + 0.5925 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79)
        - 0.4768 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0)
        + 0.05521 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        + 0.5294 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        + 9.984e-05 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 0.007791 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        - 0.718 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063)
        + 0.2381 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2)
        - 0.02274 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013)
        + 0.03614 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092)
        + 0.0008166 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10)
        - 0.0008509 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7)
        - 0.0655 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2)
        + 22.8 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7)
        - 7.976 * max(0.0, Q.C2 - 0.094)
        + 0.4361 * max(0.0, Q.LHA - 0.32)
        + 15.9 * max(0.0, Q.centroid_offset - 0.013)
        + 10.4 * max(0.0, Q.e2 - 0.028)
        - 0.04876 * max(0.0, Q.e2 - 0.051)
        - 16.04 * max(0.0, 0.043 - Q.e2)
        + 694.9 * max(0.0, 0.0088 - Q.girth2)
        - 70.06 * max(0.0, Q.lam1 - 0.016)
        - 0.003521 * max(0.0, Q.mass - 70.0)
        + 0.0001016 * max(0.0, Q.mass_top5 - 53.0)
        - 0.1751 * max(0.0, Q.max_dr - 0.15)
        + 1.877 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 185.7 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr)
        - 3.726 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012)
        + 0.1212 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5)
        + 0.45 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0)
        - 28.26 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi)
        + 3.033 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03)
        + 0.6259 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 1128.0 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08)
        - 652.1 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96)
        + 0.1274 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6)
        - 1.497 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7)
        - 101.3 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 0.04154 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6)
        + 4.611 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.1135 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr)
        - 0.0582 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098)
        - 0.0006548 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7)
        + 57.65 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7)
        + 200.3 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7)
        - 166.6 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr)
        + 0.1937 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        - 9.545 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        - 0.0006623 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7)
        - 41.93 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1)
        + 4.321 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3)
        + 3.175 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0)
        - 0.03953 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4)
        + 26.03 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4)
        + 8.593 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion)
        + 10.29 * max(0.0, Q.C2 - 0.013)
        - 71.81 * max(0.0, Q.C2 - 0.067)
        - 13.92 * max(0.0, 0.014 - Q.centroid_offset)
        - 3.476 * max(0.0, Q.e2 - 0.02)
        + 52.38 * max(0.0, Q.e2 - 0.064)
        + 78.57 * max(0.0, 0.003 - Q.e2_sq)
        + 20.55 * max(0.0, 0.017 - Q.e2_sq)
        - 3.53 * max(0.0, 0.13 - Q.girth)
        - 55.59 * max(0.0, Q.girth2 - 0.0034)
        + 110.7 * max(0.0, Q.girth2 - 0.0081)
        + 568.6 * max(0.0, 0.00033 - Q.lam2)
        + 0.00572 * max(0.0, 44.0 - Q.mass)
        + 18.99 * max(0.0, Q.mass_over_sum_pt - 0.089)
        - 0.8995 * max(0.0, Q.mass_over_sum_pt - 0.11)
        - 2.472 * max(0.0, Q.max_dr - 0.094)
        + 5.101 * max(0.0, Q.max_dr - 0.2)
        - 0.0003144 * max(0.0, 760.0 - Q.sum_pt)
        - 0.002361 * max(0.0, 430.0 - Q.sum_pt_top5)
        - 3.547 * max(0.0, 0.24 - Q.tau21)
        - 0.5664 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0)
        + 0.03685 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7)
        + 20.47 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1)
        - 522.7 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        + 0.5467 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044)
        + 76.48 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98)
        + 0.1147 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0)
        + 199.5 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 1130.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 0.0496 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass)
        + 66.57 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi)
        - 0.02991 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0)
        + 0.1614 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7)
        - 0.004355 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2)
        + 177.2 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 6.511 * max(0.0, 0.023 - Q.dr_0)
        + 20.65 * max(0.0, 0.034 - Q.e2)
        + 5766.0 * max(0.0, 5.2e-05 - Q.girth2)
        - 0.01758 * max(0.0, 53.0 - Q.pt_7)
        + 0.0009856 * max(0.0, Q.sum_pt - 870.0)
        - 0.0003565 * max(0.0, 540.0 - Q.sum_pt_top2)
        + 9.247 * max(0.0, 0.024 - Q.z_7)
        - 475.6 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034)
        + 144.9 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028)
        - 12.99 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.02867 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5)
        + 1.876 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05)
        + 17.81 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow)
        + 24.51 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063)
        + 48.12 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017)
        + 49.51 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0)
        - 45.26 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        + 11.0 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039)
        + 1659.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2)
        - 0.4706 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4)
        + 0.009988 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0)
        + 0.0004416 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.3996 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0)
        + 0.04194 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass)
        + 0.03848 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset)
        + 0.01288 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0)
        - 0.02455 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3)
        - 0.000762 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2)
        + 23370.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        - 208.3 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 1.14 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 0.0002021 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt)
        + 0.932 * max(0.0, Q.C2 - 0.033)
        - 7.388 * max(0.0, 0.034 - Q.C2)
        - 5.246 * max(0.0, Q.centroid_offset - 0.019)
        - 0.7913 * max(0.0, Q.centroid_offset - 0.05)
        + 26.3 * max(0.0, 0.051 - Q.e2)
        + 4.575 * max(0.0, Q.girth - 0.083)
        + 79.16 * max(0.0, 0.0036 - Q.girth2)
        - 139.4 * max(0.0, 0.0086 - Q.girth2)
        + 7.061 * max(0.0, 0.01 - Q.girth2_top2)
        + 23.74 * max(0.0, Q.girth2_top5 - 0.011)
        + 8.094 * max(0.0, 0.0078 - Q.lam1)
        + 224.9 * max(0.0, Q.lam2 - 0.0034)
        - 3.386 * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.01172 * max(0.0, 50.0 - Q.mass)
        - 18.68 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 0.4972 * max(0.0, Q.max_dr - 0.12)
        + 0.02088 * max(0.0, 0.11 - Q.max_dr)
        - 0.9546 * max(0.0, 0.06 - Q.planar_flow)
        + 0.005673 * max(0.0, 25.0 - Q.pt_6)
        + 0.00706 * max(0.0, Q.sum_pt - 990.0)
        - 0.0006112 * max(0.0, Q.sum_pt_top5 - 840.0)
        - 36.32 * max(0.0, 0.013 - Q.width)
        + 0.336 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0)
        - 0.01527 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass)
        - 0.0005658 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4)
        - 9.04 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2)
        + 1198.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 0.4911 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3)
        - 2921.0 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2)
        + 2.494 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5)
        + 0.5411 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0)
        + 24.43 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16)
        - 669.0 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13)
        + 968.9 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi)
        - 177.7 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014)
        - 0.6565 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 506.6 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        - 3.508 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098)
        + 0.00982 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0)
        + 0.02378 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6)
        + 0.01932 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        - 2.331 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4)
        + 41.8 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 18.36 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.07465 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15)
        + 0.01145 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        + 0.06699 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7)
        - 2.16 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        - 1.434e-05 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0)
        + 1191.0 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 135.4 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026)
        - 6.662 * max(0.0, 0.021 - Q.centroid_offset)
        + 234.3 * max(0.0, 0.0012 - Q.e2_sq)
        - 16.18 * max(0.0, 0.088 - Q.girth)
        - 156.4 * max(0.0, Q.girth2 - 0.0015)
        + 44.75 * max(0.0, Q.girth2 - 0.0044)
        + 160.1 * max(0.0, Q.girth2 - 0.0075)
        - 204.0 * max(0.0, Q.girth2 - 0.0087)
        + 14.96 * max(0.0, Q.girth2 - 0.015)
        - 86.33 * max(0.0, 0.0011 - Q.girth2_top2)
        - 148.6 * max(0.0, 0.0084 - Q.lam1)
        + 0.01058 * max(0.0, Q.mass - 80.4)
        - 2.596 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 8.516 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 40.11 * max(0.0, Q.mass_over_sum_pt - 0.091)
        + 53.63 * max(0.0, 0.00052 - Q.width)
        + 262.1 * max(0.0, 0.0055 - Q.width)
        - 0.05427 * max(0.0, Q.z_dr_0p05_0p1 - 0.66)
        - 148.4 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025)
        + 288.0 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068)
        + 0.05491 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0)
        + 0.1462 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0)
        - 0.1048 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 1.243 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 20.42 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 796.6 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055)
        - 12.82 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21)
        + 566.7 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95)
        - 88.39 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 483.9 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3)
        + 97.85 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 3.265 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0)
        - 33.74 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88)
        - 1.861 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 5.683 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50)
        - 0.1068 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92)
        + 0.0001387 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012)
        - 1.94 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        + 2.145 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2)
        - 0.2827 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 1.944 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        + 0.1612 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2)
        + 0.1058 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2)
        - 0.001012 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0)
        - 0.0002852 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow)
        - 2076.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi)
        - 16.44 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        + 58.34 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        - 3.364 * max(0.0, 0.027 - Q.C2)
        + 32.17 * max(0.0, 0.0034 - Q.centroid_offset)
        - 11.83 * max(0.0, 0.016 - Q.dr_0)
        + 14.61 * max(0.0, Q.log_sum_pt - 6.7)
        + 212.3 * max(0.0, 0.0049 - Q.width)
        + 7008.0 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075)
        - 33.14 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075)
        + 8249.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 6.537 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 11270.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 2903.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 97.68 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow)
        - 117.5 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029)
        + 0.01419 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7)
        + 0.6818 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031)
        - 3.23 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 0.0383 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 4643.0 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2)
        - 4581.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 18480.0 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045)
        - 3256.0 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2)
        + 3.999 * max(0.0, Q.C2 - 0.051)
        + 29.0 * max(0.0, 0.018 - Q.centroid_offset)
        - 8.886 * max(0.0, 0.055 - Q.girth)
        + 40.62 * max(0.0, Q.girth2 - 0.018)
        - 126.8 * max(0.0, Q.lam2 - 0.0017)
        - 64.95 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 261.9 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        + 3.274 * max(0.0, 0.23 - Q.max_dr)
        - 3.119 * max(0.0, 0.00042 - Q.mean_phi)
        + 0.1553 * max(0.0, Q.n_dr_0p2_0p4 - 0.9)
        - 0.001325 * max(0.0, Q.sum_pt - 850.0)
        + 536.0 * max(0.0, 0.0061 - Q.width)
        - 34.68 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5)
        - 0.08723 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2)
        + 0.5119 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0)
        - 376.1 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043)
        - 279.3 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033)
        + 3104.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 158.8 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01)
        - 0.4036 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7)
        - 53.06 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21)
        - 15720.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 348.5 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta)
        - 316.8 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow)
        + 5.912 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0)
        - 2957.0 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023)
        + 1.608 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0)
        - 18.46 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026)
        - 1.304 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        + 1.331 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.06028 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.08902 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 0.05648 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 11150.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 1.458 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22)
        - 1287.0 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031)
        - 10850.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        - 2.38 * max(0.0, Q.C2 - 0.055)
        + 1.165 * max(0.0, Q.LHA - 0.33)
        + 10.33 * max(0.0, Q.centroid_offset - 0.038)
        - 9.589 * max(0.0, Q.e2 - 0.036)
        + 7.691 * max(0.0, 0.037 - Q.e2)
        - 78.19 * max(0.0, Q.girth2 - 0.0085)
        + 81.04 * max(0.0, 0.0017 - Q.girth2)
        - 49.19 * max(0.0, 0.0038 - Q.girth2_top2)
        - 24.93 * max(0.0, 0.0016 - Q.girth2_top3)
        + 161.2 * max(0.0, Q.lam1 - 0.0085)
        - 55.74 * max(0.0, 0.0044 - Q.lam1)
        - 39.9 * max(0.0, 0.0034 - Q.lam2)
        + 0.6947 * max(0.0, 6.3 - Q.log_sum_pt)
        - 0.02113 * max(0.0, Q.mass - 17.0)
        - 3.155 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        - 0.01089 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        - 0.02013 * max(0.0, Q.n_dr_0p2_0p4 - 1.9)
        - 0.001415 * max(0.0, Q.pt_7 - 46.0)
        - 1.315 * max(0.0, 0.27 - Q.tau32)
        + 3.834 * max(0.0, Q.z_7 - 0.062)
        + 7.25 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21)
        + 0.01385 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2)
        - 39.42 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4)
        - 588.4 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        + 0.383 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass)
        + 7.573 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50)
        - 336.3 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21)
        + 0.001755 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        + 0.452 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 8.203 * max(0.0, 0.035 - Q.C2)
        - 2.72 * max(0.0, Q.centroid_offset - 0.014)
        + 11.99 * max(0.0, 0.04 - Q.centroid_offset)
        + 13.77 * max(0.0, 0.0062 - Q.e2_sq)
        + 4.355 * max(0.0, Q.girth - 0.076)
        + 12.78 * max(0.0, 0.021 - Q.girth)
        + 10.91 * max(0.0, 0.089 - Q.girth)
        - 28.2 * max(0.0, 0.002 - Q.girth2_top3)
        + 0.009339 * max(0.0, 5.5 - Q.mass_top5)
        - 3.87 * max(0.0, 0.22 - Q.max_dr)
        + 0.02057 * max(0.0, 2.8 - Q.n_dr_0p1_0p2)
        + 0.3665 * max(0.0, 0.27 - Q.planar_flow)
        + 0.0001683 * max(0.0, 690.0 - Q.sum_pt_top5)
        + 79.16 * max(0.0, 0.0036 - Q.width)
        + 17.56 * max(0.0, 0.0085 - Q.width)
        + 10.45 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7)
        - 22.6 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 12.88 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21)
        + 58.47 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014)
        + 375.7 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi)
        + 0.1047 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.03452 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 0.07449 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7)
        - 0.488 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0)
        - 10.45 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi)
        + 70.2 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032)
        + 1.53 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0)
        - 0.001318 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 370.2 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        - 0.005737 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass)
        - 1.627 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11)
        - 272.5 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        + 0.001516 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 40.16 * max(0.0, Q.e2 - 0.063)
        + 88.03 * max(0.0, Q.girth2 - 0.019)
        + 0.007626 * max(0.0, Q.mass - 91.2)
        - 4293.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        - 0.8279 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0)
        + 59.06 * max(0.0, Q.C2 - 0.066)
        - 2.045 * max(0.0, 0.038 - Q.centroid_offset)
        - 12.87 * max(0.0, 0.049 - Q.e2)
        + 1.952 * max(0.0, 0.15 - Q.girth)
        - 22.34 * max(0.0, 0.0062 - Q.lam1)
        + 19.88 * max(0.0, 0.015 - Q.lam1)
        - 0.003156 * max(0.0, 25.0 - Q.pt_7)
        + 0.009589 * max(0.0, Q.sum_pt - 1000.0)
        - 0.2457 * max(0.0, 0.5 - Q.tau21)
        - 289.2 * max(0.0, 0.0076 - Q.width)
        + 0.8082 * max(0.0, 0.027 - Q.z_7)
        + 2.499 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4)
        + 5.278 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.1226 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        - 546.8 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 23.23 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18)
        + 21930.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        - 0.0001908 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2)
        - 9.705e-05 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.02656 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4)
        + 0.0001064 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2)
        - 3.507e-05 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88)
        + 0.0001514 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1)
        - 0.0002927 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5)
        - 3.31e-06 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 0.001218 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32)
        - 7.298e-06 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023)
        + 1.68 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013)
        - 2.16 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0)
        + 9.389 * max(0.0, 0.067 - Q.C2)
        - 0.1003 * max(0.0, 0.8 - Q.D2)
        - 9.866 * max(0.0, Q.centroid_offset - 0.031)
        + 342.8 * max(0.0, 0.0058 - Q.e2_sq)
        + 5.119 * max(0.0, 0.034 - Q.girth)
        + 9.04 * max(0.0, 0.087 - Q.girth)
        - 140.1 * max(0.0, Q.lam1 - 0.0025)
        + 91.8 * max(0.0, Q.lam1 - 0.0042)
        - 30.39 * max(0.0, Q.lam1 - 0.0061)
        - 63.36 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 0.1878 * max(0.0, 0.078 - Q.max_dr)
        + 1.753 * max(0.0, 0.18 - Q.max_dr)
        + 0.005705 * max(0.0, 4.8 - Q.n_dr_0p05_0p1)
        + 0.0002169 * max(0.0, 310.0 - Q.sum_pt_top3)
        + 0.7657 * max(0.0, 0.14 - Q.tau21)
        - 0.02208 * max(0.0, 0.6 - Q.z_dr_0p05_0p1)
        - 5.642 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        - 5.732 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2)
        + 862.3 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 0.3349 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0)
        - 8.739 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 4.067 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 29.79 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        + 19.73 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6)
        - 14.28 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 2515.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        - 42.6 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094)
        + 36.51 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        - 4.375 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr)
        + 0.0005761 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt)
        + 198.6 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 5.203 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0)
        - 1601.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta)
        + 578.4 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        + 31.83 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 413.6 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow)
        + 0.8424 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2)
        - 0.05792 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.9591 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 1.265 * max(0.0, Q.LHA - 0.34)
        - 18.62 * max(0.0, 0.024 - Q.e2)
        + 10.18 * max(0.0, 0.041 - Q.e2)
        + 7.018 * max(0.0, Q.girth - 0.032)
        + 6.01 * max(0.0, 0.0074 - Q.girth2_top2)
        + 41.7 * max(0.0, 0.0067 - Q.lam1)
        - 10.06 * max(0.0, 0.0083 - Q.lam1)
        - 650.2 * max(0.0, 0.00031 - Q.lam2)
        - 7.43 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 73.67 * max(0.0, 0.0067 - Q.width)
        - 1.124 * max(0.0, Q.z_dr_0p05_0p1 - 0.75)
        - 0.323 * max(0.0, 0.32 - Q.z_dr_0p1_0p2)
        - 3.974 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion)
        + 0.001707 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0)
        + 110.8 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2)
        + 0.9884 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0)
        - 1.323 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3)
        + 714.4 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017)
        - 0.02301 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 0.4596 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion)
        - 61.61 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084)
        + 0.5855 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1)
        + 2.196 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 9891.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 233.2 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 497.7 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow)
    )


def score_W(Q):
    return (-12.0
        - 0.03667 * max(0.0, Q.D2 - 3.9)
        - 22.29 * max(0.0, 0.033 - Q.centroid_offset)
        - 71.44 * max(0.0, 0.078 - Q.girth)
        + 130.5 * max(0.0, 0.013 - Q.girth2)
        - 57.55 * max(0.0, 0.0085 - Q.girth2_top5)
        + 946.8 * max(0.0, 0.0006 - Q.lam1)
        + 843.9 * max(0.0, 0.0015 - Q.lam1)
        + 0.06428 * max(0.0, 22.0 - Q.mass)
        - 0.0249 * max(0.0, 30.0 - Q.mass)
        - 0.03867 * max(0.0, 59.0 - Q.mass)
        + 0.06994 * max(0.0, Q.n_dr_0_0p05 - 3.8)
        + 22.9 * max(0.0, 0.013 - Q.planar_flow)
        + 0.02207 * max(0.0, Q.sum_pt - 810.0)
        - 0.0103 * max(0.0, Q.sum_pt - 900.0)
        + 0.002779 * max(0.0, Q.sum_pt_top5 - 700.0)
        - 0.2773 * max(0.0, Q.tau32 - 0.44)
        - 1976.0 * max(0.0, 0.0044 - Q.width)
        - 3.489 * max(0.0, 0.0089 - Q.width)
        + 2.577 * max(0.0, Q.z_dr_0_0p05 - 0.85)
        - 3.67 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2)
        + 1.578 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0)
        - 721.5 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96)
        - 0.6971 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0)
        + 38.53 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0)
        - 1782.0 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        - 13.16 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0)
        + 0.6346 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024)
        + 0.2146 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2)
        + 2.057 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        + 0.08692 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13)
        - 0.002887 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056)
        - 0.0001391 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7)
        + 3.113 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50)
        + 146.3 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset)
        - 122.8 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2)
        - 0.003982 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2)
        + 10.2 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5)
        + 6.952e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7)
        + 2.743e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0)
        - 3766.0 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7)
        + 17.34 * max(0.0, Q.LHA - 0.28)
        - 12.69 * max(0.0, Q.e2 - 0.034)
        + 751.6 * max(0.0, 0.0081 - Q.e2_sq)
        - 0.857 * max(0.0, Q.log_sum_pt - 6.4)
        + 0.4816 * max(0.0, Q.log_sum_pt - 6.6)
        + 0.01457 * max(0.0, 6.1 - Q.mass_top5)
        - 1.834 * max(0.0, 0.048 - Q.max_dr)
        + 2.507 * max(0.0, 0.25 - Q.max_dr)
        - 0.0003066 * max(0.0, Q.pt_7 - 34.0)
        - 0.004152 * max(0.0, Q.pt_7 - 54.0)
        - 1557.0 * max(0.0, 0.0087 - Q.width)
        - 2.443 * max(0.0, 0.044 - Q.z_7)
        - 2.256 * max(0.0, 0.054 - Q.z_7)
        + 250.5 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 7.916 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow)
        + 71.91 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98)
        + 0.05347 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1)
        + 1.488 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32)
        - 189.3 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow)
        - 3652.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02)
        + 82.15 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037)
        + 1.439 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2)
        - 49.37 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 44.89 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3)
        + 2.699 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr)
        - 0.001077 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0)
        - 0.01408 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21)
        + 15.95 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2)
        + 0.5641 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32)
        - 0.9404 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016)
        + 8.782e-05 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass)
        + 0.0001638 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3)
        - 0.184 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr)
        + 8646.0 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2)
        - 40.96 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2)
        + 4.551 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2)
        + 11.76 * max(0.0, Q.LHA - 0.1)
        + 4.374 * max(0.0, 3.7e-05 - Q.girth2_top2)
        + 91.71 * max(0.0, 0.0057 - Q.lam1)
        - 4.095 * max(0.0, Q.log_sum_pt - 6.3)
        + 1.334 * max(0.0, Q.log_sum_pt - 6.8)
        + 7.251 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.2874 * max(0.0, 6.5 - Q.log_sum_pt)
        + 0.04838 * max(0.0, 8.5 - Q.mass)
        + 0.016 * max(0.0, 36.0 - Q.mass)
        + 0.05131 * max(0.0, 69.0 - Q.mass)
        - 40.82 * max(0.0, 0.0098 - Q.mass_over_sum_pt)
        - 465.4 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 2.218 * max(0.0, 0.16 - Q.max_dr)
        - 0.2943 * max(0.0, 0.68 - Q.planar_flow)
        + 0.01231 * max(0.0, Q.pt_7 - 31.0)
        - 0.01339 * max(0.0, 54.0 - Q.pt_7)
        - 0.002976 * max(0.0, 790.0 - Q.sum_pt)
        - 2.949 * max(0.0, 0.017 - Q.z_7)
        - 7428.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset)
        + 382.1 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078)
        + 1.389 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79)
        + 0.8775 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0)
        - 1.033 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        - 22.64 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 9.61e-05 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 0.2759 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 1.232 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063)
        - 0.4767 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2)
        + 0.5548 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013)
        + 0.2081 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092)
        + 0.005154 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10)
        - 0.00206 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7)
        - 0.3131 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2)
        + 34.55 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7)
        + 19.1 * max(0.0, Q.C2 - 0.094)
        - 57.44 * max(0.0, Q.LHA - 0.32)
        - 44.79 * max(0.0, Q.centroid_offset - 0.013)
        - 50.81 * max(0.0, Q.e2 - 0.028)
        - 19.63 * max(0.0, Q.e2 - 0.051)
        - 74.97 * max(0.0, 0.043 - Q.e2)
        + 144.1 * max(0.0, 0.0088 - Q.girth2)
        - 374.5 * max(0.0, Q.lam1 - 0.016)
        - 0.07058 * max(0.0, Q.mass - 70.0)
        + 0.01643 * max(0.0, Q.mass_top5 - 53.0)
        - 7.17 * max(0.0, Q.max_dr - 0.15)
        + 128.9 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 224.2 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr)
        + 14.83 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012)
        - 0.2511 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5)
        - 1.14 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0)
        + 84.45 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi)
        - 45.15 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03)
        - 1.709 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 1596.0 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08)
        - 1524.0 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96)
        - 4.15 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6)
        + 11.76 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7)
        - 2404.0 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        + 0.1662 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6)
        - 3.4 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        - 1.582 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr)
        + 0.4425 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098)
        + 0.001354 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7)
        + 1006.0 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7)
        - 3608.0 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7)
        - 5115.0 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr)
        - 5.776 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        + 1.507 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 0.04544 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7)
        + 225.4 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1)
        + 103.9 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3)
        - 49.18 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0)
        + 0.3828 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4)
        - 248.6 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4)
        + 854.7 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion)
        + 1.918 * max(0.0, Q.C2 - 0.013)
        + 52.54 * max(0.0, Q.C2 - 0.067)
        - 11.45 * max(0.0, 0.014 - Q.centroid_offset)
        + 51.71 * max(0.0, Q.e2 - 0.02)
        - 352.4 * max(0.0, Q.e2 - 0.064)
        - 548.0 * max(0.0, 0.003 - Q.e2_sq)
        - 24.75 * max(0.0, 0.017 - Q.e2_sq)
        + 8.106 * max(0.0, 0.13 - Q.girth)
        + 161.6 * max(0.0, Q.girth2 - 0.0034)
        - 384.0 * max(0.0, Q.girth2 - 0.0081)
        + 1697.0 * max(0.0, 0.00033 - Q.lam2)
        - 0.0401 * max(0.0, 44.0 - Q.mass)
        - 73.29 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 128.8 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 2.359 * max(0.0, Q.max_dr - 0.094)
        - 7.501 * max(0.0, Q.max_dr - 0.2)
        + 0.0003377 * max(0.0, 760.0 - Q.sum_pt)
        + 0.001754 * max(0.0, 430.0 - Q.sum_pt_top5)
        - 3.643 * max(0.0, 0.24 - Q.tau21)
        + 0.1186 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0)
        - 0.8817 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7)
        + 20.32 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1)
        - 6080.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        - 0.1997 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044)
        - 119.1 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98)
        - 0.3734 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0)
        + 539.6 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 2436.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        + 0.04913 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass)
        - 172.9 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi)
        + 0.07242 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0)
        - 0.008743 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7)
        - 0.005105 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2)
        - 2197.0 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        - 6.938 * max(0.0, 0.023 - Q.dr_0)
        - 71.87 * max(0.0, 0.034 - Q.e2)
        - 7129.0 * max(0.0, 5.2e-05 - Q.girth2)
        + 0.01448 * max(0.0, 53.0 - Q.pt_7)
        - 0.0006389 * max(0.0, Q.sum_pt - 870.0)
        - 1.35e-05 * max(0.0, 540.0 - Q.sum_pt_top2)
        + 10.11 * max(0.0, 0.024 - Q.z_7)
        + 1159.0 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034)
        - 188.7 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028)
        - 1.611 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.03828 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5)
        + 2.902 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05)
        - 41.7 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow)
        - 3.152 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063)
        - 188.4 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017)
        - 65.33 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0)
        - 173.7 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        + 0.1832 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039)
        - 2930.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2)
        - 0.2876 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4)
        - 0.02106 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0)
        + 0.005383 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 1.311 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0)
        - 0.01615 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass)
        + 0.09446 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset)
        - 0.008884 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0)
        + 0.1403 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3)
        - 0.0004185 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2)
        + 29510.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 275.2 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        + 3.738 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 0.007893 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt)
        + 15.46 * max(0.0, Q.C2 - 0.033)
        - 3.326 * max(0.0, 0.034 - Q.C2)
        - 35.24 * max(0.0, Q.centroid_offset - 0.019)
        + 44.36 * max(0.0, Q.centroid_offset - 0.05)
        - 81.15 * max(0.0, 0.051 - Q.e2)
        + 3.508 * max(0.0, Q.girth - 0.083)
        - 200.3 * max(0.0, 0.0036 - Q.girth2)
        + 538.5 * max(0.0, 0.0086 - Q.girth2)
        - 41.64 * max(0.0, 0.01 - Q.girth2_top2)
        - 81.65 * max(0.0, Q.girth2_top5 - 0.011)
        - 132.4 * max(0.0, 0.0078 - Q.lam1)
        - 1120.0 * max(0.0, Q.lam2 - 0.0034)
        + 0.1832 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.03279 * max(0.0, 50.0 - Q.mass)
        + 42.72 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 1.721 * max(0.0, Q.max_dr - 0.12)
        - 8.936 * max(0.0, 0.11 - Q.max_dr)
        - 1.697 * max(0.0, 0.06 - Q.planar_flow)
        - 0.01814 * max(0.0, 25.0 - Q.pt_6)
        - 0.02163 * max(0.0, Q.sum_pt - 990.0)
        + 0.002383 * max(0.0, Q.sum_pt_top5 - 840.0)
        + 130.5 * max(0.0, 0.013 - Q.width)
        - 0.7259 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0)
        + 0.04272 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass)
        + 0.0008531 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4)
        - 91.73 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2)
        - 4778.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        + 1.621 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3)
        + 3773.0 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2)
        - 12.34 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5)
        - 1.356 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0)
        - 88.42 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16)
        + 1209.0 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13)
        - 368.6 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi)
        - 133.2 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014)
        + 2.398 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        + 3375.0 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 13.22 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098)
        + 0.01624 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0)
        - 0.05522 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6)
        - 0.05763 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 4.085 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4)
        - 174.6 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        - 124.6 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        - 0.19 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15)
        - 0.02093 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        - 0.1684 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7)
        + 15.61 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        + 4.157e-05 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0)
        + 994.7 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 1404.0 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026)
        + 27.6 * max(0.0, 0.021 - Q.centroid_offset)
        + 253.5 * max(0.0, 0.0012 - Q.e2_sq)
        - 3.316 * max(0.0, 0.088 - Q.girth)
        - 826.9 * max(0.0, Q.girth2 - 0.0015)
        + 1311.0 * max(0.0, Q.girth2 - 0.0044)
        - 1379.0 * max(0.0, Q.girth2 - 0.0075)
        + 1681.0 * max(0.0, Q.girth2 - 0.0087)
        - 376.1 * max(0.0, Q.girth2 - 0.015)
        + 550.9 * max(0.0, 0.0011 - Q.girth2_top2)
        + 1006.0 * max(0.0, 0.0084 - Q.lam1)
        + 0.05315 * max(0.0, Q.mass - 80.4)
        + 56.56 * max(0.0, Q.mass_over_sum_pt - 0.072)
        + 72.08 * max(0.0, Q.mass_over_sum_pt - 0.085)
        - 273.6 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 1258.0 * max(0.0, 0.00052 - Q.width)
        - 1515.0 * max(0.0, 0.0055 - Q.width)
        - 1.273 * max(0.0, Q.z_dr_0p05_0p1 - 0.66)
        + 563.7 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025)
        - 1814.0 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068)
        - 0.3664 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0)
        - 0.4201 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0)
        + 0.1474 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 55.54 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 89.86 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 2476.0 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055)
        - 54.23 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21)
        + 1679.0 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95)
        + 688.1 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 1227.0 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3)
        - 725.0 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 6.124 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0)
        + 9.907 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88)
        - 392.6 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 2.79 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50)
        - 0.2917 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92)
        - 0.001112 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012)
        - 28.43 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        - 0.2432 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2)
        - 13.68 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 9.305 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 2.744 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2)
        - 0.279 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2)
        + 0.004716 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0)
        + 0.002356 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow)
        - 1616.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi)
        + 244.1 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        - 226.7 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        - 7.328 * max(0.0, 0.027 - Q.C2)
        - 120.5 * max(0.0, 0.0034 - Q.centroid_offset)
        + 45.64 * max(0.0, 0.016 - Q.dr_0)
        - 16.04 * max(0.0, Q.log_sum_pt - 6.7)
        - 874.2 * max(0.0, 0.0049 - Q.width)
        + 841.7 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075)
        + 227.5 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075)
        - 16650.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 17.51 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        + 1719.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        - 20360.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        - 34.98 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow)
        - 5265.0 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029)
        - 0.03598 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7)
        - 5.516 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031)
        + 3.786 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        - 0.03449 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 927.6 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2)
        + 17430.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        + 22440.0 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045)
        + 2013.0 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2)
        - 18.97 * max(0.0, Q.C2 - 0.051)
        + 29.2 * max(0.0, 0.018 - Q.centroid_offset)
        + 40.53 * max(0.0, 0.055 - Q.girth)
        - 289.9 * max(0.0, Q.girth2 - 0.018)
        - 233.7 * max(0.0, Q.lam2 - 0.0017)
        + 54.72 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        - 45.06 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 0.05603 * max(0.0, 0.23 - Q.max_dr)
        + 8.002 * max(0.0, 0.00042 - Q.mean_phi)
        - 0.1437 * max(0.0, Q.n_dr_0p2_0p4 - 0.9)
        - 0.0005424 * max(0.0, Q.sum_pt - 850.0)
        - 134.6 * max(0.0, 0.0061 - Q.width)
        - 3.426 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5)
        + 0.1228 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2)
        - 0.202 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0)
        + 172.9 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043)
        + 172.3 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033)
        - 1455.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        - 63.89 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01)
        - 1.186 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7)
        + 76.97 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21)
        + 9694.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 93.13 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta)
        - 9.193 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow)
        + 7.429 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0)
        - 3173.0 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023)
        + 0.679 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0)
        - 29.56 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026)
        + 3.816 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        - 3.034 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        - 0.06941 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.06939 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 0.07993 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 12430.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        - 0.2277 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22)
        - 1751.0 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031)
        + 16570.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 4.464 * max(0.0, Q.C2 - 0.055)
        + 7.012 * max(0.0, Q.LHA - 0.33)
        - 47.98 * max(0.0, Q.centroid_offset - 0.038)
        - 12.32 * max(0.0, Q.e2 - 0.036)
        + 6.032 * max(0.0, 0.037 - Q.e2)
        + 1977.0 * max(0.0, Q.girth2 - 0.0085)
        + 234.3 * max(0.0, 0.0017 - Q.girth2)
        + 75.21 * max(0.0, 0.0038 - Q.girth2_top2)
        - 7.437 * max(0.0, 0.0016 - Q.girth2_top3)
        - 628.0 * max(0.0, Q.lam1 - 0.0085)
        + 611.5 * max(0.0, 0.0044 - Q.lam1)
        + 1282.0 * max(0.0, 0.0034 - Q.lam2)
        + 2.44 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.03775 * max(0.0, Q.mass - 17.0)
        - 30.39 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 0.003967 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 0.1059 * max(0.0, Q.n_dr_0p2_0p4 - 1.9)
        - 9.892e-05 * max(0.0, Q.pt_7 - 46.0)
        + 1.351 * max(0.0, 0.27 - Q.tau32)
        + 1.734 * max(0.0, Q.z_7 - 0.062)
        - 19.89 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21)
        - 0.02661 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2)
        + 8.341 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4)
        + 1757.0 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 0.4696 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass)
        - 24.23 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50)
        + 576.5 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21)
        - 0.001828 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 1.027 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 25.54 * max(0.0, 0.035 - Q.C2)
        - 11.96 * max(0.0, Q.centroid_offset - 0.014)
        + 5.557 * max(0.0, 0.04 - Q.centroid_offset)
        - 555.5 * max(0.0, 0.0062 - Q.e2_sq)
        - 97.48 * max(0.0, Q.girth - 0.076)
        - 4.093 * max(0.0, 0.021 - Q.girth)
        - 54.03 * max(0.0, 0.089 - Q.girth)
        - 154.2 * max(0.0, 0.002 - Q.girth2_top3)
        + 0.03154 * max(0.0, 5.5 - Q.mass_top5)
        + 5.018 * max(0.0, 0.22 - Q.max_dr)
        - 0.0003074 * max(0.0, 2.8 - Q.n_dr_0p1_0p2)
        + 1.728 * max(0.0, 0.27 - Q.planar_flow)
        + 0.002186 * max(0.0, 690.0 - Q.sum_pt_top5)
        - 200.3 * max(0.0, 0.0036 - Q.width)
        - 1257.0 * max(0.0, 0.0085 - Q.width)
        + 339.0 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7)
        - 22.69 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        + 294.8 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21)
        + 5990.0 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014)
        + 8594.0 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi)
        - 5.635 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 5.798 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        + 0.07417 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7)
        + 1.291 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0)
        - 4042.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi)
        - 2702.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032)
        - 6.466 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0)
        + 0.06333 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 55.33 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        + 0.006322 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass)
        - 1.518 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11)
        - 436.3 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        - 0.01919 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        + 306.6 * max(0.0, Q.e2 - 0.063)
        + 135.3 * max(0.0, Q.girth2 - 0.019)
        - 0.1019 * max(0.0, Q.mass - 91.2)
        - 10760.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        + 1.198 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0)
        - 3.613 * max(0.0, Q.C2 - 0.066)
        - 34.24 * max(0.0, 0.038 - Q.centroid_offset)
        + 63.75 * max(0.0, 0.049 - Q.e2)
        + 1.621 * max(0.0, 0.15 - Q.girth)
        + 63.13 * max(0.0, 0.0062 - Q.lam1)
        + 468.9 * max(0.0, 0.015 - Q.lam1)
        - 0.009635 * max(0.0, 25.0 - Q.pt_7)
        - 0.001268 * max(0.0, Q.sum_pt - 1000.0)
        + 0.8425 * max(0.0, 0.5 - Q.tau21)
        + 2765.0 * max(0.0, 0.0076 - Q.width)
        - 3.65 * max(0.0, 0.027 - Q.z_7)
        + 8.095 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4)
        - 14.05 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.1115 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        + 5726.0 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 203.9 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18)
        - 40480.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        - 0.0001069 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2)
        + 0.002194 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.6824 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4)
        - 6.025e-05 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2)
        + 0.0001761 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88)
        - 0.003623 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1)
        + 0.0007675 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5)
        + 3.577e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 0.005875 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32)
        - 0.06558 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023)
        - 7.734 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013)
        + 1.577 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0)
        + 11.69 * max(0.0, 0.067 - Q.C2)
        + 0.3489 * max(0.0, 0.8 - Q.D2)
        + 56.86 * max(0.0, Q.centroid_offset - 0.031)
        - 491.0 * max(0.0, 0.0058 - Q.e2_sq)
        + 19.76 * max(0.0, 0.034 - Q.girth)
        + 108.7 * max(0.0, 0.087 - Q.girth)
        + 582.9 * max(0.0, Q.lam1 - 0.0025)
        - 1044.0 * max(0.0, Q.lam1 - 0.0042)
        - 79.12 * max(0.0, Q.lam1 - 0.0061)
        + 1541.0 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 9.54 * max(0.0, 0.078 - Q.max_dr)
        - 1.48 * max(0.0, 0.18 - Q.max_dr)
        - 0.008538 * max(0.0, 4.8 - Q.n_dr_0p05_0p1)
        + 0.0008728 * max(0.0, 310.0 - Q.sum_pt_top3)
        - 9.826 * max(0.0, 0.14 - Q.tau21)
        - 0.8206 * max(0.0, 0.6 - Q.z_dr_0p05_0p1)
        - 35.83 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        - 110.2 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2)
        + 891.4 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        - 4.379 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0)
        + 38.11 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 393.7 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        - 489.3 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        + 177.3 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6)
        + 331.0 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        + 10000.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        - 221.6 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094)
        + 434.5 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        - 17.06 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr)
        + 0.002449 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt)
        + 1222.0 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 17.75 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0)
        - 1629.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta)
        - 9940.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 206.9 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 111.0 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow)
        + 22.18 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2)
        + 0.07637 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 1.521 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 65.33 * max(0.0, Q.LHA - 0.34)
        + 192.7 * max(0.0, 0.024 - Q.e2)
        - 96.67 * max(0.0, 0.041 - Q.e2)
        - 14.53 * max(0.0, Q.girth - 0.032)
        + 16.73 * max(0.0, 0.0074 - Q.girth2_top2)
        - 786.3 * max(0.0, 0.0067 - Q.lam1)
        - 256.7 * max(0.0, 0.0083 - Q.lam1)
        - 79.47 * max(0.0, 0.00031 - Q.lam2)
        - 12.69 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        + 1159.0 * max(0.0, 0.0067 - Q.width)
        + 4.455 * max(0.0, Q.z_dr_0p05_0p1 - 0.75)
        - 0.4527 * max(0.0, 0.32 - Q.z_dr_0p1_0p2)
        - 48.08 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion)
        - 0.01593 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0)
        + 672.2 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2)
        + 3.892 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0)
        - 4.426 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3)
        - 2996.0 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017)
        + 0.1766 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 2.042 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion)
        + 187.8 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084)
        + 2.525 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1)
        - 34.87 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 71340.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        + 2377.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 1429.0 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow)
    )


def score_Z(Q):
    return (-16.24
        + 0.0146 * max(0.0, Q.D2 - 3.9)
        + 32.16 * max(0.0, 0.033 - Q.centroid_offset)
        - 1.954 * max(0.0, 0.078 - Q.girth)
        + 730.1 * max(0.0, 0.013 - Q.girth2)
        - 15.81 * max(0.0, 0.0085 - Q.girth2_top5)
        - 65.75 * max(0.0, 0.0006 - Q.lam1)
        - 341.1 * max(0.0, 0.0015 - Q.lam1)
        - 0.0003329 * max(0.0, 22.0 - Q.mass)
        + 0.03713 * max(0.0, 30.0 - Q.mass)
        + 0.001474 * max(0.0, 59.0 - Q.mass)
        - 0.02307 * max(0.0, Q.n_dr_0_0p05 - 3.8)
        + 0.8703 * max(0.0, 0.013 - Q.planar_flow)
        + 0.04233 * max(0.0, Q.sum_pt - 810.0)
        - 0.01983 * max(0.0, Q.sum_pt - 900.0)
        + 0.0006322 * max(0.0, Q.sum_pt_top5 - 700.0)
        + 0.04574 * max(0.0, Q.tau32 - 0.44)
        - 1041.0 * max(0.0, 0.0044 - Q.width)
        + 3623.0 * max(0.0, 0.0089 - Q.width)
        + 0.3391 * max(0.0, Q.z_dr_0_0p05 - 0.85)
        - 1.04 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2)
        + 0.7166 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0)
        - 499.3 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96)
        - 0.4047 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0)
        - 12.9 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0)
        + 638.9 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        + 6.443 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0)
        - 0.3586 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024)
        - 0.00311 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2)
        + 1.441 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        - 0.2305 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13)
        + 0.03523 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056)
        + 0.0005175 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7)
        - 0.4104 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50)
        - 18.85 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset)
        - 13.04 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2)
        + 0.001711 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2)
        + 3.108 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5)
        - 6.777e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7)
        - 3.169e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0)
        - 222.4 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7)
        + 7.862 * max(0.0, Q.LHA - 0.28)
        - 9.4 * max(0.0, Q.e2 - 0.034)
        - 1198.0 * max(0.0, 0.0081 - Q.e2_sq)
        + 0.9012 * max(0.0, Q.log_sum_pt - 6.4)
        + 1.758 * max(0.0, Q.log_sum_pt - 6.6)
        - 0.007217 * max(0.0, 6.1 - Q.mass_top5)
        - 0.1493 * max(0.0, 0.048 - Q.max_dr)
        - 0.7077 * max(0.0, 0.25 - Q.max_dr)
        + 0.03647 * max(0.0, Q.pt_7 - 34.0)
        - 0.02648 * max(0.0, Q.pt_7 - 54.0)
        - 189.2 * max(0.0, 0.0087 - Q.width)
        - 11.81 * max(0.0, 0.044 - Q.z_7)
        - 19.38 * max(0.0, 0.054 - Q.z_7)
        + 86.43 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        + 1.225 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow)
        + 1825.0 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98)
        - 0.1271 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1)
        - 24.39 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32)
        - 1310.0 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow)
        + 10600.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02)
        - 2.629 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037)
        + 1.827 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2)
        - 115.7 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        - 41.23 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3)
        - 0.1735 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr)
        - 0.5313 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 0.08878 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21)
        - 47.98 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2)
        - 2.263 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32)
        - 0.3028 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016)
        - 0.0005589 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass)
        + 0.0006619 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3)
        - 0.1385 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr)
        - 1345.0 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2)
        + 687.0 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2)
        - 1.236 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2)
        - 3.011 * max(0.0, Q.LHA - 0.1)
        - 240.8 * max(0.0, 3.7e-05 - Q.girth2_top2)
        + 176.7 * max(0.0, 0.0057 - Q.lam1)
        - 2.594 * max(0.0, Q.log_sum_pt - 6.3)
        + 9.69 * max(0.0, Q.log_sum_pt - 6.8)
        + 19.21 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.6141 * max(0.0, 6.5 - Q.log_sum_pt)
        + 0.01585 * max(0.0, 8.5 - Q.mass)
        + 0.03256 * max(0.0, 36.0 - Q.mass)
        - 0.03302 * max(0.0, 69.0 - Q.mass)
        - 12.45 * max(0.0, 0.0098 - Q.mass_over_sum_pt)
        - 397.0 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        - 28.28 * max(0.0, 0.16 - Q.max_dr)
        + 0.151 * max(0.0, 0.68 - Q.planar_flow)
        + 0.005137 * max(0.0, Q.pt_7 - 31.0)
        - 0.01598 * max(0.0, 54.0 - Q.pt_7)
        - 0.006899 * max(0.0, 790.0 - Q.sum_pt)
        + 7.322 * max(0.0, 0.017 - Q.z_7)
        - 4250.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset)
        - 1821.0 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078)
        + 1.254 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79)
        - 0.9318 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0)
        - 0.9442 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        - 7.834 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 5.484e-05 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 0.112 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 0.9507 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063)
        - 0.003489 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2)
        + 0.4561 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013)
        + 0.1989 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092)
        - 0.002025 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10)
        + 0.0006146 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7)
        - 0.1021 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2)
        + 13.17 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7)
        + 31.72 * max(0.0, Q.C2 - 0.094)
        - 6.885 * max(0.0, Q.LHA - 0.32)
        - 65.52 * max(0.0, Q.centroid_offset - 0.013)
        - 80.15 * max(0.0, Q.e2 - 0.028)
        - 22.46 * max(0.0, Q.e2 - 0.051)
        + 110.2 * max(0.0, 0.043 - Q.e2)
        - 3955.0 * max(0.0, 0.0088 - Q.girth2)
        - 367.3 * max(0.0, Q.lam1 - 0.016)
        + 0.03132 * max(0.0, Q.mass - 70.0)
        + 0.02624 * max(0.0, Q.mass_top5 - 53.0)
        - 3.319 * max(0.0, Q.max_dr - 0.15)
        - 496.7 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 995.9 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr)
        - 12.71 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012)
        - 2.167 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5)
        - 1.14 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0)
        + 287.5 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi)
        - 49.24 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03)
        + 1.897 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 2415.0 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08)
        + 2158.0 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96)
        - 1.836 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6)
        + 8.198 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7)
        + 1330.0 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        + 0.142 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6)
        - 8.668 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        - 0.7612 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr)
        + 0.8327 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098)
        - 0.0001549 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7)
        + 920.6 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7)
        - 2781.0 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7)
        - 3169.0 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr)
        - 1.449 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        + 63.34 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        + 0.07478 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7)
        + 186.9 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1)
        + 95.3 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3)
        - 91.1 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0)
        + 0.6543 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4)
        - 362.7 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4)
        - 3.559 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion)
        - 3.896 * max(0.0, Q.C2 - 0.013)
        + 83.71 * max(0.0, Q.C2 - 0.067)
        + 4.185 * max(0.0, 0.014 - Q.centroid_offset)
        + 59.24 * max(0.0, Q.e2 - 0.02)
        - 479.4 * max(0.0, Q.e2 - 0.064)
        - 590.8 * max(0.0, 0.003 - Q.e2_sq)
        + 26.9 * max(0.0, 0.017 - Q.e2_sq)
        + 31.17 * max(0.0, 0.13 - Q.girth)
        + 933.3 * max(0.0, Q.girth2 - 0.0034)
        - 186.1 * max(0.0, Q.girth2 - 0.0081)
        + 1660.0 * max(0.0, 0.00033 - Q.lam2)
        - 0.03129 * max(0.0, 44.0 - Q.mass)
        - 13.36 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 87.45 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 5.642 * max(0.0, Q.max_dr - 0.094)
        - 9.173 * max(0.0, Q.max_dr - 0.2)
        + 0.0003993 * max(0.0, 760.0 - Q.sum_pt)
        + 0.002621 * max(0.0, 430.0 - Q.sum_pt_top5)
        - 2.236 * max(0.0, 0.24 - Q.tau21)
        + 0.7631 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0)
        + 0.1909 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7)
        - 18.96 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1)
        - 1401.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        - 0.5613 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044)
        - 54.19 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98)
        - 0.3999 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0)
        + 1388.0 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        + 1213.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        - 0.06508 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass)
        - 14.44 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi)
        - 0.005177 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0)
        - 0.1051 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7)
        + 0.002745 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2)
        - 1374.0 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        + 6.697 * max(0.0, 0.023 - Q.dr_0)
        - 89.66 * max(0.0, 0.034 - Q.e2)
        + 58.66 * max(0.0, 5.2e-05 - Q.girth2)
        + 0.0175 * max(0.0, 53.0 - Q.pt_7)
        - 0.0009013 * max(0.0, Q.sum_pt - 870.0)
        + 8.213e-05 * max(0.0, 540.0 - Q.sum_pt_top2)
        + 17.33 * max(0.0, 0.024 - Q.z_7)
        + 536.2 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034)
        - 25.15 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028)
        + 1.002 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.03417 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5)
        - 1.167 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05)
        - 24.01 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow)
        - 8.262 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063)
        - 115.4 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017)
        - 37.39 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0)
        - 41.3 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        + 2.941 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039)
        - 1660.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2)
        - 0.323 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4)
        - 0.001093 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0)
        + 0.0005199 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 1.04 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0)
        + 0.1041 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass)
        + 0.09372 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset)
        - 0.006867 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0)
        + 0.03872 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3)
        - 0.0008475 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2)
        - 1963.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 178.3 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 0.02669 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 0.02196 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt)
        - 3.3 * max(0.0, Q.C2 - 0.033)
        + 8.582 * max(0.0, 0.034 - Q.C2)
        - 64.4 * max(0.0, Q.centroid_offset - 0.019)
        + 74.37 * max(0.0, Q.centroid_offset - 0.05)
        - 97.09 * max(0.0, 0.051 - Q.e2)
        - 17.77 * max(0.0, Q.girth - 0.083)
        - 151.2 * max(0.0, 0.0036 - Q.girth2)
        + 761.0 * max(0.0, 0.0086 - Q.girth2)
        - 66.62 * max(0.0, 0.01 - Q.girth2_top2)
        - 103.1 * max(0.0, Q.girth2_top5 - 0.011)
        - 509.0 * max(0.0, 0.0078 - Q.lam1)
        - 548.2 * max(0.0, Q.lam2 - 0.0034)
        + 5.725 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.03785 * max(0.0, 50.0 - Q.mass)
        + 30.46 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        - 1.503 * max(0.0, Q.max_dr - 0.12)
        - 0.5288 * max(0.0, 0.11 - Q.max_dr)
        + 0.8901 * max(0.0, 0.06 - Q.planar_flow)
        - 0.01688 * max(0.0, 25.0 - Q.pt_6)
        - 0.02573 * max(0.0, Q.sum_pt - 990.0)
        + 0.001664 * max(0.0, Q.sum_pt_top5 - 840.0)
        + 730.1 * max(0.0, 0.013 - Q.width)
        - 0.8657 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0)
        + 0.0442 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass)
        + 0.002652 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4)
        + 105.7 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2)
        + 1527.0 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        + 2.435 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3)
        + 5882.0 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2)
        - 11.01 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5)
        - 1.517 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0)
        + 3.972 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16)
        + 1887.0 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13)
        - 151.7 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi)
        + 591.4 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014)
        + 3.823 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 93.05 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        + 25.29 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098)
        - 0.0002224 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0)
        - 0.08809 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6)
        - 0.1142 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 4.081 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4)
        - 174.4 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        - 97.02 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        - 0.2396 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15)
        - 0.02545 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        - 0.01019 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7)
        + 12.02 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        + 5.787e-05 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0)
        - 965.4 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 3117.0 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026)
        - 23.28 * max(0.0, 0.021 - Q.centroid_offset)
        - 313.1 * max(0.0, 0.0012 - Q.e2_sq)
        + 2.593 * max(0.0, 0.088 - Q.girth)
        - 81.65 * max(0.0, Q.girth2 - 0.0015)
        + 423.1 * max(0.0, Q.girth2 - 0.0044)
        - 1417.0 * max(0.0, Q.girth2 - 0.0075)
        + 1301.0 * max(0.0, Q.girth2 - 0.0087)
        - 439.1 * max(0.0, Q.girth2 - 0.015)
        - 447.6 * max(0.0, 0.0011 - Q.girth2_top2)
        + 316.7 * max(0.0, 0.0084 - Q.lam1)
        - 0.0479 * max(0.0, Q.mass - 80.4)
        + 33.38 * max(0.0, Q.mass_over_sum_pt - 0.072)
        + 103.9 * max(0.0, Q.mass_over_sum_pt - 0.085)
        - 351.4 * max(0.0, Q.mass_over_sum_pt - 0.091)
        - 384.5 * max(0.0, 0.00052 - Q.width)
        - 158.1 * max(0.0, 0.0055 - Q.width)
        - 1.385 * max(0.0, Q.z_dr_0p05_0p1 - 0.66)
        + 506.0 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025)
        - 1345.0 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068)
        - 0.4991 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0)
        - 0.4898 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0)
        + 0.1322 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        - 145.5 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 56.69 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 1422.0 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055)
        - 12.64 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21)
        + 1513.0 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95)
        - 69.66 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        - 1243.0 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3)
        + 841.9 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 79.9 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0)
        + 38.22 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88)
        - 342.4 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        - 13.42 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50)
        - 0.6491 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92)
        - 0.002707 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012)
        - 22.23 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        + 13.54 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2)
        + 43.75 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 32.49 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 1.621 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2)
        - 0.2744 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2)
        + 0.007159 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0)
        - 0.01639 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow)
        - 3973.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi)
        - 109.9 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        - 483.3 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        + 17.15 * max(0.0, 0.027 - Q.C2)
        - 56.28 * max(0.0, 0.0034 - Q.centroid_offset)
        - 0.5336 * max(0.0, 0.016 - Q.dr_0)
        - 27.68 * max(0.0, Q.log_sum_pt - 6.7)
        + 159.7 * max(0.0, 0.0049 - Q.width)
        + 11060.0 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075)
        + 42.99 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075)
        - 2212.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 26.0 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        + 13620.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 5478.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 6.744 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow)
        - 620.5 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029)
        - 0.05764 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7)
        + 0.4757 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031)
        - 1.345 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        + 0.004098 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        - 10370.0 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2)
        + 1133.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        + 20640.0 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045)
        - 2358.0 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2)
        - 7.097 * max(0.0, Q.C2 - 0.051)
        + 50.76 * max(0.0, 0.018 - Q.centroid_offset)
        - 17.86 * max(0.0, 0.055 - Q.girth)
        - 274.9 * max(0.0, Q.girth2 - 0.018)
        - 221.0 * max(0.0, Q.lam2 - 0.0017)
        + 29.37 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        - 162.5 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 4.158 * max(0.0, 0.23 - Q.max_dr)
        + 5.561 * max(0.0, 0.00042 - Q.mean_phi)
        - 0.0005074 * max(0.0, Q.n_dr_0p2_0p4 - 0.9)
        - 0.001029 * max(0.0, Q.sum_pt - 850.0)
        - 766.9 * max(0.0, 0.0061 - Q.width)
        - 7.284 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5)
        - 0.0355 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2)
        + 0.04675 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0)
        + 25.01 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043)
        + 110.3 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033)
        + 1096.0 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        + 79.93 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01)
        - 1.775 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7)
        + 47.11 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21)
        + 15430.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        + 285.4 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta)
        - 282.8 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow)
        + 2.117 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0)
        - 5820.0 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023)
        + 1.822 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0)
        + 3.026 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026)
        + 4.0 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        - 1.669 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        - 0.07274 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.06253 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 0.03678 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 3100.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        + 0.5341 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22)
        - 2640.0 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031)
        + 15660.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 11.85 * max(0.0, Q.C2 - 0.055)
        - 19.85 * max(0.0, Q.LHA - 0.33)
        + 20.68 * max(0.0, Q.centroid_offset - 0.038)
        + 25.45 * max(0.0, Q.e2 - 0.036)
        - 9.121 * max(0.0, 0.037 - Q.e2)
        + 947.4 * max(0.0, Q.girth2 - 0.0085)
        + 627.2 * max(0.0, 0.0017 - Q.girth2)
        - 7.346 * max(0.0, 0.0038 - Q.girth2_top2)
        + 57.19 * max(0.0, 0.0016 - Q.girth2_top3)
        - 946.2 * max(0.0, Q.lam1 - 0.0085)
        + 324.1 * max(0.0, 0.0044 - Q.lam1)
        + 567.8 * max(0.0, 0.0034 - Q.lam2)
        + 1.409 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.01236 * max(0.0, Q.mass - 17.0)
        - 35.9 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 0.04113 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        - 0.01394 * max(0.0, Q.n_dr_0p2_0p4 - 1.9)
        + 0.003465 * max(0.0, Q.pt_7 - 46.0)
        + 1.114 * max(0.0, 0.27 - Q.tau32)
        - 2.183 * max(0.0, Q.z_7 - 0.062)
        - 26.16 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21)
        - 0.01236 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2)
        + 29.32 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4)
        + 389.6 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        + 3.042 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass)
        - 0.5892 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50)
        + 195.0 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21)
        + 0.00171 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 0.7629 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 3.483 * max(0.0, 0.035 - Q.C2)
        + 48.3 * max(0.0, Q.centroid_offset - 0.014)
        - 49.07 * max(0.0, 0.04 - Q.centroid_offset)
        + 80.31 * max(0.0, 0.0062 - Q.e2_sq)
        + 2.821 * max(0.0, Q.girth - 0.076)
        + 18.87 * max(0.0, 0.021 - Q.girth)
        - 85.33 * max(0.0, 0.089 - Q.girth)
        - 48.85 * max(0.0, 0.002 - Q.girth2_top3)
        - 0.003749 * max(0.0, 5.5 - Q.mass_top5)
        + 10.9 * max(0.0, 0.22 - Q.max_dr)
        - 0.0054 * max(0.0, 2.8 - Q.n_dr_0p1_0p2)
        - 0.7453 * max(0.0, 0.27 - Q.planar_flow)
        - 0.001107 * max(0.0, 690.0 - Q.sum_pt_top5)
        - 151.2 * max(0.0, 0.0036 - Q.width)
        - 541.4 * max(0.0, 0.0085 - Q.width)
        + 72.27 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7)
        + 23.01 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 134.0 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21)
        + 1255.0 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014)
        + 6190.0 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi)
        - 1.545 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 2.156 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        + 0.1452 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7)
        - 1.575 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0)
        - 1919.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi)
        - 521.7 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032)
        + 0.1479 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0)
        - 0.01594 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 480.2 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        + 0.008316 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass)
        + 4.988 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11)
        + 549.7 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        - 0.005105 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        + 474.3 * max(0.0, Q.e2 - 0.063)
        + 87.68 * max(0.0, Q.girth2 - 0.019)
        + 0.01752 * max(0.0, Q.mass - 91.2)
        - 9828.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        + 4.155 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0)
        - 50.65 * max(0.0, Q.C2 - 0.066)
        - 15.96 * max(0.0, 0.038 - Q.centroid_offset)
        + 0.2617 * max(0.0, 0.049 - Q.e2)
        + 5.357 * max(0.0, 0.15 - Q.girth)
        - 117.1 * max(0.0, 0.0062 - Q.lam1)
        + 423.1 * max(0.0, 0.015 - Q.lam1)
        + 0.001594 * max(0.0, 25.0 - Q.pt_7)
        - 0.004826 * max(0.0, Q.sum_pt - 1000.0)
        + 0.1325 * max(0.0, 0.5 - Q.tau21)
        + 1197.0 * max(0.0, 0.0076 - Q.width)
        - 9.242 * max(0.0, 0.027 - Q.z_7)
        + 12.62 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4)
        - 6.105 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.07559 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        - 34.72 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        + 110.7 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18)
        - 16450.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        + 0.0006936 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2)
        + 0.0003597 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.5472 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4)
        - 0.000245 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2)
        + 7.244e-05 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88)
        + 0.0003889 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1)
        + 0.001099 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5)
        + 2.538e-05 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        - 0.008269 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32)
        - 0.02646 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023)
        - 0.4932 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013)
        - 1.799 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0)
        - 27.03 * max(0.0, 0.067 - Q.C2)
        - 0.9048 * max(0.0, 0.8 - Q.D2)
        - 41.94 * max(0.0, Q.centroid_offset - 0.031)
        - 159.4 * max(0.0, 0.0058 - Q.e2_sq)
        - 8.849 * max(0.0, 0.034 - Q.girth)
        - 0.8211 * max(0.0, 0.087 - Q.girth)
        + 92.86 * max(0.0, Q.lam1 - 0.0025)
        - 190.4 * max(0.0, Q.lam1 - 0.0042)
        + 238.5 * max(0.0, Q.lam1 - 0.0061)
        + 2216.0 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        + 6.696 * max(0.0, 0.078 - Q.max_dr)
        - 9.1 * max(0.0, 0.18 - Q.max_dr)
        - 0.0592 * max(0.0, 4.8 - Q.n_dr_0p05_0p1)
        - 0.001109 * max(0.0, 310.0 - Q.sum_pt_top3)
        - 0.472 * max(0.0, 0.14 - Q.tau21)
        + 0.3465 * max(0.0, 0.6 - Q.z_dr_0p05_0p1)
        + 7.896 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        + 78.7 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2)
        + 6763.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 4.131 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0)
        - 50.59 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 385.7 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 141.8 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        - 156.1 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6)
        - 239.3 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        + 4012.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 284.2 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094)
        - 403.8 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 3.094 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr)
        - 0.0115 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt)
        - 538.3 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        - 15.82 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0)
        - 370.1 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta)
        - 6840.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        + 154.8 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 3120.0 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow)
        - 8.803 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2)
        + 0.2134 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.8653 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 44.9 * max(0.0, Q.LHA - 0.34)
        + 145.6 * max(0.0, 0.024 - Q.e2)
        - 43.56 * max(0.0, 0.041 - Q.e2)
        + 2.472 * max(0.0, Q.girth - 0.032)
        + 67.39 * max(0.0, 0.0074 - Q.girth2_top2)
        - 139.9 * max(0.0, 0.0067 - Q.lam1)
        + 13.11 * max(0.0, 0.0083 - Q.lam1)
        - 1063.0 * max(0.0, 0.00031 - Q.lam2)
        - 26.09 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 224.3 * max(0.0, 0.0067 - Q.width)
        - 1.443 * max(0.0, Q.z_dr_0p05_0p1 - 0.75)
        + 1.384 * max(0.0, 0.32 - Q.z_dr_0p1_0p2)
        + 4.119 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion)
        - 0.01099 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0)
        + 2.725 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2)
        - 0.6953 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0)
        + 2.103 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3)
        + 187.2 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017)
        - 0.05556 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 1.408 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion)
        + 71.44 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084)
        + 1.998 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1)
        - 1.127 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 37450.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        + 940.9 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 480.6 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow)
    )


def score_t(Q):
    return (0.2809
        + 0.01966 * max(0.0, Q.D2 - 3.9)
        + 8.056 * max(0.0, 0.033 - Q.centroid_offset)
        - 10.8 * max(0.0, 0.078 - Q.girth)
        - 29.94 * max(0.0, 0.013 - Q.girth2)
        - 9.965 * max(0.0, 0.0085 - Q.girth2_top5)
        - 223.4 * max(0.0, 0.0006 - Q.lam1)
        - 73.04 * max(0.0, 0.0015 - Q.lam1)
        - 0.006112 * max(0.0, 22.0 - Q.mass)
        - 0.005382 * max(0.0, 30.0 - Q.mass)
        + 0.01606 * max(0.0, 59.0 - Q.mass)
        - 0.002355 * max(0.0, Q.n_dr_0_0p05 - 3.8)
        + 1.734 * max(0.0, 0.013 - Q.planar_flow)
        + 0.01073 * max(0.0, Q.sum_pt - 810.0)
        - 0.003377 * max(0.0, Q.sum_pt - 900.0)
        + 0.0001979 * max(0.0, Q.sum_pt_top5 - 700.0)
        + 0.02115 * max(0.0, Q.tau32 - 0.44)
        + 128.2 * max(0.0, 0.0044 - Q.width)
        + 52.17 * max(0.0, 0.0089 - Q.width)
        + 0.6938 * max(0.0, Q.z_dr_0_0p05 - 0.85)
        - 1.443 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2)
        + 0.4566 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0)
        - 56.42 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96)
        - 0.4184 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0)
        + 10.99 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0)
        - 55.98 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2)
        - 2.157 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0)
        - 0.3364 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024)
        + 0.02477 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2)
        - 0.2679 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012)
        - 0.1888 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13)
        - 0.02801 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056)
        - 9.038e-05 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7)
        - 3.146 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50)
        - 0.2991 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset)
        + 11.85 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2)
        - 0.008488 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2)
        + 6.636 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5)
        + 9.783e-05 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7)
        - 0.0001063 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0)
        + 173.6 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7)
        - 1.724 * max(0.0, Q.LHA - 0.28)
        - 14.07 * max(0.0, Q.e2 - 0.034)
        - 521.1 * max(0.0, 0.0081 - Q.e2_sq)
        - 1.186 * max(0.0, Q.log_sum_pt - 6.4)
        - 1.796 * max(0.0, Q.log_sum_pt - 6.6)
        + 0.02668 * max(0.0, 6.1 - Q.mass_top5)
        - 1.205 * max(0.0, 0.048 - Q.max_dr)
        + 0.7071 * max(0.0, 0.25 - Q.max_dr)
        - 0.01751 * max(0.0, Q.pt_7 - 34.0)
        + 0.03074 * max(0.0, Q.pt_7 - 54.0)
        + 188.0 * max(0.0, 0.0087 - Q.width)
        - 2.869 * max(0.0, 0.044 - Q.z_7)
        - 1.34 * max(0.0, 0.054 - Q.z_7)
        + 23.49 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21)
        - 1.464 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow)
        - 992.5 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98)
        + 0.00329 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1)
        + 3.05 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32)
        + 161.9 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow)
        - 1704.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02)
        + 75.1 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037)
        + 0.8574 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2)
        - 21.79 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset)
        + 21.78 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3)
        + 0.2568 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr)
        + 0.4573 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0)
        - 0.1275 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21)
        + 73.05 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2)
        + 0.3463 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32)
        + 0.7791 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016)
        + 5.298e-05 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass)
        + 0.0001054 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3)
        - 0.03157 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr)
        - 3411.0 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2)
        - 117.9 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2)
        - 1.357 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2)
        - 1.246 * max(0.0, Q.LHA - 0.1)
        + 1086.0 * max(0.0, 3.7e-05 - Q.girth2_top2)
        + 22.12 * max(0.0, 0.0057 - Q.lam1)
        + 2.336 * max(0.0, Q.log_sum_pt - 6.3)
        + 6.59 * max(0.0, Q.log_sum_pt - 6.8)
        + 12.67 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.9052 * max(0.0, 6.5 - Q.log_sum_pt)
        + 0.003337 * max(0.0, 8.5 - Q.mass)
        + 0.005095 * max(0.0, 36.0 - Q.mass)
        + 0.01805 * max(0.0, 69.0 - Q.mass)
        - 5.161 * max(0.0, 0.0098 - Q.mass_over_sum_pt)
        + 33.69 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq)
        + 0.1079 * max(0.0, 0.16 - Q.max_dr)
        - 0.03823 * max(0.0, 0.68 - Q.planar_flow)
        + 0.01379 * max(0.0, Q.pt_7 - 31.0)
        - 0.0194 * max(0.0, 54.0 - Q.pt_7)
        - 0.001548 * max(0.0, 790.0 - Q.sum_pt)
        - 12.85 * max(0.0, 0.017 - Q.z_7)
        + 12000.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset)
        - 202.1 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078)
        + 1.95 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79)
        + 1.778 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0)
        + 0.01049 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01)
        - 3.08 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2)
        - 4.081e-06 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 0.009839 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7)
        + 0.3471 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063)
        - 0.3275 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2)
        - 0.6726 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013)
        + 0.01384 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092)
        - 0.003466 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10)
        + 0.00301 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7)
        + 0.01356 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2)
        - 25.1 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7)
        + 3.843 * max(0.0, Q.C2 - 0.094)
        + 2.099 * max(0.0, Q.LHA - 0.32)
        - 24.25 * max(0.0, Q.centroid_offset - 0.013)
        + 5.872 * max(0.0, Q.e2 - 0.028)
        - 9.389 * max(0.0, Q.e2 - 0.051)
        + 1.314 * max(0.0, 0.043 - Q.e2)
        - 28.48 * max(0.0, 0.0088 - Q.girth2)
        + 27.6 * max(0.0, Q.lam1 - 0.016)
        - 0.01463 * max(0.0, Q.mass - 70.0)
        - 0.003928 * max(0.0, Q.mass_top5 - 53.0)
        - 0.04392 * max(0.0, Q.max_dr - 0.15)
        + 27.71 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96)
        + 5.983 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr)
        + 0.2763 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012)
        + 0.2517 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5)
        + 0.09909 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0)
        + 5.89 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi)
        + 4.979 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03)
        + 0.1109 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 561.8 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08)
        + 471.4 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96)
        + 0.4276 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6)
        + 0.1626 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7)
        - 471.2 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7)
        - 0.002074 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6)
        - 6.28 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2)
        + 0.07047 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr)
        - 0.1226 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098)
        - 0.000788 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7)
        - 97.85 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7)
        + 498.3 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7)
        + 674.3 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr)
        + 0.2874 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05)
        - 3.081 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32)
        - 0.004311 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7)
        - 46.6 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1)
        - 6.153 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3)
        + 4.185 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0)
        - 0.07856 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4)
        + 35.52 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4)
        + 20.89 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion)
        - 10.13 * max(0.0, Q.C2 - 0.013)
        + 78.77 * max(0.0, Q.C2 - 0.067)
        + 4.918 * max(0.0, 0.014 - Q.centroid_offset)
        + 19.15 * max(0.0, Q.e2 - 0.02)
        + 170.3 * max(0.0, Q.e2 - 0.064)
        - 217.7 * max(0.0, 0.003 - Q.e2_sq)
        + 81.2 * max(0.0, 0.017 - Q.e2_sq)
        + 1.266 * max(0.0, 0.13 - Q.girth)
        + 238.3 * max(0.0, Q.girth2 - 0.0034)
        - 50.36 * max(0.0, Q.girth2 - 0.0081)
        - 1365.0 * max(0.0, 0.00033 - Q.lam2)
        + 0.01988 * max(0.0, 44.0 - Q.mass)
        + 6.403 * max(0.0, Q.mass_over_sum_pt - 0.089)
        + 5.573 * max(0.0, Q.mass_over_sum_pt - 0.11)
        + 2.611 * max(0.0, Q.max_dr - 0.094)
        - 3.458 * max(0.0, Q.max_dr - 0.2)
        + 8.591e-05 * max(0.0, 760.0 - Q.sum_pt)
        + 0.002533 * max(0.0, 430.0 - Q.sum_pt_top5)
        + 4.844 * max(0.0, 0.24 - Q.tau21)
        + 0.7093 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0)
        + 0.04846 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7)
        - 30.54 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1)
        + 3005.0 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016)
        - 0.4768 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044)
        + 2.607 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98)
        - 0.2053 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0)
        - 430.1 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011)
        - 2703.0 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2)
        - 0.05944 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass)
        - 28.64 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi)
        + 0.04908 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0)
        - 0.04889 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7)
        + 0.01068 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2)
        + 313.2 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2)
        + 50.2 * max(0.0, 0.023 - Q.dr_0)
        + 2.993 * max(0.0, 0.034 - Q.e2)
        - 16640.0 * max(0.0, 5.2e-05 - Q.girth2)
        - 0.01311 * max(0.0, 53.0 - Q.pt_7)
        + 0.0003698 * max(0.0, Q.sum_pt - 870.0)
        + 0.001334 * max(0.0, 540.0 - Q.sum_pt_top2)
        - 36.72 * max(0.0, 0.024 - Q.z_7)
        + 432.3 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034)
        + 31.03 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028)
        + 19.73 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.2029 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5)
        + 3.871 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05)
        - 97.2 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow)
        - 32.14 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063)
        + 248.5 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017)
        - 98.04 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0)
        + 412.1 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0)
        - 9.656 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039)
        - 3727.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2)
        + 1.685 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4)
        - 0.02656 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0)
        - 0.00228 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 1.47 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0)
        - 0.2498 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass)
        - 0.3472 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset)
        - 0.1058 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0)
        + 0.2139 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3)
        + 0.0008184 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2)
        - 11360.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset)
        + 593.6 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset)
        - 4.348 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 0.04439 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt)
        - 12.95 * max(0.0, Q.C2 - 0.033)
        + 14.14 * max(0.0, 0.034 - Q.C2)
        + 10.72 * max(0.0, Q.centroid_offset - 0.019)
        - 4.331 * max(0.0, Q.centroid_offset - 0.05)
        + 7.845 * max(0.0, 0.051 - Q.e2)
        + 5.022 * max(0.0, Q.girth - 0.083)
        - 20.51 * max(0.0, 0.0036 - Q.girth2)
        + 28.21 * max(0.0, 0.0086 - Q.girth2)
        + 5.276 * max(0.0, 0.01 - Q.girth2_top2)
        + 5.906 * max(0.0, Q.girth2_top5 - 0.011)
        + 47.82 * max(0.0, 0.0078 - Q.lam1)
        + 60.56 * max(0.0, Q.lam2 - 0.0034)
        + 0.4178 * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.001615 * max(0.0, 50.0 - Q.mass)
        + 17.31 * max(0.0, Q.mass_over_sum_pt - 0.0081)
        + 1.291 * max(0.0, Q.max_dr - 0.12)
        - 0.4793 * max(0.0, 0.11 - Q.max_dr)
        - 1.235 * max(0.0, 0.06 - Q.planar_flow)
        - 0.001386 * max(0.0, 25.0 - Q.pt_6)
        + 0.01104 * max(0.0, Q.sum_pt - 990.0)
        - 0.003583 * max(0.0, Q.sum_pt_top5 - 840.0)
        - 29.94 * max(0.0, 0.013 - Q.width)
        - 0.1084 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0)
        + 0.002224 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass)
        + 0.0003741 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4)
        + 4.27 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2)
        - 319.4 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2)
        - 0.119 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3)
        - 174.6 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2)
        + 1.831 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5)
        + 0.146 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0)
        + 48.9 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16)
        + 95.99 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13)
        + 154.6 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi)
        - 72.12 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014)
        - 0.4596 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0)
        - 176.8 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
        - 2.026 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098)
        - 0.005121 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0)
        - 0.01008 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6)
        - 0.005847 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7)
        + 1.577 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4)
        - 7.776 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
        + 15.94 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
        + 0.001656 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15)
        - 0.003989 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1)
        + 0.006573 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7)
        + 0.3584 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32)
        + 1.849e-05 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0)
        - 301.8 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        - 509.3 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026)
        + 0.02855 * max(0.0, 0.021 - Q.centroid_offset)
        - 59.85 * max(0.0, 0.0012 - Q.e2_sq)
        - 12.98 * max(0.0, 0.088 - Q.girth)
        - 26.32 * max(0.0, Q.girth2 - 0.0015)
        - 207.8 * max(0.0, Q.girth2 - 0.0044)
        + 207.7 * max(0.0, Q.girth2 - 0.0075)
        - 155.9 * max(0.0, Q.girth2 - 0.0087)
        + 73.81 * max(0.0, Q.girth2 - 0.015)
        + 144.5 * max(0.0, 0.0011 - Q.girth2_top2)
        + 120.0 * max(0.0, 0.0084 - Q.lam1)
        + 0.01549 * max(0.0, Q.mass - 80.4)
        - 3.091 * max(0.0, Q.mass_over_sum_pt - 0.072)
        - 18.3 * max(0.0, Q.mass_over_sum_pt - 0.085)
        + 47.9 * max(0.0, Q.mass_over_sum_pt - 0.091)
        + 248.8 * max(0.0, 0.00052 - Q.width)
        + 86.19 * max(0.0, 0.0055 - Q.width)
        + 0.2691 * max(0.0, Q.z_dr_0p05_0p1 - 0.66)
        - 302.7 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025)
        + 651.9 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068)
        + 0.05455 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0)
        + 0.1225 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0)
        + 0.005595 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0)
        + 19.89 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 4.144 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 476.6 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055)
        - 53.53 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21)
        - 458.3 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95)
        - 105.7 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2)
        + 63.12 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3)
        + 108.2 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4)
        + 20.07 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0)
        - 18.73 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88)
        - 2.765 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2)
        + 7.67 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50)
        + 0.1576 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92)
        - 0.00111 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012)
        - 3.866 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27)
        - 0.7953 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2)
        - 0.5943 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1)
        + 2.614 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056)
        - 0.02692 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2)
        - 0.01939 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2)
        + 0.001996 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0)
        + 0.005896 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow)
        - 818.3 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi)
        - 1.461 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2)
        - 4.503 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4)
        - 2.369 * max(0.0, 0.027 - Q.C2)
        + 0.5708 * max(0.0, 0.0034 - Q.centroid_offset)
        - 33.23 * max(0.0, 0.016 - Q.dr_0)
        - 10.77 * max(0.0, Q.log_sum_pt - 6.7)
        + 204.1 * max(0.0, 0.0049 - Q.width)
        + 5030.0 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075)
        - 122.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075)
        + 5086.0 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width)
        - 9.444 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7)
        - 2154.0 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width)
        + 7733.0 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset)
        + 90.67 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow)
        + 2115.0 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029)
        - 0.01185 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7)
        + 0.9277 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031)
        - 0.3699 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset)
        + 0.1963 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0)
        + 10770.0 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2)
        - 9129.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071)
        - 12320.0 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045)
        - 2893.0 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2)
        + 2.707 * max(0.0, Q.C2 - 0.051)
        + 1.749 * max(0.0, 0.018 - Q.centroid_offset)
        - 8.165 * max(0.0, 0.055 - Q.girth)
        + 145.5 * max(0.0, Q.girth2 - 0.018)
        + 95.6 * max(0.0, Q.lam2 - 0.0017)
        - 29.55 * max(0.0, 0.076 - Q.mass_over_sum_pt)
        + 64.67 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq)
        - 0.616 * max(0.0, 0.23 - Q.max_dr)
        - 3.464 * max(0.0, 0.00042 - Q.mean_phi)
        - 0.02175 * max(0.0, Q.n_dr_0p2_0p4 - 0.9)
        + 0.0004622 * max(0.0, Q.sum_pt - 850.0)
        - 51.61 * max(0.0, 0.0061 - Q.width)
        - 2.325 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5)
        - 0.09096 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2)
        - 0.1 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0)
        - 6.381 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043)
        - 26.98 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033)
        + 11.89 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset)
        - 29.77 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01)
        - 0.1833 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7)
        + 28.8 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21)
        + 4449.0 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011)
        - 85.48 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta)
        - 5.509 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow)
        + 19.47 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0)
        - 17210.0 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023)
        + 0.2594 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0)
        - 0.9745 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026)
        - 0.5325 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset)
        + 0.06007 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset)
        + 0.07248 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
        - 0.01825 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        - 0.01802 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow)
        + 1551.0 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2)
        - 0.004893 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22)
        + 1693.0 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031)
        + 2120.0 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029)
        + 9.844 * max(0.0, Q.C2 - 0.055)
        - 2.629 * max(0.0, Q.LHA - 0.33)
        + 18.61 * max(0.0, Q.centroid_offset - 0.038)
        + 23.45 * max(0.0, Q.e2 - 0.036)
        - 16.92 * max(0.0, 0.037 - Q.e2)
        - 144.6 * max(0.0, Q.girth2 - 0.0085)
        - 360.2 * max(0.0, 0.0017 - Q.girth2)
        + 127.7 * max(0.0, 0.0038 - Q.girth2_top2)
        + 34.07 * max(0.0, 0.0016 - Q.girth2_top3)
        - 265.7 * max(0.0, Q.lam1 - 0.0085)
        - 148.7 * max(0.0, 0.0044 - Q.lam1)
        - 348.9 * max(0.0, 0.0034 - Q.lam2)
        - 2.545 * max(0.0, 6.3 - Q.log_sum_pt)
        + 0.01347 * max(0.0, Q.mass - 17.0)
        + 19.75 * max(0.0, 0.1 - Q.mass_over_sum_pt)
        + 0.07863 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 0.3061 * max(0.0, Q.n_dr_0p2_0p4 - 1.9)
        + 0.004192 * max(0.0, Q.pt_7 - 46.0)
        + 4.217 * max(0.0, 0.27 - Q.tau32)
        - 8.104 * max(0.0, Q.z_7 - 0.062)
        - 13.69 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21)
        - 0.04154 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2)
        + 44.85 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4)
        + 690.9 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8)
        - 6.201 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass)
        - 17.13 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50)
        + 364.5 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21)
        - 0.001827 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 1.517 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 5.373 * max(0.0, 0.035 - Q.C2)
        + 34.99 * max(0.0, Q.centroid_offset - 0.014)
        - 4.267 * max(0.0, 0.04 - Q.centroid_offset)
        - 26.24 * max(0.0, 0.0062 - Q.e2_sq)
        + 12.0 * max(0.0, Q.girth - 0.076)
        - 1.519 * max(0.0, 0.021 - Q.girth)
        + 21.45 * max(0.0, 0.089 - Q.girth)
        + 41.48 * max(0.0, 0.002 - Q.girth2_top3)
        - 0.01894 * max(0.0, 5.5 - Q.mass_top5)
        + 1.119 * max(0.0, 0.22 - Q.max_dr)
        - 0.02567 * max(0.0, 2.8 - Q.n_dr_0p1_0p2)
        - 0.2473 * max(0.0, 0.27 - Q.planar_flow)
        - 0.0006043 * max(0.0, 690.0 - Q.sum_pt_top5)
        - 20.51 * max(0.0, 0.0036 - Q.width)
        + 198.7 * max(0.0, 0.0085 - Q.width)
        - 2.14 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7)
        + 0.8287 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt)
        - 19.51 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21)
        + 189.9 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014)
        + 11.5 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi)
        + 1.268 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 1.039 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50)
        - 0.09608 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7)
        - 0.4122 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0)
        + 646.0 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi)
        + 83.2 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032)
        - 0.7039 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0)
        - 0.009437 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 112.8 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015)
        + 0.0198 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass)
        + 0.5702 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11)
        + 140.1 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056)
        + 0.002989 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4)
        - 146.3 * max(0.0, Q.e2 - 0.063)
        - 200.7 * max(0.0, Q.girth2 - 0.019)
        - 0.03722 * max(0.0, Q.mass - 91.2)
        - 6217.0 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06)
        - 2.942 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0)
        - 59.27 * max(0.0, Q.C2 - 0.066)
        - 18.57 * max(0.0, 0.038 - Q.centroid_offset)
        + 18.64 * max(0.0, 0.049 - Q.e2)
        - 22.85 * max(0.0, 0.15 - Q.girth)
        - 174.3 * max(0.0, 0.0062 - Q.lam1)
        - 149.1 * max(0.0, 0.015 - Q.lam1)
        + 0.05596 * max(0.0, 25.0 - Q.pt_7)
        - 0.01805 * max(0.0, Q.sum_pt - 1000.0)
        + 0.4883 * max(0.0, 0.5 - Q.tau21)
        - 113.5 * max(0.0, 0.0076 - Q.width)
        + 47.47 * max(0.0, 0.027 - Q.z_7)
        - 40.59 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4)
        + 16.87 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt)
        + 0.35 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7)
        + 924.2 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset)
        - 168.3 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18)
        + 42250.0 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset)
        + 0.002657 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2)
        - 0.0122 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1.883 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4)
        - 0.002475 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2)
        - 0.001442 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88)
        + 0.01212 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1)
        - 0.0005105 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5)
        - 0.0001378 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7)
        + 0.009583 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32)
        + 0.1927 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023)
        + 4.88 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013)
        + 5.267 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0)
        - 14.01 * max(0.0, 0.067 - Q.C2)
        + 0.1212 * max(0.0, 0.8 - Q.D2)
        - 8.561 * max(0.0, Q.centroid_offset - 0.031)
        + 194.8 * max(0.0, 0.0058 - Q.e2_sq)
        - 7.707 * max(0.0, 0.034 - Q.girth)
        - 8.307 * max(0.0, 0.087 - Q.girth)
        + 19.48 * max(0.0, Q.lam1 - 0.0025)
        + 115.9 * max(0.0, Q.lam1 - 0.0042)
        + 54.05 * max(0.0, Q.lam1 - 0.0061)
        + 134.7 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq)
        - 0.2581 * max(0.0, 0.078 - Q.max_dr)
        - 0.2831 * max(0.0, 0.18 - Q.max_dr)
        - 0.001284 * max(0.0, 4.8 - Q.n_dr_0p05_0p1)
        - 0.0001434 * max(0.0, 310.0 - Q.sum_pt_top3)
        + 0.2692 * max(0.0, 0.14 - Q.tau21)
        - 0.1652 * max(0.0, 0.6 - Q.z_dr_0p05_0p1)
        - 1.598 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset)
        - 11.44 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2)
        - 874.3 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
        + 0.6371 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0)
        + 0.3998 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 39.92 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4)
        + 27.33 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4)
        + 11.45 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6)
        - 5.225 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41)
        - 1636.0 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr)
        + 54.72 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094)
        - 52.57 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
        + 12.78 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr)
        + 0.006694 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt)
        + 25.45 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2)
        + 1.131 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0)
        - 1510.0 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta)
        - 206.9 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi)
        + 7.724 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 446.8 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow)
        + 1.685 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2)
        - 0.007007 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.4974 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 8.149 * max(0.0, Q.LHA - 0.34)
        - 17.04 * max(0.0, 0.024 - Q.e2)
        + 3.447 * max(0.0, 0.041 - Q.e2)
        - 2.792 * max(0.0, Q.girth - 0.032)
        - 8.067 * max(0.0, 0.0074 - Q.girth2_top2)
        + 27.46 * max(0.0, 0.0067 - Q.lam1)
        + 5.392 * max(0.0, 0.0083 - Q.lam1)
        + 398.2 * max(0.0, 0.00031 - Q.lam2)
        + 0.6117 * max(0.0, 0.068 - Q.mass_over_sum_pt)
        - 90.01 * max(0.0, 0.0067 - Q.width)
        - 0.7836 * max(0.0, Q.z_dr_0p05_0p1 - 0.75)
        + 0.09525 * max(0.0, 0.32 - Q.z_dr_0p1_0p2)
        + 0.3259 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion)
        - 0.001179 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0)
        + 67.56 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2)
        + 0.9536 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0)
        - 0.2137 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3)
        + 725.3 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017)
        + 0.02365 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        - 0.08686 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion)
        + 54.62 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084)
        + 0.3505 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1)
        + 0.8477 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4)
        + 8528.0 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024)
        - 324.5 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 718.1 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.width > 0.009139599744230509:
        if s['g'] - s['t'] > -0.09985221549868584:
            if s['g'] - s['t'] > 0.30911290645599365:
                if s['g'] - s['q'] > 0.1422872394323349:
                    if s['g'] - s['Z'] > 0.15964581072330475:
                        return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 70% of the training jets here get this class from the formula
                else:
                    return 'q'   # 79% of the training jets here get this class from the formula
            else:
                if s['Z'] - s['t'] > 0.0897565484046936:
                    return 'Z'   # 88% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['q'] > 0.05554055981338024:
                        if s['g'] - s['t'] > 0.054003624245524406:
                            if s['Z'] - s['t'] > -5.295156478881836:
                                return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.0632219314575195:
                                    return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.06760667264461517:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.084766387939453:
                                return 'g'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.22333773970603943:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 72% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.15523432940244675:
                if s['Z'] - s['t'] > 0.14021676778793335:
                    return 'Z'   # 91% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 1.9611663818359375:
                        if Q.eccentricity > 0.9771870970726013:
                            return 't'   # 83% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 67% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 75% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.1271747499704361:
                    if s['q'] - s['t'] > 0.13240236788988113:
                        return 'q'   # 83% of the training jets here get this class from the formula
                    else:
                        return 't'   # 51% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.4283677935600281:
                        if s['g'] - s['t'] > -0.23633960634469986:
                            if Q.C2 > 0.07314332574605942:
                                if Q.girth2_top5 > 0.012512816581875086:
                                    return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 68% of the training jets here get this class from the formula
                            else:
                                return 't'   # 71% of the training jets here get this class from the formula
                        else:
                            return 't'   # 83% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.47518375515937805:
                            if s['W'] - s['t'] > -3.7363752126693726:
                                return 'Z'   # 53% of the training jets here get this class from the formula
                            else:
                                return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            return 't'   # 99% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.2987767457962036:
            if s['g'] - s['q'] > -0.03203224204480648:
                if s['g'] - s['t'] > 0.03114129975438118:
                    if s['g'] - s['W'] > -0.01348957372829318:
                        if s['g'] - s['q'] > 0.07270979881286621:
                            if s['g'] - s['t'] > 0.2595144808292389:
                                if s['g'] - s['W'] > 0.3276536166667938:
                                    if s['g'] - s['q'] > 0.12715010344982147:
                                        if Q.D2 > 7.976235389709473:
                                            return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.e2_sq > 0.0031671300530433655:
                                        if Q.mass_top5 > 26.506850242614746:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 5.612061977386475:
                                    return 't'   # 65% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 0.318935364484787:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 57% of the training jets here get this class from the formula
                        else:
                            if Q.dr_0 > 0.012809824664145708:
                                if s['g'] - s['q'] > 0.016638873144984245:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 724.625:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top5 > 3.4615219831466675:
                                            if s['Z'] - s['t'] > -0.7073579728603363:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 769.6875:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.2873292565345764:
                            if Q.width > 0.003263103892095387:
                                return 'W'   # 82% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.8024930357933044:
                                    return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 58% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 87% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.2418128177523613:
                        if s['W'] - s['t'] > 0.16002576053142548:
                            return 'W'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top2 > 0.0015347293228842318:
                                return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                return 't'   # 61% of the training jets here get this class from the formula
                    else:
                        return 't'   # 86% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.009744667448103428:
                    if s['q'] - s['t'] > -0.03749691694974899:
                        if s['g'] - s['q'] > -0.13798554986715317:
                            if Q.sum_pt_top3 > 641.59375:
                                return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['q'] > -0.07790838554501534:
                                    if Q.girth2_top5 > 0.0001721729859127663:
                                        if s['g'] - s['t'] > 4.153945446014404:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top5 > 2.6050511598587036:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 739.2890625:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > 3.8172407150268555:
                                            if s['q'] - s['t'] > 4.702470779418945:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 2.6824753284454346:
                                        if Q.log_sum_pt > 6.706250905990601:
                                            if Q.D2 > 1.1973251104354858:
                                                if Q.pt_dispersion > 0.3872888833284378:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 6.97138237953186:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['t'] > 1.494419276714325:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 37.578125:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.02372520975768566:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 1.5767889618873596:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 42.5:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.00027820498507935554:
                                                    if Q.log_sum_pt > 6.496480226516724:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.21414796262979507:
                                if s['g'] - s['q'] > -0.21958693861961365:
                                    if Q.sum_pt_top5 > 906.875:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 620.375:
                                            if Q.max_pair_mass > 4.892366409301758:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 0.2284354642033577:
                                        if Q.log_sum_pt > 7.2924816608428955:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 3.5628721714019775:
                                                if Q.centroid_offset > 0.0029806349193677306:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_4 > 37.953125:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 41% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.9718663692474365:
                                                    if s['g'] - s['q'] > -0.5547259747982025:
                                                        if s['g'] - s['t'] > 3.8668347597122192:
                                                            if Q.pt_4 > 47.96875:
                                                                return 'q'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.dr_1 > 0.039501724764704704:
                                    return 'W'   # 53% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 77% of the training jets here get this class from the formula
                    else:
                        return 't'   # 88% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.39346130192279816:
                        if s['q'] - s['Z'] > 0.5815601050853729:
                            if s['g'] - s['t'] > 1.0885253548622131:
                                return 'q'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 769.8359375:
                                    if s['q'] - s['Z'] > 1.1846720576286316:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 59% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 68% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 70% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['t'] > 0.05395499989390373:
                            return 'W'   # 94% of the training jets here get this class from the formula
                        else:
                            return 't'   # 56% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.019718951545655727:
                if s['g'] - s['W'] > -0.0007544458494521677:
                    if s['g'] - s['W'] > 0.19057101756334305:
                        if s['g'] - s['t'] > -0.030402486212551594:
                            if Q.mass > 30.1876277923584:
                                if s['g'] - s['Z'] > 1.2659425139427185:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.013936161994934082:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            return 't'   # 90% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.05775378458201885:
                            if Q.centroid_offset > 0.01216986682265997:
                                if s['g'] - s['Z'] > 0.6508273780345917:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 1.2243384718894958:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.4798574894666672:
                                            if s['g'] - s['W'] > 0.09307660162448883:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.03724782355129719:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01752102468162775:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 60% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 66% of the training jets here get this class from the formula
                        else:
                            return 't'   # 85% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.19702690094709396:
                        if s['g'] - s['W'] > -0.3802005648612976:
                            if s['g'] - s['Z'] > 0.42310968041419983:
                                if Q.mass_over_sum_pt > 0.055875299498438835:
                                    if s['W'] - s['t'] > 0.04399518296122551:
                                        return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.0510367676615715:
                                        if s['g'] - s['Z'] > 0.7379532158374786:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.21479064971208572:
                                    if s['g'] - s['q'] > 1.2045871019363403:
                                        if Q.mass > 24.685362815856934:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.1027478277683258:
                                if s['W'] - s['Z'] > 0.40931449830532074:
                                    if s['g'] - s['W'] > -0.6744151413440704:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > 0.4325924217700958:
                                            return 'W'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.2972334623336792:
                                                return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['W'] > -0.09104997292160988:
                                        return 'W'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.026094372384250164:
                                            if Q.sum_pt_top5 > 998.34375:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.2151520997285843:
                                    if Q.z_dr_0p1_0p2 > 0.3111940622329712:
                                        return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 93% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['Z'] > -1.018450915813446:
                            if s['Z'] - s['t'] > -0.14584175497293472:
                                if s['q'] - s['Z'] > 0.09854009374976158:
                                    return 'q'   # 49% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 84% of the training jets here get this class from the formula
                            else:
                                return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 22.096442222595215:
                                if Q.max_dr > 0.14994792640209198:
                                    if Q.centroid_offset > 0.01461395202204585:
                                        if Q.tau21 > 0.15948130935430527:
                                            return 'W'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 635.21875:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > -0.2526489123702049:
                                        if s['W'] - s['Z'] > 0.10567271709442139:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_2 > 0.09599228575825691:
                                                if Q.girth > 0.07407630980014801:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.17610833793878555:
                                    if s['q'] - s['Z'] > -1.1789096593856812:
                                        return 'W'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 1.0051358342170715:
                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 83% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > -0.08001813292503357:
                    if s['g'] - s['t'] > -0.008643952198326588:
                        if s['g'] - s['Z'] > 0.17482511699199677:
                            return 'g'   # 92% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > 0.032075537368655205:
                                if s['q'] - s['Z'] > -0.006417351774871349:
                                    return 'q'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 651.34375:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.05105233192443848:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > -0.32632970809936523:
                                    if Q.mass_over_sum_pt > 0.020727972500026226:
                                        return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 642.234375:
                                        if s['g'] - s['q'] > 0.07363441959023476:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 66% of the training jets here get this class from the formula
                    else:
                        return 't'   # 90% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.11297651380300522:
                        if s['W'] - s['Z'] > -0.26804350316524506:
                            if Q.n_dr_0p05_0p1 > 1.5:
                                if s['W'] - s['Z'] > -0.14874422550201416:
                                    if Q.z_dr_0p1_0p2 > 0.11379973590373993:
                                        if s['g'] - s['W'] > -3.2149572372436523:
                                            if Q.D2 > 0.798509418964386:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.006864835973829031:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top3 > 39.90224075317383:
                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.16647430509328842:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.0925205647945404:
                                        if Q.sum_pt_top3 > 267.171875:
                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 52% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.776349663734436:
                                    if Q.girth > 0.02597777731716633:
                                        if s['q'] - s['Z'] > -0.13695622235536575:
                                            return 'q'   # 43% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.014174175914376974:
                                        if s['W'] - s['Z'] > -0.11796439439058304:
                                            if s['q'] - s['W'] > -1.0175016522407532:
                                                if Q.LHA > 0.17631487548351288:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.029781173914670944:
                                            if Q.e2 > 0.03470836766064167:
                                                if Q.D2 > 1.002588003873825:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > -0.08368588238954544:
                                                    if Q.LHA > 0.2573733478784561:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 75% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > -0.30986393988132477:
                                if s['q'] - s['Z'] > -0.19848386198282242:
                                    return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 56% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.4351944029331207:
                                    if s['W'] - s['Z'] > -0.550099104642868:
                                        if Q.z_dr_0p05_0p1 > 0.08407343178987503:
                                            if s['g'] - s['Z'] > -2.7352843284606934:
                                                if Q.z_dr_0p1_0p2 > 0.059943610802292824:
                                                    if Q.sum_pt_top5 > 357.8125:
                                                        if Q.lam1 > 0.006895378232002258:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.006684644380584359:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > -0.42066361010074615:
                                            if s['q'] - s['W'] > 2.8654919862747192:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 3.6284422874450684:
                                                if s['q'] - s['Z'] > -0.8983859419822693:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > -0.7872259020805359:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 3.4715179204940796:
                                        if Q.lam2 > 8.836913184495643e-05:
                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 77% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.22234336286783218:
                            if s['Z'] - s['t'] > -0.04062463715672493:
                                if s['q'] - s['W'] > 3.384598731994629:
                                    return 't'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 52% of the training jets here get this class from the formula
                            else:
                                return 't'   # 69% of the training jets here get this class from the formula
                        else:
                            return 't'   # 94% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
