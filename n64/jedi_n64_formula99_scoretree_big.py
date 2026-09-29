"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities and on differences of additive class scores.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
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

Test set (50,000 jets): accuracy 80.47% (the formula: 80.45%); same class as the formula for 97.76% of jets.  221 leaves, depth 17.
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
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def score_g(Q):
    return (-0.5284
        - 0.8767 * max(0.0, 0.057 - Q.girth)
        + 58.8 * max(0.0, 0.0062 - Q.girth2_top20)
        + 92.14 * max(0.0, 0.0058 - Q.lam1)
        + 0.1231 * max(0.0, Q.mass - 80.4)
        + 0.04221 * max(0.0, Q.mass - 91.2)
        - 214.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.01807 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 228.9 * max(0.0, 0.0021 - Q.girth2_top20)
        - 13.32 * max(0.0, Q.log_sum_pt - 6.8)
        + 34.81 * max(0.0, Q.log_sum_pt - 6.9)
        - 14.97 * max(0.0, Q.log_sum_pt - 7.0)
        + 0.0117 * max(0.0, 73.0 - Q.mass_top30)
        - 0.04332 * max(0.0, 43.0 - Q.n_particles)
        + 0.00345 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 1.323 * max(0.0, Q.tau32 - 0.28)
        - 3.622 * max(0.0, 0.96 - Q.z_top20_slots)
        - 38.8 * max(0.0, Q.z_top50_slots - 0.96)
        + 0.4527 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
        + 0.8606 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
        - 0.0001904 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        + 0.0408 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        + 6526.0 * max(0.0, 0.00011 - Q.girth2_top5)
        - 1360.0 * max(0.0, 0.00052 - Q.lam2)
        - 154.5 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 9.308e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 9.978 * max(0.0, Q.C2 - 0.1)
        - 0.2199 * max(0.0, 7.7 - Q.D2)
        - 0.008627 * max(0.0, 80.4 - Q.mass)
        - 0.02418 * max(0.0, 110.0 - Q.mass)
        - 0.008018 * max(0.0, 1100.0 - Q.sum_pt)
        - 2.511 * max(0.0, 0.61 - Q.tau32)
        - 0.03188 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
        - 0.2274 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.004547 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 110.3 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        - 0.004748 * max(0.0, Q.mass_top30 - 49.0)
        + 0.01343 * max(0.0, Q.mass_top50 - 150.0)
        - 1.024 * max(0.0, Q.max_dr - 0.44)
        + 2.665 * max(0.0, Q.z_top30_slots - 0.93)
        - 11.41 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
        - 0.007465 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
        - 0.1053 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        - 0.008418 * max(0.0, 91.2 - Q.mass)
        - 0.01399 * max(0.0, 100.0 - Q.mass)
        - 26.91 * max(0.0, 0.026 - Q.e2)
        + 0.09064 * max(0.0, 120.0 - Q.mass)
        - 0.02 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        + 1.549 * max(0.0, 0.99 - Q.z_top50_slots)
        - 0.06757 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.04715 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        + 0.2649 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.08899 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 1.107 * max(0.0, 0.096 - Q.girth)
        + 56.14 * max(0.0, 0.0015 - Q.lam2)
        - 0.08214 * max(0.0, Q.mass - 71.0)
        - 0.08868 * max(0.0, Q.mass - 120.0)
        - 0.003622 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.005177 * max(0.0, Q.n_particles - 58.0)
        + 0.008563 * max(0.0, 1000.0 - Q.sum_pt)
        + 0.01868 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 8.974e-05 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
        + 0.002011 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        + 259.2 * max(0.0, 0.0056 - Q.girth2_top40)
        - 160.4 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        - 0.001728 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 0.1703 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        + 7.541 * max(0.0, 0.021 - Q.girth2_top5)
        + 0.01694 * max(0.0, Q.mass - 150.0)
        + 0.02018 * max(0.0, Q.mass_top50 - 140.0)
        + 0.0208 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.00309 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 0.03158 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        - 0.634 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 0.02565 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
        + 0.0006945 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        - 22.35 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
        + 0.002615 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
        - 0.005265 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        - 3.972 * max(0.0, Q.C2 - 0.055)
        + 0.004798 * max(0.0, Q.mass_top30 - 130.0)
        - 0.02647 * max(0.0, 63.0 - Q.mass_top50)
        + 3461.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        - 2.354 * max(0.0, 6.8 - Q.log_sum_pt)
        + 7.95 * max(0.0, 7.0 - Q.log_sum_pt)
        - 0.01487 * max(0.0, Q.mass - 140.0)
        - 0.0222 * max(0.0, Q.mass - 172.8)
        + 0.003406 * max(0.0, Q.mass_top10 - 67.0)
        - 0.00875 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 0.001504 * max(0.0, 130.0 - Q.mass)
        + 6.089 * max(0.0, 0.96 - Q.z_top50_slots)
    )


def score_q(Q):
    return (-0.2975
        - 3.645 * max(0.0, 0.057 - Q.girth)
        + 67.82 * max(0.0, 0.0062 - Q.girth2_top20)
        + 56.23 * max(0.0, 0.0058 - Q.lam1)
        + 0.1525 * max(0.0, Q.mass - 80.4)
        + 0.05902 * max(0.0, Q.mass - 91.2)
        - 56.65 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.01725 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 133.3 * max(0.0, 0.0021 - Q.girth2_top20)
        - 3.484 * max(0.0, Q.log_sum_pt - 6.8)
        + 8.119 * max(0.0, Q.log_sum_pt - 6.9)
        - 5.5 * max(0.0, Q.log_sum_pt - 7.0)
        - 0.001641 * max(0.0, 73.0 - Q.mass_top30)
        + 0.01393 * max(0.0, 43.0 - Q.n_particles)
        - 0.0007977 * max(0.0, 760.0 - Q.sum_pt_top3)
        - 0.2631 * max(0.0, Q.tau32 - 0.28)
        + 1.649 * max(0.0, 0.96 - Q.z_top20_slots)
        - 0.6276 * max(0.0, Q.z_top50_slots - 0.96)
        - 0.1243 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
        - 0.4551 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
        + 5.042e-05 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        + 0.08224 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        + 1700.0 * max(0.0, 0.00011 - Q.girth2_top5)
        + 44.26 * max(0.0, 0.00052 - Q.lam2)
        - 295.1 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 2.272e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 11.04 * max(0.0, Q.C2 - 0.1)
        - 0.22 * max(0.0, 7.7 - Q.D2)
        + 0.001956 * max(0.0, 80.4 - Q.mass)
        - 0.0323 * max(0.0, 110.0 - Q.mass)
        + 0.003583 * max(0.0, 1100.0 - Q.sum_pt)
        - 2.806 * max(0.0, 0.61 - Q.tau32)
        - 0.03133 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
        - 0.2461 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.004314 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 226.5 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        - 0.002914 * max(0.0, Q.mass_top30 - 49.0)
        + 0.01123 * max(0.0, Q.mass_top50 - 150.0)
        + 0.06454 * max(0.0, Q.max_dr - 0.44)
        + 0.5736 * max(0.0, Q.z_top30_slots - 0.93)
        + 26.96 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
        + 0.01517 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
        + 0.4924 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        + 0.002905 * max(0.0, 91.2 - Q.mass)
        - 0.02677 * max(0.0, 100.0 - Q.mass)
        - 34.19 * max(0.0, 0.026 - Q.e2)
        + 0.1012 * max(0.0, 120.0 - Q.mass)
        - 0.00473 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        - 1.005 * max(0.0, 0.99 - Q.z_top50_slots)
        - 0.165 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.07044 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        + 0.263 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.186 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 1.636 * max(0.0, 0.096 - Q.girth)
        + 33.23 * max(0.0, 0.0015 - Q.lam2)
        - 0.1147 * max(0.0, Q.mass - 71.0)
        - 0.1023 * max(0.0, Q.mass - 120.0)
        - 0.004449 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.004039 * max(0.0, Q.n_particles - 58.0)
        + 0.005027 * max(0.0, 1000.0 - Q.sum_pt)
        + 0.022 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 9.82e-05 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
        + 0.00654 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        + 340.4 * max(0.0, 0.0056 - Q.girth2_top40)
        - 283.5 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        - 0.001305 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 0.1231 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        + 6.889 * max(0.0, 0.021 - Q.girth2_top5)
        + 0.01243 * max(0.0, Q.mass - 150.0)
        + 0.003232 * max(0.0, Q.mass_top50 - 140.0)
        + 0.01535 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.001631 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 0.1998 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        - 0.5319 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 0.01691 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
        + 0.007483 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        - 52.72 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
        + 0.005313 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
        - 0.01751 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        - 4.2 * max(0.0, Q.C2 - 0.055)
        + 0.01323 * max(0.0, Q.mass_top30 - 130.0)
        - 0.02607 * max(0.0, 63.0 - Q.mass_top50)
        + 6224.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        - 5.362 * max(0.0, 6.8 - Q.log_sum_pt)
        + 2.741 * max(0.0, 7.0 - Q.log_sum_pt)
        - 0.01383 * max(0.0, Q.mass - 140.0)
        + 0.000652 * max(0.0, Q.mass - 172.8)
        + 0.002427 * max(0.0, Q.mass_top10 - 67.0)
        - 0.007931 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 9.799e-05 * max(0.0, 130.0 - Q.mass)
        + 9.167 * max(0.0, 0.96 - Q.z_top50_slots)
    )


def score_W(Q):
    return (0.9564
        - 31.08 * max(0.0, 0.057 - Q.girth)
        - 189.0 * max(0.0, 0.0062 - Q.girth2_top20)
        - 160.3 * max(0.0, 0.0058 - Q.lam1)
        - 0.2714 * max(0.0, Q.mass - 80.4)
        + 0.2205 * max(0.0, Q.mass - 91.2)
        + 2388.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.1774 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 57.75 * max(0.0, 0.0021 - Q.girth2_top20)
        + 0.02984 * max(0.0, Q.log_sum_pt - 6.8)
        + 13.29 * max(0.0, Q.log_sum_pt - 6.9)
        - 13.48 * max(0.0, Q.log_sum_pt - 7.0)
        - 0.001449 * max(0.0, 73.0 - Q.mass_top30)
        + 0.02145 * max(0.0, 43.0 - Q.n_particles)
        - 0.001088 * max(0.0, 760.0 - Q.sum_pt_top3)
        - 0.09797 * max(0.0, Q.tau32 - 0.28)
        + 0.6392 * max(0.0, 0.96 - Q.z_top20_slots)
        - 11.75 * max(0.0, Q.z_top50_slots - 0.96)
        - 0.04257 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
        - 0.4965 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
        + 5.051e-05 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        + 0.09478 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        - 3539.0 * max(0.0, 0.00011 - Q.girth2_top5)
        + 392.1 * max(0.0, 0.00052 - Q.lam2)
        - 2186.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 3.953e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 3.692 * max(0.0, Q.C2 - 0.1)
        + 0.07311 * max(0.0, 7.7 - Q.D2)
        - 0.1621 * max(0.0, 80.4 - Q.mass)
        + 0.01177 * max(0.0, 110.0 - Q.mass)
        + 0.0445 * max(0.0, 1100.0 - Q.sum_pt)
        + 0.7008 * max(0.0, 0.61 - Q.tau32)
        - 0.09839 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
        + 0.1075 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.001373 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        - 106.6 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        + 0.01196 * max(0.0, Q.mass_top30 - 49.0)
        - 0.01531 * max(0.0, Q.mass_top50 - 150.0)
        + 1.186 * max(0.0, Q.max_dr - 0.44)
        - 3.655 * max(0.0, Q.z_top30_slots - 0.93)
        + 49.44 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
        + 0.1909 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
        + 0.7861 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        + 0.2785 * max(0.0, 91.2 - Q.mass)
        - 0.06161 * max(0.0, 100.0 - Q.mass)
        + 13.67 * max(0.0, 0.026 - Q.e2)
        - 0.02632 * max(0.0, 120.0 - Q.mass)
        - 0.1134 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        + 1.052 * max(0.0, 0.99 - Q.z_top50_slots)
        + 0.0005726 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.4293 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        - 0.457 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.03709 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 7.843 * max(0.0, 0.096 - Q.girth)
        + 407.2 * max(0.0, 0.0015 - Q.lam2)
        - 0.01266 * max(0.0, Q.mass - 71.0)
        + 0.05871 * max(0.0, Q.mass - 120.0)
        + 0.06362 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        - 0.08469 * max(0.0, Q.n_particles - 58.0)
        + 0.009894 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.09014 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.0004942 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
        + 0.01861 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        - 197.0 * max(0.0, 0.0056 - Q.girth2_top40)
        + 223.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        + 0.01377 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 1.452 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 9.842 * max(0.0, 0.021 - Q.girth2_top5)
        - 0.07025 * max(0.0, Q.mass - 150.0)
        + 0.01647 * max(0.0, Q.mass_top50 - 140.0)
        + 0.03431 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.003502 * max(0.0, 960.0 - Q.sum_pt_top50)
        + 0.6149 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        + 2.558 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        + 0.04781 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
        - 0.01022 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        + 78.63 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
        - 0.0131 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
        + 0.04709 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        + 6.476 * max(0.0, Q.C2 - 0.055)
        - 0.006636 * max(0.0, Q.mass_top30 - 130.0)
        + 0.01963 * max(0.0, 63.0 - Q.mass_top50)
        - 8311.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        - 7.517 * max(0.0, 6.8 - Q.log_sum_pt)
        - 36.15 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.01645 * max(0.0, Q.mass - 140.0)
        + 0.02085 * max(0.0, Q.mass - 172.8)
        - 0.007164 * max(0.0, Q.mass_top10 - 67.0)
        - 0.01042 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 0.03177 * max(0.0, 130.0 - Q.mass)
        - 80.01 * max(0.0, 0.96 - Q.z_top50_slots)
    )


def score_Z(Q):
    return (-0.3449
        + 65.01 * max(0.0, 0.057 - Q.girth)
        + 200.7 * max(0.0, 0.0062 - Q.girth2_top20)
        + 327.8 * max(0.0, 0.0058 - Q.lam1)
        + 0.1552 * max(0.0, Q.mass - 80.4)
        - 0.2406 * max(0.0, Q.mass - 91.2)
        - 4398.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.1356 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 53.86 * max(0.0, 0.0021 - Q.girth2_top20)
        + 7.816 * max(0.0, Q.log_sum_pt - 6.8)
        - 1.622 * max(0.0, Q.log_sum_pt - 6.9)
        - 4.48 * max(0.0, Q.log_sum_pt - 7.0)
        + 0.01577 * max(0.0, 73.0 - Q.mass_top30)
        + 0.02391 * max(0.0, 43.0 - Q.n_particles)
        - 0.0003329 * max(0.0, 760.0 - Q.sum_pt_top3)
        - 0.4055 * max(0.0, Q.tau32 - 0.28)
        + 0.9454 * max(0.0, 0.96 - Q.z_top20_slots)
        - 26.35 * max(0.0, Q.z_top50_slots - 0.96)
        - 0.06371 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
        - 0.2803 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
        + 4.609e-05 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        - 0.2729 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        - 4424.0 * max(0.0, 0.00011 - Q.girth2_top5)
        + 521.5 * max(0.0, 0.00052 - Q.lam2)
        + 3479.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 6.872e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 1.356 * max(0.0, Q.C2 - 0.1)
        - 0.009388 * max(0.0, 7.7 - Q.D2)
        + 0.1718 * max(0.0, 80.4 - Q.mass)
        - 0.0003338 * max(0.0, 110.0 - Q.mass)
        + 0.04306 * max(0.0, 1100.0 - Q.sum_pt)
        + 0.01882 * max(0.0, 0.61 - Q.tau32)
        + 0.3168 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
        + 0.2151 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.000875 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        - 357.4 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        + 0.01808 * max(0.0, Q.mass_top30 - 49.0)
        + 0.01722 * max(0.0, Q.mass_top50 - 150.0)
        + 1.452 * max(0.0, Q.max_dr - 0.44)
        - 5.053 * max(0.0, Q.z_top30_slots - 0.93)
        + 32.02 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
        + 0.2151 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
        + 0.8296 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        - 0.3047 * max(0.0, 91.2 - Q.mass)
        + 0.2527 * max(0.0, 100.0 - Q.mass)
        - 6.351 * max(0.0, 0.026 - Q.e2)
        - 0.09778 * max(0.0, 120.0 - Q.mass)
        - 0.008022 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        - 38.13 * max(0.0, 0.99 - Q.z_top50_slots)
        + 0.5279 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        - 0.501 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        - 0.0009801 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.6575 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 8.454 * max(0.0, 0.096 - Q.girth)
        + 478.2 * max(0.0, 0.0015 - Q.lam2)
        - 0.05777 * max(0.0, Q.mass - 71.0)
        + 0.1399 * max(0.0, Q.mass - 120.0)
        + 0.07062 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        - 0.1122 * max(0.0, Q.n_particles - 58.0)
        + 0.03251 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.08022 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.0003704 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
        + 0.03419 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        - 129.9 * max(0.0, 0.0056 - Q.girth2_top40)
        + 391.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        + 0.01245 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 1.178 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 3.354 * max(0.0, 0.021 - Q.girth2_top5)
        - 0.1087 * max(0.0, Q.mass - 150.0)
        - 0.1407 * max(0.0, Q.mass_top50 - 140.0)
        + 0.0142 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.006554 * max(0.0, 960.0 - Q.sum_pt_top50)
        + 0.04472 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        + 3.69 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        + 0.004393 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
        - 0.004388 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        - 25.59 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
        + 0.00466 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
        - 0.005611 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        - 3.139 * max(0.0, Q.C2 - 0.055)
        - 0.00213 * max(0.0, Q.mass_top30 - 130.0)
        - 0.04222 * max(0.0, 63.0 - Q.mass_top50)
        + 1570.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        - 16.7 * max(0.0, 6.8 - Q.log_sum_pt)
        - 42.77 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.1512 * max(0.0, Q.mass - 140.0)
        + 0.06009 * max(0.0, Q.mass - 172.8)
        + 0.001594 * max(0.0, Q.mass_top10 - 67.0)
        - 0.01228 * max(0.0, 1000.0 - Q.sum_pt_top40)
        + 0.02527 * max(0.0, 130.0 - Q.mass)
        + 20.36 * max(0.0, 0.96 - Q.z_top50_slots)
    )


def score_t(Q):
    return (-0.5273
        + 11.27 * max(0.0, 0.057 - Q.girth)
        - 25.72 * max(0.0, 0.0062 - Q.girth2_top20)
        - 55.4 * max(0.0, 0.0058 - Q.lam1)
        + 0.007158 * max(0.0, Q.mass - 80.4)
        + 0.03485 * max(0.0, Q.mass - 91.2)
        + 723.5 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.006924 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 190.3 * max(0.0, 0.0021 - Q.girth2_top20)
        - 5.517 * max(0.0, Q.log_sum_pt - 6.8)
        + 3.441 * max(0.0, Q.log_sum_pt - 6.9)
        + 1.996 * max(0.0, Q.log_sum_pt - 7.0)
        + 0.003847 * max(0.0, 73.0 - Q.mass_top30)
        + 0.007468 * max(0.0, 43.0 - Q.n_particles)
        + 0.0003401 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 0.08652 * max(0.0, Q.tau32 - 0.28)
        + 0.2982 * max(0.0, 0.96 - Q.z_top20_slots)
        - 5.337 * max(0.0, Q.z_top50_slots - 0.96)
        - 0.004456 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
        - 0.2963 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
        + 2.227e-05 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        + 0.07635 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        - 2187.0 * max(0.0, 0.00011 - Q.girth2_top5)
        - 35.76 * max(0.0, 0.00052 - Q.lam2)
        - 414.4 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 1.388e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        + 1.049 * max(0.0, Q.C2 - 0.1)
        + 0.03689 * max(0.0, 7.7 - Q.D2)
        - 0.01067 * max(0.0, 80.4 - Q.mass)
        + 0.003692 * max(0.0, 110.0 - Q.mass)
        + 0.006913 * max(0.0, 1100.0 - Q.sum_pt)
        + 0.663 * max(0.0, 0.61 - Q.tau32)
        - 0.2152 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
        - 0.05943 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.001785 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 25.31 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        - 0.003633 * max(0.0, Q.mass_top30 - 49.0)
        + 0.005197 * max(0.0, Q.mass_top50 - 150.0)
        - 0.7754 * max(0.0, Q.max_dr - 0.44)
        + 3.397 * max(0.0, Q.z_top30_slots - 0.93)
        - 37.45 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
        - 0.09759 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
        - 0.4677 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        - 0.01432 * max(0.0, 91.2 - Q.mass)
        - 0.02055 * max(0.0, 100.0 - Q.mass)
        - 12.32 * max(0.0, 0.026 - Q.e2)
        + 0.01895 * max(0.0, 120.0 - Q.mass)
        - 0.05 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        + 5.061 * max(0.0, 0.99 - Q.z_top50_slots)
        - 0.06076 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.2817 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        - 0.0564 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.08433 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        + 5.363 * max(0.0, 0.096 - Q.girth)
        - 99.84 * max(0.0, 0.0015 - Q.lam2)
        + 0.01893 * max(0.0, Q.mass - 71.0)
        - 0.05924 * max(0.0, Q.mass - 120.0)
        - 0.0153 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.0189 * max(0.0, Q.n_particles - 58.0)
        + 0.02935 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.06646 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        - 3.313e-05 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
        + 0.01952 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        - 5.734 * max(0.0, 0.0056 - Q.girth2_top40)
        - 40.88 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        - 0.0005647 * max(0.0, 920.0 - Q.sum_pt_top40)
        - 0.5521 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 86.8 * max(0.0, 0.021 - Q.girth2_top5)
        - 0.1262 * max(0.0, Q.mass - 150.0)
        + 0.129 * max(0.0, Q.mass_top50 - 140.0)
        - 0.1118 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.01029 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 9.78 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        + 14.15 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        + 0.08068 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
        + 0.07542 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        + 4.696 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
        + 0.006328 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
        + 0.004209 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        + 1.947 * max(0.0, Q.C2 - 0.055)
        - 0.01649 * max(0.0, Q.mass_top30 - 130.0)
        + 0.03825 * max(0.0, 63.0 - Q.mass_top50)
        - 6389.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        - 15.99 * max(0.0, 6.8 - Q.log_sum_pt)
        + 4.18 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.03811 * max(0.0, Q.mass - 140.0)
        - 0.07248 * max(0.0, Q.mass - 172.8)
        - 0.01174 * max(0.0, Q.mass_top10 - 67.0)
        - 0.01731 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 0.005373 * max(0.0, 130.0 - Q.mass)
        + 10.05 * max(0.0, 0.96 - Q.z_top50_slots)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['q'] - s['W'] > -1.4963955879211426:
        if s['q'] - s['Z'] > -0.8536352217197418:
            if s['g'] - s['t'] > -0.28608983755111694:
                if s['g'] - s['q'] > -0.00107462058076635:
                    if s['g'] - s['W'] > -0.128572516143322:
                        if s['g'] - s['t'] > 0.08292904868721962:
                            if s['g'] - s['Z'] > 0.29883530735969543:
                                if s['g'] - s['q'] > 0.07113382965326309:
                                    if s['g'] - s['t'] > 0.5032555758953094:
                                        if s['g'] - s['W'] > 0.11539756879210472:
                                            if Q.max_dr > 0.4968833923339844:
                                                if s['g'] - s['q'] > 0.5118412375450134:
                                                    if s['g'] - s['Z'] > 2.283942699432373:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > 1.0487703680992126:
                                                            if s['g'] - s['W'] > 1.1423208713531494:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.5905841588973999:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 64.96455764770508:
                                                            return 'g'   # 35% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 809.287109375:
                                                    if s['g'] - s['Z'] > 0.5813725590705872:
                                                        if Q.z_top50_slots > 0.9278044402599335:
                                                            return 'g'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > 2.8985371589660645:
                                                                return 'g'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 1003.7939453125:
                                                                    if s['g'] - s['q'] > 1.9131689071655273:
                                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.06414096057415009:
                                                if Q.max_dr > 0.3529297262430191:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 895.65283203125:
                                            if s['q'] - s['Z'] > 1.5953343510627747:
                                                if Q.lam1 > 0.03237444534897804:
                                                    if Q.sum_pt_top40 > 898.619140625:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 912.2451171875:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 20.5:
                                                            return 't'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 0.2621142268180847:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.935274124145508:
                                                        if Q.lam1 > 0.010406507179141045:
                                                            if Q.mass > 215.8578338623047:
                                                                return 't'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 7.132404327392578:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.009740718174725771:
                                                if Q.sum_pt_top40 > 771.03125:
                                                    if s['Z'] - s['t'] > -5.841248273849487:
                                                        if Q.n_dr_0p2_0p4 > 11.5:
                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.4588704854249954:
                                        if Q.max_dr > 0.6032905280590057:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.03661065921187401:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.03481720946729183:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 998.6845703125:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top30 > 39.455020904541016:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > 3.3040443658828735:
                                    if s['g'] - s['q'] > 0.18183543533086777:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 930.3232421875:
                                if Q.sum_pt_top40 > 987.8349609375:
                                    if Q.z_dr_0p2_0p4 > 0.07214105129241943:
                                        if s['g'] - s['t'] > -0.09046154096722603:
                                            if Q.sum_pt_top50 > 1217.52734375:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 181.7289276123047:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 160.57009887695312:
                                                    if Q.mass_top30 > 161.07791137695312:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.05691978149116039:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > -0.03417224995791912:
                                                if s['W'] - s['Z'] > -0.09822221845388412:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.023123398423194885:
                                        if s['g'] - s['q'] > 2.5212098360061646:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > -0.1003105565905571:
                                            if s['g'] - s['t'] > -0.18430238217115402:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 3.6790703535079956:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 771.39453125:
                                    if Q.mass_top40 > 74.75468063354492:
                                        if Q.log_sum_pt > 6.864191055297852:
                                            if Q.mass_over_sum_pt_sq > 0.02004402596503496:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.1582920253276825:
                                                if Q.sum_pt_top50 > 861.1494140625:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > -4.700120210647583:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.48492351174354553:
                            if Q.C2 > 0.050499362871050835:
                                if s['W'] - s['Z'] > 0.4695819467306137:
                                    if Q.log_sum_pt > 6.888311862945557:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 76.00655364990234:
                                    return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 70% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 94% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 0.003918800735846162:
                        if s['q'] - s['Z'] > 0.3340637981891632:
                            if s['g'] - s['q'] > -0.07538343220949173:
                                if Q.mass_over_sum_pt_sq > 0.004148927284404635:
                                    if Q.mass > 72.10356521606445:
                                        if Q.e2 > 0.02494984772056341:
                                            return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.011463042814284563:
                                        if Q.max_dr > 0.5855439603328705:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > 0.37556028366088867:
                                    return 'q'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.08372889831662178:
                                        if Q.planar_flow > 0.41204534471035004:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 0.20257285982370377:
                                                return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 90% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 3.0338000059127808:
                                if s['q'] - s['Z'] > -0.3201216161251068:
                                    return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 70% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > 3.8192137479782104:
                                    return 'q'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 973.212158203125:
                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 67% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > -0.3451625406742096:
                            if Q.C2 > 0.06667853891849518:
                                if s['W'] - s['Z'] > 0.4226732403039932:
                                    if Q.girth2_top40 > 0.005541962571442127:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.4733220785856247:
                                            if Q.C2 > 0.08146339654922485:
                                                if Q.mass > 77.18021392822266:
                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > -0.055159829556941986:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 73.99081802368164:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 44% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt_sq > 0.00481291301548481:
                                    if s['W'] - s['Z'] > 0.6834098398685455:
                                        return 'W'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 87% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.27454225718975067:
                                if Q.mass > 68.48222351074219:
                                    if s['W'] - s['Z'] > 0.617315262556076:
                                        if s['q'] - s['W'] > -0.5979351103305817:
                                            if Q.C2 > 0.058056045323610306:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.06943634897470474:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 99% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.2608678340911865:
                                        return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.7381367683410645:
                                            return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 3.1799428462982178:
                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 75% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.29957129061222076:
                    if s['q'] - s['W'] > -0.3566359579563141:
                        if s['q'] - s['Z'] > 0.2620798647403717:
                            if s['q'] - s['t'] > -0.06449944525957108:
                                if s['q'] - s['W'] > 0.0011212142708245665:
                                    if s['q'] - s['Z'] > 1.3787073493003845:
                                        return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.845154762268066:
                                            if Q.sum_pt > 1058.382568359375:
                                                return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > 0.41714687645435333:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.43619735538959503:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 81.1951675415039:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.9682668447494507:
                                        if Q.mass > 75.91050338745117:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 77.35036087036133:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.08887634798884392:
                                    if Q.sum_pt > 954.492919921875:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > -0.18025699257850647:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.7100329697132111:
                                            if Q.mass_over_sum_pt > 0.07573346793651581:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 71% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 2.8078527450561523:
                                if Q.sum_pt > 979.8701171875:
                                    if s['q'] - s['Z'] > -0.21260308474302292:
                                        return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 938.47705078125:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 66% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.08258986845612526:
                                    if s['q'] - s['t'] > -0.07318798452615738:
                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 92% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.01290874369442463:
                            if Q.mass > 68.5615463256836:
                                if s['W'] - s['Z'] > 0.5755650997161865:
                                    return 'W'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.2872127294540405:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > -0.9190081059932709:
                                    if Q.girth > 0.03843960165977478:
                                        if s['g'] - s['Z'] > 1.7702742218971252:
                                            return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 85% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.27015355229377747:
                        if s['W'] - s['t'] > 0.001852345303632319:
                            return 'W'   # 95% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.30642005801200867:
                                return 'W'   # 62% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 59% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.25543320178985596:
                            if s['W'] - s['t'] > -2.5000449419021606:
                                return 'Z'   # 74% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -0.45852600038051605:
                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 794.697998046875:
                                if s['q'] - s['t'] > -0.4455859661102295:
                                    if Q.C2 > 0.09575581178069115:
                                        if Q.log_sum_pt > 6.873748064041138:
                                            if s['q'] - s['t'] > -0.353540301322937:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.4646214544773102:
                                            return 'q'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.47444772720336914:
                                        if Q.e2 > 0.06271159276366234:
                                            return 't'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 963.66796875:
                                                if Q.mass > 188.8658447265625:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.883631706237793:
                                                    if Q.tau21 > 0.382577583193779:
                                                        if Q.lam1 > 0.019387438893318176:
                                                            return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -2.143823981285095:
                                            if s['Z'] - s['t'] > -0.4568393975496292:
                                                if Q.mass > 85.47290802001953:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.032525982707738876:
                                                    if s['g'] - s['W'] > 4.3258466720581055:
                                                        if Q.sum_pt > 953.01171875:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > 0.036663519218564034:
                                                                if Q.sum_pt_top50 > 888.8779296875:
                                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 79.53450393676758:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.005969781195744872:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.8770895004272461:
                                    if s['Z'] - s['t'] > -6.293356657028198:
                                        if Q.girth2_top40 > 0.012567794881761074:
                                            if Q.girth2_top5 > 0.011798685416579247:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 105.94844055175781:
                                        return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > -1.2975984811782837:
                                            if Q.mass_top30 > 89.09163284301758:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 635.529296875:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 41% of the training jets here get this class from the formula
        else:
            if s['g'] - s['Z'] > -0.09809532389044762:
                if s['g'] - s['Z'] > 0.2896324098110199:
                    if Q.max_dr > 0.4619872570037842:
                        if s['g'] - s['t'] > 3.3493112325668335:
                            return 'g'   # 91% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 66% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 98% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 1.8304216861724854:
                        return 'g'   # 86% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 64% of the training jets here get this class from the formula
            else:
                if s['Z'] - s['t'] > 0.09877293184399605:
                    if s['W'] - s['Z'] > -0.08607592433691025:
                        if Q.D2 > 3.909511923789978:
                            return 'Z'   # 84% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.18476970493793488:
                                return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 84.59238815307617:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 57% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > 4.168320417404175:
                            if s['g'] - s['Z'] > -0.6429811120033264:
                                if s['q'] - s['W'] > 2.1525949239730835:
                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 90% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -3.7090580463409424:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 59% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > -3.9256511926651:
                                if s['Z'] - s['t'] > 0.3933851718902588:
                                    if Q.sum_pt_top50 > 907.490234375:
                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > -3.368917465209961:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 50% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.4564370959997177:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 85% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -2.6663286685943604:
                        if s['Z'] - s['t'] > -0.16112418472766876:
                            if s['W'] - s['Z'] > -0.19553394615650177:
                                return 'W'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 1005.1258544921875:
                                    return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 4.5:
                                return 't'   # 89% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 53% of the training jets here get this class from the formula
                    else:
                        return 't'   # 97% of the training jets here get this class from the formula
    else:
        if s['W'] - s['Z'] > 0.033476075157523155:
            if s['g'] - s['W'] > -0.07888580486178398:
                if s['g'] - s['W'] > 0.2817249596118927:
                    return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.z_top50_slots > 0.9997511506080627:
                        return 'W'   # 73% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.24389838427305222:
                            return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 75% of the training jets here get this class from the formula
            else:
                if s['W'] - s['t'] > -0.34637783467769623:
                    if s['W'] - s['Z'] > 0.2706206738948822:
                        return 'W'   # 100% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.2558833211660385:
                            if Q.mass > 84.47722625732422:
                                return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 57% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 85% of the training jets here get this class from the formula
                else:
                    if Q.z_dr_0p2_0p4 > 0.002772988402284682:
                        return 't'   # 87% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 70% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > 0.7507397830486298:
                if s['W'] - s['Z'] > -0.055002426728606224:
                    if s['Z'] - s['t'] > 1.4437920451164246:
                        return 'Z'   # 79% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 78% of the training jets here get this class from the formula
                else:
                    return 'Z'   # 100% of the training jets here get this class from the formula
            else:
                if s['W'] - s['t'] > -0.08018423616886139:
                    if s['W'] - s['Z'] > -0.2430310919880867:
                        return 'W'   # 85% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 79% of the training jets here get this class from the formula
                else:
                    return 'Z'   # 48% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
