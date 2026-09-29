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

Test set (50,000 jets): accuracy 80.24% (the formula: 80.45%); same class as the formula for 95.82% of jets.  48 leaves, depth 11.
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
    return (0.2841
        + 7.136 * max(0.0, 0.057 - Q.girth)
        + 74.45 * max(0.0, 0.0062 - Q.girth2_top20)
        + 122.3 * max(0.0, 0.0058 - Q.lam1)
        + 0.155 * max(0.0, Q.mass - 80.4)
        + 0.06006 * max(0.0, Q.mass - 91.2)
        - 151.4 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.02621 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 13.87 * max(0.0, Q.log_sum_pt - 6.8)
        + 29.89 * max(0.0, Q.log_sum_pt - 6.9)
        - 8.949 * max(0.0, Q.log_sum_pt - 7.0)
        + 0.01722 * max(0.0, 73.0 - Q.mass_top30)
        - 0.04297 * max(0.0, 43.0 - Q.n_particles)
        + 0.002712 * max(0.0, 760.0 - Q.sum_pt_top3)
        - 3.893 * max(0.0, 0.96 - Q.z_top20_slots)
        - 43.6 * max(0.0, Q.z_top50_slots - 0.96)
        - 351.2 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 0.0001104 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 0.2434 * max(0.0, 7.7 - Q.D2)
        - 0.04054 * max(0.0, 80.4 - Q.mass)
        - 0.02129 * max(0.0, 110.0 - Q.mass)
        - 0.01422 * max(0.0, 1100.0 - Q.sum_pt)
        - 5.259 * max(0.0, 0.61 - Q.tau32)
        - 0.2473 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.003198 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 73.03 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        - 0.007075 * max(0.0, Q.mass_top30 - 49.0)
        + 0.03348 * max(0.0, Q.mass_top50 - 150.0)
        - 1.214 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        - 0.019 * max(0.0, 91.2 - Q.mass)
        - 0.01957 * max(0.0, 100.0 - Q.mass)
        - 12.67 * max(0.0, 0.026 - Q.e2)
        + 0.1345 * max(0.0, 120.0 - Q.mass)
        - 0.2156 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        + 19.01 * max(0.0, 0.99 - Q.z_top50_slots)
        - 0.1466 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        - 0.02393 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        + 0.3225 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.18 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        + 8.749 * max(0.0, 0.096 - Q.girth)
        - 0.08493 * max(0.0, Q.mass - 71.0)
        - 0.1364 * max(0.0, Q.mass - 120.0)
        - 0.02573 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.01067 * max(0.0, 1000.0 - Q.sum_pt)
        + 0.01909 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.003782 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        + 335.8 * max(0.0, 0.0056 - Q.girth2_top40)
        - 151.8 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        - 0.0003905 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 0.1255 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 0.007216 * max(0.0, Q.mass - 150.0)
        + 0.03439 * max(0.0, Q.mass_top50 - 140.0)
        + 0.1276 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.002621 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 1.569 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        - 1.086 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 10.57 * max(0.0, Q.C2 - 0.055)
        - 0.00998 * max(0.0, 63.0 - Q.mass_top50)
        - 3.283 * max(0.0, 6.8 - Q.log_sum_pt)
        + 9.758 * max(0.0, 7.0 - Q.log_sum_pt)
        - 0.01569 * max(0.0, Q.mass - 140.0)
        - 0.02898 * max(0.0, Q.mass - 172.8)
        - 0.008234 * max(0.0, 1000.0 - Q.sum_pt_top40)
        + 0.00322 * max(0.0, 130.0 - Q.mass)
    )


def score_q(Q):
    return (0.7858
        - 6.977 * max(0.0, 0.057 - Q.girth)
        + 49.11 * max(0.0, 0.0062 - Q.girth2_top20)
        + 139.1 * max(0.0, 0.0058 - Q.lam1)
        + 0.143 * max(0.0, Q.mass - 80.4)
        + 0.05048 * max(0.0, Q.mass - 91.2)
        - 316.2 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.01956 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 3.778 * max(0.0, Q.log_sum_pt - 6.8)
        + 4.684 * max(0.0, Q.log_sum_pt - 6.9)
        - 1.617 * max(0.0, Q.log_sum_pt - 7.0)
        - 0.006909 * max(0.0, 73.0 - Q.mass_top30)
        + 0.01446 * max(0.0, 43.0 - Q.n_particles)
        - 0.0009583 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 1.127 * max(0.0, 0.96 - Q.z_top20_slots)
        - 6.699 * max(0.0, Q.z_top50_slots - 0.96)
        - 58.61 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 4.851e-05 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 0.3038 * max(0.0, 7.7 - Q.D2)
        + 0.01446 * max(0.0, 80.4 - Q.mass)
        - 0.03056 * max(0.0, 110.0 - Q.mass)
        + 0.0003473 * max(0.0, 1100.0 - Q.sum_pt)
        - 2.703 * max(0.0, 0.61 - Q.tau32)
        - 0.283 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.005319 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 112.4 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        + 0.005004 * max(0.0, Q.mass_top30 - 49.0)
        + 0.04092 * max(0.0, Q.mass_top50 - 150.0)
        + 0.8963 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        + 0.009476 * max(0.0, 91.2 - Q.mass)
        - 0.03092 * max(0.0, 100.0 - Q.mass)
        - 63.37 * max(0.0, 0.026 - Q.e2)
        + 0.09139 * max(0.0, 120.0 - Q.mass)
        - 0.002766 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        - 3.175 * max(0.0, 0.99 - Q.z_top50_slots)
        - 0.196 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        - 0.02825 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        + 0.3699 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.226 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 0.465 * max(0.0, 0.096 - Q.girth)
        - 0.1107 * max(0.0, Q.mass - 71.0)
        - 0.09257 * max(0.0, Q.mass - 120.0)
        - 0.0003692 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.00552 * max(0.0, 1000.0 - Q.sum_pt)
        + 0.02872 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.00911 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        + 324.7 * max(0.0, 0.0056 - Q.girth2_top40)
        - 192.8 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        - 0.001247 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 0.06054 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 0.02624 * max(0.0, Q.mass - 150.0)
        + 0.03366 * max(0.0, Q.mass_top50 - 140.0)
        + 0.0181 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.001848 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 0.3898 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        - 1.122 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 13.03 * max(0.0, Q.C2 - 0.055)
        - 0.03566 * max(0.0, 63.0 - Q.mass_top50)
        - 5.885 * max(0.0, 6.8 - Q.log_sum_pt)
        + 3.781 * max(0.0, 7.0 - Q.log_sum_pt)
        - 0.02018 * max(0.0, Q.mass - 140.0)
        - 0.003649 * max(0.0, Q.mass - 172.8)
        - 0.008249 * max(0.0, 1000.0 - Q.sum_pt_top40)
        + 0.003144 * max(0.0, 130.0 - Q.mass)
    )


def score_W(Q):
    return (-0.5523
        - 25.59 * max(0.0, 0.057 - Q.girth)
        - 206.6 * max(0.0, 0.0062 - Q.girth2_top20)
        - 500.8 * max(0.0, 0.0058 - Q.lam1)
        - 0.2569 * max(0.0, Q.mass - 80.4)
        + 0.226 * max(0.0, Q.mass - 91.2)
        + 3613.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.1711 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 4.143 * max(0.0, Q.log_sum_pt - 6.8)
        + 22.63 * max(0.0, Q.log_sum_pt - 6.9)
        - 17.43 * max(0.0, Q.log_sum_pt - 7.0)
        - 0.005006 * max(0.0, 73.0 - Q.mass_top30)
        + 0.04221 * max(0.0, 43.0 - Q.n_particles)
        - 5.978e-05 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 1.975 * max(0.0, 0.96 - Q.z_top20_slots)
        - 6.068 * max(0.0, Q.z_top50_slots - 0.96)
        - 3290.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 0.0001123 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 0.03654 * max(0.0, 7.7 - Q.D2)
        - 0.1923 * max(0.0, 80.4 - Q.mass)
        + 0.004319 * max(0.0, 110.0 - Q.mass)
        + 0.04177 * max(0.0, 1100.0 - Q.sum_pt)
        + 2.082 * max(0.0, 0.61 - Q.tau32)
        + 0.1122 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.0001427 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        - 30.34 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        + 0.02264 * max(0.0, Q.mass_top30 - 49.0)
        - 0.05666 * max(0.0, Q.mass_top50 - 150.0)
        + 1.097 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        + 0.2787 * max(0.0, 91.2 - Q.mass)
        - 0.06547 * max(0.0, 100.0 - Q.mass)
        - 0.2374 * max(0.0, 0.026 - Q.e2)
        + 0.01704 * max(0.0, 120.0 - Q.mass)
        + 0.02601 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        - 28.27 * max(0.0, 0.99 - Q.z_top50_slots)
        + 0.1949 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.2392 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        - 0.3449 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.1752 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 4.819 * max(0.0, 0.096 - Q.girth)
        - 0.01234 * max(0.0, Q.mass - 71.0)
        + 0.02696 * max(0.0, Q.mass - 120.0)
        + 0.08091 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        - 0.001201 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.06595 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.009503 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        - 61.3 * max(0.0, 0.0056 - Q.girth2_top40)
        + 117.9 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        + 0.008361 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 2.006 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        + 0.0001232 * max(0.0, Q.mass - 150.0)
        - 0.03555 * max(0.0, Q.mass_top50 - 140.0)
        - 0.03904 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.004203 * max(0.0, 960.0 - Q.sum_pt_top50)
        + 0.8332 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        - 0.6369 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 4.231 * max(0.0, Q.C2 - 0.055)
        + 0.007696 * max(0.0, 63.0 - Q.mass_top50)
        + 2.189 * max(0.0, 6.8 - Q.log_sum_pt)
        - 27.0 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.0248 * max(0.0, Q.mass - 140.0)
        + 0.03039 * max(0.0, Q.mass - 172.8)
        - 0.01062 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 0.03335 * max(0.0, 130.0 - Q.mass)
    )


def score_Z(Q):
    return (-0.3909
        + 63.74 * max(0.0, 0.057 - Q.girth)
        + 207.4 * max(0.0, 0.0062 - Q.girth2_top20)
        + 111.1 * max(0.0, 0.0058 - Q.lam1)
        + 0.1581 * max(0.0, Q.mass - 80.4)
        - 0.2707 * max(0.0, Q.mass - 91.2)
        - 4050.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.1374 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 2.417 * max(0.0, Q.log_sum_pt - 6.8)
        + 10.54 * max(0.0, Q.log_sum_pt - 6.9)
        - 10.53 * max(0.0, Q.log_sum_pt - 7.0)
        + 0.01608 * max(0.0, 73.0 - Q.mass_top30)
        + 0.0399 * max(0.0, 43.0 - Q.n_particles)
        - 0.000189 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 2.486 * max(0.0, 0.96 - Q.z_top20_slots)
        + 19.14 * max(0.0, Q.z_top50_slots - 0.96)
        + 3185.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        - 0.0001311 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        - 0.08231 * max(0.0, 7.7 - Q.D2)
        + 0.1923 * max(0.0, 80.4 - Q.mass)
        - 0.01129 * max(0.0, 110.0 - Q.mass)
        + 0.04317 * max(0.0, 1100.0 - Q.sum_pt)
        + 1.124 * max(0.0, 0.61 - Q.tau32)
        + 0.3561 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.001435 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        - 398.3 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        + 0.01937 * max(0.0, Q.mass_top30 - 49.0)
        - 0.01128 * max(0.0, Q.mass_top50 - 150.0)
        + 1.291 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        - 0.3286 * max(0.0, 91.2 - Q.mass)
        + 0.3081 * max(0.0, 100.0 - Q.mass)
        - 32.17 * max(0.0, 0.026 - Q.e2)
        - 0.1409 * max(0.0, 120.0 - Q.mass)
        + 0.02291 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        - 19.94 * max(0.0, 0.99 - Q.z_top50_slots)
        + 0.5762 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.1333 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        - 0.6066 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.7135 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        - 10.11 * max(0.0, 0.096 - Q.girth)
        - 0.07634 * max(0.0, Q.mass - 71.0)
        + 0.1761 * max(0.0, Q.mass - 120.0)
        + 0.08822 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.01857 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.06486 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.0292 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        - 7.287 * max(0.0, 0.0056 - Q.girth2_top40)
        + 425.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        + 0.01461 * max(0.0, 920.0 - Q.sum_pt_top40)
        + 1.773 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 0.08143 * max(0.0, Q.mass - 150.0)
        - 0.1363 * max(0.0, Q.mass_top50 - 140.0)
        + 0.002701 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.01154 * max(0.0, 960.0 - Q.sum_pt_top50)
        + 0.4667 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        + 1.382 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        - 9.959 * max(0.0, Q.C2 - 0.055)
        - 0.04154 * max(0.0, 63.0 - Q.mass_top50)
        - 11.22 * max(0.0, 6.8 - Q.log_sum_pt)
        - 35.12 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.1481 * max(0.0, Q.mass - 140.0)
        + 0.06017 * max(0.0, Q.mass - 172.8)
        - 0.01145 * max(0.0, 1000.0 - Q.sum_pt_top40)
        + 0.01733 * max(0.0, 130.0 - Q.mass)
    )


def score_t(Q):
    return (-0.7544
        - 4.865 * max(0.0, 0.057 - Q.girth)
        - 48.59 * max(0.0, 0.0062 - Q.girth2_top20)
        - 115.4 * max(0.0, 0.0058 - Q.lam1)
        - 0.0267 * max(0.0, Q.mass - 80.4)
        + 0.02708 * max(0.0, Q.mass - 91.2)
        + 776.1 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
        - 0.01157 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 4.267 * max(0.0, Q.log_sum_pt - 6.8)
        - 3.828 * max(0.0, Q.log_sum_pt - 6.9)
        + 7.606 * max(0.0, Q.log_sum_pt - 7.0)
        - 0.01448 * max(0.0, 73.0 - Q.mass_top30)
        + 0.008846 * max(0.0, 43.0 - Q.n_particles)
        + 0.001539 * max(0.0, 760.0 - Q.sum_pt_top3)
        + 0.6658 * max(0.0, 0.96 - Q.z_top20_slots)
        - 10.77 * max(0.0, Q.z_top50_slots - 0.96)
        - 489.2 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
        + 2.872e-06 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        + 0.3562 * max(0.0, 7.7 - Q.D2)
        + 0.04505 * max(0.0, 80.4 - Q.mass)
        + 0.008304 * max(0.0, 110.0 - Q.mass)
        + 0.00365 * max(0.0, 1100.0 - Q.sum_pt)
        - 0.2081 * max(0.0, 0.61 - Q.tau32)
        - 0.1285 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        - 0.003143 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        + 29.86 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
        - 0.01145 * max(0.0, Q.mass_top30 - 49.0)
        - 0.01966 * max(0.0, Q.mass_top50 - 150.0)
        - 0.06815 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        + 0.01325 * max(0.0, 91.2 - Q.mass)
        - 0.04009 * max(0.0, 100.0 - Q.mass)
        + 36.46 * max(0.0, 0.026 - Q.e2)
        - 0.02876 * max(0.0, 120.0 - Q.mass)
        - 0.07114 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
        + 6.217 * max(0.0, 0.99 - Q.z_top50_slots)
        + 0.01191 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
        + 0.01766 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
        + 0.2002 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
        + 0.03028 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        + 4.976 * max(0.0, 0.096 - Q.girth)
        + 0.007237 * max(0.0, Q.mass - 71.0)
        + 0.0002828 * max(0.0, Q.mass - 120.0)
        - 0.01443 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
        + 0.03376 * max(0.0, 1000.0 - Q.sum_pt)
        - 0.05632 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
        + 0.02466 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        + 46.85 * max(0.0, 0.0056 - Q.girth2_top40)
        - 98.63 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
        + 0.000495 * max(0.0, 920.0 - Q.sum_pt_top40)
        - 0.406 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        - 0.101 * max(0.0, Q.mass - 150.0)
        + 0.1086 * max(0.0, Q.mass_top50 - 140.0)
        - 0.09503 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.01016 * max(0.0, 960.0 - Q.sum_pt_top50)
        - 8.229 * max(0.0, Q.z_dr_0_0p05 - 0.77)
        + 15.74 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
        + 10.48 * max(0.0, Q.C2 - 0.055)
        + 0.01852 * max(0.0, 63.0 - Q.mass_top50)
        - 18.8 * max(0.0, 6.8 - Q.log_sum_pt)
        + 3.459 * max(0.0, 7.0 - Q.log_sum_pt)
        + 0.04084 * max(0.0, Q.mass - 140.0)
        - 0.07344 * max(0.0, Q.mass - 172.8)
        - 0.01856 * max(0.0, 1000.0 - Q.sum_pt_top40)
        - 0.007945 * max(0.0, 130.0 - Q.mass)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['q'] - s['W'] > -1.5274686813354492:
        if s['q'] - s['Z'] > -0.9933307468891144:
            if s['g'] - s['t'] > -0.38557903468608856:
                if s['g'] - s['q'] > -0.046488167718052864:
                    if s['g'] - s['t'] > 0.28840479254722595:
                        if s['g'] - s['W'] > 0.09397090971469879:
                            if s['g'] - s['q'] > 0.31094391644001007:
                                if s['g'] - s['Z'] > 0.4498296529054642:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 3.17938768863678:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.807975560426712:
                                    if s['g'] - s['Z'] > 0.3642248958349228:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 0.20387019217014313:
                                        if Q.n_particles > 47.5:
                                            if Q.tau32 > 0.7308608293533325:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 82% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > -0.5068084597587585:
                                if Q.tau32 > 0.7755908370018005:
                                    return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 64% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 87% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.06787879765033722:
                            if s['Z'] - s['t'] > 0.017689605243504047:
                                return 'Z'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 838.2470703125:
                                    if s['g'] - s['W'] > -0.5308825969696045:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0_0p05 > 0.16749460250139236:
                                return 'g'   # 51% of the training jets here get this class from the formula
                            else:
                                return 't'   # 65% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.02285820059478283:
                        if s['q'] - s['Z'] > 0.31651023030281067:
                            if s['g'] - s['q'] > -0.3339899629354477:
                                if Q.tau32 > 0.8536834716796875:
                                    if Q.n_particles > 42.5:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 98% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 84% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > -0.5135298669338226:
                            if s['q'] - s['Z'] > -0.07740594074130058:
                                if Q.C2 > 0.06030937843024731:
                                    return 'W'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.0697394497692585:
                                        return 'W'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.5723769664764404:
                                return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 52% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.2771860957145691:
                    if s['q'] - s['W'] > -0.24047749489545822:
                        if s['q'] - s['Z'] > 0.10031677410006523:
                            return 'q'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 79% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.12007712572813034:
                            return 'W'   # 86% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 73% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.2980353385210037:
                        return 'W'   # 76% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.7208874523639679:
                            if s['Z'] - s['t'] > -0.17412148416042328:
                                return 'Z'   # 64% of the training jets here get this class from the formula
                            else:
                                return 't'   # 76% of the training jets here get this class from the formula
                        else:
                            return 't'   # 98% of the training jets here get this class from the formula
        else:
            if s['g'] - s['Z'] > -0.15131960064172745:
                return 'g'   # 90% of the training jets here get this class from the formula
            else:
                if s['Z'] - s['t'] > 0.2827135771512985:
                    if s['g'] - s['W'] > 3.848861813545227:
                        return 'Z'   # 53% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > -0.07377605885267258:
                            return 'W'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 99% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -3.4343351125717163:
                        if s['Z'] - s['t'] > -0.38466696441173553:
                            return 'Z'   # 57% of the training jets here get this class from the formula
                        else:
                            return 't'   # 86% of the training jets here get this class from the formula
                    else:
                        return 't'   # 94% of the training jets here get this class from the formula
    else:
        if s['W'] - s['Z'] > 0.12177802994847298:
            if s['g'] - s['W'] > -0.28376777470111847:
                return 'g'   # 76% of the training jets here get this class from the formula
            else:
                if s['W'] - s['t'] > -0.26000499725341797:
                    return 'W'   # 100% of the training jets here get this class from the formula
                else:
                    return 't'   # 63% of the training jets here get this class from the formula
        else:
            return 'Z'   # 98% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
