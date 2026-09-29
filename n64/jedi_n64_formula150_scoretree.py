"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 80.88% (the formula: 81.02%); same class as the formula for 96.30% of jets.  25 leaves, depth 8.
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
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_9=pt[9],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def score_g(Q):
    return (-0.02278
        - 8.879 * max(0.0, 0.0555 - Q.girth)
        - 25.15 * max(0.0, 0.00622 - Q.girth2_top20)
        + 50.45 * max(0.0, 0.00578 - Q.lam1)
        - 0.1219 * max(0.0, Q.mass - 80.4)
        + 0.1911 * max(0.0, Q.mass - 91.2)
        - 194.2 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.04192 * max(0.0, 81.8 - Q.mass_top50)
        - 0.1494 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.0002862 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.02994 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.01729 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        + 31.82 * max(0.0, Q.log_sum_pt - 6.91)
        - 20.45 * max(0.0, Q.log_sum_pt - 6.99)
        - 147.7 * max(0.0, 0.435 - Q.max_dr)
        + 0.002729 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.002897 * max(0.0, 1070.0 - Q.sum_pt_top30)
        - 0.001166 * max(0.0, Q.sum_pt_top50 - 930.0)
        + 1.603 * max(0.0, Q.tau32 - 0.276)
        - 5.45 * max(0.0, 0.931 - Q.z_top20_slots)
        - 7.582 * max(0.0, Q.z_top50_slots - 0.96)
        - 1309.0 * max(0.0, 0.000569 - Q.lam2)
        + 92.99 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 3.834 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 6.989 * max(0.0, Q.C2 - 0.106)
        - 0.2057 * max(0.0, 6.52 - Q.D2)
        + 45.84 * max(0.0, 0.063 - Q.girth)
        + 0.2488 * max(0.0, 80.4 - Q.mass)
        - 0.03153 * max(0.0, 109.0 - Q.mass)
        + 0.03412 * max(0.0, Q.n_particles - 22.7)
        + 0.007671 * max(0.0, 1070.0 - Q.sum_pt)
        + 1.283 * max(0.0, 0.451 - Q.tau21)
        - 1.914 * max(0.0, 0.578 - Q.tau32)
        - 0.01419 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.1203 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        + 0.004353 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        - 4.514 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.06222 * max(0.0, Q.mass - 63.2)
        + 100.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.01602 * max(0.0, Q.mass_top50 - 102.0)
        + 2.118 * max(0.0, Q.max_dr - 0.436)
        - 1.756 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        + 43.62 * max(0.0, Q.lam1 - 0.00717)
        - 0.1862 * max(0.0, 91.2 - Q.mass)
        - 0.0009941 * max(0.0, 101.0 - Q.mass)
        - 1.011 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.05669 * max(0.0, 73.1 - Q.mass_top50)
        - 0.01299 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.1396 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 16.44 * max(0.0, 0.00812 - Q.girth2_top20)
        + 34.05 * max(0.0, 0.0129 - Q.girth2_top40)
        + 0.02342 * max(0.0, 66.0 - Q.mass_top20)
        + 33.25 * max(0.0, 0.998 - Q.z_top50_slots)
        + 53.12 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 160.0 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.0505 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.0235 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.3929 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.1679 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.05392 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 7.057 * max(0.0, Q.C2 - 0.0661)
        - 0.01231 * max(0.0, Q.mass - 104.0)
        + 0.01883 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.01052 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.1142 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.0247 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 9.46e-05 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.002981 * max(0.0, Q.mass - 61.8)
        - 0.04768 * max(0.0, Q.mass - 142.0)
        + 0.01248 * max(0.0, 128.0 - Q.mass_top50)
        - 126.4 * max(0.0, Q.width - 0.026)
        + 17.9 * max(0.0, 0.0611 - Q.e2)
        - 2.833 * max(0.0, Q.lam1 - 0.00248)
        + 0.03646 * max(0.0, Q.mass - 144.0)
        - 0.006922 * max(0.0, Q.mass - 162.0)
        - 0.005198 * max(0.0, 123.0 - Q.mass)
        + 0.004991 * max(0.0, Q.mass_top50 - 137.0)
        + 144.9 * max(0.0, 0.436 - Q.max_dr)
        + 0.1447 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.001247 * max(0.0, 954.0 - Q.sum_pt_top50)
        - 2.077 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 1.768 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        + 0.0108 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 5.864 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 0.001654 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 8.959 * max(0.0, Q.C2 - 0.0528)
        + 0.0065 * max(0.0, Q.mass_top30 - 136.0)
        + 0.007984 * max(0.0, 63.0 - Q.mass_top50)
        - 3.704 * max(0.0, 6.82 - Q.log_sum_pt)
        + 3.043 * max(0.0, 6.99 - Q.log_sum_pt)
        - 0.01914 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.00171 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.008939 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.008931 * max(0.0, 131.0 - Q.mass)
        - 20.86 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 1.594 * max(0.0, Q.max_dr - 0.237)
        - 3.961 * max(0.0, 0.961 - Q.z_top50_slots)
        - 6.925 * max(0.0, 7.04 - Q.log_sum_pt)
    )


def score_q(Q):
    return (0.6944
        - 40.67 * max(0.0, 0.0555 - Q.girth)
        + 90.22 * max(0.0, 0.00622 - Q.girth2_top20)
        + 41.06 * max(0.0, 0.00578 - Q.lam1)
        + 0.1641 * max(0.0, Q.mass - 80.4)
        - 0.0805 * max(0.0, Q.mass - 91.2)
        - 162.5 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.06289 * max(0.0, 81.8 - Q.mass_top50)
        + 0.03246 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.0006837 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.03977 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.01171 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        - 11.78 * max(0.0, Q.log_sum_pt - 6.91)
        - 2.91 * max(0.0, Q.log_sum_pt - 6.99)
        + 29.47 * max(0.0, 0.435 - Q.max_dr)
        - 0.0007512 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.0001088 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.003489 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.3967 * max(0.0, Q.tau32 - 0.276)
        + 1.289 * max(0.0, 0.931 - Q.z_top20_slots)
        - 15.15 * max(0.0, Q.z_top50_slots - 0.96)
        - 171.0 * max(0.0, 0.000569 - Q.lam2)
        + 5.588 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 2.203 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 12.04 * max(0.0, Q.C2 - 0.106)
        - 0.2675 * max(0.0, 6.52 - Q.D2)
        + 20.76 * max(0.0, 0.063 - Q.girth)
        - 0.00432 * max(0.0, 80.4 - Q.mass)
        - 0.03855 * max(0.0, 109.0 - Q.mass)
        + 0.01402 * max(0.0, Q.n_particles - 22.7)
        + 0.008359 * max(0.0, 1070.0 - Q.sum_pt)
        + 2.079 * max(0.0, 0.451 - Q.tau21)
        - 2.796 * max(0.0, 0.578 - Q.tau32)
        - 0.04248 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.1434 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        + 0.006306 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 8.629 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.08649 * max(0.0, Q.mass - 63.2)
        + 200.2 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.01219 * max(0.0, Q.mass_top50 - 102.0)
        + 2.978 * max(0.0, Q.max_dr - 0.436)
        + 0.8601 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        + 65.48 * max(0.0, Q.lam1 - 0.00717)
        + 0.08695 * max(0.0, 91.2 - Q.mass)
        - 0.007195 * max(0.0, 101.0 - Q.mass)
        - 10.44 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.07216 * max(0.0, 73.1 - Q.mass_top50)
        - 0.007558 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.6197 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 22.25 * max(0.0, 0.00812 - Q.girth2_top20)
        - 53.72 * max(0.0, 0.0129 - Q.girth2_top40)
        - 0.01426 * max(0.0, 66.0 - Q.mass_top20)
        - 18.82 * max(0.0, 0.998 - Q.z_top50_slots)
        - 0.2644 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 32.7 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.01362 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.08514 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.3762 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.1148 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.1123 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 2.802 * max(0.0, Q.C2 - 0.0661)
        - 0.01114 * max(0.0, Q.mass - 104.0)
        - 0.03008 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.006127 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.06274 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.03626 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 0.0001212 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.0211 * max(0.0, Q.mass - 61.8)
        - 0.001237 * max(0.0, Q.mass - 142.0)
        + 0.03222 * max(0.0, 128.0 - Q.mass_top50)
        - 209.3 * max(0.0, Q.width - 0.026)
        - 4.7 * max(0.0, 0.0611 - Q.e2)
        - 9.139 * max(0.0, Q.lam1 - 0.00248)
        - 0.02352 * max(0.0, Q.mass - 144.0)
        + 0.00864 * max(0.0, Q.mass - 162.0)
        + 0.004596 * max(0.0, 123.0 - Q.mass)
        - 0.004258 * max(0.0, Q.mass_top50 - 137.0)
        - 29.91 * max(0.0, 0.436 - Q.max_dr)
        - 0.003911 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.0001806 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 0.4973 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 5.809 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        - 0.02305 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 10.44 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 0.001118 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 6.513 * max(0.0, Q.C2 - 0.0528)
        + 0.01673 * max(0.0, Q.mass_top30 - 136.0)
        - 0.00978 * max(0.0, 63.0 - Q.mass_top50)
        - 7.43 * max(0.0, 6.82 - Q.log_sum_pt)
        + 2.712 * max(0.0, 6.99 - Q.log_sum_pt)
        - 0.0005514 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.00212 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.008581 * max(0.0, 1020.0 - Q.sum_pt_top40)
        - 0.0002823 * max(0.0, 131.0 - Q.mass)
        - 0.1401 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 1.638 * max(0.0, Q.max_dr - 0.237)
        + 13.95 * max(0.0, 0.961 - Q.z_top50_slots)
        - 2.196 * max(0.0, 7.04 - Q.log_sum_pt)
    )


def score_W(Q):
    return (0.1126
        - 21.93 * max(0.0, 0.0555 - Q.girth)
        - 310.2 * max(0.0, 0.00622 - Q.girth2_top20)
        - 297.3 * max(0.0, 0.00578 - Q.lam1)
        - 0.1637 * max(0.0, Q.mass - 80.4)
        - 0.01811 * max(0.0, Q.mass - 91.2)
        - 3565.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        - 0.03463 * max(0.0, 81.8 - Q.mass_top50)
        + 0.02582 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.008677 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        + 0.01397 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.1394 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        - 27.33 * max(0.0, Q.log_sum_pt - 6.91)
        - 45.89 * max(0.0, Q.log_sum_pt - 6.99)
        + 220.4 * max(0.0, 0.435 - Q.max_dr)
        - 0.0001941 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.001034 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.01253 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.6326 * max(0.0, Q.tau32 - 0.276)
        + 0.2913 * max(0.0, 0.931 - Q.z_top20_slots)
        - 65.0 * max(0.0, Q.z_top50_slots - 0.96)
        + 1428.0 * max(0.0, 0.000569 - Q.lam2)
        + 4155.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 3.911 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 4.917 * max(0.0, Q.C2 - 0.106)
        + 0.03551 * max(0.0, 6.52 - Q.D2)
        - 13.64 * max(0.0, 0.063 - Q.girth)
        - 0.2514 * max(0.0, 80.4 - Q.mass)
        - 0.02958 * max(0.0, 109.0 - Q.mass)
        - 0.003578 * max(0.0, Q.n_particles - 22.7)
        - 0.003451 * max(0.0, 1070.0 - Q.sum_pt)
        + 0.8287 * max(0.0, 0.451 - Q.tau21)
        + 0.857 * max(0.0, 0.578 - Q.tau32)
        - 0.2555 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        + 0.2481 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        - 0.0009767 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 53.96 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.1061 * max(0.0, Q.mass - 63.2)
        - 238.3 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.04705 * max(0.0, Q.mass_top50 - 102.0)
        - 1.946 * max(0.0, Q.max_dr - 0.436)
        + 1.093 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        - 98.17 * max(0.0, Q.lam1 - 0.00717)
        + 0.4355 * max(0.0, 91.2 - Q.mass)
        - 0.1707 * max(0.0, 101.0 - Q.mass)
        - 42.56 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.001627 * max(0.0, 73.1 - Q.mass_top50)
        + 0.06418 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.1525 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        + 173.0 * max(0.0, 0.00812 - Q.girth2_top20)
        - 98.7 * max(0.0, 0.0129 - Q.girth2_top40)
        - 0.0004338 * max(0.0, 66.0 - Q.mass_top20)
        - 75.64 * max(0.0, 0.998 - Q.z_top50_slots)
        - 18.79 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 127.9 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        + 0.004659 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.0148 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.6593 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.6886 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.03242 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        - 7.278 * max(0.0, Q.C2 - 0.0661)
        + 0.1459 * max(0.0, Q.mass - 104.0)
        + 0.009081 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.004342 * max(0.0, 1010.0 - Q.sum_pt)
        + 0.8511 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        - 0.2443 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 0.0005159 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.1117 * max(0.0, Q.mass - 61.8)
        + 0.02587 * max(0.0, Q.mass - 142.0)
        + 0.009315 * max(0.0, 128.0 - Q.mass_top50)
        + 385.4 * max(0.0, Q.width - 0.026)
        - 9.302 * max(0.0, 0.0611 - Q.e2)
        + 68.85 * max(0.0, Q.lam1 - 0.00248)
        - 0.01422 * max(0.0, Q.mass - 144.0)
        - 0.01692 * max(0.0, Q.mass - 162.0)
        + 0.005751 * max(0.0, 123.0 - Q.mass)
        - 0.002907 * max(0.0, Q.mass_top50 - 137.0)
        - 222.0 * max(0.0, 0.436 - Q.max_dr)
        - 0.006797 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.01086 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 0.4821 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        - 0.8363 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        - 0.03554 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        - 30.46 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        + 0.003342 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        + 7.148 * max(0.0, Q.C2 - 0.0528)
        - 0.001629 * max(0.0, Q.mass_top30 - 136.0)
        + 0.08435 * max(0.0, 63.0 - Q.mass_top50)
        - 41.15 * max(0.0, 6.82 - Q.log_sum_pt)
        + 50.93 * max(0.0, 6.99 - Q.log_sum_pt)
        + 0.1733 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.01322 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.01205 * max(0.0, 1020.0 - Q.sum_pt_top40)
        - 0.0733 * max(0.0, 131.0 - Q.mass)
        - 1.471 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        + 2.757 * max(0.0, Q.max_dr - 0.237)
        + 14.9 * max(0.0, 0.961 - Q.z_top50_slots)
        - 0.6515 * max(0.0, 7.04 - Q.log_sum_pt)
    )


def score_Z(Q):
    return (-0.6226
        + 58.71 * max(0.0, 0.0555 - Q.girth)
        + 235.1 * max(0.0, 0.00622 - Q.girth2_top20)
        + 136.8 * max(0.0, 0.00578 - Q.lam1)
        - 0.3762 * max(0.0, Q.mass - 80.4)
        + 0.4487 * max(0.0, Q.mass - 91.2)
        + 3694.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.06172 * max(0.0, 81.8 - Q.mass_top50)
        - 0.1793 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.006785 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.08426 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.062 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        - 36.85 * max(0.0, Q.log_sum_pt - 6.91)
        - 45.25 * max(0.0, Q.log_sum_pt - 6.99)
        + 346.1 * max(0.0, 0.435 - Q.max_dr)
        - 0.0003637 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.00152 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.01156 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.6245 * max(0.0, Q.tau32 - 0.276)
        - 0.7208 * max(0.0, 0.931 - Q.z_top20_slots)
        - 109.8 * max(0.0, Q.z_top50_slots - 0.96)
        + 865.2 * max(0.0, 0.000569 - Q.lam2)
        - 3656.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 4.554 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 8.737 * max(0.0, Q.C2 - 0.106)
        - 0.09451 * max(0.0, 6.52 - Q.D2)
        - 24.67 * max(0.0, 0.063 - Q.girth)
        + 0.6643 * max(0.0, 80.4 - Q.mass)
        + 0.01754 * max(0.0, 109.0 - Q.mass)
        + 0.008847 * max(0.0, Q.n_particles - 22.7)
        + 0.0005798 * max(0.0, 1070.0 - Q.sum_pt)
        + 1.796 * max(0.0, 0.451 - Q.tau21)
        + 0.2842 * max(0.0, 0.578 - Q.tau32)
        + 0.4285 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        + 0.4632 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        + 0.002544 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 66.35 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.2 * max(0.0, Q.mass - 63.2)
        - 319.9 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        - 0.05105 * max(0.0, Q.mass_top50 - 102.0)
        + 3.053 * max(0.0, Q.max_dr - 0.436)
        + 2.214 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        - 236.5 * max(0.0, Q.lam1 - 0.00717)
        - 1.029 * max(0.0, 91.2 - Q.mass)
        + 0.3773 * max(0.0, 101.0 - Q.mass)
        + 14.49 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.04031 * max(0.0, 73.1 - Q.mass_top50)
        + 0.0467 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        - 2.026 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 383.6 * max(0.0, 0.00812 - Q.girth2_top20)
        + 285.0 * max(0.0, 0.0129 - Q.girth2_top40)
        + 0.02645 * max(0.0, 66.0 - Q.mass_top20)
        - 138.8 * max(0.0, 0.998 - Q.z_top50_slots)
        - 673.4 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        + 999.5 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        + 0.1303 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        + 0.3119 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        - 0.6623 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.1719 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        - 0.3775 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        - 0.2314 * max(0.0, Q.C2 - 0.0661)
        - 0.03843 * max(0.0, Q.mass - 104.0)
        + 0.09798 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.004514 * max(0.0, 1010.0 - Q.sum_pt)
        + 1.082 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        - 0.2232 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 0.0005314 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.2 * max(0.0, Q.mass - 61.8)
        + 0.04694 * max(0.0, Q.mass - 142.0)
        - 0.04159 * max(0.0, 128.0 - Q.mass_top50)
        + 341.9 * max(0.0, Q.width - 0.026)
        - 38.19 * max(0.0, 0.0611 - Q.e2)
        + 97.31 * max(0.0, Q.lam1 - 0.00248)
        - 0.07882 * max(0.0, Q.mass - 144.0)
        - 0.02132 * max(0.0, Q.mass - 162.0)
        + 0.00266 * max(0.0, 123.0 - Q.mass)
        + 0.02249 * max(0.0, Q.mass_top50 - 137.0)
        - 354.4 * max(0.0, 0.436 - Q.max_dr)
        + 0.1084 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.02247 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 1.77 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        - 1.247 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        - 0.05121 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        - 10.73 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        + 0.001332 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 8.636 * max(0.0, Q.C2 - 0.0528)
        + 0.05412 * max(0.0, Q.mass_top30 - 136.0)
        - 0.006959 * max(0.0, 63.0 - Q.mass_top50)
        - 41.47 * max(0.0, 6.82 - Q.log_sum_pt)
        + 46.65 * max(0.0, 6.99 - Q.log_sum_pt)
        + 0.08011 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.01603 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.01397 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.01287 * max(0.0, 131.0 - Q.mass)
        - 97.77 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 3.222 * max(0.0, Q.max_dr - 0.237)
        + 142.1 * max(0.0, 0.961 - Q.z_top50_slots)
        + 0.572 * max(0.0, 7.04 - Q.log_sum_pt)
    )


def score_t(Q):
    return (-1.849
        + 5.119 * max(0.0, 0.0555 - Q.girth)
        - 125.4 * max(0.0, 0.00622 - Q.girth2_top20)
        + 25.88 * max(0.0, 0.00578 - Q.lam1)
        + 0.04667 * max(0.0, Q.mass - 80.4)
        - 0.0465 * max(0.0, Q.mass - 91.2)
        - 1062.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        - 0.006109 * max(0.0, 81.8 - Q.mass_top50)
        - 0.08832 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 0.0001858 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.1219 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        + 0.006269 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        + 16.31 * max(0.0, Q.log_sum_pt - 6.91)
        + 1.811 * max(0.0, Q.log_sum_pt - 6.99)
        - 263.2 * max(0.0, 0.435 - Q.max_dr)
        + 0.001901 * max(0.0, 662.0 - Q.sum_pt_top2)
        - 0.001472 * max(0.0, 1070.0 - Q.sum_pt_top30)
        - 0.0002926 * max(0.0, Q.sum_pt_top50 - 930.0)
        + 0.4806 * max(0.0, Q.tau32 - 0.276)
        + 2.539 * max(0.0, 0.931 - Q.z_top20_slots)
        + 41.99 * max(0.0, Q.z_top50_slots - 0.96)
        - 144.0 * max(0.0, 0.000569 - Q.lam2)
        + 984.3 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 2.086 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        + 2.04 * max(0.0, Q.C2 - 0.106)
        + 0.1802 * max(0.0, 6.52 - Q.D2)
        + 26.12 * max(0.0, 0.063 - Q.girth)
        - 0.06164 * max(0.0, 80.4 - Q.mass)
        + 0.004276 * max(0.0, 109.0 - Q.mass)
        + 0.004169 * max(0.0, Q.n_particles - 22.7)
        - 0.001553 * max(0.0, 1070.0 - Q.sum_pt)
        - 1.386 * max(0.0, 0.451 - Q.tau21)
        + 1.119 * max(0.0, 0.578 - Q.tau32)
        - 0.1676 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.122 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        - 0.00436 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        - 18.65 * max(0.0, Q.log_sum_pt - 6.85)
        + 0.02294 * max(0.0, Q.mass - 63.2)
        + 52.21 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.0123 * max(0.0, Q.mass_top50 - 102.0)
        - 1.839 * max(0.0, Q.max_dr - 0.436)
        - 0.5321 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        + 33.81 * max(0.0, Q.lam1 - 0.00717)
        + 0.1284 * max(0.0, 91.2 - Q.mass)
        - 0.05469 * max(0.0, 101.0 - Q.mass)
        + 13.17 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        + 0.03519 * max(0.0, 73.1 - Q.mass_top50)
        - 0.02131 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.45 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        + 88.53 * max(0.0, 0.00812 - Q.girth2_top20)
        + 17.97 * max(0.0, 0.0129 - Q.girth2_top40)
        + 0.008104 * max(0.0, 66.0 - Q.mass_top20)
        + 42.67 * max(0.0, 0.998 - Q.z_top50_slots)
        + 278.7 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 315.9 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.01113 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.04962 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.1365 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        + 0.08097 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.05362 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 0.475 * max(0.0, Q.C2 - 0.0661)
        + 0.001231 * max(0.0, Q.mass - 104.0)
        - 0.02327 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        - 0.0003089 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.2824 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.01834 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        - 9.951e-05 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        - 0.03433 * max(0.0, Q.mass - 61.8)
        - 0.09621 * max(0.0, Q.mass - 142.0)
        + 0.004707 * max(0.0, 128.0 - Q.mass_top50)
        - 76.72 * max(0.0, Q.width - 0.026)
        - 35.03 * max(0.0, 0.0611 - Q.e2)
        - 89.34 * max(0.0, Q.lam1 - 0.00248)
        + 0.1055 * max(0.0, Q.mass - 144.0)
        - 0.0952 * max(0.0, Q.mass - 162.0)
        - 0.03951 * max(0.0, 123.0 - Q.mass)
        + 0.07581 * max(0.0, Q.mass_top50 - 137.0)
        + 264.1 * max(0.0, 0.436 - Q.max_dr)
        - 0.006896 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.005988 * max(0.0, 954.0 - Q.sum_pt_top50)
        - 8.206 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 13.57 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        + 0.1039 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 19.05 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 1.416e-05 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        + 5.269 * max(0.0, Q.C2 - 0.0528)
        - 0.02962 * max(0.0, Q.mass_top30 - 136.0)
        + 0.01652 * max(0.0, 63.0 - Q.mass_top50)
        - 7.662 * max(0.0, 6.82 - Q.log_sum_pt)
        + 8.32 * max(0.0, 6.99 - Q.log_sum_pt)
        + 0.1457 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.01908 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.01291 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.0008246 * max(0.0, 131.0 - Q.mass)
        + 17.24 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        + 1.949 * max(0.0, Q.max_dr - 0.237)
        - 32.64 * max(0.0, 0.961 - Q.z_top50_slots)
        - 0.2368 * max(0.0, 7.04 - Q.log_sum_pt)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['q'] - s['W'] > -1.6572717428207397:
        if s['q'] - s['Z'] > -1.1128073930740356:
            if s['g'] - s['t'] > -0.31510840356349945:
                if s['g'] - s['q'] > 0.013665744569152594:
                    if s['g'] - s['t'] > 0.2615371346473694:
                        if s['g'] - s['W'] > -0.015417271293699741:
                            if s['g'] - s['q'] > 0.2756725400686264:
                                return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.23399578034877777:
                                    return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 76% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.03852515108883381:
                            return 'g'   # 60% of the training jets here get this class from the formula
                        else:
                            return 't'   # 59% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.018091714940965176:
                        if s['q'] - s['Z'] > 0.16273635625839233:
                            return 'q'   # 97% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 82% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > -0.47430284321308136:
                            if Q.D2 > 3.1557570695877075:
                                return 'W'   # 74% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 65% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.26260538399219513:
                                return 'W'   # 95% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 81% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.26128457486629486:
                    if s['q'] - s['W'] > -0.31296442449092865:
                        if s['q'] - s['Z'] > -0.011889290064573288:
                            return 'q'   # 79% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 84% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 84% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.324128657579422:
                        return 'W'   # 68% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.2967568337917328:
                            return 'Z'   # 60% of the training jets here get this class from the formula
                        else:
                            return 't'   # 98% of the training jets here get this class from the formula
        else:
            if s['g'] - s['Z'] > -0.18266305327415466:
                return 'g'   # 85% of the training jets here get this class from the formula
            else:
                if s['Z'] - s['t'] > 0.1253153756260872:
                    return 'Z'   # 99% of the training jets here get this class from the formula
                else:
                    return 't'   # 79% of the training jets here get this class from the formula
    else:
        if s['W'] - s['Z'] > 0.10052231326699257:
            if s['g'] - s['W'] > -0.12212986499071121:
                return 'g'   # 90% of the training jets here get this class from the formula
            else:
                if s['W'] - s['t'] > 0.008948798291385174:
                    return 'W'   # 100% of the training jets here get this class from the formula
                else:
                    return 't'   # 66% of the training jets here get this class from the formula
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
