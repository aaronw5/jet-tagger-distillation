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

Test set (50,000 jets): accuracy 81.03% (the formula: 81.02%); same class as the formula for 97.84% of jets.  219 leaves, depth 15.
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
    return (0.1375
        - 26.88 * max(0.0, 0.0555 - Q.girth)
        + 32.06 * max(0.0, 0.00622 - Q.girth2_top20)
        + 48.73 * max(0.0, 0.00578 - Q.lam1)
        + 0.0085 * max(0.0, Q.mass - 80.4)
        + 0.0554 * max(0.0, Q.mass - 91.2)
        - 27.95 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.04527 * max(0.0, 81.8 - Q.mass_top50)
        + 0.04875 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.04689 * max(0.0, Q.z_dr_0_0p05 - 0.883)
        + 0.0008949 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.0456 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.00225 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
        - 0.006753 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        + 588.0 * max(0.0, 0.00073 - Q.girth2_top3)
        + 34.66 * max(0.0, Q.log_sum_pt - 6.91)
        - 26.71 * max(0.0, Q.log_sum_pt - 6.99)
        - 221.5 * max(0.0, 0.435 - Q.max_dr)
        - 0.02229 * max(0.0, 33.6 - Q.pt_9)
        + 0.002035 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.00296 * max(0.0, 1070.0 - Q.sum_pt_top30)
        - 0.001414 * max(0.0, Q.sum_pt_top50 - 930.0)
        + 1.403 * max(0.0, Q.tau32 - 0.276)
        - 3.766 * max(0.0, 0.931 - Q.z_top20_slots)
        - 23.09 * max(0.0, Q.z_top50_slots - 0.96)
        - 42.34 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
        + 0.001609 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
        - 0.9423 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
        + 0.3144 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
        + 0.8027 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
        - 0.000144 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        + 0.02851 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        + 5782.0 * max(0.0, 0.000115 - Q.girth2_top5)
        - 1232.0 * max(0.0, 0.000569 - Q.lam2)
        - 0.6066 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        + 2.362 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
        - 7.571 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 7.761e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
        + 68.93 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        - 10.91 * max(0.0, Q.C2 - 0.106)
        - 0.232 * max(0.0, 6.52 - Q.D2)
        + 29.58 * max(0.0, 0.063 - Q.girth)
        + 0.1241 * max(0.0, 80.4 - Q.mass)
        - 0.02729 * max(0.0, 109.0 - Q.mass)
        + 0.009679 * max(0.0, Q.n_particles - 22.7)
        + 0.005581 * max(0.0, 1070.0 - Q.sum_pt)
        + 0.8405 * max(0.0, 0.451 - Q.tau21)
        - 2.196 * max(0.0, 0.578 - Q.tau32)
        - 0.8717 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
        - 0.04009 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.06601 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        + 0.006461 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        - 2.1 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.06578 * max(0.0, Q.mass - 63.2)
        + 97.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.01348 * max(0.0, Q.mass_top50 - 102.0)
        + 0.01098 * max(0.0, Q.mass_top50 - 157.0)
        + 0.3986 * max(0.0, Q.max_dr - 0.436)
        + 0.7044 * max(0.0, Q.z_top30_slots - 0.943)
        - 5.95 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
        - 0.01685 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
        - 0.08923 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        - 2.544 * max(0.0, Q.e2 - 0.0521)
        + 53.49 * max(0.0, Q.lam1 - 0.00717)
        - 0.05677 * max(0.0, 91.2 - Q.mass)
        + 0.00349 * max(0.0, 101.0 - Q.mass)
        - 0.9757 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        + 0.001911 * max(0.0, 92.5 - Q.mass_top10)
        - 0.05443 * max(0.0, 73.1 - Q.mass_top50)
        - 0.008472 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.1919 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 15.51 * max(0.0, 0.00812 - Q.girth2_top20)
        - 35.73 * max(0.0, 0.0129 - Q.girth2_top40)
        + 0.001825 * max(0.0, 66.0 - Q.mass_top20)
        + 2.083 * max(0.0, 0.998 - Q.z_top50_slots)
        + 21.5 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 72.72 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.01056 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.03809 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.4781 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.2898 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.06332 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 3.8 * max(0.0, Q.C2 - 0.0661)
        - 15.51 * max(0.0, 0.00815 - Q.girth2_top30)
        + 61.51 * max(0.0, 0.00143 - Q.lam2)
        - 0.01115 * max(0.0, Q.mass - 104.0)
        + 0.006682 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.007739 * max(0.0, Q.n_particles - 49.3)
        + 6.768e-05 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.08676 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.03612 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        - 0.5034 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
        + 4.223e-06 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.00153 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
        + 0.0009191 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        - 38.71 * max(0.0, 0.0191 - Q.girth2_top15)
        + 0.01135 * max(0.0, Q.mass - 61.8)
        - 0.02278 * max(0.0, Q.mass - 142.0)
        + 0.01951 * max(0.0, 128.0 - Q.mass_top50)
        - 113.8 * max(0.0, Q.width - 0.026)
        + 1.022 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
        + 0.3163 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
        + 17.71 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
        + 0.02319 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        - 2.775 * max(0.0, 0.0611 - Q.e2)
        + 1.491 * max(0.0, 0.0254 - Q.girth2_top5)
        - 33.05 * max(0.0, Q.lam1 - 0.00248)
        + 0.009161 * max(0.0, Q.mass - 144.0)
        + 0.01245 * max(0.0, Q.mass - 162.0)
        - 0.00203 * max(0.0, 123.0 - Q.mass)
        - 0.003719 * max(0.0, Q.mass_top50 - 137.0)
        + 218.6 * max(0.0, 0.436 - Q.max_dr)
        - 0.0413 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.001802 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 0.3143 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 3.989 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        + 0.08445 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
        + 0.0009507 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
        + 0.0001454 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        - 7.168 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
        - 0.001935 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
        - 0.00171 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
        + 4.484 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 0.0002999 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 6.559 * max(0.0, Q.C2 - 0.0528)
        + 0.005676 * max(0.0, Q.mass_top30 - 136.0)
        - 0.008464 * max(0.0, 63.0 - Q.mass_top50)
        + 2866.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
        - 0.1013 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        - 4.203 * max(0.0, 6.82 - Q.log_sum_pt)
        + 8.649 * max(0.0, 6.99 - Q.log_sum_pt)
        - 0.01462 * max(0.0, Q.mass - 172.8)
        + 0.001042 * max(0.0, Q.mass_top10 - 66.7)
        + 0.00942 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.003054 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.005731 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.003099 * max(0.0, 131.0 - Q.mass)
        - 31.6 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 0.5688 * max(0.0, Q.max_dr - 0.237)
        + 6.397 * max(0.0, 0.961 - Q.z_top50_slots)
        - 6.702 * max(0.0, 7.04 - Q.log_sum_pt)
        + 0.08646 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
    )


def score_q(Q):
    return (-0.5286
        - 34.21 * max(0.0, 0.0555 - Q.girth)
        + 50.71 * max(0.0, 0.00622 - Q.girth2_top20)
        + 70.6 * max(0.0, 0.00578 - Q.lam1)
        + 0.1196 * max(0.0, Q.mass - 80.4)
        - 0.04217 * max(0.0, Q.mass - 91.2)
        - 552.8 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.05626 * max(0.0, 81.8 - Q.mass_top50)
        - 0.03116 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 0.2833 * max(0.0, Q.z_dr_0_0p05 - 0.883)
        + 0.0005829 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.04221 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.002047 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
        - 0.007133 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        - 324.6 * max(0.0, 0.00073 - Q.girth2_top3)
        - 11.54 * max(0.0, Q.log_sum_pt - 6.91)
        - 9.149 * max(0.0, Q.log_sum_pt - 6.99)
        - 42.15 * max(0.0, 0.435 - Q.max_dr)
        + 0.005866 * max(0.0, 33.6 - Q.pt_9)
        - 0.000881 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.000138 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.004245 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.2785 * max(0.0, Q.tau32 - 0.276)
        + 0.5947 * max(0.0, 0.931 - Q.z_top20_slots)
        + 5.383 * max(0.0, Q.z_top50_slots - 0.96)
        + 24.32 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
        - 0.0009637 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
        + 0.2281 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
        - 0.1125 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
        - 0.3015 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
        + 4.764e-05 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        + 0.0815 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        + 1315.0 * max(0.0, 0.000115 - Q.girth2_top5)
        + 44.26 * max(0.0, 0.000569 - Q.lam2)
        + 525.6 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 1.238 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
        + 0.4342 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 2.03e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
        - 10.42 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        - 10.86 * max(0.0, Q.C2 - 0.106)
        - 0.2548 * max(0.0, 6.52 - Q.D2)
        + 31.62 * max(0.0, 0.063 - Q.girth)
        + 0.04538 * max(0.0, 80.4 - Q.mass)
        - 0.03615 * max(0.0, 109.0 - Q.mass)
        + 0.007368 * max(0.0, Q.n_particles - 22.7)
        + 0.005558 * max(0.0, 1070.0 - Q.sum_pt)
        + 1.262 * max(0.0, 0.451 - Q.tau21)
        - 2.46 * max(0.0, 0.578 - Q.tau32)
        + 0.07233 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
        - 0.04819 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.07173 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        + 0.006445 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 12.72 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.07839 * max(0.0, Q.mass - 63.2)
        + 198.2 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.01006 * max(0.0, Q.mass_top50 - 102.0)
        + 0.006538 * max(0.0, Q.mass_top50 - 157.0)
        + 0.6597 * max(0.0, Q.max_dr - 0.436)
        - 1.377 * max(0.0, Q.z_top30_slots - 0.943)
        + 16.02 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
        + 0.008358 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
        + 0.3536 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        - 10.98 * max(0.0, Q.e2 - 0.0521)
        + 18.7 * max(0.0, Q.lam1 - 0.00717)
        + 0.05017 * max(0.0, 91.2 - Q.mass)
        - 0.001743 * max(0.0, 101.0 - Q.mass)
        - 2.022 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        + 0.002337 * max(0.0, 92.5 - Q.mass_top10)
        - 0.07204 * max(0.0, 73.1 - Q.mass_top50)
        - 0.0114 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.3222 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 5.616 * max(0.0, 0.00812 - Q.girth2_top20)
        - 80.31 * max(0.0, 0.0129 - Q.girth2_top40)
        - 0.00158 * max(0.0, 66.0 - Q.mass_top20)
        + 0.9907 * max(0.0, 0.998 - Q.z_top50_slots)
        + 87.65 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 185.4 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.01666 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.06256 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.4497 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.2559 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.08854 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 4.119 * max(0.0, Q.C2 - 0.0661)
        - 4.024 * max(0.0, 0.00815 - Q.girth2_top30)
        + 40.34 * max(0.0, 0.00143 - Q.lam2)
        - 0.01278 * max(0.0, Q.mass - 104.0)
        - 0.001532 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.02002 * max(0.0, Q.n_particles - 49.3)
        + 0.002102 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.05316 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.03511 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        - 0.6275 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
        - 5.065e-06 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.001434 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
        + 0.003498 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        - 34.74 * max(0.0, 0.0191 - Q.girth2_top15)
        + 0.01776 * max(0.0, Q.mass - 61.8)
        - 0.01867 * max(0.0, Q.mass - 142.0)
        + 0.02833 * max(0.0, 128.0 - Q.mass_top50)
        - 197.0 * max(0.0, Q.width - 0.026)
        + 0.8443 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
        + 0.3693 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
        + 16.53 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
        + 0.02775 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        - 3.514 * max(0.0, 0.0611 - Q.e2)
        + 0.2409 * max(0.0, 0.0254 - Q.girth2_top5)
        + 10.62 * max(0.0, Q.lam1 - 0.00248)
        - 0.005637 * max(0.0, Q.mass - 144.0)
        - 0.004425 * max(0.0, Q.mass - 162.0)
        + 0.0005033 * max(0.0, 123.0 - Q.mass)
        - 0.005626 * max(0.0, Q.mass_top50 - 137.0)
        + 41.83 * max(0.0, 0.436 - Q.max_dr)
        + 0.04008 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.002823 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 0.2414 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 5.043 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        + 0.09643 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
        + 0.006645 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
        - 0.004338 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        - 44.81 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
        + 0.001035 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
        - 0.03461 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
        + 14.93 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 0.001728 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 8.49 * max(0.0, Q.C2 - 0.0528)
        + 0.0205 * max(0.0, Q.mass_top30 - 136.0)
        - 0.01503 * max(0.0, 63.0 - Q.mass_top50)
        + 4372.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
        - 0.2356 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        - 8.776 * max(0.0, 6.82 - Q.log_sum_pt)
        + 9.352 * max(0.0, 6.99 - Q.log_sum_pt)
        + 0.01087 * max(0.0, Q.mass - 172.8)
        - 0.0007505 * max(0.0, Q.mass_top10 - 66.7)
        + 0.01413 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.0003491 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.004968 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.001928 * max(0.0, 131.0 - Q.mass)
        - 24.54 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 0.3771 * max(0.0, Q.max_dr - 0.237)
        + 4.263 * max(0.0, 0.961 - Q.z_top50_slots)
        - 2.675 * max(0.0, 7.04 - Q.log_sum_pt)
        + 0.2457 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
    )


def score_W(Q):
    return (2.149
        - 16.85 * max(0.0, 0.0555 - Q.girth)
        - 346.3 * max(0.0, 0.00622 - Q.girth2_top20)
        - 304.5 * max(0.0, 0.00578 - Q.lam1)
        - 0.4408 * max(0.0, Q.mass - 80.4)
        + 0.2472 * max(0.0, Q.mass - 91.2)
        - 3938.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        - 0.02504 * max(0.0, 81.8 - Q.mass_top50)
        - 0.06906 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        - 1.922 * max(0.0, Q.z_dr_0_0p05 - 0.883)
        - 0.007619 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        + 0.04129 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        + 0.0167 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
        - 0.1665 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        + 8.569 * max(0.0, 0.00073 - Q.girth2_top3)
        - 46.24 * max(0.0, Q.log_sum_pt - 6.91)
        - 19.24 * max(0.0, Q.log_sum_pt - 6.99)
        + 286.3 * max(0.0, 0.435 - Q.max_dr)
        + 9.62e-05 * max(0.0, 33.6 - Q.pt_9)
        - 0.0002836 * max(0.0, 662.0 - Q.sum_pt_top2)
        - 0.0002643 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.008734 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.3086 * max(0.0, Q.tau32 - 0.276)
        - 1.022 * max(0.0, 0.931 - Q.z_top20_slots)
        - 12.83 * max(0.0, Q.z_top50_slots - 0.96)
        - 1.712 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
        + 0.0007509 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
        + 0.0229 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
        + 0.1194 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
        - 0.2486 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
        + 5.035e-06 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        + 0.03552 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        - 3070.0 * max(0.0, 0.000115 - Q.girth2_top5)
        + 506.3 * max(0.0, 0.000569 - Q.lam2)
        + 4477.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 1.567 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
        - 2.43 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        + 3.613e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
        - 31.36 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        - 4.881 * max(0.0, Q.C2 - 0.106)
        + 0.1364 * max(0.0, 6.52 - Q.D2)
        - 15.15 * max(0.0, 0.063 - Q.girth)
        + 0.023 * max(0.0, 80.4 - Q.mass)
        - 0.0275 * max(0.0, 109.0 - Q.mass)
        - 0.004757 * max(0.0, Q.n_particles - 22.7)
        + 0.002187 * max(0.0, 1070.0 - Q.sum_pt)
        + 1.134 * max(0.0, 0.451 - Q.tau21)
        + 0.6675 * max(0.0, 0.578 - Q.tau32)
        + 0.5453 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
        - 0.1899 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        + 0.1458 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        - 0.002854 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 50.18 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.04513 * max(0.0, Q.mass - 63.2)
        - 278.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.04936 * max(0.0, Q.mass_top50 - 102.0)
        - 0.00361 * max(0.0, Q.mass_top50 - 157.0)
        + 0.9261 * max(0.0, Q.max_dr - 0.436)
        - 6.237 * max(0.0, Q.z_top30_slots - 0.943)
        + 19.38 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
        + 0.01286 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
        + 0.8441 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        + 11.57 * max(0.0, Q.e2 - 0.0521)
        + 65.07 * max(0.0, Q.lam1 - 0.00717)
        + 0.1896 * max(0.0, 91.2 - Q.mass)
        - 0.1814 * max(0.0, 101.0 - Q.mass)
        - 24.98 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        + 0.0006224 * max(0.0, 92.5 - Q.mass_top10)
        - 0.005829 * max(0.0, 73.1 - Q.mass_top50)
        + 0.03956 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.6326 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        + 224.4 * max(0.0, 0.00812 - Q.girth2_top20)
        - 43.7 * max(0.0, 0.0129 - Q.girth2_top40)
        - 0.003708 * max(0.0, 66.0 - Q.mass_top20)
        + 4.677 * max(0.0, 0.998 - Q.z_top50_slots)
        - 49.45 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        + 8.414 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.0252 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.04058 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.415 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.3869 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.06091 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        - 8.256 * max(0.0, Q.C2 - 0.0661)
        - 56.85 * max(0.0, 0.00815 - Q.girth2_top30)
        + 256.0 * max(0.0, 0.00143 - Q.lam2)
        + 0.1321 * max(0.0, Q.mass - 104.0)
        + 0.005064 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        - 0.05045 * max(0.0, Q.n_particles - 49.3)
        + 0.03524 * max(0.0, 1010.0 - Q.sum_pt)
        + 0.8274 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        - 0.2748 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 1.886 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
        + 0.0004202 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        - 0.001887 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
        + 0.02755 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        + 20.91 * max(0.0, 0.0191 - Q.girth2_top15)
        + 0.06699 * max(0.0, Q.mass - 61.8)
        + 0.06728 * max(0.0, Q.mass - 142.0)
        + 0.0179 * max(0.0, 128.0 - Q.mass_top50)
        + 373.8 * max(0.0, Q.width - 0.026)
        - 1.097 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
        + 1.845 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
        - 1.855 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
        - 0.0366 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        + 3.627 * max(0.0, 0.0611 - Q.e2)
        + 0.3755 * max(0.0, 0.0254 - Q.girth2_top5)
        - 85.18 * max(0.0, Q.lam1 - 0.00248)
        - 0.05322 * max(0.0, Q.mass - 144.0)
        - 0.0009502 * max(0.0, Q.mass - 162.0)
        + 0.008396 * max(0.0, 123.0 - Q.mass)
        - 0.008828 * max(0.0, Q.mass_top50 - 137.0)
        - 289.4 * max(0.0, 0.436 - Q.max_dr)
        + 0.08102 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.006724 * max(0.0, 954.0 - Q.sum_pt_top50)
        + 0.416 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        - 1.288 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        - 0.3003 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
        + 0.008066 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
        - 0.03633 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 98.75 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
        - 0.006889 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
        + 0.05059 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
        - 24.01 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        + 0.004236 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        + 13.58 * max(0.0, Q.C2 - 0.0528)
        - 0.002578 * max(0.0, Q.mass_top30 - 136.0)
        + 0.06789 * max(0.0, 63.0 - Q.mass_top50)
        - 6271.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
        - 0.04958 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        - 30.24 * max(0.0, 6.82 - Q.log_sum_pt)
        + 16.11 * max(0.0, 6.99 - Q.log_sum_pt)
        - 0.004501 * max(0.0, Q.mass - 172.8)
        - 0.0002662 * max(0.0, Q.mass_top10 - 66.7)
        + 0.07628 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        - 0.001491 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.01057 * max(0.0, 1020.0 - Q.sum_pt_top40)
        - 0.07736 * max(0.0, 131.0 - Q.mass)
        - 9.082 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        + 1.475 * max(0.0, Q.max_dr - 0.237)
        - 70.95 * max(0.0, 0.961 - Q.z_top50_slots)
        + 2.767 * max(0.0, 7.04 - Q.log_sum_pt)
        - 0.02618 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
    )


def score_Z(Q):
    return (0.5819
        + 56.0 * max(0.0, 0.0555 - Q.girth)
        + 235.2 * max(0.0, 0.00622 - Q.girth2_top20)
        + 215.3 * max(0.0, 0.00578 - Q.lam1)
        - 0.6189 * max(0.0, Q.mass - 80.4)
        + 0.6856 * max(0.0, Q.mass - 91.2)
        + 3224.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        + 0.04911 * max(0.0, 81.8 - Q.mass_top50)
        - 0.1611 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 5.587 * max(0.0, Q.z_dr_0_0p05 - 0.883)
        + 0.007379 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.11 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        + 0.01353 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
        - 0.08067 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        + 36.59 * max(0.0, 0.00073 - Q.girth2_top3)
        - 51.97 * max(0.0, Q.log_sum_pt - 6.91)
        - 14.56 * max(0.0, Q.log_sum_pt - 6.99)
        + 306.0 * max(0.0, 0.435 - Q.max_dr)
        + 0.0001875 * max(0.0, 33.6 - Q.pt_9)
        - 0.0001282 * max(0.0, 662.0 - Q.sum_pt_top2)
        + 0.0007305 * max(0.0, 1070.0 - Q.sum_pt_top30)
        + 0.006846 * max(0.0, Q.sum_pt_top50 - 930.0)
        - 0.305 * max(0.0, Q.tau32 - 0.276)
        - 1.331 * max(0.0, 0.931 - Q.z_top20_slots)
        - 51.49 * max(0.0, Q.z_top50_slots - 0.96)
        + 6.222 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
        - 0.0003425 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
        + 0.04132 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
        + 0.06118 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
        - 0.3645 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
        - 1.062e-05 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        - 0.2602 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        - 3673.0 * max(0.0, 0.000115 - Q.girth2_top5)
        + 803.5 * max(0.0, 0.000569 - Q.lam2)
        - 3010.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        - 0.3601 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
        - 2.147 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        + 1.208e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
        + 38.62 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        - 6.009 * max(0.0, Q.C2 - 0.106)
        - 0.02004 * max(0.0, 6.52 - Q.D2)
        - 14.0 * max(0.0, 0.063 - Q.girth)
        + 0.9028 * max(0.0, 80.4 - Q.mass)
        + 0.02845 * max(0.0, 109.0 - Q.mass)
        + 0.005233 * max(0.0, Q.n_particles - 22.7)
        + 0.005472 * max(0.0, 1070.0 - Q.sum_pt)
        + 0.1591 * max(0.0, 0.451 - Q.tau21)
        - 0.03529 * max(0.0, 0.578 - Q.tau32)
        + 3.045 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
        + 0.4427 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        + 0.3664 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        - 0.0002056 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        + 56.34 * max(0.0, Q.log_sum_pt - 6.85)
        - 0.1338 * max(0.0, Q.mass - 63.2)
        - 321.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        - 0.06578 * max(0.0, Q.mass_top50 - 102.0)
        + 0.02238 * max(0.0, Q.mass_top50 - 157.0)
        + 3.155 * max(0.0, Q.max_dr - 0.436)
        - 4.5 * max(0.0, Q.z_top30_slots - 0.943)
        + 25.03 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
        + 0.0909 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
        + 1.46 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        + 20.2 * max(0.0, Q.e2 - 0.0521)
        - 183.0 * max(0.0, Q.lam1 - 0.00717)
        - 1.248 * max(0.0, 91.2 - Q.mass)
        + 0.3689 * max(0.0, 101.0 - Q.mass)
        + 41.3 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.006865 * max(0.0, 92.5 - Q.mass_top10)
        - 0.02372 * max(0.0, 73.1 - Q.mass_top50)
        + 0.05082 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        - 1.911 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        - 344.9 * max(0.0, 0.00812 - Q.girth2_top20)
        + 293.2 * max(0.0, 0.0129 - Q.girth2_top40)
        + 0.02777 * max(0.0, 66.0 - Q.mass_top20)
        - 47.61 * max(0.0, 0.998 - Q.z_top50_slots)
        - 666.3 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        + 981.9 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        + 0.1347 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        + 0.2834 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        - 0.806 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        + 0.06346 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        - 0.3515 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        + 0.1517 * max(0.0, Q.C2 - 0.0661)
        - 83.8 * max(0.0, 0.00815 - Q.girth2_top30)
        + 225.7 * max(0.0, 0.00143 - Q.lam2)
        - 0.03604 * max(0.0, Q.mass - 104.0)
        + 0.0775 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        - 0.07009 * max(0.0, Q.n_particles - 49.3)
        + 0.03048 * max(0.0, 1010.0 - Q.sum_pt)
        + 0.9559 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        - 0.2629 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        + 2.599 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
        + 0.0004716 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        - 0.005555 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
        + 0.03663 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        + 7.799 * max(0.0, 0.0191 - Q.girth2_top15)
        + 0.1477 * max(0.0, Q.mass - 61.8)
        + 0.08577 * max(0.0, Q.mass - 142.0)
        - 0.03004 * max(0.0, 128.0 - Q.mass_top50)
        + 347.9 * max(0.0, Q.width - 0.026)
        - 0.514 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
        + 0.418 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
        - 22.46 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
        + 0.004431 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        - 11.56 * max(0.0, 0.0611 - Q.e2)
        + 9.281 * max(0.0, 0.0254 - Q.girth2_top5)
        + 14.55 * max(0.0, Q.lam1 - 0.00248)
        - 0.08431 * max(0.0, Q.mass - 144.0)
        - 0.0683 * max(0.0, Q.mass - 162.0)
        + 0.003111 * max(0.0, 123.0 - Q.mass)
        - 0.005274 * max(0.0, Q.mass_top50 - 137.0)
        - 314.7 * max(0.0, 0.436 - Q.max_dr)
        + 0.1017 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.01657 * max(0.0, 954.0 - Q.sum_pt_top50)
        - 0.09813 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        - 1.873 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        - 0.2668 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
        + 0.005417 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
        - 0.03192 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 5.782 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
        + 0.0007939 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
        - 0.04383 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
        - 7.736 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        + 0.0004686 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        - 5.882 * max(0.0, Q.C2 - 0.0528)
        + 0.04418 * max(0.0, Q.mass_top30 - 136.0)
        - 0.01554 * max(0.0, 63.0 - Q.mass_top50)
        - 1183.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
        - 0.2199 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        - 20.38 * max(0.0, 6.82 - Q.log_sum_pt)
        + 8.443 * max(0.0, 6.99 - Q.log_sum_pt)
        + 0.02862 * max(0.0, Q.mass - 172.8)
        + 0.005252 * max(0.0, Q.mass_top10 - 66.7)
        + 0.03911 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.002845 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.01292 * max(0.0, 1020.0 - Q.sum_pt_top40)
        + 0.01028 * max(0.0, 131.0 - Q.mass)
        - 127.5 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        - 3.22 * max(0.0, Q.max_dr - 0.237)
        + 40.83 * max(0.0, 0.961 - Q.z_top50_slots)
        + 4.03 * max(0.0, 7.04 - Q.log_sum_pt)
        + 1.795 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
    )


def score_t(Q):
    return (-1.152
        + 11.5 * max(0.0, 0.0555 - Q.girth)
        - 89.99 * max(0.0, 0.00622 - Q.girth2_top20)
        - 29.23 * max(0.0, 0.00578 - Q.lam1)
        + 0.3265 * max(0.0, Q.mass - 80.4)
        - 0.3201 * max(0.0, Q.mass - 91.2)
        - 622.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
        - 0.01069 * max(0.0, 81.8 - Q.mass_top50)
        - 0.08241 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
        + 2.003 * max(0.0, Q.z_dr_0_0p05 - 0.883)
        + 0.0004185 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
        - 0.1081 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
        - 0.005036 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
        + 0.02175 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        - 154.8 * max(0.0, 0.00073 - Q.girth2_top3)
        + 6.995 * max(0.0, Q.log_sum_pt - 6.91)
        + 3.723 * max(0.0, Q.log_sum_pt - 6.99)
        - 247.2 * max(0.0, 0.435 - Q.max_dr)
        - 0.000983 * max(0.0, 33.6 - Q.pt_9)
        + 3.077e-05 * max(0.0, 662.0 - Q.sum_pt_top2)
        - 0.0003914 * max(0.0, 1070.0 - Q.sum_pt_top30)
        - 0.0004661 * max(0.0, Q.sum_pt_top50 - 930.0)
        + 0.04169 * max(0.0, Q.tau32 - 0.276)
        + 1.029 * max(0.0, 0.931 - Q.z_top20_slots)
        + 39.64 * max(0.0, Q.z_top50_slots - 0.96)
        + 4.876 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
        + 7.818e-05 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
        + 0.07918 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
        - 0.01349 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
        + 0.136 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
        + 2.069e-05 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        + 0.04124 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        - 1790.0 * max(0.0, 0.000115 - Q.girth2_top5)
        - 49.69 * max(0.0, 0.000569 - Q.lam2)
        + 403.7 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
        + 0.3514 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
        - 1.386 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
        - 2.994e-06 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
        + 53.9 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        + 0.7099 * max(0.0, Q.C2 - 0.106)
        + 0.05045 * max(0.0, 6.52 - Q.D2)
        + 10.69 * max(0.0, 0.063 - Q.girth)
        - 0.3401 * max(0.0, 80.4 - Q.mass)
        + 0.003875 * max(0.0, 109.0 - Q.mass)
        - 0.004445 * max(0.0, Q.n_particles - 22.7)
        + 0.001619 * max(0.0, 1070.0 - Q.sum_pt)
        - 0.5062 * max(0.0, 0.451 - Q.tau21)
        + 0.7561 * max(0.0, 0.578 - Q.tau32)
        - 2.999 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
        - 0.1615 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
        - 0.03743 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
        - 0.002633 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        - 12.7 * max(0.0, Q.log_sum_pt - 6.85)
        + 0.006296 * max(0.0, Q.mass - 63.2)
        + 32.4 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
        + 0.005227 * max(0.0, Q.mass_top50 - 102.0)
        + 0.0006993 * max(0.0, Q.mass_top50 - 157.0)
        - 1.262 * max(0.0, Q.max_dr - 0.436)
        + 3.424 * max(0.0, Q.z_top30_slots - 0.943)
        - 32.74 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
        - 0.06022 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
        - 0.2994 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        - 0.07099 * max(0.0, Q.e2 - 0.0521)
        + 26.28 * max(0.0, Q.lam1 - 0.00717)
        + 0.3961 * max(0.0, 91.2 - Q.mass)
        - 0.04587 * max(0.0, 101.0 - Q.mass)
        - 5.15 * max(0.0, Q.mass_over_sum_pt - 0.0534)
        - 0.002216 * max(0.0, 92.5 - Q.mass_top10)
        + 0.03851 * max(0.0, 73.1 - Q.mass_top50)
        - 0.0116 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
        + 0.008762 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        + 73.92 * max(0.0, 0.00812 - Q.girth2_top20)
        - 24.34 * max(0.0, 0.0129 - Q.girth2_top40)
        - 0.00188 * max(0.0, 66.0 - Q.mass_top20)
        + 28.37 * max(0.0, 0.998 - Q.z_top50_slots)
        + 281.4 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
        - 308.6 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
        - 0.03005 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
        - 0.04994 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
        + 0.2409 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
        - 0.09609 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
        + 0.05958 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        - 1.047 * max(0.0, Q.C2 - 0.0661)
        + 3.449 * max(0.0, 0.00815 - Q.girth2_top30)
        - 111.9 * max(0.0, 0.00143 - Q.lam2)
        + 0.006946 * max(0.0, Q.mass - 104.0)
        + 0.0009722 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
        + 0.02403 * max(0.0, Q.n_particles - 49.3)
        + 0.00404 * max(0.0, 1010.0 - Q.sum_pt)
        - 0.2197 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
        + 0.03833 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
        - 0.8079 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
        - 0.0001982 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
        + 0.01498 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
        - 0.01137 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        + 0.4957 * max(0.0, 0.0191 - Q.girth2_top15)
        - 0.0162 * max(0.0, Q.mass - 61.8)
        - 0.04549 * max(0.0, Q.mass - 142.0)
        + 0.007864 * max(0.0, 128.0 - Q.mass_top50)
        - 76.48 * max(0.0, Q.width - 0.026)
        + 0.8819 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
        + 0.2486 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
        - 0.8735 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
        + 0.002524 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        - 34.83 * max(0.0, 0.0611 - Q.e2)
        - 84.73 * max(0.0, 0.0254 - Q.girth2_top5)
        - 58.34 * max(0.0, Q.lam1 - 0.00248)
        + 0.03228 * max(0.0, Q.mass - 144.0)
        - 0.03882 * max(0.0, Q.mass - 162.0)
        - 0.03327 * max(0.0, 123.0 - Q.mass)
        + 0.08529 * max(0.0, Q.mass_top50 - 137.0)
        + 247.0 * max(0.0, 0.436 - Q.max_dr)
        - 0.01667 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
        - 0.005629 * max(0.0, 954.0 - Q.sum_pt_top50)
        - 9.01 * max(0.0, Q.z_dr_0_0p05 - 0.769)
        + 12.64 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
        + 0.6311 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
        + 0.1091 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
        + 0.06416 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        + 5.369 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
        + 0.001901 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
        + 0.009849 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
        + 17.02 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
        - 0.0004775 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        + 3.967 * max(0.0, Q.C2 - 0.0528)
        - 0.03218 * max(0.0, Q.mass_top30 - 136.0)
        + 0.006357 * max(0.0, 63.0 - Q.mass_top50)
        - 3437.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
        + 0.3653 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        - 12.0 * max(0.0, 6.82 - Q.log_sum_pt)
        + 0.3001 * max(0.0, 6.99 - Q.log_sum_pt)
        - 0.05559 * max(0.0, Q.mass - 172.8)
        - 0.01469 * max(0.0, Q.mass_top10 - 66.7)
        + 0.1292 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
        + 0.01627 * max(0.0, 1020.0 - Q.sum_pt)
        - 0.009418 * max(0.0, 1020.0 - Q.sum_pt_top40)
        - 0.006557 * max(0.0, 131.0 - Q.mass)
        + 46.44 * max(0.0, 0.089 - Q.mass_over_sum_pt)
        + 1.109 * max(0.0, Q.max_dr - 0.237)
        - 23.78 * max(0.0, 0.961 - Q.z_top50_slots)
        - 0.5618 * max(0.0, 7.04 - Q.log_sum_pt)
        - 1.09 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['q'] - s['W'] > -1.6853109002113342:
        if s['q'] - s['Z'] > -1.1045098304748535:
            if s['g'] - s['t'] > -0.19970905780792236:
                if s['g'] - s['q'] > -0.024554334580898285:
                    if s['g'] - s['W'] > 0.051825057715177536:
                        if s['g'] - s['t'] > 0.19363577663898468:
                            if s['g'] - s['Z'] > 0.3042866885662079:
                                if s['g'] - s['q'] > 0.06359722092747688:
                                    if s['g'] - s['t'] > 0.6481331586837769:
                                        if s['g'] - s['W'] > 0.20854128152132034:
                                            if Q.sum_pt_top30 > 729.44921875:
                                                if Q.max_dr > 0.5093667507171631:
                                                    if s['g'] - s['q'] > 0.4961162656545639:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.594063013792038:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 0.5603179931640625:
                                                        if Q.girth2 > 0.04723294638097286:
                                                            if Q.sum_pt_top50 > 1012.6484375:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top20 > 88.77266693115234:
                                                    if s['g'] - s['t'] > 0.9699320793151855:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.5297921895980835:
                                                if Q.max_dr > 0.2909587323665619:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 894.4208984375:
                                            if Q.mass_over_sum_pt_sq > 0.03353581950068474:
                                                if Q.sum_pt_top50 > 1057.6923828125:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 218.19112396240234:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top20 > 128.83826446533203:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 74.05160522460938:
                                        if Q.e2 > 0.02761991135776043:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.4800656586885452:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > 3.278036117553711:
                                    if s['g'] - s['Z'] > -0.20425095409154892:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > -0.093571737408638:
                                        if Q.sum_pt_top40 > 1121.31640625:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 2.542064905166626:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 1145.4033203125:
                                            return 'g'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 90% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.016166066750884056:
                                if Q.sum_pt_top50 > 909.93798828125:
                                    if s['Z'] - s['t'] > 0.011557164136320353:
                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 2.4164226055145264:
                                            if Q.mass_over_sum_pt_sq > 0.03184996917843819:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 0.030109049752354622:
                                                    if Q.sum_pt_top50 > 936.01953125:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.16470972448587418:
                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > 0.077644232660532:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.015862290747463703:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 993.77392578125:
                                                        if Q.D2 > 2.559549331665039:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.1812112182378769:
                                                                return 't'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -1.6678151488304138:
                                        return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > -0.05582370422780514:
                                    if s['q'] - s['W'] > 2.6150180101394653:
                                        if Q.sum_pt_top50 > 928.7734375:
                                            if s['Z'] - s['t'] > -6.726746320724487:
                                                if s['q'] - s['W'] > 3.1179378032684326:
                                                    if Q.girth2 > 0.025203347206115723:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top20 > 133.50057220458984:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top40 > 989.224609375:
                                                            return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.35024070739746094:
                                                    return 't'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 712.10546875:
                                                if Q.z_dr_0p2_0p4 > 0.3208436220884323:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top50 > 0.007393994368612766:
                                            if Q.lam1 > 0.015515283681452274:
                                                if Q.mass_top20 > 140.68920135498047:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 83% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.3952672481536865:
                            if s['g'] - s['Z'] > 0.0474324319511652:
                                if Q.C2 > 0.051247840747237206:
                                    if s['g'] - s['W'] > -0.13757146894931793:
                                        if Q.mass_top20 > 51.74909591674805:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.06418971717357635:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 7.065571069717407:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top15 > 0.004391508409753442:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 73.63446807861328:
                                        return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 90% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.23817437887191772:
                                return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 79% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 0.05261661671102047:
                        if s['q'] - s['Z'] > 0.1580142304301262:
                            if s['g'] - s['q'] > -0.10370056331157684:
                                if s['g'] - s['q'] > -0.07405883446335793:
                                    if Q.n_dr_0p2_0p4 > 6.5:
                                        if Q.log_sum_pt > 6.868006229400635:
                                            if Q.sum_pt_top3 > 738.84375:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 8.801763760857284e-05:
                                            return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 78% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > 0.4169529527425766:
                                    if s['q'] - s['Z'] > 0.4520179033279419:
                                        if s['g'] - s['q'] > -0.20767737179994583:
                                            return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.038549765944480896:
                                                if s['g'] - s['q'] > -0.5714931488037109:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 4.769692420959473:
                                        if Q.planar_flow > 0.4653812199831009:
                                            if Q.sum_pt_top2 > 398.375:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > 0.2988123297691345:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0030106552876532078:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 92% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > -0.3045003116130829:
                                if s['W'] - s['t'] > -1.858214020729065:
                                    if Q.log_sum_pt > 6.90201210975647:
                                        if s['g'] - s['t'] > 1.8012099862098694:
                                            return 'q'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > 2.9594677686691284:
                                    if Q.LHA > 0.2075968086719513:
                                        if Q.sum_pt_top50 > 982.5499267578125:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 97% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > -0.37329454720020294:
                            if Q.D2 > 3.3835434913635254:
                                if Q.mass_top50 > 78.44907760620117:
                                    if s['W'] - s['Z'] > 0.26464465260505676:
                                        return 'q'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.5711697041988373:
                                        return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.14208068698644638:
                                            if Q.mass_top50 > 75.26679992675781:
                                                return 'q'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 4.585673570632935:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.004303656984120607:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top50 > 0.004867015173658729:
                                    if s['q'] - s['Z'] > 0.05316326580941677:
                                        if s['q'] - s['W'] > -0.1252540424466133:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 85% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > 0.2770881950855255:
                                if Q.mass > 68.83484268188477:
                                    return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.023005611263215542:
                                        if s['q'] - s['W'] > -0.7879992127418518:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.03345266729593277:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 80% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.19497019052505493:
                    if s['q'] - s['W'] > -0.3378528654575348:
                        if s['q'] - s['Z'] > 0.07579827308654785:
                            if s['q'] - s['t'] > -0.00747003429569304:
                                if s['q'] - s['W'] > 0.00884327944368124:
                                    if s['q'] - s['t'] > 0.25197678804397583:
                                        return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.874328851699829:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top50 > 0.007145127514377236:
                                                return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.02029347699135542:
                                        if Q.mass_over_sum_pt > 0.08139373734593391:
                                            return 'q'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 1005.33349609375:
                                    if s['q'] - s['t'] > -0.13451670110225677:
                                        if s['Z'] - s['t'] > -2.4193848371505737:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 968.4384765625:
                                        if Q.D2 > 1.6279872059822083:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 2.0012764930725098:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.052395785227417946:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 3.115035891532898:
                                if s['q'] - s['Z'] > -0.759412556886673:
                                    return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 94% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.06321351230144501:
                            if s['q'] - s['W'] > -0.778190016746521:
                                if Q.mass > 71.19627380371094:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.029635454528033733:
                                        return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.49617235362529755:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 75% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > 0.46466317772865295:
                                    return 'W'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.2686100006103516:
                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 89% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 92% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.20048751682043076:
                        if s['W'] - s['Z'] > 0.09357357397675514:
                            if s['W'] - s['t'] > 0.18412910401821136:
                                return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 1.8587809205055237:
                                    return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 86% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.10888952389359474:
                            if s['W'] - s['t'] > -3.1403385400772095:
                                return 'Z'   # 86% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.24619103223085403:
                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 737.96875:
                                if s['q'] - s['t'] > -0.3924388885498047:
                                    if Q.sum_pt_top40 > 990.59619140625:
                                        if Q.mass_top50 > 174.29291534423828:
                                            return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.0874776840209961:
                                            if Q.log_sum_pt > 6.88593602180481:
                                                if s['q'] - s['t'] > -0.28455832600593567:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 889.283203125:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 49% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.4236491173505783:
                                        if Q.mass_top50 > 173.90445709228516:
                                            if Q.sum_pt_top50 > 1034.642578125:
                                                if Q.girth2_top5 > 0.013955434318631887:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > -0.3456025570631027:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > -0.5890864431858063:
                                                if s['g'] - s['q'] > -1.7311829328536987:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.03280402533710003:
                                                    if s['g'] - s['W'] > 4.716263055801392:
                                                        if s['g'] - s['q'] > 0.3940744698047638:
                                                            if Q.tau21 > 0.35719089210033417:
                                                                return 't'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -1.2684306502342224:
                                    if s['g'] - s['Z'] > 5.347486257553101:
                                        if Q.sum_pt_top40 > 787.529296875:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > -0.7355778515338898:
                                                if Q.tau21 > 0.4275037348270416:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.558425426483154:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 95% of the training jets here get this class from the formula
        else:
            if s['g'] - s['Z'] > -0.049638781696558:
                if s['g'] - s['Z'] > 0.30452851951122284:
                    if s['g'] - s['t'] > 0.42054593563079834:
                        return 'g'   # 97% of the training jets here get this class from the formula
                    else:
                        return 't'   # 56% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > 1.0667842030525208:
                        return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top50 > 101.00012969970703:
                            return 'g'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.20987924188375473:
                                return 'Z'   # 66% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
            else:
                if s['Z'] - s['t'] > 0.17485816776752472:
                    if s['W'] - s['Z'] > -0.062260910868644714:
                        if Q.C2 > 0.06517406925559044:
                            if s['W'] - s['Z'] > 0.25030607730150223:
                                return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 86% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 88% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > -0.5351375937461853:
                            if s['g'] - s['W'] > 3.6046794652938843:
                                if Q.sum_pt_top2 > 324.6875:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > 4.695892333984375:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.21312199532985687:
                                    if Q.mass > 99.12939453125:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 87.5628433227539:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 80% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > 0.46152688562870026:
                                if s['g'] - s['W'] > 3.9414790868759155:
                                    if Q.mass_top30 > 76.54303741455078:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 100% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['W'] > 2.409511685371399:
                                    return 't'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 88% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > -0.19989313185214996:
                        if s['W'] - s['t'] > -3.583101987838745:
                            if s['W'] - s['Z'] > -0.1551487147808075:
                                return 'W'   # 65% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -1.3274556398391724:
                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 968.2091064453125:
                                        if s['Z'] - s['t'] > -0.04200655221939087:
                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 83% of the training jets here get this class from the formula
                        else:
                            return 't'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.490475133061409:
                            return 'Z'   # 64% of the training jets here get this class from the formula
                        else:
                            return 't'   # 94% of the training jets here get this class from the formula
    else:
        if s['W'] - s['Z'] > 0.04459928162395954:
            if s['g'] - s['W'] > -0.044942695647478104:
                if s['g'] - s['W'] > 0.11924801021814346:
                    return 'g'   # 93% of the training jets here get this class from the formula
                else:
                    if Q.mass_top20 > 52.81925392150879:
                        return 'g'   # 78% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 75% of the training jets here get this class from the formula
            else:
                if s['W'] - s['t'] > -0.16566985845565796:
                    if s['W'] - s['Z'] > 0.43877893686294556:
                        return 'W'   # 100% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.22182630747556686:
                            if Q.C2 > 0.06966394186019897:
                                if s['W'] - s['Z'] > 0.18646961450576782:
                                    return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 90% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 78% of the training jets here get this class from the formula
                else:
                    if Q.mass > 79.10045623779297:
                        return 't'   # 88% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -3.2887178659439087:
                            return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 68% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.2541256844997406:
                if s['W'] - s['t'] > 1.9069432616233826:
                    if Q.max_dr > 0.4213980734348297:
                        return 'W'   # 53% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.mass_top10 > 61.91860008239746:
                        if Q.n_dr_0p2_0p4 > 2.5:
                            return 'W'   # 79% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 56% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 65% of the training jets here get this class from the formula
            else:
                return 'Z'   # 100% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
