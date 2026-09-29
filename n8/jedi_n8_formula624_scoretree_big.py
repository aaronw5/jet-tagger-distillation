"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 65.47% (the formula: 65.58%); same class as the formula for 95.83% of jets.  826 leaves, depth 20.
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
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_top5_slots=sum(pt[:5]) / tot,
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_1=phi[1],
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
    return (0.2061
        + 90.02 * max(0.0, 0.004372139331 - Q.width)
        - 19.51 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.0007329 * max(0.0, 64.618731689453 - Q.mass)
        - 0.01771 * max(0.0, 21.784077072144 - Q.mass)
        + 13.68 * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 40.67 * max(0.0, 0.013238675334 - Q.girth2)
        + 233.3 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.00322 * max(0.0, Q.sum_pt - 901.59375)
        + 0.007435 * max(0.0, 56.920347213745 - Q.mass)
        - 101.6 * max(0.0, 0.008678044951 - Q.width)
        - 1.876e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        + 0.09634 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 17.14 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 0.349 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        + 0.002373 * max(0.0, 29.644699859619 - Q.mass)
        + 0.03147 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        + 2.946 * max(0.0, 0.087236513197 - Q.girth)
        - 1.815 * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 0.3018 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 13.65 * max(0.0, 0.020459658932 - Q.e2)
        + 1052.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 184.0 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 8.173e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875)
        + 532.4 * max(0.0, 0.000504949057 - Q.lam1)
        + 0.001053 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 11.15 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4)
        + 132.3 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.05235 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 61.36 * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 9.228 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0004078 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 0.03412 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        + 112.5 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 72.63 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.01238 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4)
        - 52.98 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 320.0 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.1552 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 1.312 * max(0.0, 0.293190627853 - Q.LHA)
        - 5.235 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3)
        - 0.03734 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0)
        + 0.1276 * max(0.0, Q.sum_pt - 813.415625)
        - 31.06 * max(0.0, 0.000657050184 - Q.girth2_top5)
        + 305.7 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7)
        + 0.06144 * max(0.0, Q.pt_7 - 34.53125)
        + 6104.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0006699 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        + 3.16 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 31.16 * max(0.0, 0.055577157257 - Q.z_7)
        - 28690.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2)
        + 1933.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2)
        + 4.47 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.04627 * max(0.0, 53.332374954224 - Q.mass)
        - 0.04311 * max(0.0, Q.pt_7 - 53.4375)
        - 4.556 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 15.04 * max(0.0, Q.log_sum_pt - 6.572937922293)
        + 1166.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 52.65 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 0.2724 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        + 1050.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 6.375 * max(0.0, Q.e2 - 0.032346998155)
        - 19.59 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 61.74 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1002.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 57.02 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 11.73 * max(0.0, 0.043044721986 - Q.z_7)
        + 0.02201 * max(0.0, 49.668099212646 - Q.mass)
        - 15.04 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.00249 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 3.784 * max(0.0, Q.LHA - 0.266912960293)
        - 2.744 * max(0.0, 0.250761204958 - Q.max_dr)
        + 1111.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2)
        - 112.4 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 0.01164 * max(0.0, 69.611351776123 - Q.mass)
        + 0.5365 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 85.18 * max(0.0, 0.005954149834 - Q.lam1)
        + 0.01998 * max(0.0, Q.pt_7 - 30.484375)
        - 0.2031 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 18.49 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 5.077 * max(0.0, Q.LHA - 0.111565049159)
        - 1121.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.05157 * max(0.0, 36.229410171509 - Q.mass)
        + 13.04 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.1448 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 18.3 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 106.7 * max(0.0, 0.003377388461 - Q.lam1)
        - 0.03011 * max(0.0, 53.4375 - Q.pt_7)
        + 0.01576 * max(0.0, 43.5 - Q.pt_7)
        + 0.003251 * max(0.0, 788.4484375 - Q.sum_pt)
        + 100.9 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 4790.0 * max(0.0, 0.000172198326 - Q.width)
        - 14370.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 59.28 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 1.342 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 24.73 * max(0.0, Q.width - 0.018827653081)
        + 0.01931 * max(0.0, Q.mass - 36.229410171509)
        + 10.53 * max(0.0, Q.e2 - 0.028531698044)
        + 33.26 * max(0.0, 0.006679471358 - Q.width)
        - 13.61 * max(0.0, 0.04447356835 - Q.e2)
        + 89.69 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.2051 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.007568 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 0.0081 * max(0.0, Q.mass - 69.611351776123)
        - 101.6 * max(0.0, 0.008678044751 - Q.girth2)
        + 3.678 * max(0.0, Q.LHA - 0.325582223496)
        - 0.00575 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        - 9.471 * max(0.0, 0.038466955721 - Q.e2)
        - 73.27 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582)
        - 72.05 * max(0.0, Q.lam1 - 0.016433749775)
        + 118.2 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        - 171.7 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 90.03 * max(0.0, 0.004372139461 - Q.girth2)
        + 45.91 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 128.7 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        - 65.32 * max(0.0, Q.lam1 - 0.012003726523)
        - 135.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 0.03591 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        + 0.3334 * max(0.0, Q.max_dr - 0.145231109113)
        + 2.729 * max(0.0, Q.C2 - 0.094821243733)
        - 79.33 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7)
        + 0.01193 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 0.2846 * max(0.0, Q.LHA - 0.423592510895)
        - 38.54 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1)
        - 57.29 * max(0.0, Q.lam1 - 0.008375572068)
        - 39.22 * max(0.0, Q.lam1 - 0.004183811014)
        - 210.6 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        - 223.7 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7)
        + 0.03383 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125)
        + 309.7 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7)
        + 3.228 * max(0.0, Q.e2 - 0.050284641981)
        + 0.06305 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015)
        - 0.6297 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.03071 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        - 0.8669 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        - 178.7 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        + 61.86 * max(0.0, Q.width - 0.000319370692)
        + 5.401 * max(0.0, Q.e2 - 0.041109715588)
        - 0.0009624 * max(0.0, 763.825 - Q.sum_pt)
        + 7.038 * max(0.0, 0.009530300104 - Q.girth2_top2)
        + 1727.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 0.3232 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        + 77.6 * max(0.0, Q.width - 0.001653836415)
        + 0.0001203 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 1.088 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 1.475 * max(0.0, Q.C2 - 0.067292226106)
        - 27.47 * max(0.0, 0.001130644719 - Q.lam2)
        - 1.444 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 4.512 * max(0.0, Q.girth - 0.101940929517)
        + 311.8 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 9.465 * max(0.0, Q.girth2 - 0.013238675334)
        - 4.67 * max(0.0, Q.e2 - 0.063441075385)
        - 7.842 * max(0.0, Q.e2 - 0.007078157854)
        - 0.005088 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        - 0.03371 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 9.075 * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 0.2123 * max(0.0, Q.max_dr - 0.102758520097)
        + 1.723 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 2.867 * max(0.0, Q.max_dr - 0.197968879342)
        + 0.09997 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        + 1.337 * max(0.0, Q.C2 - 0.014943876117)
        - 0.1057 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        + 1.557 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 6.401 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 3.22 * max(0.0, 0.047915700823 - Q.girth)
        + 239.4 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 83.95 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        + 11.81 * max(0.0, Q.girth2 - 0.007520088344)
        + 1.209 * max(0.0, 0.007639643088 - Q.girth2_top2)
        - 213.5 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 8.112 * max(0.0, Q.width - 0.002635417778)
        - 88.64 * max(0.0, 0.003013300392 - Q.e2_sq)
        - 6.45 * max(0.0, 0.003111083776 - Q.girth2_top2)
        + 2.405 * max(0.0, Q.e2 - 0.020459658932)
        + 32.98 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.02428 * max(0.0, 41.377904891968 - Q.mass)
        + 2.704 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25)
        + 4.248 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944)
        - 714.5 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        + 66.01 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116)
        - 0.5418 * max(0.0, 0.124553743005 - Q.girth)
        + 0.01639 * max(0.0, 80.4 - Q.mass)
        - 1.121 * max(0.0, 0.216055863061 - Q.LHA)
        - 0.1319 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 7.866 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 1.628 * max(0.0, 0.028865759995 - Q.z_6)
        + 1.474 * max(0.0, 0.071488645583 - Q.z_7)
        + 216.8 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 0.02513 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 16.59 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow)
        + 0.7821 * max(0.0, 0.035560912266 - Q.e2)
        - 38.24 * max(0.0, 0.002635417778 - Q.width)
        + 33.16 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.003041 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 5991.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.001025 * max(0.0, 548.196875 - Q.sum_pt_top2)
        + 1026.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.0008095 * max(0.0, Q.sum_pt - 868.509375)
        - 0.1809 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 201.4 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 4.13 * max(0.0, 0.154689112391 - Q.LHA)
        - 6241.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.1065 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 154.3 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        + 237.8 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 57.29 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 197.4 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 9274.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.03941 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 1.669 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5)
        - 36910.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 39.07 * max(0.0, 0.021588001063 - Q.dr_0)
        + 128.6 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516)
        + 0.4184 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 9.001 * max(0.0, 0.05096141791 - Q.z_6)
        + 0.2997 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5)
        - 59.55 * max(0.0, 0.023207568189 - Q.z_7)
        - 3.495 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5)
        + 0.7727 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 73.05 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        - 8.212 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 368.5 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471)
        + 1262.0 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset)
        + 0.00146 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 25.77 * max(0.0, 0.004007841607 - Q.girth2_top2)
        + 0.001164 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2)
        - 40.66 * max(0.0, 0.013238675006 - Q.width)
        - 53.89 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 3465.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 11.73 * max(0.0, 0.050284641981 - Q.e2)
        - 500.3 * max(0.0, Q.lam2 - 0.000537286005)
        + 10.54 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        - 2.635 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 38.08 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 29.6 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        - 1.749 * max(0.0, Q.centroid_offset - 0.049903668404)
        - 0.663 * max(0.0, Q.LHA - 0.312727471086)
        - 1.039 * Q.max_dr
        + 770.7 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 41.75 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        - 4.455 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 14.22 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 5.813 * max(0.0, Q.girth - 0.087236513197)
        - 2.3 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        + 0.01995 * max(0.0, 1.679198372364 - Q.D2)
        + 34.28 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 97.48 * max(0.0, Q.lam2 - 0.003408388935)
        + 0.005298 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 6.922 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386)
        - 2.298 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.0007285 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 239.6 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056)
        + 0.03991 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 0.007178 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 4.465e-05 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        + 23.08 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        - 0.004537 * max(0.0, 5.351076855015 - Q.pt1_dr01)
        - 0.1449 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475)
        + 0.006647 * max(0.0, 31.125 - Q.pt_4)
        - 0.5171 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625)
        - 5.809 * max(0.0, 0.037477688199 - Q.z_4)
        - 8.881 * max(0.0, 0.024419631481 - Q.girth2_top5)
        + 209.4 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 0.05156 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7)
        - 0.1073 * max(0.0, Q.sum_pt - 988.4078125)
        - 0.0008224 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        - 3.783e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625)
        + 0.007077 * max(0.0, 24.421875 - Q.pt_6)
        + 255.1 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        + 153.2 * max(0.0, Q.girth2 - 0.004372139461)
        - 4.961 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 77.6 * max(0.0, Q.girth2 - 0.0016538364)
        - 17.85 * max(0.0, 0.024547699839 - Q.e2)
        + 9.61 * max(0.0, 0.04081947431 - Q.girth)
        + 0.0002396 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        + 121.2 * max(0.0, 0.005590288644 - Q.width)
        - 9.104 * max(0.0, 0.02076709205 - Q.centroid_offset)
        - 54.81 * max(0.0, Q.girth2 - 0.008678044751)
        + 152.4 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.02047 * max(0.0, Q.mass - 80.4)
        - 444.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        + 20.62 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 2.188 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 129.1 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.1702 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 0.004515 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 16.47 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 472.2 * max(0.0, 0.000561123155 - Q.width)
        + 0.08216 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 0.5037 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 27.76 * max(0.0, 0.002464291268 - Q.lam1)
        + 1.194 * max(0.0, 0.15984864831 - Q.max_dr)
        - 0.5825 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 1.632 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        - 0.1132 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 790.8 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257)
        + 11.32 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581)
        - 11.14 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 3.84 * max(0.0, Q.e2 - 0.016554418951)
        + 0.001444 * max(0.0, Q.mass_top5 - 53.607658247923)
        + 0.04278 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012)
        + 3.129 * max(0.0, 0.067292226106 - Q.C2)
        - 18.62 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 70.6 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 26.17 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 33.27 * max(0.0, 0.006679471442 - Q.girth2)
        - 38.43 * max(0.0, 0.005019718802 - Q.width)
        + 0.791 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7)
        + 1893.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 5.418 * max(0.0, 0.196739721581 - Q.LHA)
        + 64.2 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 13.62 * max(0.0, 0.061086014472 - Q.girth)
        - 0.002624 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 0.3571 * max(0.0, 0.177304983139 - Q.max_dr)
        + 1332.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 34.44 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 4.425 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 5804.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 1665.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 5381.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 3092.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        + 0.03155 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 468.2 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 6005.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        + 1.083 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 3098.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 154.1 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 2.085 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 9861.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 11610.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        - 2.794 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 28.58 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 5786.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 6116.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.02143 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        - 0.2461 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 614.5 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 609.1 * max(0.0, 0.000964142894 - Q.girth2)
        + 1.105 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021)
        + 0.1537 * max(0.0, 0.000222950415 - Q.girth2_top5)
        - 1.929 * max(0.0, 0.027029510401 - Q.C2)
        - 1543.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        + 5680.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width)
        - 954.8 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442)
        - 627.7 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 0.9178 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2)
        + 151.9 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 13.81 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2)
        - 19.09 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width)
        - 361.5 * max(0.0, 0.000194798295 - Q.lam2)
        - 150.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787)
        + 77.5 * max(0.0, 0.000319370692 - Q.width)
        - 40.9 * max(0.0, 0.000823693417 - Q.girth2_top3)
        - 773.2 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        + 18.15 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 0.0448 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5)
        - 7.993 * max(0.0, 0.054649224505 - Q.girth)
        + 130.2 * max(0.0, Q.lam2 - 0.001130644719)
        + 1.096 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 301.1 * max(0.0, 0.006096650059 - Q.width)
        - 11.1 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 24.72 * max(0.0, Q.girth2 - 0.018827652745)
        + 1.608 * max(0.0, 0.032346998155 - Q.e2)
        + 134.7 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.06458 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 2.461 * max(0.0, 0.221586732566 - Q.max_dr)
        - 4.627 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5)
        - 538.8 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 0.7465 * max(0.0, Q.C2 - 0.051192347892)
        + 7.855 * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 8872.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 35.37 * max(0.0, 0.007520088344 - Q.girth2)
        + 14.23 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5)
        - 82.47 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5)
        + 0.09535 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.2683 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 230.2 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 328.5 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        - 0.02786 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 21.51 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 37.32 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 4.258 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 7984.0 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 0.01182 * max(0.0, 35.5 - Q.pt_5)
        - 0.786 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 0.004083 * max(0.0, Q.sum_pt - 840.01953125)
        + 0.0471 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3)
        + 4.168 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2)
        - 725.7 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2)
        + 645.6 * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 21530.0 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189)
        + 39.63 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875)
        - 0.1913 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125)
        - 1.772 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 1998.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 7533.0 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421)
        - 8816.0 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2)
        + 236.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        + 0.7765 * max(0.0, 0.033604209498 - Q.girth)
        - 14380.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 9.033 * Q.e2
        - 53.67 * max(0.0, Q.lam2 - 0.000194798295)
        + 2.709 * max(0.0, Q.LHA - 0.303313749495)
        + 24.7 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 6.208 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 75.99 * max(0.0, 0.004183811014 - Q.lam1)
        + 5.715 * max(0.0, Q.centroid_offset - 0.037760993714)
        - 0.00858 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        - 0.2755 * max(0.0, 0.391541349888 - Q.tau21)
        - 25.94 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 26.94 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        - 14.75 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        + 0.06225 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 71.47 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 312.6 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.0005623 * max(0.0, Q.mass - 15.454033088684)
        - 0.01761 * max(0.0, Q.mass - 53.332374954224)
        + 23.44 * max(0.0, 0.0016538364 - Q.girth2)
        + 2.169 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 0.698 * max(0.0, Q.LHA - 0.346713497427)
        - 101.1 * max(0.0, 0.003408388935 - Q.lam2)
        + 0.04662 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 41.11 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        + 0.267 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2)
        - 317.7 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334)
        + 11.46 * max(0.0, Q.girth - 0.076081777364)
        - 0.02102 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        + 10.83 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.0811 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 173.1 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 15.34 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 13.01 * Q.centroid_offset
        - 26.44 * max(0.0, 0.003562611091 - Q.width)
        + 212.3 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2)
        - 32.24 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        + 0.3878 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.0002571 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 0.0001283 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 49.15 * max(0.0, 0.002151567843 - Q.girth2_top3)
        + 36.14 * max(0.0, 0.004839980301 - Q.lam1)
        + 0.6033 * max(0.0, 0.035786485299 - Q.C2)
        + 0.4184 * max(0.0, 0.111761856824 - Q.max_dr)
        - 0.386 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 3.575 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2)
        - 1.529 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.3016 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7)
        - 7.338 * max(0.0, 0.02054281719 - Q.girth)
        + 0.09739 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0)
        - 0.1662 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 2.516 * max(0.0, 0.148408418149 - Q.girth)
        + 9.048 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.197 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 55.94 * max(0.0, 0.016433749775 - Q.lam1)
        - 170.3 * max(0.0, 0.000306123359 - Q.lam2)
        + 0.1353 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        + 0.0019 * max(0.0, 31.90625 - Q.pt_6)
        + 10290.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 3.579e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 0.2185 * max(0.0, Q.z_top5_slots - 0.930764273368)
        + 18.24 * max(0.0, 0.028070914944 - Q.z_7)
        - 0.002569 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.03551 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        - 1.578 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        - 0.001067 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32)
        - 234.6 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 17.53 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 0.3401 * max(0.0, Q.LHA - 0.09323897448)
        + 0.0006031 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 0.301 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        - 35.38 * max(0.0, 0.00752008842 - Q.width)
        + 0.002147 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0001951 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0)
        - 5.421 * max(0.0, 0.006506575659 - Q.lam1)
        + 0.02538 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375)
        + 0.0005779 * max(0.0, 25.578125 - Q.pt_7)
        - 0.001864 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0005904 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        + 0.01214 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 23.78 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 15.16 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 4.404 * max(0.0, Q.lam1 - 0.00543336053)
        - 39.4 * max(0.0, Q.lam1 - 0.007330079875)
        + 1.885 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        - 2228.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 25.75 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        - 0.07371 * max(0.0, 0.080507021025 - Q.max_dr)
        - 43.74 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.04697 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 115.7 * max(0.0, Q.lam1 - 0.002464291268)
        + 58.64 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 42.4 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        - 458.5 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2)
        + 0.1082 * max(0.0, 1.122624260187 - Q.D2)
        + 50.65 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.009978 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 4.902 * max(0.0, 0.303313749495 - Q.LHA)
        + 43.56 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 6.53 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 1.511 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.07496 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578)
        + 0.009946 * max(0.0, 86.4 - Q.mass)
        + 3170.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 318.0 * max(0.0, 0.005834489329 - Q.e2_sq)
        + 13.2 * max(0.0, 0.041109715588 - Q.e2)
        - 29.35 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 23.61 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 0.4777 * max(0.0, 0.135767506063 - Q.tau21)
        - 4.776 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 3.922 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364)
        - 44.9 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 10.4 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364)
        + 0.173 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 1.339 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 16.52 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537)
        + 7127.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 63.15 * max(0.0, 0.012003726523 - Q.lam1)
        + 38.03 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 899.7 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.002651 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 0.6543 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 5.569 * max(0.0, 0.063441075385 - Q.e2)
        - 0.1138 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 0.3871 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.01834 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 623.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744)
        - 11.34 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        - 601.8 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.7565 * max(0.0, 0.083662731125 - Q.planar_flow)
        - 60.02 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 0.002799 * max(0.0, 0.74595130682 - Q.D2)
        + 3.181 * max(0.0, Q.girth - 0.033604209498)
        - 296.8 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi)
    )


def score_q(Q):
    return (0.2844
        + 147.3 * max(0.0, 0.004372139331 - Q.width)
        + 74.89 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.0085 * max(0.0, 64.618731689453 - Q.mass)
        - 0.007371 * max(0.0, 21.784077072144 - Q.mass)
        + 3.801 * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 10.86 * max(0.0, 0.013238675334 - Q.girth2)
        + 85.85 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.00481 * max(0.0, Q.sum_pt - 901.59375)
        - 0.00794 * max(0.0, 56.920347213745 - Q.mass)
        - 36.75 * max(0.0, 0.008678044951 - Q.width)
        + 3.354e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        + 0.2978 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        - 0.2684 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 0.7204 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        - 0.01166 * max(0.0, 29.644699859619 - Q.mass)
        + 0.05207 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        + 2.624 * max(0.0, 0.087236513197 - Q.girth)
        - 2.928 * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 0.04076 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 10.22 * max(0.0, 0.020459658932 - Q.e2)
        + 1612.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 275.1 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        + 5.794e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875)
        - 342.2 * max(0.0, 0.000504949057 - Q.lam1)
        + 0.001285 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 1.909 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4)
        + 65.99 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.1235 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 13.42 * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 1.237 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0001952 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 0.008812 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        + 133.0 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 78.82 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.003464 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4)
        - 20.61 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 323.2 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.09117 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 0.8153 * max(0.0, 0.293190627853 - Q.LHA)
        + 7.333 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3)
        + 0.02549 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0)
        + 0.0666 * max(0.0, Q.sum_pt - 813.415625)
        + 14.95 * max(0.0, 0.000657050184 - Q.girth2_top5)
        + 263.4 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7)
        + 0.01456 * max(0.0, Q.pt_7 - 34.53125)
        + 1163.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0001244 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        + 0.7689 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 2.103 * max(0.0, 0.055577157257 - Q.z_7)
        + 5378.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2)
        + 1319.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2)
        + 44.53 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.02958 * max(0.0, 53.332374954224 - Q.mass)
        - 0.003369 * max(0.0, Q.pt_7 - 53.4375)
        - 3.653 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 8.58 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 199.4 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 49.27 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 0.07059 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.5097 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.1967 * max(0.0, Q.e2 - 0.032346998155)
        - 6.675 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 12.93 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 124.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 12.69 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 5.246 * max(0.0, 0.043044721986 - Q.z_7)
        + 0.008877 * max(0.0, 49.668099212646 - Q.mass)
        - 8.099 * max(0.0, Q.log_sum_pt - 6.638338705138)
        + 0.0001246 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 3.528 * max(0.0, Q.LHA - 0.266912960293)
        - 0.2122 * max(0.0, 0.250761204958 - Q.max_dr)
        - 18.18 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2)
        - 36.25 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 0.05419 * max(0.0, 69.611351776123 - Q.mass)
        - 0.3045 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 27.64 * max(0.0, 0.005954149834 - Q.lam1)
        - 0.006614 * max(0.0, Q.pt_7 - 30.484375)
        + 0.1442 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 9.143 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 2.914 * max(0.0, Q.LHA - 0.111565049159)
        - 383.9 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        + 0.1496 * max(0.0, 36.229410171509 - Q.mass)
        - 5.549 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.05079 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 9.782 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 16.41 * max(0.0, 0.003377388461 - Q.lam1)
        + 0.001541 * max(0.0, 53.4375 - Q.pt_7)
        - 0.001235 * max(0.0, 43.5 - Q.pt_7)
        + 7.067e-05 * max(0.0, 788.4484375 - Q.sum_pt)
        + 57.19 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 3380.0 * max(0.0, 0.000172198326 - Q.width)
        + 2559.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 78.94 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 0.5898 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 19.35 * max(0.0, Q.width - 0.018827653081)
        - 0.1424 * max(0.0, Q.mass - 36.229410171509)
        + 11.58 * max(0.0, Q.e2 - 0.028531698044)
        + 10.58 * max(0.0, 0.006679471358 - Q.width)
        - 8.617 * max(0.0, 0.04447356835 - Q.e2)
        + 21.3 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.3593 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.01683 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 0.06371 * max(0.0, Q.mass - 69.611351776123)
        - 36.74 * max(0.0, 0.008678044751 - Q.girth2)
        + 1.628 * max(0.0, Q.LHA - 0.325582223496)
        - 0.004498 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        - 2.914 * max(0.0, 0.038466955721 - Q.e2)
        - 60.39 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582)
        - 37.21 * max(0.0, Q.lam1 - 0.016433749775)
        + 95.92 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        - 270.8 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 147.3 * max(0.0, 0.004372139461 - Q.girth2)
        + 31.59 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 171.3 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        - 21.5 * max(0.0, Q.lam1 - 0.012003726523)
        - 449.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 0.05809 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        + 0.3102 * max(0.0, Q.max_dr - 0.145231109113)
        + 1.637 * max(0.0, Q.C2 - 0.094821243733)
        - 20.42 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7)
        - 0.0504 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 1.932 * max(0.0, Q.LHA - 0.423592510895)
        - 37.66 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1)
        + 19.44 * max(0.0, Q.lam1 - 0.008375572068)
        + 44.52 * max(0.0, Q.lam1 - 0.004183811014)
        - 327.7 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 24.85 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7)
        + 0.05102 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125)
        + 382.1 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7)
        + 7.495 * max(0.0, Q.e2 - 0.050284641981)
        - 0.2802 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015)
        - 1.713 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.04145 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        + 53.7 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 101.5 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        + 253.4 * max(0.0, Q.width - 0.000319370692)
        + 2.042 * max(0.0, Q.e2 - 0.041109715588)
        - 0.002275 * max(0.0, 763.825 - Q.sum_pt)
        + 20.6 * max(0.0, 0.009530300104 - Q.girth2_top2)
        + 1281.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 2.128 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        + 67.79 * max(0.0, Q.width - 0.001653836415)
        - 0.0005978 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 1.328 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 0.706 * max(0.0, Q.C2 - 0.067292226106)
        + 50.23 * max(0.0, 0.001130644719 - Q.lam2)
        - 8.433 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 3.527 * max(0.0, Q.girth - 0.101940929517)
        + 338.2 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 75.73 * max(0.0, Q.girth2 - 0.013238675334)
        + 2.803 * max(0.0, Q.e2 - 0.063441075385)
        - 22.3 * max(0.0, Q.e2 - 0.007078157854)
        - 0.0151 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        - 0.03452 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 7.761 * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 0.7311 * max(0.0, Q.max_dr - 0.102758520097)
        + 6.116 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 4.085 * max(0.0, Q.max_dr - 0.197968879342)
        + 0.03007 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        + 5.068 * max(0.0, Q.C2 - 0.014943876117)
        - 0.1665 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        + 9.912 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 38.14 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 6.279 * max(0.0, 0.047915700823 - Q.girth)
        + 253.8 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 91.28 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        - 45.58 * max(0.0, Q.girth2 - 0.007520088344)
        + 2.978 * max(0.0, 0.007639643088 - Q.girth2_top2)
        - 380.8 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 19.46 * max(0.0, Q.width - 0.002635417778)
        - 15.2 * max(0.0, 0.003013300392 - Q.e2_sq)
        - 28.79 * max(0.0, 0.003111083776 - Q.girth2_top2)
        + 5.166 * max(0.0, Q.e2 - 0.020459658932)
        - 0.5346 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.02436 * max(0.0, 41.377904891968 - Q.mass)
        + 1.623 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25)
        - 24.21 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944)
        - 818.4 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        + 6.978 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116)
        + 1.485 * max(0.0, 0.124553743005 - Q.girth)
        - 0.1283 * max(0.0, 80.4 - Q.mass)
        + 3.889 * max(0.0, 0.216055863061 - Q.LHA)
        + 0.0157 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        - 0.7601 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 9.529 * max(0.0, 0.028865759995 - Q.z_6)
        + 3.255 * max(0.0, 0.071488645583 - Q.z_7)
        - 5353.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 0.01394 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 7.038 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow)
        + 5.627 * max(0.0, 0.035560912266 - Q.e2)
        + 97.36 * max(0.0, 0.002635417778 - Q.width)
        + 84.77 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.001103 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 986.6 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.0001116 * max(0.0, 548.196875 - Q.sum_pt_top2)
        + 10580.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.002507 * max(0.0, Q.sum_pt - 868.509375)
        + 0.1015 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        - 339.5 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 0.1068 * max(0.0, 0.154689112391 - Q.LHA)
        + 2001.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        + 0.0146 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        + 41.9 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        - 38.06 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 20.03 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        + 6.46 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        + 2603.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.1295 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        - 0.305 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5)
        + 39160.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        - 7.44 * max(0.0, 0.021588001063 - Q.dr_0)
        - 174.3 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516)
        + 0.1587 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 1.876 * max(0.0, 0.05096141791 - Q.z_6)
        + 0.02808 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5)
        + 7.201 * max(0.0, 0.023207568189 - Q.z_7)
        + 1.245 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5)
        - 4.523 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 132.5 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 4.574 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 684.6 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471)
        + 233.1 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset)
        + 0.001355 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 28.3 * max(0.0, 0.004007841607 - Q.girth2_top2)
        - 7.674e-05 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2)
        + 10.86 * max(0.0, 0.013238675006 - Q.width)
        - 52.05 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 2184.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 12.25 * max(0.0, 0.050284641981 - Q.e2)
        - 237.4 * max(0.0, Q.lam2 - 0.000537286005)
        + 2.438 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        - 0.04126 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 18.8 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 20.22 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        + 0.9482 * max(0.0, Q.centroid_offset - 0.049903668404)
        - 1.864 * max(0.0, Q.LHA - 0.312727471086)
        - 2.549 * Q.max_dr
        + 769.9 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 38.86 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 13.09 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 20.44 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 7.346 * max(0.0, Q.girth - 0.087236513197)
        - 2.472 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        + 0.04291 * max(0.0, 1.679198372364 - Q.D2)
        + 33.35 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 83.01 * max(0.0, Q.lam2 - 0.003408388935)
        + 0.02722 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 7.207 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386)
        - 16.43 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.001257 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 289.1 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056)
        + 0.02647 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 0.1128 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        + 0.0026 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        + 32.69 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        - 0.004582 * max(0.0, 5.351076855015 - Q.pt1_dr01)
        - 0.317 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475)
        + 0.004951 * max(0.0, 31.125 - Q.pt_4)
        - 0.04328 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625)
        - 2.521 * max(0.0, 0.037477688199 - Q.z_4)
        - 8.818 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 194.1 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 0.08696 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7)
        - 0.05012 * max(0.0, Q.sum_pt - 988.4078125)
        - 0.0009336 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        - 3.977e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625)
        + 0.01328 * max(0.0, 24.421875 - Q.pt_6)
        + 261.1 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        + 65.93 * max(0.0, Q.girth2 - 0.004372139461)
        - 3.206 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 67.79 * max(0.0, Q.girth2 - 0.0016538364)
        - 5.065 * max(0.0, 0.024547699839 - Q.e2)
        - 6.73 * max(0.0, 0.04081947431 - Q.girth)
        + 0.001537 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        + 155.5 * max(0.0, 0.005590288644 - Q.width)
        - 10.59 * max(0.0, 0.02076709205 - Q.centroid_offset)
        - 123.0 * max(0.0, Q.girth2 - 0.008678044751)
        + 249.2 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 0.1161 * max(0.0, Q.mass - 80.4)
        - 553.6 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        + 64.29 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 0.6461 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 261.2 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.1168 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 5.006e-05 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 13.9 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 432.5 * max(0.0, 0.000561123155 - Q.width)
        + 0.08583 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 1.756 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        - 66.82 * max(0.0, 0.002464291268 - Q.lam1)
        + 1.491 * max(0.0, 0.15984864831 - Q.max_dr)
        - 0.7902 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 2.227 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        - 0.07744 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 127.1 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257)
        - 50.44 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581)
        - 20.86 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 5.523 * max(0.0, Q.e2 - 0.016554418951)
        + 0.0001239 * max(0.0, Q.mass_top5 - 53.607658247923)
        + 0.04119 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012)
        + 6.355 * max(0.0, 0.067292226106 - Q.C2)
        - 23.75 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 53.93 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 24.2 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 10.59 * max(0.0, 0.006679471442 - Q.girth2)
        + 68.53 * max(0.0, 0.005019718802 - Q.width)
        - 0.7649 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7)
        + 5941.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.002756 * max(0.0, 0.196739721581 - Q.LHA)
        + 100.5 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 11.89 * max(0.0, 0.061086014472 - Q.girth)
        + 0.000871 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 0.3666 * max(0.0, 0.177304983139 - Q.max_dr)
        + 1575.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 27.32 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 1.533 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 9217.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 602.5 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 8584.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 4815.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        + 0.0787 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 16.44 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 4490.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        + 0.3279 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 6621.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 113.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 2.382 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 4786.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 2676.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 658.9 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 585.3 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 1295.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 1004.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.2167 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        - 0.509 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 906.9 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 1034.0 * max(0.0, 0.000964142894 - Q.girth2)
        + 1.364 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 278.5 * max(0.0, 0.000222950415 - Q.girth2_top5)
        - 6.179 * max(0.0, 0.027029510401 - Q.C2)
        - 3155.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        + 1799.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width)
        + 1564.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442)
        + 976.9 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 1.266 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2)
        + 258.2 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 76.08 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2)
        - 37.89 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width)
        + 148.4 * max(0.0, 0.000194798295 - Q.lam2)
        - 110.5 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787)
        + 266.0 * max(0.0, 0.000319370692 - Q.width)
        - 43.54 * max(0.0, 0.000823693417 - Q.girth2_top3)
        - 706.7 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        + 24.01 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 0.6205 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5)
        - 12.14 * max(0.0, 0.054649224505 - Q.girth)
        + 55.04 * max(0.0, Q.lam2 - 0.001130644719)
        + 1.401 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 470.7 * max(0.0, 0.006096650059 - Q.width)
        - 12.31 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 19.36 * max(0.0, Q.girth2 - 0.018827652745)
        + 4.225 * max(0.0, 0.032346998155 - Q.e2)
        + 210.3 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.04506 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 0.5825 * max(0.0, 0.221586732566 - Q.max_dr)
        - 7.514 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5)
        - 2083.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        - 1.485 * max(0.0, Q.C2 - 0.051192347892)
        + 9.07 * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 11620.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 40.76 * max(0.0, 0.007520088344 - Q.girth2)
        + 31.87 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5)
        - 182.1 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5)
        + 0.08489 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.4132 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 393.9 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 250.7 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        - 0.04263 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 49.75 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 53.79 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 6.492 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 10630.0 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 0.01737 * max(0.0, 35.5 - Q.pt_5)
        - 0.9207 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 0.002937 * max(0.0, Q.sum_pt - 840.01953125)
        + 0.03188 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3)
        + 18.07 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2)
        - 1576.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2)
        + 396.4 * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 9560.0 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189)
        + 21.88 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875)
        - 1.672 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125)
        - 2.36 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 3513.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 12700.0 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421)
        - 15890.0 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2)
        + 176.7 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        - 8.284 * max(0.0, 0.033604209498 - Q.girth)
        - 18430.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 5.268 * Q.e2
        + 23.44 * max(0.0, Q.lam2 - 0.000194798295)
        + 7.651 * max(0.0, Q.LHA - 0.303313749495)
        + 16.49 * max(0.0, Q.centroid_offset - 0.00231612516)
        + 0.01167 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 100.5 * max(0.0, 0.004183811014 - Q.lam1)
        + 3.821 * max(0.0, Q.centroid_offset - 0.037760993714)
        - 0.0232 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 0.0735 * max(0.0, 0.391541349888 - Q.tau21)
        - 38.63 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 13.87 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        - 73.48 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 1.84 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 36.75 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 475.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.009605 * max(0.0, Q.mass - 15.454033088684)
        - 0.0571 * max(0.0, Q.mass - 53.332374954224)
        + 176.4 * max(0.0, 0.0016538364 - Q.girth2)
        - 58.12 * max(0.0, 0.002270363079 - Q.girth2_top5)
        + 3.125 * max(0.0, Q.LHA - 0.346713497427)
        + 77.22 * max(0.0, 0.003408388935 - Q.lam2)
        + 0.6051 * max(0.0, 0.253403707141 - Q.planar_flow)
        - 68.94 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        + 0.34 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2)
        - 176.7 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334)
        + 2.906 * max(0.0, Q.girth - 0.076081777364)
        - 0.01959 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        + 8.78 * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 0.306 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 318.6 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 0.729 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 9.602 * Q.centroid_offset
        + 80.02 * max(0.0, 0.003562611091 - Q.width)
        - 256.5 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2)
        - 8.09 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        + 0.5294 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.0005716 * max(0.0, 687.4375 - Q.sum_pt_top5)
        + 3.623e-05 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 2.112 * max(0.0, 0.002151567843 - Q.girth2_top3)
        + 36.75 * max(0.0, 0.004839980301 - Q.lam1)
        + 1.052 * max(0.0, 0.035786485299 - Q.C2)
        - 1.376 * max(0.0, 0.111761856824 - Q.max_dr)
        - 0.6119 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 5.826 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2)
        + 8.188 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.2936 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7)
        - 9.829 * max(0.0, 0.02054281719 - Q.girth)
        + 0.05097 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0)
        - 0.4573 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 4.901 * max(0.0, 0.148408418149 - Q.girth)
        + 8.663 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.1497 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 20.39 * max(0.0, 0.016433749775 - Q.lam1)
        - 116.7 * max(0.0, 0.000306123359 - Q.lam2)
        + 0.06078 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 0.001087 * max(0.0, 31.90625 - Q.pt_6)
        + 4388.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 7.124e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 3.226 * max(0.0, Q.z_top5_slots - 0.930764273368)
        + 9.113 * max(0.0, 0.028070914944 - Q.z_7)
        - 0.001032 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.05707 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 1.409 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        + 0.001002 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32)
        - 80.83 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 11.08 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 1.434 * max(0.0, Q.LHA - 0.09323897448)
        - 0.0001278 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 0.2039 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 40.75 * max(0.0, 0.00752008842 - Q.width)
        + 0.0002768 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0001346 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0)
        - 2.452 * max(0.0, 0.006506575659 - Q.lam1)
        + 0.08612 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375)
        + 0.002406 * max(0.0, 25.578125 - Q.pt_7)
        - 0.0002221 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1.434e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        + 0.02919 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 64.38 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 4.061 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 4.018 * max(0.0, Q.lam1 - 0.00543336053)
        - 37.89 * max(0.0, Q.lam1 - 0.007330079875)
        - 1.272 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        - 1814.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 51.3 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 0.2251 * max(0.0, 0.080507021025 - Q.max_dr)
        - 33.13 * max(0.0, Q.lam1 - 0.005954149834)
        + 0.001368 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 122.8 * max(0.0, Q.lam1 - 0.002464291268)
        + 308.9 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 206.5 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        - 269.7 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2)
        - 0.02384 * max(0.0, 1.122624260187 - Q.D2)
        + 49.83 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.00861 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 6.491 * max(0.0, 0.303313749495 - Q.LHA)
        - 10.42 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 12.73 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 1.589 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.06595 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578)
        + 0.009798 * max(0.0, 86.4 - Q.mass)
        + 4000.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 453.7 * max(0.0, 0.005834489329 - Q.e2_sq)
        + 7.249 * max(0.0, 0.041109715588 - Q.e2)
        - 60.99 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 52.33 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 0.4225 * max(0.0, 0.135767506063 - Q.tau21)
        - 5.043 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 9.757 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364)
        - 30.68 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 14.21 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364)
        + 0.6889 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 2.747 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 53.9 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537)
        + 9694.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 35.21 * max(0.0, 0.012003726523 - Q.lam1)
        + 22.05 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 1549.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        - 0.0003254 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 1.422 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 5.588 * max(0.0, 0.063441075385 - Q.e2)
        - 0.1722 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 0.6754 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.01607 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 458.5 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744)
        - 12.48 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 1543.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.2592 * max(0.0, 0.083662731125 - Q.planar_flow)
        - 85.56 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.05121 * max(0.0, 0.74595130682 - Q.D2)
        - 4.251 * max(0.0, Q.girth - 0.033604209498)
        - 163.8 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi)
    )


def score_W(Q):
    return (-0.1646
        - 402.4 * max(0.0, 0.004372139331 - Q.width)
        - 263.9 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.03231 * max(0.0, 64.618731689453 - Q.mass)
        + 0.162 * max(0.0, 21.784077072144 - Q.mass)
        - 25.63 * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 78.19 * max(0.0, 0.013238675334 - Q.girth2)
        - 644.8 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.0156 * max(0.0, Q.sum_pt - 901.59375)
        - 0.03266 * max(0.0, 56.920347213745 - Q.mass)
        + 136.7 * max(0.0, 0.008678044951 - Q.width)
        + 0.000226 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.04182 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 24.56 * max(0.0, 0.00832969537 - Q.girth2_top5)
        - 1.739 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        - 0.02097 * max(0.0, 29.644699859619 - Q.mass)
        - 0.04666 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        + 81.67 * max(0.0, 0.087236513197 - Q.girth)
        + 1.878 * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 0.8828 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 52.08 * max(0.0, 0.020459658932 - Q.e2)
        - 11600.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 419.4 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        + 9.353e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875)
        + 452.4 * max(0.0, 0.000504949057 - Q.lam1)
        + 0.002244 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 27.47 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4)
        + 52.45 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.05416 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        + 192.6 * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 54.15 * max(0.0, 0.076081777364 - Q.girth)
        - 0.0001956 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 0.1225 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        - 326.1 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 78.23 * max(0.0, 0.008375572068 - Q.lam1)
        - 0.02366 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4)
        - 11.42 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 1109.0 * max(0.0, 0.001503553356 - Q.lam1)
        + 3.021 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 1.202 * max(0.0, 0.293190627853 - Q.LHA)
        + 14.16 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3)
        - 0.07553 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0)
        + 0.3061 * max(0.0, Q.sum_pt - 813.415625)
        - 527.6 * max(0.0, 0.000657050184 - Q.girth2_top5)
        + 1165.0 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7)
        - 0.01919 * max(0.0, Q.pt_7 - 34.53125)
        - 11130.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        + 0.000415 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 2.236 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 12.48 * max(0.0, 0.055577157257 - Q.z_7)
        + 102500.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2)
        + 558.6 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2)
        - 36.72 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.1251 * max(0.0, 53.332374954224 - Q.mass)
        + 0.006526 * max(0.0, Q.pt_7 - 53.4375)
        + 4.424 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 41.64 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 1376.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 693.6 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.04634 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        + 1106.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 20.1 * max(0.0, Q.e2 - 0.032346998155)
        + 54.78 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 15.6 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 579.7 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 9.99 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 9.517 * max(0.0, 0.043044721986 - Q.z_7)
        - 0.02384 * max(0.0, 49.668099212646 - Q.mass)
        - 40.05 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 8.932e-05 * max(0.0, Q.sum_pt_top5 - 579.875)
        + 11.62 * max(0.0, Q.LHA - 0.266912960293)
        + 0.6501 * max(0.0, 0.250761204958 - Q.max_dr)
        + 326.9 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2)
        + 184.7 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 0.2317 * max(0.0, 69.611351776123 - Q.mass)
        + 1.065 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        - 28.69 * max(0.0, 0.005954149834 - Q.lam1)
        + 0.02006 * max(0.0, Q.pt_7 - 30.484375)
        - 0.5485 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 40.4 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 6.139 * max(0.0, Q.LHA - 0.111565049159)
        + 776.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.5048 * max(0.0, 36.229410171509 - Q.mass)
        - 4.207 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.01745 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 41.79 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        - 158.3 * max(0.0, 0.003377388461 - Q.lam1)
        - 0.01495 * max(0.0, 53.4375 - Q.pt_7)
        - 0.01058 * max(0.0, 43.5 - Q.pt_7)
        + 0.009119 * max(0.0, 788.4484375 - Q.sum_pt)
        + 255.8 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 772.6 * max(0.0, 0.000172198326 - Q.width)
        - 8965.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        - 372.6 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 0.3525 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 57.51 * max(0.0, Q.width - 0.018827653081)
        + 0.5349 * max(0.0, Q.mass - 36.229410171509)
        - 47.03 * max(0.0, Q.e2 - 0.028531698044)
        + 282.2 * max(0.0, 0.006679471358 - Q.width)
        - 51.04 * max(0.0, 0.04447356835 - Q.e2)
        - 13.46 * max(0.0, 0.007330079875 - Q.lam1)
        - 11.26 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.2267 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 0.2144 * max(0.0, Q.mass - 69.611351776123)
        + 136.6 * max(0.0, 0.008678044751 - Q.girth2)
        - 12.87 * max(0.0, Q.LHA - 0.325582223496)
        - 0.06262 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        - 2.332 * max(0.0, 0.038466955721 - Q.e2)
        + 840.1 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582)
        - 133.9 * max(0.0, Q.lam1 - 0.016433749775)
        + 678.8 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 1960.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 402.4 * max(0.0, 0.004372139461 - Q.girth2)
        - 327.5 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 258.6 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        + 37.98 * max(0.0, Q.lam1 - 0.012003726523)
        - 5162.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 2.494 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 8.231 * max(0.0, Q.max_dr - 0.145231109113)
        + 2.334 * max(0.0, Q.C2 - 0.094821243733)
        - 2241.0 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7)
        - 0.1238 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        - 17.34 * max(0.0, Q.LHA - 0.423592510895)
        + 152.6 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1)
        - 33.43 * max(0.0, Q.lam1 - 0.008375572068)
        - 259.1 * max(0.0, Q.lam1 - 0.004183811014)
        + 245.1 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 213.0 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7)
        - 0.864 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125)
        - 485.0 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7)
        - 47.01 * max(0.0, Q.e2 - 0.050284641981)
        + 0.9045 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015)
        - 1.203 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.03101 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        + 313.4 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 157.2 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        - 758.4 * max(0.0, Q.width - 0.000319370692)
        - 46.5 * max(0.0, Q.e2 - 0.041109715588)
        - 0.007433 * max(0.0, 763.825 - Q.sum_pt)
        - 48.16 * max(0.0, 0.009530300104 - Q.girth2_top2)
        - 4636.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 2.491 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 139.2 * max(0.0, Q.width - 0.001653836415)
        + 0.003726 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 0.6078 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 10.91 * max(0.0, Q.C2 - 0.067292226106)
        - 212.4 * max(0.0, 0.001130644719 - Q.lam2)
        - 59.42 * max(0.0, 0.005011406868 - Q.girth2_top3)
        + 49.17 * max(0.0, Q.girth - 0.101940929517)
        - 856.8 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 216.0 * max(0.0, Q.girth2 - 0.013238675334)
        + 3.821 * max(0.0, Q.e2 - 0.063441075385)
        + 54.99 * max(0.0, Q.e2 - 0.007078157854)
        - 0.005196 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        + 0.09016 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        - 21.74 * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 4.004 * max(0.0, Q.max_dr - 0.102758520097)
        + 21.51 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 10.38 * max(0.0, Q.max_dr - 0.197968879342)
        + 0.04169 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        + 13.64 * max(0.0, Q.C2 - 0.014943876117)
        - 0.2513 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 3.005 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 3.734 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 12.22 * max(0.0, 0.047915700823 - Q.girth)
        - 744.1 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 119.0 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        + 774.5 * max(0.0, Q.girth2 - 0.007520088344)
        + 33.05 * max(0.0, 0.007639643088 - Q.girth2_top2)
        - 79.97 * max(0.0, 0.007182789718 - Q.e2_sq)
        - 162.4 * max(0.0, Q.width - 0.002635417778)
        + 4546.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        - 66.84 * max(0.0, 0.003111083776 - Q.girth2_top2)
        + 59.32 * max(0.0, Q.e2 - 0.020459658932)
        - 101.7 * max(0.0, Q.girth2 - 0.003562611155)
        - 0.02043 * max(0.0, 41.377904891968 - Q.mass)
        - 3.265 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25)
        + 51.44 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944)
        + 2004.0 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        - 147.4 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116)
        - 12.81 * max(0.0, 0.124553743005 - Q.girth)
        + 0.477 * max(0.0, 80.4 - Q.mass)
        - 14.09 * max(0.0, 0.216055863061 - Q.LHA)
        - 0.06305 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 10.46 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 2.026 * max(0.0, 0.028865759995 - Q.z_6)
        - 5.575 * max(0.0, 0.071488645583 - Q.z_7)
        + 5595.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 0.01611 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 4.131 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow)
        - 0.8449 * max(0.0, 0.035560912266 - Q.e2)
        - 456.6 * max(0.0, 0.002635417778 - Q.width)
        - 678.8 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 0.000713 * max(0.0, Q.sum_pt_top5 - 752.1)
        + 460.9 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.0001256 * max(0.0, 548.196875 - Q.sum_pt_top2)
        - 11180.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.01101 * max(0.0, Q.sum_pt - 868.509375)
        + 0.006995 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        - 36.71 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 3.911 * max(0.0, 0.154689112391 - Q.LHA)
        - 1590.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        + 0.02868 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        + 25.11 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        - 80.01 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        - 5.847 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 93.42 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 2834.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        + 0.1221 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.2925 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5)
        + 81340.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        - 7.152 * max(0.0, 0.021588001063 - Q.dr_0)
        + 539.1 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516)
        - 0.5568 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 5.449 * max(0.0, 0.05096141791 - Q.z_6)
        - 0.127 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5)
        + 45.96 * max(0.0, 0.023207568189 - Q.z_7)
        - 0.8612 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5)
        - 10.07 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 449.9 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        - 4.212 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 3781.0 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471)
        - 835.8 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset)
        + 0.005311 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 85.02 * max(0.0, 0.004007841607 - Q.girth2_top2)
        + 0.0002657 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2)
        - 78.27 * max(0.0, 0.013238675006 - Q.width)
        + 206.2 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 4970.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        - 43.04 * max(0.0, 0.050284641981 - Q.e2)
        - 26.16 * max(0.0, Q.lam2 - 0.000537286005)
        - 20.76 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        + 53.98 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 81.99 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 19.75 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        + 8.525 * max(0.0, Q.centroid_offset - 0.049903668404)
        - 36.16 * max(0.0, Q.LHA - 0.312727471086)
        + 12.02 * Q.max_dr
        - 2216.0 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        - 176.1 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 11.19 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 124.5 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 63.86 * max(0.0, Q.girth - 0.087236513197)
        + 4.233 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.09474 * max(0.0, 1.679198372364 - Q.D2)
        - 130.6 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 152.1 * max(0.0, Q.lam2 - 0.003408388935)
        - 0.07687 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 13.25 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386)
        + 93.21 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.0005359 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        + 401.4 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056)
        - 0.07627 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 0.3027 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.002671 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 43.28 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        + 0.01032 * max(0.0, 5.351076855015 - Q.pt1_dr01)
        + 1.954 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475)
        - 0.01738 * max(0.0, 31.125 - Q.pt_4)
        - 0.2771 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625)
        + 10.36 * max(0.0, 0.037477688199 - Q.z_4)
        + 33.03 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 1719.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 0.007155 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7)
        - 0.2854 * max(0.0, Q.sum_pt - 988.4078125)
        + 0.001822 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 3.885e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625)
        - 0.01261 * max(0.0, 24.421875 - Q.pt_6)
        + 190.4 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 131.6 * max(0.0, Q.girth2 - 0.004372139461)
        + 76.35 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 139.2 * max(0.0, Q.girth2 - 0.0016538364)
        + 152.2 * max(0.0, 0.024547699839 - Q.e2)
        - 17.36 * max(0.0, 0.04081947431 - Q.girth)
        + 0.004502 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 1592.0 * max(0.0, 0.005590288644 - Q.width)
        + 16.64 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 422.1 * max(0.0, Q.girth2 - 0.008678044751)
        - 1461.0 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.4322 * max(0.0, Q.mass - 80.4)
        - 1567.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 391.8 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        - 43.07 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 18.68 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        - 0.4991 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        - 0.007768 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        + 48.49 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 1393.0 * max(0.0, 0.000561123155 - Q.width)
        - 0.3327 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 18.51 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 722.4 * max(0.0, 0.002464291268 - Q.lam1)
        + 0.2258 * max(0.0, 0.15984864831 - Q.max_dr)
        + 0.2798 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 29.09 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        + 0.217 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 317.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257)
        + 437.4 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581)
        + 253.4 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 30.76 * max(0.0, Q.e2 - 0.016554418951)
        + 0.00428 * max(0.0, Q.mass_top5 - 53.607658247923)
        - 0.01385 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012)
        - 20.94 * max(0.0, 0.067292226106 - Q.C2)
        + 129.7 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 186.7 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 93.62 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 282.1 * max(0.0, 0.006679471442 - Q.girth2)
        - 958.9 * max(0.0, 0.005019718802 - Q.width)
        + 1.725 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7)
        - 8721.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 11.32 * max(0.0, 0.196739721581 - Q.LHA)
        - 647.4 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 126.0 * max(0.0, 0.061086014472 - Q.girth)
        + 0.001444 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 0.8148 * max(0.0, 0.177304983139 - Q.max_dr)
        - 2284.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 278.5 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 13.55 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 34020.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 35870.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 24770.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 8493.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 0.08821 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        - 553.6 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 7491.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        - 6.09 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 6325.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 1433.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 7.653 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 50640.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 34640.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        - 3735.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 2044.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 2252.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 5057.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.1787 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 0.7964 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 3569.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        - 1133.0 * max(0.0, 0.000964142894 - Q.girth2)
        - 5.633 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 197.5 * max(0.0, 0.000222950415 - Q.girth2_top5)
        + 41.14 * max(0.0, 0.027029510401 - Q.C2)
        + 49260.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 36660.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width)
        - 30160.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442)
        - 29060.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        + 0.9073 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2)
        - 40.05 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 6.029 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2)
        + 93.86 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width)
        - 628.6 * max(0.0, 0.000194798295 - Q.lam2)
        + 567.2 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787)
        - 1006.0 * max(0.0, 0.000319370692 - Q.width)
        - 25.98 * max(0.0, 0.000823693417 - Q.girth2_top3)
        + 2791.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 89.98 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 0.5956 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5)
        + 7.379 * max(0.0, 0.054649224505 - Q.girth)
        + 191.7 * max(0.0, Q.lam2 - 0.001130644719)
        - 2.07 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 21.32 * max(0.0, 0.006096650059 - Q.width)
        + 33.58 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 57.54 * max(0.0, Q.girth2 - 0.018827652745)
        - 21.98 * max(0.0, 0.032346998155 - Q.e2)
        - 50.63 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        - 0.0104 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 6.376 * max(0.0, 0.221586732566 - Q.max_dr)
        + 1.919 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5)
        - 2825.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        - 5.227 * max(0.0, Q.C2 - 0.051192347892)
        + 10.48 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 16830.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 486.9 * max(0.0, 0.007520088344 - Q.girth2)
        - 5.947 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5)
        + 27.66 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5)
        - 0.02651 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 0.1591 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        + 101.7 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 248.8 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        + 0.01286 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 42.9 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 214.9 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 8.917 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 533.8 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 0.002974 * max(0.0, 35.5 - Q.pt_5)
        - 0.9194 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 0.01081 * max(0.0, Q.sum_pt - 840.01953125)
        - 0.05755 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3)
        - 5.229 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2)
        + 521.5 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2)
        - 330.7 * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 777.3 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189)
        + 2.502 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875)
        - 2.355 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125)
        + 2.375 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 2844.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 18080.0 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421)
        + 15000.0 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 4435.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        - 1.859 * max(0.0, 0.033604209498 - Q.girth)
        - 10700.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 5.203 * Q.e2
        - 315.6 * max(0.0, Q.lam2 - 0.000194798295)
        + 29.51 * max(0.0, Q.LHA - 0.303313749495)
        - 84.85 * max(0.0, Q.centroid_offset - 0.00231612516)
        + 10.47 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        - 109.8 * max(0.0, 0.004183811014 - Q.lam1)
        - 30.36 * max(0.0, Q.centroid_offset - 0.037760993714)
        - 0.003887 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 1.352 * max(0.0, 0.391541349888 - Q.tau21)
        + 79.22 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 10.54 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        - 13.46 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 2.426 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 164.0 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 206.9 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.06001 * max(0.0, Q.mass - 15.454033088684)
        + 0.1633 * max(0.0, Q.mass - 53.332374954224)
        - 536.3 * max(0.0, 0.0016538364 - Q.girth2)
        + 7.877 * max(0.0, 0.002270363079 - Q.girth2_top5)
        + 5.304 * max(0.0, Q.LHA - 0.346713497427)
        - 156.2 * max(0.0, 0.003408388935 - Q.lam2)
        + 1.887 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 314.3 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        + 0.2078 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2)
        + 54.17 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334)
        - 80.33 * max(0.0, Q.girth - 0.076081777364)
        - 0.03747 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        - 27.77 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.5102 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        - 701.8 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 4.716 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 33.38 * Q.centroid_offset
        - 382.0 * max(0.0, 0.003562611091 - Q.width)
        - 2192.0 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2)
        + 64.48 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        + 4.845 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.0002507 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 0.0003846 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 153.7 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 63.66 * max(0.0, 0.004839980301 - Q.lam1)
        - 6.412 * max(0.0, 0.035786485299 - Q.C2)
        - 11.8 * max(0.0, 0.111761856824 - Q.max_dr)
        - 7.477 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 1.654 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2)
        - 36.94 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.1414 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7)
        + 16.37 * max(0.0, 0.02054281719 - Q.girth)
        - 3.753 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0)
        + 0.3492 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        + 41.16 * max(0.0, 0.148408418149 - Q.girth)
        - 22.02 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.09972 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        - 35.98 * max(0.0, 0.016433749775 - Q.lam1)
        + 1258.0 * max(0.0, 0.000306123359 - Q.lam2)
        - 0.1464 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 0.001215 * max(0.0, 31.90625 - Q.pt_6)
        - 10600.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 2.218e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 7.29 * max(0.0, Q.z_top5_slots - 0.930764273368)
        - 28.96 * max(0.0, 0.028070914944 - Q.z_7)
        + 0.002484 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.05134 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 6.517 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        - 0.007241 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32)
        + 2807.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 42.07 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 3.55 * max(0.0, Q.LHA - 0.09323897448)
        - 0.0008684 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        - 5.325 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 487.0 * max(0.0, 0.00752008842 - Q.width)
        - 0.002719 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.0001831 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0)
        - 373.2 * max(0.0, 0.006506575659 - Q.lam1)
        - 0.6483 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375)
        - 0.001047 * max(0.0, 25.578125 - Q.pt_7)
        + 0.002072 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.0006207 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        - 1.284 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 21.19 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 9.181 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        - 46.69 * max(0.0, Q.lam1 - 0.00543336053)
        - 124.2 * max(0.0, Q.lam1 - 0.007330079875)
        + 12.78 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        - 998.8 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 50.19 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 7.543 * max(0.0, 0.080507021025 - Q.max_dr)
        - 133.0 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.1295 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 636.9 * max(0.0, Q.lam1 - 0.002464291268)
        - 1199.0 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 1045.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 807.0 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2)
        + 0.5896 * max(0.0, 1.122624260187 - Q.D2)
        - 42.06 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.03202 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 15.19 * max(0.0, 0.303313749495 - Q.LHA)
        + 464.8 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 206.2 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 36.08 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 1.209 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578)
        - 0.04848 * max(0.0, 86.4 - Q.mass)
        - 4693.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        + 1323.0 * max(0.0, 0.005834489329 - Q.e2_sq)
        - 45.67 * max(0.0, 0.041109715588 - Q.e2)
        + 531.8 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        - 485.7 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 2.364 * max(0.0, 0.135767506063 - Q.tau21)
        - 35.43 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 251.1 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364)
        - 18.04 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 153.0 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364)
        + 4.05 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 16.26 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 160.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537)
        - 81820.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 162.9 * max(0.0, 0.012003726523 - Q.lam1)
        + 709.6 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 12370.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        - 0.006499 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        + 9.012 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 1.456 * max(0.0, 0.063441075385 - Q.e2)
        + 1.528 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        - 3.878 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.08611 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 3453.0 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744)
        + 52.25 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 4905.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.3376 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 413.0 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 0.5622 * max(0.0, 0.74595130682 - Q.D2)
        - 25.28 * max(0.0, Q.girth - 0.033604209498)
        - 1060.0 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi)
    )


def score_Z(Q):
    return (-0.801
        - 142.7 * max(0.0, 0.004372139331 - Q.width)
        - 129.3 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.0007892 * max(0.0, 64.618731689453 - Q.mass)
        + 0.04006 * max(0.0, 21.784077072144 - Q.mass)
        - 7.107 * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 309.3 * max(0.0, 0.013238675334 - Q.girth2)
        + 596.2 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.01068 * max(0.0, Q.sum_pt - 901.59375)
        + 0.01006 * max(0.0, 56.920347213745 - Q.mass)
        + 372.9 * max(0.0, 0.008678044951 - Q.width)
        - 5.677e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.4366 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 67.22 * max(0.0, 0.00832969537 - Q.girth2_top5)
        - 1.999 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        + 0.06683 * max(0.0, 29.644699859619 - Q.mass)
        - 0.0245 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        - 12.97 * max(0.0, 0.087236513197 - Q.girth)
        + 31.06 * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 0.8493 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 16.16 * max(0.0, 0.020459658932 - Q.e2)
        - 14810.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 50.47 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 2.161e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875)
        - 199.5 * max(0.0, 0.000504949057 - Q.lam1)
        - 0.001555 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 1.718 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4)
        + 56.07 * max(0.0, 0.00543336053 - Q.lam1)
        + 0.06061 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 26.28 * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 11.27 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0004831 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 0.03155 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        - 420.5 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 159.6 * max(0.0, 0.008375572068 - Q.lam1)
        - 0.001178 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4)
        + 11.32 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 434.3 * max(0.0, 0.001503553356 - Q.lam1)
        - 0.5924 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 4.56 * max(0.0, 0.293190627853 - Q.LHA)
        - 0.8022 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3)
        - 0.225 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0)
        + 0.318 * max(0.0, Q.sum_pt - 813.415625)
        - 148.1 * max(0.0, 0.000657050184 - Q.girth2_top5)
        + 216.1 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7)
        + 0.02398 * max(0.0, Q.pt_7 - 34.53125)
        - 5368.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 7e-05 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 0.206 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 24.61 * max(0.0, 0.055577157257 - Q.z_7)
        + 16970.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2)
        + 3681.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2)
        - 63.24 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.1382 * max(0.0, 53.332374954224 - Q.mass)
        - 0.01823 * max(0.0, Q.pt_7 - 53.4375)
        - 2.5 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 47.4 * max(0.0, Q.log_sum_pt - 6.572937922293)
        + 1418.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 325.2 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.1175 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        + 2245.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 6.231 * max(0.0, Q.e2 - 0.032346998155)
        - 21.91 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 4.342 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1593.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 4.854 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 15.55 * max(0.0, 0.043044721986 - Q.z_7)
        + 0.01364 * max(0.0, 49.668099212646 - Q.mass)
        - 39.71 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.000475 * max(0.0, Q.sum_pt_top5 - 579.875)
        + 10.55 * max(0.0, Q.LHA - 0.266912960293)
        - 1.0 * max(0.0, 0.250761204958 - Q.max_dr)
        + 537.9 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2)
        - 21.37 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 0.1752 * max(0.0, 69.611351776123 - Q.mass)
        + 0.4267 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 73.85 * max(0.0, 0.005954149834 - Q.lam1)
        + 0.0121 * max(0.0, Q.pt_7 - 30.484375)
        - 0.2448 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 41.77 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 1.445 * max(0.0, Q.LHA - 0.111565049159)
        - 1422.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.4635 * max(0.0, 36.229410171509 - Q.mass)
        + 2.186 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0328 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 49.44 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 277.6 * max(0.0, 0.003377388461 - Q.lam1)
        - 0.01492 * max(0.0, 53.4375 - Q.pt_7)
        - 0.01844 * max(0.0, 43.5 - Q.pt_7)
        + 0.007152 * max(0.0, 788.4484375 - Q.sum_pt)
        + 274.0 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 495.0 * max(0.0, 0.000172198326 - Q.width)
        - 1857.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        - 465.0 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 13.48 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 20.52 * max(0.0, Q.width - 0.018827653081)
        + 0.5055 * max(0.0, Q.mass - 36.229410171509)
        - 75.43 * max(0.0, Q.e2 - 0.028531698044)
        - 360.6 * max(0.0, 0.006679471358 - Q.width)
        + 122.5 * max(0.0, 0.04447356835 - Q.e2)
        + 23.74 * max(0.0, 0.007330079875 - Q.lam1)
        - 4.743 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.112 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 0.1925 * max(0.0, Q.mass - 69.611351776123)
        + 372.9 * max(0.0, 0.008678044751 - Q.girth2)
        - 23.41 * max(0.0, Q.LHA - 0.325582223496)
        - 0.07189 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 102.9 * max(0.0, 0.038466955721 - Q.e2)
        + 482.2 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582)
        - 77.65 * max(0.0, Q.lam1 - 0.016433749775)
        + 472.1 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 2417.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 142.7 * max(0.0, 0.004372139461 - Q.girth2)
        - 416.1 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 1083.0 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        - 2.602 * max(0.0, Q.lam1 - 0.012003726523)
        - 3397.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 0.2059 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 3.262 * max(0.0, Q.max_dr - 0.145231109113)
        + 26.18 * max(0.0, Q.C2 - 0.094821243733)
        - 1388.0 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7)
        - 0.4651 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        - 16.14 * max(0.0, Q.LHA - 0.423592510895)
        + 129.8 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1)
        - 224.2 * max(0.0, Q.lam1 - 0.008375572068)
        + 164.4 * max(0.0, Q.lam1 - 0.004183811014)
        + 367.9 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 1375.0 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7)
        - 0.944 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125)
        + 628.0 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7)
        - 22.36 * max(0.0, Q.e2 - 0.050284641981)
        + 0.7704 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015)
        - 2.81 * max(0.0, 0.23799610585 - Q.tau21)
        - 0.04441 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        + 584.9 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 1101.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        - 415.3 * max(0.0, Q.width - 0.000319370692)
        - 47.49 * max(0.0, Q.e2 - 0.041109715588)
        - 0.01518 * max(0.0, 763.825 - Q.sum_pt)
        - 75.87 * max(0.0, 0.009530300104 - Q.girth2_top2)
        - 745.9 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 3.165 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 126.0 * max(0.0, Q.width - 0.001653836415)
        + 0.003868 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 0.8973 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 7.582 * max(0.0, Q.C2 - 0.067292226106)
        - 132.4 * max(0.0, 0.001130644719 - Q.lam2)
        + 49.16 * max(0.0, 0.005011406868 - Q.girth2_top3)
        + 136.2 * max(0.0, Q.girth - 0.101940929517)
        - 1082.0 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 388.7 * max(0.0, Q.girth2 - 0.013238675334)
        + 34.04 * max(0.0, Q.e2 - 0.063441075385)
        - 10.03 * max(0.0, Q.e2 - 0.007078157854)
        + 0.008294 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        + 0.03403 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 4.941 * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 7.342 * max(0.0, Q.max_dr - 0.102758520097)
        - 9.784 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 8.815 * max(0.0, Q.max_dr - 0.197968879342)
        - 0.1092 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        - 9.268 * max(0.0, Q.C2 - 0.014943876117)
        + 0.09436 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 7.735 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 53.39 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 4.288 * max(0.0, 0.047915700823 - Q.girth)
        - 896.3 * max(0.0, 0.023780909279 - Q.e2_sq)
        - 483.1 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        - 122.4 * max(0.0, Q.girth2 - 0.007520088344)
        + 65.58 * max(0.0, 0.007639643088 - Q.girth2_top2)
        + 434.4 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 13.69 * max(0.0, Q.width - 0.002635417778)
        + 1404.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        + 28.55 * max(0.0, 0.003111083776 - Q.girth2_top2)
        + 76.58 * max(0.0, Q.e2 - 0.020459658932)
        + 3.126 * max(0.0, Q.girth2 - 0.003562611155)
        - 0.01914 * max(0.0, 41.377904891968 - Q.mass)
        - 3.037 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25)
        - 2.778 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944)
        + 2602.0 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        - 3.755 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116)
        + 0.5992 * max(0.0, 0.124553743005 - Q.girth)
        + 0.4012 * max(0.0, 80.4 - Q.mass)
        - 3.617 * max(0.0, 0.216055863061 - Q.LHA)
        - 0.04638 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 1.482 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.1673 * max(0.0, 0.028865759995 - Q.z_6)
        - 2.012 * max(0.0, 0.071488645583 - Q.z_7)
        + 8666.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 0.006593 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 0.5279 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow)
        - 49.8 * max(0.0, 0.035560912266 - Q.e2)
        - 57.08 * max(0.0, 0.002635417778 - Q.width)
        - 700.4 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 0.0008657 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 2548.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 3.318e-05 * max(0.0, 548.196875 - Q.sum_pt_top2)
        - 27840.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.01067 * max(0.0, Q.sum_pt - 868.509375)
        + 0.01121 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 71.94 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 7.052 * max(0.0, 0.154689112391 - Q.LHA)
        - 1361.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.01608 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 58.84 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        - 21.65 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 370.4 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 7.342 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 2236.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        + 0.03773 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        - 0.1456 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5)
        + 21170.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 19.66 * max(0.0, 0.021588001063 - Q.dr_0)
        - 14.73 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516)
        - 0.1793 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 2.318 * max(0.0, 0.05096141791 - Q.z_6)
        + 0.0334 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5)
        + 23.29 * max(0.0, 0.023207568189 - Q.z_7)
        + 0.03596 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5)
        - 11.68 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 523.0 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 1.426 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 1281.0 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471)
        - 1095.0 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset)
        - 0.003482 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 56.23 * max(0.0, 0.004007841607 - Q.girth2_top2)
        - 0.0002199 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2)
        + 309.2 * max(0.0, 0.013238675006 - Q.width)
        + 251.1 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 6647.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        - 102.9 * max(0.0, 0.050284641981 - Q.e2)
        - 158.7 * max(0.0, Q.lam2 - 0.000537286005)
        - 22.61 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        + 11.12 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 85.03 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 47.38 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        - 48.4 * max(0.0, Q.centroid_offset - 0.049903668404)
        - 13.08 * max(0.0, Q.LHA - 0.312727471086)
        + 0.4857 * Q.max_dr
        - 2572.0 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        - 162.1 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        - 2.253 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 131.6 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 51.8 * max(0.0, Q.girth - 0.087236513197)
        + 4.95 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        + 0.1149 * max(0.0, 1.679198372364 - Q.D2)
        - 32.9 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 62.39 * max(0.0, Q.lam2 - 0.003408388935)
        - 0.1107 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        - 53.68 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386)
        + 90.79 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.001326 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        + 750.2 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056)
        - 0.08694 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 0.4957 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.02185 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 90.89 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        + 0.01613 * max(0.0, 5.351076855015 - Q.pt1_dr01)
        + 1.975 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475)
        - 0.02416 * max(0.0, 31.125 - Q.pt_4)
        + 1.191 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625)
        + 13.94 * max(0.0, 0.037477688199 - Q.z_4)
        + 10.45 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 337.6 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 0.2047 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7)
        - 0.2981 * max(0.0, Q.sum_pt - 988.4078125)
        + 0.001059 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 5.396e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625)
        - 0.02774 * max(0.0, 24.421875 - Q.pt_6)
        - 1564.0 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 75.14 * max(0.0, Q.girth2 - 0.004372139461)
        + 25.29 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 126.0 * max(0.0, Q.girth2 - 0.0016538364)
        + 191.2 * max(0.0, 0.024547699839 - Q.e2)
        + 0.4817 * max(0.0, 0.04081947431 - Q.girth)
        + 0.005346 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 169.3 * max(0.0, 0.005590288644 - Q.width)
        + 4.5 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 416.2 * max(0.0, Q.girth2 - 0.008678044751)
        - 1618.0 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.3888 * max(0.0, Q.mass - 80.4)
        + 4826.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 517.1 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 12.52 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 287.0 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        - 0.8064 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        - 0.01457 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        + 71.82 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 707.5 * max(0.0, 0.000561123155 - Q.width)
        - 0.6884 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 37.91 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 1.889 * max(0.0, 0.002464291268 - Q.lam1)
        - 24.15 * max(0.0, 0.15984864831 - Q.max_dr)
        + 45.36 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 21.48 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        + 0.1325 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 2075.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257)
        - 305.6 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581)
        - 122.1 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.6413 * max(0.0, Q.e2 - 0.016554418951)
        + 0.01652 * max(0.0, Q.mass_top5 - 53.607658247923)
        - 0.0726 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012)
        - 28.5 * max(0.0, 0.067292226106 - Q.C2)
        + 162.5 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 867.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 163.5 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 360.7 * max(0.0, 0.006679471442 - Q.girth2)
        - 594.3 * max(0.0, 0.005019718802 - Q.width)
        + 4.828 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7)
        + 192.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 1.49 * max(0.0, 0.196739721581 - Q.LHA)
        + 55.39 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 56.04 * max(0.0, 0.061086014472 - Q.girth)
        - 0.0001095 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 6.718 * max(0.0, 0.177304983139 - Q.max_dr)
        + 6233.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 561.8 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 2.98 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 170.6 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 10870.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 18610.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        - 11040.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 0.03387 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        - 366.9 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 4926.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        - 2.712 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 4239.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 1864.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 0.5841 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 18080.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 8973.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 606.3 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 1055.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 19670.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 2637.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.346 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 0.169 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 715.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        - 123.1 * max(0.0, 0.000964142894 - Q.girth2)
        - 0.8014 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 189.2 * max(0.0, 0.000222950415 - Q.girth2_top5)
        + 16.69 * max(0.0, 0.027029510401 - Q.C2)
        + 6361.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 3214.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width)
        + 169.1 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442)
        - 3543.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 0.5255 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2)
        - 377.2 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 80.33 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2)
        + 38.26 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width)
        + 727.9 * max(0.0, 0.000194798295 - Q.lam2)
        + 19.57 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787)
        - 412.2 * max(0.0, 0.000319370692 - Q.width)
        - 23.35 * max(0.0, 0.000823693417 - Q.girth2_top3)
        + 2318.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        + 92.79 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 0.4687 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5)
        - 1.166 * max(0.0, 0.054649224505 - Q.girth)
        + 86.06 * max(0.0, Q.lam2 - 0.001130644719)
        - 0.09971 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 934.1 * max(0.0, 0.006096650059 - Q.width)
        + 1.091 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 20.51 * max(0.0, Q.girth2 - 0.018827652745)
        - 47.61 * max(0.0, 0.032346998155 - Q.e2)
        + 1.257 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        - 0.0614 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 5.328 * max(0.0, 0.221586732566 - Q.max_dr)
        - 1.451 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5)
        + 273.7 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 6.234 * max(0.0, Q.C2 - 0.051192347892)
        - 1.719 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 3760.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 186.4 * max(0.0, 0.007520088344 - Q.girth2)
        - 9.638 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5)
        + 69.36 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5)
        - 0.01527 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.1816 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 105.1 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 347.0 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        + 0.0484 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 42.35 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 177.5 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 11.2 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        + 3888.0 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 0.00424 * max(0.0, 35.5 - Q.pt_5)
        - 1.342 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 0.01115 * max(0.0, Q.sum_pt - 840.01953125)
        - 0.02472 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3)
        - 13.25 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2)
        + 1091.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2)
        - 85.05 * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 15590.0 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189)
        + 14.39 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875)
        - 0.8226 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125)
        + 3.341 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 364.8 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 86.11 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421)
        + 2301.0 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 1615.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        + 24.33 * max(0.0, 0.033604209498 - Q.girth)
        - 4619.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 60.16 * Q.e2
        - 121.6 * max(0.0, Q.lam2 - 0.000194798295)
        + 24.09 * max(0.0, Q.LHA - 0.303313749495)
        - 21.23 * max(0.0, Q.centroid_offset - 0.00231612516)
        + 14.49 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 206.4 * max(0.0, 0.004183811014 - Q.lam1)
        + 1.277 * max(0.0, Q.centroid_offset - 0.037760993714)
        + 0.003881 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 0.8285 * max(0.0, 0.391541349888 - Q.tau21)
        + 13.35 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 139.1 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        - 53.52 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        + 0.8789 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 174.2 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 837.2 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.00513 * max(0.0, Q.mass - 15.454033088684)
        + 0.1566 * max(0.0, Q.mass - 53.332374954224)
        - 154.1 * max(0.0, 0.0016538364 - Q.girth2)
        - 3.258 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 30.51 * max(0.0, Q.LHA - 0.346713497427)
        - 15.76 * max(0.0, 0.003408388935 - Q.lam2)
        - 1.145 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 122.6 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        - 0.254 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2)
        + 202.3 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334)
        - 1.248 * max(0.0, Q.girth - 0.076081777364)
        + 0.006921 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        - 25.98 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.7983 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        - 231.6 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 19.54 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 14.37 * Q.centroid_offset
        - 67.85 * max(0.0, 0.003562611091 - Q.width)
        + 644.9 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2)
        + 170.3 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        - 1.048 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.0007471 * max(0.0, 687.4375 - Q.sum_pt_top5)
        + 8.739e-05 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 40.94 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 85.98 * max(0.0, 0.004839980301 - Q.lam1)
        + 6.856 * max(0.0, 0.035786485299 - Q.C2)
        - 3.87 * max(0.0, 0.111761856824 - Q.max_dr)
        - 0.6729 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 33.52 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2)
        - 36.49 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.06601 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7)
        + 0.1757 * max(0.0, 0.02054281719 - Q.girth)
        + 0.7318 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0)
        + 3.43 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        + 39.32 * max(0.0, 0.148408418149 - Q.girth)
        - 8.2 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.07158 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        - 52.81 * max(0.0, 0.016433749775 - Q.lam1)
        - 48.86 * max(0.0, 0.000306123359 - Q.lam2)
        - 0.0516 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 0.0001197 * max(0.0, 31.90625 - Q.pt_6)
        - 4520.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 1.551e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 7.842 * max(0.0, Q.z_top5_slots - 0.930764273368)
        - 24.97 * max(0.0, 0.028070914944 - Q.z_7)
        + 0.002885 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.1102 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 7.03 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        - 0.005378 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32)
        - 1777.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 24.56 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 1.25 * max(0.0, Q.LHA - 0.09323897448)
        - 0.001141 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        - 2.007 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        - 186.3 * max(0.0, 0.00752008842 - Q.width)
        + 0.001806 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1.276e-05 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0)
        - 69.56 * max(0.0, 0.006506575659 - Q.lam1)
        + 1.137 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375)
        + 0.001401 * max(0.0, 25.578125 - Q.pt_7)
        - 0.001655 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.001307 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        + 0.728 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 269.2 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 30.36 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 16.55 * max(0.0, Q.lam1 - 0.00543336053)
        - 10.72 * max(0.0, Q.lam1 - 0.007330079875)
        - 9.562 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        + 3593.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 241.0 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 4.027 * max(0.0, 0.080507021025 - Q.max_dr)
        + 32.04 * max(0.0, Q.lam1 - 0.005954149834)
        + 0.003541 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 42.05 * max(0.0, Q.lam1 - 0.002464291268)
        - 527.0 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 514.9 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 258.5 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2)
        - 0.3054 * max(0.0, 1.122624260187 - Q.D2)
        - 354.4 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        - 0.04492 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 15.63 * max(0.0, 0.303313749495 - Q.LHA)
        - 131.0 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 175.1 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 38.42 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.2652 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578)
        - 0.002764 * max(0.0, 86.4 - Q.mass)
        - 15390.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        + 1450.0 * max(0.0, 0.005834489329 - Q.e2_sq)
        - 96.77 * max(0.0, 0.041109715588 - Q.e2)
        - 436.7 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 304.7 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        + 1.907 * max(0.0, 0.135767506063 - Q.tau21)
        - 18.24 * max(0.0, Q.centroid_offset - 0.012587644117)
        + 458.7 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364)
        + 264.5 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        - 309.8 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364)
        + 2.564 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 3.088 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 100.2 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537)
        - 63330.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 87.78 * max(0.0, 0.012003726523 - Q.lam1)
        + 209.4 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 13690.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.002529 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 1.858 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        - 24.29 * max(0.0, 0.063441075385 - Q.e2)
        + 0.5873 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        - 0.1789 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.1805 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 31.47 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744)
        + 0.6156 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 1152.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 3.434 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 459.3 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.002454 * max(0.0, 0.74595130682 - Q.D2)
        + 5.38 * max(0.0, Q.girth - 0.033604209498)
        - 361.9 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi)
    )


def score_t(Q):
    return (0.14
        + 20.19 * max(0.0, 0.004372139331 - Q.width)
        - 12.42 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.00265 * max(0.0, 64.618731689453 - Q.mass)
        - 0.01018 * max(0.0, 21.784077072144 - Q.mass)
        + 1.224 * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 30.48 * max(0.0, 0.013238675334 - Q.girth2)
        - 204.5 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        + 0.07465 * max(0.0, Q.sum_pt - 901.59375)
        + 0.006019 * max(0.0, 56.920347213745 - Q.mass)
        + 70.58 * max(0.0, 0.008678044951 - Q.width)
        - 0.0004319 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.1421 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        - 6.222 * max(0.0, 0.00832969537 - Q.girth2_top5)
        - 0.3297 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        - 0.004172 * max(0.0, 29.644699859619 - Q.mass)
        + 0.0426 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        + 9.042 * max(0.0, 0.087236513197 - Q.girth)
        - 5.935 * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 0.1006 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 17.77 * max(0.0, 0.020459658932 - Q.e2)
        + 1253.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 608.3 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.000248 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875)
        + 400.1 * max(0.0, 0.000504949057 - Q.lam1)
        + 0.008793 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 3.628 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4)
        + 138.2 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.2427 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 7.013 * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 13.97 * max(0.0, 0.076081777364 - Q.girth)
        + 5.879e-06 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        + 0.01247 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        - 146.8 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 37.08 * max(0.0, 0.008375572068 - Q.lam1)
        - 0.01049 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4)
        - 115.4 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 46.0 * max(0.0, 0.001503553356 - Q.lam1)
        - 1.446 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50)
        - 0.9265 * max(0.0, 0.293190627853 - Q.LHA)
        - 8.147 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3)
        + 0.04507 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0)
        - 1.906 * max(0.0, Q.sum_pt - 813.415625)
        + 134.4 * max(0.0, 0.000657050184 - Q.girth2_top5)
        - 234.3 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7)
        + 0.01745 * max(0.0, Q.pt_7 - 34.53125)
        - 477.5 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0001279 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        + 0.1413 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 12.67 * max(0.0, 0.055577157257 - Q.z_7)
        - 28170.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2)
        - 572.9 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2)
        - 0.7053 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.2982 * max(0.0, 53.332374954224 - Q.mass)
        - 0.02155 * max(0.0, Q.pt_7 - 53.4375)
        - 0.805 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        + 261.3 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 600.1 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 51.76 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 0.04367 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 372.9 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 3.832 * max(0.0, Q.e2 - 0.032346998155)
        - 0.1163 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 15.92 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 506.5 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 16.68 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 19.54 * max(0.0, 0.043044721986 - Q.z_7)
        + 0.00269 * max(0.0, 49.668099212646 - Q.mass)
        + 259.7 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.002114 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 0.192 * max(0.0, Q.LHA - 0.266912960293)
        - 0.4949 * max(0.0, 0.250761204958 - Q.max_dr)
        + 30.97 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2)
        - 59.23 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 0.4416 * max(0.0, 69.611351776123 - Q.mass)
        - 0.04285 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 30.82 * max(0.0, 0.005954149834 - Q.lam1)
        + 0.008984 * max(0.0, Q.pt_7 - 30.484375)
        + 0.118 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        - 260.6 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 1.645 * max(0.0, Q.LHA - 0.111565049159)
        - 126.3 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        + 1.083 * max(0.0, 36.229410171509 - Q.mass)
        - 0.5279 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.0114 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        - 262.5 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 74.24 * max(0.0, 0.003377388461 - Q.lam1)
        + 0.01397 * max(0.0, 53.4375 - Q.pt_7)
        + 0.007831 * max(0.0, 43.5 - Q.pt_7)
        - 0.001 * max(0.0, 788.4484375 - Q.sum_pt)
        - 1658.0 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 963.6 * max(0.0, 0.000172198326 - Q.width)
        + 7728.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 40.92 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 4.568 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 35.71 * max(0.0, Q.width - 0.018827653081)
        - 1.08 * max(0.0, Q.mass - 36.229410171509)
        + 8.388 * max(0.0, Q.e2 - 0.028531698044)
        + 9.608 * max(0.0, 0.006679471358 - Q.width)
        - 0.6131 * max(0.0, 0.04447356835 - Q.e2)
        + 24.74 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.8547 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.02266 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 0.4334 * max(0.0, Q.mass - 69.611351776123)
        + 70.59 * max(0.0, 0.008678044751 - Q.girth2)
        - 0.03907 * max(0.0, Q.LHA - 0.325582223496)
        - 0.002879 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        - 4.202 * max(0.0, 0.038466955721 - Q.e2)
        - 59.18 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582)
        - 34.63 * max(0.0, Q.lam1 - 0.016433749775)
        + 6.384 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        - 262.2 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 20.18 * max(0.0, 0.004372139461 - Q.girth2)
        + 169.6 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 37.83 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        - 40.89 * max(0.0, Q.lam1 - 0.012003726523)
        + 496.5 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 0.04403 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 1.144 * max(0.0, Q.max_dr - 0.145231109113)
        - 1.725 * max(0.0, Q.C2 - 0.094821243733)
        + 241.4 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7)
        + 0.05391 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 17.82 * max(0.0, Q.LHA - 0.423592510895)
        - 32.55 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1)
        - 68.4 * max(0.0, Q.lam1 - 0.008375572068)
        - 105.7 * max(0.0, Q.lam1 - 0.004183811014)
        - 115.8 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        - 222.2 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7)
        + 0.1263 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125)
        - 195.1 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7)
        + 8.644 * max(0.0, Q.e2 - 0.050284641981)
        + 0.823 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015)
        + 2.897 * max(0.0, 0.23799610585 - Q.tau21)
        - 0.04593 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        - 225.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        - 1505.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        + 399.7 * max(0.0, Q.width - 0.000319370692)
        - 0.7408 * max(0.0, Q.e2 - 0.041109715588)
        + 0.004578 * max(0.0, 763.825 - Q.sum_pt)
        - 13.41 * max(0.0, 0.009530300104 - Q.girth2_top2)
        + 2852.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 5.177 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 79.41 * max(0.0, Q.width - 0.001653836415)
        + 0.0006449 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 3.288 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 12.89 * max(0.0, Q.C2 - 0.067292226106)
        + 176.4 * max(0.0, 0.001130644719 - Q.lam2)
        + 24.43 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 13.78 * max(0.0, Q.girth - 0.101940929517)
        + 204.9 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 56.95 * max(0.0, Q.girth2 - 0.013238675334)
        + 6.51 * max(0.0, Q.e2 - 0.063441075385)
        + 28.77 * max(0.0, Q.e2 - 0.007078157854)
        + 0.01228 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        + 0.01601 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 0.9518 * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 1.56 * max(0.0, Q.max_dr - 0.102758520097)
        - 6.29 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 1.576 * max(0.0, Q.max_dr - 0.197968879342)
        - 0.1081 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        - 6.795 * max(0.0, Q.C2 - 0.014943876117)
        + 0.1729 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 11.87 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 44.13 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 3.447 * max(0.0, 0.047915700823 - Q.girth)
        - 142.8 * max(0.0, 0.023780909279 - Q.e2_sq)
        - 165.5 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        - 52.03 * max(0.0, Q.girth2 - 0.007520088344)
        - 11.92 * max(0.0, 0.007639643088 - Q.girth2_top2)
        - 143.0 * max(0.0, 0.007182789718 - Q.e2_sq)
        - 27.85 * max(0.0, Q.width - 0.002635417778)
        - 28.81 * max(0.0, 0.003013300392 - Q.e2_sq)
        + 16.5 * max(0.0, 0.003111083776 - Q.girth2_top2)
        + 2.569 * max(0.0, Q.e2 - 0.020459658932)
        + 71.68 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.01121 * max(0.0, 41.377904891968 - Q.mass)
        + 2.854 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25)
        + 40.37 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944)
        + 471.7 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        + 5.375 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116)
        + 1.478 * max(0.0, 0.124553743005 - Q.girth)
        - 0.9304 * max(0.0, 80.4 - Q.mass)
        - 9.184 * max(0.0, 0.216055863061 - Q.LHA)
        - 0.1988 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 12.35 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 2.617 * max(0.0, 0.028865759995 - Q.z_6)
        + 3.128 * max(0.0, 0.071488645583 - Q.z_7)
        + 724.8 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 0.004454 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 53.42 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow)
        - 1.59 * max(0.0, 0.035560912266 - Q.e2)
        + 9.074 * max(0.0, 0.002635417778 - Q.width)
        + 53.51 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.001903 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 7735.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.001156 * max(0.0, 548.196875 - Q.sum_pt_top2)
        + 8620.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.06554 * max(0.0, Q.sum_pt - 868.509375)
        - 0.4842 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 594.2 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 8.497 * max(0.0, 0.154689112391 - Q.LHA)
        - 8913.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.07688 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 103.4 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        + 275.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 4.278 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 115.8 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 12000.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        + 0.04624 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.9323 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5)
        - 88970.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 24.94 * max(0.0, 0.021588001063 - Q.dr_0)
        + 290.4 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516)
        + 0.5263 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 11.25 * max(0.0, 0.05096141791 - Q.z_6)
        + 0.218 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5)
        - 114.0 * max(0.0, 0.023207568189 - Q.z_7)
        - 2.348 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5)
        + 74.1 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 86.57 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 0.7925 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 459.7 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471)
        + 1086.0 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset)
        - 0.001868 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 113.1 * max(0.0, 0.004007841607 - Q.girth2_top2)
        + 0.0007318 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2)
        - 30.53 * max(0.0, 0.013238675006 - Q.width)
        + 45.0 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 3204.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 21.81 * max(0.0, 0.050284641981 - Q.e2)
        + 56.57 * max(0.0, Q.lam2 - 0.000537286005)
        + 13.78 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        - 3.287 * max(0.0, Q.centroid_offset - 0.018377780003)
        + 521.1 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 11.35 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        + 2.764 * max(0.0, Q.centroid_offset - 0.049903668404)
        + 3.002 * max(0.0, Q.LHA - 0.312727471086)
        + 0.1921 * Q.max_dr
        + 593.6 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 4.594 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 3.605 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 20.09 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 14.36 * max(0.0, Q.girth - 0.087236513197)
        + 1.415 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        + 0.0476 * max(0.0, 1.679198372364 - Q.D2)
        + 32.88 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        - 148.7 * max(0.0, Q.lam2 - 0.003408388935)
        - 0.02768 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        - 0.8711 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386)
        - 15.4 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.0003401 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 12.29 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056)
        - 0.016 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 0.07157 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.001909 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        + 12.08 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        + 0.003143 * max(0.0, 5.351076855015 - Q.pt1_dr01)
        + 0.07695 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475)
        - 0.002011 * max(0.0, 31.125 - Q.pt_4)
        - 0.05528 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625)
        + 9.9 * max(0.0, 0.037477688199 - Q.z_4)
        - 5.423 * max(0.0, 0.024419631481 - Q.girth2_top5)
        + 411.2 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 0.109 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7)
        + 1.732 * max(0.0, Q.sum_pt - 988.4078125)
        - 0.0004562 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        - 2.193e-05 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625)
        - 0.007045 * max(0.0, 24.421875 - Q.pt_6)
        + 46.96 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 5.586 * max(0.0, Q.girth2 - 0.004372139461)
        - 10.99 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 79.4 * max(0.0, Q.girth2 - 0.0016538364)
        - 16.2 * max(0.0, 0.024547699839 - Q.e2)
        + 26.14 * max(0.0, 0.04081947431 - Q.girth)
        - 0.0002419 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        + 21.94 * max(0.0, 0.005590288644 - Q.width)
        + 4.22 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 45.68 * max(0.0, Q.girth2 - 0.008678044751)
        + 135.7 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 0.9311 * max(0.0, Q.mass - 80.4)
        - 70.34 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 33.02 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        - 4.515 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 68.81 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.1025 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 0.007386 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 7.34 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 458.6 * max(0.0, 0.000561123155 - Q.width)
        + 0.05147 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 0.323 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 5.906 * max(0.0, 0.002464291268 - Q.lam1)
        + 1.13 * max(0.0, 0.15984864831 - Q.max_dr)
        - 1.416 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 1.254 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361)
        - 0.02735 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        + 226.9 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257)
        + 23.55 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581)
        + 16.35 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 13.55 * max(0.0, Q.e2 - 0.016554418951)
        - 0.002777 * max(0.0, Q.mass_top5 - 53.607658247923)
        + 0.04217 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012)
        + 1.908 * max(0.0, 0.067292226106 - Q.C2)
        - 0.3662 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 13.61 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 13.98 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 9.609 * max(0.0, 0.006679471442 - Q.girth2)
        + 124.5 * max(0.0, 0.005019718802 - Q.width)
        - 0.1095 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7)
        + 6178.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 4.986 * max(0.0, 0.196739721581 - Q.LHA)
        + 164.1 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 26.68 * max(0.0, 0.061086014472 - Q.girth)
        - 0.009457 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 0.6914 * max(0.0, 0.177304983139 - Q.max_dr)
        + 2085.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 64.77 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 3.192 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 8005.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 1265.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 20020.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 14540.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        + 0.07425 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 423.9 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 9055.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        + 0.5271 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 2148.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 144.8 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 1.485 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 11660.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 19020.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 583.6 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 246.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 6217.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 3023.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.09106 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 0.8998 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 934.3 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 901.0 * max(0.0, 0.000964142894 - Q.girth2)
        - 0.386 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 216.1 * max(0.0, 0.000222950415 - Q.girth2_top5)
        - 4.781 * max(0.0, 0.027029510401 - Q.C2)
        - 2386.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 1474.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width)
        + 7785.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442)
        - 812.7 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 0.2076 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2)
        + 140.8 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 58.19 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2)
        - 25.19 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width)
        - 247.6 * max(0.0, 0.000194798295 - Q.lam2)
        - 99.67 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787)
        + 405.7 * max(0.0, 0.000319370692 - Q.width)
        - 57.1 * max(0.0, 0.000823693417 - Q.girth2_top3)
        - 609.7 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 26.14 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 0.01704 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5)
        + 7.87 * max(0.0, 0.054649224505 - Q.girth)
        + 120.4 * max(0.0, Q.lam2 - 0.001130644719)
        - 0.05388 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 46.74 * max(0.0, 0.006096650059 - Q.width)
        - 7.272 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 35.7 * max(0.0, Q.girth2 - 0.018827652745)
        + 8.782 * max(0.0, 0.032346998155 - Q.e2)
        - 63.01 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.05802 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 0.6155 * max(0.0, 0.221586732566 - Q.max_dr)
        - 1.819 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5)
        + 1621.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 1.272 * max(0.0, Q.C2 - 0.051192347892)
        + 0.1759 * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 670.4 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 31.96 * max(0.0, 0.007520088344 - Q.girth2)
        - 5.948 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5)
        + 50.95 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5)
        + 0.002861 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.1695 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 129.1 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 30.6 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        - 0.0421 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 12.93 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 7.525 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 0.935 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 14.1 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 0.005797 * max(0.0, 35.5 - Q.pt_5)
        - 0.118 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        + 0.06503 * max(0.0, Q.sum_pt - 840.01953125)
        + 0.05948 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3)
        + 3.008 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2)
        - 273.5 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2)
        + 273.2 * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 9537.0 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189)
        + 11.88 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875)
        + 1.848 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125)
        + 0.22 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 1912.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 809.1 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421)
        - 964.7 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2)
        - 291.5 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        + 18.39 * max(0.0, 0.033604209498 - Q.girth)
        - 2365.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 13.26 * Q.e2
        + 62.45 * max(0.0, Q.lam2 - 0.000194798295)
        - 10.02 * max(0.0, Q.LHA - 0.303313749495)
        - 11.25 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 25.97 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        - 88.62 * max(0.0, 0.004183811014 - Q.lam1)
        + 8.322 * max(0.0, Q.centroid_offset - 0.037760993714)
        + 0.06237 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        - 0.8352 * max(0.0, 0.391541349888 - Q.tau21)
        + 47.56 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 83.96 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        + 177.8 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 3.039 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        + 1043.0 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 1919.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.01822 * max(0.0, Q.mass - 15.454033088684)
        - 0.3048 * max(0.0, Q.mass - 53.332374954224)
        - 64.67 * max(0.0, 0.0016538364 - Q.girth2)
        + 195.8 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 5.031 * max(0.0, Q.LHA - 0.346713497427)
        - 146.4 * max(0.0, 0.003408388935 - Q.lam2)
        - 1.833 * max(0.0, 0.253403707141 - Q.planar_flow)
        - 16.81 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        - 0.1302 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2)
        + 45.69 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334)
        - 9.446 * max(0.0, Q.girth - 0.076081777364)
        + 0.03777 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        + 3.084 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.1282 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 54.93 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 12.2 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.3676 * Q.centroid_offset
        + 103.6 * max(0.0, 0.003562611091 - Q.width)
        + 209.2 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2)
        - 173.1 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        - 0.4977 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.001363 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 4.669e-05 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 16.84 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 3.215 * max(0.0, 0.004839980301 - Q.lam1)
        - 0.514 * max(0.0, 0.035786485299 - Q.C2)
        + 0.3204 * max(0.0, 0.111761856824 - Q.max_dr)
        + 1.347 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 7.696 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2)
        - 10.06 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.2117 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7)
        - 17.22 * max(0.0, 0.02054281719 - Q.girth)
        + 0.1483 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0)
        - 3.096 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 45.22 * max(0.0, 0.148408418149 - Q.girth)
        + 6.637 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.4019 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        - 4.087 * max(0.0, 0.016433749775 - Q.lam1)
        - 1041.0 * max(0.0, 0.000306123359 - Q.lam2)
        + 0.6806 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        + 0.03244 * max(0.0, 31.90625 - Q.pt_6)
        + 49390.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 4.288e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 2.786 * max(0.0, Q.z_top5_slots - 0.930764273368)
        + 37.74 * max(0.0, 0.028070914944 - Q.z_7)
        - 0.01591 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.708 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        - 44.85 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        + 0.01343 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32)
        + 137.9 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 7.892 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 1.158 * max(0.0, Q.LHA - 0.09323897448)
        + 0.003284 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 2.217 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        - 31.97 * max(0.0, 0.00752008842 - Q.width)
        + 0.01018 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0009114 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0)
        - 48.87 * max(0.0, 0.006506575659 - Q.lam1)
        + 0.1337 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375)
        + 0.0537 * max(0.0, 25.578125 - Q.pt_7)
        - 0.006696 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.003217 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        - 0.008005 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 110.1 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.8974 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 100.3 * max(0.0, Q.lam1 - 0.00543336053)
        - 0.4752 * max(0.0, Q.lam1 - 0.007330079875)
        - 3.129 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        - 1304.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        - 113.2 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 0.1956 * max(0.0, 0.080507021025 - Q.max_dr)
        - 4.357 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.000287 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 20.41 * max(0.0, Q.lam1 - 0.002464291268)
        - 955.4 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 819.3 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        - 55.11 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2)
        + 0.1293 * max(0.0, 1.122624260187 - Q.D2)
        + 33.65 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.009249 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        + 6.886 * max(0.0, 0.303313749495 - Q.LHA)
        + 196.2 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 11.59 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.9841 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.1564 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578)
        + 0.0003497 * max(0.0, 86.4 - Q.mass)
        + 2266.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 52.7 * max(0.0, 0.005834489329 - Q.e2_sq)
        + 11.42 * max(0.0, 0.041109715588 - Q.e2)
        - 47.08 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 37.06 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        + 0.3203 * max(0.0, 0.135767506063 - Q.tau21)
        + 7.345 * max(0.0, Q.centroid_offset - 0.012587644117)
        + 1.689 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364)
        - 45.7 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 2.998 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364)
        + 0.8107 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 0.5713 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 10.13 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537)
        + 8617.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 14.38 * max(0.0, 0.012003726523 - Q.lam1)
        + 9.998 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 969.6 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.001511 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 0.8214 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 18.95 * max(0.0, 0.063441075385 - Q.e2)
        + 0.2389 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 0.3877 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.01127 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 786.1 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744)
        - 20.57 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 2439.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.4675 * max(0.0, 0.083662731125 - Q.planar_flow)
        - 41.41 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 0.1054 * max(0.0, 0.74595130682 - Q.D2)
        - 4.688 * max(0.0, Q.girth - 0.033604209498)
        - 54.65 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['W'] - s['t'] > -3.656342029571533:
        if s['q'] - s['Z'] > 0.4213292747735977:
            if s['g'] - s['q'] > -0.07172336801886559:
                if s['g'] - s['W'] > -0.09008387103676796:
                    if s['g'] - s['t'] > -0.03588720224797726:
                        if s['g'] - s['q'] > 0.02086247969418764:
                            if s['g'] - s['W'] > 0.20757047832012177:
                                if s['g'] - s['q'] > 0.0819404236972332:
                                    if s['g'] - s['t'] > 0.20716923475265503:
                                        if Q.mass > 33.10029220581055:
                                            if s['g'] - s['W'] > 0.5909198224544525:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.05244150571525097:
                                                    if Q.mass > 36.75882911682129:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0001462176878703758:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 0.13569191843271255:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0008072535274550319:
                                                    if Q.sum_pt_top5 > 532.40625:
                                                        if Q.centroid_offset > 0.009849337860941887:
                                                            if s['g'] - s['t'] > 1.6051827669143677:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.2612517476081848:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.08852948620915413:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.02021710015833378:
                                        if Q.sum_pt_top5 > 522.078125:
                                            if Q.sum_pt_top5 > 654.140625:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0010722725419327617:
                                                    if Q.centroid_offset > 0.008254257962107658:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > 0.36447378993034363:
                                                        if Q.mass > 12.826893329620361:
                                                            if Q.centroid_offset > 0.010182218626141548:
                                                                if Q.dr_3 > 0.021860795095562935:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.11633766815066338:
                                            if Q.pt_7 > 40.234375:
                                                if Q.centroid_offset > 0.004258869215846062:
                                                    if Q.sum_pt > 800.734375:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.059264132753014565:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > 0.05017380230128765:
                                                                return 'g'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 34.54186820983887:
                                    if Q.width > 0.003037990187294781:
                                        return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 0.07685444131493568:
                                        if Q.lam2 > 0.0001305349578615278:
                                            if Q.mass > 31.833274841308594:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top2 > 0.0007976466149557382:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.006786818616092205:
                                if Q.log_sum_pt > 6.677339315414429:
                                    if Q.centroid_offset > 0.01340660359710455:
                                        return 'q'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top2 > 377.625:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.0537590142339468:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 60% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > -0.013725018594413996:
                                        if Q.sum_pt_top5 > 533.0625:
                                            if Q.girth2_top3 > 0.0005090545164421201:
                                                if Q.mass > 30.511170387268066:
                                                    return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 37.671875:
                                                    if Q.mass > 18.244544982910156:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.015275482088327408:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.592766046524048:
                                                                return 'q'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_pair_mass > 7.560944080352783:
                                                if Q.sum_pt_top5 > 497.5625:
                                                    if Q.girth2_top5 > 0.0020929572638124228:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 496.4140625:
                                            if Q.girth2_top2 > 0.000463643460534513:
                                                if Q.eccentricity > 0.987347275018692:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.038828786462545395:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 36.90625:
                                                    if Q.sum_pt_top5 > 542.40625:
                                                        if Q.max_dr > 0.07458188012242317:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.00029747100779786706:
                                                        if s['g'] - s['q'] > -0.03303825855255127:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > -0.061811523512005806:
                                                                if Q.dr_7 > 0.021480930037796497:
                                                                    return 'q'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_6 > 0.07565342262387276:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > 0.4913906753063202:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.dr_0 > 0.010865391697734594:
                                    if Q.pt_7 > 35.21875:
                                        if Q.sum_pt > 827.140625:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.6413657665252686:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.004948618356138468:
                                                    if Q.log_sum_pt > 6.598998546600342:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.7094564437866211:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.02863399311900139:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > -0.027432959526777267:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05232924222946167:
                                            if Q.dr_0 > 0.0059750855434685946:
                                                if Q.log_sum_pt > 6.685918807983398:
                                                    if Q.sum_pt > 877.34375:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.607409954071045:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 548.3125:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.28801967203617096:
                            if s['W'] - s['t'] > -0.8030828833580017:
                                if Q.mass > 35.43067169189453:
                                    return 'W'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 20.4609375:
                                    if Q.z_6 > 0.05398978479206562:
                                        if s['q'] - s['Z'] > 1.0443887114524841:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 56% of the training jets here get this class from the formula
                        else:
                            return 't'   # 98% of the training jets here get this class from the formula
                else:
                    if Q.mass > 33.70472717285156:
                        if s['W'] - s['t'] > -0.15390240401029587:
                            if s['g'] - s['t'] > 1.6810532808303833:
                                if s['g'] - s['W'] > -0.38402150571346283:
                                    return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.019689837470650673:
                                return 't'   # 100% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 57% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.45858150720596313:
                            if s['g'] - s['q'] > 0.08964121714234352:
                                if Q.mass > 31.90170192718506:
                                    if s['q'] - s['W'] > -0.7948423326015472:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 405.65625:
                                        if Q.mass_top5 > 16.25725221633911:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 35.5625:
                                    return 'q'   # 47% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 57% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > -0.6092826724052429:
                                if Q.pt_5 > 49.734375:
                                    return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 70% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 94% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > -0.016686185263097286:
                    if s['g'] - s['q'] > -0.1499565690755844:
                        if Q.sum_pt_top5 > 791.625:
                            if Q.pt_7 > 46.484375:
                                return 'q'   # 55% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.lam1 > 0.00014771275891689584:
                                if s['g'] - s['t'] > 0.03525386843830347:
                                    if Q.log_sum_pt > 6.713164806365967:
                                        if Q.lam2 > 6.0549618865479715e-05:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 45.015625:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.03922056220471859:
                                            if Q.eccentricity > 0.98699089884758:
                                                if Q.sum_pt_top5 > 482.4375:
                                                    if Q.max_dr > 0.09086654707789421:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > 0.06209161318838596:
                                                if Q.eccentricity > 0.8724725544452667:
                                                    if s['g'] - s['q'] > -0.1140897199511528:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 36.484375:
                                    if Q.sum_pt > 803.6875:
                                        if s['g'] - s['q'] > -0.08990217000246048:
                                            if s['W'] - s['Z'] > -0.0229840325191617:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.0061457648407667875:
                                                if Q.tau32 > 0.6926512122154236:
                                                    if Q.sum_pt > 849.984375:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 885.53125:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 60.65625:
                                                        return 'q'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_3 > 0.10114331543445587:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 572.53125:
                                            if Q.pt_7 > 38.765625:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.006292588543146849:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > -0.10403946414589882:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_2 > 72.5:
                                        if Q.mass > 6.2076990604400635:
                                            if s['g'] - s['q'] > -0.11186743900179863:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 64% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > 0.026596135459840298:
                            if s['q'] - s['W'] > 0.2478545382618904:
                                if Q.sum_pt > 1327.48046875:
                                    if s['g'] - s['W'] > 2.2484596967697144:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > -0.21382001787424088:
                                        if Q.log_sum_pt > 6.919992208480835:
                                            if Q.dr_3 > 0.007283798418939114:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr01 > 0.00426083174534142:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 758.0:
                                                if Q.z_top5_slots > 0.828692227602005:
                                                    if Q.sum_pt_top3 > 627.5:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.051959285512566566:
                                    return 'W'   # 79% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 0.6571898460388184:
                                        if Q.max_dr > 0.2096061259508133:
                                            if Q.dr_1 > 0.0143133201636374:
                                                if Q.lam2 > 1.899906146718422e-05:
                                                    if s['W'] - s['t'] > 1.9948075413703918:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 9.434159801458009e-05:
                                                if Q.tau32 > 0.6853070855140686:
                                                    if Q.eccentricity > 0.8833225965499878:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.37599651515483856:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.666050672531128:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.744749069213867:
                                            if s['q'] - s['Z'] > 0.5733045935630798:
                                                if Q.pt_6 > 39.640625:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 7.003085136413574:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > -1.4895304441452026:
                                                    if Q.dr_5 > 0.011944754514843225:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0004134370246902108:
                                                if Q.eccentricity > 0.8198212087154388:
                                                    if Q.mean_phi > 0.007614883594214916:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_2 > 0.027541404590010643:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['t'] > -0.07994922623038292:
                                if Q.z_dr_0_0p05 > 0.8552281558513641:
                                    return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 68% of the training jets here get this class from the formula
                            else:
                                return 't'   # 86% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.4162856340408325:
                        if s['q'] - s['Z'] > 0.7475619614124298:
                            if Q.width > 0.0028175737243145704:
                                if s['q'] - s['t'] > -0.19305088371038437:
                                    return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.16776200383901596:
                                    if Q.centroid_offset > 0.010025989264249802:
                                        if Q.mass > 34.74825859069824:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3060484379529953:
                                            if Q.lam2 > 3.2200090572587214e-05:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00013183725241106004:
                                        return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.1987355649471283:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > -0.24970068037509918:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['t'] > 1.6462758779525757:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 746.4375:
                                return 'W'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.0016072193393483758:
                                    if s['Z'] - s['t'] > -0.418323278427124:
                                        return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 48% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.8751579821109772:
                                        if Q.log_sum_pt > 6.726701736450195:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 85% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['t'] > -0.012726445216685534:
                            if s['q'] - s['W'] > -0.6230399310588837:
                                if s['q'] - s['Z'] > 1.3168842792510986:
                                    if Q.z_7 > 0.031162959523499012:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            return 't'   # 72% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > 0.030281074345111847:
                if s['g'] - s['W'] > -0.09939562529325485:
                    if s['g'] - s['t'] > 0.025082753971219063:
                        if s['g'] - s['W'] > 0.16001125425100327:
                            if Q.mass > 30.561052322387695:
                                if s['g'] - s['W'] > 0.45833712816238403:
                                    if s['g'] - s['t'] > 0.2785532772541046:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_7 > 0.06495708972215652:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 42.16705322265625:
                                        return 'W'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_4 > 66.5625:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 0.3551919311285019:
                                                if Q.girth > 0.05079027824103832:
                                                    if Q.mass_top5 > 25.307276725769043:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi > 0.0005174510588403791:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.41366295516490936:
                                    return 'g'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03620018996298313:
                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > 0.12396382540464401:
                                if Q.mass > 31.184309005737305:
                                    if Q.lam1 > 0.002443617326207459:
                                        if Q.planar_flow > 0.5736563205718994:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1022.5:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.1533163785934448:
                                                    if s['q'] - s['t'] > -0.08185278624296188:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.202725172042847:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.02718606311827898:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.013596555683761835:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.04215773940086365:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > -0.00029170729976613075:
                                        if s['g'] - s['Z'] > 0.3502069115638733:
                                            if Q.m012 > 15.272252082824707:
                                                if s['g'] - s['Z'] > 0.8068530559539795:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.5875720977783203:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01941568683832884:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.0003608535771491006:
                                                            if Q.phi_1 > 0.004231452941894531:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.510560512542725:
                                                if Q.z_7 > 0.07519722729921341:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.1786630004644394:
                                                        if Q.sum_pt > 729.09375:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02099397126585245:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.18568485975265503:
                                                    if Q.mass > 20.85184669494629:
                                                        if Q.dr_0 > 0.04964970052242279:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 624.4375:
                                                            if s['g'] - s['Z'] > 0.204405277967453:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 1.7448652386665344:
                                            if Q.max_dr > 0.033670639619231224:
                                                if Q.planar_flow > 0.32470373809337616:
                                                    if Q.mass > 16.364209175109863:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.4865802526473999:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_eta > 0.015567902941256762:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.01065652770921588:
                                                if s['g'] - s['Z'] > 0.5586956441402435:
                                                    if Q.dr01 > 0.04904220625758171:
                                                        if s['W'] - s['Z'] > 0.9836760461330414:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.2640531063079834:
                                                        if Q.mass > 17.01967144012451:
                                                            if Q.girth2_top5 > 0.002531133475713432:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_pair_mass > 1.9550586342811584:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 502.75:
                                                                if Q.LHA > 0.18721164762973785:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.0267066964879632:
                                    if Q.pt_6 > 36.0:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 30.9375:
                                        if s['q'] - s['Z'] > -1.3533760905265808:
                                            return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 36% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 55% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.23307395726442337:
                            if Q.lam2 > 0.00015277823695214465:
                                return 't'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.0272762356325984:
                                    return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > 0.3788738548755646:
                                        return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 65% of the training jets here get this class from the formula
                        else:
                            return 't'   # 98% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.22128607332706451:
                        if s['g'] - s['W'] > -0.4343436062335968:
                            if s['W'] - s['t'] > 0.08528303727507591:
                                if s['g'] - s['Z'] > 0.47188661992549896:
                                    if Q.mass > 29.707866668701172:
                                        if Q.width > 0.002880239742808044:
                                            if Q.mass > 32.377113342285156:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9836409091949463:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > -0.22757045179605484:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.2560960501432419:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['W'] > -1.4342595338821411:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.6871640384197235:
                                                if Q.tau21 > 0.22269344329833984:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 851.890625:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 733.84375:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top5 > 21.466361045837402:
                                            if s['g'] - s['Z'] > 0.8282325267791748:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.10242651030421257:
                                                if Q.mean_phi > -0.01713559590280056:
                                                    if Q.z_4 > 0.09887212887406349:
                                                        if Q.eccentricity > 0.8895268440246582:
                                                            if Q.centroid_offset > 0.018534038215875626:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > 2.2161515951156616:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.026037058793008327:
                                        if Q.D2 > 1.14820796251297:
                                            if Q.lam1 > 0.000783622992457822:
                                                if Q.e2 > 0.02108147367835045:
                                                    if Q.pt_dispersion > 0.38907214999198914:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.10326474159955978:
                                                if s['g'] - s['q'] > 1.0374789237976074:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -0.25435495376586914:
                                            if Q.centroid_offset > 0.01988859847187996:
                                                if s['g'] - s['Z'] > 0.18606936186552048:
                                                    if Q.lam2 > 0.00013136152847437188:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.570664167404175:
                                                            if Q.mass_over_sum_pt > 0.013628303073346615:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.17825409770011902:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_5 > 52.90625:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 6.172293433337472e-05:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['t'] > 0.9618055820465088:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 34.296875:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 5.247312307357788:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.18431681394577026:
                                    if Q.z_dr_0p1_0p2 > 0.06147405505180359:
                                        return 'W'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 98% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.06873461976647377:
                                if s['W'] - s['Z'] > 0.3674102872610092:
                                    if Q.mass > 79.39935302734375:
                                        if s['g'] - s['q'] > 0.8020533621311188:
                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -0.6713499128818512:
                                            if s['W'] - s['t'] > 0.31532590091228485:
                                                if s['g'] - s['Z'] > 0.3795367330312729:
                                                    if Q.mass > 29.657462120056152:
                                                        if Q.girth2 > 0.002782045863568783:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.342131644487381:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > 0.7079617977142334:
                                                            if Q.C2 > 0.02939148899167776:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.3635285198688507:
                                                                if Q.dr01 > 0.03128264099359512:
                                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_7 > 0.07328669726848602:
                                                    return 't'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 0.5294786691665649:
                                                if s['q'] - s['W'] > -0.2871907502412796:
                                                    if Q.centroid_offset > 0.02149895206093788:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 8.967090980149806e-05:
                                                        if Q.centroid_offset > 0.03791450895369053:
                                                            if Q.lam2 > 0.00010440839832881466:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.06672833114862442:
                                                                if Q.mass_over_sum_pt_sq > 0.00547476252540946:
                                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.018200612626969814:
                                                                        if Q.tau21 > 0.3928883522748947:
                                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_2 > 0.02610842604190111:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > -1.0125470161437988:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 0.2587316036224365:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.24190139025449753:
                                            if Q.mass_over_sum_pt > 0.05059343762695789:
                                                if Q.e2_sq > 0.005030386848375201:
                                                    if Q.D2 > 2.1903719902038574:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2608901411294937:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 76.36629867553711:
                                                if s['g'] - s['q'] > 0.3900008350610733:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0006861493457108736:
                                                    if Q.e2 > 0.0036756854970008135:
                                                        if Q.sum_pt_top2 > 455.203125:
                                                            if s['W'] - s['Z'] > 0.2699148803949356:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.011165931820869446:
                                                                    if Q.mean_phi > -0.006587477633729577:
                                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau21 > 0.11955910548567772:
                                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.07993278279900551:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.031252953223884106:
                                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.09067528322339058:
                                    if Q.z_top5_slots > 0.8322616815567017:
                                        return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 90% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.10964836552739143:
                            if Q.girth > 0.024895813316106796:
                                if Q.e2 > 0.0037190094590187073:
                                    if s['W'] - s['t'] > -0.07351279631257057:
                                        if Q.z_dr_0p05_0p1 > 0.1314820870757103:
                                            if Q.log_sum_pt > 6.846975326538086:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03835102543234825:
                                                    if Q.max_dr > 0.07972485944628716:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > -3.1156245470046997:
                                                        if Q.mass_over_sum_pt_sq > 0.0008406060806009918:
                                                            if Q.C2 > 0.030705785378813744:
                                                                if Q.girth > 0.07378539815545082:
                                                                    if Q.pt_4 > 59.234375:
                                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.05002984777092934:
                                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        if s['g'] - s['W'] > -1.6785358786582947:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.z_3 > 0.08866617828607559:
                                                                                if Q.mass_top5 > 35.72702598571777:
                                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006608139723539352:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_2 > 0.09247495234012604:
                                                                if s['q'] - s['t'] > -1.5081470012664795:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -1.0322272181510925:
                                                if s['q'] - s['Z'] > 0.21201874315738678:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.02703838236629963:
                                                        if Q.centroid_offset > 0.02607625536620617:
                                                            if Q.lam1 > 0.0008096125675365329:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02695064153522253:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.015270049218088388:
                                                    if Q.mean_phi > -0.008908764459192753:
                                                        if Q.z_3 > 0.12229614332318306:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['W'] > -1.8012346029281616:
                                                                if Q.mass > 51.57291793823242:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                if s['W'] - s['Z'] > 0.14952827990055084:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.9793999791145325:
                                                                        if Q.centroid_offset > 0.011806359514594078:
                                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.23345492780208588:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 340.0625:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.18121371418237686:
                                                        if Q.e2 > 0.006515519460663199:
                                                            if Q.mass > 35.65789794921875:
                                                                if Q.centroid_offset > 0.023170556873083115:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mean_phi > -0.0073430342599749565:
                                                                        if Q.max_dr > 0.23537186533212662:
                                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['Z'] > -0.49797293543815613:
                                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['t'] > 1.9967437982559204:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.1806468814611435:
                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 2.5094757080078125:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.0026578597025945783:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > 0.1435728594660759:
                                    return 'q'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.025059746578335762:
                                if Q.z_dr_0p05_0p1 > 0.026670907624065876:
                                    if Q.mass > 17.856791496276855:
                                        if s['Z'] - s['t'] > 0.1465037390589714:
                                            if Q.max_dr > 0.14670879393815994:
                                                if Q.eccentricity > 0.9876608848571777:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.3020899146795273:
                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -1.0979625582695007:
                                        if Q.e2 > 0.004624583525583148:
                                            if Q.eccentricity > 0.9171980321407318:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.026060800999403:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.00313429266680032:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -1.1825730800628662:
                                    if s['g'] - s['t'] > 0.6195141971111298:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > -0.03378794342279434:
                                            if Q.eccentricity > 0.9808289706707001:
                                                if s['g'] - s['t'] > -0.5403714179992676:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 494.671875:
                                        if Q.lam2 > 0.00011048485976061784:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010737589560449123:
                                                if Q.girth2_top2 > 0.0004881108470726758:
                                                    if Q.phi_1 > -0.01740264892578125:
                                                        if Q.tau32 > 0.7217623889446259:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 629.59375:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top2 > 491.84375:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau32 > 0.6452513039112091:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.5509928464889526:
                                                    if Q.max_dr > 0.09552757069468498:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.m012 > 1.3251863718032837:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > -0.07565896585583687:
                                            if Q.lam1 > 0.006556378677487373:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 6.123175262473524e-05:
                                                    if s['g'] - s['Z'] > -2.220358371734619:
                                                        if Q.eccentricity > 0.7152161002159119:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00028064030630048364:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_1 > 0.05345790274441242:
                                                        if Q.z_dr_0p1_0p2 > 0.24319295585155487:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.00114705681335181:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.5416314005851746:
                                                                if s['g'] - s['W'] > -0.7429044842720032:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > -0.009589617140591145:
                    if s['g'] - s['t'] > 0.02151847630739212:
                        if s['g'] - s['Z'] > 0.19565021991729736:
                            if s['g'] - s['t'] > 0.18814144283533096:
                                if s['g'] - s['Z'] > 0.333301842212677:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 0.11283142492175102:
                                        if Q.lam1 > 0.0038407084066420794:
                                            if Q.pt_5 > 40.09375:
                                                if Q.mean_phi > 0.013744103256613016:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 343.53125:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.9498854279518127:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03914253227412701:
                                                    if Q.dr_5 > 0.03712325729429722:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top2 > 0.001281363656744361:
                                    if Q.D2 > 0.9042850732803345:
                                        if Q.z_dr_0p05_0p1 > 0.43565648794174194:
                                            if s['g'] - s['t'] > 0.1406703144311905:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.00492410478182137:
                                                    if Q.centroid_offset > 0.04174889996647835:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.003453698707744479:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.3701281398534775:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 65% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > 0.06692903116345406:
                                if Q.log_sum_pt > 6.568648338317871:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 0.21991850435733795:
                                        if Q.centroid_offset > 0.02732005901634693:
                                            if Q.log_sum_pt > 6.444923639297485:
                                                if Q.pt_7 > 31.359375:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 35% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03365492448210716:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.3307534158229828:
                                            if Q.z_7 > 0.07814211770892143:
                                                if s['g'] - s['Z'] > 0.14909815043210983:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > 0.004714712500572205:
                                                        if Q.mean_eta > -0.01190887438133359:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.21038837730884552:
                                                    if s['q'] - s['t'] > -0.012917026411741972:
                                                        if Q.mean_eta2 > 0.0014521311386488378:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.5167457461357117:
                                                                if Q.sum_pt_top2 > 291.375:
                                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.2602732330560684:
                                                if Q.dr_4 > 0.06394177675247192:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['Z'] > -0.8432106673717499:
                                    if Q.D2 > 1.1103472709655762:
                                        if Q.pt_7 > 33.171875:
                                            if Q.girth2_top2 > 0.0008698110177647322:
                                                if Q.tau21 > 0.6200878322124481:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.phi_1 > -0.03046417236328125:
                                                        if Q.mass_top5 > 6.549988031387329:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.06997410207986832:
                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_4 > 0.08492294326424599:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 26.1328125:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 38% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > 0.5889523923397064:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.3321487754583359:
                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.13568494468927383:
                            if Q.log_sum_pt > 6.022538900375366:
                                if Q.pt_7 > 47.046875:
                                    return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.0708172619342804:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0034331511706113815:
                                            if s['Z'] - s['t'] > -0.30885277688503265:
                                                if Q.mass > 35.67272186279297:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 69% of the training jets here get this class from the formula
                        else:
                            return 't'   # 96% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.07778757438063622:
                        if s['W'] - s['Z'] > -0.19750725477933884:
                            if Q.mass > 7.309357643127441:
                                if s['W'] - s['Z'] > -0.08566584065556526:
                                    if Q.max_dr > 0.14695853739976883:
                                        if Q.lam2 > 0.00010604575072648004:
                                            if Q.sum_pt_top3 > 317.453125:
                                                if Q.girth2_top2 > 0.0027589009841904044:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.0047757262364029884:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_2 > 99.15625:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.0035946102580055594:
                                                                if s['W'] - s['Z'] > -0.03095258492976427:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.0035456718178465962:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt1_dr01 > 0.828443318605423:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.012423614040017128:
                                                if Q.dr_0 > 0.049994584172964096:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.42190833389759064:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.03022200521081686:
                                                            if Q.z_dr_0p1_0p2 > 0.02479131519794464:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if s['W'] - s['Z'] > 0.0031127234688028693:
                                                                    if Q.dr_1 > 0.03168261609971523:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.004875447601079941:
                                                    if Q.mass > 52.91705894470215:
                                                        if Q.dr_4 > 0.018729008734226227:
                                                            if Q.dr_5 > 0.04458015039563179:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['Z'] > -2.3934414386749268:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['g'] - s['t'] > -1.0764406323432922:
                                                                        if Q.pt1_dr01 > 0.8069440424442291:
                                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.07210465148091316:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.011341363191604614:
                                            if Q.mass > 60.79006385803223:
                                                if Q.tau32 > 0.45198841392993927:
                                                    if Q.z_dr_0p05_0p1 > 0.47289039194583893:
                                                        if Q.sum_pt_top5 > 630.21875:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.000807512435130775:
                                                    if Q.eccentricity > 0.8844726383686066:
                                                        if s['g'] - s['W'] > -0.4026716500520706:
                                                            if Q.centroid_offset > 0.033986056223511696:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.10793115571141243:
                                                                if Q.lam1 > 0.006706172600388527:
                                                                    if Q.z_7 > 0.05725834332406521:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['g'] - s['Z'] > -2.3122477531433105:
                                                                        if Q.D2 > 0.8521276414394379:
                                                                            if Q.pt_2 > 81.71875:
                                                                                if Q.eccentricity > 0.9756551384925842:
                                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 0.6273821592330933:
                                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 12.260416030883789:
                                                                    if Q.dr_0 > 0.09128084033727646:
                                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.037744808942079544:
                                                                            if Q.sum_pt_top5 > 547.140625:
                                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.025118034332990646:
                                                            if Q.dr_1 > 0.03953937441110611:
                                                                if Q.z_6 > 0.0602226834744215:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.04639975726604462:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if s['q'] - s['W'] > -0.8869735300540924:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 0.0009127209486905485:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.025715233758091927:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.09617987275123596:
                                                    if Q.max_pair_mass > 26.521002769470215:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006656207609921694:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 544.0:
                                                        if Q.centroid_offset > 0.02406969480216503:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_2 > 115.0:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 452.953125:
                                        if s['q'] - s['Z'] > -0.9186210632324219:
                                            if s['W'] - s['t'] > 1.3812682628631592:
                                                if Q.centroid_offset > 0.02564182970672846:
                                                    if Q.sum_pt_top5 > 726.234375:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.015397564508020878:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > -0.09234565496444702:
                                                    if s['g'] - s['t'] > 0.32368887960910797:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.0025592950405552983:
                                                if Q.sum_pt_top5 > 619.40625:
                                                    if Q.width > 0.005211265757679939:
                                                        if Q.max_dr > 0.09403584524989128:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.006795406341552734:
                                                        if Q.eccentricity > 0.9904931485652924:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.5137473940849304:
                                                            if Q.centroid_offset > 0.025566214695572853:
                                                                if s['g'] - s['W'] > -1.3067901134490967:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2680003345012665:
                                                                if Q.dr_4 > 0.10539735481142998:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_2 > 0.08805956318974495:
                                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        if s['g'] - s['t'] > -0.3254171758890152:
                                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.dr01 > 0.1703120395541191:
                                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 8.635440826765262e-05:
                                                    if Q.e2 > 0.01388725358992815:
                                                        if Q.LHA > 0.26031753420829773:
                                                            if s['W'] - s['Z'] > -0.11337459087371826:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['t'] > -0.2100011482834816:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 56.27879333496094:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.planar_flow > 0.34757906198501587:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.00476012728177011:
                                                        if s['q'] - s['Z'] > -1.0311683416366577:
                                                            if Q.LHA > 0.2002403512597084:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_5 > 59.234375:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.7921203374862671:
                                            if Q.z_6 > 0.09733695536851883:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13472700119018555:
                                                    if Q.dr_7 > 0.03361596539616585:
                                                        if Q.pt_2 > 64.171875:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.043082095682621:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.09594873711466789:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.024292937479913235:
                                    if Q.sum_pt > 627.1015625:
                                        if s['q'] - s['Z'] > -0.20501700043678284:
                                            return 'q'   # 42% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.025400209240615368:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -1.159336805343628:
                                                    if Q.e2 > 0.00284252327401191:
                                                        if Q.sum_pt_top2 > 437.46875:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -1.238940179347992:
                                        if Q.log_sum_pt > 6.702229022979736:
                                            if s['q'] - s['Z'] > -1.0999363660812378:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.328020378947258:
                                if Q.log_sum_pt > 6.617152214050293:
                                    if s['g'] - s['W'] > 0.4047546088695526:
                                        if Q.pt_5 > 59.65625:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.8296753764152527:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.0043020411394536495:
                                            if s['q'] - s['t'] > 0.9029998183250427:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.010481260251253843:
                                        if s['g'] - s['Z'] > -0.1027761809527874:
                                            if Q.C2 > 0.014838500414043665:
                                                if Q.dr_1 > 0.03131643496453762:
                                                    if Q.sum_pt > 455.203125:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.29104699194431305:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['t'] > -0.07837919518351555:
                                                if Q.dr_1 > 0.02981876116245985:
                                                    if Q.tau21 > 0.2743922024965286:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt1_dr01 > 3.920615792274475:
                                                            if Q.mass > 34.22078514099121:
                                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 36.3125:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 0.445050910115242:
                                            if Q.pt_7 > 26.890625:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.8322923481464386:
                                                if Q.m012 > 1.6909275650978088:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.35717542469501495:
                                    if s['W'] - s['Z'] > -0.4338178336620331:
                                        if Q.lam2 > 7.749720316496678e-05:
                                            if s['W'] - s['Z'] > -0.2771598994731903:
                                                if Q.dr_0 > 0.0184546560049057:
                                                    if Q.dr_0 > 0.049967437982559204:
                                                        if Q.max_dr > 0.15005365759134293:
                                                            if Q.tau21 > 0.3475246727466583:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['W'] > -2.625810146331787:
                                                                if s['g'] - s['t'] > 0.0652172788977623:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.09402132034301758:
                                                                        if Q.z_dr_0p1_0p2 > 0.29823242127895355:
                                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam2 > 0.00018425541202304885:
                                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 277.453125:
                                                    if Q.dr_0 > 0.022124451585114002:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009664712473750114:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0002661909529706463:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.12146766856312752:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.006836626213043928:
                                                if Q.C2 > 0.018663205206394196:
                                                    if Q.z_dr_0_0p05 > 0.40056878328323364:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > 0.14232294261455536:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006514829816296697:
                                                        if Q.girth2_top3 > 0.0025083954678848386:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['Z'] > -0.7460023760795593:
                                                                if Q.dr_7 > 0.05218156427145004:
                                                                    return 'W'   # 47% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.004558162996545434:
                                                            if Q.tau21 > 0.08954464644193649:
                                                                if Q.z_dr_0p1_0p2 > 0.15610244870185852:
                                                                    if Q.D2 > 0.577288419008255:
                                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > -0.18774700164794922:
                                            if Q.D2 > 3.884416103363037:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['t'] > 0.19409509003162384:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 2.4899595975875854:
                                                if Q.e2 > 0.02024439349770546:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -1.3444228768348694:
                                                        if Q.tau21 > 0.18933672457933426:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 2.751264214515686:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.020084191113710403:
                                                        if s['Z'] - s['t'] > 0.5078878998756409:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['W'] > 1.6739458441734314:
                                                                if Q.dr_5 > 0.05846093408763409:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2_sq > 0.004357384284958243:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.018124514259397984:
                                                                return 'W'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.015336876269429922:
                                        if s['Z'] - s['t'] > 0.20193924754858017:
                                            if s['W'] - s['Z'] > -0.5467869937419891:
                                                if Q.log_sum_pt > 6.236511945724487:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 1.5588109493255615:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07172292098402977:
                                                    if Q.pt_7 > 31.3359375:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04424590803682804:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.m012 > 2.8473392724990845:
                                                        if Q.centroid_offset > 0.017092909663915634:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > -0.16176262497901917:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.1538778766989708:
                            if s['Z'] - s['t'] > -0.030226143077015877:
                                if s['W'] - s['t'] > -0.23443962633609772:
                                    if Q.sum_pt_top3 > 306.765625:
                                        return 'Z'   # 37% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.018696977756917477:
                                        if Q.dr_0 > 0.026590673252940178:
                                            if Q.pt_5 > 30.3984375:
                                                if s['g'] - s['Z'] > -0.1053168848156929:
                                                    return 't'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > -1.6368194818496704:
                                                        if s['W'] - s['t'] > -3.0569772720336914:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.07921140268445015:
                                                            if Q.pt_2 > 58.640625:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > -0.520968496799469:
                                            return 'q'   # 36% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.012354729231446981:
                                    if s['q'] - s['Z'] > -0.8875136375427246:
                                        if Q.girth2_top2 > 0.00275622121989727:
                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.5155692100524902:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 27.2578125:
                                                if Q.mass_over_sum_pt > 0.07650614157319069:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.007177300751209259:
                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.38514213263988495:
                                            return 'W'   # 38% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 70% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -0.42374785244464874:
                                if Q.lam1 > 0.009262274485081434:
                                    if Q.lam2 > 0.00016418188897660002:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.49776217341423035:
                                        return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.8480333387851715:
                                            if Q.max_dr > 0.11575545743107796:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                return 't'   # 96% of the training jets here get this class from the formula
    else:
        if s['g'] - s['t'] > -0.022786367684602737:
            if s['g'] - s['q'] > 0.02146686427295208:
                if s['g'] - s['t'] > 0.3197198808193207:
                    if s['g'] - s['Z'] > -0.5203883349895477:
                        if s['Z'] - s['t'] > -8.89783763885498:
                            if Q.mean_eta2 > 0.0292422566562891:
                                if s['g'] - s['t'] > 0.7371603548526764:
                                    return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mean_eta > 0.0007145332056097686:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['q'] > 0.1329510509967804:
                                    if s['g'] - s['t'] > 0.6239916682243347:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 6.489368915557861:
                                            if Q.mass > 25.42194652557373:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 57% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > 0.92580446600914:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03264234401285648:
                                    if Q.mean_phi2 > 0.0007653492211829871:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 62% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 8.767481327056885:
                                        if s['q'] - s['Z'] > 10.395373344421387:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 81% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > 0.0703195221722126:
                        if s['q'] - s['Z'] > 6.625925302505493:
                            if Q.dr_5 > 0.09546161442995071:
                                if Q.tau32 > 0.3189469426870346:
                                    return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 867.53125:
                                if s['g'] - s['q'] > 1.0520838499069214:
                                    if Q.mass > 100.26227188110352:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.09411533921957016:
                                    if Q.mean_phi2 > 0.014265627600252628:
                                        return 'g'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.05128968693315983:
                                        if s['g'] - s['t'] > 0.14674250781536102:
                                            if Q.centroid_offset > 0.04967140778899193:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0_0p05 > 0.7928476333618164:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.36221475899219513:
                                                if Q.z_dr_0p2_0p4 > 0.0835961252450943:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_3 > 0.10913749411702156:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > 12.700868606567383:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.05740760639309883:
                                                if Q.phi_1 > 0.048187255859375:
                                                    if Q.dr_0 > 0.06670403480529785:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_4 > 0.06745367124676704:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.15296277403831482:
                            if Q.mean_phi2 > 0.013955545146018267:
                                if Q.girth2_top5 > 0.024519536644220352:
                                    return 't'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0016565661644563079:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.04181361757218838:
                                return 't'   # 68% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 65% of the training jets here get this class from the formula
            else:
                if Q.z_dr_0p2_0p4 > 0.10037799179553986:
                    return 'q'   # 52% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['q'] > -0.13088949769735336:
                        if Q.e2 > 0.012999885249882936:
                            return 'q'   # 79% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 58% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 93% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > 0.04093630984425545:
                if s['q'] - s['Z'] > -2.4938712120056152:
                    if s['q'] - s['t'] > 0.06024308130145073:
                        return 'q'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.1494714319705963:
                            if s['Z'] - s['t'] > 0.6386825442314148:
                                return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.dr_7 > 0.03982895240187645:
                                    if s['W'] - s['t'] > -4.350381851196289:
                                        if Q.pt_4 > 52.0625:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0_0p05 > 0.6001070439815521:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 64% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > 0.14360404014587402:
                                return 'Z'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.pt1_dr01 > 15.640034675598145:
                                    return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.08386852219700813:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 55% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.3497075289487839:
                        if s['W'] - s['t'] > -4.96636176109314:
                            return 'Z'   # 97% of the training jets here get this class from the formula
                        else:
                            return 't'   # 51% of the training jets here get this class from the formula
                    else:
                        if Q.LHA > 0.3244035542011261:
                            if Q.log_sum_pt > 6.5043110847473145:
                                if Q.z_top5_slots > 0.8109349310398102:
                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 84% of the training jets here get this class from the formula
                        else:
                            return 't'   # 68% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.11913459002971649:
                    if s['q'] - s['t'] > 0.2640051990747452:
                        return 'q'   # 98% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > 0.0038243656745180488:
                            return 'q'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0_0p05 > 0.6563746631145477:
                                return 'q'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.mean_eta2 > 0.004020522581413388:
                                    if Q.pt_2 > 80.5:
                                        return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 73% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.2852315157651901:
                        if s['g'] - s['t'] > -0.11866417899727821:
                            if Q.centroid_offset > 0.05877924896776676:
                                if Q.z_6 > 0.06738267093896866:
                                    if Q.mean_eta2 > 0.009644593577831984:
                                        if Q.dr_1 > 0.14838581532239914:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0831703282892704:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.0757373720407486:
                                    if Q.z_dr_0p2_0p4 > 0.06818601489067078:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 65% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -4.634319067001343:
                                        if Q.z_dr_0p1_0p2 > 0.3989965468645096:
                                            if Q.e2 > 0.05461944080889225:
                                                if s['q'] - s['Z'] > 4.379261493682861:
                                                    if Q.dr_0 > 0.11589239910244942:
                                                        return 't'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.8634369075298309:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.05478145182132721:
                                if Q.pt_2 > 61.203125:
                                    if s['W'] - s['t'] > -11.484848976135254:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 0.0074498027097433805:
                                            return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 25.34375:
                                        if s['W'] - s['Z'] > 0.8733616471290588:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_phi > 0.048924317583441734:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.3966374546289444:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 344.625:
                                    return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mean_eta2 > 0.015987010672688484:
                                        return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.3330262005329132:
                            if Q.z_6 > 0.027682782150804996:
                                return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.m012 > 3.5703446865081787:
                                    return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 69% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.5124592781066895:
                                if Q.centroid_offset > 0.06486207619309425:
                                    return 't'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 5.888444900512695:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.21875:
                                            return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_eta2 > 0.007119073532521725:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > -0.0789390504360199:
                                    if Q.max_dr > 0.13518434762954712:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 419.640625:
                                            if s['g'] - s['t'] > -1.7858107686042786:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 100% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 0
    print('class:', classify(pt, eta, phi))
