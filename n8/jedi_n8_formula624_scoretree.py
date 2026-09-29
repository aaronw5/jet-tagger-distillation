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

Test set (50,000 jets): accuracy 65.14% (the formula: 65.58%); same class as the formula for 93.20% of jets.  54 leaves, depth 10.
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
    return (-0.2444
        + 60.68 * max(0.0, 0.004372139331 - Q.width)
        - 14.75 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.002677 * max(0.0, 64.618731689453 - Q.mass)
        - 0.02815 * max(0.0, 21.784077072144 - Q.mass)
        - 64.0 * max(0.0, 0.013238675334 - Q.girth2)
        - 0.003247 * max(0.0, Q.sum_pt - 901.59375)
        + 0.003725 * max(0.0, 56.920347213745 - Q.mass)
        - 117.0 * max(0.0, 0.008678044951 - Q.width)
        + 0.002151 * max(0.0, 29.644699859619 - Q.mass)
        + 8.807 * max(0.0, 0.087236513197 - Q.girth)
        + 1473.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 0.002067 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 244.9 * max(0.0, 0.00543336053 - Q.lam1)
        + 14.16 * max(0.0, 0.076081777364 - Q.girth)
        + 115.8 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 269.0 * max(0.0, 0.008375572068 - Q.lam1)
        - 85.72 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 612.0 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.07866 * max(0.0, Q.sum_pt - 813.415625)
        + 2.781 * max(0.0, Q.log_sum_pt - 6.377722943814)
        + 15.43 * max(0.0, 0.055577157257 - Q.z_7)
        + 4.602 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.01197 * max(0.0, 53.332374954224 - Q.mass)
        - 6.232 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 612.7 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 174.1 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.01779 * max(0.0, 49.668099212646 - Q.mass)
        - 11.81 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.003325 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 0.01868 * max(0.0, 69.611351776123 - Q.mass)
        + 151.6 * max(0.0, 0.005954149834 - Q.lam1)
        + 13.38 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 3.773 * max(0.0, Q.LHA - 0.111565049159)
        + 0.009447 * max(0.0, 36.229410171509 - Q.mass)
        + 7.876 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        - 0.07529 * max(0.0, 53.4375 - Q.pt_7)
        + 0.006537 * max(0.0, 788.4484375 - Q.sum_pt)
        + 70.81 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 57.02 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 17.12 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 9.449 * max(0.0, Q.width - 0.018827653081)
        - 0.03501 * max(0.0, Q.mass - 36.229410171509)
        + 19.63 * max(0.0, Q.e2 - 0.028531698044)
        + 20.69 * max(0.0, 0.006679471358 - Q.width)
        - 16.57 * max(0.0, 0.04447356835 - Q.e2)
        + 241.6 * max(0.0, 0.007330079875 - Q.lam1)
        + 1.341 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.02535 * max(0.0, Q.mass - 69.611351776123)
        - 117.0 * max(0.0, 0.008678044751 - Q.girth2)
        + 0.1987 * max(0.0, Q.LHA - 0.325582223496)
        - 1.002 * max(0.0, 0.038466955721 - Q.e2)
        - 147.5 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 60.68 * max(0.0, 0.004372139461 - Q.girth2)
        + 46.16 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        - 159.1 * max(0.0, Q.lam1 - 0.012003726523)
        + 0.1368 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 87.04 * max(0.0, Q.lam1 - 0.008375572068)
        - 132.1 * max(0.0, Q.lam1 - 0.004183811014)
        - 123.5 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 8.658 * max(0.0, Q.e2 - 0.050284641981)
        + 489.9 * max(0.0, Q.width - 0.000319370692)
        + 0.003916 * max(0.0, 763.825 - Q.sum_pt)
        + 3491.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 194.3 * max(0.0, Q.width - 0.001653836415)
        + 0.002227 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 2.781 * max(0.0, Q.girth - 0.101940929517)
        + 410.5 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 34.8 * max(0.0, Q.girth2 - 0.013238675334)
        - 8.29 * max(0.0, Q.e2 - 0.007078157854)
        + 1.348 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 350.1 * max(0.0, 0.023780909279 - Q.e2_sq)
        - 63.79 * max(0.0, Q.girth2 - 0.007520088344)
        - 515.9 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 69.03 * max(0.0, Q.width - 0.002635417778)
        - 4407.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        - 17.25 * max(0.0, Q.e2 - 0.020459658932)
        + 137.4 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.009538 * max(0.0, 41.377904891968 - Q.mass)
        - 818.1 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        - 0.03254 * max(0.0, 80.4 - Q.mass)
        + 8.33 * max(0.0, 0.216055863061 - Q.LHA)
        - 10850.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 13.68 * max(0.0, 0.035560912266 - Q.e2)
        + 29.68 * max(0.0, 0.002635417778 - Q.width)
        + 3.412 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.0008843 * max(0.0, Q.sum_pt - 868.509375)
        + 1.393 * max(0.0, 0.154689112391 - Q.LHA)
        + 320.7 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        - 252.1 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        + 0.4998 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 69.71 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 0.003941 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 64.01 * max(0.0, 0.013238675006 - Q.width)
        - 84.7 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 13.62 * max(0.0, 0.050284641981 - Q.e2)
        - 7.503 * max(0.0, Q.lam2 - 0.000537286005)
        - 8.156 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 24.33 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 4.801 * max(0.0, Q.LHA - 0.312727471086)
        + 0.518 * Q.max_dr
        - 10.31 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 20.47 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 8.132 * max(0.0, Q.girth - 0.087236513197)
        - 22.78 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.149 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 14.23 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 0.07561 * max(0.0, Q.sum_pt - 988.4078125)
        + 65.11 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        + 103.4 * max(0.0, Q.girth2 - 0.004372139461)
        - 8.042 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 194.3 * max(0.0, Q.girth2 - 0.0016538364)
        - 44.04 * max(0.0, 0.024547699839 - Q.e2)
        - 3.715 * max(0.0, 0.04081947431 - Q.girth)
        - 28.58 * max(0.0, 0.005590288644 - Q.width)
        - 94.9 * max(0.0, Q.girth2 - 0.008678044751)
        + 120.1 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 0.03524 * max(0.0, Q.mass - 80.4)
        - 126.8 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 22.71 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 0.8463 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 49.89 * max(0.0, 0.002464291268 - Q.lam1)
        + 0.9477 * max(0.0, 0.15984864831 - Q.max_dr)
        + 1.097 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 0.1015 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 7.78 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 3.6 * max(0.0, Q.e2 - 0.016554418951)
        + 3.349 * max(0.0, 0.067292226106 - Q.C2)
        + 200.9 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 20.7 * max(0.0, 0.006679471442 - Q.girth2)
        + 370.2 * max(0.0, 0.005019718802 - Q.width)
        + 5570.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 2.773 * max(0.0, 0.196739721581 - Q.LHA)
        + 13.23 * max(0.0, 0.061086014472 - Q.girth)
        + 0.0008773 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 2.658 * max(0.0, 0.177304983139 - Q.max_dr)
        - 1148.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 57.37 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 14020.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 3525.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 581.2 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 2.154 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 745.5 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 2.75 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 12300.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 63.51 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 2944.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 329.2 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 727.2 * max(0.0, 0.000964142894 - Q.girth2)
        - 12380.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        + 3571.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 1430.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 175.3 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 9.405 * max(0.0, 0.054649224505 - Q.girth)
        + 143.5 * max(0.0, Q.lam2 - 0.001130644719)
        + 0.7047 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 322.5 * max(0.0, 0.006096650059 - Q.width)
        - 16.38 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 9.445 * max(0.0, Q.girth2 - 0.018827652745)
        - 10.37 * max(0.0, 0.032346998155 - Q.e2)
        + 0.04331 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        - 1.093 * max(0.0, 0.221586732566 - Q.max_dr)
        - 12670.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 82.77 * max(0.0, 0.007520088344 - Q.girth2)
        + 141.7 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 0.002047 * max(0.0, Q.sum_pt - 840.01953125)
        - 1.472 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 2919.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 4452.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        - 5.65 * Q.e2
        - 1123.0 * max(0.0, Q.lam2 - 0.000194798295)
        - 1.725 * max(0.0, Q.LHA - 0.303313749495)
        + 72.47 * max(0.0, Q.centroid_offset - 0.00231612516)
        + 222.5 * max(0.0, 0.004183811014 - Q.lam1)
        - 6.936 * max(0.0, Q.centroid_offset - 0.037760993714)
        + 19.62 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 0.1876 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 44.1 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 1246.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.002044 * max(0.0, Q.mass - 15.454033088684)
        - 0.01531 * max(0.0, Q.mass - 53.332374954224)
        + 182.7 * max(0.0, 0.0016538364 - Q.girth2)
        - 2.07 * max(0.0, Q.LHA - 0.346713497427)
        + 193.5 * max(0.0, 0.003408388935 - Q.lam2)
        + 12.46 * max(0.0, Q.girth - 0.076081777364)
        + 163.6 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 45.46 * Q.centroid_offset
        + 88.79 * max(0.0, 0.003562611091 - Q.width)
        - 0.003871 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 25.26 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 8.551 * max(0.0, 0.148408418149 - Q.girth)
        + 10.03 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 10.81 * max(0.0, 0.016433749775 - Q.lam1)
        - 8.14 * max(0.0, 0.028070914944 - Q.z_7)
        - 1170.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 40.66 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 2.887 * max(0.0, Q.LHA - 0.09323897448)
        - 82.77 * max(0.0, 0.00752008842 - Q.width)
        + 71.57 * max(0.0, 0.006506575659 - Q.lam1)
        - 111.0 * max(0.0, Q.lam1 - 0.00543336053)
        - 112.8 * max(0.0, Q.lam1 - 0.007330079875)
        - 203.5 * max(0.0, Q.lam1 - 0.005954149834)
        - 279.7 * max(0.0, Q.lam1 - 0.002464291268)
        - 9.556 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 6.444 * max(0.0, 0.303313749495 - Q.LHA)
        + 4.535 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 2.884 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.00458 * max(0.0, 86.4 - Q.mass)
        + 3944.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 973.3 * max(0.0, 0.005834489329 - Q.e2_sq)
        - 2.273 * max(0.0, 0.041109715588 - Q.e2)
        - 40.05 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 55.21 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        + 1.459 * max(0.0, Q.centroid_offset - 0.012587644117)
        + 12240.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 198.6 * max(0.0, 0.012003726523 - Q.lam1)
        + 91.43 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 172.9 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 3.98 * max(0.0, 0.063441075385 - Q.e2)
        - 0.4103 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        - 504.6 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 59.33 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 13.03 * max(0.0, Q.girth - 0.033604209498)
    )


def score_q(Q):
    return (0.09491
        + 277.7 * max(0.0, 0.004372139331 - Q.width)
        + 437.4 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.00099 * max(0.0, 64.618731689453 - Q.mass)
        + 0.002602 * max(0.0, 21.784077072144 - Q.mass)
        + 178.8 * max(0.0, 0.013238675334 - Q.girth2)
        - 0.003311 * max(0.0, Q.sum_pt - 901.59375)
        - 0.00213 * max(0.0, 56.920347213745 - Q.mass)
        + 116.9 * max(0.0, 0.008678044951 - Q.width)
        - 0.008934 * max(0.0, 29.644699859619 - Q.mass)
        + 6.403 * max(0.0, 0.087236513197 - Q.girth)
        + 2642.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 0.001632 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 123.7 * max(0.0, 0.00543336053 - Q.lam1)
        - 2.138 * max(0.0, 0.076081777364 - Q.girth)
        + 147.6 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 26.9 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.7314 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 267.3 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.06284 * max(0.0, Q.sum_pt - 813.415625)
        - 0.09464 * max(0.0, Q.log_sum_pt - 6.377722943814)
        + 16.7 * max(0.0, 0.055577157257 - Q.z_7)
        + 112.6 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.1376 * max(0.0, 53.332374954224 - Q.mass)
        - 7.234 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 624.5 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 343.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.009017 * max(0.0, 49.668099212646 - Q.mass)
        - 9.208 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.0001744 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 0.2026 * max(0.0, 69.611351776123 - Q.mass)
        - 128.8 * max(0.0, 0.005954149834 - Q.lam1)
        + 8.52 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 9.708 * max(0.0, Q.LHA - 0.111565049159)
        + 0.509 * max(0.0, 36.229410171509 - Q.mass)
        + 7.02 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        - 0.02588 * max(0.0, 53.4375 - Q.pt_7)
        - 0.000707 * max(0.0, 788.4484375 - Q.sum_pt)
        + 59.56 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 106.6 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 32.53 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 217.4 * max(0.0, Q.width - 0.018827653081)
        - 0.5089 * max(0.0, Q.mass - 36.229410171509)
        + 13.15 * max(0.0, Q.e2 - 0.028531698044)
        + 10.94 * max(0.0, 0.006679471358 - Q.width)
        - 5.624 * max(0.0, 0.04447356835 - Q.e2)
        - 119.6 * max(0.0, 0.007330079875 - Q.lam1)
        + 1.239 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.2085 * max(0.0, Q.mass - 69.611351776123)
        + 116.9 * max(0.0, 0.008678044751 - Q.girth2)
        - 2.356 * max(0.0, Q.LHA - 0.325582223496)
        + 0.03578 * max(0.0, 0.038466955721 - Q.e2)
        - 332.2 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 277.7 * max(0.0, 0.004372139461 - Q.girth2)
        + 51.09 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 144.8 * max(0.0, Q.lam1 - 0.012003726523)
        + 0.3399 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        + 219.1 * max(0.0, Q.lam1 - 0.008375572068)
        + 214.3 * max(0.0, Q.lam1 - 0.004183811014)
        - 257.6 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 21.1 * max(0.0, Q.e2 - 0.050284641981)
        + 1990.0 * max(0.0, Q.width - 0.000319370692)
        + 0.002652 * max(0.0, 763.825 - Q.sum_pt)
        + 2805.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 142.6 * max(0.0, Q.width - 0.001653836415)
        - 0.0003599 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 0.4841 * max(0.0, Q.girth - 0.101940929517)
        + 452.5 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 467.5 * max(0.0, Q.girth2 - 0.013238675334)
        - 64.52 * max(0.0, Q.e2 - 0.007078157854)
        + 7.226 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 404.8 * max(0.0, 0.023780909279 - Q.e2_sq)
        - 444.4 * max(0.0, Q.girth2 - 0.007520088344)
        - 575.8 * max(0.0, 0.007182789718 - Q.e2_sq)
        - 226.1 * max(0.0, Q.width - 0.002635417778)
        - 4718.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        - 5.939 * max(0.0, Q.e2 - 0.020459658932)
        - 229.6 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.01799 * max(0.0, 41.377904891968 - Q.mass)
        - 965.9 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        - 0.4352 * max(0.0, 80.4 - Q.mass)
        + 8.383 * max(0.0, 0.216055863061 - Q.LHA)
        - 9044.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 1.726 * max(0.0, 0.035560912266 - Q.e2)
        + 397.4 * max(0.0, 0.002635417778 - Q.width)
        + 27.99 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.002043 * max(0.0, Q.sum_pt - 868.509375)
        - 6.417 * max(0.0, 0.154689112391 - Q.LHA)
        - 75.79 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 99.83 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 7.41 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 155.0 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 0.001333 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 178.8 * max(0.0, 0.013238675006 - Q.width)
        - 94.69 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 4.299 * max(0.0, 0.050284641981 - Q.e2)
        + 142.5 * max(0.0, Q.lam2 - 0.000537286005)
        - 27.23 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 17.03 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 3.589 * max(0.0, Q.LHA - 0.312727471086)
        + 0.4897 * Q.max_dr
        + 33.78 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 12.87 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 12.2 * max(0.0, Q.girth - 0.087236513197)
        - 27.3 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.08882 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 31.06 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 0.05507 * max(0.0, Q.sum_pt - 988.4078125)
        + 213.8 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 369.1 * max(0.0, Q.girth2 - 0.004372139461)
        - 6.327 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 142.6 * max(0.0, Q.girth2 - 0.0016538364)
        - 14.0 * max(0.0, 0.024547699839 - Q.e2)
        - 25.28 * max(0.0, 0.04081947431 - Q.girth)
        + 52.81 * max(0.0, 0.005590288644 - Q.width)
        - 529.0 * max(0.0, Q.girth2 - 0.008678044751)
        + 235.7 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 0.4258 * max(0.0, Q.mass - 80.4)
        - 867.7 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 27.83 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 0.9806 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        - 257.3 * max(0.0, 0.002464291268 - Q.lam1)
        + 1.027 * max(0.0, 0.15984864831 - Q.max_dr)
        + 1.183 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 0.1063 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 24.74 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 2.792 * max(0.0, Q.e2 - 0.016554418951)
        - 3.465 * max(0.0, 0.067292226106 - Q.C2)
        + 101.6 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 10.94 * max(0.0, 0.006679471442 - Q.girth2)
        + 408.6 * max(0.0, 0.005019718802 - Q.width)
        + 6260.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 3.794 * max(0.0, 0.196739721581 - Q.LHA)
        - 32.0 * max(0.0, 0.061086014472 - Q.girth)
        + 0.0008752 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 5.75 * max(0.0, 0.177304983139 - Q.max_dr)
        - 1511.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 61.65 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 16090.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 773.5 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 41.08 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 0.7317 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 518.8 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 1.678 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 5019.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 485.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 4632.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 7056.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 692.4 * max(0.0, 0.000964142894 - Q.girth2)
        - 6425.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 543.6 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 658.7 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        + 95.59 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 14.06 * max(0.0, 0.054649224505 - Q.girth)
        + 258.6 * max(0.0, Q.lam2 - 0.001130644719)
        + 1.635 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 509.8 * max(0.0, 0.006096650059 - Q.width)
        - 16.82 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 217.4 * max(0.0, Q.girth2 - 0.018827652745)
        - 6.687 * max(0.0, 0.032346998155 - Q.e2)
        + 0.05485 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        - 2.951 * max(0.0, 0.221586732566 - Q.max_dr)
        - 17740.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 204.3 * max(0.0, 0.007520088344 - Q.girth2)
        + 182.4 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 0.002763 * max(0.0, Q.sum_pt - 840.01953125)
        - 2.221 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 4003.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 4852.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        + 28.65 * Q.e2
        + 687.7 * max(0.0, Q.lam2 - 0.000194798295)
        - 5.14 * max(0.0, Q.LHA - 0.303313749495)
        + 38.18 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 22.86 * max(0.0, 0.004183811014 - Q.lam1)
        + 10.76 * max(0.0, Q.centroid_offset - 0.037760993714)
        + 36.55 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 2.028 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 33.45 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 344.1 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.01331 * max(0.0, Q.mass - 15.454033088684)
        - 0.1508 * max(0.0, Q.mass - 53.332374954224)
        + 527.1 * max(0.0, 0.0016538364 - Q.girth2)
        + 1.437 * max(0.0, Q.LHA - 0.346713497427)
        + 339.1 * max(0.0, 0.003408388935 - Q.lam2)
        + 3.748 * max(0.0, Q.girth - 0.076081777364)
        + 303.0 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 14.55 * Q.centroid_offset
        + 420.3 * max(0.0, 0.003562611091 - Q.width)
        - 0.003161 * max(0.0, 687.4375 - Q.sum_pt_top5)
        + 5.615 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.393 * max(0.0, 0.148408418149 - Q.girth)
        - 1.557 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 7.197 * max(0.0, 0.016433749775 - Q.lam1)
        + 15.39 * max(0.0, 0.028070914944 - Q.z_7)
        - 640.5 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 18.59 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 1.529 * max(0.0, Q.LHA - 0.09323897448)
        + 204.3 * max(0.0, 0.00752008842 - Q.width)
        + 81.24 * max(0.0, 0.006506575659 - Q.lam1)
        + 111.0 * max(0.0, Q.lam1 - 0.00543336053)
        + 123.4 * max(0.0, Q.lam1 - 0.007330079875)
        + 107.0 * max(0.0, Q.lam1 - 0.005954149834)
        + 47.06 * max(0.0, Q.lam1 - 0.002464291268)
        + 245.8 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 2.656 * max(0.0, 0.303313749495 - Q.LHA)
        + 13.11 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 5.303 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.01079 * max(0.0, 86.4 - Q.mass)
        + 3721.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 1290.0 * max(0.0, 0.005834489329 - Q.e2_sq)
        + 7.028 * max(0.0, 0.041109715588 - Q.e2)
        - 78.96 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 88.48 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 10.63 * max(0.0, Q.centroid_offset - 0.012587644117)
        + 9817.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 104.2 * max(0.0, 0.012003726523 - Q.lam1)
        + 102.5 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 911.8 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 7.433 * max(0.0, 0.063441075385 - Q.e2)
        - 0.2553 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 1836.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 100.6 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 17.18 * max(0.0, Q.girth - 0.033604209498)
    )


def score_W(Q):
    return (0.7577
        - 456.8 * max(0.0, 0.004372139331 - Q.width)
        - 283.5 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.0358 * max(0.0, 64.618731689453 - Q.mass)
        + 0.1439 * max(0.0, 21.784077072144 - Q.mass)
        - 113.3 * max(0.0, 0.013238675334 - Q.girth2)
        - 0.01404 * max(0.0, Q.sum_pt - 901.59375)
        - 0.04309 * max(0.0, 56.920347213745 - Q.mass)
        + 226.3 * max(0.0, 0.008678044951 - Q.width)
        - 0.05565 * max(0.0, 29.644699859619 - Q.mass)
        + 69.73 * max(0.0, 0.087236513197 - Q.girth)
        - 6380.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 0.003074 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 166.0 * max(0.0, 0.00543336053 - Q.lam1)
        - 60.89 * max(0.0, 0.076081777364 - Q.girth)
        - 309.2 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 392.8 * max(0.0, 0.008375572068 - Q.lam1)
        - 71.08 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 1006.0 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.3684 * max(0.0, Q.sum_pt - 813.415625)
        - 0.5986 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 27.74 * max(0.0, 0.055577157257 - Q.z_7)
        - 56.06 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.1939 * max(0.0, 53.332374954224 - Q.mass)
        - 54.99 * max(0.0, Q.log_sum_pt - 6.572937922293)
        + 602.1 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 2352.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.02998 * max(0.0, 49.668099212646 - Q.mass)
        - 48.05 * max(0.0, Q.log_sum_pt - 6.638338705138)
        + 0.001084 * max(0.0, Q.sum_pt_top5 - 579.875)
        + 0.3558 * max(0.0, 69.611351776123 - Q.mass)
        - 276.7 * max(0.0, 0.005954149834 - Q.lam1)
        + 49.9 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 0.1401 * max(0.0, Q.LHA - 0.111565049159)
        - 0.8014 * max(0.0, 36.229410171509 - Q.mass)
        + 55.92 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 0.01746 * max(0.0, 53.4375 - Q.pt_7)
        + 0.009168 * max(0.0, 788.4484375 - Q.sum_pt)
        + 294.2 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 192.5 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 7.029 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 28.99 * max(0.0, Q.width - 0.018827653081)
        + 0.8032 * max(0.0, Q.mass - 36.229410171509)
        - 48.07 * max(0.0, Q.e2 - 0.028531698044)
        + 303.4 * max(0.0, 0.006679471358 - Q.width)
        - 71.42 * max(0.0, 0.04447356835 - Q.e2)
        - 406.6 * max(0.0, 0.007330079875 - Q.lam1)
        - 13.99 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.3559 * max(0.0, Q.mass - 69.611351776123)
        + 226.2 * max(0.0, 0.008678044751 - Q.girth2)
        - 20.95 * max(0.0, Q.LHA - 0.325582223496)
        + 40.27 * max(0.0, 0.038466955721 - Q.e2)
        + 1058.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 456.8 * max(0.0, 0.004372139461 - Q.girth2)
        - 49.4 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 353.2 * max(0.0, Q.lam1 - 0.012003726523)
        - 3.755 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        + 159.3 * max(0.0, Q.lam1 - 0.008375572068)
        + 308.2 * max(0.0, Q.lam1 - 0.004183811014)
        + 33.79 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        - 97.65 * max(0.0, Q.e2 - 0.050284641981)
        - 2931.0 * max(0.0, Q.width - 0.000319370692)
        - 0.01151 * max(0.0, 763.825 - Q.sum_pt)
        - 6412.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 76.53 * max(0.0, Q.width - 0.001653836415)
        + 0.002836 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 79.41 * max(0.0, Q.girth - 0.101940929517)
        - 1141.0 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 186.9 * max(0.0, Q.girth2 - 0.013238675334)
        + 102.0 * max(0.0, Q.e2 - 0.007078157854)
        + 10.12 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 1140.0 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 984.5 * max(0.0, Q.girth2 - 0.007520088344)
        - 265.7 * max(0.0, 0.007182789718 - Q.e2_sq)
        - 288.8 * max(0.0, Q.width - 0.002635417778)
        - 1580.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        + 83.29 * max(0.0, Q.e2 - 0.020459658932)
        - 197.9 * max(0.0, Q.girth2 - 0.003562611155)
        - 0.01265 * max(0.0, 41.377904891968 - Q.mass)
        + 2157.0 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        + 0.7282 * max(0.0, 80.4 - Q.mass)
        - 23.2 * max(0.0, 0.216055863061 - Q.LHA)
        + 17430.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 21.66 * max(0.0, 0.035560912266 - Q.e2)
        - 565.9 * max(0.0, 0.002635417778 - Q.width)
        - 636.3 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.01277 * max(0.0, Q.sum_pt - 868.509375)
        + 8.077 * max(0.0, 0.154689112391 - Q.LHA)
        - 181.6 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 226.1 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 10.58 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 355.9 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 0.01496 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 113.2 * max(0.0, 0.013238675006 - Q.width)
        + 300.7 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 20.61 * max(0.0, 0.050284641981 - Q.e2)
        + 258.9 * max(0.0, Q.lam2 - 0.000537286005)
        + 70.01 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 100.7 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 48.81 * max(0.0, Q.LHA - 0.312727471086)
        - 4.524 * Q.max_dr
        + 37.59 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 140.3 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 51.12 * max(0.0, Q.girth - 0.087236513197)
        + 110.4 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.1261 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 69.01 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 0.3277 * max(0.0, Q.sum_pt - 988.4078125)
        - 708.8 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 146.2 * max(0.0, Q.girth2 - 0.004372139461)
        + 70.86 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 76.53 * max(0.0, Q.girth2 - 0.0016538364)
        + 264.5 * max(0.0, 0.024547699839 - Q.e2)
        + 10.43 * max(0.0, 0.04081947431 - Q.girth)
        - 1573.0 * max(0.0, 0.005590288644 - Q.width)
        + 494.5 * max(0.0, Q.girth2 - 0.008678044751)
        - 1045.0 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.6227 * max(0.0, Q.mass - 80.4)
        + 2037.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        + 69.36 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 3.564 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        + 205.8 * max(0.0, 0.002464291268 - Q.lam1)
        - 8.64 * max(0.0, 0.15984864831 - Q.max_dr)
        - 1.46 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 0.1925 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        + 266.2 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 39.35 * max(0.0, Q.e2 - 0.016554418951)
        - 16.86 * max(0.0, 0.067292226106 - Q.C2)
        - 112.6 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 303.4 * max(0.0, 0.006679471442 - Q.girth2)
        - 1603.0 * max(0.0, 0.005019718802 - Q.width)
        - 15090.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 23.73 * max(0.0, 0.196739721581 - Q.LHA)
        - 108.3 * max(0.0, 0.061086014472 - Q.girth)
        - 0.003287 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 7.475 * max(0.0, 0.177304983139 - Q.max_dr)
        + 1324.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 340.5 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 51980.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 42800.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 321.8 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 7.448 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 1668.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 8.0 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 74780.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 3900.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 13920.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 34100.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 1594.0 * max(0.0, 0.000964142894 - Q.girth2)
        + 44650.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 27120.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        + 2961.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 74.52 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 16.83 * max(0.0, 0.054649224505 - Q.girth)
        + 87.45 * max(0.0, Q.lam2 - 0.001130644719)
        - 3.035 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 44.02 * max(0.0, 0.006096650059 - Q.width)
        + 40.69 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 29.01 * max(0.0, Q.girth2 - 0.018827652745)
        - 53.71 * max(0.0, 0.032346998155 - Q.e2)
        - 0.02681 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 13.22 * max(0.0, 0.221586732566 - Q.max_dr)
        + 23990.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 723.8 * max(0.0, 0.007520088344 - Q.girth2)
        - 32.66 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 0.01396 * max(0.0, Q.sum_pt - 840.01953125)
        + 1.822 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 5463.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 1624.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        - 76.06 * Q.e2
        + 2127.0 * max(0.0, Q.lam2 - 0.000194798295)
        + 65.68 * max(0.0, Q.LHA - 0.303313749495)
        - 126.9 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 222.7 * max(0.0, 0.004183811014 - Q.lam1)
        - 61.2 * max(0.0, Q.centroid_offset - 0.037760993714)
        - 189.5 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 2.85 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 204.8 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 877.1 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.006724 * max(0.0, Q.mass - 15.454033088684)
        + 0.2441 * max(0.0, Q.mass - 53.332374954224)
        - 363.6 * max(0.0, 0.0016538364 - Q.girth2)
        - 0.2959 * max(0.0, Q.LHA - 0.346713497427)
        - 598.6 * max(0.0, 0.003408388935 - Q.lam2)
        - 84.51 * max(0.0, Q.girth - 0.076081777364)
        - 585.5 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 56.88 * Q.centroid_offset
        - 527.2 * max(0.0, 0.003562611091 - Q.width)
        + 0.00222 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 34.06 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 47.92 * max(0.0, 0.148408418149 - Q.girth)
        - 24.23 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 259.7 * max(0.0, 0.016433749775 - Q.lam1)
        - 2.289 * max(0.0, 0.028070914944 - Q.z_7)
        + 5957.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 76.57 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 7.787 * max(0.0, Q.LHA - 0.09323897448)
        + 723.9 * max(0.0, 0.00752008842 - Q.width)
        - 618.5 * max(0.0, 0.006506575659 - Q.lam1)
        + 363.9 * max(0.0, Q.lam1 - 0.00543336053)
        + 138.7 * max(0.0, Q.lam1 - 0.007330079875)
        + 264.7 * max(0.0, Q.lam1 - 0.005954149834)
        + 735.1 * max(0.0, Q.lam1 - 0.002464291268)
        - 1512.0 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 25.69 * max(0.0, 0.303313749495 - Q.LHA)
        - 292.6 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 57.25 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.03942 * max(0.0, 86.4 - Q.mass)
        - 5134.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        + 40.07 * max(0.0, 0.005834489329 - Q.e2_sq)
        - 112.8 * max(0.0, 0.041109715588 - Q.e2)
        + 271.4 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        - 290.8 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 64.18 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 104300.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 206.6 * max(0.0, 0.012003726523 - Q.lam1)
        + 251.7 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 9973.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.2393 * max(0.0, 0.063441075385 - Q.e2)
        + 3.767 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 6271.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 219.6 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 34.18 * max(0.0, Q.girth - 0.033604209498)
    )


def score_Z(Q):
    return (0.145
        - 220.8 * max(0.0, 0.004372139331 - Q.width)
        - 310.1 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.005581 * max(0.0, 64.618731689453 - Q.mass)
        - 0.006362 * max(0.0, 21.784077072144 - Q.mass)
        + 308.5 * max(0.0, 0.013238675334 - Q.girth2)
        - 0.009118 * max(0.0, Q.sum_pt - 901.59375)
        - 0.01203 * max(0.0, 56.920347213745 - Q.mass)
        + 378.9 * max(0.0, 0.008678044951 - Q.width)
        + 0.0697 * max(0.0, 29.644699859619 - Q.mass)
        - 26.37 * max(0.0, 0.087236513197 - Q.girth)
        - 22600.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 0.0001298 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 152.2 * max(0.0, 0.00543336053 - Q.lam1)
        + 82.8 * max(0.0, 0.076081777364 - Q.girth)
        - 449.1 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 682.5 * max(0.0, 0.008375572068 - Q.lam1)
        + 60.83 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 371.6 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.3275 * max(0.0, Q.sum_pt - 813.415625)
        + 0.6512 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 23.51 * max(0.0, 0.055577157257 - Q.z_7)
        - 172.0 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.2644 * max(0.0, 53.332374954224 - Q.mass)
        - 50.67 * max(0.0, Q.log_sum_pt - 6.572937922293)
        + 1265.0 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 2071.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.001774 * max(0.0, 49.668099212646 - Q.mass)
        - 43.74 * max(0.0, Q.log_sum_pt - 6.638338705138)
        + 0.001532 * max(0.0, Q.sum_pt_top5 - 579.875)
        + 0.3668 * max(0.0, 69.611351776123 - Q.mass)
        - 55.53 * max(0.0, 0.005954149834 - Q.lam1)
        + 45.01 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 3.965 * max(0.0, Q.LHA - 0.111565049159)
        - 0.9378 * max(0.0, 36.229410171509 - Q.mass)
        + 51.83 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        + 0.008439 * max(0.0, 53.4375 - Q.pt_7)
        + 0.01071 * max(0.0, 788.4484375 - Q.sum_pt)
        + 270.9 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 632.0 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 5.555 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 95.67 * max(0.0, Q.width - 0.018827653081)
        + 0.9513 * max(0.0, Q.mass - 36.229410171509)
        - 109.3 * max(0.0, Q.e2 - 0.028531698044)
        - 636.0 * max(0.0, 0.006679471358 - Q.width)
        + 41.22 * max(0.0, 0.04447356835 - Q.e2)
        - 60.47 * max(0.0, 0.007330079875 - Q.lam1)
        - 5.13 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.3757 * max(0.0, Q.mass - 69.611351776123)
        + 378.9 * max(0.0, 0.008678044751 - Q.girth2)
        + 9.792 * max(0.0, Q.LHA - 0.325582223496)
        + 29.07 * max(0.0, 0.038466955721 - Q.e2)
        + 3515.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 220.8 * max(0.0, 0.004372139461 - Q.girth2)
        - 302.3 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 79.12 * max(0.0, Q.lam1 - 0.012003726523)
        - 0.9657 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 480.5 * max(0.0, Q.lam1 - 0.008375572068)
        + 134.9 * max(0.0, Q.lam1 - 0.004183811014)
        + 181.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        - 8.51 * max(0.0, Q.e2 - 0.050284641981)
        - 2073.0 * max(0.0, Q.width - 0.000319370692)
        - 0.01661 * max(0.0, 763.825 - Q.sum_pt)
        - 3045.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 21.51 * max(0.0, Q.width - 0.001653836415)
        + 0.00719 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 77.87 * max(0.0, Q.girth - 0.101940929517)
        - 1423.0 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 691.6 * max(0.0, Q.girth2 - 0.013238675334)
        + 34.12 * max(0.0, Q.e2 - 0.007078157854)
        - 31.32 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 1405.0 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 28.29 * max(0.0, Q.girth2 - 0.007520088344)
        + 1007.0 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 176.5 * max(0.0, Q.width - 0.002635417778)
        + 798.2 * max(0.0, 0.003013300392 - Q.e2_sq)
        + 34.75 * max(0.0, Q.e2 - 0.020459658932)
        + 259.0 * max(0.0, Q.girth2 - 0.003562611155)
        - 0.02536 * max(0.0, 41.377904891968 - Q.mass)
        + 3028.0 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        + 0.7838 * max(0.0, 80.4 - Q.mass)
        - 8.907 * max(0.0, 0.216055863061 - Q.LHA)
        + 10270.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 39.76 * max(0.0, 0.035560912266 - Q.e2)
        - 162.1 * max(0.0, 0.002635417778 - Q.width)
        - 1345.0 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 0.01116 * max(0.0, Q.sum_pt - 868.509375)
        + 13.95 * max(0.0, 0.154689112391 - Q.LHA)
        - 111.2 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 392.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 11.73 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 932.1 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        + 0.008705 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 308.6 * max(0.0, 0.013238675006 - Q.width)
        + 386.1 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 100.6 * max(0.0, 0.050284641981 - Q.e2)
        - 621.7 * max(0.0, Q.lam2 - 0.000537286005)
        - 9.395 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 89.44 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 27.34 * max(0.0, Q.LHA - 0.312727471086)
        - 6.21 * Q.max_dr
        - 24.41 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 45.04 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 111.5 * max(0.0, Q.girth - 0.087236513197)
        + 43.27 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.1043 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 27.84 * max(0.0, 0.024419631481 - Q.girth2_top5)
        - 0.2943 * max(0.0, Q.sum_pt - 988.4078125)
        - 1067.0 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        + 133.6 * max(0.0, Q.girth2 - 0.004372139461)
        + 6.288 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 21.51 * max(0.0, Q.girth2 - 0.0016538364)
        + 267.9 * max(0.0, 0.024547699839 - Q.e2)
        + 2.355 * max(0.0, 0.04081947431 - Q.girth)
        + 62.0 * max(0.0, 0.005590288644 - Q.width)
        + 718.4 * max(0.0, Q.girth2 - 0.008678044751)
        - 2582.0 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.8228 * max(0.0, Q.mass - 80.4)
        + 4684.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        + 83.95 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 13.9 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        - 101.0 * max(0.0, 0.002464291268 - Q.lam1)
        - 14.06 * max(0.0, 0.15984864831 - Q.max_dr)
        + 17.0 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 0.1893 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 83.78 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 5.249 * max(0.0, Q.e2 - 0.016554418951)
        - 15.55 * max(0.0, 0.067292226106 - Q.C2)
        - 234.5 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 636.1 * max(0.0, 0.006679471442 - Q.girth2)
        - 460.1 * max(0.0, 0.005019718802 - Q.width)
        - 4338.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 7.828 * max(0.0, 0.196739721581 - Q.LHA)
        - 43.52 * max(0.0, 0.061086014472 - Q.girth)
        - 0.002269 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 9.723 * max(0.0, 0.177304983139 - Q.max_dr)
        + 6280.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 494.3 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 5675.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 9866.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 32.02 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 0.7884 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 2076.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 3.943 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 18960.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 515.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 18990.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 46870.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 306.1 * max(0.0, 0.000964142894 - Q.girth2)
        + 35040.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        - 20930.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        + 2942.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 14.43 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 29.0 * max(0.0, 0.054649224505 - Q.girth)
        - 110.8 * max(0.0, Q.lam2 - 0.001130644719)
        - 1.026 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 938.9 * max(0.0, 0.006096650059 - Q.width)
        - 8.294 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 95.7 * max(0.0, Q.girth2 - 0.018827652745)
        - 60.97 * max(0.0, 0.032346998155 - Q.e2)
        - 0.03902 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 5.141 * max(0.0, 0.221586732566 - Q.max_dr)
        + 8261.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 328.0 * max(0.0, 0.007520088344 - Q.girth2)
        - 162.3 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 0.01225 * max(0.0, Q.sum_pt - 840.01953125)
        + 4.253 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 1994.0 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 1004.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        + 61.58 * Q.e2
        + 936.8 * max(0.0, Q.lam2 - 0.000194798295)
        - 21.45 * max(0.0, Q.LHA - 0.303313749495)
        - 37.25 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 83.76 * max(0.0, 0.004183811014 - Q.lam1)
        - 36.17 * max(0.0, Q.centroid_offset - 0.037760993714)
        - 142.0 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 1.199 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 184.4 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 1309.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.002379 * max(0.0, Q.mass - 15.454033088684)
        + 0.2895 * max(0.0, Q.mass - 53.332374954224)
        - 428.7 * max(0.0, 0.0016538364 - Q.girth2)
        - 42.02 * max(0.0, Q.LHA - 0.346713497427)
        - 643.2 * max(0.0, 0.003408388935 - Q.lam2)
        + 10.76 * max(0.0, Q.girth - 0.076081777364)
        + 146.4 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 3.802 * Q.centroid_offset
        - 87.15 * max(0.0, 0.003562611091 - Q.width)
        + 0.002282 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 70.46 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 55.61 * max(0.0, 0.148408418149 - Q.girth)
        - 18.37 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 228.8 * max(0.0, 0.016433749775 - Q.lam1)
        - 13.2 * max(0.0, 0.028070914944 - Q.z_7)
        + 1277.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 45.48 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 3.733 * max(0.0, Q.LHA - 0.09323897448)
        - 327.9 * max(0.0, 0.00752008842 - Q.width)
        + 304.4 * max(0.0, 0.006506575659 - Q.lam1)
        + 62.64 * max(0.0, Q.lam1 - 0.00543336053)
        + 148.9 * max(0.0, Q.lam1 - 0.007330079875)
        + 152.0 * max(0.0, Q.lam1 - 0.005954149834)
        + 7.404 * max(0.0, Q.lam1 - 0.002464291268)
        - 2414.0 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 4.341 * max(0.0, 0.303313749495 - Q.LHA)
        + 174.1 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 36.55 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.005318 * max(0.0, 86.4 - Q.mass)
        - 14220.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        + 1172.0 * max(0.0, 0.005834489329 - Q.e2_sq)
        + 7.951 * max(0.0, 0.041109715588 - Q.e2)
        - 139.4 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 34.7 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        - 29.78 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 87930.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 108.9 * max(0.0, 0.012003726523 - Q.lam1)
        - 89.95 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 22490.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        - 7.367 * max(0.0, 0.063441075385 - Q.e2)
        + 1.154 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 3153.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 607.7 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 44.47 * max(0.0, Q.girth - 0.033604209498)
    )


def score_t(Q):
    return (-0.3918
        + 52.65 * max(0.0, 0.004372139331 - Q.width)
        + 42.51 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.008348 * max(0.0, 64.618731689453 - Q.mass)
        + 0.01429 * max(0.0, 21.784077072144 - Q.mass)
        - 16.1 * max(0.0, 0.013238675334 - Q.girth2)
        + 0.07756 * max(0.0, Q.sum_pt - 901.59375)
        - 0.0005342 * max(0.0, 56.920347213745 - Q.mass)
        + 114.7 * max(0.0, 0.008678044951 - Q.width)
        - 0.002702 * max(0.0, 29.644699859619 - Q.mass)
        + 4.585 * max(0.0, 0.087236513197 - Q.girth)
        + 1330.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 0.01268 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 214.6 * max(0.0, 0.00543336053 - Q.lam1)
        + 7.502 * max(0.0, 0.076081777364 - Q.girth)
        - 172.6 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 115.4 * max(0.0, 0.008375572068 - Q.lam1)
        - 232.8 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 517.8 * max(0.0, 0.001503553356 - Q.lam1)
        - 2.088 * max(0.0, Q.sum_pt - 813.415625)
        + 0.5375 * max(0.0, Q.log_sum_pt - 6.377722943814)
        + 10.14 * max(0.0, 0.055577157257 - Q.z_7)
        - 38.89 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.3394 * max(0.0, 53.332374954224 - Q.mass)
        + 290.1 * max(0.0, Q.log_sum_pt - 6.572937922293)
        + 148.4 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 3667.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.006283 * max(0.0, 49.668099212646 - Q.mass)
        + 282.2 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.002963 * max(0.0, Q.sum_pt_top5 - 579.875)
        - 0.4933 * max(0.0, 69.611351776123 - Q.mass)
        - 32.34 * max(0.0, 0.005954149834 - Q.lam1)
        - 285.5 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 6.732 * max(0.0, Q.LHA - 0.111565049159)
        + 1.203 * max(0.0, 36.229410171509 - Q.mass)
        - 293.3 * max(0.0, 6.638338705138 - Q.log_sum_pt)
        - 0.01299 * max(0.0, 53.4375 - Q.pt_7)
        + 0.003102 * max(0.0, 788.4484375 - Q.sum_pt)
        - 1782.0 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 57.25 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 8.43 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 114.7 * max(0.0, Q.width - 0.018827653081)
        - 1.201 * max(0.0, Q.mass - 36.229410171509)
        - 4.398 * max(0.0, Q.e2 - 0.028531698044)
        + 53.52 * max(0.0, 0.006679471358 - Q.width)
        + 6.905 * max(0.0, 0.04447356835 - Q.e2)
        - 20.67 * max(0.0, 0.007330079875 - Q.lam1)
        - 0.2006 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.4873 * max(0.0, Q.mass - 69.611351776123)
        + 114.7 * max(0.0, 0.008678044751 - Q.girth2)
        - 1.656 * max(0.0, Q.LHA - 0.325582223496)
        - 13.31 * max(0.0, 0.038466955721 - Q.e2)
        - 276.8 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 52.65 * max(0.0, 0.004372139461 - Q.girth2)
        + 174.5 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 15.07 * max(0.0, Q.lam1 - 0.012003726523)
        - 0.2942 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2)
        - 58.78 * max(0.0, Q.lam1 - 0.008375572068)
        - 5.894 * max(0.0, Q.lam1 - 0.004183811014)
        - 89.48 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907)
        + 15.98 * max(0.0, Q.e2 - 0.050284641981)
        + 137.7 * max(0.0, Q.width - 0.000319370692)
        + 0.01349 * max(0.0, 763.825 - Q.sum_pt)
        + 5157.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 4.919 * max(0.0, Q.width - 0.001653836415)
        + 0.0005668 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 3.359 * max(0.0, Q.girth - 0.101940929517)
        + 280.7 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 177.0 * max(0.0, Q.girth2 - 0.013238675334)
        + 12.16 * max(0.0, Q.e2 - 0.007078157854)
        - 19.69 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 222.1 * max(0.0, 0.023780909279 - Q.e2_sq)
        - 38.99 * max(0.0, Q.girth2 - 0.007520088344)
        - 467.9 * max(0.0, 0.007182789718 - Q.e2_sq)
        + 37.84 * max(0.0, Q.width - 0.002635417778)
        - 1226.0 * max(0.0, 0.003013300392 - Q.e2_sq)
        + 15.87 * max(0.0, Q.e2 - 0.020459658932)
        + 16.83 * max(0.0, Q.girth2 - 0.003562611155)
        + 0.00853 * max(0.0, 41.377904891968 - Q.mass)
        + 512.6 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq)
        - 1.036 * max(0.0, 80.4 - Q.mass)
        - 9.876 * max(0.0, 0.216055863061 - Q.LHA)
        - 3683.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 1.825 * max(0.0, 0.035560912266 - Q.e2)
        + 192.3 * max(0.0, 0.002635417778 - Q.width)
        + 36.76 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 0.06926 * max(0.0, Q.sum_pt - 868.509375)
        + 5.068 * max(0.0, 0.154689112391 - Q.LHA)
        + 360.6 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        - 107.3 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        + 74.46 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 51.94 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt)
        - 0.002338 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 16.1 * max(0.0, 0.013238675006 - Q.width)
        + 64.3 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 20.81 * max(0.0, 0.050284641981 - Q.e2)
        - 158.8 * max(0.0, Q.lam2 - 0.000537286005)
        + 14.28 * max(0.0, Q.centroid_offset - 0.018377780003)
        + 571.4 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 1.142 * max(0.0, Q.LHA - 0.312727471086)
        + 0.5343 * Q.max_dr
        + 5.959 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 30.07 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 0.3687 * max(0.0, Q.girth - 0.087236513197)
        - 21.23 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.07525 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 2.988 * max(0.0, 0.024419631481 - Q.girth2_top5)
        + 1.858 * max(0.0, Q.sum_pt - 988.4078125)
        - 108.5 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 113.6 * max(0.0, Q.girth2 - 0.004372139461)
        - 10.37 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 4.919 * max(0.0, Q.girth2 - 0.0016538364)
        + 15.44 * max(0.0, 0.024547699839 - Q.e2)
        + 71.74 * max(0.0, 0.04081947431 - Q.girth)
        - 36.51 * max(0.0, 0.005590288644 - Q.width)
        - 41.4 * max(0.0, Q.girth2 - 0.008678044751)
        + 158.6 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 1.032 * max(0.0, Q.mass - 80.4)
        + 280.4 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 10.41 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 1.93 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378)
        - 127.1 * max(0.0, 0.002464291268 - Q.lam1)
        + 2.456 * max(0.0, 0.15984864831 - Q.max_dr)
        + 2.352 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 0.02817 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875)
        - 10.14 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 18.52 * max(0.0, Q.e2 - 0.016554418951)
        + 19.39 * max(0.0, 0.067292226106 - Q.C2)
        + 68.11 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 53.52 * max(0.0, 0.006679471442 - Q.girth2)
        + 268.7 * max(0.0, 0.005019718802 - Q.width)
        + 9451.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 7.868 * max(0.0, 0.196739721581 - Q.LHA)
        - 57.78 * max(0.0, 0.061086014472 - Q.girth)
        + 0.01103 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 8.682 * max(0.0, 0.177304983139 - Q.max_dr)
        + 938.9 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 261.3 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 11390.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 6001.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 298.3 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 0.06071 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 1286.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936)
        - 0.04591 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 13240.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 1519.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 9326.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 4060.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 585.5 * max(0.0, 0.000964142894 - Q.girth2)
        - 9210.0 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width)
        + 739.4 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width)
        - 1354.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width)
        - 276.1 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936)
        + 23.17 * max(0.0, 0.054649224505 - Q.girth)
        - 19.62 * max(0.0, Q.lam2 - 0.001130644719)
        - 0.581 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 67.02 * max(0.0, 0.006096650059 - Q.width)
        + 5.045 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 114.7 * max(0.0, Q.girth2 - 0.018827652745)
        + 10.99 * max(0.0, 0.032346998155 - Q.e2)
        + 0.03621 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 4.967 * max(0.0, 0.221586732566 - Q.max_dr)
        + 590.4 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 104.5 * max(0.0, 0.007520088344 - Q.girth2)
        + 59.66 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 0.07291 * max(0.0, Q.sum_pt - 840.01953125)
        - 0.1431 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 647.5 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 1091.0 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq)
        - 4.545 * Q.e2
        + 544.7 * max(0.0, Q.lam2 - 0.000194798295)
        - 1.694 * max(0.0, Q.LHA - 0.303313749495)
        - 39.01 * max(0.0, Q.centroid_offset - 0.00231612516)
        - 70.79 * max(0.0, 0.004183811014 - Q.lam1)
        + 21.14 * max(0.0, Q.centroid_offset - 0.037760993714)
        + 187.5 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq)
        - 2.155 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        + 1146.0 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 3583.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.01604 * max(0.0, Q.mass - 15.454033088684)
        - 0.3319 * max(0.0, Q.mass - 53.332374954224)
        + 157.0 * max(0.0, 0.0016538364 - Q.girth2)
        - 2.134 * max(0.0, Q.LHA - 0.346713497427)
        - 238.5 * max(0.0, 0.003408388935 - Q.lam2)
        + 2.517 * max(0.0, Q.girth - 0.076081777364)
        - 31.58 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 17.24 * Q.centroid_offset
        + 188.0 * max(0.0, 0.003562611091 - Q.width)
        - 0.003052 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 11.86 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 20.08 * max(0.0, 0.148408418149 - Q.girth)
        + 9.688 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 30.92 * max(0.0, 0.016433749775 - Q.lam1)
        + 30.34 * max(0.0, 0.028070914944 - Q.z_7)
        + 465.3 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 0.8743 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 2.5 * max(0.0, Q.LHA - 0.09323897448)
        + 104.5 * max(0.0, 0.00752008842 - Q.width)
        - 136.9 * max(0.0, 0.006506575659 - Q.lam1)
        + 272.3 * max(0.0, Q.lam1 - 0.00543336053)
        + 41.05 * max(0.0, Q.lam1 - 0.007330079875)
        + 18.47 * max(0.0, Q.lam1 - 0.005954149834)
        - 23.54 * max(0.0, Q.lam1 - 0.002464291268)
        - 552.3 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 1.213 * max(0.0, 0.303313749495 - Q.LHA)
        - 18.78 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 7.552 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.004866 * max(0.0, 86.4 - Q.mass)
        + 2584.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021)
        - 298.6 * max(0.0, 0.005834489329 - Q.e2_sq)
        - 23.8 * max(0.0, 0.041109715588 - Q.e2)
        - 82.45 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435)
        + 82.95 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435)
        + 18.57 * max(0.0, Q.centroid_offset - 0.012587644117)
        - 3232.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 48.51 * max(0.0, 0.012003726523 - Q.lam1)
        - 31.04 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 630.9 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 22.14 * max(0.0, 0.063441075385 - Q.e2)
        - 0.005104 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 1874.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 63.91 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 8.428 * max(0.0, Q.girth - 0.033604209498)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if s['W'] - s['t'] > -3.7087724208831787:
        if s['q'] - s['Z'] > 0.3887976258993149:
            if s['g'] - s['q'] > -2.7633052013698034e-05:
                if s['g'] - s['W'] > 0.0466865599155426:
                    if s['g'] - s['q'] > 0.1285959929227829:
                        if s['g'] - s['t'] > -0.1408286765217781:
                            return 'g'   # 98% of the training jets here get this class from the formula
                        else:
                            return 't'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 39.171875:
                            if Q.sum_pt_top5 > 699.09375:
                                return 'g'   # 83% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.dr_0 > 0.01710436213761568:
                                if Q.sum_pt_top5 > 510.484375:
                                    return 'q'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 86% of the training jets here get this class from the formula
                else:
                    if Q.mass > 33.76364517211914:
                        return 'W'   # 82% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 61% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.12144129350781441:
                    if s['g'] - s['q'] > -0.19058070331811905:
                        if s['g'] - s['Z'] > 2.443682312965393:
                            if Q.dr_0 > 0.01093333587050438:
                                return 'q'   # 61% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 729.34375:
                                return 'q'   # 51% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 1.5010413527488708:
                                    return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 5.647509813308716:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > 2.639635682106018:
                            return 'q'   # 73% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.3895184397697449:
                                if s['q'] - s['t'] > -0.06083803437650204:
                                    return 'q'   # 99% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 78% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.4170846939086914:
                        return 'q'   # 50% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 88% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.03591829538345337:
                if s['g'] - s['W'] > -0.07617003843188286:
                    if s['g'] - s['W'] > 0.19962169975042343:
                        if s['g'] - s['t'] > -0.031638745218515396:
                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            return 't'   # 84% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > 0.00030595826683565974:
                            if s['g'] - s['Z'] > 0.20501083880662918:
                                return 'g'   # 68% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 42% of the training jets here get this class from the formula
                        else:
                            return 't'   # 73% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.28752192854881287:
                        if s['g'] - s['W'] > -0.6589205265045166:
                            if s['W'] - s['t'] > 0.06521953642368317:
                                return 'W'   # 73% of the training jets here get this class from the formula
                            else:
                                return 't'   # 70% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.1694198101758957:
                                return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                return 't'   # 74% of the training jets here get this class from the formula
                    else:
                        if Q.lam1 > 0.0005954582884442061:
                            if Q.e2 > 0.0037007677601650357:
                                if Q.max_dr > 0.14695899188518524:
                                    if Q.lam2 > 3.882028795487713e-05:
                                        return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.010282072238624096:
                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 72% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 76% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 96% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.034439366310834885:
                    if s['g'] - s['t'] > 0.05757214315235615:
                        return 'g'   # 86% of the training jets here get this class from the formula
                    else:
                        return 't'   # 74% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.13110974431037903:
                        if s['W'] - s['Z'] > -0.2790039926767349:
                            if s['W'] - s['Z'] > -0.13960587233304977:
                                if Q.LHA > 0.17477062344551086:
                                    if Q.lam2 > 3.9826658394304104e-05:
                                        return 'W'   # 49% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 75% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 70% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.3339713215827942:
                                return 'Z'   # 69% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 0.6765598356723785:
                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 23.1640625:
                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 46% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.19676385074853897:
                            return 'Z'   # 52% of the training jets here get this class from the formula
                        else:
                            return 't'   # 77% of the training jets here get this class from the formula
    else:
        if s['g'] - s['t'] > -0.06894927099347115:
            return 'g'   # 74% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > 0.030778286047279835:
                if Q.max_dr > 0.1577732190489769:
                    return 't'   # 54% of the training jets here get this class from the formula
                else:
                    return 'Z'   # 79% of the training jets here get this class from the formula
            else:
                if s['g'] - s['t'] > -0.46360163390636444:
                    return 't'   # 76% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['t'] > -0.47452840209007263:
                        return 'q'   # 55% of the training jets here get this class from the formula
                    else:
                        return 't'   # 99% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 0
    print('class:', classify(pt, eta, phi))
