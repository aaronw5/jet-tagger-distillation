"""JEDI-linear jet tagger, 8 particles, 3 features: the tuned formula (start): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 65.47% (the formula: 65.56%); same class as the formula for 95.61% of jets.  886 leaves, depth 19.
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
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        phi_1=phi[1],
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
    return (-0.2834
        - 1.772 * max(0.0, 0.148419710734 - Q.planar_flow)
        + 66.28 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 275.6 * max(0.0, 0.004372139331 - Q.width)
        + 382.5 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.0115 * max(0.0, 64.618731689453 - Q.mass)
        + 0.06413 * max(0.0, 21.784077072144 - Q.mass)
        + 16.99 * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 133.6 * max(0.0, 0.013238675334 - Q.girth2)
        + 4.787 * max(0.0, 0.006506575659 - Q.lam1)
        - 172.4 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        + 0.2014 * max(0.0, Q.sum_pt - 901.59375)
        + 0.008919 * max(0.0, 56.920347213745 - Q.mass)
        + 19.55 * max(0.0, 0.008678044951 - Q.width)
        + 0.0001148 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        + 0.001229 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2)
        + 0.002373 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2)
        + 0.04891 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 29.43 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 0.1014 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        + 0.02565 * max(0.0, 29.644699859619 - Q.mass)
        - 0.03705 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        - 24.75 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr)
        + 0.3232 * max(0.0, 0.087236513197 - Q.girth)
        - 0.4137 * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 3382.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 0.4726 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 17.52 * max(0.0, 0.020459658932 - Q.e2)
        + 468.9 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 44.79 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 8.476 * max(0.0, Q.log_sum_pt - 6.638338705138)
        + 0.00173 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 0.001939 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6)
        + 66.04 * max(0.0, 0.00543336053 - Q.lam1)
        + 1.024 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 0.1435 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 77.51 * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 11.14 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0003874 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        + 1.372 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7)
        + 0.02895 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        + 14.29 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0)
        - 30.81 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 9.992 * max(0.0, 0.012569162668 - Q.planar_flow)
        + 66.19 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.05218 * max(0.0, Q.pt_7 - 34.53125)
        + 5550.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0006711 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 9.295 * max(0.0, 0.035560912266 - Q.e2)
        - 18.16 * max(0.0, 0.055577157257 - Q.z_7)
        + 0.67 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 3.36 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.06065 * max(0.0, 53.332374954224 - Q.mass)
        + 0.0007103 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3)
        - 4.993 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        + 4.705 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 0.02707 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21)
        - 844.5 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 1402.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 0.5785 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 284.9 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2)
        - 0.7877 * max(0.0, 0.197783735394 - Q.tau21)
        + 39.78 * max(0.0, 0.008168570676 - Q.e2_sq)
        + 0.01322 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.0005972 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3)
        - 12.39 * max(0.0, 0.042322802544 - Q.C2)
        - 0.06973 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6)
        - 338.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3)
        + 2201.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 9206.0 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2)
        + 1428.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 51.38 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32)
        - 0.9672 * max(0.0, 0.346713497427 - Q.LHA)
        + 13.66 * max(0.0, Q.e2 - 0.032346998155)
        + 37.04 * max(0.0, Q.mass_over_sum_pt - 0.054892207095)
        + 32.82 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 59.62 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1321.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 29.6 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 26.06 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow)
        - 1.024 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0)
        - 2.02 * max(0.0, 0.046566883102 - Q.max_dr)
        - 17.96 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32)
        - 0.2488 * max(0.0, 69.611351776123 - Q.mass)
        - 0.03371 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7)
        - 5.72 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 2.053 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        - 14.24 * max(0.0, 0.028070914944 - Q.z_7)
        + 60.75 * max(0.0, 0.005954149834 - Q.lam1)
        + 17380.0 * max(0.0, 9.1213921e-05 - Q.width)
        + 0.04064 * max(0.0, Q.pt_7 - 30.484375)
        - 0.5849 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        - 7.999 * max(0.0, Q.z_7 - 0.046240320761)
        - 4552.0 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width)
        + 0.7102 * max(0.0, Q.LHA - 0.111565049159)
        - 1004.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.2887 * max(0.0, 0.694781820497 - Q.planar_flow)
        + 0.04045 * max(0.0, 36.229410171509 - Q.mass)
        - 7.578e-05 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746)
        + 15.29 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.1624 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        - 40.67 * max(0.0, 0.003377388461 - Q.lam1)
        - 38920.0 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset)
        - 50.17 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset)
        - 53.04 * max(0.0, 0.016858545121 - Q.z_7)
        - 0.001638 * max(0.0, 687.4375 - Q.sum_pt_top5)
        + 0.04463 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7)
        - 0.04002 * max(0.0, 53.4375 - Q.pt_7)
        + 0.0137 * max(0.0, 43.5 - Q.pt_7)
        + 0.003459 * max(0.0, 788.4484375 - Q.sum_pt)
        + 3.311 * max(0.0, Q.log_sum_pt - 6.267538488641)
        + 212.6 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 6135.0 * max(0.0, 4.8108519e-05 - Q.girth2)
        - 183.2 * max(0.0, Q.log_sum_pt - 6.804164030582)
        - 2.183 * max(0.0, 0.15984864831 - Q.max_dr)
        - 0.0007036 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731)
        + 2174.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 72.3 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 10.33 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 4.659 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32)
        - 155.5 * max(0.0, Q.width - 0.018827653081)
        - 0.06805 * max(0.0, Q.mass - 36.229410171509)
        + 3.765 * max(0.0, Q.e2 - 0.028531698044)
        - 3.525 * max(0.0, Q.LHA - 0.312727471086)
        - 18.27 * max(0.0, 0.006679471358 - Q.width)
        - 0.5417 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 9.752 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2)
        - 12.67 * max(0.0, 0.04447356835 - Q.e2)
        + 57.82 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.2181 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.003234 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 0.258 * max(0.0, Q.mass - 69.611351776123)
        + 2.297 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1)
        + 19.54 * max(0.0, 0.008678044751 - Q.girth2)
        + 2.235 * max(0.0, Q.LHA - 0.325582223496)
        - 0.001976 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 3.919 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0)
        + 0.5323 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828)
        + 0.0001329 * max(0.0, Q.mass_top5 - 53.607658247923)
        + 0.2667 * max(0.0, -0.012844925793 - Q.mean_eta)
        + 8.024 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0)
        - 147.4 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion)
        - 0.05709 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6)
        - 19.0 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 37.22 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 275.7 * max(0.0, 0.004372139461 - Q.girth2)
        - 74.31 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 114.9 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        + 5.135 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781)
        - 21.74 * max(0.0, Q.lam1 - 0.012003726523)
        + 332.9 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        + 0.1835 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr)
        - 0.9645 * max(0.0, Q.max_dr - 0.145231109113)
        - 4.826 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3)
        - 463.3 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153)
        + 0.006656 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4)
        - 10.55 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4)
        - 0.3687 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5)
        + 87.84 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153)
        - 35.27 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.05484 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        - 0.3839 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6)
        - 0.04111 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7)
        - 3.095 * max(0.0, Q.LHA - 0.423592510895)
        - 0.0001863 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7)
        - 1.483 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.06003 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        - 58.76 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 568.2 * max(0.0, 0.000306123359 - Q.lam2)
        - 585.7 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        + 2151.0 * max(0.0, Q.width - 0.000319370692)
        + 0.01119 * max(0.0, 763.825 - Q.sum_pt)
        + 150.8 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2)
        + 813.7 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 0.9298 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 56.29 * max(0.0, Q.width - 0.001653836415)
        + 0.0002796 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 104.2 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.6036 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        - 0.757 * max(0.0, Q.C2 - 0.067292226106)
        + 342.2 * max(0.0, 0.001130644719 - Q.lam2)
        - 25.56 * max(0.0, 0.005011406868 - Q.girth2_top3)
        + 14.63 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0)
        - 8.64 * max(0.0, Q.girth - 0.101940929517)
        + 103.2 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 402.1 * max(0.0, Q.girth2 - 0.013238675334)
        - 0.001709 * max(0.0, 430.75 - Q.sum_pt_top5)
        + 21.79 * max(0.0, Q.e2 - 0.063441075385)
        - 70.27 * max(0.0, Q.e2 - 0.007078157854)
        - 0.01083 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        + 16.3 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi)
        - 0.06231 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        - 18.65 * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 5.671 * max(0.0, Q.max_dr - 0.102758520097)
        + 3.696 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 3.216 * max(0.0, Q.max_dr - 0.197968879342)
        + 0.04966 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        - 1.993 * max(0.0, Q.C2 - 0.014943876117)
        - 0.1083 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 0.7842 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 11.2 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 1.038 * max(0.0, 0.047915700823 - Q.girth)
        - 0.0006062 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875)
        - 66.01 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 66.97 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        + 8.743 * max(0.0, 0.216055863061 - Q.LHA)
        - 4.646 * max(0.0, 0.049399692737 - Q.z_7)
        - 0.2182 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        - 10.01 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 161.1 * max(0.0, 0.00528466865 - Q.e2_sq)
        - 18.62 * max(0.0, 0.028865759995 - Q.z_6)
        - 3.011 * max(0.0, 0.071488645583 - Q.z_7)
        - 749.5 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 646.7 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 0.03954 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        - 11640.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1)
        + 0.5739 * max(0.0, 0.03243272066 - Q.z_7)
        + 14.59 * max(0.0, 0.026454043164 - Q.dr_0)
        + 56.99 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 3.541 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0)
        - 1.856 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 84.53 * max(0.0, 0.002635417778 - Q.width)
        - 66.03 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 28.59 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 0.002803 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 4780.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0008377 * max(0.0, 548.196875 - Q.sum_pt_top2)
        + 56.27 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 277.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.4748 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625)
        + 0.01673 * max(0.0, 24.578125 - Q.pt_5)
        + 0.004105 * max(0.0, Q.sum_pt - 868.509375)
        - 0.09492 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 0.1778 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085)
        + 254.7 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 1063.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.1366 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 136.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        + 141.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 321.3 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2)
        - 66.81 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 85.12 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        + 0.437 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 197.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.1782 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.2512 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142)
        - 7.167 * max(0.0, 0.01426135283 - Q.mean_phi2)
        - 0.1075 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass)
        - 133.9 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 51850.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 43.06 * max(0.0, 0.021588001063 - Q.dr_0)
        - 10.86 * max(0.0, Q.centroid_offset - 0.008092360237)
        + 133.7 * max(0.0, 0.013238675006 - Q.width)
        + 13.07 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 1.273 * max(0.0, 6.327378592257 - Q.log_sum_pt)
        + 3505.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 11.47 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7)
        + 11.99 * max(0.0, 0.050284641981 - Q.e2)
        + 10.57 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        - 77.72 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2)
        + 17.15 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.02557 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125)
        - 6.994 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 76.42 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        - 0.8261 * max(0.0, Q.centroid_offset - 0.049903668404)
        + 3.843 * Q.max_dr
        + 1.127 * max(0.0, Q.C2 - 0.010539266048)
        + 199.2 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 26.99 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 0.3552 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1)
        + 11.88 * max(0.0, 0.000537286005 - Q.lam2)
        + 17.06 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072)
        - 86.38 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 7.237 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 3.975 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537)
        + 15.55 * max(0.0, Q.girth - 0.087236513197)
        - 3.215 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.008204 * max(0.0, 1.679198372364 - Q.D2)
        + 0.1089 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5)
        + 19.88 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2)
        + 56.54 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 25.24 * max(0.0, Q.lam2 - 0.003408388935)
        - 1.477 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 0.003649 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 5.275 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5)
        + 777.1 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105)
        - 10890.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105)
        + 7.718 * max(0.0, 0.003562611155 - Q.girth2)
        + 0.01841 * max(0.0, 49.668099212646 - Q.mass)
        + 7.441 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.0002815 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 3.41 * max(0.0, Q.z_7 - 0.06164517166)
        + 0.04264 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 0.06881 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        + 0.003327 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 0.2911 * max(0.0, 0.195013533663 - Q.planar_flow)
        + 69.68 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 67.22 * max(0.0, 0.001056655216 - Q.girth2_top2)
        - 387.2 * max(0.0, Q.girth2 - 0.007520088344)
        - 263.3 * max(0.0, Q.girth2 - 0.004372139461)
        - 0.7156 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 1378.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 56.29 * max(0.0, Q.girth2 - 0.0016538364)
        - 28.81 * max(0.0, 0.024547699839 - Q.e2)
        + 0.005787 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6)
        - 19.52 * max(0.0, 0.04081947431 - Q.girth)
        - 0.0241 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439)
        - 0.005388 * max(0.0, 48.71875 - Q.pt_7)
        + 0.001816 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 13.47 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 137.3 * max(0.0, 0.005590288644 - Q.width)
        - 35.41 * max(0.0, 0.02076709205 - Q.centroid_offset)
        - 518.9 * max(0.0, Q.girth2 - 0.008678044751)
        + 55.15 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 0.2915 * max(0.0, 0.001101266364 - Q.e2_sq)
        - 0.004185 * max(0.0, Q.mass - 80.4)
        - 285.5 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 48.75 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        - 3.763 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 18.15 * max(0.0, 0.038466955721 - Q.e2)
        - 5.176 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.08691 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 0.00512 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 0.2205 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 4.49 * max(0.0, 0.293190627853 - Q.LHA)
        - 3.317 * max(0.0, 0.005884990035 - Q.girth2_top3)
        + 1095.0 * max(0.0, 0.000561123155 - Q.width)
        + 0.08449 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5)
        + 1172.0 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1)
        + 0.02319 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        - 1.2 * max(0.0, 0.197968879342 - Q.max_dr)
        - 67.65 * max(0.0, 0.005019718802 - Q.width)
        + 50.47 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21)
        + 8937.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 1.029 * max(0.0, 0.196739721581 - Q.LHA)
        - 0.6156 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 42.51 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 0.1352 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2)
        + 10.2 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0)
        - 22.79 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264)
        + 9.604 * max(0.0, 0.061086014472 - Q.girth)
        - 0.0005766 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 54.95 * max(0.0, 0.003343241496 - Q.centroid_offset)
        + 1.443 * max(0.0, 0.177304983139 - Q.max_dr)
        + 2799.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 4.225 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 1.073 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 14680.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 5410.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 1722.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 3263.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 151.9 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.02109 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 268.4 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 4930.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        - 12.95 * max(0.0, 0.016554418951 - Q.e2)
        - 273.4 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 5.301 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 3990.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        + 37.91 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394)
        - 3.685 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.01689 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375)
        - 5528.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 1959.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 230.8 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 175.8 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 12640.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 5078.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.1615 * max(0.0, 8.379955863953 - Q.mass)
        - 0.3914 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 16.61 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121)
        + 0.2005 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 830.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 883.3 * max(0.0, 0.000964142894 - Q.girth2)
        + 131.7 * max(0.0, 0.000222950415 - Q.girth2_top5)
        - 16.48 * max(0.0, 0.054649224505 - Q.girth)
        - 175.7 * max(0.0, Q.lam2 - 0.001130644719)
        + 1.634 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 76.5 * max(0.0, 0.006096650059 - Q.width)
        - 4.319 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 155.5 * max(0.0, Q.girth2 - 0.018827652745)
        - 17.28 * max(0.0, 0.032346998155 - Q.e2)
        - 205.7 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 56.85 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.1741 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 0.07293 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 0.9901 * max(0.0, 0.221586732566 - Q.max_dr)
        - 1988.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 4.972 * max(0.0, Q.C2 - 0.051192347892)
        + 15.85 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 3438.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 6144.0 * max(0.0, 0.000172198326 - Q.width)
        + 22.32 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752)
        + 150.6 * max(0.0, 0.007520088344 - Q.girth2)
        - 1.074 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 0.0931 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.5273 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 323.9 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 215.4 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        + 13.22 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity)
        + 0.02548 * max(0.0, 41.377904891968 - Q.mass)
        - 0.07293 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 18.32 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 14.97 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 85.58 * max(0.0, 0.001503553356 - Q.lam1)
        + 1.942 * max(0.0, Q.max_dr - 0.15984864831)
        + 2.54 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        + 0.3894 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264)
        + 0.002105 * max(0.0, 35.5 - Q.pt_5)
        + 0.5192 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 193.7 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta)
        - 0.001257 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962)
        - 1.699 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
        + 28.63 * Q.e2
        + 196.9 * max(0.0, Q.lam2 - 0.000194798295)
        - 3.694 * max(0.0, Q.LHA - 0.303313749495)
        - 109.1 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668)
        - 2.365 * max(0.0, 6.267538488641 - Q.log_sum_pt)
        - 0.1017 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7)
        - 1.449 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        - 0.06828 * max(0.0, Q.n_dr_0p2_0p4 - 2.0)
        + 76.25 * max(0.0, 0.004183811014 - Q.lam1)
        - 83.79 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt)
        - 0.01178 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 4.675 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21)
        + 0.2967 * max(0.0, 0.269169217348 - Q.tau32)
        + 1.542 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21)
        - 0.04574 * max(0.0, 0.391541349888 - Q.tau21)
        - 38.56 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.1639 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 11.12 * max(0.0, 0.002412890926 - Q.girth2_top2)
        + 0.007334 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125)
        + 1.047 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7)
        - 10.6 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        + 10.86 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7)
        - 3.226 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386)
        + 1.265 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 815.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.257 * max(0.0, Q.mass - 15.454033088684)
        + 0.09748 * max(0.0, Q.max_dr - 0.121680960059)
        + 0.003041 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 403.9 * max(0.0, 0.0016538364 - Q.girth2)
        + 0.009148 * max(0.0, Q.sum_pt - 813.415625)
        - 18.27 * max(0.0, 0.006679471442 - Q.girth2)
        - 0.5856 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 8.634 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        - 1.696 * max(0.0, 0.154689112391 - Q.LHA)
        + 26.24 * max(0.0, Q.girth - 0.076081777364)
        + 0.01006 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        - 1.833 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097)
        - 0.894 * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 0.07612 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 0.9174 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
        + 44.09 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 2.605 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.0113 * max(0.0, 7.0 - Q.n_dr_0_0p05)
        + 6.068 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2)
        - 0.3011 * Q.centroid_offset
        + 7.717 * max(0.0, 0.003562611091 - Q.width)
        + 0.0006653 * max(0.0, 9.257203159811 - Q.mass_top5)
        + 197.5 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        + 0.2331 * max(0.0, 15.454033088684 - Q.mass)
        - 0.4502 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.03929 * max(0.0, 29.0421875 - Q.pt_7)
        - 38.5 * max(0.0, 0.004839980301 - Q.lam1)
        + 2.098 * max(0.0, 0.035786485299 - Q.C2)
        + 3.879 * max(0.0, 0.111761856824 - Q.max_dr)
        + 0.04943 * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.0005722 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2)
        - 0.05274 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625)
        - 0.001744 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.01904 * max(0.0, Q.mass - 91.19)
        - 7538.0 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005)
        - 0.1899 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        + 2.483 * max(0.0, Q.mean_phi - 0.026127964072)
        + 3.104 * max(0.0, 0.148408418149 - Q.girth)
        - 1.924 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.1844 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 34.81 * max(0.0, 0.016433749775 - Q.lam1)
        + 0.1149 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 9246.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 8.181e-05 * max(0.0, 531.1875 - Q.sum_pt_top5)
        - 1.392e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        - 0.0004635 * max(0.0, Q.sum_pt_top5 - 902.40625)
        - 0.0005351 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 0.01497 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 4.282 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        - 0.1987 * max(0.0, 0.501026660204 - Q.tau21)
        + 191.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 0.9786 * max(0.0, 0.067272114405 - Q.z_6)
        + 9.076 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 5.799 * max(0.0, Q.LHA - 0.09323897448)
        + 0.000784 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 0.003556 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2)
        + 1.657 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 150.6 * max(0.0, 0.00752008842 - Q.width)
        - 0.02978 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4)
        - 0.02675 * max(0.0, 25.578125 - Q.pt_7)
        - 0.2274 * max(0.0, Q.sum_pt - 988.4078125)
        - 0.0005545 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0007675 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        - 4.245 * max(0.0, 0.111513564951 - Q.planar_flow)
        + 0.06501 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 74.12 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 4.615 * max(0.0, Q.lam1 - 0.008375572068)
        + 12.63 * max(0.0, Q.lam1 - 0.004183811014)
        + 44.5 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        - 0.6891 * max(0.0, Q.lam1 - 0.00543336053)
        + 0.01148 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt)
        - 4.761 * max(0.0, Q.lam1 - 0.007330079875)
        + 5.611 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        + 4746.0 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr)
        - 1.504 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        - 2887.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        - 64.6 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 0.009241 * max(0.0, 76.655700683594 - Q.mass)
        - 11.89 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164)
        - 0.5652 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 5.844 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.03824 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 121.9 * max(0.0, Q.lam1 - 0.002464291268)
        + 503.4 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 0.01288 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2)
        + 810.8 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow)
        - 737.4 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 101.8 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 0.3063 * max(0.0, 1.122624260187 - Q.D2)
        - 47.1 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.02768 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 7.059 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 11.36 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116)
        - 0.2591 * max(0.0, 0.74595130682 - Q.D2)
        + 157.4 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 122.8 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 3.104 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.5034 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 41.79 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 8.231 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.1081 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 1.385 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr)
        + 4.489 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 5007.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 38.68 * max(0.0, 0.012003726523 - Q.lam1)
        - 197.3 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 45.04 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.008634 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 3.965 * max(0.0, 0.041109715588 - Q.e2)
        - 11.01 * max(0.0, 0.063441075385 - Q.e2)
        - 0.1299 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 1.145 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.02785 * max(0.0, 2.0 - Q.n_dr_0_0p05)
        + 1.798 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264)
        - 1.293 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377)
        - 0.01419 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 11.08 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        - 1021.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 0.4732 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276)
        + 267.7 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow)
        - 48.81 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.01342 * max(0.0, Q.n_dr_0p05_0p1 - 5.0)
        - 1.025 * max(0.0, Q.LHA - 0.346713497427)
        - 9.08 * max(0.0, Q.girth - 0.033604209498)
        + 18.38 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 3.236 * max(0.0, 0.007639643088 - Q.girth2_top2)
    )


def score_q(Q):
    return (-0.47
        - 0.7683 * max(0.0, 0.148419710734 - Q.planar_flow)
        + 22.39 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 280.4 * max(0.0, 0.004372139331 - Q.width)
        + 440.6 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.003922 * max(0.0, 64.618731689453 - Q.mass)
        + 0.1414 * max(0.0, 21.784077072144 - Q.mass)
        - 7.276 * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 172.3 * max(0.0, 0.013238675334 - Q.girth2)
        + 34.92 * max(0.0, 0.006506575659 - Q.lam1)
        - 228.1 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        + 0.01113 * max(0.0, Q.sum_pt - 901.59375)
        + 0.0003197 * max(0.0, 56.920347213745 - Q.mass)
        + 107.8 * max(0.0, 0.008678044951 - Q.width)
        - 7.793e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.0005239 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2)
        - 0.02814 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2)
        + 0.2658 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 16.89 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 0.5328 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        + 0.01484 * max(0.0, 29.644699859619 - Q.mass)
        + 0.008209 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        - 22.46 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr)
        - 11.71 * max(0.0, 0.087236513197 - Q.girth)
        + 1.164 * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 3057.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 0.1132 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 9.149 * max(0.0, 0.020459658932 - Q.e2)
        + 1054.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 229.9 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 5.462 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.0001328 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 0.1184 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6)
        + 147.6 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.5529 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 0.06818 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 62.88 * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 8.413 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0001912 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        + 1.59 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7)
        + 0.03288 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        - 0.4303 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0)
        - 40.76 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 0.0994 * max(0.0, 0.012569162668 - Q.planar_flow)
        + 198.8 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.01077 * max(0.0, Q.pt_7 - 34.53125)
        + 99.38 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0001851 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 0.239 * max(0.0, 0.035560912266 - Q.e2)
        + 2.851 * max(0.0, 0.055577157257 - Q.z_7)
        + 0.266 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101)
        + 120.6 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.04576 * max(0.0, 53.332374954224 - Q.mass)
        + 5.054e-06 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3)
        - 3.908 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        + 4.328 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 0.007449 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21)
        - 730.2 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 552.6 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        - 1.678 * max(0.0, 0.083662731125 - Q.planar_flow)
        - 15.7 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2)
        - 0.3115 * max(0.0, 0.197783735394 - Q.tau21)
        - 218.6 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.02682 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.0001639 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3)
        - 3.695 * max(0.0, 0.042322802544 - Q.C2)
        + 0.003917 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6)
        + 48.55 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3)
        - 252.3 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 2053.0 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2)
        - 1314.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 1.534 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32)
        - 7.822 * max(0.0, 0.346713497427 - Q.LHA)
        + 21.26 * max(0.0, Q.e2 - 0.032346998155)
        + 30.28 * max(0.0, Q.mass_over_sum_pt - 0.054892207095)
        + 15.02 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 39.89 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 521.2 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 32.22 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 7.099 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow)
        - 0.9919 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 1.784 * max(0.0, 0.046566883102 - Q.max_dr)
        + 3.698 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32)
        - 0.3187 * max(0.0, 69.611351776123 - Q.mass)
        + 0.02331 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7)
        - 1.121 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 0.1873 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        - 22.64 * max(0.0, 0.028070914944 - Q.z_7)
        + 149.0 * max(0.0, 0.005954149834 - Q.lam1)
        + 28720.0 * max(0.0, 9.1213921e-05 - Q.width)
        - 0.02199 * max(0.0, Q.pt_7 - 30.484375)
        + 0.331 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 3.825 * max(0.0, Q.z_7 - 0.046240320761)
        + 2155.0 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width)
        + 2.119 * max(0.0, Q.LHA - 0.111565049159)
        - 785.8 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        + 0.1616 * max(0.0, 0.694781820497 - Q.planar_flow)
        + 0.07282 * max(0.0, 36.229410171509 - Q.mass)
        - 1.467e-05 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746)
        + 5.632 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0312 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        - 51.24 * max(0.0, 0.003377388461 - Q.lam1)
        - 39320.0 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset)
        - 89.3 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset)
        - 6.688 * max(0.0, 0.016858545121 - Q.z_7)
        + 0.002156 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 0.01 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7)
        - 0.004019 * max(0.0, 53.4375 - Q.pt_7)
        - 0.001251 * max(0.0, 43.5 - Q.pt_7)
        - 0.00097 * max(0.0, 788.4484375 - Q.sum_pt)
        + 4.044 * max(0.0, Q.log_sum_pt - 6.267538488641)
        + 4.468 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 53960.0 * max(0.0, 4.8108519e-05 - Q.girth2)
        - 9.289 * max(0.0, Q.log_sum_pt - 6.804164030582)
        - 0.5092 * max(0.0, 0.15984864831 - Q.max_dr)
        - 5.402e-05 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731)
        + 8062.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 106.2 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 11.77 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 7.376 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32)
        - 170.1 * max(0.0, Q.width - 0.018827653081)
        - 0.07219 * max(0.0, Q.mass - 36.229410171509)
        + 10.82 * max(0.0, Q.e2 - 0.028531698044)
        - 4.746 * max(0.0, Q.LHA - 0.312727471086)
        - 89.06 * max(0.0, 0.006679471358 - Q.width)
        - 0.4275 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 8.091 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2)
        - 7.703 * max(0.0, 0.04447356835 - Q.e2)
        + 136.2 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.9142 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.006428 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 0.3293 * max(0.0, Q.mass - 69.611351776123)
        - 5.131 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1)
        + 107.8 * max(0.0, 0.008678044751 - Q.girth2)
        + 0.8063 * max(0.0, Q.LHA - 0.325582223496)
        + 0.001353 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 14.18 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0)
        + 0.4573 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828)
        + 0.0008051 * max(0.0, Q.mass_top5 - 53.607658247923)
        + 0.7738 * max(0.0, -0.012844925793 - Q.mean_eta)
        + 10.36 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0)
        - 147.2 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion)
        - 0.05406 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6)
        - 19.55 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 27.54 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        + 280.4 * max(0.0, 0.004372139461 - Q.girth2)
        - 135.1 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 128.2 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        + 10.17 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781)
        - 121.2 * max(0.0, Q.lam1 - 0.012003726523)
        - 7.148 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 0.06763 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr)
        - 0.1256 * max(0.0, Q.max_dr - 0.145231109113)
        + 4.206 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3)
        - 659.0 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.04087 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4)
        + 2.541 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4)
        - 0.3947 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5)
        + 45.27 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153)
        + 84.66 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.003067 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        - 0.6392 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6)
        + 0.02367 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7)
        - 3.521 * max(0.0, Q.LHA - 0.423592510895)
        + 0.0002615 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7)
        - 2.818 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.06116 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        - 22.02 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 655.4 * max(0.0, 0.000306123359 - Q.lam2)
        + 42.7 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        + 2841.0 * max(0.0, Q.width - 0.000319370692)
        + 0.004985 * max(0.0, 763.825 - Q.sum_pt)
        + 147.9 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2)
        + 358.1 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 2.091 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 91.69 * max(0.0, Q.width - 0.001653836415)
        + 0.0006051 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 123.6 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 1.656 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 1.316 * max(0.0, Q.C2 - 0.067292226106)
        + 357.5 * max(0.0, 0.001130644719 - Q.lam2)
        - 15.82 * max(0.0, 0.005011406868 - Q.girth2_top3)
        + 36.15 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0)
        - 9.798 * max(0.0, Q.girth - 0.101940929517)
        + 113.7 * max(0.0, 0.017162483186 - Q.e2_sq)
        - 435.7 * max(0.0, Q.girth2 - 0.013238675334)
        - 0.002625 * max(0.0, 430.75 - Q.sum_pt_top5)
        + 30.27 * max(0.0, Q.e2 - 0.063441075385)
        - 99.12 * max(0.0, Q.e2 - 0.007078157854)
        - 0.01862 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        + 35.18 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi)
        - 0.09425 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        - 16.28 * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 1.245 * max(0.0, Q.max_dr - 0.102758520097)
        + 7.701 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        + 2.772 * max(0.0, Q.max_dr - 0.197968879342)
        + 0.1731 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        + 0.8441 * max(0.0, Q.C2 - 0.014943876117)
        - 0.7526 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        + 4.325 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        + 23.99 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 7.737 * max(0.0, 0.047915700823 - Q.girth)
        - 0.0005281 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875)
        - 106.3 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 64.69 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        + 18.38 * max(0.0, 0.216055863061 - Q.LHA)
        + 0.07321 * max(0.0, 0.049399692737 - Q.z_7)
        + 0.04606 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        - 31.86 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 255.5 * max(0.0, 0.00528466865 - Q.e2_sq)
        + 0.5337 * max(0.0, 0.028865759995 - Q.z_6)
        + 8.294 * max(0.0, 0.071488645583 - Q.z_7)
        - 4941.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 205.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 0.01362 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        - 13190.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1)
        + 2.987 * max(0.0, 0.03243272066 - Q.z_7)
        - 6.235 * max(0.0, 0.026454043164 - Q.dr_0)
        + 8.368 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.6054 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0)
        - 1.693 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 37.43 * max(0.0, 0.002635417778 - Q.width)
        - 93.33 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 52.26 * max(0.0, 0.002270363079 - Q.girth2_top5)
        + 0.002311 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 1792.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        - 0.0006172 * max(0.0, 548.196875 - Q.sum_pt_top2)
        + 21070.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 237.5 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 0.1902 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625)
        - 0.00269 * max(0.0, 24.578125 - Q.pt_5)
        - 0.002699 * max(0.0, Q.sum_pt - 868.509375)
        + 0.1445 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 0.2598 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085)
        - 373.5 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 3340.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        + 0.03332 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        + 27.03 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        - 50.82 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        - 87.88 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2)
        - 79.43 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 61.9 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        + 0.06366 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 6302.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.005532 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        - 0.03839 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142)
        - 1.091 * max(0.0, 0.01426135283 - Q.mean_phi2)
        + 0.2173 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass)
        - 47.36 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 32210.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        - 8.593 * max(0.0, 0.021588001063 - Q.dr_0)
        - 13.35 * max(0.0, Q.centroid_offset - 0.008092360237)
        + 172.2 * max(0.0, 0.013238675006 - Q.width)
        + 21.63 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        + 1.232 * max(0.0, 6.327378592257 - Q.log_sum_pt)
        + 4442.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 29.13 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7)
        + 14.46 * max(0.0, 0.050284641981 - Q.e2)
        + 14.75 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        - 106.4 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2)
        + 19.49 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.03362 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125)
        - 0.7255 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 59.63 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        + 3.167 * max(0.0, Q.centroid_offset - 0.049903668404)
        + 0.9989 * Q.max_dr
        - 3.757 * max(0.0, Q.C2 - 0.010539266048)
        + 265.9 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 36.65 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 0.256 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1)
        + 14.86 * max(0.0, 0.000537286005 - Q.lam2)
        - 74.69 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072)
        - 117.8 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 15.13 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 5.619 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537)
        + 24.49 * max(0.0, Q.girth - 0.087236513197)
        - 1.397 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.04463 * max(0.0, 1.679198372364 - Q.D2)
        + 0.2034 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5)
        - 34.49 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2)
        + 49.67 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 48.14 * max(0.0, Q.lam2 - 0.003408388935)
        - 3.246 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        + 0.00712 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 5.465 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5)
        + 792.7 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105)
        - 11280.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105)
        + 52.56 * max(0.0, 0.003562611155 - Q.girth2)
        + 0.02238 * max(0.0, 49.668099212646 - Q.mass)
        + 0.6563 * max(0.0, Q.girth2_top5 - 0.002270363079)
        - 0.0007462 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        + 1.019 * max(0.0, Q.z_7 - 0.06164517166)
        + 0.02908 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        + 0.3426 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        + 0.003971 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 0.6529 * max(0.0, 0.195013533663 - Q.planar_flow)
        + 260.0 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 43.23 * max(0.0, 0.001056655216 - Q.girth2_top2)
        - 434.6 * max(0.0, Q.girth2 - 0.007520088344)
        - 331.6 * max(0.0, Q.girth2 - 0.004372139461)
        + 22.43 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 3728.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 91.69 * max(0.0, Q.girth2 - 0.0016538364)
        - 9.871 * max(0.0, 0.024547699839 - Q.e2)
        + 0.01352 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6)
        - 38.61 * max(0.0, 0.04081947431 - Q.girth)
        - 0.02457 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439)
        + 0.005041 * max(0.0, 48.71875 - Q.pt_7)
        + 0.003942 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 42.43 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 156.8 * max(0.0, 0.005590288644 - Q.width)
        - 43.78 * max(0.0, 0.02076709205 - Q.centroid_offset)
        - 500.2 * max(0.0, Q.girth2 - 0.008678044751)
        + 93.57 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 43.4 * max(0.0, 0.001101266364 - Q.e2_sq)
        - 0.0009904 * max(0.0, Q.mass - 80.4)
        - 674.9 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 58.66 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 3.013 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 6.925 * max(0.0, 0.038466955721 - Q.e2)
        + 44.44 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.05378 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 0.001169 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        + 1.35 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 0.7479 * max(0.0, 0.293190627853 - Q.LHA)
        - 1.125 * max(0.0, 0.005884990035 - Q.girth2_top3)
        + 1202.0 * max(0.0, 0.000561123155 - Q.width)
        + 0.08987 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5)
        + 1612.0 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1)
        - 0.07548 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 0.5408 * max(0.0, 0.197968879342 - Q.max_dr)
        + 169.0 * max(0.0, 0.005019718802 - Q.width)
        + 8.429 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21)
        + 11580.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 6.012 * max(0.0, 0.196739721581 - Q.LHA)
        + 6.622 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 1.75 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 0.5177 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2)
        + 8.296 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0)
        - 67.25 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264)
        + 13.69 * max(0.0, 0.061086014472 - Q.girth)
        - 0.0002246 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 78.48 * max(0.0, 0.003343241496 - Q.centroid_offset)
        + 1.175 * max(0.0, 0.177304983139 - Q.max_dr)
        + 2012.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 61.45 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 0.68 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 22720.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 5068.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 9735.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 10140.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 256.3 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.03578 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 136.1 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 11730.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        - 6.262 * max(0.0, 0.016554418951 - Q.e2)
        - 145.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 8.648 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 4422.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        + 28.34 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394)
        - 2.941 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.04176 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375)
        - 8018.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 3707.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 751.2 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 436.5 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 8387.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 6250.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.2537 * max(0.0, 8.379955863953 - Q.mass)
        - 0.04743 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 12.86 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121)
        - 0.3787 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 1224.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 1126.0 * max(0.0, 0.000964142894 - Q.girth2)
        - 230.9 * max(0.0, 0.000222950415 - Q.girth2_top5)
        - 19.5 * max(0.0, 0.054649224505 - Q.girth)
        - 90.46 * max(0.0, Q.lam2 - 0.001130644719)
        + 2.358 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 127.9 * max(0.0, 0.006096650059 - Q.width)
        - 4.468 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 170.1 * max(0.0, Q.girth2 - 0.018827652745)
        - 16.02 * max(0.0, 0.032346998155 - Q.e2)
        - 247.9 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 96.08 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.1497 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 0.04708 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 0.07359 * max(0.0, 0.221586732566 - Q.max_dr)
        - 3328.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 0.7242 * max(0.0, Q.C2 - 0.051192347892)
        + 17.31 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 3355.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 8756.0 * max(0.0, 0.000172198326 - Q.width)
        - 47.41 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752)
        + 173.7 * max(0.0, 0.007520088344 - Q.girth2)
        - 2.026 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2)
        + 0.07723 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.429 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 285.1 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 148.3 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        + 34.45 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity)
        + 0.0278 * max(0.0, 41.377904891968 - Q.mass)
        - 0.07717 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 8.012 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 20.1 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 128.9 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.5279 * max(0.0, Q.max_dr - 0.15984864831)
        + 5.969 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        + 0.4194 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264)
        + 0.004319 * max(0.0, 35.5 - Q.pt_5)
        + 0.8917 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 295.7 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta)
        - 0.002904 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962)
        - 1.446 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
        + 36.36 * Q.e2
        - 436.1 * max(0.0, Q.lam2 - 0.000194798295)
        - 1.259 * max(0.0, Q.LHA - 0.303313749495)
        - 159.0 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668)
        - 3.298 * max(0.0, 6.267538488641 - Q.log_sum_pt)
        - 0.1795 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7)
        + 1.241 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        - 0.06058 * max(0.0, Q.n_dr_0p2_0p4 - 2.0)
        + 198.5 * max(0.0, 0.004183811014 - Q.lam1)
        - 159.0 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt)
        - 0.04043 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        - 267.5 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21)
        - 1.794 * max(0.0, 0.269169217348 - Q.tau32)
        + 6.307 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21)
        - 0.265 * max(0.0, 0.391541349888 - Q.tau21)
        - 52.02 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.5774 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 62.57 * max(0.0, 0.002412890926 - Q.girth2_top2)
        + 0.02064 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125)
        + 0.2834 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7)
        + 25.76 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        - 170.3 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7)
        - 13.52 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386)
        + 0.0339 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        - 887.9 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 0.322 * max(0.0, Q.mass - 15.454033088684)
        + 0.6072 * max(0.0, Q.max_dr - 0.121680960059)
        + 0.002823 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 510.7 * max(0.0, 0.0016538364 - Q.girth2)
        - 0.006179 * max(0.0, Q.sum_pt - 813.415625)
        - 89.06 * max(0.0, 0.006679471442 - Q.girth2)
        - 0.3089 * max(0.0, 0.253403707141 - Q.planar_flow)
        - 31.62 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        - 5.483 * max(0.0, 0.154689112391 - Q.LHA)
        + 27.76 * max(0.0, Q.girth - 0.076081777364)
        + 0.01043 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        - 1.655 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097)
        + 0.8864 * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 0.07756 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 1.202 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
        + 123.1 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 2.134 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.009967 * max(0.0, 7.0 - Q.n_dr_0_0p05)
        + 39.27 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2)
        + 0.08578 * Q.centroid_offset
        + 52.56 * max(0.0, 0.003562611091 - Q.width)
        - 0.001714 * max(0.0, 9.257203159811 - Q.mass_top5)
        + 72.64 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        + 0.3152 * max(0.0, 15.454033088684 - Q.mass)
        - 0.3674 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        + 0.02434 * max(0.0, 29.0421875 - Q.pt_7)
        - 43.51 * max(0.0, 0.004839980301 - Q.lam1)
        + 0.8064 * max(0.0, 0.035786485299 - Q.C2)
        - 1.013 * max(0.0, 0.111761856824 - Q.max_dr)
        + 0.07697 * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 0.0002914 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2)
        - 0.04995 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625)
        - 0.002645 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.005688 * max(0.0, Q.mass - 91.19)
        - 9162.0 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005)
        + 0.4499 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        + 2.862 * max(0.0, Q.mean_phi - 0.026127964072)
        + 3.15 * max(0.0, 0.148408418149 - Q.girth)
        + 1.734 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.1169 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 54.09 * max(0.0, 0.016433749775 - Q.lam1)
        + 0.09635 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        + 7937.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.0004525 * max(0.0, 531.1875 - Q.sum_pt_top5)
        + 6.945e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 0.001458 * max(0.0, Q.sum_pt_top5 - 902.40625)
        - 0.0001206 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 0.1228 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 4.773 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        - 0.01241 * max(0.0, 0.501026660204 - Q.tau21)
        - 247.8 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 0.6511 * max(0.0, 0.067272114405 - Q.z_6)
        + 14.23 * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 9.96 * max(0.0, Q.LHA - 0.09323897448)
        - 0.000564 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 0.0005642 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2)
        - 0.2961 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 173.7 * max(0.0, 0.00752008842 - Q.width)
        + 0.06838 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4)
        - 0.006575 * max(0.0, 25.578125 - Q.pt_7)
        - 0.001888 * max(0.0, Q.sum_pt - 988.4078125)
        + 0.0006911 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.0006497 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        - 3.549 * max(0.0, 0.111513564951 - Q.planar_flow)
        + 0.217 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 18.56 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 56.2 * max(0.0, Q.lam1 - 0.008375572068)
        - 62.35 * max(0.0, Q.lam1 - 0.004183811014)
        + 19.6 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        - 111.8 * max(0.0, Q.lam1 - 0.00543336053)
        + 0.008164 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt)
        - 118.9 * max(0.0, Q.lam1 - 0.007330079875)
        - 4.576 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        + 4810.0 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr)
        - 1.729 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        - 2133.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 6.42 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        + 0.007368 * max(0.0, 76.655700683594 - Q.mass)
        - 4.896 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164)
        - 0.09914 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 108.4 * max(0.0, Q.lam1 - 0.005954149834)
        + 0.02427 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 33.84 * max(0.0, Q.lam1 - 0.002464291268)
        + 655.6 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 0.01445 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2)
        - 184.6 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow)
        + 1215.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 138.5 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 0.284 * max(0.0, 1.122624260187 - Q.D2)
        - 23.63 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.0248 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 4.401 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 11.18 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116)
        - 0.3631 * max(0.0, 0.74595130682 - Q.D2)
        + 124.3 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 11.67 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 2.789 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2)
        + 6.395 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 55.2 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 13.26 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.665 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 3.512 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr)
        + 6.742 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 9110.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 137.0 * max(0.0, 0.012003726523 - Q.lam1)
        - 151.4 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 39.97 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.01097 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        + 2.223 * max(0.0, 0.041109715588 - Q.e2)
        - 12.41 * max(0.0, 0.063441075385 - Q.e2)
        - 0.2137 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 0.9926 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.0334 * max(0.0, 2.0 - Q.n_dr_0_0p05)
        + 1.056 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264)
        - 0.5146 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377)
        - 0.01994 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 9.757 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 678.4 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 0.705 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276)
        + 198.9 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow)
        - 81.14 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        + 0.0007903 * max(0.0, Q.n_dr_0p05_0p1 - 5.0)
        + 8.267 * max(0.0, Q.LHA - 0.346713497427)
        - 27.55 * max(0.0, Q.girth - 0.033604209498)
        + 8.549 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 1.324 * max(0.0, 0.007639643088 - Q.girth2_top2)
    )


def score_W(Q):
    return (2.596
        + 4.745 * max(0.0, 0.148419710734 - Q.planar_flow)
        + 60.04 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 472.9 * max(0.0, 0.004372139331 - Q.width)
        - 599.0 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.03858 * max(0.0, 64.618731689453 - Q.mass)
        + 0.1499 * max(0.0, 21.784077072144 - Q.mass)
        - 59.6 * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 321.7 * max(0.0, 0.013238675334 - Q.girth2)
        - 517.8 * max(0.0, 0.006506575659 - Q.lam1)
        - 707.5 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.02283 * max(0.0, Q.sum_pt - 901.59375)
        - 0.03121 * max(0.0, 56.920347213745 - Q.mass)
        - 135.8 * max(0.0, 0.008678044951 - Q.width)
        + 0.0001151 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.01006 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2)
        - 0.01144 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2)
        - 0.01639 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        - 34.27 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 1.514 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        - 0.04745 * max(0.0, 29.644699859619 - Q.mass)
        + 0.1219 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        - 52.49 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr)
        + 58.56 * max(0.0, 0.087236513197 - Q.girth)
        + 16.64 * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 10810.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101)
        + 1.317 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 30.7 * max(0.0, 0.020459658932 - Q.e2)
        - 13570.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 650.8 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        + 16.35 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.001743 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 0.2726 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6)
        + 157.7 * max(0.0, 0.00543336053 - Q.lam1)
        - 1.388 * max(0.0, Q.log_sum_pt - 6.377722943814)
        + 0.0984 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        + 132.9 * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 80.0 * max(0.0, 0.076081777364 - Q.girth)
        - 0.00047 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 3.548 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7)
        - 0.02126 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        - 164.0 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0)
        - 45.0 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 28.75 * max(0.0, 0.012569162668 - Q.planar_flow)
        + 309.3 * max(0.0, 0.008375572068 - Q.lam1)
        - 0.003075 * max(0.0, Q.pt_7 - 34.53125)
        - 6045.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        + 0.0002991 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 21.55 * max(0.0, 0.035560912266 - Q.e2)
        - 16.71 * max(0.0, 0.055577157257 - Q.z_7)
        - 0.6881 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 63.72 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.05219 * max(0.0, 53.332374954224 - Q.mass)
        + 0.0007639 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3)
        + 7.273 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 1.85 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 0.03687 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21)
        + 3362.0 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 222.6 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        - 3.013 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 168.1 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2)
        - 1.323 * max(0.0, 0.197783735394 - Q.tau21)
        + 597.7 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.1487 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.0009179 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3)
        + 4.767 * max(0.0, 0.042322802544 - Q.C2)
        - 0.01328 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6)
        + 48.89 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3)
        + 1609.0 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 4877.0 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2)
        + 1623.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 18.74 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32)
        - 39.64 * max(0.0, 0.346713497427 - Q.LHA)
        - 18.32 * max(0.0, Q.e2 - 0.032346998155)
        - 23.78 * max(0.0, Q.mass_over_sum_pt - 0.054892207095)
        - 28.14 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 13.58 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1888.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 15.65 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 8.412 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow)
        + 0.07056 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0)
        - 4.999 * max(0.0, 0.046566883102 - Q.max_dr)
        - 30.71 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32)
        + 0.6642 * max(0.0, 69.611351776123 - Q.mass)
        - 0.01191 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7)
        + 2.547 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 1.36 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 20.73 * max(0.0, 0.028070914944 - Q.z_7)
        + 155.4 * max(0.0, 0.005954149834 - Q.lam1)
        - 6023.0 * max(0.0, 9.1213921e-05 - Q.width)
        + 0.01591 * max(0.0, Q.pt_7 - 30.484375)
        - 0.1538 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        - 7.056 * max(0.0, Q.z_7 - 0.046240320761)
        - 2232.0 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width)
        - 0.5901 * max(0.0, Q.LHA - 0.111565049159)
        + 3374.0 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.059 * max(0.0, 0.694781820497 - Q.planar_flow)
        - 0.1014 * max(0.0, 36.229410171509 - Q.mass)
        - 1.27e-05 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746)
        + 7.196 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.262 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 191.3 * max(0.0, 0.003377388461 - Q.lam1)
        - 15410.0 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset)
        + 12.31 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset)
        + 1.803 * max(0.0, 0.016858545121 - Q.z_7)
        + 0.002075 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 0.004515 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7)
        + 0.005125 * max(0.0, 53.4375 - Q.pt_7)
        + 0.01084 * max(0.0, 43.5 - Q.pt_7)
        + 0.00431 * max(0.0, 788.4484375 - Q.sum_pt)
        - 0.7539 * max(0.0, Q.log_sum_pt - 6.267538488641)
        - 68.87 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 13080.0 * max(0.0, 4.8108519e-05 - Q.girth2)
        + 17.91 * max(0.0, Q.log_sum_pt - 6.804164030582)
        - 4.774 * max(0.0, 0.15984864831 - Q.max_dr)
        - 0.000355 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731)
        - 10750.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        - 73.79 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        - 11.34 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 43.2 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32)
        + 244.1 * max(0.0, Q.width - 0.018827653081)
        + 0.1132 * max(0.0, Q.mass - 36.229410171509)
        - 77.42 * max(0.0, Q.e2 - 0.028531698044)
        - 42.53 * max(0.0, Q.LHA - 0.312727471086)
        + 191.2 * max(0.0, 0.006679471358 - Q.width)
        - 0.2107 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0)
        - 31.54 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2)
        + 2.059 * max(0.0, 0.04447356835 - Q.e2)
        + 308.3 * max(0.0, 0.007330079875 - Q.lam1)
        - 6.938 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.1215 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 0.6484 * max(0.0, Q.mass - 69.611351776123)
        + 35.42 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1)
        - 135.8 * max(0.0, 0.008678044751 - Q.girth2)
        - 10.81 * max(0.0, Q.LHA - 0.325582223496)
        + 0.01994 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 61.15 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0)
        - 23.72 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828)
        + 0.0184 * max(0.0, Q.mass_top5 - 53.607658247923)
        - 2.519 * max(0.0, -0.012844925793 - Q.mean_eta)
        + 47.0 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0)
        + 155.1 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion)
        + 0.04472 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6)
        + 219.0 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 1069.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 472.9 * max(0.0, 0.004372139461 - Q.girth2)
        + 201.3 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 1051.0 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        + 15.56 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781)
        - 153.3 * max(0.0, Q.lam1 - 0.012003726523)
        - 8152.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 1.814 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr)
        - 0.168 * max(0.0, Q.max_dr - 0.145231109113)
        + 91.55 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3)
        + 1644.0 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153)
        + 0.4625 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4)
        - 242.3 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4)
        - 0.9015 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5)
        - 550.3 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153)
        + 872.8 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153)
        + 0.3037 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 1.098 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6)
        + 0.8147 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7)
        - 2.018 * max(0.0, Q.LHA - 0.423592510895)
        + 0.001414 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7)
        - 5.596 * max(0.0, 0.23799610585 - Q.tau21)
        + 0.0315 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        + 603.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 691.8 * max(0.0, 0.000306123359 - Q.lam2)
        + 5054.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        - 2563.0 * max(0.0, Q.width - 0.000319370692)
        - 0.02426 * max(0.0, 763.825 - Q.sum_pt)
        - 31.67 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2)
        - 3977.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 4.423 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        + 10.53 * max(0.0, Q.width - 0.001653836415)
        + 0.000271 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 580.2 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 3.706 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 10.82 * max(0.0, Q.C2 - 0.067292226106)
        - 903.7 * max(0.0, 0.001130644719 - Q.lam2)
        - 1.394 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 10.61 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0)
        + 54.55 * max(0.0, Q.girth - 0.101940929517)
        + 59.67 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 520.8 * max(0.0, Q.girth2 - 0.013238675334)
        + 0.000907 * max(0.0, 430.75 - Q.sum_pt_top5)
        - 56.75 * max(0.0, Q.e2 - 0.063441075385)
        + 72.74 * max(0.0, Q.e2 - 0.007078157854)
        - 0.000702 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        - 20.59 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi)
        + 0.07788 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 35.47 * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 14.11 * max(0.0, Q.max_dr - 0.102758520097)
        + 2.815 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 4.479 * max(0.0, Q.max_dr - 0.197968879342)
        - 0.3358 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        - 2.998 * max(0.0, Q.C2 - 0.014943876117)
        + 0.1361 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        + 1.966 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 1.622 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 18.41 * max(0.0, 0.047915700823 - Q.girth)
        + 0.0003089 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875)
        + 82.48 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 475.6 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        - 4.141 * max(0.0, 0.216055863061 - Q.LHA)
        - 7.129 * max(0.0, 0.049399692737 - Q.z_7)
        - 0.1476 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 10.76 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 41.96 * max(0.0, 0.00528466865 - Q.e2_sq)
        - 1.227 * max(0.0, 0.028865759995 - Q.z_6)
        - 4.018 * max(0.0, 0.071488645583 - Q.z_7)
        + 22510.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 6.777 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 0.05702 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 2840.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1)
        + 3.262 * max(0.0, 0.03243272066 - Q.z_7)
        - 5.26 * max(0.0, 0.026454043164 - Q.dr_0)
        + 3.772 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.9366 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0)
        + 4.549 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 409.3 * max(0.0, 0.002635417778 - Q.width)
        - 1215.0 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 119.1 * max(0.0, 0.002270363079 - Q.girth2_top5)
        + 0.0008002 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 1525.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0002931 * max(0.0, 548.196875 - Q.sum_pt_top2)
        - 4277.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 287.2 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.06874 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625)
        - 0.001378 * max(0.0, 24.578125 - Q.pt_5)
        + 0.0009345 * max(0.0, Q.sum_pt - 868.509375)
        - 0.07311 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 0.2799 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085)
        + 172.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 1816.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.002418 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 70.17 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        - 128.7 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        - 10.45 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2)
        - 31.52 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 110.6 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 0.06001 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 3957.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.009355 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.0486 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142)
        + 9.44 * max(0.0, 0.01426135283 - Q.mean_phi2)
        - 0.3188 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass)
        - 26.16 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 80770.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 15.25 * max(0.0, 0.021588001063 - Q.dr_0)
        + 4.745 * max(0.0, Q.centroid_offset - 0.008092360237)
        - 321.7 * max(0.0, 0.013238675006 - Q.width)
        - 46.91 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 1.174 * max(0.0, 6.327378592257 - Q.log_sum_pt)
        - 13160.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        - 143.7 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7)
        - 78.76 * max(0.0, 0.050284641981 - Q.e2)
        - 47.82 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        + 174.7 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2)
        + 19.97 * max(0.0, Q.centroid_offset - 0.018377780003)
        + 0.07727 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125)
        + 14.03 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 53.84 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        - 27.02 * max(0.0, Q.centroid_offset - 0.049903668404)
        + 4.437 * Q.max_dr
        + 0.3149 * max(0.0, Q.C2 - 0.010539266048)
        - 557.3 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        - 30.57 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 4.57 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1)
        + 207.4 * max(0.0, 0.000537286005 - Q.lam2)
        - 620.5 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072)
        + 44.14 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 109.0 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 9.832 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537)
        + 6.687 * max(0.0, Q.girth - 0.087236513197)
        + 9.693 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        + 0.1457 * max(0.0, 1.679198372364 - Q.D2)
        - 0.1039 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5)
        + 263.5 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2)
        - 92.18 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 511.6 * max(0.0, Q.lam2 - 0.003408388935)
        - 0.003387 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 0.03055 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        - 17.42 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5)
        - 1998.0 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105)
        + 17390.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105)
        - 249.8 * max(0.0, 0.003562611155 - Q.girth2)
        - 0.01239 * max(0.0, 49.668099212646 - Q.mass)
        + 28.74 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.0005897 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 4.598 * max(0.0, Q.z_7 - 0.06164517166)
        - 0.09211 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 0.5597 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.01613 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        + 0.2011 * max(0.0, 0.195013533663 - Q.planar_flow)
        + 569.5 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 414.9 * max(0.0, 0.001056655216 - Q.girth2_top2)
        + 1551.0 * max(0.0, Q.girth2 - 0.007520088344)
        + 389.3 * max(0.0, Q.girth2 - 0.004372139461)
        + 61.41 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 27020.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 10.53 * max(0.0, Q.girth2 - 0.0016538364)
        + 225.1 * max(0.0, 0.024547699839 - Q.e2)
        - 0.04165 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6)
        - 19.7 * max(0.0, 0.04081947431 - Q.girth)
        + 0.03224 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439)
        + 0.002323 * max(0.0, 48.71875 - Q.pt_7)
        + 0.005202 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 96.73 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 1071.0 * max(0.0, 0.005590288644 - Q.width)
        + 17.99 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 688.8 * max(0.0, Q.girth2 - 0.008678044751)
        - 1354.0 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 49.62 * max(0.0, 0.001101266364 - Q.e2_sq)
        + 0.02657 * max(0.0, Q.mass - 80.4)
        - 1333.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 282.8 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 46.53 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        - 6.691 * max(0.0, 0.038466955721 - Q.e2)
        - 187.1 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.01056 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        - 0.003178 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        + 6.169 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 12.52 * max(0.0, 0.293190627853 - Q.LHA)
        + 35.08 * max(0.0, 0.005884990035 - Q.girth2_top3)
        - 1790.0 * max(0.0, 0.000561123155 - Q.width)
        - 0.4553 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5)
        - 1345.0 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1)
        - 0.6924 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        - 4.214 * max(0.0, 0.197968879342 - Q.max_dr)
        - 1068.0 * max(0.0, 0.005019718802 - Q.width)
        + 292.7 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21)
        - 17410.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 21.64 * max(0.0, 0.196739721581 - Q.LHA)
        + 11.87 * max(0.0, Q.log_sum_pt - 6.701242202626)
        - 258.4 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 0.3936 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2)
        + 73.06 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0)
        + 137.1 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264)
        - 146.7 * max(0.0, 0.061086014472 - Q.girth)
        + 0.003023 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 94.44 * max(0.0, 0.003343241496 - Q.centroid_offset)
        + 1.91 * max(0.0, 0.177304983139 - Q.max_dr)
        - 4256.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 277.3 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 1.135 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 62620.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 46030.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 1469.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 9179.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        + 175.8 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.03483 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        - 236.7 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 1221.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        + 9.931 * max(0.0, 0.016554418951 - Q.e2)
        + 140.4 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 5.335 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 9780.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        + 36.89 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394)
        + 9.024 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.08166 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375)
        + 58450.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        - 29900.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        - 3590.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 1702.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 3245.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 3176.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.03564 * max(0.0, 8.379955863953 - Q.mass)
        + 0.821 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        - 74.57 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121)
        - 3.518 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 7311.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        - 1721.0 * max(0.0, 0.000964142894 - Q.girth2)
        + 1313.0 * max(0.0, 0.000222950415 - Q.girth2_top5)
        + 40.45 * max(0.0, 0.054649224505 - Q.girth)
        + 930.5 * max(0.0, Q.lam2 - 0.001130644719)
        - 1.611 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 803.5 * max(0.0, 0.006096650059 - Q.width)
        + 44.21 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 244.1 * max(0.0, Q.girth2 - 0.018827652745)
        - 45.5 * max(0.0, 0.032346998155 - Q.e2)
        + 218.4 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 67.35 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        - 0.2823 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        - 0.01958 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 5.314 * max(0.0, 0.221586732566 - Q.max_dr)
        + 3979.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        - 13.31 * max(0.0, Q.C2 - 0.051192347892)
        + 33.25 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 10190.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 1191.0 * max(0.0, 0.000172198326 - Q.width)
        + 23.73 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752)
        + 738.7 * max(0.0, 0.007520088344 - Q.girth2)
        + 2.273 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 0.136 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.2538 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 203.0 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        + 109.0 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        - 42.66 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity)
        - 0.02591 * max(0.0, 41.377904891968 - Q.mass)
        + 0.1323 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 10.72 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 15.53 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        + 1037.0 * max(0.0, 0.001503553356 - Q.lam1)
        - 2.712 * max(0.0, Q.max_dr - 0.15984864831)
        - 7.362 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 0.159 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264)
        - 0.001239 * max(0.0, 35.5 - Q.pt_5)
        - 0.9536 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        + 148.3 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta)
        + 0.0004207 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962)
        + 5.433 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
        + 36.02 * Q.e2
        - 1466.0 * max(0.0, Q.lam2 - 0.000194798295)
        + 44.64 * max(0.0, Q.LHA - 0.303313749495)
        - 261.5 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668)
        - 1.391 * max(0.0, 6.267538488641 - Q.log_sum_pt)
        - 0.4228 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7)
        + 8.349 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 0.368 * max(0.0, Q.n_dr_0p2_0p4 - 2.0)
        + 276.9 * max(0.0, 0.004183811014 - Q.lam1)
        + 1207.0 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt)
        - 0.02486 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 58.93 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21)
        - 1.19 * max(0.0, 0.269169217348 - Q.tau32)
        - 17.76 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21)
        + 0.5682 * max(0.0, 0.391541349888 - Q.tau21)
        + 106.0 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.4347 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.3407 * max(0.0, 0.002412890926 - Q.girth2_top2)
        + 0.007904 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125)
        - 0.0395 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7)
        - 102.9 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        + 230.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7)
        - 11.21 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386)
        - 1.773 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        + 1569.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.5736 * max(0.0, Q.mass - 15.454033088684)
        - 5.885 * max(0.0, Q.max_dr - 0.121680960059)
        - 0.007545 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 675.6 * max(0.0, 0.0016538364 - Q.girth2)
        - 0.0288 * max(0.0, Q.sum_pt - 813.415625)
        + 191.2 * max(0.0, 0.006679471442 - Q.girth2)
        + 0.4348 * max(0.0, 0.253403707141 - Q.planar_flow)
        - 169.1 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        - 8.774 * max(0.0, 0.154689112391 - Q.LHA)
        - 138.5 * max(0.0, Q.girth - 0.076081777364)
        - 0.008435 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        + 1.053 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097)
        + 1.376 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.3142 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        - 0.3758 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
        - 527.6 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 8.792 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.1129 * max(0.0, 7.0 - Q.n_dr_0_0p05)
        - 223.2 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2)
        - 12.49 * Q.centroid_offset
        - 249.8 * max(0.0, 0.003562611091 - Q.width)
        + 0.01198 * max(0.0, 9.257203159811 - Q.mass_top5)
        + 129.4 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        - 0.5962 * max(0.0, 15.454033088684 - Q.mass)
        - 0.6022 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.04701 * max(0.0, 29.0421875 - Q.pt_7)
        + 181.7 * max(0.0, 0.004839980301 - Q.lam1)
        - 19.81 * max(0.0, 0.035786485299 - Q.C2)
        + 7.602 * max(0.0, 0.111761856824 - Q.max_dr)
        - 0.3496 * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.0008226 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2)
        - 0.09359 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625)
        - 0.01137 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.04002 * max(0.0, Q.mass - 91.19)
        - 5148.0 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005)
        + 1.964 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 8.307 * max(0.0, Q.mean_phi - 0.026127964072)
        + 22.85 * max(0.0, 0.148408418149 - Q.girth)
        + 15.78 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.2027 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 0.8365 * max(0.0, 0.016433749775 - Q.lam1)
        - 0.1847 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 9414.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.0001535 * max(0.0, 531.1875 - Q.sum_pt_top5)
        - 4.606e-07 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        - 0.0004769 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.0003385 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 0.3193 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 4.833 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        + 0.01495 * max(0.0, 0.501026660204 - Q.tau21)
        - 1.095 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 2.404 * max(0.0, 0.067272114405 - Q.z_6)
        - 52.8 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 5.337 * max(0.0, Q.LHA - 0.09323897448)
        - 0.0002904 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        + 0.004667 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2)
        - 4.737 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 738.8 * max(0.0, 0.00752008842 - Q.width)
        + 0.3694 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4)
        - 0.01216 * max(0.0, 25.578125 - Q.pt_7)
        + 0.05094 * max(0.0, Q.sum_pt - 988.4078125)
        + 0.001236 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.0001227 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        + 2.434 * max(0.0, 0.111513564951 - Q.planar_flow)
        - 0.5405 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 248.5 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 214.9 * max(0.0, Q.lam1 - 0.008375572068)
        - 212.5 * max(0.0, Q.lam1 - 0.004183811014)
        + 1.342 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        - 311.7 * max(0.0, Q.lam1 - 0.00543336053)
        + 0.0005186 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt)
        - 226.4 * max(0.0, Q.lam1 - 0.007330079875)
        - 4.095 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        - 58530.0 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr)
        + 6.15 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 9618.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        - 86.61 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        - 0.01343 * max(0.0, 76.655700683594 - Q.mass)
        - 2.946 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164)
        - 3.747 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 333.7 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.1385 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 743.6 * max(0.0, Q.lam1 - 0.002464291268)
        - 778.9 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 0.001434 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2)
        + 435.8 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow)
        - 1641.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        - 1317.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 0.9754 * max(0.0, 1.122624260187 - Q.D2)
        + 58.68 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.01316 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        - 6.696 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 32.38 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116)
        + 0.6841 * max(0.0, 0.74595130682 - Q.D2)
        - 629.1 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 1260.0 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 57.39 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2)
        - 77.83 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 26.86 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 47.01 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 5.751 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 44.49 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr)
        - 24.22 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 82980.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 372.0 * max(0.0, 0.012003726523 - Q.lam1)
        + 94.17 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 14910.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        - 0.01461 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        - 114.1 * max(0.0, 0.041109715588 - Q.e2)
        - 41.54 * max(0.0, 0.063441075385 - Q.e2)
        - 0.06811 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        - 0.4728 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.01199 * max(0.0, 2.0 - Q.n_dr_0_0p05)
        + 3.548 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264)
        - 4.099 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377)
        + 0.2615 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 28.71 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 5229.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.624 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276)
        + 571.5 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow)
        + 68.79 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.04019 * max(0.0, Q.n_dr_0p05_0p1 - 5.0)
        + 71.58 * max(0.0, Q.LHA - 0.346713497427)
        - 57.53 * max(0.0, Q.girth - 0.033604209498)
        + 3.634 * max(0.0, 0.002151567843 - Q.girth2_top3)
        + 8.575 * max(0.0, 0.007639643088 - Q.girth2_top2)
    )


def score_Z(Q):
    return (0.1038
        - 0.02939 * max(0.0, 0.148419710734 - Q.planar_flow)
        + 152.7 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 246.3 * max(0.0, 0.004372139331 - Q.width)
        - 60.79 * max(0.0, 0.018827652745 - Q.girth2)
        - 0.01941 * max(0.0, 64.618731689453 - Q.mass)
        + 0.01243 * max(0.0, 21.784077072144 - Q.mass)
        - 10.74 * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 267.7 * max(0.0, 0.013238675334 - Q.girth2)
        + 66.66 * max(0.0, 0.006506575659 - Q.lam1)
        + 365.2 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        - 0.03106 * max(0.0, Q.sum_pt - 901.59375)
        + 0.002325 * max(0.0, 56.920347213745 - Q.mass)
        + 604.3 * max(0.0, 0.008678044951 - Q.width)
        - 2.659e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        + 0.001917 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2)
        - 0.00715 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2)
        - 0.2435 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        + 52.83 * max(0.0, 0.00832969537 - Q.girth2_top5)
        - 0.2752 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        + 0.01629 * max(0.0, 29.644699859619 - Q.mass)
        - 0.06619 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        + 66.71 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr)
        - 36.43 * max(0.0, 0.087236513197 - Q.girth)
        - 8.207 * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 10060.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101)
        + 0.8704 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 74.66 * max(0.0, 0.020459658932 - Q.e2)
        - 5683.0 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        - 1266.0 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        + 19.97 * max(0.0, Q.log_sum_pt - 6.638338705138)
        - 0.002041 * max(0.0, Q.sum_pt_top5 - 687.4375)
        + 0.02581 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6)
        + 12.46 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.4835 * max(0.0, Q.log_sum_pt - 6.377722943814)
        + 0.09754 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        + 46.67 * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 6.007 * max(0.0, 0.076081777364 - Q.girth)
        + 0.0004427 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 6.46 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7)
        - 0.01385 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        + 151.1 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0)
        + 74.61 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        - 1.476 * max(0.0, 0.012569162668 - Q.planar_flow)
        - 140.1 * max(0.0, 0.008375572068 - Q.lam1)
        + 0.03154 * max(0.0, Q.pt_7 - 34.53125)
        - 1162.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.0003099 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 48.33 * max(0.0, 0.035560912266 - Q.e2)
        - 10.78 * max(0.0, 0.055577157257 - Q.z_7)
        - 0.2283 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 114.0 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.02228 * max(0.0, 53.332374954224 - Q.mass)
        + 0.00181 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3)
        + 1.613 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 6.011 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 0.06066 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21)
        + 708.7 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 1315.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        + 2.774 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 185.3 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2)
        + 0.3442 * max(0.0, 0.197783735394 - Q.tau21)
        - 415.7 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.1472 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.001949 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3)
        - 6.504 * max(0.0, 0.042322802544 - Q.C2)
        - 0.02001 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6)
        - 150.5 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3)
        + 518.2 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 0.733 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2)
        + 1469.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        - 13.72 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32)
        - 27.34 * max(0.0, 0.346713497427 - Q.LHA)
        + 5.322 * max(0.0, Q.e2 - 0.032346998155)
        - 39.96 * max(0.0, Q.mass_over_sum_pt - 0.054892207095)
        - 23.99 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 13.03 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 1513.0 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 25.4 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 15.05 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow)
        - 0.7188 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 2.902 * max(0.0, 0.046566883102 - Q.max_dr)
        - 35.48 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32)
        + 0.1507 * max(0.0, 69.611351776123 - Q.mass)
        + 0.06055 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7)
        + 0.8133 * max(0.0, Q.log_sum_pt - 6.842716632804)
        - 1.454 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        - 8.031 * max(0.0, 0.028070914944 - Q.z_7)
        + 53.81 * max(0.0, 0.005954149834 - Q.lam1)
        - 1813.0 * max(0.0, 9.1213921e-05 - Q.width)
        + 0.03156 * max(0.0, Q.pt_7 - 30.484375)
        - 0.5827 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        - 5.262 * max(0.0, Q.z_7 - 0.046240320761)
        + 2681.0 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width)
        + 1.065 * max(0.0, Q.LHA - 0.111565049159)
        + 53.37 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        + 0.05708 * max(0.0, 0.694781820497 - Q.planar_flow)
        + 0.01125 * max(0.0, 36.229410171509 - Q.mass)
        - 0.000143 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746)
        - 9.355 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.1666 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        + 258.6 * max(0.0, 0.003377388461 - Q.lam1)
        - 5984.0 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset)
        + 4.761 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset)
        + 5.405 * max(0.0, 0.016858545121 - Q.z_7)
        - 0.0007342 * max(0.0, 687.4375 - Q.sum_pt_top5)
        - 0.003174 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7)
        + 0.002624 * max(0.0, 53.4375 - Q.pt_7)
        + 0.009957 * max(0.0, 43.5 - Q.pt_7)
        + 0.002853 * max(0.0, 788.4484375 - Q.sum_pt)
        - 4.687 * max(0.0, Q.log_sum_pt - 6.267538488641)
        - 43.87 * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 3617.0 * max(0.0, 4.8108519e-05 - Q.girth2)
        + 30.54 * max(0.0, Q.log_sum_pt - 6.804164030582)
        + 4.581 * max(0.0, 0.15984864831 - Q.max_dr)
        - 9.451e-05 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731)
        - 4367.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        + 1.678 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 27.59 * max(0.0, Q.centroid_offset - 0.014379521101)
        + 65.14 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32)
        + 54.62 * max(0.0, Q.width - 0.018827653081)
        + 0.02521 * max(0.0, Q.mass - 36.229410171509)
        - 52.69 * max(0.0, Q.e2 - 0.028531698044)
        - 1.87 * max(0.0, Q.LHA - 0.312727471086)
        - 263.2 * max(0.0, 0.006679471358 - Q.width)
        - 0.4297 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0)
        - 14.47 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2)
        + 31.79 * max(0.0, 0.04447356835 - Q.e2)
        - 251.4 * max(0.0, 0.007330079875 - Q.lam1)
        - 2.837 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        + 0.1291 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 0.1209 * max(0.0, Q.mass - 69.611351776123)
        + 10.87 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1)
        + 604.2 * max(0.0, 0.008678044751 - Q.girth2)
        - 8.48 * max(0.0, Q.LHA - 0.325582223496)
        - 0.01171 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 151.9 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0)
        - 23.73 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828)
        + 0.02023 * max(0.0, Q.mass_top5 - 53.607658247923)
        - 3.405 * max(0.0, -0.012844925793 - Q.mean_eta)
        + 65.0 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0)
        - 114.9 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion)
        + 0.09752 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6)
        + 383.2 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        + 462.2 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 246.3 * max(0.0, 0.004372139461 - Q.girth2)
        - 134.5 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        + 473.1 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        + 3.833 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781)
        + 95.44 * max(0.0, Q.lam1 - 0.012003726523)
        - 3428.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        - 1.046 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr)
        - 3.667 * max(0.0, Q.max_dr - 0.145231109113)
        + 96.14 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3)
        + 3933.0 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153)
        + 0.6436 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4)
        - 324.0 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4)
        + 0.2231 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5)
        - 319.7 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153)
        - 156.5 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153)
        + 0.1842 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 0.5568 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6)
        + 0.5682 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7)
        + 1.422 * max(0.0, Q.LHA - 0.423592510895)
        + 0.000799 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7)
        - 2.569 * max(0.0, 0.23799610585 - Q.tau21)
        - 0.001329 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        + 1424.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        + 748.1 * max(0.0, 0.000306123359 - Q.lam2)
        + 528.6 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        - 526.6 * max(0.0, Q.width - 0.000319370692)
        - 0.02678 * max(0.0, 763.825 - Q.sum_pt)
        - 802.9 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2)
        - 1420.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 3.596 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        + 110.9 * max(0.0, Q.width - 0.001653836415)
        - 0.0002067 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 316.2 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 5.638 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        - 6.694 * max(0.0, Q.C2 - 0.067292226106)
        + 55.65 * max(0.0, 0.001130644719 - Q.lam2)
        - 13.43 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 57.05 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0)
        + 81.08 * max(0.0, Q.girth - 0.101940929517)
        - 354.9 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 300.7 * max(0.0, Q.girth2 - 0.013238675334)
        + 0.001959 * max(0.0, 430.75 - Q.sum_pt_top5)
        + 4.798 * max(0.0, Q.e2 - 0.063441075385)
        - 3.602 * max(0.0, Q.e2 - 0.007078157854)
        + 0.01144 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        - 46.19 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi)
        + 0.06316 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        + 45.19 * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 9.357 * max(0.0, Q.max_dr - 0.102758520097)
        + 4.457 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 7.723 * max(0.0, Q.max_dr - 0.197968879342)
        - 0.3751 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        + 5.592 * max(0.0, Q.C2 - 0.014943876117)
        + 0.9494 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 3.038 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 22.17 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        - 5.59 * max(0.0, 0.047915700823 - Q.girth)
        + 0.003004 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875)
        + 131.6 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 143.4 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        + 0.8417 * max(0.0, 0.216055863061 - Q.LHA)
        - 9.968 * max(0.0, 0.049399692737 - Q.z_7)
        - 0.03654 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 9.35 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 3.893 * max(0.0, 0.00528466865 - Q.e2_sq)
        + 0.1122 * max(0.0, 0.028865759995 - Q.z_6)
        + 3.654 * max(0.0, 0.071488645583 - Q.z_7)
        + 18840.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        - 4.658 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 0.06275 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 2443.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1)
        + 0.8088 * max(0.0, 0.03243272066 - Q.z_7)
        - 4.747 * max(0.0, 0.026454043164 - Q.dr_0)
        - 15.73 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 2.73 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0)
        + 0.8224 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 431.5 * max(0.0, 0.002635417778 - Q.width)
        - 462.4 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        - 31.99 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 2.108e-05 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 4872.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0004389 * max(0.0, 548.196875 - Q.sum_pt_top2)
        - 13980.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 106.8 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 0.03623 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625)
        - 8.627e-05 * max(0.0, 24.578125 - Q.pt_5)
        + 0.0004652 * max(0.0, Q.sum_pt - 868.509375)
        + 0.05557 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        - 0.07884 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085)
        + 91.66 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 1628.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        + 0.006826 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 30.57 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        + 15.85 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 52.03 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2)
        + 84.14 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 16.54 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        - 0.09749 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 3492.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        + 0.06198 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.06917 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142)
        + 5.837 * max(0.0, 0.01426135283 - Q.mean_phi2)
        - 0.08952 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass)
        - 21.0 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 39950.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 8.77 * max(0.0, 0.021588001063 - Q.dr_0)
        - 4.398 * max(0.0, Q.centroid_offset - 0.008092360237)
        + 267.8 * max(0.0, 0.013238675006 - Q.width)
        - 73.96 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 1.789 * max(0.0, 6.327378592257 - Q.log_sum_pt)
        - 4278.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        - 173.4 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7)
        - 145.7 * max(0.0, 0.050284641981 - Q.e2)
        - 21.07 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        + 46.94 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2)
        - 13.04 * max(0.0, Q.centroid_offset - 0.018377780003)
        + 0.07284 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125)
        + 9.671 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        - 89.61 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        - 5.643 * max(0.0, Q.centroid_offset - 0.049903668404)
        - 6.809 * Q.max_dr
        - 4.08 * max(0.0, Q.C2 - 0.010539266048)
        - 889.5 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        - 31.87 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 2.978 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1)
        + 124.1 * max(0.0, 0.000537286005 - Q.lam2)
        - 1360.0 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072)
        + 9.522 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        - 121.4 * max(0.0, Q.girth2_top5 - 0.011482925368)
        + 14.46 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537)
        - 73.41 * max(0.0, Q.girth - 0.087236513197)
        + 12.13 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.07714 * max(0.0, 1.679198372364 - Q.D2)
        - 0.1196 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5)
        - 164.4 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2)
        - 62.12 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        + 272.1 * max(0.0, Q.lam2 - 0.003408388935)
        + 3.311 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 0.05501 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        - 18.01 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5)
        - 2940.0 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105)
        + 39680.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105)
        - 4.687 * max(0.0, 0.003562611155 - Q.girth2)
        + 0.01347 * max(0.0, 49.668099212646 - Q.mass)
        + 67.29 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.002144 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        - 3.71 * max(0.0, Q.z_7 - 0.06164517166)
        - 0.1049 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 1.41 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.02265 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 2.854 * max(0.0, 0.195013533663 - Q.planar_flow)
        - 2382.0 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 316.4 * max(0.0, 0.001056655216 - Q.girth2_top2)
        - 300.3 * max(0.0, Q.girth2 - 0.007520088344)
        - 168.2 * max(0.0, Q.girth2 - 0.004372139461)
        + 44.39 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        + 21930.0 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 110.9 * max(0.0, Q.girth2 - 0.0016538364)
        + 153.9 * max(0.0, 0.024547699839 - Q.e2)
        - 0.07848 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6)
        - 13.61 * max(0.0, 0.04081947431 - Q.girth)
        + 0.06503 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439)
        - 0.01059 * max(0.0, 48.71875 - Q.pt_7)
        + 0.01023 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 54.22 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 684.3 * max(0.0, 0.005590288644 - Q.width)
        + 7.658 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 677.1 * max(0.0, Q.girth2 - 0.008678044751)
        - 447.7 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        + 6.601 * max(0.0, 0.001101266364 - Q.e2_sq)
        - 0.05117 * max(0.0, Q.mass - 80.4)
        + 7185.0 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 417.6 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        + 108.4 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 93.47 * max(0.0, 0.038466955721 - Q.e2)
        + 464.0 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        - 0.4264 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        - 0.006554 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 6.434 * max(0.0, Q.centroid_offset - 0.031170772021)
        + 18.65 * max(0.0, 0.293190627853 - Q.LHA)
        + 83.01 * max(0.0, 0.005884990035 - Q.girth2_top3)
        - 43.84 * max(0.0, 0.000561123155 - Q.width)
        - 1.461 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5)
        - 3677.0 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1)
        - 0.6449 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        + 0.9872 * max(0.0, 0.197968879342 - Q.max_dr)
        - 406.2 * max(0.0, 0.005019718802 - Q.width)
        - 71.4 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21)
        - 4560.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 5.737 * max(0.0, 0.196739721581 - Q.LHA)
        - 0.009019 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 88.07 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 0.4086 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2)
        + 12.18 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0)
        + 1.632 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264)
        - 46.46 * max(0.0, 0.061086014472 - Q.girth)
        + 0.001836 * max(0.0, Q.sum_pt_top5 - 658.125)
        - 13.42 * max(0.0, 0.003343241496 - Q.centroid_offset)
        - 8.942 * max(0.0, 0.177304983139 - Q.max_dr)
        + 585.9 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        - 342.9 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 3.946 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 11800.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 10230.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        + 5318.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        - 6394.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 11.53 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.08741 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        - 220.7 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        - 3242.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        - 11.5 * max(0.0, 0.016554418951 - Q.e2)
        - 10.53 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.8808 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 7291.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        + 7.287 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394)
        + 2.062 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.08674 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375)
        + 20320.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 6325.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        - 43.34 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 1764.0 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 25520.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 4195.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.02216 * max(0.0, 8.379955863953 - Q.mass)
        - 0.8784 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        - 58.6 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121)
        - 0.5105 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 456.8 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        - 844.1 * max(0.0, 0.000964142894 - Q.girth2)
        - 193.0 * max(0.0, 0.000222950415 - Q.girth2_top5)
        + 6.898 * max(0.0, 0.054649224505 - Q.girth)
        + 90.64 * max(0.0, Q.lam2 - 0.001130644719)
        + 0.4906 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        - 197.7 * max(0.0, 0.006096650059 - Q.width)
        + 19.86 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        + 54.61 * max(0.0, Q.girth2 - 0.018827652745)
        - 52.14 * max(0.0, 0.032346998155 - Q.e2)
        + 61.84 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 80.89 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        - 0.239 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        - 0.03131 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 2.326 * max(0.0, 0.221586732566 - Q.max_dr)
        + 501.5 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 8.401 * max(0.0, Q.C2 - 0.051192347892)
        - 4.106 * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 902.2 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        + 462.6 * max(0.0, 0.000172198326 - Q.width)
        + 19.43 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752)
        - 379.5 * max(0.0, 0.007520088344 - Q.girth2)
        - 0.0284 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 0.06998 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 0.2706 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        - 189.9 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 195.3 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        - 11.03 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity)
        - 0.01472 * max(0.0, 41.377904891968 - Q.mass)
        + 0.09499 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 48.08 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 23.02 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 89.72 * max(0.0, 0.001503553356 - Q.lam1)
        - 0.4863 * max(0.0, Q.max_dr - 0.15984864831)
        - 6.466 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        - 0.2188 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264)
        - 0.00149 * max(0.0, 35.5 - Q.pt_5)
        - 1.614 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        + 89.73 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta)
        + 0.0001306 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962)
        - 1.759 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
        + 65.9 * Q.e2
        + 234.6 * max(0.0, Q.lam2 - 0.000194798295)
        + 8.925 * max(0.0, Q.LHA - 0.303313749495)
        - 450.4 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668)
        + 2.328 * max(0.0, 6.267538488641 - Q.log_sum_pt)
        - 0.0107 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7)
        + 9.549 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 0.2159 * max(0.0, Q.n_dr_0p2_0p4 - 2.0)
        - 31.42 * max(0.0, 0.004183811014 - Q.lam1)
        + 445.0 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt)
        - 0.01821 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 180.2 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21)
        - 0.5758 * max(0.0, 0.269169217348 - Q.tau32)
        - 13.92 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21)
        + 0.3016 * max(0.0, 0.391541349888 - Q.tau21)
        + 6.844 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.1084 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 12.51 * max(0.0, 0.002412890926 - Q.girth2_top2)
        + 0.006641 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125)
        - 7.699 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7)
        - 62.72 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        + 245.5 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7)
        - 8.221 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386)
        - 1.091 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        + 183.0 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.1171 * max(0.0, Q.mass - 15.454033088684)
        - 0.3351 * max(0.0, Q.max_dr - 0.121680960059)
        - 0.003138 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 80.66 * max(0.0, 0.0016538364 - Q.girth2)
        - 0.006364 * max(0.0, Q.sum_pt - 813.415625)
        - 263.2 * max(0.0, 0.006679471442 - Q.girth2)
        + 0.05668 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 216.2 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        + 2.293 * max(0.0, 0.154689112391 - Q.LHA)
        - 28.06 * max(0.0, Q.girth - 0.076081777364)
        + 0.003504 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        + 7.558 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097)
        - 8.169 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.4507 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 0.1545 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
        + 211.6 * max(0.0, 0.006390124748 - Q.e2_sq)
        + 14.65 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.04231 * max(0.0, 7.0 - Q.n_dr_0_0p05)
        + 124.0 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2)
        - 0.08048 * Q.centroid_offset
        - 4.682 * max(0.0, 0.003562611091 - Q.width)
        + 0.001921 * max(0.0, 9.257203159811 - Q.mass_top5)
        - 46.3 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        - 0.0846 * max(0.0, 15.454033088684 - Q.mass)
        - 1.603 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.02136 * max(0.0, 29.0421875 - Q.pt_7)
        + 71.61 * max(0.0, 0.004839980301 - Q.lam1)
        + 8.929 * max(0.0, 0.035786485299 - Q.C2)
        - 7.547 * max(0.0, 0.111761856824 - Q.max_dr)
        - 0.1767 * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.0003452 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2)
        - 0.01037 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625)
        - 0.008036 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 0.0531 * max(0.0, Q.mass - 91.19)
        - 3606.0 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005)
        + 0.7949 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 6.998 * max(0.0, Q.mean_phi - 0.026127964072)
        + 51.76 * max(0.0, 0.148408418149 - Q.girth)
        - 14.52 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.04254 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        + 44.04 * max(0.0, 0.016433749775 - Q.lam1)
        - 0.1381 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        - 46310.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 7.377e-05 * max(0.0, 531.1875 - Q.sum_pt_top5)
        + 4.59e-05 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 0.002111 * max(0.0, Q.sum_pt_top5 - 902.40625)
        + 0.000568 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 0.08673 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        + 10.24 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        + 0.7237 * max(0.0, 0.501026660204 - Q.tau21)
        - 3365.0 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 0.7559 * max(0.0, 0.067272114405 - Q.z_6)
        - 3.638 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 1.768 * max(0.0, Q.LHA - 0.09323897448)
        - 0.0002175 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        - 0.002338 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2)
        - 3.388 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        - 379.5 * max(0.0, 0.00752008842 - Q.width)
        + 0.5137 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4)
        - 0.00285 * max(0.0, 25.578125 - Q.pt_7)
        + 0.0353 * max(0.0, Q.sum_pt - 988.4078125)
        + 0.0001413 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 0.0003963 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        + 0.2153 * max(0.0, 0.111513564951 - Q.planar_flow)
        + 0.1445 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        - 341.9 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 147.2 * max(0.0, Q.lam1 - 0.008375572068)
        - 14.6 * max(0.0, Q.lam1 - 0.004183811014)
        - 53.18 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 28.5 * max(0.0, Q.lam1 - 0.00543336053)
        - 0.01339 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt)
        - 243.3 * max(0.0, Q.lam1 - 0.007330079875)
        - 11.6 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        + 33670.0 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr)
        - 3.853 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        + 1718.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        + 413.5 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        - 0.05099 * max(0.0, 76.655700683594 - Q.mass)
        - 2.915 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164)
        + 5.239 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 62.75 * max(0.0, Q.lam1 - 0.005954149834)
        + 0.2187 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 47.56 * max(0.0, Q.lam1 - 0.002464291268)
        + 188.0 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        - 0.02195 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2)
        - 763.7 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow)
        + 1820.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 940.1 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 1.146 * max(0.0, 1.122624260187 - Q.D2)
        - 38.19 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        - 0.002984 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        + 16.62 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 14.17 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116)
        + 0.7736 * max(0.0, 0.74595130682 - Q.D2)
        + 167.2 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 1210.0 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 264.1 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2)
        + 99.7 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 131.8 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 9.288 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 1.224 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 19.98 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr)
        - 3.853 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 54560.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        + 108.9 * max(0.0, 0.012003726523 - Q.lam1)
        + 331.4 * max(0.0, 0.011657374702 - Q.e2_sq)
        + 6054.0 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        - 0.00178 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        + 10.73 * max(0.0, 0.041109715588 - Q.e2)
        - 54.12 * max(0.0, 0.063441075385 - Q.e2)
        + 1.514 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        - 0.542 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.03857 * max(0.0, 2.0 - Q.n_dr_0_0p05)
        + 0.06663 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264)
        + 1.199 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377)
        - 0.05786 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 3.441 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        + 1775.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 0.3224 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276)
        - 418.7 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow)
        - 16.65 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.04689 * max(0.0, Q.n_dr_0p05_0p1 - 5.0)
        + 24.28 * max(0.0, Q.LHA - 0.346713497427)
        - 9.142 * max(0.0, Q.girth - 0.033604209498)
        - 39.46 * max(0.0, 0.002151567843 - Q.girth2_top3)
        + 9.982 * max(0.0, 0.007639643088 - Q.girth2_top2)
    )


def score_t(Q):
    return (-0.4271
        - 0.9698 * max(0.0, 0.148419710734 - Q.planar_flow)
        + 45.13 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 24.14 * max(0.0, 0.004372139331 - Q.width)
        - 55.78 * max(0.0, 0.018827652745 - Q.girth2)
        + 0.01063 * max(0.0, 64.618731689453 - Q.mass)
        - 0.003334 * max(0.0, 21.784077072144 - Q.mass)
        - 1.117 * max(0.0, 0.007929074034 - Q.girth2_top3)
        - 33.11 * max(0.0, 0.013238675334 - Q.girth2)
        - 168.9 * max(0.0, 0.006506575659 - Q.lam1)
        - 157.8 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2)
        + 0.13 * max(0.0, Q.sum_pt - 901.59375)
        + 0.01919 * max(0.0, 56.920347213745 - Q.mass)
        - 28.9 * max(0.0, 0.008678044951 - Q.width)
        + 5.686e-05 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7)
        - 0.002388 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2)
        + 0.06352 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2)
        - 0.1554 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605)
        - 6.65 * max(0.0, 0.00832969537 - Q.girth2_top5)
        + 0.2685 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971)
        - 0.008255 * max(0.0, 29.644699859619 - Q.mass)
        - 0.02736 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2)
        - 5.309 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr)
        + 21.43 * max(0.0, 0.087236513197 - Q.girth)
        - 4.913 * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 1155.0 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101)
        - 0.1557 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117)
        + 7.612 * max(0.0, 0.020459658932 - Q.e2)
        + 654.4 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq)
        + 6.697 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153)
        - 7.251 * max(0.0, Q.log_sum_pt - 6.638338705138)
        + 0.003544 * max(0.0, Q.sum_pt_top5 - 687.4375)
        - 0.251 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6)
        - 22.68 * max(0.0, 0.00543336053 - Q.lam1)
        - 0.06694 * max(0.0, Q.log_sum_pt - 6.377722943814)
        - 0.1416 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189)
        - 6.491 * max(0.0, 0.003952581551 - Q.girth2_top3)
        + 26.83 * max(0.0, 0.076081777364 - Q.girth)
        + 8.164e-05 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7)
        - 2.05 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7)
        - 0.02846 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539)
        + 11.91 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0)
        - 37.87 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt)
        + 1.872 * max(0.0, 0.012569162668 - Q.planar_flow)
        - 47.06 * max(0.0, 0.008375572068 - Q.lam1)
        - 0.005062 * max(0.0, Q.pt_7 - 34.53125)
        + 164.0 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205)
        - 0.000122 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass)
        - 2.758 * max(0.0, 0.035560912266 - Q.e2)
        - 0.7851 * max(0.0, 0.055577157257 - Q.z_7)
        + 0.09072 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101)
        + 32.71 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 0.02 * max(0.0, 53.332374954224 - Q.mass)
        + 0.0002393 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3)
        + 0.2335 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr)
        - 0.02563 * max(0.0, Q.log_sum_pt - 6.572937922293)
        - 0.007745 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21)
        - 197.9 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 326.8 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925)
        - 1.017 * max(0.0, 0.083662731125 - Q.planar_flow)
        + 27.62 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2)
        + 0.5065 * max(0.0, 0.197783735394 - Q.tau21)
        + 148.6 * max(0.0, 0.008168570676 - Q.e2_sq)
        - 0.1977 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr)
        - 0.0002758 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3)
        - 0.4604 * max(0.0, 0.042322802544 - Q.C2)
        + 0.002464 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6)
        - 36.29 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3)
        - 387.4 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3)
        + 1420.0 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2)
        + 142.4 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2)
        + 17.41 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32)
        + 7.337 * max(0.0, 0.346713497427 - Q.LHA)
        + 8.173 * max(0.0, Q.e2 - 0.032346998155)
        - 6.037 * max(0.0, Q.mass_over_sum_pt - 0.054892207095)
        + 11.72 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 5.006 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 746.7 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow)
        + 16.74 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 7.103 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow)
        + 0.5031 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0)
        + 1.621 * max(0.0, 0.046566883102 - Q.max_dr)
        + 0.8886 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32)
        + 0.1513 * max(0.0, 69.611351776123 - Q.mass)
        + 0.04525 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7)
        - 6.157 * max(0.0, Q.log_sum_pt - 6.842716632804)
        + 0.05475 * max(0.0, 6.464150123592 - Q.log_sum_pt)
        + 33.43 * max(0.0, 0.028070914944 - Q.z_7)
        - 37.48 * max(0.0, 0.005954149834 - Q.lam1)
        + 4050.0 * max(0.0, 9.1213921e-05 - Q.width)
        + 0.02235 * max(0.0, Q.pt_7 - 30.484375)
        - 0.2177 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2)
        + 0.5657 * max(0.0, Q.z_7 - 0.046240320761)
        - 1092.0 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width)
        + 0.7226 * max(0.0, Q.LHA - 0.111565049159)
        - 612.2 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025)
        - 0.0405 * max(0.0, 0.694781820497 - Q.planar_flow)
        - 0.005856 * max(0.0, 36.229410171509 - Q.mass)
        - 6.839e-06 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746)
        - 6.505 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.03758 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299)
        - 13.62 * max(0.0, 0.003377388461 - Q.lam1)
        - 6039.0 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset)
        - 11.96 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset)
        - 3.231 * max(0.0, 0.016858545121 - Q.z_7)
        - 0.0008832 * max(0.0, 687.4375 - Q.sum_pt_top5)
        + 0.004709 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7)
        + 0.002361 * max(0.0, 53.4375 - Q.pt_7)
        - 0.0008019 * max(0.0, 43.5 - Q.pt_7)
        - 0.0007498 * max(0.0, 788.4484375 - Q.sum_pt)
        - 1.096 * max(0.0, Q.log_sum_pt - 6.267538488641)
        + 163.4 * max(0.0, Q.log_sum_pt - 6.896095378249)
        + 1264.0 * max(0.0, 4.8108519e-05 - Q.girth2)
        - 118.8 * max(0.0, Q.log_sum_pt - 6.804164030582)
        - 0.3284 * max(0.0, 0.15984864831 - Q.max_dr)
        + 3.17e-05 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731)
        + 4135.0 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7)
        - 12.37 * max(0.0, Q.mass_over_sum_pt - 0.06813910019)
        + 10.2 * max(0.0, Q.centroid_offset - 0.014379521101)
        - 2.623 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.1876 * max(0.0, Q.width - 0.018827653081)
        + 0.01558 * max(0.0, Q.mass - 36.229410171509)
        + 3.62 * max(0.0, Q.e2 - 0.028531698044)
        + 0.1495 * max(0.0, Q.LHA - 0.312727471086)
        + 38.7 * max(0.0, 0.006679471358 - Q.width)
        + 0.3792 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0)
        + 2.328 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2)
        + 1.571 * max(0.0, 0.04447356835 - Q.e2)
        - 42.56 * max(0.0, 0.007330079875 - Q.lam1)
        + 0.6433 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05)
        - 0.02108 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 0.1603 * max(0.0, Q.mass - 69.611351776123)
        + 1.14 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1)
        - 28.88 * max(0.0, 0.008678044751 - Q.girth2)
        - 0.08966 * max(0.0, Q.LHA - 0.325582223496)
        - 0.01569 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573)
        + 30.87 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0)
        + 2.303 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828)
        - 0.004891 * max(0.0, Q.mass_top5 - 53.607658247923)
        - 0.4361 * max(0.0, -0.012844925793 - Q.mean_eta)
        - 3.558 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0)
        + 181.8 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion)
        - 0.04181 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6)
        - 38.55 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7)
        - 95.32 * max(0.0, Q.mass_over_sum_pt - 0.090413827016)
        - 24.14 * max(0.0, 0.004372139461 - Q.girth2)
        + 117.8 * max(0.0, Q.mass_over_sum_pt - 0.107985668755)
        - 44.64 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr)
        - 1.319 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781)
        + 28.02 * max(0.0, Q.lam1 - 0.012003726523)
        + 1135.0 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr)
        + 0.08994 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr)
        + 0.6297 * max(0.0, Q.max_dr - 0.145231109113)
        - 5.848 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3)
        + 139.6 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.02625 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4)
        + 18.51 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4)
        + 0.1461 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5)
        + 59.16 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153)
        - 77.22 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153)
        - 0.04633 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6)
        + 1.189 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6)
        + 0.008756 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7)
        + 19.05 * max(0.0, Q.LHA - 0.423592510895)
        - 0.0004772 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7)
        + 3.592 * max(0.0, 0.23799610585 - Q.tau21)
        - 0.09624 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass)
        - 319.9 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702)
        - 1469.0 * max(0.0, 0.000306123359 - Q.lam2)
        - 2421.0 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2)
        - 189.7 * max(0.0, Q.width - 0.000319370692)
        + 0.01151 * max(0.0, 763.825 - Q.sum_pt)
        + 4.652 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2)
        + 2333.0 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848)
        + 3.174 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346)
        - 138.1 * max(0.0, Q.width - 0.001653836415)
        - 0.0001303 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 136.8 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 2.936 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7)
        + 22.76 * max(0.0, Q.C2 - 0.067292226106)
        + 91.82 * max(0.0, 0.001130644719 - Q.lam2)
        + 58.9 * max(0.0, 0.005011406868 - Q.girth2_top3)
        - 31.68 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0)
        - 7.673 * max(0.0, Q.girth - 0.101940929517)
        + 230.6 * max(0.0, 0.017162483186 - Q.e2_sq)
        + 22.97 * max(0.0, Q.girth2 - 0.013238675334)
        + 0.0009204 * max(0.0, 430.75 - Q.sum_pt_top5)
        + 23.08 * max(0.0, Q.e2 - 0.063441075385)
        + 0.7244 * max(0.0, Q.e2 - 0.007078157854)
        + 0.01335 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3)
        - 58.1 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi)
        + 0.06807 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875)
        - 2.966 * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 0.4122 * max(0.0, Q.max_dr - 0.102758520097)
        + 5.59 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2)
        - 0.7144 * max(0.0, Q.max_dr - 0.197968879342)
        - 0.2831 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625)
        - 2.221 * max(0.0, Q.C2 - 0.014943876117)
        + 0.8739 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125)
        - 10.68 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3)
        - 20.87 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1)
        + 12.13 * max(0.0, 0.047915700823 - Q.girth)
        + 0.003577 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875)
        - 106.9 * max(0.0, 0.023780909279 - Q.e2_sq)
        + 48.68 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq)
        - 10.77 * max(0.0, 0.216055863061 - Q.LHA)
        - 8.644 * max(0.0, 0.049399692737 - Q.z_7)
        - 0.2523 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5)
        + 7.155 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 74.42 * max(0.0, 0.00528466865 - Q.e2_sq)
        - 28.75 * max(0.0, 0.028865759995 - Q.z_6)
        - 9.941 * max(0.0, 0.071488645583 - Q.z_7)
        + 7811.0 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset)
        + 809.7 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset)
        + 0.01482 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt)
        + 4567.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1)
        - 28.13 * max(0.0, 0.03243272066 - Q.z_7)
        + 22.29 * max(0.0, 0.026454043164 - Q.dr_0)
        - 10.01 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0)
        + 4.34 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0)
        - 3.986 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 84.66 * max(0.0, 0.002635417778 - Q.width)
        + 55.84 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt)
        + 103.5 * max(0.0, 0.002270363079 - Q.girth2_top5)
        - 0.004103 * max(0.0, Q.sum_pt_top5 - 752.1)
        - 2840.0 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2)
        + 0.0008758 * max(0.0, 548.196875 - Q.sum_pt_top2)
        - 4082.0 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 205.4 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.5616 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625)
        + 0.01493 * max(0.0, 24.578125 - Q.pt_5)
        + 0.005411 * max(0.0, Q.sum_pt - 868.509375)
        - 0.2413 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset)
        + 0.7919 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085)
        + 505.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset)
        - 5669.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2)
        - 0.1982 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0)
        - 192.7 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0)
        + 236.3 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0)
        + 509.6 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2)
        - 29.08 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1)
        - 39.12 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2)
        + 0.3801 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3)
        - 10010.0 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2)
        - 0.08293 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass)
        + 0.2593 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142)
        - 9.962 * max(0.0, 0.01426135283 - Q.mean_phi2)
        - 0.3139 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass)
        - 96.86 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow)
        - 63790.0 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2)
        + 52.71 * max(0.0, 0.021588001063 - Q.dr_0)
        - 16.77 * max(0.0, Q.centroid_offset - 0.008092360237)
        - 33.14 * max(0.0, 0.013238675006 - Q.width)
        + 37.97 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt)
        - 0.1216 * max(0.0, 6.327378592257 - Q.log_sum_pt)
        + 3499.0 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2)
        + 14.08 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7)
        + 17.62 * max(0.0, 0.050284641981 - Q.e2)
        + 15.43 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781)
        + 82.87 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2)
        + 2.459 * max(0.0, Q.centroid_offset - 0.018377780003)
        - 0.006547 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125)
        - 9.492 * max(0.0, 6.701242202626 - Q.log_sum_pt)
        + 26.15 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7)
        + 5.277 * max(0.0, Q.centroid_offset - 0.049903668404)
        + 0.5331 * Q.max_dr
        - 2.489 * max(0.0, Q.C2 - 0.010539266048)
        - 1.962 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2)
        + 3.395 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787)
        + 0.05309 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1)
        + 120.3 * max(0.0, 0.000537286005 - Q.lam2)
        + 533.9 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072)
        + 6.21 * max(0.0, Q.mass_over_sum_pt - 0.008374148675)
        + 9.751 * max(0.0, Q.girth2_top5 - 0.011482925368)
        - 0.9179 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537)
        - 25.52 * max(0.0, Q.girth - 0.087236513197)
        + 1.012 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32)
        - 0.02409 * max(0.0, 1.679198372364 - Q.D2)
        - 0.03545 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5)
        + 45.4 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2)
        + 35.14 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179)
        - 44.6 * max(0.0, Q.lam2 - 0.003408388935)
        + 1.519 * max(0.0, 6.572937922293 - Q.log_sum_pt)
        - 0.008292 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6)
        + 4.321 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5)
        + 72.7 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105)
        - 5544.0 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105)
        + 50.74 * max(0.0, 0.003562611155 - Q.girth2)
        + 0.003714 * max(0.0, 49.668099212646 - Q.mass)
        - 2.595 * max(0.0, Q.girth2_top5 - 0.002270363079)
        + 0.001125 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4)
        + 1.243 * max(0.0, Q.z_7 - 0.06164517166)
        - 0.005054 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7)
        - 0.1018 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375)
        - 0.006326 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1)
        - 0.05344 * max(0.0, 0.195013533663 - Q.planar_flow)
        - 49.45 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842)
        - 7.604 * max(0.0, 0.001056655216 - Q.girth2_top2)
        + 71.1 * max(0.0, Q.girth2 - 0.007520088344)
        + 42.45 * max(0.0, Q.girth2 - 0.004372139461)
        - 4.158 * max(0.0, Q.mass_over_sum_pt - 0.072690732432)
        - 964.8 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 138.1 * max(0.0, Q.girth2 - 0.0016538364)
        - 13.31 * max(0.0, 0.024547699839 - Q.e2)
        + 0.04113 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6)
        + 29.67 * max(0.0, 0.04081947431 - Q.girth)
        - 0.0215 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439)
        + 0.0001672 * max(0.0, 48.71875 - Q.pt_7)
        + 0.0005529 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875)
        - 27.86 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        + 27.68 * max(0.0, 0.005590288644 - Q.width)
        - 1.13 * max(0.0, 0.02076709205 - Q.centroid_offset)
        + 29.18 * max(0.0, Q.girth2 - 0.008678044751)
        + 29.03 * max(0.0, Q.mass_over_sum_pt - 0.084751611895)
        - 110.2 * max(0.0, 0.001101266364 - Q.e2_sq)
        + 0.01678 * max(0.0, Q.mass - 80.4)
        + 33.89 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852)
        - 74.21 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2)
        - 7.582 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2)
        + 1.905 * max(0.0, 0.038466955721 - Q.e2)
        - 62.82 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605)
        + 0.01048 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478)
        + 0.003032 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow)
        - 2.047 * max(0.0, Q.centroid_offset - 0.031170772021)
        - 1.862 * max(0.0, 0.293190627853 - Q.LHA)
        - 11.72 * max(0.0, 0.005884990035 - Q.girth2_top3)
        - 221.0 * max(0.0, 0.000561123155 - Q.width)
        + 0.139 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5)
        - 1003.0 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1)
        + 0.1315 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5)
        - 1.948 * max(0.0, 0.197968879342 - Q.max_dr)
        + 180.1 * max(0.0, 0.005019718802 - Q.width)
        - 113.6 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21)
        + 4275.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset)
        - 9.758 * max(0.0, 0.196739721581 - Q.LHA)
        - 10.19 * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 86.75 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow)
        + 0.5008 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2)
        - 1.701 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0)
        - 69.06 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264)
        - 5.235 * max(0.0, 0.061086014472 - Q.girth)
        - 0.001444 * max(0.0, Q.sum_pt_top5 - 658.125)
        + 23.55 * max(0.0, 0.003343241496 - Q.centroid_offset)
        + 0.5159 * max(0.0, 0.177304983139 - Q.max_dr)
        + 451.4 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4)
        + 52.86 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        + 0.09134 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 10510.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266)
        - 2922.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256)
        - 2243.0 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649)
        + 4880.0 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2)
        - 53.25 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2)
        + 0.04693 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7)
        + 127.1 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2)
        + 7433.0 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq)
        + 3.636 * max(0.0, 0.016554418951 - Q.e2)
        - 232.2 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.9721 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 3480.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width)
        - 35.65 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394)
        - 0.5208 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset)
        + 0.02492 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375)
        - 9223.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width)
        + 21150.0 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2)
        + 906.4 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266)
        + 267.2 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 2082.0 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003)
        - 8270.0 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2)
        - 0.01219 * max(0.0, 8.379955863953 - Q.mass)
        - 0.7018 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7)
        + 158.4 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121)
        + 0.09751 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848)
        - 2761.0 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2)
        + 147.1 * max(0.0, 0.000964142894 - Q.girth2)
        - 1575.0 * max(0.0, 0.000222950415 - Q.girth2_top5)
        + 4.536 * max(0.0, 0.054649224505 - Q.girth)
        + 15.83 * max(0.0, Q.lam2 - 0.001130644719)
        + 0.3227 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset)
        + 2.586 * max(0.0, 0.006096650059 - Q.width)
        - 6.859 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1)
        - 0.1735 * max(0.0, Q.girth2 - 0.018827652745)
        - 3.597 * max(0.0, 0.032346998155 - Q.e2)
        - 51.4 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow)
        - 49.32 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01)
        + 0.08416 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 0.05741 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt)
        + 1.167 * max(0.0, 0.221586732566 - Q.max_dr)
        - 589.1 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516)
        + 8.842 * max(0.0, Q.C2 - 0.051192347892)
        - 1.267 * max(0.0, 0.018377780003 - Q.centroid_offset)
        + 1798.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496)
        - 319.8 * max(0.0, 0.000172198326 - Q.width)
        + 14.11 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752)
        + 5.527 * max(0.0, 0.007520088344 - Q.girth2)
        - 0.1453 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2)
        - 0.004794 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        - 0.03869 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375)
        + 2.486 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878)
        - 54.54 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696)
        + 6.167 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity)
        - 0.0001876 * max(0.0, 41.377904891968 - Q.mass)
        + 0.004686 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow)
        + 27.84 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21)
        - 1.066 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt)
        - 34.59 * max(0.0, 0.001503553356 - Q.lam1)
        + 0.1077 * max(0.0, Q.max_dr - 0.15984864831)
        - 1.626 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7)
        + 0.1458 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264)
        - 0.0009956 * max(0.0, 35.5 - Q.pt_5)
        - 0.2305 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7)
        - 15.32 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta)
        + 0.000952 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962)
        - 0.7869 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
        + 12.46 * Q.e2
        + 725.5 * max(0.0, Q.lam2 - 0.000194798295)
        - 4.533 * max(0.0, Q.LHA - 0.303313749495)
        - 204.2 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668)
        + 0.8458 * max(0.0, 6.267538488641 - Q.log_sum_pt)
        + 0.5137 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7)
        - 16.6 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50)
        + 0.03387 * max(0.0, Q.n_dr_0p2_0p4 - 2.0)
        - 109.1 * max(0.0, 0.004183811014 - Q.lam1)
        - 68.3 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt)
        + 0.09023 * max(0.0, 3.0 - Q.n_dr_0p05_0p1)
        + 679.0 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21)
        + 4.514 * max(0.0, 0.269169217348 - Q.tau32)
        - 14.35 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21)
        + 0.4394 * max(0.0, 0.391541349888 - Q.tau21)
        + 49.51 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4)
        - 1.637 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 214.7 * max(0.0, 0.002412890926 - Q.girth2_top2)
        - 0.02972 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125)
        - 9.714 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7)
        - 64.22 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2)
        + 386.1 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7)
        + 32.68 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386)
        + 0.7532 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125)
        + 804.2 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626)
        + 0.1268 * max(0.0, Q.mass - 15.454033088684)
        - 1.249 * max(0.0, Q.max_dr - 0.121680960059)
        + 0.001115 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 235.4 * max(0.0, 0.0016538364 - Q.girth2)
        + 0.01921 * max(0.0, Q.sum_pt - 813.415625)
        + 38.7 * max(0.0, 0.006679471442 - Q.girth2)
        - 0.4872 * max(0.0, 0.253403707141 - Q.planar_flow)
        + 22.44 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059)
        + 3.262 * max(0.0, 0.154689112391 - Q.LHA)
        - 19.48 * max(0.0, Q.girth - 0.076081777364)
        + 0.02494 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass)
        - 0.8624 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097)
        + 0.09325 * max(0.0, 0.049903668404 - Q.centroid_offset)
        + 0.06932 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7)
        + 0.392 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0)
        + 1.231 * max(0.0, 0.006390124748 - Q.e2_sq)
        - 3.304 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 0.005627 * max(0.0, 7.0 - Q.n_dr_0_0p05)
        - 26.75 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2)
        + 3.625 * Q.centroid_offset
        + 50.74 * max(0.0, 0.003562611091 - Q.width)
        + 0.002999 * max(0.0, 9.257203159811 - Q.mass_top5)
        + 213.1 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7)
        - 0.08485 * max(0.0, 15.454033088684 - Q.mass)
        + 0.436 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50)
        - 0.03678 * max(0.0, 29.0421875 - Q.pt_7)
        + 2.888 * max(0.0, 0.004839980301 - Q.lam1)
        - 0.1038 * max(0.0, 0.035786485299 - Q.C2)
        + 0.6414 * max(0.0, 0.111761856824 - Q.max_dr)
        + 0.02 * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 6.252e-06 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2)
        + 0.06558 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625)
        + 0.008187 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 0.02822 * max(0.0, Q.mass - 91.19)
        - 10630.0 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005)
        - 3.568 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625)
        - 4.986 * max(0.0, Q.mean_phi - 0.026127964072)
        - 55.69 * max(0.0, 0.148408418149 - Q.girth)
        + 9.133 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        + 0.5292 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7)
        - 84.39 * max(0.0, 0.016433749775 - Q.lam1)
        + 0.2207 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189)
        + 34160.0 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset)
        - 0.001596 * max(0.0, 531.1875 - Q.sum_pt_top5)
        - 0.0001908 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7)
        + 0.0005003 * max(0.0, Q.sum_pt_top5 - 902.40625)
        - 0.007538 * max(0.0, Q.sum_pt_top5 - 839.9546875)
        + 0.507 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6)
        - 50.68 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362)
        + 0.6914 * max(0.0, 0.501026660204 - Q.tau21)
        + 951.9 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset)
        - 7.954 * max(0.0, 0.067272114405 - Q.z_6)
        - 5.167 * max(0.0, 0.037760993714 - Q.centroid_offset)
        + 2.139 * max(0.0, Q.LHA - 0.09323897448)
        - 0.001 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2)
        - 0.005911 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2)
        + 3.501 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841)
        + 5.524 * max(0.0, 0.00752008842 - Q.width)
        - 1.636 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4)
        + 0.09103 * max(0.0, 25.578125 - Q.pt_7)
        - 0.1524 * max(0.0, Q.sum_pt - 988.4078125)
        - 0.006251 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0)
        - 0.0002842 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2)
        - 0.5283 * max(0.0, 0.111513564951 - Q.planar_flow)
        + 0.09663 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 15.72 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003)
        + 34.23 * max(0.0, Q.lam1 - 0.008375572068)
        - 26.31 * max(0.0, Q.lam1 - 0.004183811014)
        + 19.56 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr)
        + 54.0 * max(0.0, Q.lam1 - 0.00543336053)
        + 0.004476 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt)
        + 49.58 * max(0.0, Q.lam1 - 0.007330079875)
        - 0.9738 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2)
        + 3457.0 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr)
        - 0.6385 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235)
        - 1801.0 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr)
        - 34.78 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835)
        - 0.005644 * max(0.0, 76.655700683594 - Q.mass)
        - 2.389 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164)
        + 0.0798 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        + 41.01 * max(0.0, Q.lam1 - 0.005954149834)
        - 0.01394 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        - 15.41 * max(0.0, Q.lam1 - 0.002464291268)
        - 359.7 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq)
        + 0.009667 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2)
        + 74.29 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow)
        - 596.2 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164)
        + 105.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        + 0.2421 * max(0.0, 1.122624260187 - Q.D2)
        + 5.901 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2)
        + 0.0127 * max(0.0, 5.0 - Q.n_dr_0p05_0p1)
        + 3.604 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset)
        - 6.247 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116)
        + 0.01446 * max(0.0, 0.74595130682 - Q.D2)
        + 43.68 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt)
        - 44.99 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2)
        - 15.03 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2)
        - 3.024 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 37.15 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4)
        - 0.668 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2)
        + 0.6893 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1)
        + 14.57 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr)
        + 1.875 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        + 5400.0 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839)
        - 57.52 * max(0.0, 0.012003726523 - Q.lam1)
        - 216.1 * max(0.0, 0.011657374702 - Q.e2_sq)
        - 546.9 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq)
        + 0.003803 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625)
        + 4.912 * max(0.0, 0.041109715588 - Q.e2)
        + 6.444 * max(0.0, 0.063441075385 - Q.e2)
        + 0.05639 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2)
        + 0.4206 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 3.961e-05 * max(0.0, 2.0 - Q.n_dr_0_0p05)
        + 1.2 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264)
        - 1.783 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377)
        + 0.03143 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4)
        - 34.04 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136)
        - 2280.0 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249)
        - 0.2054 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276)
        + 19.87 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow)
        + 14.01 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt)
        - 0.02734 * max(0.0, Q.n_dr_0p05_0p1 - 5.0)
        - 6.837 * max(0.0, Q.LHA - 0.346713497427)
        + 26.46 * max(0.0, Q.girth - 0.033604209498)
        - 50.04 * max(0.0, 0.002151567843 - Q.girth2_top3)
        - 1.85 * max(0.0, 0.007639643088 - Q.girth2_top2)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.width > 0.009197441395372152:
        if s['g'] - s['t'] > -0.0796944685280323:
            if s['g'] - s['q'] > -0.03662191703915596:
                if s['g'] - s['t'] > 0.1061202771961689:
                    if s['g'] - s['Z'] > 0.07397183403372765:
                        if Q.e2 > 0.09787071496248245:
                            if Q.pt_7 > 37.671875:
                                return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > 0.6921992301940918:
                                    return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 66% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > 0.42837822437286377:
                                if s['g'] - s['q'] > 0.09086447209119797:
                                    if s['q'] - s['Z'] > 7.959142446517944:
                                        if s['g'] - s['t'] > 1.1688838601112366:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.8083146810531616:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.032608333975076675:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > 8.397993087768555:
                                    if Q.z_top5 > 0.8058714866638184:
                                        if Q.pt_5 > 31.109375:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 4.764932870864868:
                                        if Q.girth2_top2 > 0.0269379373639822:
                                            if Q.C2 > 0.061301348730921745:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.phi_1 > 0.03261566162109375:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 64.89310836791992:
                            return 'g'   # 67% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 89% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['Z'] > 0.08410898968577385:
                        if s['g'] - s['t'] > 0.03546742722392082:
                            if s['Z'] - s['t'] > -7.87116551399231:
                                return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p05_0p1 > 0.5291006565093994:
                                    return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p1_0p2 > 0.9139785468578339:
                                return 't'   # 69% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > 12.584245204925537:
                                    return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.z_6 > 0.10455435141921043:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.38423700630664825:
                                            if Q.phi_7 > 0.1331787109375:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_4 > 0.09824599325656891:
                                                if Q.lam1 > 0.00911121815443039:
                                                    if Q.girth2_top5 > 0.006106771063059568:
                                                        if Q.e2 > 0.05308009311556816:
                                                            if Q.e2 > 0.06888448819518089:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.07604088634252548:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 84% of the training jets here get this class from the formula
            else:
                return 'q'   # 82% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.12900714576244354:
                if s['Z'] - s['t'] > 0.1727282777428627:
                    if Q.log_sum_pt > 6.617673397064209:
                        if s['Z'] - s['t'] > 0.5467167496681213:
                            return 'Z'   # 96% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > 4.019908308982849:
                                return 't'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 98% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt > 632.125:
                        if Q.z_dr_0_0p05 > 0.06659437716007233:
                            return 'Z'   # 61% of the training jets here get this class from the formula
                        else:
                            if Q.tau32 > 0.4712686240673065:
                                return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 55% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.02717881929129362:
                            return 't'   # 59% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 82% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.13820552080869675:
                    if s['q'] - s['t'] > 0.09677619114518166:
                        return 'q'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top3 > 571.0:
                            if s['q'] - s['W'] > 5.5054943561553955:
                                if Q.m012 > 3.588148832321167:
                                    return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.00014027501310920343:
                                return 'q'   # 74% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -0.044631609693169594:
                                    return 'q'   # 59% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 74% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.330395832657814:
                        if Q.centroid_offset > 0.05585586838424206:
                            if Q.girth > 0.16690687835216522:
                                if s['q'] - s['W'] > 6.79108738899231:
                                    return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.1835973933339119:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_3 > 0.18475893139839172:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.phi_7 > 0.0388946533203125:
                                    return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.7689020335674286:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.eta_0 > 0.107025146484375:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_eta > 0.04430334083735943:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > -0.28431694209575653:
                                                    if s['q'] - s['W'] > 10.365607738494873:
                                                        return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.07652655616402626:
                                if Q.pt_7 > 24.5703125:
                                    if Q.z_dr_0p2_0p4 > 0.06207798980176449:
                                        if Q.max_dr > 0.32305601239204407:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_phi > 0.012563118245452642:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > -0.22266460955142975:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 66% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.18868695199489594:
                                    if Q.mass > 48.88730812072754:
                                        if Q.lam2 > 0.00021584776550298557:
                                            if Q.max_dr > 0.16602251678705215:
                                                if s['g'] - s['W'] > 7.703374862670898:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_eta2 > 0.010936936363577843:
                                                        return 't'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 0.019213643856346607:
                                        if Q.dr_6 > 0.23004848510026932:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 93% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.5712519586086273:
                            if s['W'] - s['t'] > -3.915930151939392:
                                if Q.lam2 > 0.00020205848704790697:
                                    if s['W'] - s['t'] > -3.341766595840454:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.007874831324443221:
                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 519.5078125:
                                    return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 52% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.5265758037567139:
                                if Q.girth2 > 0.036471955478191376:
                                    if Q.pt_4 > 50.875:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_4 > 30.078125:
                                        return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.08656174317002296:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                return 't'   # 100% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.24157562851905823:
            if s['g'] - s['q'] > 0.00968681275844574:
                if s['g'] - s['t'] > -0.04817958548665047:
                    if s['g'] - s['W'] > 0.020514756441116333:
                        if s['g'] - s['q'] > 0.1167370080947876:
                            if s['g'] - s['t'] > 0.2658756524324417:
                                if s['g'] - s['q'] > 0.22331993281841278:
                                    if s['g'] - s['W'] > 0.3278306573629379:
                                        if Q.z_7 > 0.011696463450789452:
                                            if s['g'] - s['q'] > 0.2894338518381119:
                                                if s['q'] - s['Z'] > 5.055893182754517:
                                                    if s['g'] - s['t'] > 1.3734924793243408:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.04486602544784546:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top2 > 0.0011975730303674936:
                                                                if Q.pt_0 > 173.375:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0026582726277410984:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > 2.005220949649811:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > 3.123893141746521:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.030515204183757305:
                                            if Q.mass > 29.920292854309082:
                                                if s['Z'] - s['t'] > -0.6220933794975281:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 1.2386369705200195:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_2 > 0.013481453992426395:
                                                    if Q.pt1_dr01 > 10.692437171936035:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['t'] > 3.221286654472351:
                                                            if Q.centroid_offset > 0.017673330381512642:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -0.07602706551551819:
                                        if s['q'] - s['W'] > 0.1725233942270279:
                                            if Q.centroid_offset > 0.004696927964687347:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 2.4364482164382935:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 737.96875:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 12.207359790802002:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 41.078125:
                                                                return 'q'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 2.6818633159564342e-05:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt1_dr01 > 0.6816844344139099:
                                                if s['q'] - s['W'] > -0.044355105608701706:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.004894237732514739:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 2.4138710498809814:
                                                if Q.z_7 > 0.031230059452354908:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > 3.107279896736145:
                                                    if Q.centroid_offset > 0.0032977957744151354:
                                                        if Q.pt_2 > 83.53125:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.003111023106612265:
                                                            if Q.mass > 8.43979549407959:
                                                                return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > 0.06412098184227943:
                                    if s['Z'] - s['t'] > -6.693366289138794:
                                        if Q.mean_phi > -0.039315128698945045:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 0.7694889903068542:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -2.0439735651016235:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 43.59670448303223:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.17637038230896:
                                                if Q.sum_pt > 621.953125:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.04730154387652874:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > -0.0006521366012748331:
                                if s['g'] - s['q'] > 0.05004796013236046:
                                    if s['g'] - s['Z'] > 0.8817261159420013:
                                        if Q.C2 > 0.011225885711610317:
                                            if s['g'] - s['q'] > 0.09014243632555008:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 634.0625:
                                                    if Q.e2 > 0.006392693147063255:
                                                        if Q.LHA > 0.1789105385541916:
                                                            if Q.centroid_offset > 0.010427873115986586:
                                                                if s['q'] - s['W'] > 0.5850704610347748:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.012936275452375412:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9841589331626892:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0049066320061683655:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 3.3904683589935303:
                                                if Q.sum_pt_top2 > 440.375:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.07632556930184364:
                                                    if Q.dr_0 > 0.007838987279683352:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 0.07303731143474579:
                                            if Q.sum_pt_top5 > 701.1875:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.000759895279770717:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 653.4375:
                                        if s['q'] - s['Z'] > 1.2809012532234192:
                                            if Q.sum_pt_top2 > 460.5:
                                                if Q.pt_7 > 33.828125:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.009933783207088709:
                                                    if Q.centroid_offset > 0.011078526265919209:
                                                        if Q.dr_6 > 0.024440297856926918:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['Z'] > 0.13631483167409897:
                                                            if Q.sum_pt_top2 > 345.125:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.004431511275470257:
                                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008898417465388775:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 60.125:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 695.015625:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.0004746483318740502:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.27539949119091034:
                                            if Q.lam2 > 4.853556674788706e-05:
                                                return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.6500163078308105:
                                    if Q.log_sum_pt > 7.099457025527954:
                                        return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.04016513004899025:
                                            if Q.mass_top5 > 16.65973472595215:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 459.6875:
                                                if Q.z_7 > 0.05739951692521572:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.0055977217853069305:
                                        if Q.pt_5 > 48.03125:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 32.359375:
                                                if Q.C2 > 0.02360313106328249:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.549684286117554:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.023359217680990696:
                                                    return 'g'   # 35% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 1.6913933157920837:
                                            if s['g'] - s['Z'] > 2.4700292348861694:
                                                if s['g'] - s['q'] > 0.08632079884409904:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 667.953125:
                                                        if Q.mass > 14.38560152053833:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.009498116094619036:
                                                                return 'q'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_4 > 44.53125:
                                                    if s['q'] - s['t'] > 2.773103952407837:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > 1.8585510849952698:
                                                            if s['g'] - s['q'] > 0.0964500866830349:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_5 > 52.875:
                                                return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.4212913513183594:
                            if s['g'] - s['Z'] > 1.027489960193634:
                                if Q.e2 > 0.027622753754258156:
                                    if Q.eccentricity > 0.9911943674087524:
                                        if s['g'] - s['Z'] > 1.6048640608787537:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top5 > 23.260089874267578:
                                            if Q.girth2_top2 > 0.0016878959140740335:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.10313606634736061:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > -0.18647074699401855:
                                        if Q.mass_top5 > 25.285232543945312:
                                            if Q.pt_dispersion > 0.40336230397224426:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0_0p05 > 6.5:
                                            if Q.mass > 27.921642303466797:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.04740709997713566:
                                                if Q.girth2_top3 > 0.0012227247352711856:
                                                    if Q.log_sum_pt > 6.390617847442627:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                            else:
                                if s['Z'] - s['t'] > 2.6386799812316895:
                                    return 'W'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.001580533164087683:
                                        return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top5 > 0.8616653084754944:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.050343750044703484:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > -0.7297484278678894:
                                if s['g'] - s['Z'] > 1.019107699394226:
                                    if s['q'] - s['t'] > 0.9966084957122803:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 98% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.014638297725468874:
                        if s['W'] - s['t'] > 0.33441998064517975:
                            return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top3 > 0.0019594630575738847:
                                return 'W'   # 83% of the training jets here get this class from the formula
                            else:
                                return 't'   # 64% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.25594596564769745:
                            if s['W'] - s['Z'] > -2.6206436157226562:
                                if s['g'] - s['t'] > -0.15814178436994553:
                                    if Q.log_sum_pt > 6.2020769119262695:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_3 > 0.047167250886559486:
                                        if Q.tau32 > 0.5182288885116577:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 461.2109375:
                                    if Q.log_sum_pt > 6.450618505477905:
                                        if s['g'] - s['t'] > -0.1371639519929886:
                                            if Q.pt_6 > 38.609375:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.14167584478855133:
                                        return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.LHA > 0.16581083834171295:
                                if s['g'] - s['t'] > -0.5373515188694:
                                    if Q.C2 > 0.025063255801796913:
                                        return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 441.1171875:
                                            return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 99% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 45% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.022190628573298454:
                    if s['q'] - s['t'] > -0.0452562402933836:
                        if s['g'] - s['q'] > -0.08378492295742035:
                            if s['g'] - s['q'] > -0.02863105572760105:
                                if Q.sum_pt_top5 > 582.796875:
                                    if Q.pt_dispersion > 0.39906004071235657:
                                        if Q.log_sum_pt > 6.920008182525635:
                                            if Q.sum_pt > 1154.8984375:
                                                return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 3.1307607889175415:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > 0.0431353785097599:
                                                    if Q.C2 > 0.011484798975288868:
                                                        if Q.centroid_offset > 0.010837608017027378:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.006177812349051237:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 32.1328125:
                                                            if Q.sum_pt > 790.25:
                                                                if Q.lam2 > 1.4780587662244216e-05:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 2.941126227378845:
                                                        if Q.girth2_top5 > 0.00012822761345887557:
                                                            if Q.z_top5 > 0.8202373683452606:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.02210389729589224:
                                            if Q.dr_0 > 0.014501032419502735:
                                                return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 9.746405339683406e-05:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if s['Z'] - s['t'] > 2.6352829933166504:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > 0.7966238260269165:
                                        if Q.planar_flow > 0.05867652967572212:
                                            return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 5.635746955871582:
                                            if Q.max_dr > 0.0720541961491108:
                                                if Q.pt_4 > 44.421875:
                                                    if s['q'] - s['t'] > 2.087091326713562:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 677.984375:
                                                    if Q.width > 0.00039497119723819196:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 2.097409725189209:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.916296720504761:
                                    if Q.sum_pt > 1168.265625:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_5 > 72.78125:
                                        if s['W'] - s['Z'] > -0.08186877891421318:
                                            if Q.dr_0 > 0.005426678340882063:
                                                if Q.z_7 > 0.043562235310673714:
                                                    if Q.centroid_offset > 0.00637213047593832:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9906152486801147:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05707894265651703:
                                                if s['q'] - s['Z'] > 1.8305876851081848:
                                                    if Q.sum_pt > 831.15625:
                                                        if Q.girth > 0.007421525660902262:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > 0.08310091868042946:
                                                    if Q.centroid_offset > 0.0022727252217009664:
                                                        if Q.z_5 > 0.07463983818888664:
                                                            if Q.centroid_offset > 0.0042628091759979725:
                                                                if Q.C2 > 0.010676536709070206:
                                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2 > 9.960008173948154e-05:
                                                                        if Q.eta_0 > 0.011272430419921875:
                                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.04617995768785477:
                                                                if Q.width > 0.00011341008212184533:
                                                                    if Q.C2 > 0.008594101760536432:
                                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        if s['q'] - s['Z'] > 1.8151332139968872:
                                                                            if Q.pt1_dr01 > 0.7197777032852173:
                                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'q'   # 70% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.07854808494448662:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.26457706093788147:
                                if s['g'] - s['q'] > -0.1793014481663704:
                                    if Q.sum_pt_top3 > 671.25:
                                        if Q.sum_pt > 1213.6796875:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1031.1484375:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.00012513643378042616:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.004973460221663117:
                                            if Q.z_7 > 0.032538075000047684:
                                                if Q.planar_flow > 0.03278930485248566:
                                                    if Q.centroid_offset > 0.0011697381851263344:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.84375:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 5.2919587687938474e-05:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 3.833770513534546:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -0.25447978079319:
                                        if s['W'] - s['t'] > -2.0776132345199585:
                                            if Q.sum_pt > 1082.904296875:
                                                if s['g'] - s['q'] > -0.4646504819393158:
                                                    if Q.log_sum_pt > 7.056525468826294:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.03273449465632439:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['t'] > 2.2668557167053223:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.008968389127403498:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 2.489355206489563:
                                            if s['q'] - s['t'] > 0.1220199428498745:
                                                if Q.pt_5 > 34.921875:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 9.8041749879485e-05:
                                                        if Q.sum_pt > 596.78515625:
                                                            if s['g'] - s['q'] > -0.8197149634361267:
                                                                if Q.centroid_offset > 0.0027838615933433175:
                                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 13.381628036499023:
                                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0020306509686633945:
                                                            if Q.sum_pt_top5 > 666.078125:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.7034947872161865:
                                                                if Q.pt_5 > 20.0625:
                                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                if s['q'] - s['t'] > 2.8979005813598633:
                                                                    if Q.girth2_top3 > 2.3343127395492047e-05:
                                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 558.2109375:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 4.261015601514373e-05:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > 3.892188787460327:
                                                    if Q.pt_4 > 51.5625:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['t'] > 1.756902277469635:
                                                            return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 3.7393705497379415e-05:
                                                        if Q.pt_5 > 45.15625:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['t'] > 2.7513744831085205:
                                                                if Q.pt_4 > 45.046875:
                                                                    if s['q'] - s['t'] > 3.506688952445984:
                                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 78.15625:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2584681063890457:
                                    if Q.dr_3 > 0.011823304928839207:
                                        return 'W'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.0024995473213493824:
                                        if s['W'] - s['Z'] > 0.9791829288005829:
                                            if Q.girth > 0.04958475008606911:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 7.287758126039989e-05:
                                            if s['W'] - s['Z'] > 0.7441483736038208:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.019937463104724884:
                                                    if Q.e2_sq > 0.0014419269282370806:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.5759466886520386:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.017846979200839996:
                                                if Q.sum_pt > 777.6171875:
                                                    if Q.width > 0.00047589837049599737:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi2 > 0.00018489172362023965:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.3558952808380127:
                            if s['g'] - s['t'] > -1.5623847246170044:
                                if s['q'] - s['W'] > 0.5942469537258148:
                                    return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 43% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 61% of the training jets here get this class from the formula
                        else:
                            return 't'   # 96% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['W'] > -0.3794710338115692:
                        if s['q'] - s['Z'] > 0.9006775319576263:
                            if Q.girth > 0.04558306746184826:
                                if Q.girth > 0.053652314469218254:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.989656001329422:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > 0.5260114371776581:
                                    if Q.max_dr > 0.2556522339582443:
                                        return 'W'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 30% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > -0.08668403699994087:
                                if Q.girth > 0.018048682250082493:
                                    if Q.mass > 29.961583137512207:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.000441340496763587:
                                            if s['q'] - s['Z'] > 0.34340275824069977:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.dr01 > 0.00790836475789547:
                                        return 'W'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.001986216171644628:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > 0.8717204630374908:
                                        if Q.eccentricity > 0.8768350780010223:
                                            if Q.phi_1 > 0.008174896240234375:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.020998612977564335:
                                            return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.0006343194690998644:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > -1.821287214756012:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 30.495671272277832:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['Z'] > 0.62610924243927:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['Z'] > 0.9777295589447021:
                            if Q.sum_pt_top5 > 664.671875:
                                if s['q'] - s['W'] > -1.000352680683136:
                                    if Q.pt_0 > 491.5:
                                        return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 0.0021491030929610133:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 74% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.004974273033440113:
                                    return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 88% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > 0.09477170184254646:
                                if Q.girth2 > 0.0019322876469232142:
                                    return 'W'   # 98% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > 2.906546711921692:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -0.651981920003891:
                                            if Q.lam2 > 3.035368717974052e-05:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                return 't'   # 56% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > 0.019345716573297977:
                if s['g'] - s['W'] > 0.03812866844236851:
                    if s['g'] - s['W'] > 0.2473410815000534:
                        if s['g'] - s['t'] > -0.03971278667449951:
                            if Q.mass > 52.468828201293945:
                                if Q.D2 > 0.5329417884349823:
                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 76% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > 0.39516179263591766:
                                    if Q.mass > 28.427733421325684:
                                        if s['g'] - s['W'] > 0.7216320335865021:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.008058912586420774:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_4 > 0.10328696668148041:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.012080663815140724:
                                        if Q.tau21 > 0.11407258361577988:
                                            if s['q'] - s['t'] > 2.588512659072876:
                                                if Q.girth2_top5 > 0.0003873535606544465:
                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005391060141846538:
                                            if Q.dr_3 > 0.0480219442397356:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 75% of the training jets here get this class from the formula
                        else:
                            return 't'   # 94% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.04278164356946945:
                            if s['q'] - s['t'] > 1.8379679322242737:
                                if Q.centroid_offset > 0.01817108504474163:
                                    if s['g'] - s['Z'] > 0.7321158647537231:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 4.194216126052197e-05:
                                            if Q.LHA > 0.17573970556259155:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['t'] > 2.277426242828369:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 2.6727787256240845:
                                        if s['g'] - s['W'] > 0.12206387892365456:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 28.430347442626953:
                                    if s['g'] - s['t'] > 2.0728384256362915:
                                        if Q.girth > 0.05110333301126957:
                                            return 'W'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.09973963350057602:
                                            if Q.pt_7 > 30.90625:
                                                if Q.girth > 0.04931744933128357:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.14757710695266724:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.09988154098391533:
                                                if Q.z_6 > 0.06056595407426357:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.3941841572523117:
                                        if Q.sum_pt > 689.078125:
                                            if s['q'] - s['W'] > -1.5338323712348938:
                                                if Q.z_6 > 0.07590470463037491:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.01336690690368414:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.007533564465120435:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.1511698961257935:
                                                if Q.dr_0 > 0.030941524542868137:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > 1.3432767391204834:
                                            if Q.pt_dispersion > 0.4002366214990616:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.24142051488161087:
                                                if Q.mean_phi2 > 0.00046560418559238315:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06872151792049408:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > 1.056895673274994:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.7262945175170898:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.1924702152609825:
                                                        if Q.z_dr_0p05_0p1 > 0.04904802702367306:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                        else:
                            return 't'   # 85% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.21169757843017578:
                        if s['g'] - s['W'] > -0.4111604392528534:
                            if s['W'] - s['t'] > 0.07352503761649132:
                                if s['g'] - s['q'] > 1.250560998916626:
                                    if Q.mass_over_sum_pt > 0.05523904971778393:
                                        if Q.mass > 28.079474449157715:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 0.7960081696510315:
                                                if Q.min_pair_mass > 1.0027563571929932:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 3.3425414585508406e-05:
                                            if s['g'] - s['W'] > -0.19010917842388153:
                                                if s['g'] - s['q'] > 1.581018090248108:
                                                    if s['g'] - s['Z'] > 0.2910754233598709:
                                                        if Q.mass_over_sum_pt_sq > 0.0021557784639298916:
                                                            if Q.e2 > 0.02245150413364172:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.464781999588013:
                                                        if Q.C2 > 0.011540873907506466:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr01 > 0.011380020529031754:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_2 > 0.03685710392892361:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.957253873348236:
                                                    if Q.lam1 > 0.0025503956712782383:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_5 > 50.796875:
                                                        if Q.z_dr_0p05_0p1 > 0.16046881675720215:
                                                            if s['q'] - s['Z'] > -0.758102536201477:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.7890625:
                                                if s['g'] - s['Z'] > 0.3183394819498062:
                                                    if Q.mass > 43.46396827697754:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['W'] > -1.8380487561225891:
                                                            if s['q'] - s['Z'] > -0.7702944576740265:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['W'] > -0.11380261555314064:
                                        if s['Z'] - s['t'] > 2.036615014076233:
                                            if Q.centroid_offset > 0.017961003817617893:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_6 > 0.026256482116878033:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.023027585819363594:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 13.961106777191162:
                                                if s['W'] - s['Z'] > 0.9455797076225281:
                                                    if Q.mass > 26.665892601013184:
                                                        if Q.width > 0.0033605792559683323:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_7 > 0.040858346968889236:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['W'] > -1.0380293726921082:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.4980568671016954e-05:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if s['Z'] - s['t'] > 1.6432032585144043:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > 0.27393946051597595:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.8555842936038971:
                                            if Q.e2_sq > 0.0031295923981815577:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.07848170399665833:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06590691953897476:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 27.88024139404297:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1016.015625:
                                                if Q.mass_over_sum_pt_sq > 0.0008444332343060523:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                return 't'   # 78% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > -0.01815913151949644:
                                if s['W'] - s['Z'] > 0.32599757611751556:
                                    if s['g'] - s['W'] > -0.7386038303375244:
                                        if s['g'] - s['q'] > 1.3481563329696655:
                                            if Q.mass_over_sum_pt > 0.04449912905693054:
                                                if s['W'] - s['t'] > 0.5131962895393372:
                                                    if s['g'] - s['t'] > 2.2296624183654785:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 29.63777732849121:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > 1.2452722191810608:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.069403555855388e-05:
                                                    if s['g'] - s['Z'] > 0.19574882835149765:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.915265798568726:
                                                if Q.max_dr > 0.06001468375325203:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > 0.435343399643898:
                                            return 'W'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.1559278964996338:
                                                if s['g'] - s['Z'] > -0.04561842139810324:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.0007384900818578899:
                                        if Q.max_dr > 0.07933345809578896:
                                            if Q.C2 > 0.05117647349834442:
                                                if Q.mass > 48.84498405456543:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.20800895243883133:
                                                        if Q.z_dr_0p2_0p4 > 0.023373437114059925:
                                                            if Q.mass_top3 > 2.487568497657776:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -0.9757693111896515:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02652821410447359:
                                                    if s['g'] - s['q'] > 1.0537848472595215:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.010729529429227114:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.18617716431617737:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.32327964901924133:
                                    if Q.lam2 > 0.0001501956139691174:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > -1.1173010468482971:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 95% of the training jets here get this class from the formula
                    else:
                        if Q.girth > 0.02466746512800455:
                            if s['W'] - s['Z'] > 0.12913545966148376:
                                if s['Z'] - s['t'] > -0.20088700950145721:
                                    if Q.mass_over_sum_pt_sq > 5.1263557907077484e-05:
                                        if s['g'] - s['q'] > 1.1194080114364624:
                                            if s['g'] - s['W'] > -0.26725761592388153:
                                                if Q.C2 > 0.017122283577919006:
                                                    return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0018606522353366017:
                                                    if Q.centroid_offset > 0.013689549639821053:
                                                        if Q.max_dr > 0.08477899432182312:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.03471091017127037:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 70.19386291503906:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > -0.09382518753409386:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1577581912279129:
                                                    if Q.sum_pt_top5 > 624.859375:
                                                        if Q.centroid_offset > 0.02133567351847887:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013570392038673162:
                                                                if Q.mass > 48.55790710449219:
                                                                    if Q.C2 > 0.04619952291250229:
                                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.planar_flow > 0.03878808952867985:
                                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.z_dr_0p2_0p4 > 0.022408942691981792:
                                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 14.5390625:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.04798689857125282:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 9.738136941450648e-05:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.12912403792142868:
                                                                if Q.centroid_offset > 0.013584536965936422:
                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_pair_mass > 26.409567832946777:
                                                        if s['Z'] - s['t'] > 2.4034000635147095:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.005789558868855238:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top3 > 0.006436681374907494:
                                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.0815984383225441:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.02589491754770279:
                                                                if s['g'] - s['t'] > 2.1613664627075195:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 0.00010470664710737765:
                                                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > -1.0089054703712463:
                                            if Q.planar_flow > 0.03370004892349243:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0007261193240992725:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 789.0:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0_0p05 > 0.9489171802997589:
                                    if s['q'] - s['Z'] > -0.939909815788269:
                                        if Q.planar_flow > 0.03732035122811794:
                                            if s['q'] - s['Z'] > 0.0006697264034301043:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.03843030892312527:
                                                    if Q.z_dr_0p2_0p4 > 0.02039247751235962:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 13.831812858581543:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > 0.0634253527969122:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_7 > 0.029289714992046356:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.026338830590248108:
                                            if Q.D2 > 3.0658438205718994:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_dispersion > 0.38486574590206146:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -1.2760841846466064:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 845.265625:
                                                if Q.C2 > 0.008313626516610384:
                                                    if Q.mass_over_sum_pt > 0.04708144627511501:
                                                        if Q.mass_over_sum_pt_sq > 0.0035456994082778692:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.587099015712738:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.03796064294874668:
                                                    if Q.z_7 > 0.045694855973124504:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_2 > 0.028548700734972954:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > 0.06355894170701504:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -1.2903835773468018:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['t'] > 0.0073189961276511895:
                                        if Q.max_dr > 0.15739060193300247:
                                            if Q.LHA > 0.27015599608421326:
                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01555635454133153:
                                                    if Q.eccentricity > 0.9771754145622253:
                                                        if Q.sum_pt_top5 > 619.84375:
                                                            if Q.centroid_offset > 0.02172062359750271:
                                                                if Q.log_sum_pt > 6.6784796714782715:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top2 > 415.0625:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.464051723480225:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.7422388195991516:
                                                        if s['Z'] - s['t'] > 1.1401989459991455:
                                                            if s['g'] - s['t'] > -0.06552191264927387:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['t'] > -1.8880527019500732:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.0547995250672102:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.016036249697208405:
                                                if s['g'] - s['W'] > -0.503185510635376:
                                                    if s['q'] - s['W'] > -1.3649035692214966:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04068353213369846:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 19.26154327392578:
                                                        if Q.planar_flow > 0.0704188123345375:
                                                            if Q.tau21 > 0.1466706097126007:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02531410474330187:
                                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mean_eta2 > 0.002891939366236329:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 35.515625:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.018929493613541126:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.006718594813719392:
                                                    if Q.sum_pt_top5 > 632.46875:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10129605233669281:
                                                        if Q.z_dr_0p05_0p1 > 0.2384372279047966:
                                                            if Q.min_pair_mass > 0.5508710741996765:
                                                                if s['W'] - s['t'] > 2.2615031003952026:
                                                                    if Q.dr_2 > 0.06231165491044521:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > -0.43731679022312164:
                                            if Q.mass_top3 > 17.015119552612305:
                                                return 'W'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.9412171840667725:
                                return 'g'   # 41% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 93% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > -0.02305565308779478:
                    if s['g'] - s['t'] > -0.03026519250124693:
                        if s['g'] - s['Z'] > 0.2332133799791336:
                            if s['g'] - s['Z'] > 0.41075992584228516:
                                if s['g'] - s['t'] > 0.13642297685146332:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['q'] > 1.2363606691360474:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2_sq > 0.005477826809510589:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.010461268480867147:
                                    if Q.planar_flow > 0.10920636728405952:
                                        if Q.mass > 16.08976173400879:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0019357585115358233:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.006235213717445731:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.9523594081401825:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.507825613021851:
                                if s['g'] - s['W'] > 0.1625412032008171:
                                    if Q.mass_top5 > 27.86705780029297:
                                        if s['W'] - s['t'] > -3.0738641023635864:
                                            if s['g'] - s['W'] > 0.8575927913188934:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > 0.03081041667610407:
                                            return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.35069213807582855:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.027368362061679363:
                                                    if Q.max_dr > 0.04177124984562397:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.024709745310246944:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 76% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.12047670409083366:
                                    if s['W'] - s['Z'] > -0.023193408735096455:
                                        return 'W'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.113203525543213:
                                            if s['Z'] - s['t'] > -0.057903725653886795:
                                                if Q.tau21 > 0.4845268130302429:
                                                    if Q.sum_pt_top2 > 240.40625:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_2 > 0.03362562507390976:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 37% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 400.109375:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.104862928390503:
                                        if Q.dr_7 > 0.09648124128580093:
                                            if Q.width > 0.003955001709982753:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.04686429537832737:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.0822916030883789:
                                                    if Q.eta_0 > 0.02156829833984375:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_6 > 0.06517250090837479:
                                                            if Q.mean_phi > 0.005757395178079605:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.22389885038137436:
                            if Q.C2 > 0.07522205635905266:
                                return 'g'   # 68% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > -0.11031519249081612:
                                    if Q.log_sum_pt > 6.43860125541687:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 454.4140625:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.z_6 > 0.06698008254170418:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.3879949450492859:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.380530908703804:
                                if Q.C2 > 0.07389610633254051:
                                    return 'g'   # 44% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 92% of the training jets here get this class from the formula
                            else:
                                return 't'   # 99% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.0413708109408617:
                        if s['W'] - s['Z'] > -0.21502411365509033:
                            if Q.z_dr_0_0p05 > 0.9268824756145477:
                                if Q.girth > 0.025535689666867256:
                                    if Q.mass > 10.82301950454712:
                                        if Q.mass_over_sum_pt_sq > 0.0017720134346745908:
                                            if Q.centroid_offset > 0.013016424141824245:
                                                if Q.z_dr_0p1_0p2 > 0.023591346107423306:
                                                    if Q.lam1 > 0.003254050388932228:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.676540851593018:
                                                            if Q.max_dr > 0.1898490935564041:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.27201323211193085:
                                                    if Q.tau21 > 0.3502846658229828:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 0.22820008546113968:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['Z'] > -0.10795053467154503:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.026135909371078014:
                                                if Q.planar_flow > 0.07089425250887871:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.002195473527535796:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -0.06310565583407879:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.0015409880434162915:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.020598500967025757:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.8900429308414459:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > -0.1350734904408455:
                                            if Q.z_6 > 0.04223456233739853:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -0.048287633806467056:
                                                if s['q'] - s['Z'] > -1.1424156427383423:
                                                    if Q.dr_7 > 0.02158635575324297:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 831.46875:
                                        if Q.width > 0.0005813358293380588:
                                            if Q.LHA > 0.17507828772068024:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['Z'] > -0.10969248786568642:
                                                    if Q.z_7 > 0.029028630815446377:
                                                        if Q.z_7 > 0.041336914524436:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.10052350163459778:
                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.11512145400047302:
                                    if s['W'] - s['Z'] > -0.06932038068771362:
                                        if Q.centroid_offset > 0.024966977536678314:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.012618721462786198:
                                                if Q.centroid_offset > 0.01589584071189165:
                                                    if Q.max_dr > 0.14340786635875702:
                                                        if Q.tau32 > 0.30331768095493317:
                                                            if Q.z_dr_0p05_0p1 > 0.21772648394107819:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_2 > 0.04174627177417278:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['q'] > -0.1816004514694214:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.22041060030460358:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006743859965354204:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.04771718010306358:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p1_0p2 > 0.19478625804185867:
                                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.012015155516564846:
                                                    if Q.D2 > 1.925692856311798:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 9.05311135284137e-05:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > -2.8446682691574097:
                                                        if s['q'] - s['W'] > -2.4913487434387207:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['t'] > -1.7383081912994385:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt_sq > 0.0034370197681710124:
                                            if Q.lam1 > 0.006617635954171419:
                                                if Q.z_dr_0p05_0p1 > 0.5675254762172699:
                                                    if Q.pt_dispersion > 0.40512004494667053:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.min_pair_mass > 1.1669487953186035:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.12036509439349174:
                                                    if Q.z_dr_0_0p05 > 0.048684608191251755:
                                                        if Q.max_dr > 0.13428901135921478:
                                                            if Q.mass > 62.88058853149414:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_2 > 0.03488273359835148:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau21 > 0.5049663931131363:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01694537978619337:
                                                                if Q.dr_0 > 0.050113750621676445:
                                                                    if Q.z_dr_0p1_0p2 > 0.17104265093803406:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 52.7285213470459:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 0.00591406668536365:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.033363182097673416:
                                                if s['g'] - s['t'] > -0.15379727631807327:
                                                    if Q.log_sum_pt > 6.589871168136597:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.0021196886664256454:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 48.31266784667969:
                                        if Q.centroid_offset > 0.00537285185419023:
                                            if Q.z_dr_0p1_0p2 > 0.09289305284619331:
                                                if Q.LHA > 0.30870547890663147:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.49445588886737823:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.0893452949821949:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.07868412882089615:
                                            if Q.C2 > 0.023327927105128765:
                                                if Q.width > 0.006699346471577883:
                                                    if Q.phi_1 > 0.0033130645751953125:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.05173249542713165:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top5 > 0.0020700530149042606:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 31.578125:
                                                    if s['W'] - s['Z'] > -0.02764345332980156:
                                                        if Q.pt_4 > 54.40625:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.0493182297796011:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 372.46875:
                                                if Q.max_dr > 0.1060781255364418:
                                                    if Q.width > 0.005713357124477625:
                                                        if Q.sum_pt_top5 > 545.265625:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > -1.203504204750061:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.37119270861148834:
                                if Q.log_sum_pt > 6.527935028076172:
                                    if s['g'] - s['W'] > 0.2900523841381073:
                                        if Q.e2_sq > 0.0001781523387762718:
                                            if s['g'] - s['W'] > 3.292982578277588:
                                                if Q.pt_7 > 29.984375:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.016047751996666193:
                                                    if s['q'] - s['W'] > -1.3082353472709656:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 0.051176151260733604:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > -1.8615316152572632:
                                            if s['q'] - s['Z'] > -0.1823100969195366:
                                                return 'q'   # 43% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9810534715652466:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top2 > 123.0625:
                                        if s['W'] - s['t'] > 0.34436582028865814:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.4439074993133545:
                                                if Q.mean_eta2 > 0.0015343865379691124:
                                                    if Q.girth > 0.06179788149893284:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > 1.3691802024841309:
                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 74% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.31382937729358673:
                                    if s['W'] - s['t'] > 0.3764757812023163:
                                        if s['q'] - s['Z'] > 0.05735963582992554:
                                            return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.028147157281637192:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.00227447843644768:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['W'] > 0.235971137881279:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > -0.05972849205136299:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.42704707384109497:
                                                if Q.e2 > 0.01964691001921892:
                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > 0.3129778206348419:
                                        if s['W'] - s['Z'] > -0.4308314770460129:
                                            if Q.girth2_top2 > 0.0035461189690977335:
                                                if Q.mass > 54.385427474975586:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9813495576381683:
                                                        if Q.max_dr > 0.10335437208414078:
                                                            if Q.lam1 > 0.0067153554409742355:
                                                                if s['Z'] - s['t'] > 1.3239113688468933:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_top5 > 0.7749331891536713:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.027699051424860954:
                                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr01 > 0.15897180885076523:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top3 > 0.005033703288063407:
                                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.080006193369627:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.07869626209139824:
                                                                if s['W'] - s['t'] > 1.0123097896575928:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.006591911660507321:
                                                    if Q.girth2_top2 > 0.0025896886363625526:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9601736068725586:
                                                        if Q.girth > 0.024612722918391228:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 648.6875:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 12.267364978790283:
                                                            if Q.lam2 > 7.808667214703746e-05:
                                                                if s['W'] - s['Z'] > -0.24651547521352768:
                                                                    if Q.pt_0 > 267.875:
                                                                        if Q.sum_pt_top2 > 524.03125:
                                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.057780783623456955:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 4.764816522598267:
                                                if Q.e2 > 0.021442475728690624:
                                                    if s['Z'] - s['t'] > 0.6278603374958038:
                                                        if s['g'] - s['Z'] > -0.7352888584136963:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9894559681415558:
                                                            return 't'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > -0.14574698358774185:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 1.9679707288742065:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > -0.772753894329071:
                                                    if s['W'] - s['Z'] > -5.886834383010864:
                                                        if Q.pt_7 > 59.125:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_eta > -0.03661000169813633:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['Z'] > -0.4490258991718292:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if s['Z'] - s['t'] > 0.5620115399360657:
                                                        if s['W'] - s['Z'] > -0.6681448519229889:
                                                            if Q.lam1 > 0.006874591112136841:
                                                                if Q.mass_top5 > 35.3648567199707:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['g'] - s['t'] > -1.1447942852973938:
                                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if s['q'] - s['Z'] > -0.8006963729858398:
                                                                if s['q'] - s['W'] > 2.4110379219055176:
                                                                    if Q.e2 > 0.020955142565071583:
                                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass_over_sum_pt_sq > 0.003534813644364476:
                                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['W'] > 2.966511368751526:
                                                            if Q.LHA > 0.3285272866487503:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.007890125270932913:
                                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 3.5625596046447754:
                                            if Q.max_dr > 0.11951570585370064:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if s['Z'] - s['t'] > 0.18906434625387192:
                                                if s['W'] - s['t'] > -0.24819643050432205:
                                                    if Q.z_7 > 0.053744686767458916:
                                                        return 'W'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.020967128686606884:
                                                    if Q.tau32 > 0.46753130853176117:
                                                        if Q.z_dr_0p1_0p2 > 0.16825546324253082:
                                                            if Q.girth > 0.07827091217041016:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.07255642861127853:
                                                                    return 't'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.03925277478992939:
                                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.16501522809267044:
                            if Q.e2 > 0.03993668034672737:
                                if Q.z_dr_0p1_0p2 > 0.40377581119537354:
                                    return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.07327953726053238:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p1_0p2 > 1.5:
                                            if Q.pt_0 > 117.84375:
                                                if Q.tau32 > 0.6668021976947784:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 420.671875:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 53% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['t'] > -0.0926649458706379:
                                    return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['W'] > 2.839674472808838:
                                        if Q.dr_0 > 0.036959752440452576:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.19313446432352066:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2_sq > 0.0049927425570786:
                                                if Q.max_pair_mass > 15.869011402130127:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eta_0 > 0.0322723388671875:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > 1.4806710481643677:
                                                            if Q.dr_7 > 0.11609364673495293:
                                                                return 't'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if s['W'] - s['t'] > -2.300351858139038:
                                                    if Q.LHA > 0.30539315938949585:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > -0.014645619317889214:
                                                            if Q.girth > 0.06246479041874409:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['t'] > -0.9154555201530457:
                                                                if Q.max_dr > 0.12346918880939484:
                                                                    return 't'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 45% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -0.3751242309808731:
                                if Q.tau32 > 0.38024336099624634:
                                    if Q.z_dr_0p05_0p1 > 0.9485243856906891:
                                        if Q.centroid_offset > 0.03410710580646992:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.08064394816756248:
                                            if Q.girth > 0.0777372233569622:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 63% of the training jets here get this class from the formula
                            else:
                                return 't'   # 97% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    Q = quantities(pt, eta, phi)
    return decide(Q, scores(Q))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
