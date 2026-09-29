"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities and on differences of additive class scores.

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

Test set (50,000 jets): accuracy 65.24% (the formula: 65.33%); same class as the formula for 95.49% of jets.  906 leaves, depth 20.
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
        z_4=z[4],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_1=phi[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
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
    return (7.713
        + 10.46 * max(0.0, 0.0338 - Q.centroid_offset)
        - 46.27 * max(0.0, Q.eccentricity - 0.997)
        + 26.29 * max(0.0, 0.0775 - Q.girth)
        - 127.2 * max(0.0, 0.0121 - Q.girth2)
        - 0.04245 * max(0.0, 22.5 - Q.mass)
        + 0.001141 * max(0.0, 71.3 - Q.mass)
        + 0.0003499 * max(0.0, Q.sum_pt - 895.0)
        + 58.24 * max(0.0, 0.00449 - Q.width)
        - 0.5065 * max(0.0, Q.z_dr_0_0p05 - 0.852)
        - 955.9 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 150.5 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 0.069 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
        - 1.227 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.008596 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
        + 0.0004638 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        + 0.004991 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
        + 7.781e-05 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
        + 0.1018 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        - 6.964 * max(0.0, 0.0491 - Q.C2)
        - 230.2 * max(0.0, 0.00775 - Q.e2_sq)
        + 1.57 * max(0.0, Q.log_sum_pt - 6.41)
        - 5.865 * max(0.0, 0.0455 - Q.max_dr)
        + 0.1759 * max(0.0, Q.pt_7 - 30.4)
        + 427.4 * max(0.0, 0.009 - Q.width)
        - 5.964 * max(0.0, 0.0562 - Q.z_7)
        + 1007.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 20.02 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 1903.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 8103.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 36.29 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        - 429.3 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
        + 2677.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 1.351 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
        - 1.218 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
        + 0.6536 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
        - 0.0006104 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        - 4.722 * max(0.0, Q.LHA - 0.116)
        - 24900.0 * max(0.0, 4.87e-05 - Q.girth2)
        + 181.1 * max(0.0, 0.0053 - Q.lam1)
        + 4.035 * max(0.0, Q.log_sum_pt - 6.83)
        + 9.575 * max(0.0, Q.log_sum_pt - 6.89)
        + 1.688 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.6374 * max(0.0, 0.654 - Q.planar_flow)
        - 0.08319 * max(0.0, Q.pt_7 - 30.2)
        - 0.02546 * max(0.0, 54.2 - Q.pt_7)
        + 0.001426 * max(0.0, 788.0 - Q.sum_pt)
        - 21.97 * max(0.0, Q.z_7 - 0.045)
        - 76.73 * max(0.0, 0.0185 - Q.z_7)
        - 27720.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.3226 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        - 0.2048 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 11.13 * max(0.0, Q.centroid_offset - 0.0103)
        + 12.42 * max(0.0, Q.e2 - 0.0278)
        + 3.693 * max(0.0, 0.0433 - Q.e2)
        - 695.7 * max(0.0, 0.00905 - Q.girth2)
        + 45.49 * max(0.0, 0.0126 - Q.girth2)
        + 0.01909 * max(0.0, Q.mass - 71.9)
        + 26.96 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 179.0 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
        + 73.29 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 519.6 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 0.1147 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
        - 15.71 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
        + 302.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        + 0.7205 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        - 2.399 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 11.07 * max(0.0, Q.C2 - 0.00878)
        + 432.8 * max(0.0, 0.000389 - Q.lam2)
        + 41.9 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 11.73 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        - 0.001654 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 0.6549 * max(0.0, 0.279 - Q.tau21)
        - 129.6 * max(0.0, Q.width - 0.00219)
        - 0.1865 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
        + 0.1455 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
        + 0.04688 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
        - 4.596 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
        - 0.09033 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        + 20.9 * max(0.0, 0.0238 - Q.dr_0)
        - 13.43 * max(0.0, 0.0374 - Q.e2)
        + 2362.0 * max(0.0, 6.4e-05 - Q.girth2)
        - 23.24 * max(0.0, 0.0333 - Q.z_7)
        - 18.72 * max(0.0, 0.0683 - Q.z_7)
        - 21.75 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        - 74010.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
        - 77.55 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        + 64.69 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 0.2935 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
        - 34110.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 635.8 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 0.04419 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 15.16 * max(0.0, Q.C2 - 0.0101)
        + 10.5 * max(0.0, Q.centroid_offset - 0.0212)
        + 166.7 * max(0.0, 0.0508 - Q.e2)
        + 306.1 * max(0.0, 0.00328 - Q.e2_sq)
        + 21.35 * max(0.0, Q.girth - 0.0868)
        - 638.0 * max(0.0, 0.00862 - Q.girth2)
        - 354.4 * max(0.0, 0.00802 - Q.lam1)
        + 80.57 * max(0.0, Q.lam2 - 0.0029)
        - 193.3 * max(0.0, 0.000518 - Q.lam2)
        - 40.88 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        + 1.76 * max(0.0, Q.max_dr - 0.0279)
        - 21.42 * max(0.0, 0.0133 - Q.width)
        + 0.04223 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 1856.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 61.3 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        + 0.02157 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 55.12 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 102.1 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.001933 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
        - 3.752 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 31.32 * max(0.0, 0.0201 - Q.centroid_offset)
        + 26.92 * max(0.0, 0.0253 - Q.e2)
        + 7.356 * max(0.0, 0.0372 - Q.e2)
        - 186.3 * max(0.0, 0.00115 - Q.e2_sq)
        + 17.12 * max(0.0, 0.089 - Q.girth)
        + 25.76 * max(0.0, Q.girth2 - 0.00445)
        - 607.9 * max(0.0, 0.000708 - Q.girth2)
        - 0.009847 * max(0.0, Q.mass - 80.4)
        - 12.37 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 19.48 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 0.7204 * max(0.0, 0.184 - Q.planar_flow)
        + 212.1 * max(0.0, 0.00564 - Q.width)
        - 305.9 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
        + 0.8007 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 18.13 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 380.7 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        + 250.6 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        + 0.2076 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
        + 0.0005385 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 221.7 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.01468 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        + 53.66 * max(0.0, 0.00366 - Q.centroid_offset)
        - 2.999 * max(0.0, Q.log_sum_pt - 6.71)
        + 0.008341 * max(0.0, Q.pt_7 - 37.1)
        + 11500.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        + 0.1316 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 2557.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
        + 60.77 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 1.155 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 14770.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 1535.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 10.54 * max(0.0, 0.0174 - Q.centroid_offset)
        + 381.6 * max(0.0, Q.lam2 - 0.00136)
        + 0.08352 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.01274 * max(0.0, 31.3 - Q.mass)
        + 1.525 * max(0.0, 0.257 - Q.max_dr)
        + 0.0007864 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 215.1 * max(0.0, 0.00662 - Q.width)
        - 119.5 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
        + 1.662 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
        + 0.9154 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.07051 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 2646.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        - 1.981 * max(0.0, Q.C2 - 0.0596)
        - 13.93 * max(0.0, Q.e2 - 0.0458)
        + 1.182 * max(0.0, 0.0467 - Q.e2)
        + 71.71 * max(0.0, 0.00182 - Q.girth2)
        - 221.7 * max(0.0, Q.lam2 - 0.000227)
        + 5.982 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.5017 * max(0.0, 6.24 - Q.log_sum_pt)
        - 0.04222 * max(0.0, Q.mass - 9.7)
        + 0.3945 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 0.01935 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
        + 0.1289 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
        - 0.3709 * max(0.0, 0.271 - Q.tau32)
        - 1.363 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        - 40.33 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 1774.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        + 2.178 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        + 107.6 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 0.03694 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 4.758 * max(0.0, 0.0355 - Q.C2)
        - 0.09862 * max(0.0, 0.157 - Q.LHA)
        + 0.6545 * max(0.0, Q.centroid_offset - 0.0144)
        - 1.288 * max(0.0, 0.0498 - Q.centroid_offset)
        - 6.765 * max(0.0, Q.girth - 0.0766)
        - 19.77 * max(0.0, 0.0883 - Q.girth)
        + 0.0706 * max(0.0, 0.271 - Q.planar_flow)
        + 0.0009413 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 308.7 * max(0.0, 0.00365 - Q.width)
        + 252.8 * max(0.0, 0.0087 - Q.width)
        - 8.867 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
        - 0.4815 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
        + 0.009458 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 1.713 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 389.8 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 17.13 * max(0.0, Q.e2 - 0.0622)
        + 1.514 * max(0.0, Q.girth2 - 0.0188)
        + 0.0001522 * max(0.0, Q.mass - 91.2)
        - 5998.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 0.1823 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 2.899 * max(0.0, Q.C2 - 0.0664)
        - 152.2 * max(0.0, 0.0509 - Q.e2)
        + 1.161 * max(0.0, 0.15 - Q.girth)
        - 9.832e-05 * max(0.0, Q.pt_0 - 179.0)
        - 0.01033 * max(0.0, 24.8 - Q.pt_7)
        - 0.01901 * max(0.0, Q.sum_pt - 998.0)
        - 17.52 * max(0.0, 0.0159 - Q.width)
        - 3.087 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        - 0.2881 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        - 0.001046 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
        - 0.02034 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
        + 3.378e-06 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        - 0.1365 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
        - 0.2021 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        + 3.407 * max(0.0, 0.0385 - Q.e2)
        - 2.499 * max(0.0, 0.0869 - Q.girth)
        - 67.29 * max(0.0, 0.0131 - Q.girth2)
        + 110.4 * max(0.0, Q.lam1 - 0.00732)
        + 0.006655 * max(0.0, Q.mass - 5.61)
        + 2.579 * max(0.0, 0.177 - Q.max_dr)
        - 153.2 * max(0.0, 0.00741 - Q.width)
        - 0.1612 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
        + 1.161 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        - 19.8 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
        + 238.9 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 5.661 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 2477.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 44.41 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 0.0213 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
        - 77.74 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        + 73.01 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        + 33.46 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
        + 0.0003536 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
        - 15.89 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 116.8 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        + 389.3 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        + 7.591 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        - 0.3703 * max(0.0, 0.711 - Q.D2)
        - 2.251 * max(0.0, Q.LHA - 0.342)
        - 74.42 * max(0.0, 0.0244 - Q.e2)
        + 3.205 * max(0.0, 0.041 - Q.e2)
        + 12.82 * max(0.0, 0.00198 - Q.girth2_top3)
        + 16.62 * max(0.0, 0.00679 - Q.lam1)
        + 385.3 * max(0.0, 0.00813 - Q.lam1)
        + 8.073 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.01145 * max(0.0, 37.4 - Q.mass)
        - 0.031 * max(0.0, 2.12 - Q.n_dr_0_0p05)
        + 7.636 * max(0.0, 0.00695 - Q.width)
        + 0.908 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
        - 0.2491 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
        + 0.01131 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        - 0.002485 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
        - 0.2959 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 0.1613 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 14000.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        - 617.3 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 309.1 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_q(Q):
    return (5.611
        + 12.06 * max(0.0, 0.0338 - Q.centroid_offset)
        + 4.462 * max(0.0, Q.eccentricity - 0.997)
        + 8.421 * max(0.0, 0.0775 - Q.girth)
        - 127.2 * max(0.0, 0.0121 - Q.girth2)
        - 0.02568 * max(0.0, 22.5 - Q.mass)
        + 0.006222 * max(0.0, 71.3 - Q.mass)
        + 0.001762 * max(0.0, Q.sum_pt - 895.0)
        + 152.7 * max(0.0, 0.00449 - Q.width)
        - 0.2861 * max(0.0, Q.z_dr_0_0p05 - 0.852)
        - 653.7 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 215.5 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 0.0299 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
        - 1.118 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        - 0.004185 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
        + 0.0002251 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        + 0.002579 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
        - 3.951e-05 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
        + 0.1026 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        - 5.483 * max(0.0, 0.0491 - Q.C2)
        - 366.4 * max(0.0, 0.00775 - Q.e2_sq)
        - 0.2078 * max(0.0, Q.log_sum_pt - 6.41)
        - 4.971 * max(0.0, 0.0455 - Q.max_dr)
        + 0.06576 * max(0.0, Q.pt_7 - 30.4)
        + 873.7 * max(0.0, 0.009 - Q.width)
        - 1.988 * max(0.0, 0.0562 - Q.z_7)
        - 155.9 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 4.139 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        + 983.5 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 4873.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        + 132.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        + 31.67 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
        - 1104.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        + 2.784 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
        - 0.7017 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
        + 0.06719 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
        - 0.0002785 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        - 2.319 * max(0.0, Q.LHA - 0.116)
        - 1642.0 * max(0.0, 4.87e-05 - Q.girth2)
        + 109.3 * max(0.0, 0.0053 - Q.lam1)
        - 1.234 * max(0.0, Q.log_sum_pt - 6.83)
        + 2.362 * max(0.0, Q.log_sum_pt - 6.89)
        + 0.1302 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.05173 * max(0.0, 0.654 - Q.planar_flow)
        - 0.05944 * max(0.0, Q.pt_7 - 30.2)
        - 0.000202 * max(0.0, 54.2 - Q.pt_7)
        - 0.002054 * max(0.0, 788.0 - Q.sum_pt)
        + 0.2294 * max(0.0, Q.z_7 - 0.045)
        + 13.31 * max(0.0, 0.0185 - Q.z_7)
        - 30190.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        + 0.2196 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        - 0.1126 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 22.2 * max(0.0, Q.centroid_offset - 0.0103)
        + 21.38 * max(0.0, Q.e2 - 0.0278)
        + 9.361 * max(0.0, 0.0433 - Q.e2)
        - 947.5 * max(0.0, 0.00905 - Q.girth2)
        + 73.72 * max(0.0, 0.0126 - Q.girth2)
        + 0.009498 * max(0.0, Q.mass - 71.9)
        + 10.08 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 169.1 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
        + 71.91 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 357.7 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 0.432 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
        - 51.79 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
        - 11.06 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        + 0.6432 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        - 8.47 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 19.58 * max(0.0, Q.C2 - 0.00878)
        + 722.4 * max(0.0, 0.000389 - Q.lam2)
        + 59.51 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 11.47 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        - 0.002884 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 1.084 * max(0.0, 0.279 - Q.tau21)
        - 1.832 * max(0.0, Q.width - 0.00219)
        - 0.8892 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
        + 0.3253 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
        + 0.07115 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
        - 7.888 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
        - 0.08898 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        - 3.883 * max(0.0, 0.0238 - Q.dr_0)
        - 29.3 * max(0.0, 0.0374 - Q.e2)
        + 8339.0 * max(0.0, 6.4e-05 - Q.girth2)
        + 15.15 * max(0.0, 0.0333 - Q.z_7)
        + 8.158 * max(0.0, 0.0683 - Q.z_7)
        - 42.4 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        + 22710.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
        + 56.84 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 48.3 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 0.03094 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
        - 20580.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        - 298.3 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        - 0.01442 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 23.81 * max(0.0, Q.C2 - 0.0101)
        + 19.83 * max(0.0, Q.centroid_offset - 0.0212)
        + 87.63 * max(0.0, 0.0508 - Q.e2)
        + 118.4 * max(0.0, 0.00328 - Q.e2_sq)
        + 8.517 * max(0.0, Q.girth - 0.0868)
        - 855.5 * max(0.0, 0.00862 - Q.girth2)
        - 712.6 * max(0.0, 0.00802 - Q.lam1)
        + 69.23 * max(0.0, Q.lam2 - 0.0029)
        - 299.9 * max(0.0, 0.000518 - Q.lam2)
        - 46.82 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        + 2.016 * max(0.0, Q.max_dr - 0.0279)
        - 60.68 * max(0.0, 0.0133 - Q.width)
        + 0.3427 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 2155.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 63.19 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        + 0.03756 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 69.56 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 90.39 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.001734 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
        - 0.6064 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 49.05 * max(0.0, 0.0201 - Q.centroid_offset)
        + 51.81 * max(0.0, 0.0253 - Q.e2)
        + 49.53 * max(0.0, 0.0372 - Q.e2)
        - 321.2 * max(0.0, 0.00115 - Q.e2_sq)
        + 19.99 * max(0.0, 0.089 - Q.girth)
        - 123.2 * max(0.0, Q.girth2 - 0.00445)
        - 1127.0 * max(0.0, 0.000708 - Q.girth2)
        - 0.001765 * max(0.0, Q.mass - 80.4)
        - 14.19 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 20.92 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 0.5763 * max(0.0, 0.184 - Q.planar_flow)
        + 290.1 * max(0.0, 0.00564 - Q.width)
        - 456.2 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
        + 0.7703 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 21.06 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 147.2 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        + 475.9 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        + 0.0797 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
        + 0.003493 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 488.5 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.005563 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        + 62.08 * max(0.0, 0.00366 - Q.centroid_offset)
        - 2.235 * max(0.0, Q.log_sum_pt - 6.71)
        - 0.003903 * max(0.0, Q.pt_7 - 37.1)
        - 666.1 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        - 0.02186 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 549.4 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
        + 123.0 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 1.035 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 14600.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 965.9 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 35.07 * max(0.0, 0.0174 - Q.centroid_offset)
        + 492.5 * max(0.0, Q.lam2 - 0.00136)
        - 0.7962 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.02671 * max(0.0, 31.3 - Q.mass)
        + 0.5002 * max(0.0, 0.257 - Q.max_dr)
        + 0.04729 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 183.5 * max(0.0, 0.00662 - Q.width)
        - 237.9 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
        + 4.489 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
        + 2.141 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.0645 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 2210.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        - 3.156 * max(0.0, Q.C2 - 0.0596)
        - 28.15 * max(0.0, Q.e2 - 0.0458)
        + 16.57 * max(0.0, 0.0467 - Q.e2)
        + 246.3 * max(0.0, 0.00182 - Q.girth2)
        - 310.3 * max(0.0, Q.lam2 - 0.000227)
        + 2.625 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.3099 * max(0.0, 6.24 - Q.log_sum_pt)
        - 0.04196 * max(0.0, Q.mass - 9.7)
        - 2.529 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 0.04385 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
        + 0.05956 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
        - 2.273 * max(0.0, 0.271 - Q.tau32)
        + 5.43 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        - 66.01 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 1144.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 1.141 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        - 234.6 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        + 0.6865 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 4.351 * max(0.0, 0.0355 - Q.C2)
        - 0.3052 * max(0.0, 0.157 - Q.LHA)
        + 6.378 * max(0.0, Q.centroid_offset - 0.0144)
        + 1.672 * max(0.0, 0.0498 - Q.centroid_offset)
        + 6.421 * max(0.0, Q.girth - 0.0766)
        - 18.89 * max(0.0, 0.0883 - Q.girth)
        + 0.3931 * max(0.0, 0.271 - Q.planar_flow)
        + 0.001797 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 23.1 * max(0.0, 0.00365 - Q.width)
        + 447.6 * max(0.0, 0.0087 - Q.width)
        - 8.281 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
        - 0.5012 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
        + 0.001116 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 1.543 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 515.6 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 16.73 * max(0.0, Q.e2 - 0.0622)
        + 15.1 * max(0.0, Q.girth2 - 0.0188)
        - 0.000402 * max(0.0, Q.mass - 91.2)
        - 7382.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 0.1573 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 4.492 * max(0.0, Q.C2 - 0.0664)
        - 73.82 * max(0.0, 0.0509 - Q.e2)
        + 2.412 * max(0.0, 0.15 - Q.girth)
        + 4.073e-05 * max(0.0, Q.pt_0 - 179.0)
        - 0.002875 * max(0.0, 24.8 - Q.pt_7)
        + 0.005236 * max(0.0, Q.sum_pt - 998.0)
        + 4.057 * max(0.0, 0.0159 - Q.width)
        - 4.218 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        - 0.1511 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 0.0002886 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
        - 0.07802 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
        + 1.24e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        - 0.09868 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
        + 0.4729 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 12.2 * max(0.0, 0.0385 - Q.e2)
        + 5.211 * max(0.0, 0.0869 - Q.girth)
        - 63.83 * max(0.0, 0.0131 - Q.girth2)
        + 84.22 * max(0.0, Q.lam1 - 0.00732)
        + 0.01086 * max(0.0, Q.mass - 5.61)
        + 2.312 * max(0.0, 0.177 - Q.max_dr)
        - 165.8 * max(0.0, 0.00741 - Q.width)
        - 0.127 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
        + 9.855 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        - 12.06 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
        + 817.2 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 32.72 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 1788.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 1272.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 0.01751 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
        + 167.5 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        - 178.8 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        + 13.16 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
        - 0.001838 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
        - 265.1 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 195.0 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 438.2 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        + 3.607 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        - 0.2186 * max(0.0, 0.711 - Q.D2)
        - 1.989 * max(0.0, Q.LHA - 0.342)
        - 106.4 * max(0.0, 0.0244 - Q.e2)
        + 5.794 * max(0.0, 0.041 - Q.e2)
        - 27.08 * max(0.0, 0.00198 - Q.girth2_top3)
        + 79.67 * max(0.0, 0.00679 - Q.lam1)
        + 686.3 * max(0.0, 0.00813 - Q.lam1)
        - 8.154 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.009499 * max(0.0, 37.4 - Q.mass)
        - 0.00404 * max(0.0, 2.12 - Q.n_dr_0_0p05)
        - 83.45 * max(0.0, 0.00695 - Q.width)
        + 0.6122 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
        - 0.04737 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
        + 0.01658 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        - 0.007956 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
        - 0.2006 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        + 1.626 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 15540.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 971.2 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 678.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_W(Q):
    return (-10.53
        - 29.99 * max(0.0, 0.0338 - Q.centroid_offset)
        + 130.2 * max(0.0, Q.eccentricity - 0.997)
        - 185.2 * max(0.0, 0.0775 - Q.girth)
        + 106.1 * max(0.0, 0.0121 - Q.girth2)
        + 0.05751 * max(0.0, 22.5 - Q.mass)
        + 0.01712 * max(0.0, 71.3 - Q.mass)
        - 0.001369 * max(0.0, Q.sum_pt - 895.0)
        - 1245.0 * max(0.0, 0.00449 - Q.width)
        + 3.637 * max(0.0, Q.z_dr_0_0p05 - 0.852)
        - 179.3 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 892.5 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        + 0.1168 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
        + 1.775 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        - 0.003031 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
        - 0.0005186 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        - 0.0125 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
        + 7.064e-05 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
        - 0.1486 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        - 6.421 * max(0.0, 0.0491 - Q.C2)
        + 676.8 * max(0.0, 0.00775 - Q.e2_sq)
        - 0.4346 * max(0.0, Q.log_sum_pt - 6.41)
        - 0.002362 * max(0.0, 0.0455 - Q.max_dr)
        - 0.1948 * max(0.0, Q.pt_7 - 30.4)
        - 1177.0 * max(0.0, 0.009 - Q.width)
        - 3.449 * max(0.0, 0.0562 - Q.z_7)
        - 1467.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 13.62 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 1437.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        - 24370.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 27.15 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        - 116.6 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
        + 834.2 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 0.4772 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
        + 0.3873 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
        + 0.2002 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
        + 0.0001917 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        + 3.166 * max(0.0, Q.LHA - 0.116)
        + 3466.0 * max(0.0, 4.87e-05 - Q.girth2)
        + 61.73 * max(0.0, 0.0053 - Q.lam1)
        + 1.104 * max(0.0, Q.log_sum_pt - 6.83)
        + 3.243 * max(0.0, Q.log_sum_pt - 6.89)
        + 1.033 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.5331 * max(0.0, 0.654 - Q.planar_flow)
        + 0.1798 * max(0.0, Q.pt_7 - 30.2)
        + 0.004404 * max(0.0, 54.2 - Q.pt_7)
        + 0.001658 * max(0.0, 788.0 - Q.sum_pt)
        + 1.835 * max(0.0, Q.z_7 - 0.045)
        + 16.2 * max(0.0, 0.0185 - Q.z_7)
        - 40770.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.4578 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        + 0.2952 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 34.35 * max(0.0, Q.centroid_offset - 0.0103)
        - 54.31 * max(0.0, Q.e2 - 0.0278)
        - 132.4 * max(0.0, 0.0433 - Q.e2)
        + 1788.0 * max(0.0, 0.00905 - Q.girth2)
        + 33.36 * max(0.0, 0.0126 - Q.girth2)
        - 0.06959 * max(0.0, Q.mass - 71.9)
        - 438.6 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 842.7 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
        + 137.4 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        + 4567.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 0.6671 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
        + 151.4 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
        - 8146.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        - 7.695 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 28.21 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 63.89 * max(0.0, Q.C2 - 0.00878)
        + 1279.0 * max(0.0, 0.000389 - Q.lam2)
        - 78.45 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        - 6.007 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.0008559 * max(0.0, 486.0 - Q.sum_pt_top5)
        + 2.82 * max(0.0, 0.279 - Q.tau21)
        - 212.3 * max(0.0, Q.width - 0.00219)
        + 0.05111 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
        - 0.3469 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
        - 0.01126 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
        + 2.739 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
        + 0.06583 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        - 0.1798 * max(0.0, 0.0238 - Q.dr_0)
        + 255.1 * max(0.0, 0.0374 - Q.e2)
        - 5175.0 * max(0.0, 6.4e-05 - Q.girth2)
        + 4.02 * max(0.0, 0.0333 - Q.z_7)
        - 11.6 * max(0.0, 0.0683 - Q.z_7)
        + 19.54 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        + 231700.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
        - 30.76 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 123.9 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        - 0.06536 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
        + 48240.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 291.9 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 0.001357 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 40.67 * max(0.0, Q.C2 - 0.0101)
        + 22.85 * max(0.0, Q.centroid_offset - 0.0212)
        - 33.78 * max(0.0, 0.0508 - Q.e2)
        - 811.0 * max(0.0, 0.00328 - Q.e2_sq)
        - 64.69 * max(0.0, Q.girth - 0.0868)
        + 3659.0 * max(0.0, 0.00862 - Q.girth2)
        + 3837.0 * max(0.0, 0.00802 - Q.lam1)
        + 356.4 * max(0.0, Q.lam2 - 0.0029)
        + 168.5 * max(0.0, 0.000518 - Q.lam2)
        + 61.1 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        - 3.593 * max(0.0, Q.max_dr - 0.0279)
        + 486.1 * max(0.0, 0.0133 - Q.width)
        - 0.9317 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        - 9663.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        - 58.34 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        - 0.1039 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        - 252.1 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        - 197.0 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.00606 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
        + 12.9 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        + 83.74 * max(0.0, 0.0201 - Q.centroid_offset)
        - 272.2 * max(0.0, 0.0253 - Q.e2)
        - 265.4 * max(0.0, 0.0372 - Q.e2)
        + 352.7 * max(0.0, 0.00115 - Q.e2_sq)
        + 53.69 * max(0.0, 0.089 - Q.girth)
        + 833.5 * max(0.0, Q.girth2 - 0.00445)
        + 575.0 * max(0.0, 0.000708 - Q.girth2)
        + 0.09282 * max(0.0, Q.mass - 80.4)
        + 36.88 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 7.156 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 0.9135 * max(0.0, 0.184 - Q.planar_flow)
        - 1922.0 * max(0.0, 0.00564 - Q.width)
        + 429.0 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
        - 0.9493 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        + 55.9 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        - 215.9 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        - 1487.0 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        - 0.2788 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
        + 0.001594 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 1115.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.0009693 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        - 63.96 * max(0.0, 0.00366 - Q.centroid_offset)
        - 0.8789 * max(0.0, Q.log_sum_pt - 6.71)
        + 0.009391 * max(0.0, Q.pt_7 - 37.1)
        + 35550.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        - 0.08044 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        + 9623.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
        - 275.5 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        - 1.552 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        + 47310.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        - 4120.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        - 4.144 * max(0.0, 0.0174 - Q.centroid_offset)
        - 151.2 * max(0.0, Q.lam2 - 0.00136)
        - 0.2658 * max(0.0, Q.log_sum_pt - 6.36)
        - 0.009393 * max(0.0, 31.3 - Q.mass)
        + 5.161 * max(0.0, 0.257 - Q.max_dr)
        - 0.05042 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        - 1162.0 * max(0.0, 0.00662 - Q.width)
        + 22.15 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
        + 5.041 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
        - 1.35 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        - 0.02947 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 2883.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        + 15.85 * max(0.0, Q.C2 - 0.0596)
        + 115.3 * max(0.0, Q.e2 - 0.0458)
        - 79.74 * max(0.0, 0.0467 - Q.e2)
        + 356.7 * max(0.0, 0.00182 - Q.girth2)
        - 510.5 * max(0.0, Q.lam2 - 0.000227)
        + 0.4 * max(0.0, Q.log_sum_pt - 6.69)
        - 0.01194 * max(0.0, 6.24 - Q.log_sum_pt)
        + 0.04578 * max(0.0, Q.mass - 9.7)
        - 41.9 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 0.0445 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
        + 0.1359 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
        - 0.3696 * max(0.0, 0.271 - Q.tau32)
        - 21.41 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 38.34 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        + 1119.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 4.486 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        + 313.5 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        - 0.2187 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        - 26.53 * max(0.0, 0.0355 - Q.C2)
        - 1.824 * max(0.0, 0.157 - Q.LHA)
        - 40.8 * max(0.0, Q.centroid_offset - 0.0144)
        - 6.175 * max(0.0, 0.0498 - Q.centroid_offset)
        + 19.92 * max(0.0, Q.girth - 0.0766)
        - 6.031 * max(0.0, 0.0883 - Q.girth)
        + 2.347 * max(0.0, 0.271 - Q.planar_flow)
        + 0.001732 * max(0.0, 688.0 - Q.sum_pt_top5)
        + 285.8 * max(0.0, 0.00365 - Q.width)
        - 2453.0 * max(0.0, 0.0087 - Q.width)
        - 4.641 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
        + 1.91 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
        - 0.05607 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 17.84 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 1149.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        - 169.6 * max(0.0, Q.e2 - 0.0622)
        + 116.0 * max(0.0, Q.girth2 - 0.0188)
        - 0.04848 * max(0.0, Q.mass - 91.2)
        - 4808.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 1.99 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 1.29 * max(0.0, Q.C2 - 0.0664)
        - 2.799 * max(0.0, 0.0509 - Q.e2)
        - 12.68 * max(0.0, 0.15 - Q.girth)
        + 5.776e-05 * max(0.0, Q.pt_0 - 179.0)
        - 0.02433 * max(0.0, 24.8 - Q.pt_7)
        + 0.006116 * max(0.0, Q.sum_pt - 998.0)
        + 37.58 * max(0.0, 0.0159 - Q.width)
        - 1.338 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        + 0.1729 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 0.002224 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
        + 0.4094 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
        - 1.707e-07 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 0.1171 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
        - 4.394 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 71.17 * max(0.0, 0.0385 - Q.e2)
        + 96.32 * max(0.0, 0.0869 - Q.girth)
        - 325.7 * max(0.0, 0.0131 - Q.girth2)
        - 542.7 * max(0.0, Q.lam1 - 0.00732)
        + 0.0183 * max(0.0, Q.mass - 5.61)
        - 4.567 * max(0.0, 0.177 - Q.max_dr)
        + 2109.0 * max(0.0, 0.00741 - Q.width)
        + 0.03309 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
        - 35.42 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        + 9.938 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
        + 589.8 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        - 201.3 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        + 7884.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        - 50960.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 0.004827 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
        - 194.9 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        + 257.6 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        - 11.06 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
        + 0.005188 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
        + 1400.0 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        - 567.2 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        + 1038.0 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 18.04 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        + 0.8954 * max(0.0, 0.711 - Q.D2)
        + 39.49 * max(0.0, Q.LHA - 0.342)
        + 636.2 * max(0.0, 0.0244 - Q.e2)
        - 93.04 * max(0.0, 0.041 - Q.e2)
        + 275.9 * max(0.0, 0.00198 - Q.girth2_top3)
        - 1033.0 * max(0.0, 0.00679 - Q.lam1)
        - 2655.0 * max(0.0, 0.00813 - Q.lam1)
        - 31.59 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.025 * max(0.0, 37.4 - Q.mass)
        - 0.09311 * max(0.0, 2.12 - Q.n_dr_0_0p05)
        + 1106.0 * max(0.0, 0.00695 - Q.width)
        + 2.14 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
        - 0.5506 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
        - 0.0138 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 0.1392 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
        + 9.929 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 33.58 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 99120.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 6974.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 2796.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_Z(Q):
    return (-6.797
        - 38.51 * max(0.0, 0.0338 - Q.centroid_offset)
        + 3.845 * max(0.0, Q.eccentricity - 0.997)
        + 22.34 * max(0.0, 0.0775 - Q.girth)
        + 442.0 * max(0.0, 0.0121 - Q.girth2)
        + 0.01849 * max(0.0, 22.5 - Q.mass)
        - 0.006656 * max(0.0, 71.3 - Q.mass)
        + 0.002285 * max(0.0, Q.sum_pt - 895.0)
        - 373.5 * max(0.0, 0.00449 - Q.width)
        + 0.5557 * max(0.0, Q.z_dr_0_0p05 - 0.852)
        - 700.8 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        + 852.0 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        + 0.004528 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
        + 1.273 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        + 0.002786 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
        + 3.357e-05 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        + 0.0005942 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
        - 4.27e-05 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
        + 0.04647 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        + 2.82 * max(0.0, 0.0491 - Q.C2)
        + 64.64 * max(0.0, 0.00775 - Q.e2_sq)
        + 0.526 * max(0.0, Q.log_sum_pt - 6.41)
        + 4.261 * max(0.0, 0.0455 - Q.max_dr)
        - 0.3356 * max(0.0, Q.pt_7 - 30.4)
        - 1103.0 * max(0.0, 0.009 - Q.width)
        - 0.6842 * max(0.0, 0.0562 - Q.z_7)
        + 1011.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        - 26.28 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        - 2322.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 5.884 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 116.5 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        - 98.29 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
        + 1519.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        - 1.558 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
        - 0.3489 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
        + 0.3865 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
        + 5.233e-05 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        + 0.7596 * max(0.0, Q.LHA - 0.116)
        + 2117.0 * max(0.0, 4.87e-05 - Q.girth2)
        + 297.4 * max(0.0, 0.0053 - Q.lam1)
        - 1.87 * max(0.0, Q.log_sum_pt - 6.83)
        + 0.07069 * max(0.0, Q.log_sum_pt - 6.89)
        + 0.6953 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.2255 * max(0.0, 0.654 - Q.planar_flow)
        + 0.3357 * max(0.0, Q.pt_7 - 30.2)
        + 0.004324 * max(0.0, 54.2 - Q.pt_7)
        + 0.0008581 * max(0.0, 788.0 - Q.sum_pt)
        - 2.551 * max(0.0, Q.z_7 - 0.045)
        + 4.517 * max(0.0, 0.0185 - Q.z_7)
        - 12580.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.446 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        + 0.1323 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 30.77 * max(0.0, Q.centroid_offset - 0.0103)
        - 88.84 * max(0.0, Q.e2 - 0.0278)
        - 56.87 * max(0.0, 0.0433 - Q.e2)
        + 1494.0 * max(0.0, 0.00905 - Q.girth2)
        + 53.28 * max(0.0, 0.0126 - Q.girth2)
        + 0.01268 * max(0.0, Q.mass - 71.9)
        - 375.4 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 230.5 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
        + 134.8 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        + 8773.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 0.4131 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
        + 353.9 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
        - 3916.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        - 3.377 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 44.9 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 15.31 * max(0.0, Q.C2 - 0.00878)
        - 492.6 * max(0.0, 0.000389 - Q.lam2)
        + 102.0 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 10.01 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.001747 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 1.204 * max(0.0, 0.279 - Q.tau21)
        + 477.1 * max(0.0, Q.width - 0.00219)
        + 0.8992 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
        - 0.3161 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
        + 0.03446 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
        + 1.954 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
        + 0.09417 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        + 1.347 * max(0.0, 0.0238 - Q.dr_0)
        + 382.8 * max(0.0, 0.0374 - Q.e2)
        - 4532.0 * max(0.0, 6.4e-05 - Q.girth2)
        + 6.712 * max(0.0, 0.0333 - Q.z_7)
        - 8.583 * max(0.0, 0.0683 - Q.z_7)
        + 3.394 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        + 39120.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
        - 26.84 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        - 105.8 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        - 0.06799 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
        + 19550.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 154.9 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 0.01319 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 14.8 * max(0.0, Q.C2 - 0.0101)
        - 0.2597 * max(0.0, Q.centroid_offset - 0.0212)
        - 603.2 * max(0.0, 0.0508 - Q.e2)
        - 780.9 * max(0.0, 0.00328 - Q.e2_sq)
        - 16.62 * max(0.0, Q.girth - 0.0868)
        + 2672.0 * max(0.0, 0.00862 - Q.girth2)
        + 1126.0 * max(0.0, 0.00802 - Q.lam1)
        + 120.2 * max(0.0, Q.lam2 - 0.0029)
        + 476.6 * max(0.0, 0.000518 - Q.lam2)
        + 7.558 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        - 6.733 * max(0.0, Q.max_dr - 0.0279)
        + 689.3 * max(0.0, 0.0133 - Q.width)
        - 1.118 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        - 3807.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 21.69 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        - 0.1314 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        - 305.8 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        - 238.4 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.002767 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
        + 18.67 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 13.02 * max(0.0, 0.0201 - Q.centroid_offset)
        - 45.39 * max(0.0, 0.0253 - Q.e2)
        - 383.3 * max(0.0, 0.0372 - Q.e2)
        + 599.9 * max(0.0, 0.00115 - Q.e2_sq)
        - 69.34 * max(0.0, 0.089 - Q.girth)
        + 63.72 * max(0.0, Q.girth2 - 0.00445)
        - 665.8 * max(0.0, 0.000708 - Q.girth2)
        - 0.09251 * max(0.0, Q.mass - 80.4)
        + 44.34 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        - 166.1 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        - 1.547 * max(0.0, 0.184 - Q.planar_flow)
        - 966.2 * max(0.0, 0.00564 - Q.width)
        + 347.4 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
        - 1.566 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        + 68.55 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 2584.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        - 432.4 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        - 0.4599 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
        + 0.01292 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        - 2861.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        - 0.02477 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        - 4.12 * max(0.0, 0.00366 - Q.centroid_offset)
        - 1.141 * max(0.0, Q.log_sum_pt - 6.71)
        + 0.01218 * max(0.0, Q.pt_7 - 37.1)
        - 1159.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        - 0.06836 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 1205.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
        + 157.3 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        - 2.065 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        + 16230.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        - 4590.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 12.54 * max(0.0, 0.0174 - Q.centroid_offset)
        - 29.74 * max(0.0, Q.lam2 - 0.00136)
        - 0.1018 * max(0.0, Q.log_sum_pt - 6.36)
        + 0.005218 * max(0.0, 31.3 - Q.mass)
        - 0.2376 * max(0.0, 0.257 - Q.max_dr)
        - 0.06761 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        - 412.5 * max(0.0, 0.00662 - Q.width)
        + 20.29 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
        + 3.871 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
        + 1.175 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        - 0.009333 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        + 1771.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        + 14.49 * max(0.0, Q.C2 - 0.0596)
        + 155.6 * max(0.0, Q.e2 - 0.0458)
        - 94.52 * max(0.0, 0.0467 - Q.e2)
        + 8.139 * max(0.0, 0.00182 - Q.girth2)
        - 328.2 * max(0.0, Q.lam2 - 0.000227)
        + 0.7281 * max(0.0, Q.log_sum_pt - 6.69)
        + 0.3379 * max(0.0, 6.24 - Q.log_sum_pt)
        + 0.009903 * max(0.0, Q.mass - 9.7)
        - 39.79 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        - 0.0341 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
        + 0.03579 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
        - 0.2199 * max(0.0, 0.271 - Q.tau32)
        - 5.489 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 5.43 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        + 506.1 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        - 1.272 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        - 167.2 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        - 0.2019 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 6.41 * max(0.0, 0.0355 - Q.C2)
        + 5.466 * max(0.0, 0.157 - Q.LHA)
        + 1.238 * max(0.0, Q.centroid_offset - 0.0144)
        + 20.23 * max(0.0, 0.0498 - Q.centroid_offset)
        - 11.14 * max(0.0, Q.girth - 0.0766)
        + 61.94 * max(0.0, 0.0883 - Q.girth)
        - 1.73 * max(0.0, 0.271 - Q.planar_flow)
        - 0.0003556 * max(0.0, 688.0 - Q.sum_pt_top5)
        + 646.6 * max(0.0, 0.00365 - Q.width)
        - 1856.0 * max(0.0, 0.0087 - Q.width)
        + 13.93 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
        + 1.15 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
        + 0.01792 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        + 14.08 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        + 1225.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        - 131.3 * max(0.0, Q.e2 - 0.0622)
        + 90.41 * max(0.0, Q.girth2 - 0.0188)
        + 0.08359 * max(0.0, Q.mass - 91.2)
        - 3681.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 1.475 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        - 11.78 * max(0.0, Q.C2 - 0.0664)
        + 513.2 * max(0.0, 0.0509 - Q.e2)
        + 26.85 * max(0.0, 0.15 - Q.girth)
        + 0.0001499 * max(0.0, Q.pt_0 - 179.0)
        - 0.02575 * max(0.0, 24.8 - Q.pt_7)
        + 0.004343 * max(0.0, Q.sum_pt - 998.0)
        + 134.5 * max(0.0, 0.0159 - Q.width)
        - 19.04 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        + 0.008784 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        + 4.697e-05 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
        + 0.5247 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
        + 1.342e-05 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        - 0.04755 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
        + 0.64 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        - 13.76 * max(0.0, 0.0385 - Q.e2)
        - 58.18 * max(0.0, 0.0869 - Q.girth)
        - 227.2 * max(0.0, 0.0131 - Q.girth2)
        - 487.3 * max(0.0, Q.lam1 - 0.00732)
        + 0.02537 * max(0.0, Q.mass - 5.61)
        - 5.951 * max(0.0, 0.177 - Q.max_dr)
        + 134.4 * max(0.0, 0.00741 - Q.width)
        + 0.3122 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
        + 9.027 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        - 27.61 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
        + 5210.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 54.93 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        + 1043.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 29170.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        - 0.02899 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
        + 321.8 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        - 352.8 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        - 18.61 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
        - 0.01506 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
        - 181.7 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 139.4 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 1706.0 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 20.76 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        + 0.2581 * max(0.0, 0.711 - Q.D2)
        + 24.46 * max(0.0, Q.LHA - 0.342)
        + 274.0 * max(0.0, 0.0244 - Q.e2)
        + 16.77 * max(0.0, 0.041 - Q.e2)
        + 34.52 * max(0.0, 0.00198 - Q.girth2_top3)
        - 108.6 * max(0.0, 0.00679 - Q.lam1)
        - 1169.0 * max(0.0, 0.00813 - Q.lam1)
        - 10.89 * max(0.0, Q.log_sum_pt - 6.9)
        + 0.02235 * max(0.0, 37.4 - Q.mass)
        + 0.02851 * max(0.0, 2.12 - Q.n_dr_0_0p05)
        - 84.67 * max(0.0, 0.00695 - Q.width)
        - 0.1314 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
        + 1.123 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
        - 0.007277 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 0.02416 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
        + 2.332 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        - 6.377 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        - 59810.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        + 2234.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        + 625.1 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def score_t(Q):
    return (4.367
        + 10.07 * max(0.0, 0.0338 - Q.centroid_offset)
        + 12.7 * max(0.0, Q.eccentricity - 0.997)
        + 4.461 * max(0.0, 0.0775 - Q.girth)
        + 51.95 * max(0.0, 0.0121 - Q.girth2)
        + 0.02047 * max(0.0, 22.5 - Q.mass)
        - 0.006113 * max(0.0, 71.3 - Q.mass)
        - 0.001264 * max(0.0, Q.sum_pt - 895.0)
        + 59.41 * max(0.0, 0.00449 - Q.width)
        + 0.2122 * max(0.0, Q.z_dr_0_0p05 - 0.852)
        - 424.4 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
        - 148.7 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
        - 0.0523 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
        - 0.5849 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
        - 0.0112 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
        + 0.000112 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
        + 0.0008441 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
        + 8.139e-05 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
        + 0.1062 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        + 0.6696 * max(0.0, 0.0491 - Q.C2)
        - 304.0 * max(0.0, 0.00775 - Q.e2_sq)
        + 0.2333 * max(0.0, Q.log_sum_pt - 6.41)
        - 3.335 * max(0.0, 0.0455 - Q.max_dr)
        + 0.008322 * max(0.0, Q.pt_7 - 30.4)
        - 156.0 * max(0.0, 0.009 - Q.width)
        - 2.909 * max(0.0, 0.0562 - Q.z_7)
        - 553.5 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
        + 7.577 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
        + 911.4 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
        + 3141.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
        - 15.74 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
        + 45.06 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
        + 59.89 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
        + 1.921 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
        + 0.7589 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
        - 0.03269 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
        + 3.336e-05 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        - 4.747 * max(0.0, Q.LHA - 0.116)
        - 6280.0 * max(0.0, 4.87e-05 - Q.girth2)
        + 102.7 * max(0.0, 0.0053 - Q.lam1)
        + 0.4962 * max(0.0, Q.log_sum_pt - 6.83)
        + 10.78 * max(0.0, Q.log_sum_pt - 6.89)
        - 0.2246 * max(0.0, 6.46 - Q.log_sum_pt)
        - 0.06252 * max(0.0, 0.654 - Q.planar_flow)
        - 0.02058 * max(0.0, Q.pt_7 - 30.2)
        - 0.01118 * max(0.0, 54.2 - Q.pt_7)
        + 0.0002735 * max(0.0, 788.0 - Q.sum_pt)
        - 3.026 * max(0.0, Q.z_7 - 0.045)
        - 10.82 * max(0.0, 0.0185 - Q.z_7)
        + 22530.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
        - 0.05056 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
        + 0.03907 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        - 4.161 * max(0.0, Q.centroid_offset - 0.0103)
        + 6.453 * max(0.0, Q.e2 - 0.0278)
        + 12.79 * max(0.0, 0.0433 - Q.e2)
        + 134.7 * max(0.0, 0.00905 - Q.girth2)
        - 97.19 * max(0.0, 0.0126 - Q.girth2)
        + 0.008395 * max(0.0, Q.mass - 71.9)
        + 46.05 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
        + 79.34 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
        + 81.19 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
        - 1077.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
        - 0.09577 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
        - 37.69 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
        + 1010.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
        + 0.3216 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
        + 0.5286 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        - 13.25 * max(0.0, Q.C2 - 0.00878)
        - 636.5 * max(0.0, 0.000389 - Q.lam2)
        + 0.9437 * max(0.0, Q.mass_over_sum_pt - 0.0885)
        + 12.79 * max(0.0, 0.111 - Q.mass_over_sum_pt)
        + 0.001309 * max(0.0, 486.0 - Q.sum_pt_top5)
        - 0.3232 * max(0.0, 0.279 - Q.tau21)
        + 78.03 * max(0.0, Q.width - 0.00219)
        + 1.11 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
        - 0.2824 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
        - 0.01702 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
        + 6.68 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
        + 0.08419 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        + 18.03 * max(0.0, 0.0238 - Q.dr_0)
        - 30.72 * max(0.0, 0.0374 - Q.e2)
        - 3080.0 * max(0.0, 6.4e-05 - Q.girth2)
        - 45.32 * max(0.0, 0.0333 - Q.z_7)
        - 18.61 * max(0.0, 0.0683 - Q.z_7)
        + 3.773 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
        - 78130.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
        - 171.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
        + 90.8 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
        + 0.3402 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
        + 5098.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
        + 869.4 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
        + 0.03392 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        + 4.741 * max(0.0, Q.C2 - 0.0101)
        + 5.645 * max(0.0, Q.centroid_offset - 0.0212)
        + 524.4 * max(0.0, 0.0508 - Q.e2)
        + 100.4 * max(0.0, 0.00328 - Q.e2_sq)
        + 9.694 * max(0.0, Q.girth - 0.0868)
        - 292.0 * max(0.0, 0.00862 - Q.girth2)
        + 206.3 * max(0.0, 0.00802 - Q.lam1)
        - 22.35 * max(0.0, Q.lam2 - 0.0029)
        - 23.42 * max(0.0, 0.000518 - Q.lam2)
        - 4.229 * max(0.0, Q.mass_over_sum_pt - 0.00941)
        + 0.4315 * max(0.0, Q.max_dr - 0.0279)
        - 95.1 * max(0.0, 0.0133 - Q.width)
        - 0.1246 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
        + 2148.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
        + 64.95 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
        - 0.01389 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
        + 24.62 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
        + 30.65 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
        - 0.004287 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
        - 2.249 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        - 17.59 * max(0.0, 0.0201 - Q.centroid_offset)
        - 0.4724 * max(0.0, 0.0253 - Q.e2)
        + 23.53 * max(0.0, 0.0372 - Q.e2)
        - 176.7 * max(0.0, 0.00115 - Q.e2_sq)
        - 5.31 * max(0.0, 0.089 - Q.girth)
        - 105.6 * max(0.0, Q.girth2 - 0.00445)
        - 331.3 * max(0.0, 0.000708 - Q.girth2)
        + 0.01208 * max(0.0, Q.mass - 80.4)
        - 11.65 * max(0.0, Q.mass_over_sum_pt - 0.0729)
        + 15.69 * max(0.0, Q.mass_over_sum_pt - 0.0906)
        + 0.3479 * max(0.0, 0.184 - Q.planar_flow)
        + 71.66 * max(0.0, 0.00564 - Q.width)
        - 198.0 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
        + 0.7577 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
        - 14.98 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
        + 443.6 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
        + 17.36 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
        - 0.04429 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
        + 0.002697 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
        + 247.9 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
        + 0.01922 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        + 18.94 * max(0.0, 0.00366 - Q.centroid_offset)
        - 1.809 * max(0.0, Q.log_sum_pt - 6.71)
        + 0.004062 * max(0.0, Q.pt_7 - 37.1)
        - 26750.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
        + 0.3325 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
        - 1020.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
        + 98.9 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
        + 0.05884 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
        - 6693.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
        + 1723.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        + 9.038 * max(0.0, 0.0174 - Q.centroid_offset)
        + 95.82 * max(0.0, Q.lam2 - 0.00136)
        - 0.4096 * max(0.0, Q.log_sum_pt - 6.36)
        + 0.0003829 * max(0.0, 31.3 - Q.mass)
        - 0.1682 * max(0.0, 0.257 - Q.max_dr)
        + 0.06896 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
        + 26.9 * max(0.0, 0.00662 - Q.width)
        + 13.35 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
        - 2.421 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
        + 0.02319 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
        + 0.04359 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
        - 1410.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        + 12.64 * max(0.0, Q.C2 - 0.0596)
        - 2.142 * max(0.0, Q.e2 - 0.0458)
        - 14.54 * max(0.0, 0.0467 - Q.e2)
        - 704.2 * max(0.0, 0.00182 - Q.girth2)
        + 126.0 * max(0.0, Q.lam2 - 0.000227)
        - 0.912 * max(0.0, Q.log_sum_pt - 6.69)
        - 1.536 * max(0.0, 6.24 - Q.log_sum_pt)
        - 0.01637 * max(0.0, Q.mass - 9.7)
        + 13.25 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
        + 0.1004 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
        + 0.1077 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
        + 5.293 * max(0.0, 0.271 - Q.tau32)
        - 15.38 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
        + 59.38 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
        - 337.4 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
        + 2.845 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
        + 879.3 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
        - 1.869 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        + 2.771 * max(0.0, 0.0355 - Q.C2)
        - 1.251 * max(0.0, 0.157 - Q.LHA)
        + 7.054 * max(0.0, Q.centroid_offset - 0.0144)
        + 1.944 * max(0.0, 0.0498 - Q.centroid_offset)
        + 0.3142 * max(0.0, Q.girth - 0.0766)
        + 1.479 * max(0.0, 0.0883 - Q.girth)
        + 0.07489 * max(0.0, 0.271 - Q.planar_flow)
        + 0.0001052 * max(0.0, 688.0 - Q.sum_pt_top5)
        - 84.17 * max(0.0, 0.00365 - Q.width)
        + 179.6 * max(0.0, 0.0087 - Q.width)
        - 6.685 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
        + 0.08094 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
        + 0.009933 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
        - 0.7883 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
        - 258.2 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        + 62.64 * max(0.0, Q.e2 - 0.0622)
        - 24.03 * max(0.0, Q.girth2 - 0.0188)
        - 0.03846 * max(0.0, Q.mass - 91.2)
        - 10370.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
        - 3.655 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        + 21.42 * max(0.0, Q.C2 - 0.0664)
        - 506.3 * max(0.0, 0.0509 - Q.e2)
        - 25.17 * max(0.0, 0.15 - Q.girth)
        - 0.0009602 * max(0.0, Q.pt_0 - 179.0)
        + 0.08769 * max(0.0, 24.8 - Q.pt_7)
        - 0.01231 * max(0.0, Q.sum_pt - 998.0)
        - 75.24 * max(0.0, 0.0159 - Q.width)
        + 9.137 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
        + 0.3524 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
        - 0.007375 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
        - 1.525 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
        - 0.0001821 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
        + 0.05437 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
        + 6.025 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        + 9.257 * max(0.0, 0.0385 - Q.e2)
        - 9.609 * max(0.0, 0.0869 - Q.girth)
        + 106.2 * max(0.0, 0.0131 - Q.girth2)
        - 15.01 * max(0.0, Q.lam1 - 0.00732)
        + 0.00945 * max(0.0, Q.mass - 5.61)
        + 0.2299 * max(0.0, 0.177 - Q.max_dr)
        - 19.9 * max(0.0, 0.00741 - Q.width)
        - 0.01345 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
        + 7.162 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
        - 7.717 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
        + 477.3 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
        + 13.53 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
        - 1703.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
        + 391.7 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
        + 0.01546 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
        - 55.44 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
        + 32.78 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
        + 11.31 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
        + 0.00185 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
        + 90.24 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
        + 88.65 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
        - 468.9 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
        - 0.4559 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        - 0.1929 * max(0.0, 0.711 - Q.D2)
        - 1.207 * max(0.0, Q.LHA - 0.342)
        - 26.21 * max(0.0, 0.0244 - Q.e2)
        - 0.03018 * max(0.0, 0.041 - Q.e2)
        - 41.44 * max(0.0, 0.00198 - Q.girth2_top3)
        + 19.76 * max(0.0, 0.00679 - Q.lam1)
        - 242.9 * max(0.0, 0.00813 - Q.lam1)
        + 17.73 * max(0.0, Q.log_sum_pt - 6.9)
        - 0.01554 * max(0.0, 37.4 - Q.mass)
        + 0.007928 * max(0.0, 2.12 - Q.n_dr_0_0p05)
        + 21.76 * max(0.0, 0.00695 - Q.width)
        + 0.5091 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
        + 0.1816 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
        + 0.002115 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
        + 0.01842 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
        + 0.7851 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
        + 2.383 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        + 12740.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
        - 2457.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        - 63.24 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
    )


def scores(Q):
    return {c: f(Q) for c, f in zip(CLASSES, (score_g, score_q, score_W, score_Z, score_t))}


def decide(Q, s):
    if Q.width > 0.009295286610722542:
        if s['g'] - s['t'] > -0.11247846111655235:
            if s['g'] - s['q'] > 0.0619152057915926:
                if s['g'] - s['t'] > 0.1045367531478405:
                    if s['q'] - s['Z'] > 6.580484867095947:
                        if Q.LHA > 0.4573848992586136:
                            if Q.pt_7 > 33.390625:
                                return 'g'   # 100% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['t'] > 0.6136209070682526:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > 0.8866987228393555:
                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.06279439106583595:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 343.125:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['Z'] > 0.20602097362279892:
                            if s['g'] - s['t'] > 0.20145820826292038:
                                if s['g'] - s['q'] > 0.1845184937119484:
                                    if s['g'] - s['t'] > 0.5105668902397156:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.2017124593257904:
                                            return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.022997112944722176:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > -5.327348947525024:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > 4.425429344177246:
                                return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 70% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['Z'] > 0.06189354509115219:
                        if Q.sum_pt_top5 > 339.953125:
                            if Q.sum_pt > 892.8984375:
                                return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 39.285526275634766:
                                    if s['q'] - s['Z'] > 8.86635684967041:
                                        return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['t'] > -0.017102383077144623:
                                            if Q.e2 > 0.056363483890891075:
                                                if Q.dr_7 > 0.12013531103730202:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.020568177103996277:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 76% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.05259438045322895:
                                if Q.centroid_offset > 0.033263321965932846:
                                    if Q.max_dr > 0.16320669651031494:
                                        if Q.e2 > 0.03947511687874794:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > 0.04656731151044369:
                                        if Q.LHA > 0.32524654269218445:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 49.64478302001953:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.432105615735054:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 334.0625:
                                    return 't'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 77% of the training jets here get this class from the formula
            else:
                if s['g'] - s['q'] > -0.1646491438150406:
                    if Q.pt_7 > 27.3203125:
                        if Q.sum_pt_top5 > 547.234375:
                            return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 56% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 75% of the training jets here get this class from the formula
                else:
                    return 'q'   # 91% of the training jets here get this class from the formula
        else:
            if s['Z'] - s['t'] > -0.15404583513736725:
                if s['Z'] - s['t'] > 0.22427601367235184:
                    return 'Z'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.tau32 > 0.4860352724790573:
                        if Q.z_dr_0_0p05 > 0.05025222525000572:
                            return 'Z'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 62.85290718078613:
                                return 't'   # 93% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > 5.171180248260498:
                                    return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 86% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 85% of the training jets here get this class from the formula
            else:
                if s['q'] - s['t'] > -0.12926305830478668:
                    if s['q'] - s['t'] > 0.20811005681753159:
                        return 'q'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.mass_over_sum_pt > 0.1444346010684967:
                            return 'q'   # 72% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['Z'] > -2.8089781999588013:
                                return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 757.25:
                                    return 'q'   # 75% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > 0.08331763744354248:
                                        return 'q'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                else:
                    if s['g'] - s['t'] > -0.3366287648677826:
                        if Q.log_sum_pt > 5.721489429473877:
                            if Q.mass > 69.29921340942383:
                                if s['g'] - s['t'] > -0.18033819645643234:
                                    if Q.centroid_offset > 0.01798394974321127:
                                        return 'g'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.022415501065552235:
                                        if Q.pt_0 > 128.9375:
                                            if Q.z_4 > 0.09709679335355759:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.060702238231897354:
                                    if Q.z_dr_0p05_0p1 > 0.7085288763046265:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > -0.9997442662715912:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03193249553442001:
                                                if Q.z_7 > 0.07237431034445763:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 26.2890625:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > -1.5965962409973145:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 62% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.46019308269023895:
                            if Q.girth2 > 0.024885364808142185:
                                if Q.dr_7 > 0.20466572791337967:
                                    return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 61% of the training jets here get this class from the formula
                            else:
                                return 't'   # 91% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -0.5304411053657532:
                                if Q.tau32 > 0.38163912296295166:
                                    if Q.dr_7 > 0.04404188506305218:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.6703190803527832:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.048999007791280746:
                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 292.546875:
                                    return 't'   # 100% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.689575731754303:
                                        return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 88% of the training jets here get this class from the formula
    else:
        if s['q'] - s['Z'] > 0.11151516810059547:
            if s['g'] - s['q'] > 0.0009253336756955832:
                if s['g'] - s['t'] > 0.00693640299141407:
                    if s['g'] - s['W'] > 0.03812413290143013:
                        if s['g'] - s['q'] > 0.10490540787577629:
                            if s['g'] - s['W'] > 0.3688780814409256:
                                if s['g'] - s['t'] > 0.19668573141098022:
                                    if s['g'] - s['q'] > 0.19696792215108871:
                                        if Q.z_7 > 0.011800992302596569:
                                            if s['q'] - s['W'] > 3.522853136062622:
                                                if s['g'] - s['q'] > 0.9244566559791565:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005355544621124864:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9315855503082275:
                                                            if Q.mass > 17.67582130432129:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > 0.5226997137069702:
                                                    if Q.z_7 > 0.01604436617344618:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.5201008915901184:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['Z'] > 0.04530036449432373:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 542.859375:
                                                if s['g'] - s['t'] > 4.184320449829102:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 3.126647243334446e-05:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > 1.5758156776428223:
                                                    if Q.centroid_offset > 0.005771565949544311:
                                                        if Q.width > 0.0005581122823059559:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.018093405291438103:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.011375728528946638:
                                                            if Q.sum_pt > 523.625:
                                                                if Q.dr_0 > 0.022067046724259853:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth > 0.01738243456929922:
                                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.09594366326928139:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.3790624141693115:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -0.3058084100484848:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['t'] > 0.09860102832317352:
                                                    if Q.LHA > 0.25829291343688965:
                                                        if Q.centroid_offset > 0.05510413646697998:
                                                            if Q.pt_0 > 112.84375:
                                                                return 't'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.546875:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 399.90625:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.9241431653499603:
                                    if Q.e2 > 0.03282530605792999:
                                        if Q.mass > 29.227079391479492:
                                            if Q.eccentricity > 0.959901750087738:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 35.34311485290527:
                                            if Q.e2_sq > 0.0025717979297041893:
                                                if s['g'] - s['W'] > 0.1809374839067459:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 6.457336187362671:
                                        if Q.z_dr_0p05_0p1 > 0.0764164850115776:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.639569520950317:
                                                if Q.centroid_offset > 0.014147354755550623:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['W'] > 0.21128136664628983:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02208772301673889:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.640488386154175:
                                            if Q.centroid_offset > 0.017963859252631664:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.628950357437134:
                                if s['q'] - s['Z'] > 0.9728586673736572:
                                    if Q.log_sum_pt > 7.122059345245361:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.05481025576591492:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.01345576299354434:
                                                if Q.max_dr > 0.031200037337839603:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 621.71875:
                                                    if Q.sum_pt_top3 > 558.53125:
                                                        if s['Z'] - s['t'] > 2.5861968994140625:
                                                            if s['g'] - s['q'] > 0.02470763772726059:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 2.423717498779297:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.003233489580452442:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.02956122998148203:
                                        if s['g'] - s['W'] > 0.4806079715490341:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 2.698761463165283:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.005632841493934393:
                                    if s['g'] - s['q'] > 0.03216453269124031:
                                        if Q.sum_pt > 698.3046875:
                                            if Q.dr_0 > 0.0193465705960989:
                                                if Q.dr_7 > 0.0180541779845953:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.00040789971535559744:
                                                if Q.centroid_offset > 0.008411212358623743:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.06573054566979408:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.014619804453104734:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.007680431008338928:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.007967406418174505:
                                            if Q.max_dr > 0.06314929574728012:
                                                if Q.z_4 > 0.056503901258111:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_pt_above_50 > 5.5:
                                                    if Q.D2 > 1.249209702014923:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.016913093626499176:
                                                        if s['q'] - s['Z'] > 0.8669865429401398:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 1.7403125762939453:
                                        if s['g'] - s['q'] > 0.07485083118081093:
                                            if Q.dr_0 > 0.009648749604821205:
                                                if Q.tau21 > 0.24356983602046967:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.004466152051463723:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > 1.712678074836731:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['W'] > 1.779702365398407:
                                                if s['W'] - s['Z'] > -0.16499760746955872:
                                                    if Q.centroid_offset > 0.0046823283191770315:
                                                        if Q.sum_pt > 714.921875:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.024538559839129448:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 45.328125:
                                                            if Q.eccentricity > 0.9319826364517212:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_4 > 0.11383501440286636:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['W'] > -0.3159303218126297:
                            if s['g'] - s['Z'] > 1.1035254001617432:
                                if Q.e2 > 0.029428978450596333:
                                    if Q.planar_flow > 0.05715668573975563:
                                        return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 36.153499603271484:
                                        if Q.pt_7 > 40.0:
                                            if Q.max_dr > 0.1125066801905632:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 4.644338130950928:
                                    if s['g'] - s['Z'] > 0.7540779411792755:
                                        if Q.LHA > 0.20505647361278534:
                                            if s['g'] - s['t'] > 1.516157865524292:
                                                if s['g'] - s['Z'] > 0.9835931658744812:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.1618601381778717:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 71% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > 1.416422963142395:
                                if Q.e2 > 0.028420050628483295:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.017345826141536236:
                                        if Q.mass > 31.54427719116211:
                                            return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['W'] > -0.7190722525119781:
                                    if Q.sum_pt_top2 > 565.5625:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['Z'] > 0.8183349967002869:
                                            if Q.tau21 > 0.19285178184509277:
                                                if Q.e2 > 0.027008432894945145:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.7490326166152954:
                                                        if Q.LHA > 0.2092265635728836:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 98% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['t'] > -0.14910589903593063:
                        if s['W'] - s['t'] > 0.29128777980804443:
                            return 'W'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p1_0p2 > 0.209961898624897:
                                return 't'   # 64% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 79% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.22940541803836823:
                            if s['W'] - s['Z'] > -0.11478215083479881:
                                return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 619.25:
                                    if s['Z'] - s['t'] > -2.750811219215393:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.1969687938690186:
                                        if Q.e2_sq > 0.006717996904626489:
                                            if s['g'] - s['t'] > -0.10309892147779465:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.12900548428297043:
                                            if Q.pt_7 > 27.28125:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.width > 0.005498616024851799:
                                if Q.log_sum_pt > 6.053375244140625:
                                    return 't'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.12949036061763763:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 3.113533854484558:
                                    return 'g'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 83% of the training jets here get this class from the formula
            else:
                if s['q'] - s['W'] > 0.10917873308062553:
                    if s['q'] - s['t'] > -0.023689215071499348:
                        if s['g'] - s['q'] > -0.09763812273740768:
                            if s['g'] - s['q'] > -0.03790639154613018:
                                if Q.log_sum_pt > 6.653568506240845:
                                    if s['q'] - s['Z'] > 1.010639727115631:
                                        if Q.sum_pt_top2 > 367.5625:
                                            if Q.max_dr > 0.02811212930828333:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.008289271034300327:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['Z'] > 2.3776694536209106:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 44.78125:
                                                if Q.z_7 > 0.06033644638955593:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > 0.6853949427604675:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.019620414823293686:
                                            if Q.e2_sq > 0.0021417660173028708:
                                                return 'g'   # 39% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 1.7971354126930237:
                                                if Q.centroid_offset > 0.004815545864403248:
                                                    if Q.sum_pt > 712.640625:
                                                        if Q.planar_flow > 0.340712308883667:
                                                            if Q.z_7 > 0.04845239035785198:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                if s['g'] - s['Z'] > 2.0607346296310425:
                                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 14.88774585723877:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 2.123140335083008:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 575.203125:
                                                    if s['g'] - s['q'] > -0.022153057157993317:
                                                        if s['Z'] - s['t'] > 1.4175866842269897:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1050.9140625:
                                    if Q.sum_pt_top5 > 1098.65625:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.11682995781302452:
                                        if Q.dr_0 > 0.016739017330110073:
                                            if Q.sum_pt_top3 > 456.1875:
                                                return 'q'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.007864418439567089:
                                            if s['g'] - s['W'] > 0.38856498897075653:
                                                if Q.max_dr > 0.06208500266075134:
                                                    if Q.sum_pt > 727.578125:
                                                        if Q.pt_0 > 201.9375:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['Z'] > -0.500287652015686:
                                                        if Q.planar_flow > 0.07630367577075958:
                                                            return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 559.609375:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0030352541944012046:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.038567716255784035:
                                                    if Q.z_dr_0p05_0p1 > 0.03954440355300903:
                                                        return 'g'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['t'] > 3.1001460552215576:
                                                if Q.centroid_offset > 0.005393586587160826:
                                                    if Q.sum_pt > 751.8984375:
                                                        if Q.sum_pt_top2 > 377.0:
                                                            if s['g'] - s['q'] > -0.05363003350794315:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                if s['Z'] - s['t'] > 2.6893635988235474:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > 1.7688549160957336:
                                                        if Q.planar_flow > 0.38175566494464874:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.694368600845337:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['W'] > 0.4876326471567154:
                                if Q.log_sum_pt > 6.583506345748901:
                                    if s['g'] - s['q'] > -0.20710153877735138:
                                        if Q.log_sum_pt > 6.957726955413818:
                                            if Q.sum_pt > 1237.421875:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 17.612879753112793:
                                                if Q.dr_0 > 0.011926911771297455:
                                                    if Q.sum_pt > 910.4453125:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > -0.13562453538179398:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > 0.8716937005519867:
                                            if Q.sum_pt > 752.421875:
                                                if Q.log_sum_pt > 6.988917589187622:
                                                    if s['g'] - s['q'] > -0.5425141453742981:
                                                        if Q.sum_pt > 1179.8671875:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['Z'] > 1.8168505430221558:
                                                                if Q.C2 > 0.008642764762043953:
                                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['g'] - s['q'] > -0.27656257152557373:
                                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 40.72369194030762:
                                                        if s['q'] - s['W'] > 0.883754700422287:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.986767590045929:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.002045724540948868:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 8.052594184875488:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.08623967319726944:
                                                            if Q.z_4 > 0.1005607359111309:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0008785025565885007:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['t'] > 0.18173113465309143:
                                                                if Q.lam1 > 4.652157804230228e-05:
                                                                    if Q.n_pt_above_50 > 4.5:
                                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.012054577004164457:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.002752698725089431:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0017986297607421875:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.09957025200128555:
                                        if s['W'] - s['Z'] > -0.3696220964193344:
                                            if Q.centroid_offset > 0.002957112155854702:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 12.280951976776123:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_4 > 0.09044191613793373:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['W'] > 1.9953202605247498:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0005897241062484682:
                                                if Q.log_sum_pt > 6.39434814453125:
                                                    if s['q'] - s['t'] > 0.13647743314504623:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.007458610227331519:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > 0.28444769978523254:
                                                        if Q.lam1 > 0.0008301092893816531:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.49622106552124:
                                                    if Q.mass > 11.693252086639404:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 696.265625:
                                                            if Q.z_4 > 0.09058558568358421:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0024061674484983087:
                                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.001895141729619354:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.003199479542672634:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0034840552834793925:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 1.6898007988929749:
                                                if Q.z_7 > 0.02268733736127615:
                                                    if Q.sum_pt > 709.4609375:
                                                        if Q.z_4 > 0.08401340991258621:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['q'] > -0.29368850588798523:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 26.1796875:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if s['W'] - s['t'] > -2.3448575735092163:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > -0.2417328804731369:
                                                    if s['g'] - s['Z'] > 1.4766632318496704:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 25.84375:
                                                        if s['g'] - s['t'] > 3.54865825176239:
                                                            return 'q'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.10818885266780853:
                                                                return 'q'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2399943396449089:
                                    if s['g'] - s['t'] > 0.12402819842100143:
                                        if Q.lam1 > 0.0015132178668864071:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 5.160484215593897e-05:
                                        if s['q'] - s['Z'] > 0.8007034957408905:
                                            if Q.width > 0.0024890362983569503:
                                                return 'q'   # 46% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.00817863829433918:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.666518926620483:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0_0p05 > 0.8898020386695862:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.049537088721990585:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > 0.5695284605026245:
                                                if s['W'] - s['t'] > 4.064108848571777:
                                                    if s['q'] - s['W'] > 0.21053427457809448:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 6.02520267420914e-05:
                                                    if Q.lam1 > 0.0009193017613142729:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['t'] > -0.22843369841575623:
                            if Q.sum_pt_top5 > 586.8125:
                                if Q.tau21 > 0.36518624424934387:
                                    return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.28948256373405457:
                                        return 'q'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.08287195861339569:
                                    return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 39% of the training jets here get this class from the formula
                        else:
                            return 't'   # 95% of the training jets here get this class from the formula
                else:
                    if s['q'] - s['Z'] > 1.2397823929786682:
                        if Q.lam1 > 0.0021333080949261785:
                            if s['q'] - s['W'] > -0.3354034125804901:
                                if s['W'] - s['t'] > 0.20688311010599136:
                                    if Q.centroid_offset > 0.012331622652709484:
                                        return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['Z'] > 1.4350807070732117:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 56% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 41.9085578918457:
                                return 'W'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 83% of the training jets here get this class from the formula
                    else:
                        if s['q'] - s['W'] > -0.12219055369496346:
                            if Q.mass > 5.221503257751465:
                                if s['q'] - s['Z'] > 0.795320063829422:
                                    if Q.lam1 > 0.0017385768587701023:
                                        if Q.z_dr_0p1_0p2 > 0.042838601395487785:
                                            if Q.log_sum_pt > 6.603811740875244:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > 0.4154193252325058:
                                        if Q.lam2 > 3.9486812966060825e-05:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.015303307212889194:
                                                if Q.z_4 > 0.07940806448459625:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['t'] > 0.030207958072423935:
                                            return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 41% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.0003092029655817896:
                                    if Q.sum_pt > 785.375:
                                        return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9801025092601776:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 89% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > 0.694399356842041:
                                if Q.lam1 > 0.001821977086365223:
                                    if s['W'] - s['t'] > 0.12262824922800064:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 37.96730995178223:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > 1.412840485572815:
                                            if Q.max_dr > 0.20507927238941193:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 96% of the training jets here get this class from the formula
        else:
            if s['W'] - s['Z'] > -0.00962398573756218:
                if s['g'] - s['W'] > 0.11191751062870026:
                    if s['g'] - s['W'] > 0.3402712345123291:
                        if Q.mass > 54.28262519836426:
                            if Q.D2 > 0.5140701830387115:
                                if s['g'] - s['t'] > 2.4359253644943237:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 7.032044887542725:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 63% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 88% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['t'] > -0.053634217008948326:
                                if s['g'] - s['Z'] > 0.7669487595558167:
                                    if s['g'] - s['W'] > 0.5419220626354218:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.012687630020081997:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0004881529457634315:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.538930892944336:
                                        if Q.sum_pt > 828.109375:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 4.4396205339580774e-05:
                                                if Q.centroid_offset > 0.02468522172421217:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.0007335488335229456:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > 0.6134608685970306:
                                                    if Q.dr_0 > 0.02292189747095108:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                return 't'   # 83% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.536566972732544:
                            if s['g'] - s['Z'] > 0.6075557172298431:
                                if Q.centroid_offset > 0.008614813908934593:
                                    if s['g'] - s['q'] > 1.288939356803894:
                                        if s['g'] - s['t'] > 2.2938225269317627:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.027954579330980778:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02241786103695631:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.623027086257935:
                                                if s['g'] - s['Z'] > 0.7551634013652802:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0034295329824090004:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.0006359936960507184:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.734628677368164:
                                    if s['g'] - s['W'] > 0.2058926746249199:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.5801008939743042:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -0.9648875594139099:
                                        if Q.sum_pt > 712.546875:
                                            if Q.width > 0.00037774456723127514:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02387609612196684:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 7.30382016627118e-05:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > 0.0926116593182087:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 28.714993476867676:
                                if s['g'] - s['t'] > -0.030423596501350403:
                                    if Q.LHA > 0.2078133150935173:
                                        if Q.width > 0.004806679440662265:
                                            if Q.e2 > 0.04077594727277756:
                                                return 'W'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.003472057287581265:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_0 > 142.1875:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 88% of the training jets here get this class from the formula
                            else:
                                if s['g'] - s['Z'] > 0.3552451431751251:
                                    if Q.centroid_offset > 0.03742020204663277:
                                        return 'W'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.6149699985980988:
                                            if Q.lam2 > 0.00011310593254165724:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.0007219618128146976:
                                                    if Q.z_dr_0_0p05 > 0.7803621292114258:
                                                        if Q.z_dr_0p05_0p1 > 0.058451179414987564:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.4874677658081055:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 673.9765625:
                                        return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                else:
                    if s['W'] - s['Z'] > 0.2065310999751091:
                        if s['g'] - s['W'] > -0.23258205503225327:
                            if s['g'] - s['q'] > 1.213414192199707:
                                if Q.e2_sq > 0.003372144070453942:
                                    if Q.centroid_offset > 0.021480518393218517:
                                        if Q.z_dr_0p1_0p2 > 0.16331768035888672:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 197.1875:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.21322105824947357:
                                            return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 2.8873071642010473e-05:
                                        if Q.mass > 27.519179344177246:
                                            if Q.z_7 > 0.055764215067029:
                                                if s['g'] - s['W'] > 0.025192931294441223:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 2.333723783493042:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if s['W'] - s['Z'] > 1.244419276714325:
                                                            if Q.pt_7 > 49.53125:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.020713524892926216:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.21616951376199722:
                                                    if Q.lam2 > 6.364034925354645e-05:
                                                        if s['g'] - s['q'] > 1.4774987697601318:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 23.170988082885742:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 48.66342544555664:
                                            return 'W'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['q'] > 1.6888944506645203:
                                                if s['W'] - s['Z'] > 0.3555953800678253:
                                                    if Q.dr_7 > 0.029723108746111393:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > 0.0015320726670324802:
                                    if s['g'] - s['Z'] > 0.4712957590818405:
                                        if s['g'] - s['W'] > -0.04008885659277439:
                                            if Q.sum_pt > 723.75:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.027142612263560295:
                                                    if Q.pt_7 > 33.671875:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.2641582190990448:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.562422037124634:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0005494742945302278:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 75% of the training jets here get this class from the formula
                        else:
                            if s['W'] - s['t'] > -0.003477067220956087:
                                if s['W'] - s['Z'] > 0.3735039532184601:
                                    if s['g'] - s['W'] > -0.6151103377342224:
                                        if s['g'] - s['q'] > 1.510798454284668:
                                            if Q.lam1 > 0.0022870589746162295:
                                                if Q.mass > 28.06234645843506:
                                                    if Q.LHA > 0.18753266334533691:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > 1.8586957454681396:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if s['Z'] - s['t'] > 1.6830204129219055:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 389.90625:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > 0.5709832310676575:
                                                if Q.LHA > 0.1967957243323326:
                                                    if Q.max_dr > 0.06952636316418648:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 471.6875:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 36.87703323364258:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > 0.36786705255508423:
                                            if s['W'] - s['Z'] > 0.585706889629364:
                                                if Q.log_sum_pt > 6.994950294494629:
                                                    if s['W'] - s['Z'] > 0.7673408389091492:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > -0.8459288477897644:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.26819002628326416:
                                                if Q.n_dr_0_0p05 > 2.5:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.15961936116218567:
                                        if s['W'] - s['Z'] > 0.2744559496641159:
                                            if Q.z_dr_0_0p05 > 0.408285528421402:
                                                if Q.sum_pt_top5 > 1000.515625:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['q'] > -1.3911665081977844:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00012424429587554187:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.03572767227888107:
                                                if Q.centroid_offset > 0.013233103789389133:
                                                    if Q.z_dr_0p2_0p4 > 0.007624061778187752:
                                                        if Q.centroid_offset > 0.01850940939038992:
                                                            if Q.LHA > 0.21191798895597458:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0039059893460944295:
                                                            if Q.z_dr_0_0p05 > 0.7849781513214111:
                                                                if Q.z_dr_0p1_0p2 > 0.09750760719180107:
                                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['t'] > -2.7308359146118164:
                                            if s['g'] - s['q'] > 0.656775563955307:
                                                if Q.girth > 0.049239762127399445:
                                                    if Q.log_sum_pt > 6.943800449371338:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04790206626057625:
                                                            if Q.e2_sq > 0.005776341538876295:
                                                                if Q.max_dr > 0.134685717523098:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 9.041399243869819e-05:
                                                        if Q.C2 > 0.03131028451025486:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.04267910681664944:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 49% of the training jets here get this class from the formula
                            else:
                                if s['W'] - s['t'] > -0.2994656413793564:
                                    if Q.sum_pt > 637.0:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['q'] > 0.6790133416652679:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 94% of the training jets here get this class from the formula
                    else:
                        if s['W'] - s['Z'] > 0.10832621529698372:
                            if s['W'] - s['t'] > -0.07320800796151161:
                                if s['q'] - s['Z'] > -0.8511970341205597:
                                    if Q.width > 0.0029866512631997466:
                                        if s['g'] - s['q'] > -0.20601807534694672:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > 1.3109171986579895:
                                            if Q.lam2 > 7.486976755899377e-05:
                                                if Q.e2_sq > 0.00018444859597366303:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.020796742290258408:
                                        if Q.centroid_offset > 0.01737223193049431:
                                            if Q.max_dr > 0.15077345073223114:
                                                if Q.planar_flow > 0.04787617176771164:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.033264296129345894:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.02224417496472597:
                                                if s['g'] - s['t'] > -2.275079131126404:
                                                    if Q.log_sum_pt > 6.920705080032349:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.038565076887607574:
                                                            if Q.max_dr > 0.11528046429157257:
                                                                if Q.z_dr_0p05_0p1 > 0.5734991431236267:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.03432312048971653:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1582595407962799:
                                                                if Q.mass > 53.2710018157959:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if s['g'] - s['W'] > -2.5441726446151733:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.4592726528644562:
                                                        if Q.pt_7 > 28.9921875:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.23592153191566467:
                                            if Q.C2 > 0.023235765285789967:
                                                if Q.tau21 > 0.2554914206266403:
                                                    if Q.z_dr_0p2_0p4 > 0.01936828065663576:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.022763993591070175:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.0529816597700119:
                                                if Q.centroid_offset > 0.013250674586743116:
                                                    if Q.max_dr > 0.19974932819604874:
                                                        if Q.centroid_offset > 0.016803620383143425:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 50.43229675292969:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 610.40625:
                                                            if Q.z_dr_0_0p05 > 0.9139242768287659:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.6277676522731781:
                                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.16542251408100128:
                                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 59.72058296203613:
                                                        if Q.lam1 > 0.004395072115585208:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.017018966376781464:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > 1.5692936182022095:
                                                        if Q.centroid_offset > 0.02334075514227152:
                                                            if s['q'] - s['W'] > -1.6657683849334717:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.019795679487287998:
                                                            if s['q'] - s['Z'] > -1.105443000793457:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                            else:
                                return 't'   # 84% of the training jets here get this class from the formula
                        else:
                            if s['q'] - s['Z'] > -0.785763293504715:
                                if Q.z_dr_0p2_0p4 > 0.022830822505056858:
                                    if s['g'] - s['t'] > 0.07243314385414124:
                                        if Q.centroid_offset > 0.014732420910149813:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['t'] > -0.4551539719104767:
                                        if Q.planar_flow > 0.3128467947244644:
                                            if Q.mass > 13.545962810516357:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.023212023079395294:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 40% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.021826057694852352:
                                    if Q.centroid_offset > 0.017364555969834328:
                                        if Q.max_dr > 0.14171335101127625:
                                            if Q.girth2 > 0.0044399588368833065:
                                                if s['W'] - s['Z'] > 0.052967118099331856:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.1249053105711937:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['t'] > 0.0902545303106308:
                                                if Q.max_dr > 0.08918003365397453:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.024655009619891644:
                                                        if Q.C2 > 0.016796918585896492:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.04530277103185654:
                                            if s['g'] - s['Z'] > -3.6159791946411133:
                                                if Q.C2 > 0.04614441469311714:
                                                    if s['q'] - s['t'] > -0.5393641293048859:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['t'] > 0.11208483204245567:
                                                        if Q.tau21 > 0.16328492015600204:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if s['Z'] - s['t'] > 0.10798215493559837:
                                                            if s['g'] - s['Z'] > -3.039324164390564:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.07471628859639168:
                                                                    if Q.girth2 > 0.006811213679611683:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 59.23764419555664:
                                                                        if Q.z_dr_0p1_0p2 > 0.14747519046068192:
                                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.5548980534076691:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.16723380982875824:
                                                        if Q.max_dr > 0.18728594481945038:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.02369927428662777:
                                        if Q.C2 > 0.04198707826435566:
                                            if Q.z_dr_0p2_0p4 > 0.021799640730023384:
                                                if Q.centroid_offset > 0.02093608770519495:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00026109650207217783:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.24435850977897644:
                                                            if s['g'] - s['t'] > 0.2322777584195137:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.30173325538635254:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.03356616757810116:
                                                                    if Q.e2_sq > 0.0036969498032703996:
                                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0035097651416435838:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 19.34375:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.03316282480955124:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.01456748554483056:
                                                if Q.e2_sq > 0.0033169164089486003:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 8.740036355447955e-05:
                                                        if Q.tau21 > 0.2652781158685684:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.01296346029266715:
                                                            if Q.max_dr > 0.1546177640557289:
                                                                if Q.sum_pt_top5 > 636.390625:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -1.2209680676460266:
                                                    if Q.width > 0.0006235228502191603:
                                                        if Q.planar_flow > 0.2874557822942734:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.6528873443603516:
                                                                if Q.D2 > 2.3201942443847656:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.0016212179325520992:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.014376026578247547:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.03782368823885918:
                                                            if s['g'] - s['t'] > -0.20638538897037506:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['t'] > 1.7083483338356018:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
            else:
                if s['g'] - s['Z'] > 0.040866922587156296:
                    if s['g'] - s['t'] > 0.03363275155425072:
                        if s['g'] - s['Z'] > 0.3788582533597946:
                            if s['g'] - s['t'] > 0.18500177562236786:
                                if s['g'] - s['Z'] > 0.5834778547286987:
                                    if Q.D2 > 0.2659897059202194:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if s['W'] - s['t'] > -1.4859241247177124:
                                            return 'Z'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.08063189312815666:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04182609170675278:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03761695511639118:
                                                if Q.LHA > 0.22294148057699203:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 45.46990776062012:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.3005252182483673:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.03530445881187916:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['W'] > 1.0743300318717957:
                                if Q.sum_pt > 642.2109375:
                                    if Q.centroid_offset > 0.007630081847310066:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                else:
                                    if s['g'] - s['Z'] > 0.20041001588106155:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.6968905329704285:
                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.0757453553378582:
                                    if s['g'] - s['Z'] > 0.23729593306779861:
                                        if Q.centroid_offset > 0.036179689690470695:
                                            if Q.C2 > 0.015387843828648329:
                                                if Q.z_dr_0p05_0p1 > 0.6969832479953766:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.4986584186553955:
                                                if s['q'] - s['Z'] > -1.5874081254005432:
                                                    if Q.pt_7 > 40.234375:
                                                        if Q.max_dr > 0.04485715925693512:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.05684184469282627:
                                            if Q.max_dr > 0.07896270975470543:
                                                if Q.centroid_offset > 0.02115570567548275:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['q'] > 1.8848568201065063:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -0.6588291227817535:
                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['Z'] > 0.1702044978737831:
                                                            if Q.pt_7 > 45.40625:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.03643602505326271:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['W'] > 0.2299937754869461:
                                                if Q.centroid_offset > 0.034748589619994164:
                                                    if Q.max_dr > 0.06306912377476692:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['t'] > 1.4122342467308044:
                                                        return 'Z'   # 41% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > -0.6118969321250916:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['t'] > -0.11380640789866447:
                                        if Q.z_7 > 0.05921882018446922:
                                            if s['W'] - s['Z'] > -0.10733604803681374:
                                                if s['q'] - s['Z'] > -0.837166428565979:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0012716020573861897:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 43% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 0.002815891639329493:
                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if s['g'] - s['t'] > -0.17981518805027008:
                            if Q.sum_pt > 575.34375:
                                if Q.tau21 > 0.31351710855960846:
                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 44% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.23473502695560455:
                                    if Q.centroid_offset > 0.029026365838944912:
                                        if Q.sum_pt_top5 > 309.140625:
                                            return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 48% of the training jets here get this class from the formula
                        else:
                            return 't'   # 98% of the training jets here get this class from the formula
                else:
                    if s['Z'] - s['t'] > 0.06686468422412872:
                        if s['W'] - s['Z'] > -0.23791970312595367:
                            if s['W'] - s['Z'] > -0.10632981732487679:
                                if s['q'] - s['Z'] > -0.7167125344276428:
                                    if s['q'] - s['t'] > 1.0347992777824402:
                                        if Q.lam2 > 5.6874398069339804e-05:
                                            if Q.mass > 10.722201824188232:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 47% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.25664135813713074:
                                        if Q.centroid_offset > 0.016705244779586792:
                                            if Q.max_dr > 0.15419603139162064:
                                                if Q.sum_pt_top2 > 358.875:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0032302136532962322:
                                                    if s['W'] - s['t'] > 1.0351772904396057:
                                                        if Q.max_dr > 0.12460814416408539:
                                                            if Q.lam1 > 0.005119801731780171:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 49.61777305603027:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['t'] > 0.9238104522228241:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.00682051875628531:
                                                if Q.sum_pt_top3 > 423.453125:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.991327553987503:
                                                    if Q.dr_0 > 0.051177578046917915:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.07678430527448654:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10447581112384796:
                                                        if Q.z_dr_0_0p05 > 0.5375006198883057:
                                                            if Q.C2 > 0.04261981323361397:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.13765837252140045:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.03839588351547718:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 4.0095062255859375:
                                            if Q.z_dr_0p2_0p4 > 0.018084068782627583:
                                                if Q.LHA > 0.19895294308662415:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 17.203125:
                                                        if Q.tau21 > 0.23974261432886124:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13655031472444534:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.01956047210842371:
                                                if Q.max_dr > 0.15825440734624863:
                                                    if Q.z_dr_0p1_0p2 > 0.08912144228816032:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.021208930760622025:
                                                            if s['g'] - s['t'] > -0.4838160127401352:
                                                                if Q.dr_7 > 0.037387194111943245:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.024659860879182816:
                                                    if Q.z_dr_0p1_0p2 > 0.028357241302728653:
                                                        if Q.lam1 > 0.0033434651559218764:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 497.8125:
                                                                if s['W'] - s['t'] > 2.5605857372283936:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -0.9187146723270416:
                                                            if Q.lam1 > 0.0006944685883354396:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.05060455575585365:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.09760818630456924:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -1.3939566016197205:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02253230568021536:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top3 > 0.002197718247771263:
                                    if Q.centroid_offset > 0.016659426502883434:
                                        if Q.max_dr > 0.13411995768547058:
                                            if Q.dr_0 > 0.050105856731534004:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.15358269959688187:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.648547410964966:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9839853942394257:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 27.657676696777344:
                                                if s['W'] - s['t'] > 0.9934834837913513:
                                                    if s['g'] - s['t'] > -1.6807039380073547:
                                                        if Q.girth2 > 0.003648204728960991:
                                                            if Q.max_dr > 0.12123644724488258:
                                                                if Q.lam1 > 0.005167405353859067:
                                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p05_0p1 > 0.7398099303245544:
                                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.031031912192702293:
                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.planar_flow > 0.08384998887777328:
                                                                        if Q.C2 > 0.022240511141717434:
                                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.07947206124663353:
                                                        if s['W'] - s['Z'] > -0.20619367063045502:
                                                            if Q.planar_flow > 0.10377845540642738:
                                                                if Q.centroid_offset > 0.02246715221554041:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.006833625491708517:
                                            if s['g'] - s['t'] > -1.995718240737915:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.5045925080776215:
                                                if Q.e2 > 0.04430250823497772:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.07298620417714119:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.1860094964504242:
                                                    if Q.z_dr_0_0p05 > 0.6848644614219666:
                                                        if Q.e2_sq > 0.005983870010823011:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['t'] > -1.9109165668487549:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if s['q'] - s['Z'] > -0.6503863036632538:
                                        if Q.e2_sq > 0.0025705911684781313:
                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1767859160900116:
                                                if Q.centroid_offset > 0.03178640641272068:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_0 > 172.4375:
                                                        if Q.z_dr_0p1_0p2 > 0.019971461035311222:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.03554404340684414:
                                                                if s['q'] - s['Z'] > -0.18073994666337967:
                                                                    return 'W'   # 40% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p05_0p1 > 0.04257238656282425:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.003213830175809562:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 5.687402009963989:
                                            if Q.z_dr_0p2_0p4 > 0.019557729363441467:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0005375074979383498:
                                                if Q.z_dr_0p1_0p2 > 0.004483467433601618:
                                                    if Q.girth2 > 0.003332299063913524:
                                                        if Q.max_dr > 0.14989487081766129:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1428338885307312:
                                                            if Q.z_7 > 0.02704277355223894:
                                                                if Q.D2 > 4.040201902389526:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                        else:
                            if s['g'] - s['Z'] > -0.6267774105072021:
                                if s['g'] - s['q'] > 2.1180225610733032:
                                    if Q.e2 > 0.016024275682866573:
                                        if s['g'] - s['W'] > 3.579878807067871:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if s['g'] - s['Z'] > -0.13019566237926483:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 57.734375:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if s['g'] - s['W'] > 0.0491128358989954:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04709920659661293:
                                        if Q.e2 > 0.010920697823166847:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if s['q'] - s['W'] > 3.627328634262085:
                                            if s['g'] - s['t'] > -0.06161220371723175:
                                                if Q.centroid_offset > 0.01581811998039484:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 40% of the training jets here get this class from the formula
                                        else:
                                            if s['q'] - s['Z'] > -0.2813965082168579:
                                                if s['q'] - s['W'] > 0.49893712997436523:
                                                    if Q.e2 > 0.012914158403873444:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if s['q'] - s['Z'] > -0.054266005754470825:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['W'] > 0.17638907581567764:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                if s['g'] - s['Z'] > -0.18827614188194275:
                                                    if Q.width > 0.0016565340338274837:
                                                        if Q.e2 > 0.006169179920107126:
                                                            if Q.e2 > 0.048972371965646744:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.026311437599360943:
                                                                    if Q.centroid_offset > 0.03734302893280983:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau32 > 0.3409142941236496:
                                                                            if Q.girth2_top3 > 0.0016802260652184486:
                                                                                if Q.girth2 > 0.003936818102374673:
                                                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.001837908464949578:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if s['q'] - s['Z'] > -1.868360161781311:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.04229683801531792:
                                                        if Q.z_dr_0p05_0p1 > 0.21837644279003143:
                                                            if s['W'] - s['Z'] > -0.7579405605792999:
                                                                if Q.D2 > 1.5377522110939026:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.23340459913015366:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                            else:
                                if s['q'] - s['Z'] > -0.49263550341129303:
                                    if s['q'] - s['W'] > 0.40794534981250763:
                                        if Q.centroid_offset > 0.019998746924102306:
                                            if Q.z_dr_0p1_0p2 > 0.03465949185192585:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.17537827789783478:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                else:
                                    if s['Z'] - s['t'] > 0.337875172495842:
                                        if s['W'] - s['Z'] > -0.46445436775684357:
                                            if Q.z_dr_0_0p05 > 0.8036449551582336:
                                                if Q.D2 > 7.445019245147705:
                                                    if Q.z_dr_0p2_0p4 > 0.016106409020721912:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.017175729386508465:
                                                        if Q.width > 0.002985268598422408:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1506289690732956:
                                                                if Q.width > 0.0022483437787741423:
                                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.092356264591217:
                                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top3 > 0.0009956996073015034:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.11930324137210846:
                                                    if Q.z_dr_0p05_0p1 > 0.5413030683994293:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006513234693557024:
                                                            if s['Z'] - s['t'] > 1.603614628314972:
                                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.13889256864786148:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.06887566670775414:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 0.759670615196228:
                                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.02153619471937418:
                                                        if Q.girth2 > 0.006463231053203344:
                                                            if Q.lam1 > 0.006950875278562307:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if s['g'] - s['W'] > -2.0766345262527466:
                                                                if Q.tau32 > 0.24044162034988403:
                                                                    if s['W'] - s['Z'] > -0.3074818551540375:
                                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 54.9433708190918:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.006949451752007008:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 580.625:
                                                                    if Q.centroid_offset > 0.024689124897122383:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.log_sum_pt > 6.341042995452881:
                                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.049084337428212166:
                                                if s['g'] - s['t'] > 0.710411787033081:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['Z'] > -0.8219205439090729:
                                                    if s['W'] - s['Z'] > -1.4154821038246155:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.020292526111006737:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 611.453125:
                                                                if s['W'] - s['t'] > -1.421673595905304:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if s['g'] - s['W'] > 8.089314460754395:
                                                        if s['g'] - s['Z'] > -1.3236139416694641:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if s['g'] - s['t'] > 1.4675456881523132:
                                                            if s['g'] - s['q'] > 1.7463202476501465:
                                                                if s['g'] - s['Z'] > -0.9200248718261719:
                                                                    if s['g'] - s['W'] > -0.04264133423566818:
                                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.034966807812452316:
                                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if s['W'] - s['Z'] > -0.6778706014156342:
                                                                if Q.lam1 > 0.006895671598613262:
                                                                    if s['g'] - s['Z'] > -2.576870918273926:
                                                                        if Q.eccentricity > 0.9885790050029755:
                                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if s['Z'] - s['t'] > 0.557737797498703:
                                                                    if s['q'] - s['Z'] > -1.1755523085594177:
                                                                        if Q.max_dr > 0.31877103447914124:
                                                                            if Q.centroid_offset > 0.02052840031683445:
                                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.02149481326341629:
                                                                        if s['W'] - s['Z'] > -6.515212297439575:
                                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.phi_1 > 0.010126113891601562:
                                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.20289810746908188:
                                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if s['Z'] - s['t'] > 0.16269994527101517:
                                            if Q.e2 > 0.02173386886715889:
                                                if s['g'] - s['W'] > 5.538910150527954:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 724.6953125:
                                                        if Q.lam1 > 0.007951028645038605:
                                                            return 't'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.20608215034008026:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.3097802549600601:
                                                if s['q'] - s['W'] > 2.934508442878723:
                                                    if Q.LHA > 0.3351272791624069:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if s['q'] - s['W'] > 1.6979240775108337:
                                                    if Q.tau21 > 0.18660244345664978:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.21581117808818817:
                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                    else:
                        if s['Z'] - s['t'] > -0.19302915781736374:
                            if s['Z'] - s['t'] > -0.04929077439010143:
                                if Q.e2 > 0.0444705355912447:
                                    if Q.e2_sq > 0.00836723716929555:
                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.9629460573196411:
                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.17269782721996307:
                                            if s['q'] - s['t'] > -0.23584482073783875:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.005440633278340101:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.02305649872869253:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if s['W'] - s['Z'] > -0.2535959929227829:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13241012394428253:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.18925441056489944:
                                                        if Q.e2 > 0.042562464252114296:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.00842608092352748:
                                    if s['W'] - s['t'] > -4.541313409805298:
                                        if Q.z_dr_0p1_0p2 > 0.23292290419340134:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.33678556978702545:
                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.9287764728069305:
                                        if s['q'] - s['W'] > -0.548058807849884:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.23975347727537155:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2_sq > 0.006393992807716131:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 65% of the training jets here get this class from the formula
                        else:
                            if s['Z'] - s['t'] > -0.41223685443401337:
                                if Q.dr_7 > 0.039938896894454956:
                                    return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if s['W'] - s['Z'] > -3.8663840293884277:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
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
