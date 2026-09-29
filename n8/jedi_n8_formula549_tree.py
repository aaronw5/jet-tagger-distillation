"""JEDI-linear jet tagger, 8 particles, 3 features: the simplest formula at the network's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.78% (the formula: 65.80%); same class as the formula for 84.33% of jets.  55 leaves, depth 10.
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


def decide(Q):
    if Q.girth2 > 0.009139599744230509:
        if Q.log_sum_pt > 6.777339696884155:
            return 'g'   # 58% of the training jets here get this class from the formula
        else:
            if Q.width > 0.009906836785376072:
                return 't'   # 93% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.04764731973409653:
                    return 'Z'   # 54% of the training jets here get this class from the formula
                else:
                    return 't'   # 76% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.245094299316406:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.027192377485334873:
                    if Q.max_dr > 0.14311284571886063:
                        return 't'   # 64% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 49% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.006914052646607161:
                        return 'Z'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.040092840790748596:
                            return 'W'   # 52% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 88% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1473577842116356:
                    if Q.girth2 > 0.005036225542426109:
                        if Q.centroid_offset > 0.010997515171766281:
                            if Q.centroid_offset > 0.03173349052667618:
                                return 't'   # 37% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.17563309520483017:
                                if Q.width > 0.005610994296148419:
                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 49% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.00603402778506279:
                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.018207107670605183:
                            if Q.lam1 > 0.003343206364661455:
                                if Q.max_dr > 0.1685866415500641:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02617923729121685:
                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19978925585746765:
                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.021229864098131657:
                                if Q.max_dr > 0.2197299674153328:
                                    if Q.width > 0.00327483844012022:
                                        if Q.centroid_offset > 0.012613121885806322:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 70% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.0028181112138554454:
                        if Q.centroid_offset > 0.024551325477659702:
                            if Q.width > 0.005047633312642574:
                                if Q.max_dr > 0.10757903382182121:
                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 50% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006298912689089775:
                                if Q.e2 > 0.037772081792354584:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1059.015625:
                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.01454621460288763:
                            return 'W'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 38.484375:
                                return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 62% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top3 > 465.09375:
                if Q.centroid_offset > 0.01596675906330347:
                    if Q.centroid_offset > 0.025867613963782787:
                        return 'Z'   # 58% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.515625:
                            return 'g'   # 62% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 64% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 41.640625:
                        if Q.centroid_offset > 0.00406828336417675:
                            return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.898178339004517:
                                return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.006408276502043009:
                            if Q.pt_7 > 34.140625:
                                return 'g'   # 51% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1014.0:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.00669264793396:
                                if Q.pt_7 > 23.2734375:
                                    return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 98% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top3 > 395.734375:
                    if Q.centroid_offset > 0.01910274662077427:
                        if Q.centroid_offset > 0.02864647377282381:
                            return 'Z'   # 49% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 50% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.003971492638811469:
                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 64% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.023215947672724724:
                        if Q.mass > 24.064191818237305:
                            if Q.girth2 > 0.004747697617858648:
                                return 'g'   # 40% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 92% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
