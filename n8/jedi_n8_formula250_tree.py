"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.55% (the formula: 65.33%); same class as the formula for 85.87% of jets.  54 leaves, depth 9.
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


def decide(Q):
    if Q.girth2 > 0.009295286610722542:
        if Q.sum_pt > 876.59375:
            return 'g'   # 57% of the training jets here get this class from the formula
        else:
            if Q.mass > 42.22504234313965:
                return 't'   # 95% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.04874814487993717:
                    return 't'   # 81% of the training jets here get this class from the formula
                else:
                    return 'g'   # 62% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.245094299316406:
            if Q.girth2 > 0.006661437917500734:
                if Q.centroid_offset > 0.028851264156401157:
                    return 't'   # 56% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.23070676624774933:
                        return 'Z'   # 45% of the training jets here get this class from the formula
                    else:
                        if Q.girth2 > 0.006880460539832711:
                            return 'Z'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04009445011615753:
                                return 'W'   # 51% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 88% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.15000691264867783:
                    if Q.girth2 > 0.004923145519569516:
                        if Q.centroid_offset > 0.010989651549607515:
                            return 'Z'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0058754789642989635:
                                if Q.max_dr > 0.169140063226223:
                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.22699681669473648:
                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.018207107670605183:
                            if Q.girth2 > 0.003664114628918469:
                                return 'Z'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19974350184202194:
                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.027724992483854294:
                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.02015361748635769:
                                if Q.max_dr > 0.24060555547475815:
                                    if Q.girth2_top3 > 0.0006298379448708147:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 83% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 71% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.024785758927464485:
                        if Q.girth2 > 0.00483433622866869:
                            if Q.max_dr > 0.1125669926404953:
                                return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 53% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 66% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.0026827623369172215:
                            if Q.max_dr > 0.12343917414546013:
                                if Q.z_dr_0p05_0p1 > 0.6104320585727692:
                                    if Q.width > 0.006061566760763526:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 37.171875:
                                return 'g'   # 66% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.012786442879587412:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 63% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top3 > 462.46875:
                if Q.centroid_offset > 0.01520640728995204:
                    if Q.centroid_offset > 0.02448699064552784:
                        if Q.e2_sq > 0.0004996637289877981:
                            return 'W'   # 49% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 74% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.328125:
                            return 'g'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.66852593421936:
                                return 'W'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.01938918698579073:
                                    return 'W'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 54% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 39.515625:
                        if Q.centroid_offset > 0.004444011254236102:
                            return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.893069744110107:
                                return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 51.203125:
                                    if Q.width > 4.741107113659382e-05:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.988492488861084:
                            if Q.pt_7 > 21.6796875:
                                return 'g'   # 68% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 90% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 94% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top5 > 533.484375:
                    if Q.centroid_offset > 0.021947954781353474:
                        if Q.centroid_offset > 0.027416111901402473:
                            return 'Z'   # 58% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 56% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.004252334823831916:
                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 57% of the training jets here get this class from the formula
                else:
                    if Q.mass > 24.713354110717773:
                        if Q.centroid_offset > 0.018959971144795418:
                            if Q.girth2 > 0.004768843529745936:
                                return 'g'   # 55% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 91% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
