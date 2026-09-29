"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.62% (the formula: 65.58%); same class as the formula for 87.45% of jets.  62 leaves, depth 9.
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


def decide(Q):
    if Q.girth2 > 0.00906778173521161:
        if Q.width > 0.0096655348315835:
            if Q.sum_pt > 883.203125:
                return 'g'   # 54% of the training jets here get this class from the formula
            else:
                return 't'   # 96% of the training jets here get this class from the formula
        else:
            if Q.e2 > 0.04606887698173523:
                return 'Z'   # 55% of the training jets here get this class from the formula
            else:
                return 't'   # 81% of the training jets here get this class from the formula
    else:
        if Q.mass > 31.293143272399902:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.027192377485334873:
                    if Q.max_dr > 0.1433788537979126:
                        return 't'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.037238309159874916:
                            return 't'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 70% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.006930230185389519:
                        if Q.z_dr_0p2_0p4 > 0.027957831509411335:
                            if Q.centroid_offset > 0.013856560457497835:
                                return 't'   # 68% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.04040752165019512:
                            return 'W'   # 55% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 86% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.14520669728517532:
                    if Q.width > 0.005255302879959345:
                        if Q.centroid_offset > 0.008072602096945047:
                            if Q.centroid_offset > 0.030012542381882668:
                                return 't'   # 47% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.19862522929906845:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.006041594780981541:
                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.02080067340284586:
                            if Q.width > 0.0033396052895113826:
                                return 'Z'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19993796944618225:
                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.20089343190193176:
                                if Q.width > 0.00354658963624388:
                                    if Q.centroid_offset > 0.01263404544442892:
                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.020703423768281937:
                                        return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.030567455105483532:
                                    return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 50% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.002735531306825578:
                        if Q.centroid_offset > 0.02699857112020254:
                            if Q.girth2 > 0.005164559464901686:
                                return 'Z'   # 64% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006295684026554227:
                                if Q.e2 > 0.03774021193385124:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 35.3755989074707:
                                    return 'W'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.012573959771543741:
                                        return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.953125:
                            return 'g'   # 75% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013564847875386477:
                                return 'W'   # 75% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 72% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 603.484375:
                if Q.centroid_offset > 0.015207459684461355:
                    if Q.centroid_offset > 0.024315545335412025:
                        if Q.mass_over_sum_pt > 0.02637974638491869:
                            return 'W'   # 65% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 71% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.796875:
                            return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.0177337983623147:
                                return 'W'   # 74% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 43% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 44.671875:
                        if Q.girth > 0.006905150134116411:
                            if Q.pt_7 > 48.515625:
                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.005600777454674244:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 68% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.890163898468018:
                                return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.008630370255559683:
                            if Q.pt_7 > 35.671875:
                                return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.039064645767212:
                                if Q.pt_7 > 26.1875:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 98% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.022137119434773922:
                    if Q.log_sum_pt > 6.4331159591674805:
                        if Q.centroid_offset > 0.02702755481004715:
                            if Q.girth2 > 0.0017588083283044398:
                                if Q.e2_sq > 0.0008995103416964412:
                                    return 'W'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 77% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 48% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 24.497224807739258:
                            if Q.sum_pt > 455.09375:
                                return 'W'   # 41% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 75% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 554.078125:
                        if Q.mass_over_sum_pt > 0.01695671770721674:
                            if Q.z_7 > 0.059098176658153534:
                                return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 62% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 83% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 95% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 0
    print('class:', classify(pt, eta, phi))
