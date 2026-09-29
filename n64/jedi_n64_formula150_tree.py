"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 79.15% (the formula: 81.02%); same class as the formula for 91.82% of jets.  56 leaves, depth 9.
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


def decide(Q):
    if Q.mass > 84.83890914916992:
        if Q.mass_over_sum_pt_sq > 0.009028869681060314:
            if Q.e2 > 0.038742437958717346:
                if Q.sum_pt > 1087.34814453125:
                    if Q.tau32 > 0.48273229598999023:
                        if Q.mass > 178.00318145751953:
                            return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.047336241230368614:
                                return 't'   # 62% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 188.2408676147461:
                            return 'g'   # 70% of the training jets here get this class from the formula
                        else:
                            return 't'   # 93% of the training jets here get this class from the formula
                else:
                    if Q.lam1 > 0.03233633004128933:
                        if Q.tau21 > 0.35214319825172424:
                            return 't'   # 82% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 40% of the training jets here get this class from the formula
                    else:
                        return 't'   # 95% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1035.702880859375:
                    if Q.n_particles > 59.5:
                        return 'g'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1076.89306640625:
                            return 'g'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 41% of the training jets here get this class from the formula
                else:
                    if Q.z_top50_slots > 0.9715259969234467:
                        if Q.log_sum_pt > 6.895995140075684:
                            if Q.sum_pt_top3 > 426.71875:
                                return 'q'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.030681317672133446:
                                    return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 44% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023077521473169327:
                                return 't'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 57% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.03076271992176771:
                            return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 84% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.977758884429932:
                    return 'g'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.0233558164909482:
                        if Q.log_sum_pt > 6.885926008224487:
                            return 'Z'   # 82% of the training jets here get this class from the formula
                        else:
                            return 't'   # 51% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 72% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.28694534301758:
                    return 'g'   # 77% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top50 > 965.404052734375:
                        if Q.girth2_top20 > 0.0036466537276282907:
                            if Q.mass > 86.01634216308594:
                                return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.046930862590670586:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.007090584607794881:
                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1081.9466552734375:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.32737329602241516:
                            return 't'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 85% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.sum_pt_top50 > 963.5062255859375:
                if Q.girth2_top20 > 0.0029040308436378837:
                    return 'W'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.planar_flow > 0.420708030462265:
                        if Q.sum_pt > 1059.768310546875:
                            return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 48% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 75% of the training jets here get this class from the formula
            else:
                if Q.mass_over_sum_pt > 0.08445882424712181:
                    return 't'   # 79% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.3130234032869339:
                        return 't'   # 55% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 77% of the training jets here get this class from the formula
        else:
            if Q.n_real_top50 > 45.5:
                if Q.log_sum_pt > 6.952676296234131:
                    return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top20 > 44.60230255126953:
                            if Q.log_sum_pt > 6.882007598876953:
                                if Q.girth2_top30 > 0.004073749762028456:
                                    return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 43% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 61% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top40 > 942.18017578125:
                            if Q.girth2_top15 > 0.0005189494404476136:
                                if Q.mass_top50 > 70.28935623168945:
                                    return 'W'   # 45% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9997548162937164:
                                    return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 69% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1113.7164306640625:
                    if Q.n_real_top50 > 33.5:
                        return 'g'   # 83% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 39.5:
                        if Q.log_sum_pt > 6.965435743331909:
                            return 'g'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top30 > 64.39311981201172:
                            if Q.lam2 > 0.00038030183350201696:
                                return 'q'   # 75% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 70% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 98% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
