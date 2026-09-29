"""JEDI-linear jet tagger, 64 particles, 3 features: the formula with the fewest quantities (31) at the network's accuracy: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 79.13% (the formula: 81.07%); same class as the formula for 91.81% of jets.  56 leaves, depth 9.
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def decide(Q):
    if Q.mass > 84.8592414855957:
        if Q.mass_over_sum_pt > 0.09500768035650253:
            if Q.e2 > 0.038742437958717346:
                if Q.sum_pt > 1087.218017578125:
                    if Q.tau32 > 0.47517435252666473:
                        if Q.mass > 179.3884048461914:
                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04736979305744171:
                                return 't'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 196.10855102539062:
                            return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            return 't'   # 92% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top40 > 751.96044921875:
                        if Q.lam1 > 0.0323406457901001:
                            if Q.tau21 > 0.36105668544769287:
                                return 't'   # 89% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 40% of the training jets here get this class from the formula
                        else:
                            return 't'   # 96% of the training jets here get this class from the formula
                    else:
                        return 't'   # 55% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1029.9437255859375:
                    if Q.n_particles > 60.5:
                        return 'g'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.979100465774536:
                            return 'g'   # 84% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 40% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02890962827950716:
                        if Q.log_sum_pt > 6.902575254440308:
                            if Q.z_top50_slots > 0.9760033786296844:
                                return 'q'   # 45% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 765.974609375:
                                return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.z_top50_slots > 0.9793778657913208:
                            if Q.sum_pt_top40 > 943.0625:
                                return 'q'   # 66% of the training jets here get this class from the formula
                            else:
                                return 't'   # 72% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 81% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.979418992996216:
                    return 'g'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02336365170776844:
                        if Q.mass_over_sum_pt > 0.09147834777832031:
                            return 'g'   # 36% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 85% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 75% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.56694793701172:
                    return 'g'   # 79% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top50 > 963.7841796875:
                        if Q.girth2_top20 > 0.0038280574372038245:
                            if Q.mass > 86.29475021362305:
                                return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.08283007889986038:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.330394983291626:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.9649951457977295:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 80% of the training jets here get this class from the formula
                    else:
                        return 't'   # 71% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.51444244384766:
            if Q.sum_pt_top50 > 961.5888671875:
                if Q.girth2_top20 > 0.0030367206782102585:
                    if Q.mass_over_sum_pt > 0.08326738327741623:
                        return 'Z'   # 39% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 97% of the training jets here get this class from the formula
                else:
                    if Q.log_sum_pt > 6.967252016067505:
                        if Q.mass_top30 > 69.5692253112793:
                            return 'W'   # 69% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 63% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.02524426206946373:
                    if Q.n_dr_0p2_0p4 > 1.5:
                        return 't'   # 79% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 63% of the training jets here get this class from the formula
                else:
                    return 'g'   # 37% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 45.5:
                if Q.sum_pt > 1045.9378662109375:
                    return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.74402046203613:
                            if Q.mass_over_sum_pt_sq > 0.006308590993285179:
                                return 'g'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 57% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 87% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top40 > 935.4697265625:
                            if Q.girth2_top20 > 0.0007474646554328501:
                                if Q.mass_top50 > 69.37508010864258:
                                    return 'W'   # 44% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 68% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 73% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1124.5762939453125:
                    if Q.n_particles > 33.5:
                        return 'g'   # 84% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 39.5:
                        if Q.sum_pt > 1058.0333251953125:
                            return 'g'   # 58% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.lam1 > 0.003404597518965602:
                            if Q.sum_pt > 989.4381103515625:
                                return 'W'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 80% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 98% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
