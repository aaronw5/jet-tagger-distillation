"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 78.68% (the formula: 80.45%); same class as the formula for 91.80% of jets.  59 leaves, depth 8.
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
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
    if Q.mass > 84.76322174072266:
        if Q.mass_over_sum_pt > 0.09499715268611908:
            if Q.sum_pt > 1084.866455078125:
                if Q.tau32 > 0.4206092059612274:
                    if Q.e2 > 0.04310574010014534:
                        if Q.mass > 179.3884048461914:
                            return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.05019816569983959:
                                return 't'   # 72% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 57% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.040633074939250946:
                        if Q.mass > 191.57489776611328:
                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            return 't'   # 94% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 71% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.03476156108081341:
                    if Q.e2 > 0.04258318990468979:
                        if Q.lam1 > 0.03234411031007767:
                            return 't'   # 68% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.009727940894663334:
                                return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 50% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.944725036621094:
                            return 't'   # 47% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 768.064697265625:
                                return 't'   # 85% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 64% of the training jets here get this class from the formula
                else:
                    if Q.z_top50_slots > 0.9754540324211121:
                        if Q.sum_pt_top40 > 949.234375:
                            if Q.z_top30_slots > 0.9518505036830902:
                                return 'q'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.942647695541382:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 40% of the training jets here get this class from the formula
                        else:
                            return 't'   # 81% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.029353326186537743:
                            if Q.log_sum_pt > 6.901112079620361:
                                return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                return 't'   # 57% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 87% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.976256847381592:
                    return 'g'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02443495485931635:
                        return 'Z'   # 76% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 74% of the training jets here get this class from the formula
            else:
                if Q.mass > 98.54790496826172:
                    if Q.n_real_top50 > 39.5:
                        return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 52% of the training jets here get this class from the formula
                else:
                    if Q.mass_top50 > 85.86580657958984:
                        if Q.girth2_top20 > 0.0035427871625870466:
                            if Q.sum_pt_top50 > 963.1859130859375:
                                return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                return 't'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 56% of the training jets here get this class from the formula
                    else:
                        if Q.C2 > 0.04242449440062046:
                            if Q.girth2_top40 > 0.005356325069442391:
                                if Q.log_sum_pt > 6.87085223197937:
                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt > 0.08474580198526382:
                                return 'Z'   # 58% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 66% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.mass_over_sum_pt > 0.0849158950150013:
                if Q.mass_over_sum_pt_sq > 0.00753265293315053:
                    return 't'   # 69% of the training jets here get this class from the formula
                else:
                    return 'W'   # 37% of the training jets here get this class from the formula
            else:
                if Q.girth2_top20 > 0.0029952499317005277:
                    if Q.sum_pt_top50 > 948.9957275390625:
                        if Q.mass > 83.58809661865234:
                            if Q.D2 > 3.0743356943130493:
                                return 'Z'   # 63% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 87% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 98% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 43% of the training jets here get this class from the formula
                else:
                    if Q.planar_flow > 0.40477268397808075:
                        if Q.log_sum_pt > 6.965287685394287:
                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 45% of the training jets here get this class from the formula
                    else:
                        return 'W'   # 77% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 45.5:
                if Q.sum_pt > 1043.142333984375:
                    return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.79022789001465:
                            if Q.mass_over_sum_pt_sq > 0.006536588305607438:
                                return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 86% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top10 > 11.137710094451904:
                            if Q.log_sum_pt > 6.822614669799805:
                                if Q.lam1 > 0.003927669022232294:
                                    return 'W'   # 46% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 47% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 5.5:
                                if Q.sum_pt_top3 > 472.1875:
                                    return 'q'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 77% of the training jets here get this class from the formula
            else:
                if Q.log_sum_pt > 7.005598783493042:
                    if Q.n_particles > 35.5:
                        return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 86% of the training jets here get this class from the formula
                else:
                    if Q.lam1 > 0.003603111137636006:
                        if Q.lam2 > 0.00035261810990050435:
                            return 'q'   # 76% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 66% of the training jets here get this class from the formula
                    else:
                        if Q.n_particles > 40.5:
                            if Q.sum_pt_top50 > 1051.2225341796875:
                                return 'g'   # 61% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 89% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 99% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
