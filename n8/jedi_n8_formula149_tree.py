"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.52% (the formula: 64.64%); same class as the formula for 87.72% of jets.  56 leaves, depth 10.
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
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
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
    if Q.girth2 > 0.009165859781205654:
        if Q.mass > 43.67366409301758:
            if Q.sum_pt > 882.47265625:
                return 'g'   # 48% of the training jets here get this class from the formula
            else:
                return 't'   # 95% of the training jets here get this class from the formula
        else:
            if Q.sum_pt > 414.2109375:
                return 't'   # 85% of the training jets here get this class from the formula
            else:
                return 'g'   # 61% of the training jets here get this class from the formula
    else:
        if Q.mass > 29.730234146118164:
            if Q.width > 0.0066669650841504335:
                if Q.centroid_offset > 0.03028692863881588:
                    return 't'   # 67% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.2370653823018074:
                        return 'Z'   # 55% of the training jets here get this class from the formula
                    else:
                        if Q.girth2 > 0.006853858241811395:
                            if Q.sum_pt > 509.921875:
                                return 'Z'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.546875:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.040887435898184776:
                                return 'W'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 88% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1599615514278412:
                    if Q.girth2 > 0.004335511475801468:
                        if Q.centroid_offset > 0.012019566260278225:
                            return 'Z'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.005514880875125527:
                                if Q.max_dr > 0.1796054244041443:
                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.22798065841197968:
                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.021438544616103172:
                            if Q.lam1 > 0.0030065097380429506:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0017226602067239583:
                                if Q.max_dr > 0.22609303891658783:
                                    if Q.centroid_offset > 0.01511643873527646:
                                        if Q.lam1 > 0.0030092053348198533:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 66% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.02600828278809786:
                        if Q.width > 0.004224083386361599:
                            if Q.max_dr > 0.11861122772097588:
                                return 'Z'   # 75% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 45% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.0025453109992668033:
                            if Q.max_dr > 0.12358416616916656:
                                if Q.z_dr_0p05_0p1 > 0.565461128950119:
                                    if Q.centroid_offset > 0.012362429406493902:
                                        if Q.lam1 > 0.0054953983053565025:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.0031823458848521113:
                                    return 'W'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.011155994608998299:
                                        return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.012802626937627792:
                                return 'W'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 35.796875:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 72% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 600.140625:
                if Q.centroid_offset > 0.015202948357909918:
                    if Q.centroid_offset > 0.025169466622173786:
                        if Q.mass_over_sum_pt > 0.02235360909253359:
                            return 'W'   # 66% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 41.078125:
                            return 'g'   # 55% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 70% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 43.671875:
                        if Q.centroid_offset > 0.0054222275502979755:
                            return 'g'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.913753032684326:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 60.640625:
                                    return 'g'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1090.4140625:
                            if Q.pt_7 > 21.7890625:
                                return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 91% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.009808523580431938:
                                if Q.z_7 > 0.04390028305351734:
                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 89% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 98% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top5 > 541.015625:
                    if Q.centroid_offset > 0.02323187328875065:
                        if Q.centroid_offset > 0.029145861975848675:
                            return 'Z'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 56% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 71% of the training jets here get this class from the formula
                else:
                    if Q.mass > 25.665185928344727:
                        if Q.centroid_offset > 0.01871349010616541:
                            if Q.log_sum_pt > 6.245288848876953:
                                return 'W'   # 59% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 62% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 84% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 92% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
