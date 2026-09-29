"""JEDI-linear jet tagger, 8 particles, 3 features: the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.54% (the formula: 65.49%); same class as the formula for 85.84% of jets.  59 leaves, depth 10.
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
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def decide(Q):
    if Q.girth2 > 0.009197166189551353:
        if Q.sum_pt > 882.578125:
            return 'g'   # 55% of the training jets here get this class from the formula
        else:
            if Q.mass > 45.29232215881348:
                if Q.girth2 > 0.009854927193373442:
                    return 't'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.047675637528300285:
                        return 'Z'   # 60% of the training jets here get this class from the formula
                    else:
                        return 't'   # 81% of the training jets here get this class from the formula
            else:
                return 't'   # 66% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.15652084350586:
            if Q.width > 0.006653153337538242:
                if Q.centroid_offset > 0.028985573910176754:
                    return 't'   # 53% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.006919040577486157:
                        return 'Z'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.040062110871076584:
                            if Q.lam2 > 0.0002717693569138646:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 61% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 90% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.15002717822790146:
                    if Q.girth2 > 0.004923145519569516:
                        if Q.centroid_offset > 0.011066232342272997:
                            if Q.centroid_offset > 0.03423669748008251:
                                return 't'   # 37% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 84% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.005849120439961553:
                                return 'Z'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.20662906020879745:
                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.02011914551258087:
                            if Q.lam1 > 0.0033407355658710003:
                                return 'Z'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2038055956363678:
                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.02247863169759512:
                                if Q.max_dr > 0.22487571090459824:
                                    if Q.girth2 > 0.0032770195975899696:
                                        if Q.centroid_offset > 0.012668156065046787:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 73% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.0026827623369172215:
                        if Q.centroid_offset > 0.024128050543367863:
                            if Q.girth2 > 0.005054939771071076:
                                if Q.max_dr > 0.11049194633960724:
                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03223733976483345:
                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006298912689089775:
                                if Q.e2 > 0.03761158883571625:
                                    return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 35.38075065612793:
                                    if Q.log_sum_pt > 6.944674730300903:
                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 69% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 37.171875:
                            return 'g'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013288761489093304:
                                return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 66% of the training jets here get this class from the formula
        else:
            if Q.z_7 > 0.04599064402282238:
                if Q.centroid_offset > 0.004021958913654089:
                    if Q.centroid_offset > 0.019990808330476284:
                        if Q.log_sum_pt > 6.435824632644653:
                            if Q.centroid_offset > 0.027593708597123623:
                                if Q.centroid_offset > 0.04686124436557293:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 63% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 45% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 24.859736442565918:
                                return 'W'   # 43% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.z_7 > 0.052114056423306465:
                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.007006986532360315:
                                return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 52% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 527.421875:
                        if Q.pt_7 > 49.171875:
                            if Q.mass > 5.134068489074707:
                                return 'g'   # 74% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 83% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 84% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 83% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.015976176597177982:
                    if Q.centroid_offset > 0.02553854137659073:
                        if Q.sum_pt > 700.6796875:
                            return 'Z'   # 64% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 46% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.5641419887542725:
                            if Q.centroid_offset > 0.018392334692180157:
                                return 'W'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 44% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 62% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 593.015625:
                        if Q.pt_7 > 38.546875:
                            if Q.log_sum_pt > 6.898320198059082:
                                return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.0062101553194224834:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1098.32421875:
                                if Q.pt_7 > 23.59375:
                                    return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 96% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 64% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
