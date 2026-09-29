"""JEDI-linear jet tagger, 8 particles, 3 features: the tuned formula (start): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 64.59% (the formula: 65.56%); same class as the formula for 91.20% of jets.  2668 leaves, depth 22.
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


def decide(Q):
    if Q.width > 0.009197441395372152:
        if Q.sum_pt > 874.8046875:
            if Q.C2 > 0.07149962335824966:
                if Q.pt_7 > 47.84375:
                    if Q.max_dr > 0.25824199616909027:
                        return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        return 't'   # 72% of the training jets here get this class from the formula
                else:
                    if Q.mass_over_sum_pt_sq > 0.00986765930429101:
                        return 't'   # 91% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 54% of the training jets here get this class from the formula
            else:
                if Q.pt_7 > 23.9140625:
                    if Q.log_sum_pt > 6.8280439376831055:
                        if Q.lam2 > 0.0006781388365197927:
                            if Q.centroid_offset > 0.017515700310468674:
                                return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                return 't'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.z_7 > 0.042144255712628365:
                            if Q.max_dr > 0.15215439349412918:
                                return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.z_top5 > 0.798931211233139:
                                    return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.033838871866464615:
                                if Q.centroid_offset > 0.01420521643012762:
                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 59% of the training jets here get this class from the formula
                            else:
                                return 't'   # 84% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt > 940.03125:
                        if Q.pt_7 > 16.828125:
                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1027.6796875:
                                return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.24062366038560867:
                            if Q.centroid_offset > 0.015617330558598042:
                                return 'q'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 842.296875:
                                    return 'q'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.lam1 > 0.01646796427667141:
                                return 'q'   # 48% of the training jets here get this class from the formula
                            else:
                                return 't'   # 90% of the training jets here get this class from the formula
        else:
            if Q.mass > 45.279863357543945:
                if Q.girth2 > 0.009846583474427462:
                    if Q.lam1 > 0.033716289326548576:
                        if Q.pt_7 > 40.390625:
                            if Q.tau32 > 0.33933117985725403:
                                if Q.planar_flow > 0.05977956764400005:
                                    if Q.log_sum_pt > 6.296149015426636:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.03730244189500809:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.265625:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.20870544016361237:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_2 > 0.18217995017766953:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.0907110758125782:
                                if Q.tau32 > 0.4820861220359802:
                                    if Q.mass > 119.63004684448242:
                                        if Q.pt_7 > 34.515625:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.048572810366749763:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 36.796875:
                                                if Q.lam2 > 0.0004130457527935505:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.03716009855270386:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 306.6953125:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.040707388892769814:
                                        if Q.mass_top5 > 46.704952239990234:
                                            if Q.log_sum_pt > 6.366312503814697:
                                                return 't'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.2330111637711525:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.26897773146629333:
                                                        if Q.e2 > 0.1138823963701725:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.1369691863656044:
                                                                if Q.lam1 > 0.043422313407063484:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.05245906859636307:
                                    if Q.e2 > 0.07838685065507889:
                                        if Q.lam1 > 0.0364230964332819:
                                            if Q.phi_7 > 0.08856201171875:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_3 > 0.18357642740011215:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.07164992019534111:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.7172369956970215:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 115.79060363769531:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_pair_mass > 28.121374130249023:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 48% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 51.703125:
                            if Q.lam2 > 0.0009815104422159493:
                                return 't'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 107.26811599731445:
                                    if Q.tau32 > 0.16588938981294632:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.6084908843040466:
                                        if Q.e2 > 0.05597006343305111:
                                            if Q.tau32 > 0.19044025242328644:
                                                if Q.pt_6 > 59.34375:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 73.17882919311523:
                                                if Q.tau21 > 0.2975221872329712:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.010174327995628119:
                                                    if Q.D2 > 1.0609716773033142:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.7664158344268799:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 773.65625:
                                            if Q.centroid_offset > 0.015600038692355156:
                                                if Q.eccentricity > 0.9908520877361298:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.025922490283846855:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.04789326339960098:
                                                    if Q.mass > 72.44083786010742:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.0007833836716599762:
                                if Q.girth2 > 0.010596078354865313:
                                    if Q.sum_pt > 299.90625:
                                        if Q.centroid_offset > 0.11146353557705879:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 51.23271942138672:
                                                return 't'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02305657137185335:
                                                    return 't'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.5145831406116486:
                                                        if Q.C2 > 0.08660853654146194:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.8404606282711029:
                                            if Q.pt_7 > 25.75:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.050638096407055855:
                                        if Q.pt_7 > 32.625:
                                            if Q.z_7 > 0.06813745945692062:
                                                if Q.centroid_offset > 0.022588573396205902:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 119.11079025268555:
                                    if Q.tau32 > 0.3745507001876831:
                                        if Q.pt_7 > 38.203125:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 127.1347427368164:
                                                if Q.pt_7 > 24.921875:
                                                    if Q.log_sum_pt > 6.675803184509277:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02281002141535282:
                                                    if Q.lam1 > 0.027430537156760693:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 20.1859712600708:
                                        if Q.sum_pt > 379.0703125:
                                            if Q.max_pair_mass > 11.000325202941895:
                                                if Q.girth2 > 0.010343240574002266:
                                                    if Q.pt_7 > 45.109375:
                                                        if Q.girth > 0.1630864143371582:
                                                            if Q.lam1 > 0.03140348009765148:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 835.890625:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 297.671875:
                                                            if Q.pt_4 > 20.140625:
                                                                if Q.mass > 111.1190414428711:
                                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 0.6820335686206818:
                                                                        if Q.lam1 > 0.012179916724562645:
                                                                            return 't'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.pt_7 > 38.140625:
                                                                                if Q.C2 > 0.048154715448617935:
                                                                                    if Q.tau32 > 0.34682950377464294:
                                                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 't'   # 91% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.015945599414408207:
                                                                if Q.centroid_offset > 0.07091891393065453:
                                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau32 > 0.4981785714626312:
                                                                        if Q.C2 > 0.036245763301849365:
                                                                            if Q.planar_flow > 0.043495019897818565:
                                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.041614826768636703:
                                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.22112084925174713:
                                                        if Q.mass_over_sum_pt_sq > 0.009656406473368406:
                                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.35740916430950165:
                                                            return 't'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.061041755601763725:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.1612313613295555:
                                                    if Q.e2 > 0.07354943454265594:
                                                        if Q.log_sum_pt > 6.431959867477417:
                                                            return 'g'   # 38% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top5 > 0.7862945199012756:
                                                            if Q.z_7 > 0.052651889622211456:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 799.2734375:
                                                        if Q.pt_7 > 41.46875:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.033419014886021614:
                                                                if Q.girth2_top2 > 0.004941016668453813:
                                                                    if Q.C2 > 0.024466841481626034:
                                                                        if Q.pt_7 > 29.1953125:
                                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_6 > 0.03203999809920788:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.040885357186198235:
                                                            if Q.mass > 57.849355697631836:
                                                                if Q.pt_7 > 45.28125:
                                                                    if Q.tau32 > 0.2709432691335678:
                                                                        if Q.e2 > 0.05461557023227215:
                                                                            return 't'   # 81% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_6 > 0.027889540418982506:
                                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.18691116571426392:
                                                                    if Q.sum_pt_top3 > 300.671875:
                                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 37.703125:
                                                                            if Q.tau32 > 0.35167166590690613:
                                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.019645621068775654:
                                                                                if Q.pt_4 > 44.140625:
                                                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.mean_eta2 > 0.009859026875346899:
                                                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.mass > 52.36644744873047:
                                                                                    return 't'   # 63% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 104.72340774536133:
                                                                if Q.pt_7 > 38.203125:
                                                                    if Q.centroid_offset > 0.02979314513504505:
                                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 44.703125:
                                                                    if Q.sum_pt_top5 > 609.5:
                                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.04594946838915348:
                                                                            if Q.eccentricity > 0.9952459931373596:
                                                                                return 'g'   # 51% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 5.802165508270264:
                                                if Q.dr_0 > 0.15179037302732468:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.10841035842895508:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.5239178836345673:
                                                            if Q.planar_flow > 0.07185705378651619:
                                                                if Q.centroid_offset > 0.024822603911161423:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_4 > 39.65625:
                                                                return 't'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.022651851177215576:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.026859387755393982:
                                                    if Q.eccentricity > 0.9885388910770416:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 785.640625:
                                            if Q.pt_7 > 32.28125:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.07665246352553368:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.03930792585015297:
                                                if Q.pt_7 > 38.453125:
                                                    if Q.lam2 > 0.0002061832492472604:
                                                        if Q.centroid_offset > 0.021514486521482468:
                                                            if Q.z_7 > 0.07558457553386688:
                                                                if Q.tau21 > 0.310802161693573:
                                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 56.25887107849121:
                                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt_sq > 0.01304115355014801:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.2591776102781296:
                                                            if Q.z_dr_0p2_0p4 > 0.11029922589659691:
                                                                return 't'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.011790880002081394:
                                                                    if Q.z_7 > 0.0796911008656025:
                                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top5 > 472.234375:
                                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 474.546875:
                                                        if Q.sum_pt_top3 > 477.5625:
                                                            if Q.lam2 > 0.00011398519927752204:
                                                                return 't'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2_sq > 0.009644117671996355:
                                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00014633924001827836:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.02231080923229456:
                                                                    if Q.mass > 60.1320743560791:
                                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 35.078125:
                                                                            if Q.min_pair_mass > 0.5504505634307861:
                                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 74% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9856363236904144:
                                                            if Q.mass_top5 > 8.46327543258667:
                                                                return 't'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.017734212800860405:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.149291031062603:
                                                    if Q.width > 0.027145614847540855:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.04686764441430569:
                        if Q.tau32 > 0.4565730541944504:
                            if Q.width > 0.009367589838802814:
                                if Q.pt_7 > 39.625:
                                    if Q.max_dr > 0.11550052464008331:
                                        if Q.z_dr_0_0p05 > 0.06550467386841774:
                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.15072356909513474:
                                        if Q.girth2_top3 > 0.010598697699606419:
                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 37.15625:
                                    if Q.mass_top3 > 25.615859031677246:
                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 42.296875:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 674.125:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.6640625:
                                            if Q.m01 > 20.627148628234863:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 32.625:
                                if Q.centroid_offset > 0.028521733358502388:
                                    return 't'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.009508721996098757:
                                        if Q.z_dr_0p05_0p1 > 0.5871067643165588:
                                            if Q.tau21 > 0.16203132271766663:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top5 > 49.54878044128418:
                                    if Q.tau32 > 0.33299924433231354:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2219378501176834:
                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 36.359375:
                            if Q.D2 > 1.1064780354499817:
                                if Q.lam2 > 0.0003685594565467909:
                                    if Q.m01 > 1.0825269222259521:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 2.7455936670303345:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 70.08917617797852:
                                    if Q.centroid_offset > 0.015097796451300383:
                                        return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.04200104624032974:
                                        if Q.dr_7 > 0.06485529243946075:
                                            if Q.max_dr > 0.11268895119428635:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top5 > 43.06203651428223:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023154514841735363:
                                if Q.log_sum_pt > 6.710427761077881:
                                    if Q.z_dr_0p1_0p2 > 0.16705083847045898:
                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 55.09119987487793:
                                        if Q.pt_7 > 33.296875:
                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 408.9375:
                                    return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 60% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.03272750414907932:
                    if Q.log_sum_pt > 5.771416902542114:
                        if Q.lam2 > 0.00022784526663599536:
                            if Q.tau21 > 0.2214067503809929:
                                if Q.centroid_offset > 0.04200576990842819:
                                    if Q.girth2 > 0.01035652169957757:
                                        if Q.eccentricity > 0.920948714017868:
                                            if Q.mass_over_sum_pt > 0.09207228198647499:
                                                if Q.pt_2 > 42.71875:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.2817666381597519:
                                            if Q.sum_pt_top5 > 344.421875:
                                                if Q.tau21 > 0.3868708908557892:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mean_eta2 > 0.0048917054664343596:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 37.38034248352051:
                                        if Q.tau32 > 0.47058549523353577:
                                            if Q.sum_pt_top5 > 306.125:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.3318641781806946:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 53% of the training jets here get this class from the formula
                            else:
                                return 't'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 5.999847412109375:
                                if Q.pt_7 > 44.109375:
                                    if Q.D2 > 0.9974052309989929:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.13710904866456985:
                                        if Q.sum_pt > 440.3203125:
                                            if Q.dr_7 > 0.1863854080438614:
                                                if Q.tau32 > 0.6034205555915833:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1178823933005333:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.008726037573069334:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.01249596057459712:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.eta_0 > -0.0381011962890625:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.03298013471066952:
                                    if Q.z_dr_0p05_0p1 > 0.12366056814789772:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.phi_7 > 0.01906585693359375:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.07836367562413216:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.lam2 > 0.001152106502559036:
                            if Q.sum_pt > 281.578125:
                                if Q.girth2 > 0.01761873997747898:
                                    return 't'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.mean_phi > 0.025056702084839344:
                                        return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0449055302888155:
                                            return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.08216642215847969:
                                    return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 14.76628828048706:
                                        if Q.z_7 > 0.09185677021741867:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0_0p05 > 0.2517380863428116:
                                return 't'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.mean_phi > -0.043449634686112404:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 52% of the training jets here get this class from the formula
                else:
                    if Q.C2 > 0.03447742573916912:
                        if Q.mass > 40.55801010131836:
                            if Q.centroid_offset > 0.022645432502031326:
                                if Q.eccentricity > 0.9376624524593353:
                                    if Q.C2 > 0.05206484906375408:
                                        return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.3454841077327728:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.m012 > 18.820838928222656:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.eta_0 > -0.042938232421875:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.09580238163471222:
                                    if Q.pt_6 > 29.765625:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 426.828125:
                                        if Q.D2 > 1.243140161037445:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_dispersion > 0.3739532232284546:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.2198135033249855:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.024356620386242867:
                                if Q.mass > 37.080055236816406:
                                    if Q.C2 > 0.04273252747952938:
                                        if Q.LHA > 0.33072349429130554:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 372.828125:
                            if Q.tau21 > 0.1506655141711235:
                                if Q.sum_pt > 438.890625:
                                    return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 409.046875:
                                    return 't'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0001644105213927105:
                                        return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.min_pair_mass > 0.7005419433116913:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.0004261750727891922:
                                return 't'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 78% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.866321563720703:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.02942987810820341:
                    if Q.max_dr > 0.13843219727277756:
                        if Q.C2 > 0.043965164572000504:
                            if Q.pt_7 > 34.484375:
                                if Q.D2 > 1.7609201073646545:
                                    if Q.pt_7 > 38.0625:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.452293395996094:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.35169997811317444:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.10381463170051575:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.007466801907867193:
                                        if Q.tau21 > 0.3517310470342636:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 40.9375:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_0 > 124.75:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 561.921875:
                                    if Q.lam2 > 0.0003905096818925813:
                                        if Q.log_sum_pt > 6.686182737350464:
                                            return 'g'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_dispersion > 0.5009471476078033:
                                                if Q.z_6 > 0.03264218755066395:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 68.29306411743164:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.189683675765991:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 45% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.238645076751709:
                                        if Q.planar_flow > 0.04595012404024601:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.9302090406417847:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04455028846859932:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 31.28125:
                                                if Q.z_dr_0p05_0p1 > 0.8231154084205627:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.07175498455762863:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 838.6875:
                                if Q.z_7 > 0.02944410778582096:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 40% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 45.265625:
                                    if Q.D2 > 1.1790621280670166:
                                        if Q.lam2 > 0.0002696206502150744:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00016327405319316313:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.02231156174093485:
                                            if Q.lam1 > 0.0075797513127326965:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.030566101893782616:
                                                    if Q.sum_pt_top2 > 274.71875:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 49.301225662231445:
                                                        if Q.pt_7 > 28.7109375:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 553.015625:
                                                if Q.pt_7 > 33.15625:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.03767408803105354:
                            if Q.C2 > 0.0398507472127676:
                                if Q.z_7 > 0.06370462477207184:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top2 > 0.007282446837052703:
                                        return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0001175215293187648:
                                    if Q.tau21 > 0.19792355597019196:
                                        if Q.z_7 > 0.07843811437487602:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.67430305480957:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 43.578125:
                                        if Q.width > 0.007740125292912126:
                                            return 'g'   # 42% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.043626585975289345:
                                            return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.006903115892782807:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.734375:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 32.421875:
                                if Q.z_dr_0p1_0p2 > 0.22377489507198334:
                                    if Q.lam2 > 0.00021562603797065094:
                                        if Q.tau21 > 0.2593148797750473:
                                            if Q.mass_top5 > 29.855191230773926:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.026378927752375603:
                                                if Q.centroid_offset > 0.032279014587402344:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.12000543624162674:
                                            if Q.tau21 > 0.1286659613251686:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.274055480957031:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 9.525681161903776e-05:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 35.701202392578125:
                                        if Q.lam2 > 0.0005675216089002788:
                                            if Q.tau21 > 0.17285025864839554:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 563.8828125:
                                    if Q.girth2 > 0.00838126428425312:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.03317568637430668:
                                            if Q.D2 > 0.684706300497055:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.20256023854017258:
                                        if Q.mass > 36.432762145996094:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.21452561765909195:
                        if Q.centroid_offset > 0.012098278384655714:
                            if Q.z_7 > 0.053809188306331635:
                                if Q.C2 > 0.04483569599688053:
                                    if Q.tau32 > 0.16759898513555527:
                                        if Q.pt_7 > 36.015625:
                                            if Q.girth2 > 0.007736645173281431:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01930237654596567:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_2 > 0.04107950069010258:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.07778185606002808:
                                                if Q.centroid_offset > 0.02665156777948141:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 494.953125:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0009723450639285147:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 46.578125:
                                        if Q.lam1 > 0.007715559098869562:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.670776128768921:
                                    if Q.sum_pt > 957.046875:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.2823161333799362:
                                            if Q.pt_7 > 25.8515625:
                                                if Q.centroid_offset > 0.01905393786728382:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.017373712733387947:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.008273776154965162:
                                                return 't'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.007653298554942012:
                                        if Q.e2 > 0.025608175434172153:
                                            if Q.sum_pt_top5 > 453.1875:
                                                if Q.dr_0 > 0.04536401852965355:
                                                    if Q.C2 > 0.050946176052093506:
                                                        if Q.z_7 > 0.044038115069270134:
                                                            return 'g'   # 36% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 46% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 54.53532791137695:
                                            if Q.max_dr > 0.25507429242134094:
                                                if Q.pt_6 > 36.046875:
                                                    return 't'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 469.625:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.07687368616461754:
                                                    return 't'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.008280066773295403:
                                if Q.D2 > 1.1550599336624146:
                                    if Q.z_7 > 0.053342895582318306:
                                        if Q.max_pair_mass > 10.928449153900146:
                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 833.421875:
                                            return 'g'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.665872573852539:
                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.538930892944336:
                                    if Q.dr_7 > 0.2870803773403168:
                                        if Q.mass_top3 > 3.0740818977355957:
                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.007242680760100484:
                                                return 't'   # 41% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1046.525390625:
                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 791.5859375:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 24.8515625:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 33.21875:
                                        if Q.girth2_top5 > 0.0008304086222779006:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 36% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 525.80859375:
                                            if Q.tau21 > 0.1425839588046074:
                                                if Q.max_dr > 0.26029855012893677:
                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.006919583771377802:
                            if Q.log_sum_pt > 6.234992742538452:
                                if Q.girth2 > 0.008627574425190687:
                                    if Q.e2 > 0.04468709044158459:
                                        if Q.pt_7 > 31.0859375:
                                            if Q.sum_pt > 899.890625:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 36.046875:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.11963195726275444:
                                                        if Q.mass > 64.3611068725586:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.1297374591231346:
                                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.008894748985767365:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.56365704536438:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.04725216515362263:
                                                    if Q.width > 0.008803610224276781:
                                                        if Q.tau32 > 0.5286003351211548:
                                                            if Q.max_dr > 0.11473605409264565:
                                                                return 't'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9944333434104919:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 65.80976867675781:
                                            if Q.centroid_offset > 0.0204566465690732:
                                                return 't'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.80115270614624:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.007490144344046712:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.018086143769323826:
                                                            return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 35.796875:
                                                if Q.centroid_offset > 0.010695830918848515:
                                                    if Q.e2 > 0.04267347417771816:
                                                        if Q.max_dr > 0.1408078670501709:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 54.7320499420166:
                                        if Q.sum_pt > 1089.703125:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.022562747821211815:
                                                if Q.max_dr > 0.15708794444799423:
                                                    if Q.width > 0.008174298331141472:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.18007688969373703:
                                                    if Q.centroid_offset > 0.017351622693240643:
                                                        if Q.width > 0.007641542237251997:
                                                            if Q.eccentricity > 0.9877479374408722:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.00824839947745204:
                                                            if Q.log_sum_pt > 6.636916399002075:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.04560248740017414:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 699.53125:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_6 > 0.047632791101932526:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.007017125142738223:
                                                        if Q.sum_pt > 712.921875:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 23.453125:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_7 > 0.0993070900440216:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 703.546875:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top2 > 0.006366674788296223:
                                                                if Q.centroid_offset > 0.008604976814240217:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_2 > 0.08086156472563744:
                                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_dr_0p05_0p1 > 0.6091111302375793:
                                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.765625:
                                            if Q.girth2 > 0.007083090487867594:
                                                if Q.max_dr > 0.16525571048259735:
                                                    if Q.centroid_offset > 0.019209887832403183:
                                                        if Q.tau21 > 0.1203274242579937:
                                                            if Q.dr_0 > 0.058494605123996735:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 36.046875:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.00799271883442998:
                                                            if Q.z_7 > 0.061513615772128105:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 30.5703125:
                                                        if Q.centroid_offset > 0.025478661060333252:
                                                            if Q.girth2 > 0.008250124752521515:
                                                                if Q.max_dr > 0.11718139797449112:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0004326850612415001:
                                                                    if Q.tau21 > 0.12037695944309235:
                                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.007216220255941153:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.5887821316719055:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 7.167207877500914e-05:
                                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_6 > 48.28125:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.10741309076547623:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_6 > 0.12057231739163399:
                                                                return 't'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.331821441650391:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.012137142475694418:
                                                    if Q.z_dr_0p1_0p2 > 0.30503493547439575:
                                                        if Q.tau21 > 0.21411245316267014:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.48225177824497223:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.05560816265642643:
                                                        if Q.dr01 > 0.17038418352603912:
                                                            if Q.lam2 > 0.00023536907247034833:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.0069819537457078695:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.005443862173706293:
                                                            if Q.max_dr > 0.10234679281711578:
                                                                if Q.z_dr_0p05_0p1 > 0.6572526395320892:
                                                                    if Q.girth2 > 0.0069874615874141455:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.2314222902059555:
                                                                if Q.z_dr_0p05_0p1 > 0.5761633217334747:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.0859375:
                                                if Q.dr_0 > 0.06584720686078072:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.7700962126255035:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2845573127269745:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 30.2890625:
                                    if Q.log_sum_pt > 5.9742467403411865:
                                        if Q.max_dr > 0.15658565610647202:
                                            if Q.tau21 > 0.18741337954998016:
                                                if Q.centroid_offset > 0.017496381886303425:
                                                    if Q.z_6 > 0.07883523404598236:
                                                        if Q.e2_sq > 0.007101994240656495:
                                                            if Q.tau32 > 0.4198751002550125:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.19139296561479568:
                                                        if Q.pt_7 > 37.03125:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.015654597897082567:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.02552871685475111:
                                                        if Q.z_dr_0_0p05 > 0.4532759189605713:
                                                            return 't'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.12865083664655685:
                                                if Q.planar_flow > 0.8637567460536957:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.025760643184185028:
                                                        if Q.tau21 > 0.2051049843430519:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_5 > 44.15625:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.11598388478159904:
                                                    if Q.pt_7 > 35.640625:
                                                        if Q.girth2 > 0.00875116977840662:
                                                            if Q.pt_7 > 40.6875:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.48761458694934845:
                                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.00844165962189436:
                                                            return 't'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.04239262081682682:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.14347121119499207:
                                            if Q.max_dr > 0.12332425266504288:
                                                if Q.girth2 > 0.00833932077512145:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.2623656392097473:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.23276959359645844:
                                        if Q.log_sum_pt > 6.126220941543579:
                                            if Q.pt_7 > 28.75:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.0399557948112488:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 33.703125:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 5.998528718948364:
                                            if Q.centroid_offset > 0.009306508582085371:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005537797696888447:
                                                    return 'Z'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.0398881770670414:
                                if Q.lam2 > 0.00026015067123807967:
                                    if Q.mass > 32.66198921203613:
                                        if Q.D2 > 0.9278418719768524:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.5314057767391205:
                                                if Q.max_dr > 0.09921126440167427:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006720425561070442:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.0065263870637863874:
                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 64.25715637207031:
                                        if Q.width > 0.006766056874766946:
                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.7488694190979:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.006615940481424332:
                                                    if Q.pt_7 > 38.765625:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.012282670009881258:
                                            if Q.log_sum_pt > 6.4912331104278564:
                                                if Q.z_dr_0_0p05 > 0.043910274282097816:
                                                    if Q.pt1_dr01 > 15.90766954421997:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006719052791595459:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.593725681304932:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.8397247791290283:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006836265325546265:
                                                        if Q.e2 > 0.04131337068974972:
                                                            if Q.lam2 > 6.911965829203837e-05:
                                                                if Q.e2 > 0.042860452085733414:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.005732079967856407:
                                                            if Q.planar_flow > 0.08011530712246895:
                                                                if Q.tau21 > 0.12698138505220413:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.004998168908059597:
                                                if Q.width > 0.006801884388551116:
                                                    if Q.mass > 60.1185188293457:
                                                        if Q.z_dr_0_0p05 > 0.0425145048648119:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00017180623399326578:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 435.15625:
                                                                if Q.max_dr > 0.10858844965696335:
                                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0067581485491245985:
                                                    if Q.sum_pt > 670.390625:
                                                        if Q.n_dr_0_0p05 > 0.5:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9906964004039764:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_2 > 0.06866375729441643:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.4477763175964355:
                                    if Q.e2 > 0.03793034702539444:
                                        if Q.z_dr_0p05_0p1 > 0.48484279215335846:
                                            if Q.log_sum_pt > 6.574683427810669:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.014214854687452316:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006736683659255505:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.9192017614841461:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 64.49981307983398:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.004725532839074731:
                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.14983227103948593:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07229997217655182:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 29.7265625:
                                        if Q.z_dr_0p1_0p2 > 0.24986866116523743:
                                            if Q.z_dr_0p05_0p1 > 0.7086266577243805:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.013791298493742943:
                                                    return 't'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.008755779825150967:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.20068155974149704:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 27.3046875:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1473577842116356:
                    if Q.girth2 > 0.004858389729633927:
                        if Q.centroid_offset > 0.011311578564345837:
                            if Q.centroid_offset > 0.03229784406721592:
                                if Q.max_dr > 0.20233392715454102:
                                    if Q.sum_pt_top5 > 553.8828125:
                                        if Q.pt_7 > 31.15625:
                                            if Q.sum_pt > 767.359375:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9913395047187805:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0008259757596533746:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.04869711212813854:
                                            if Q.pt_7 > 34.46875:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04100470431149006:
                                        if Q.sum_pt_top5 > 552.8125:
                                            if Q.pt_7 > 41.1875:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.04708391800522804:
                                                    if Q.pt_7 > 27.1640625:
                                                        return 'g'   # 32% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.00559942820109427:
                                                if Q.mass_top3 > 3.445390820503235:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.05349688045680523:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_6 > 28.234375:
                                            if Q.lam2 > 0.0004038683546241373:
                                                if Q.tau21 > 0.21646541357040405:
                                                    if Q.pt_6 > 44.65625:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.00597226619720459:
                                                    if Q.sum_pt > 658.71875:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.2473883107304573:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 552.875:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.22374576330184937:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 541.125:
                                    if Q.max_dr > 0.28487783670425415:
                                        if Q.log_sum_pt > 6.606671094894409:
                                            if Q.centroid_offset > 0.019150510430336:
                                                if Q.sum_pt > 965.828125:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.828125:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3133646994829178:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.29929184913635254:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.051162442192435265:
                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.014910838566720486:
                                            if Q.max_dr > 0.22705081850290298:
                                                if Q.centroid_offset > 0.025102252140641212:
                                                    if Q.sum_pt_top5 > 573.9375:
                                                        if Q.pt_7 > 28.46875:
                                                            if Q.pt_7 > 35.875:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 702.5859375:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.8396660089492798:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 2.4997291564941406:
                                                                return 'g'   # 39% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 718.734375:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 26.7890625:
                                                            if Q.D2 > 1.9289621710777283:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_5 > 47.8125:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 645.15625:
                                                    if Q.lam1 > 0.005022814963012934:
                                                        if Q.lam2 > 0.00011024086779798381:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.015703161247074604:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.005269072717055678:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2_top2 > 0.0018933851388283074:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016633332706987858:
                                                            if Q.max_dr > 0.15597443282604218:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02014083694666624:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.26933756470680237:
                                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.16598248481750488:
                                                                if Q.e2 > 0.025425143539905548:
                                                                    if Q.lam2 > 0.00026757029991131276:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 25.921875:
                                                        if Q.z_dr_0p05_0p1 > 0.06345182284712791:
                                                            if Q.max_dr > 0.1590978130698204:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.27383820712566376:
                                                                    if Q.width > 0.005631347419694066:
                                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.02081983909010887:
                                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.021015978418290615:
                                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.1044428944587708:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.8679251968860626:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 35.984375:
                                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 24.0390625:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.17018230259418488:
                                                if Q.width > 0.005180988227948546:
                                                    if Q.mass > 48.75456237792969:
                                                        if Q.mass > 75.41369247436523:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 752.234375:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 24.1796875:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 31.6640625:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_top3 > 2.717851400375366:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.10346611961722374:
                                                        if Q.sum_pt_top5 > 670.203125:
                                                            if Q.girth2_top2 > 0.0013920654309913516:
                                                                if Q.centroid_offset > 0.013045981526374817:
                                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.4628456830978394:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005634835222736001:
                                                    if Q.mass > 53.71748161315918:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.005931137362495065:
                                                            if Q.z_dr_0p05_0p1 > 0.1604386866092682:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_top3 > 2.25276517868042:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.005398809676989913:
                                                        if Q.centroid_offset > 0.013040916994214058:
                                                            if Q.max_dr > 0.1578664556145668:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 756.84375:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 63.285865783691406:
                                                            if Q.z_top5 > 0.8905142843723297:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 32.71875:
                                        if Q.tau21 > 0.20449002087116241:
                                            if Q.LHA > 0.23892412334680557:
                                                if Q.max_dr > 0.15729912370443344:
                                                    if Q.log_sum_pt > 6.1495361328125:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.47989653050899506:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.06670721620321274:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.018010984174907207:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.19008994102478:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 41.40625:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.08188417926430702:
                                                    if Q.min_pair_mass > 0.5930780470371246:
                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.33383166790008545:
                                            if Q.pt_6 > 33.4375:
                                                if Q.z_dr_0_0p05 > 0.6550456881523132:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.140742540359497:
                                                if Q.tau21 > 0.22065837681293488:
                                                    if Q.pt_7 > 28.171875:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.17949608713388443:
                                if Q.girth2 > 0.005485784262418747:
                                    if Q.width > 0.0058292909525334835:
                                        if Q.mass > 58.22811508178711:
                                            if Q.centroid_offset > 0.0018689993303269148:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.005981513299047947:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 24.3984375:
                                                if Q.e2 > 0.02384741697460413:
                                                    if Q.log_sum_pt > 6.2353293895721436:
                                                        if Q.C2 > 0.03860917128622532:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.006052273558452725:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.027557539753615856:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 39.203125:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 681.3203125:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.20660477131605148:
                                            if Q.centroid_offset > 0.004444202408194542:
                                                if Q.log_sum_pt > 6.616870164871216:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 28.34375:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0_0p05 > 0.9145703911781311:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 7.070908759487793e-05:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi2 > 0.002937886514700949:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.008332975674420595:
                                                if Q.mass > 56.07004165649414:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.5382882356643677:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 45% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9570913016796112:
                                                    if Q.mass > 67.9097785949707:
                                                        if Q.pt_7 > 15.78515625:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.006211145780980587:
                                                            if Q.pt1_dr01 > 0.5564241111278534:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.5421990156173706:
                                        if Q.max_dr > 0.24637088179588318:
                                            if Q.mass > 54.43105697631836:
                                                if Q.centroid_offset > 0.003842703881673515:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.2852422744035721:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 26.921875:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.20918067544698715:
                                                if Q.D2 > 1.971644401550293:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.005125052062794566:
                                                        if Q.centroid_offset > 0.007453353377059102:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.005065884906798601:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top5 > 0.0051622651517391205:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.009030353277921677:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.22558725625276566:
                                            if Q.centroid_offset > 0.008103810250759125:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.903311491012573:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.005297389114275575:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 25.703125:
                                                            if Q.sum_pt_top5 > 706.640625:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.912812948226929:
                                                if Q.z_7 > 0.018781157210469246:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0099331671372056:
                                                    if Q.lam1 > 0.005120672285556793:
                                                        if Q.z_dr_0p1_0p2 > 0.11272498592734337:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.005362190771847963:
                                                        if Q.e2 > 0.024604649282991886:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.00601978600025177:
                                    if Q.centroid_offset > 0.006447848631069064:
                                        if Q.mass > 62.90322303771973:
                                            if Q.mass_over_sum_pt > 0.0787464864552021:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03293488919734955:
                                                    if Q.z_dr_0p05_0p1 > 0.1579560935497284:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.18532004952430725:
                                                if Q.z_dr_0p05_0p1 > 0.10919735580682755:
                                                    if Q.width > 0.006299575790762901:
                                                        if Q.girth2_top3 > 0.006525650154799223:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.008913753554224968:
                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.006127439672127366:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1669931560754776:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.16700749844312668:
                                            if Q.girth > 0.061394261196255684:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 69.91641616821289:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.03025589045137167:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 73.78035736083984:
                                                if Q.e2 > 0.03454837203025818:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03982306644320488:
                                                    if Q.eccentricity > 0.9653524160385132:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006381600396707654:
                                                        if Q.mass > 63.56193923950195:
                                                            if Q.z_dr_0p1_0p2 > 0.21709661185741425:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 73.16212463378906:
                                        if Q.pt_7 > 19.640625:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 77.89127731323242:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0003078086447203532:
                                            if Q.girth2 > 0.005729888333007693:
                                                if Q.max_dr > 0.15762921422719955:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1712310090661049:
                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005875503644347191:
                                                if Q.centroid_offset > 0.007415890460833907:
                                                    if Q.e2 > 0.03072729893028736:
                                                        if Q.sum_pt_top3 > 489.65625:
                                                            if Q.C2 > 0.019980876706540585:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1700378879904747:
                                                    if Q.centroid_offset > 0.00979720102623105:
                                                        if Q.width > 0.0055610788986086845:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 69.0274772644043:
                                                            if Q.sum_pt_top5 > 861.421875:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.01820436865091324:
                            if Q.width > 0.0035425531677901745:
                                if Q.max_dr > 0.17066329717636108:
                                    if Q.centroid_offset > 0.037249114364385605:
                                        if Q.z_dr_0p1_0p2 > 0.03953494317829609:
                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 29.8125:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 656.40625:
                                            if Q.z_dr_0p1_0p2 > 0.06051413528621197:
                                                if Q.girth > 0.04970710724592209:
                                                    if Q.centroid_offset > 0.02011914551258087:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.06218465603888035:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.17803658545017242:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.003832402639091015:
                                                        if Q.centroid_offset > 0.019183357246220112:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02367039956152439:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.886394739151001:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 4.2196738719940186:
                                                    if Q.centroid_offset > 0.03241650573909283:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 733.609375:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1031.5546875:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 15.60546875:
                                                            if Q.girth2_top5 > 0.0008956646197475493:
                                                                if Q.tau21 > 0.6430593430995941:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.0032490965677425265:
                                                                        if Q.pt_4 > 30.7578125:
                                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.02783862967044115:
                                                                                return 'q'   # 48% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 27.0546875:
                                                if Q.LHA > 0.24284017831087112:
                                                    if Q.centroid_offset > 0.022428419440984726:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.06302081048488617:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.19144833087921143:
                                                        if Q.LHA > 0.21934060007333755:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.15317843109369278:
                                                            if Q.centroid_offset > 0.021830346435308456:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p05_0p1 > 1.5:
                                                    if Q.sum_pt > 612.0859375:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.025469444692134857:
                                        if Q.LHA > 0.25362761318683624:
                                            if Q.lam1 > 0.0036423966521397233:
                                                if Q.max_dr > 0.1530841886997223:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.2584628760814667:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 49% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0008313026919495314:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.01766914501786232:
                                                        if Q.lam2 > 0.0002929181937361136:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 550.953125:
                                                if Q.max_dr > 0.15522608906030655:
                                                    if Q.dr_0 > 0.044727565720677376:
                                                        if Q.e2 > 0.017467843368649483:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9688514173030853:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_6 > 43.140625:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 35% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.00439949007704854:
                                            if Q.max_dr > 0.15767449885606766:
                                                if Q.e2 > 0.029116653837263584:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0201207147911191:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 46.31736755371094:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.021473548375070095:
                                                    if Q.mass > 42.4554386138916:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.022682280279695988:
                                                if Q.mass > 41.23322296142578:
                                                    if Q.max_dr > 0.15955259650945663:
                                                        if Q.centroid_offset > 0.02344781532883644:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 46.36042785644531:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 52.55367851257324:
                                                    if Q.centroid_offset > 0.019960147328674793:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.16103409230709076:
                                                        if Q.girth > 0.05282606743276119:
                                                            if Q.mean_eta2 > 0.0015936200506985188:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19975372403860092:
                                    if Q.width > 0.0024864786537364125:
                                        if Q.centroid_offset > 0.021510161459445953:
                                            if Q.pt_5 > 21.6953125:
                                                if Q.mass > 34.34202003479004:
                                                    if Q.sum_pt_top5 > 881.5:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 38.49407386779785:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.022694945335388184:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.0509672649204731:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 22.3828125:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 5.240069150924683:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.22696180641651154:
                                                if Q.girth2 > 0.002849882934242487:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.24660912156105042:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.019403242506086826:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0030986348865553737:
                                                    if Q.max_dr > 0.20962543040513992:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.9262195825576782:
                                                            if Q.mass > 40.04319953918457:
                                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.028877507895231247:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 44.83325958251953:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.021888066083192825:
                                            if Q.z_top5 > 0.933080404996872:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 5.071772575378418:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.26115885376930237:
                                                if Q.centroid_offset > 0.01959015615284443:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.027861222624778748:
                                        if Q.centroid_offset > 0.0319047924131155:
                                            if Q.pt_7 > 21.8359375:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.17699996381998062:
                                                if Q.girth > 0.04422583431005478:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.18892348557710648:
                                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9466891288757324:
                                                    if Q.mass > 38.0432014465332:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 34.503530502319336:
                                                            if Q.centroid_offset > 0.029609511606395245:
                                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.024107802659273148:
                                            if Q.mass > 40.63101577758789:
                                                if Q.lam2 > 5.8585917940945365e-05:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.18056288361549377:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.9316775500774384:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00011357102266629227:
                                                    if Q.mass > 36.54315376281738:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.3187865614891052:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 50.57655906677246:
                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_5 > 22.015625:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 5.1930763220298104e-05:
                                                        if Q.girth2 > 0.0028704331489279866:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0017601520521566272:
                                if Q.max_dr > 0.22487571090459824:
                                    if Q.girth2 > 0.0035206981701776385:
                                        if Q.centroid_offset > 0.012668156065046787:
                                            if Q.mass > 45.49706840515137:
                                                if Q.girth2 > 0.003984889248386025:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.2402154728770256:
                                                        if Q.centroid_offset > 0.01412623981013894:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.26738105714321136:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.014597342349588871:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.2131180688738823:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.24215605854988098:
                                                        if Q.LHA > 0.20108363032341003:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 38% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.29023729264736176:
                                                if Q.width > 0.003993304446339607:
                                                    if Q.centroid_offset > 0.005777026759460568:
                                                        if Q.sum_pt_top5 > 636.65625:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3266126364469528:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009721499402076006:
                                                        if Q.D2 > 3.920505404472351:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 5.044949054718018:
                                                            if Q.girth2_top2 > 0.00030090131622273475:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 56.59893226623535:
                                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.004284910392016172:
                                                    if Q.centroid_offset > 0.00857596192508936:
                                                        if Q.max_dr > 0.23703954368829727:
                                                            if Q.girth2 > 0.004593112971633673:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.05535003915429115:
                                                                    if Q.centroid_offset > 0.010663768276572227:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.04801100306212902:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.055859100073575974:
                                                            if Q.centroid_offset > 0.005521741695702076:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.0542441513389349:
                                                        if Q.centroid_offset > 0.010415463708341122:
                                                            if Q.lam1 > 0.003808974171988666:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 64.11368179321289:
                                                            if Q.pt_5 > 30.0234375:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.01511643873527646:
                                            if Q.girth2 > 0.0026600839337334037:
                                                if Q.max_dr > 0.2549952417612076:
                                                    if Q.girth2 > 0.00308197527192533:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 4.9719624519348145:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 1.6715523997845594e-05:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.003281182376667857:
                                                        if Q.centroid_offset > 0.016067102551460266:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.02702033333480358:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2834015041589737:
                                                    if Q.lam1 > 0.0022671513725072145:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.01822596602141857:
                                                if Q.mass > 39.00785255432129:
                                                    if Q.log_sum_pt > 7.0079505443573:
                                                        if Q.pt_7 > 17.1015625:
                                                            if Q.mass > 61.92066955566406:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3313913643360138:
                                                            if Q.girth > 0.02624465338885784:
                                                                if Q.pt_7 > 18.71875:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.36007367074489594:
                                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 41.463584899902344:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013448677957057953:
                                                                if Q.mass > 48.00473213195801:
                                                                    if Q.C2 > 0.05543194338679314:
                                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_6 > 10.91015625:
                                                                    if Q.lam2 > 0.00010364035188104026:
                                                                        if Q.girth2 > 0.0033344728872179985:
                                                                            if Q.centroid_offset > 0.011065852828323841:
                                                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 46.94162940979004:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_6 > 28.5859375:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.738170623779297:
                                                            if Q.centroid_offset > 0.010602387599647045:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 638.1171875:
                                                                if Q.pt_5 > 24.8046875:
                                                                    if Q.girth2_top5 > 0.000349124675267376:
                                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 47% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 41.673696517944336:
                                                    if Q.z_6 > 0.0130095100030303:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 45.65154266357422:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.051919180899858475:
                                        if Q.LHA > 0.22194450348615646:
                                            if Q.centroid_offset > 0.01274554105475545:
                                                if Q.centroid_offset > 0.01429416798055172:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_7 > 0.034682001918554306:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00021159028256079182:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.010445076040923595:
                                                        if Q.girth2 > 0.004647297086194158:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00015089927182998508:
                                                if Q.girth > 0.04087335057556629:
                                                    if Q.z_dr_0_0p05 > 0.8906897902488708:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.015278616454452276:
                                                    if Q.lam1 > 0.003526726155541837:
                                                        if Q.D2 > 2.2366058826446533:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1021.90625:
                                                        if Q.pt_7 > 22.5078125:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 34.33606147766113:
                                            if Q.mass > 67.64015197753906:
                                                if Q.log_sum_pt > 6.985381603240967:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006852707825601101:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.002281446009874344:
                                                    if Q.centroid_offset > 0.01558465650305152:
                                                        if Q.width > 0.004337467486038804:
                                                            if Q.max_dr > 0.1704552248120308:
                                                                if Q.mass > 49.6636905670166:
                                                                    if Q.centroid_offset > 0.016531387344002724:
                                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 0.004576458595693111:
                                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.09227029606699944:
                                                                        if Q.z_dr_0_0p05 > 0.8783126771450043:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 57.13878631591797:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.003926175180822611:
                                                                    if Q.max_dr > 0.19108950346708298:
                                                                        if Q.pt_2 > 88.5625:
                                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.17651591449975967:
                                                                            if Q.width > 0.004163059173151851:
                                                                                if Q.centroid_offset > 0.016953430138528347:
                                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00044509755389299244:
                                                            if Q.max_dr > 0.1755855455994606:
                                                                if Q.width > 0.0040454380214214325:
                                                                    if Q.z_dr_0p05_0p1 > 0.1890433430671692:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.width > 0.004493636079132557:
                                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 40.1711368560791:
                                                                if Q.sum_pt > 1076.6015625:
                                                                    if Q.z_7 > 0.03357964754104614:
                                                                        return 'g'   # 36% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.013235478661954403:
                                                                        if Q.mass > 62.444589614868164:
                                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.mass_over_sum_pt > 0.06554552167654037:
                                                                                if Q.max_dr > 0.18481923639774323:
                                                                                    if Q.z_dr_0p1_0p2 > 0.08626345545053482:
                                                                                        if Q.z_dr_0p1_0p2 > 0.10029556602239609:
                                                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 25.7890625:
                                                                    if Q.centroid_offset > 0.006913520861417055:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_6 > 0.052361926063895226:
                                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.889095962047577:
                                                                        if Q.centroid_offset > 0.009446080308407545:
                                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 35% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top3 > 500.5625:
                                                                            return 'q'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009567680303007364:
                                                        if Q.pt_7 > 43.40625:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013711617793887854:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.1851092204451561:
                                                                    if Q.mass > 38.54176330566406:
                                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 32.6875:
                                                            if Q.pt_7 > 40.625:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 42.5814151763916:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 3.220170021057129:
                                                                    if Q.mass > 38.8725700378418:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010530794970691204:
                                                if Q.pt_7 > 22.9375:
                                                    if Q.pt_4 > 37.140625:
                                                        if Q.girth2_top2 > 0.0003381841233931482:
                                                            if Q.e2_sq > 0.0038036376936361194:
                                                                if Q.max_dr > 0.1718355268239975:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.0017587910406291485:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.003034002147614956:
                                                    if Q.pt_6 > 38.25:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04065621457993984:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.015613558702170849:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.009987178724259138:
                                    if Q.pt_6 > 16.5:
                                        if Q.log_sum_pt > 6.72174072265625:
                                            if Q.pt_7 > 36.28125:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.012987683527171612:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 34.84408187866211:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 21.5:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.9680446982383728:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 916.37109375:
                                            if Q.z_dr_0p2_0p4 > 0.011540131643414497:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 41.979806900024414:
                                        if Q.girth2_top3 > 0.00010904869850492105:
                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 22.6171875:
                                            if Q.sum_pt > 976.0078125:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.0026776777813211083:
                        if Q.centroid_offset > 0.024785758927464485:
                            if Q.width > 0.00483433622866869:
                                if Q.max_dr > 0.11201531067490578:
                                    if Q.z_dr_0p05_0p1 > 0.3536168187856674:
                                        if Q.centroid_offset > 0.041160840541124344:
                                            if Q.girth2 > 0.005833829287439585:
                                                if Q.pt_7 > 34.171875:
                                                    if Q.e2 > 0.0243621077388525:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 30.25:
                                                    if Q.lam2 > 0.00021965344058116898:
                                                        return 'g'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 44.145111083984375:
                                                if Q.max_dr > 0.12119575217366219:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.07195981964468956:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02793126180768013:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 48.10567283630371:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1268514320254326:
                                                    if Q.pt_6 > 27.359375:
                                                        if Q.sum_pt > 522.3125:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 32.609375:
                                                                if Q.z_dr_0_0p05 > 0.23097550123929977:
                                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0296576377004385:
                                                        if Q.z_dr_0_0p05 > 0.13244304805994034:
                                                            if Q.centroid_offset > 0.033900290727615356:
                                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_6 > 33.140625:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.07425864040851593:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.05912608839571476:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.11716947704553604:
                                                                    if Q.centroid_offset > 0.02757427841424942:
                                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.7879356443881989:
                                            if Q.eccentricity > 0.8698162138462067:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03533809632062912:
                                        if Q.pt_7 > 33.828125:
                                            if Q.lam2 > 0.000663065817207098:
                                                if Q.tau21 > 0.21042588353157043:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 36.89175224304199:
                                                    if Q.centroid_offset > 0.04470590315759182:
                                                        return 't'   # 39% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0405141431838274:
                                                        if Q.girth > 0.07474247366189957:
                                                            return 't'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2948293685913086:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0001684262097114697:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.3055216521024704:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 35.618080139160156:
                                                    if Q.planar_flow > 0.06482390686869621:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6024618446826935:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 42% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.3113904744386673:
                                            if Q.dr_0 > 0.07907762378454208:
                                                if Q.girth > 0.07992677018046379:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9618653357028961:
                                                if Q.mass > 51.28518295288086:
                                                    if Q.centroid_offset > 0.027142071165144444:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.06768304109573364:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.030849000439047813:
                                                        if Q.mass > 45.042030334472656:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 30.6328125:
                                                            if Q.z_dr_0p1_0p2 > 0.21241766214370728:
                                                                return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 48.05388069152832:
                                                                    if Q.width > 0.005599841941148043:
                                                                        if Q.e2 > 0.034129390493035316:
                                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.02189022209495306:
                                                    if Q.pt_7 > 28.6875:
                                                        if Q.centroid_offset > 0.028976325877010822:
                                                            if Q.mass > 36.722917556762695:
                                                                if Q.girth > 0.07289372012019157:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1038982942700386:
                                                                if Q.girth > 0.07325557619333267:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 31% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.294716835021973:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03221680968999863:
                                    if Q.max_dr > 0.11575163155794144:
                                        if Q.LHA > 0.2634444683790207:
                                            if Q.pt_7 > 24.4921875:
                                                if Q.max_dr > 0.12832684814929962:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.06824048236012459:
                                                        if Q.centroid_offset > 0.033800218254327774:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 39% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9875820875167847:
                                                if Q.centroid_offset > 0.03484182991087437:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.31551244854927063:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.11366824433207512:
                                            if Q.sum_pt > 782.25:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.19540882855653763:
                                                    if Q.centroid_offset > 0.03788859769701958:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 464.71875:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 696.84375:
                                                if Q.max_pair_mass > 1.5720650553703308:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.min_pair_mass > 0.38600651919841766:
                                                        return 'W'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 42.372467041015625:
                                        if Q.max_dr > 0.12172861024737358:
                                            if Q.centroid_offset > 0.02765305619686842:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 48.02740287780762:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0045117661356925964:
                                                        if Q.min_pair_mass > 0.5428805351257324:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13540297746658325:
                                                            if Q.dr_2 > 0.04811353422701359:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 8.198694922612049e-05:
                                                if Q.C2 > 0.02211423870176077:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00017636422126088291:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00014318520697997883:
                                            if Q.tau21 > 0.1927974671125412:
                                                if Q.max_dr > 0.1318374201655388:
                                                    if Q.centroid_offset > 0.028333362191915512:
                                                        if Q.pt_7 > 32.296875:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top5 > 0.856317549943924:
                                                        if Q.centroid_offset > 0.027575124986469746:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.026606165803968906:
                                                    if Q.max_dr > 0.10680790990591049:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 712.3828125:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.13327942043542862:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.9276996552944183:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.13234799355268478:
                                                if Q.LHA > 0.26669231057167053:
                                                    if Q.centroid_offset > 0.027707193978130817:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.028898307122290134:
                                                        if Q.mass > 38.447181701660156:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006397217977792025:
                                if Q.e2 > 0.03773641400039196:
                                    if Q.lam2 > 0.00027055240934714675:
                                        if Q.D2 > 0.9548735916614532:
                                            if Q.dr_0 > 0.08511956036090851:
                                                if Q.z_6 > 0.08450690284371376:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.041699472814798355:
                                                if Q.width > 0.0065585949923843145:
                                                    if Q.girth2_top2 > 0.006319267675280571:
                                                        if Q.max_dr > 0.10836191475391388:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.11001204326748848:
                                                        if Q.z_dr_0_0p05 > 0.06225714273750782:
                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.3028091937303543:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.min_pair_mass > 0.8434911370277405:
                                                        return 'Z'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 73.01411437988281:
                                            if Q.pt_7 > 25.5625:
                                                if Q.mass > 75.9472885131836:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 35.734375:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.006547899451106787:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_6 > 0.09002469480037689:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03979112394154072:
                                                if Q.max_dr > 0.1309201866388321:
                                                    if Q.n_dr_0_0p05 > 0.5:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006563725881278515:
                                                        if Q.sum_pt_top5 > 626.671875:
                                                            if Q.dr01 > 0.15546928346157074:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.10791115090250969:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_4 > 0.08794290199875832:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.012394126038998365:
                                                    if Q.width > 0.006581456400454044:
                                                        if Q.girth2_top5 > 0.006506673293188214:
                                                            if Q.width > 0.0066213482059538364:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 395.75:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.08699452877044678:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_6 > 0.08606851473450661:
                                                                    if Q.log_sum_pt > 6.530307769775391:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006550839403644204:
                                                        if Q.z_dr_0p05_0p1 > 0.6436180174350739:
                                                            if Q.log_sum_pt > 6.693352937698364:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1255217120051384:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 70.74878311157227:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 420.5625:
                                                                if Q.D2 > 0.3540823757648468:
                                                                    if Q.girth2_top2 > 0.004581630229949951:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.4694657474756241:
                                        if Q.e2 > 0.03688640519976616:
                                            if Q.girth2 > 0.006440700497478247:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0063720205798745155:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005075494293123484:
                                            if Q.sum_pt > 712.09375:
                                                if Q.z_dr_0_0p05 > 0.4777771830558777:
                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 35.37722587585449:
                                    if Q.centroid_offset > 0.017157841473817825:
                                        if Q.mass > 52.82663536071777:
                                            if Q.max_dr > 0.11703246459364891:
                                                if Q.girth2 > 0.005168081261217594:
                                                    if Q.lam1 > 0.005483994958922267:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1354226991534233:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.018307295627892017:
                                                                if Q.mean_phi2 > 0.0007775897101964802:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02177246194332838:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.019776610657572746:
                                                            if Q.mass_over_sum_pt > 0.06635475531220436:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 61.337738037109375:
                                                    if Q.lam1 > 0.0057178467977792025:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02182544581592083:
                                                        if Q.max_dr > 0.10977231711149216:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 9.278888319386169e-05:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00021526339696720243:
                                                            if Q.tau21 > 0.12887639552354813:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.005997818196192384:
                                                                if Q.e2 > 0.03631543554365635:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005276516079902649:
                                                if Q.max_dr > 0.12323670461773872:
                                                    if Q.n_dr_0_0p05 > 1.5:
                                                        if Q.max_dr > 0.1379571482539177:
                                                            if Q.centroid_offset > 0.020456325262784958:
                                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.15495503693819046:
                                                                    if Q.girth > 0.06673649698495865:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.13818058371543884:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p1_0p2 > 0.2690366357564926:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.12997262179851532:
                                                            if Q.z_dr_0p1_0p2 > 0.08256244286894798:
                                                                if Q.centroid_offset > 0.019353095442056656:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.13435974717140198:
                                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.0060694164130836725:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.06697442755103111:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.15871404111385345:
                                                                        if Q.centroid_offset > 0.019602874293923378:
                                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 43% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p1_0p2 > 1.5:
                                                        if Q.girth > 0.07528574764728546:
                                                            if Q.centroid_offset > 0.019216308370232582:
                                                                if Q.mass > 48.86878776550293:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_3 > 0.07242852076888084:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.022530346177518368:
                                                                if Q.max_dr > 0.11933818832039833:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 28.6171875:
                                                            if Q.lam2 > 0.0001879229093901813:
                                                                if Q.tau21 > 0.12945368140935898:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 0.0003473648976068944:
                                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02263699285686016:
                                                    if Q.max_dr > 0.13118073344230652:
                                                        if Q.mass_over_sum_pt > 0.06243531405925751:
                                                            if Q.mass > 42.90348434448242:
                                                                if Q.mass_over_sum_pt_sq > 0.004399623489007354:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top3 > 464.984375:
                                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.004883011803030968:
                                                        if Q.max_dr > 0.13645249605178833:
                                                            if Q.z_top5 > 0.8355934023857117:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 76.24139022827148:
                                            if Q.pt_7 > 25.0:
                                                if Q.mass > 85.30367660522461:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1025.96875:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 36.21875:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top2 > 0.005730783799663186:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.942054748535156:
                                                    if Q.pt_7 > 14.94921875:
                                                        if Q.mass > 79.74367141723633:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0032721051247790456:
                                                if Q.max_dr > 0.127171091735363:
                                                    if Q.z_dr_0p05_0p1 > 0.655394434928894:
                                                        if Q.girth2 > 0.005896579008549452:
                                                            if Q.e2 > 0.03527330234646797:
                                                                if Q.girth2 > 0.006269665900617838:
                                                                    if Q.log_sum_pt > 6.6018335819244385:
                                                                        if Q.centroid_offset > 0.0076402665581554174:
                                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.13435956835746765:
                                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.00867800461128354:
                                                                        if Q.mass > 58.409873962402344:
                                                                            if Q.girth2 > 0.006143394159153104:
                                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.011096890550106764:
                                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2_sq > 0.005989609751850367:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013191836886107922:
                                                                if Q.girth2 > 0.005541075021028519:
                                                                    if Q.mass > 54.33717727661133:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.06839723885059357:
                                                            if Q.centroid_offset > 0.009823569096624851:
                                                                if Q.max_dr > 0.13585255295038223:
                                                                    if Q.mass > 53.39711952209473:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.014459081925451756:
                                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.9791014194488525:
                                                                if Q.dr_0 > 0.034601807594299316:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.39374440908432007:
                                                                    if Q.sum_pt > 991.078125:
                                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.girth2_top2 > 0.0010164623381569982:
                                                                            if Q.girth > 0.06568032130599022:
                                                                                if Q.centroid_offset > 0.014050168450921774:
                                                                                    if Q.girth2_top2 > 0.0027218402829021215:
                                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        if Q.max_dr > 0.14082463830709457:
                                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1025.5234375:
                                                        if Q.mass > 72.67289733886719:
                                                            if Q.pt_7 > 35.203125:
                                                                if Q.centroid_offset > 0.003475709934718907:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.009528237394988537:
                                                                if Q.mass > 66.48260116577148:
                                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 37.66605758666992:
                                                            if Q.girth2 > 0.003610009327530861:
                                                                if Q.max_dr > 0.11844239011406898:
                                                                    if Q.girth > 0.07411134988069534:
                                                                        if Q.e2 > 0.03678821958601475:
                                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 812.0078125:
                                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_dr_0_0p05 > 0.05700664222240448:
                                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.centroid_offset > 0.013002284336835146:
                                                                                if Q.lam1 > 0.00587257300503552:
                                                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 23.5234375:
                                                                        if Q.mass > 73.9755744934082:
                                                                            if Q.z_7 > 0.04369296878576279:
                                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 47.56280708312988:
                                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.girth2_top3 > 0.003942607552744448:
                                                                                return 't'   # 78% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 681.765625:
                                                                    if Q.centroid_offset > 0.0024130988167598844:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 45.30072593688965:
                                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 44% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.007420195499435067:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.09474079310894012:
                                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.00400668871589005:
                                                                if Q.pt_7 > 27.25:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.008372560609132051:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 0.7875559628009796:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.058790843933820724:
                                                    if Q.centroid_offset > 0.01190845062956214:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 40.0557804107666:
                                                            if Q.lam1 > 0.0029499748488888144:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.planar_flow > 0.02972833812236786:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.8096021711826324:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009357034228742123:
                                                        if Q.mass > 59.82021141052246:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 46.78125:
                                                                if Q.eccentricity > 0.9895972609519958:
                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 43.01729965209961:
                                                            if Q.log_sum_pt > 7.149940729141235:
                                                                return 'g'   # 38% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 46.248870849609375:
                                                                    if Q.lam2 > 6.778441820642911e-05:
                                                                        if Q.pt_7 > 36.453125:
                                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_5 > 38.71875:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.10661524906754494:
                                                                if Q.pt_5 > 40.4375:
                                                                    if Q.mass > 38.17897033691406:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 36.984375:
                                                                    if Q.eccentricity > 0.9834924936294556:
                                                                        return 'W'   # 44% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 8.441994941676967e-05:
                                                                        return 'g'   # 38% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.013811859302222729:
                                        if Q.pt_7 > 27.125:
                                            if Q.centroid_offset > 0.01624922826886177:
                                                if Q.z_dr_0p1_0p2 > 0.21258001029491425:
                                                    if Q.pt_7 > 32.578125:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.005483138374984264:
                                                        if Q.z_7 > 0.07039816677570343:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.05603492259979248:
                                                    if Q.mass > 32.57296943664551:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top5 > 0.774320125579834:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 31.880215644836426:
                                                        if Q.max_dr > 0.09735861420631409:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_6 > 0.07149521261453629:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.392610311508179:
                                                if Q.centroid_offset > 0.018645377829670906:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 33.44009208679199:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004028848838061094:
                                            if Q.pt_7 > 28.6953125:
                                                if Q.pt_7 > 32.6875:
                                                    if Q.girth2 > 0.004224943928420544:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.7242207825183868:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.2128264084458351:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 33.52941703796387:
                                                if Q.centroid_offset > 0.011709289159625769:
                                                    if Q.max_dr > 0.11857766285538673:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.09012766554951668:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0035275634145364165:
                                                        if Q.D2 > 0.7144179344177246:
                                                            if Q.centroid_offset > 0.007749935379251838:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top5 > 0.8187782168388367:
                                                            if Q.girth2_top2 > 0.0023793381405994296:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_2 > 0.10606935992836952:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.004205992911010981:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 480.09375:
                                                            if Q.planar_flow > 0.08571840450167656:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 38.515625:
                            if Q.centroid_offset > 0.014328745659440756:
                                if Q.sum_pt > 902.234375:
                                    if Q.eccentricity > 0.985468715429306:
                                        if Q.pt_7 > 55.265625:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.0017119463300332427:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.12562628835439682:
                                        if Q.log_sum_pt > 6.7055816650390625:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 50.546875:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 43.609375:
                                                    if Q.min_pair_mass > 0.8285157084465027:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.08873307704925537:
                                            return 'W'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.06681964918971062:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.9873974025249481:
                                    if Q.mass > 48.194976806640625:
                                        if Q.pt_7 > 75.03125:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 44.4375:
                                            if Q.z_dr_0p1_0p2 > 0.07103989273309708:
                                                if Q.dr_0 > 0.026267041452229023:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.716575384140015:
                                                if Q.centroid_offset > 0.008340885397046804:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0024820499820634723:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10386811196804047:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 43.484375:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9775159060955048:
                                            if Q.mass > 39.25607109069824:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.006726363208144903:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.012678318191319704:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 32.403987884521484:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.021579323336482048:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013291803188621998:
                                if Q.centroid_offset > 0.01610014121979475:
                                    if Q.sum_pt > 993.7578125:
                                        if Q.eccentricity > 0.9785043299198151:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 36.359100341796875:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00014869724691379815:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 25.0546875:
                                                if Q.max_dr > 0.10845482721924782:
                                                    if Q.mass > 32.66533660888672:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.19346846640110016:
                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 42.458099365234375:
                                    if Q.centroid_offset > 0.004782527917996049:
                                        if Q.girth2 > 0.0022129302378743887:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.987155944108963:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_7 > 0.06497793272137642:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 22.8828125:
                                            if Q.lam1 > 0.002004532841965556:
                                                if Q.mass > 45.60860252380371:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.908512592315674:
                                        if Q.pt_7 > 26.0859375:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.003744205692782998:
                                                if Q.log_sum_pt > 7.035742282867432:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.010844696313142776:
                                            if Q.mass > 36.66753959655762:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.021288820542395115:
                                                    if Q.pt_7 > 28.65625:
                                                        if Q.max_dr > 0.11053277179598808:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.390625:
                                                if Q.C2 > 0.02552000992000103:
                                                    if Q.centroid_offset > 0.008359966333955526:
                                                        return 'q'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 5.763063563790638e-05:
                                                        if Q.sum_pt_top5 > 563.8125:
                                                            if Q.dr_0 > 0.029462135396897793:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.56802940368652:
                                                    if Q.centroid_offset > 0.006894408259540796:
                                                        if Q.z_7 > 0.02493143267929554:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 596.453125:
                if Q.centroid_offset > 0.015240416396409273:
                    if Q.centroid_offset > 0.02524277288466692:
                        if Q.girth2 > 0.0016775433905422688:
                            if Q.centroid_offset > 0.037225041538476944:
                                if Q.pt_7 > 29.3828125:
                                    if Q.centroid_offset > 0.04584885016083717:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.684133052825928:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_2 > 107.3125:
                                                if Q.planar_flow > 0.036533813923597336:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 24.0625:
                                        if Q.centroid_offset > 0.04254935681819916:
                                            if Q.log_sum_pt > 6.616232633590698:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.06790929101407528:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03018404357135296:
                                    if Q.planar_flow > 0.0748366191983223:
                                        if Q.pt_7 > 23.0703125:
                                            if Q.pt_7 > 41.109375:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.0010230413754470646:
                                                    if Q.lam2 > 0.00010588688746793196:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.032517166808247566:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.6492578983306885:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.21875:
                                            if Q.z_dr_0p1_0p2 > 0.038439417257905006:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.65240478515625:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 21.078125:
                                        if Q.eccentricity > 0.8090555965900421:
                                            if Q.lam2 > 8.358927152585238e-05:
                                                if Q.sum_pt_top5 > 642.4375:
                                                    if Q.tau21 > 0.2924734950065613:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 749.9140625:
                                                return 'g'   # 45% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.61494517326355:
                                            if Q.z_dr_0p1_0p2 > 0.02246727142482996:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 46.328125:
                                if Q.mass > 4.999197721481323:
                                    if Q.sum_pt > 810.46875:
                                        if Q.pt_4 > 77.4375:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 3.797387944359798e-05:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.0298692649230361:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.0010484500089660287:
                                            if Q.z_7 > 0.0629965029656887:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 325.875:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 934.078125:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_3 > 0.03663162514567375:
                                            return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.02773525658994913:
                                    if Q.sum_pt > 939.8359375:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 723.859375:
                                            if Q.pt_5 > 23.625:
                                                if Q.C2 > 0.03174120746552944:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0013811792014166713:
                                                        if Q.sum_pt > 826.171875:
                                                            if Q.pt_6 > 39.75:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.620177507400513:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.19068072736263275:
                                                                if Q.pt_7 > 22.7421875:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top5 > 3.3149263858795166:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.4375:
                                                if Q.centroid_offset > 0.029642801731824875:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.12436846271157265:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.66390061378479:
                                        if Q.lam1 > 0.0010584009578451514:
                                            if Q.girth2_top5 > 0.0011681589530780911:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 960.859375:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 815.7265625:
                                                    if Q.e2 > 0.007355975452810526:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 1.005232888928731e-05:
                                                            if Q.pt_7 > 40.4375:
                                                                if Q.C2 > 0.006380745908245444:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.026062078773975372:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.74089789390564:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.18093471974134445:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 801.8828125:
                                                            if Q.girth2_top3 > 0.0007000582409091294:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.8671875:
                                            if Q.planar_flow > 0.24240364134311676:
                                                if Q.centroid_offset > 0.02635265327990055:
                                                    if Q.sum_pt > 729.421875:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04091229848563671:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.026531093753874302:
                                                    if Q.log_sum_pt > 6.633493185043335:
                                                        if Q.eccentricity > 0.9847013652324677:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 753.640625:
                                                return 'Z'   # 41% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.515625:
                            if Q.centroid_offset > 0.01832941360771656:
                                if Q.mass > 6.759165525436401:
                                    if Q.pt_7 > 45.671875:
                                        if Q.planar_flow > 0.07603209093213081:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.785269021987915:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.024025713093578815:
                                                if Q.n_dr_0p05_0p1 > 0.5:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.39865072071552277:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.038281893357634544:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.840613126754761:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 50.765625:
                                            if Q.eccentricity > 0.9706037938594818:
                                                if Q.log_sum_pt > 6.758548021316528:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.023401320911943913:
                                                if Q.sum_pt > 825.203125:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9499720335006714:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 47.015625:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 6.173706879053498e-06:
                                    if Q.pt_7 > 45.390625:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.01727050542831421:
                                            if Q.mass_over_sum_pt > 0.005370560102164745:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 584.15625:
                                                if Q.mean_phi2 > 0.00011834767792606726:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 47.9375:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 63% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.018050862476229668:
                                if Q.sum_pt > 744.765625:
                                    if Q.centroid_offset > 0.02345799095928669:
                                        if Q.log_sum_pt > 6.729142427444458:
                                            if Q.lam1 > 0.0008864483970683068:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.024440116249024868:
                                                    if Q.sum_pt > 953.578125:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.06937004253268242:
                                                            if Q.centroid_offset > 0.024298492819070816:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 24.03125:
                                                                    if Q.pt_6 > 38.375:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 883.046875:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.812551259994507:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.2462940737605095:
                                                if Q.sum_pt_top5 > 681.5:
                                                    if Q.e2_sq > 0.00048241435433737934:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.8210515081882477:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.029762020334601402:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 20.78125:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 794.109375:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.95115852355957:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.670023441314697:
                                                if Q.pt_5 > 25.5859375:
                                                    if Q.log_sum_pt > 6.878833293914795:
                                                        if Q.z_7 > 0.034201715141534805:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.0217601815238595:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0005492592172231525:
                                                            if Q.pt_7 > 38.171875:
                                                                if Q.C2 > 0.014956120401620865:
                                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.029201358556747437:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.019617916084825993:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_7 > 0.02114979363977909:
                                                                            if Q.eccentricity > 0.8769591748714447:
                                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.centroid_offset > 0.021555799059569836:
                                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 860.91015625:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 17.8203125:
                                                            if Q.centroid_offset > 0.019622466526925564:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 24.390625:
                                                    if Q.centroid_offset > 0.01977991033345461:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 29.9296875:
                                                            if Q.pt_0 > 261.625:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.017248577438294888:
                                                                    if Q.max_dr > 0.02723192609846592:
                                                                        if Q.tau21 > 0.31904512643814087:
                                                                            if Q.max_dr > 0.03459564037621021:
                                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_eta2 > 0.00037191987212281674:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.0005616616108454764:
                                                        if Q.e2 > 0.01155221275985241:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 21.9609375:
                                                                return 'W'   # 47% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 26.3515625:
                                        if Q.centroid_offset > 0.020889476872980595:
                                            if Q.log_sum_pt > 6.564516305923462:
                                                if Q.pt_7 > 27.7109375:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.17687269300222397:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0006727234285790473:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.02230223547667265:
                                                if Q.max_dr > 0.09975112974643707:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.8786920309066772:
                                                        if Q.z_7 > 0.044940611347556114:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.04266427084803581:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0037012577522546053:
                                                        if Q.dr_0 > 0.019112002104520798:
                                                            return 'q'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.011973726097494364:
                                            if Q.pt_7 > 22.2109375:
                                                if Q.centroid_offset > 0.021878286264836788:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 21.78125:
                                                if Q.mass > 23.854254722595215:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.6439841985702515:
                                                        return 'g'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 840.8515625:
                                    if Q.sum_pt > 1006.5703125:
                                        if Q.pt_7 > 17.859375:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 873.0703125:
                                            if Q.pt_7 > 18.5703125:
                                                if Q.pt_7 > 32.765625:
                                                    if Q.C2 > 0.01271247724071145:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.814727306365967:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01675028447061777:
                                                        if Q.pt_7 > 15.6953125:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.7734375:
                                                if Q.z_dr_0p05_0p1 > 0.03755825199186802:
                                                    return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0002911373012466356:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.017075215466320515:
                                                    if Q.z_dr_0_0p05 > 0.9764637351036072:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 5.765750029240735e-05:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.044240016490221024:
                                        if Q.dr_0 > 0.02217391226440668:
                                            if Q.eccentricity > 0.945939689874649:
                                                if Q.max_dr > 0.09522419795393944:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2964444011449814:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.0453974325209856:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.030352515168488026:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.6015625:
                                            if Q.eccentricity > 0.7787016034126282:
                                                if Q.dr_0 > 0.011907673440873623:
                                                    if Q.log_sum_pt > 6.70550537109375:
                                                        if Q.centroid_offset > 0.016699605621397495:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.11703846976161003:
                                                            if Q.mass_over_sum_pt > 0.03653980419039726:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.015228526666760445:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 2.8428220502974e-05:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 680.171875:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 7.518232087022625e-05:
                                                if Q.pt_6 > 26.8515625:
                                                    if Q.max_dr > 0.1320064514875412:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 636.375:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.720898389816284:
                                                    if Q.centroid_offset > 0.01677078567445278:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 42.296875:
                        if Q.centroid_offset > 0.004032833967357874:
                            if Q.pt_7 > 47.203125:
                                if Q.girth2 > 4.232040737406351e-05:
                                    if Q.pt_7 > 49.609375:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005102550610899925:
                                            if Q.planar_flow > 0.0583552997559309:
                                                if Q.centroid_offset > 0.00684365164488554:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_2 > 97.46875:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 1.937477554747602e-05:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.00915990024805069:
                                                if Q.z_7 > 0.059468869119882584:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.814163684844971:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 3.962509072152898e-05:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.006460712291300297:
                                    if Q.planar_flow > 0.45409028232097626:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 825.09375:
                                            if Q.max_dr > 0.014662584289908409:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 894.65625:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010213146451860666:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.009775552898645401:
                                                    if Q.lam2 > 4.080224607605487e-05:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_dispersion > 0.4023934453725815:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 1.240519895873149e-05:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.685373306274414:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 912.03125:
                                        if Q.mass > 4.666942596435547:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 2.0387297809065785e-05:
                                            if Q.dr_0 > 0.009176729712635279:
                                                if Q.pt_7 > 44.078125:
                                                    if Q.sum_pt > 852.15625:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.005516166565939784:
                                                            if Q.dr_0 > 0.012322834692895412:
                                                                return 'q'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.689641587494407e-05:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_0 > 235.6875:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.004517991794273257:
                                                if Q.pt_dispersion > 0.39211010932922363:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 1.0018251941801282e-05:
                                                        if Q.centroid_offset > 0.004926193505525589:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.006234155735000968:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.891260623931885:
                                if Q.sum_pt > 1048.421875:
                                    if Q.lam1 > 1.620227612875169e-05:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1087.125:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 51.796875:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 3.422894951654598e-05:
                                        if Q.pt_7 > 47.234375:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 4.0592547520645894e-05:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0018643703078851104:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 50.46875:
                                            if Q.centroid_offset > 0.0015336488140746951:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mean_phi2 > 1.1144256859552115e-05:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.046875:
                                    if Q.width > 4.453965448192321e-05:
                                        if Q.centroid_offset > 0.001786219363566488:
                                            if Q.log_sum_pt > 6.723569393157959:
                                                if Q.lam2 > 9.72093221207615e-06:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 878.171875:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.002879671170376241:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.4827713072299957:
                                                    if Q.centroid_offset > 0.0023730152752250433:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top5 > 0.7764352858066559:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.03077117819339037:
                                                        if Q.C2 > 0.01380481431260705:
                                                            return 'g'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 55.140625:
                                                                return 'g'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.00284039624966681:
                                                            if Q.dr_0 > 0.009487034287303686:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.812877893447876:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.01087017497047782:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 62.15625:
                                                        if Q.centroid_offset > 0.0014806364779360592:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_eta2 > 3.5451088479021564e-05:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 2.5045103939191904e-05:
                                                            return 'g'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 3.572316018107813e-05:
                                            if Q.log_sum_pt > 6.835520505905151:
                                                if Q.pt_4 > 89.75:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0028620921075344086:
                                                    if Q.pt_7 > 59.25:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 940.53125:
                                                if Q.girth > 0.004978731973096728:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.014964202884584665:
                                        if Q.lam2 > 2.979928376589669e-05:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.006330269388854504:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 9.234230518341064:
                                            if Q.log_sum_pt > 6.817060947418213:
                                                if Q.centroid_offset > 0.0019406313076615334:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.5176933407783508:
                                                    if Q.pt_7 > 45.390625:
                                                        if Q.dr_7 > 0.013274340890347958:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0031451331451535225:
                                                if Q.z_7 > 0.06027277745306492:
                                                    if Q.max_dr > 0.014379022177308798:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 48.0625:
                                                        if Q.mass > 5.4617393016815186:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 33.796875:
                            if Q.centroid_offset > 0.007019019220024347:
                                if Q.sum_pt > 926.484375:
                                    if Q.log_sum_pt > 6.867665529251099:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.00012965760834049433:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 3.4030772440019064e-05:
                                        if Q.pt_7 > 37.234375:
                                            if Q.eccentricity > 0.8966525793075562:
                                                if Q.D2 > 2.498302698135376:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04979872703552246:
                                                        if Q.centroid_offset > 0.010182637255638838:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.7776643335819244:
                                                    if Q.centroid_offset > 0.009697210974991322:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.012816831469535828:
                                                            if Q.z_5 > 0.0636901929974556:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.38061201572418213:
                                                if Q.sum_pt_top3 > 525.21875:
                                                    if Q.centroid_offset > 0.010624569840729237:
                                                        if Q.dr_0 > 0.015070946887135506:
                                                            if Q.lam2 > 7.118222492863424e-05:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.009539324790239334:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_eta2 > 7.095394175848924e-05:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008963429369032383:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.010378210805356503:
                                                            if Q.eccentricity > 0.716986745595932:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.014119949657469988:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 37.765625:
                                            if Q.mass > 9.345941543579102:
                                                if Q.dr_0 > 0.0106898108497262:
                                                    if Q.log_sum_pt > 6.721237421035767:
                                                        if Q.max_dr > 0.06404897943139076:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 1.4779813682253007e-05:
                                                    if Q.pt_dispersion > 0.412134125828743:
                                                        if Q.max_dr > 0.01716170646250248:
                                                            if Q.centroid_offset > 0.0085755898617208:
                                                                if Q.dr_0 > 0.009999689646065235:
                                                                    if Q.dr_2 > 0.01152371196076274:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 39.453125:
                                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top5 > 0.00010492778164916672:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.011393394321203232:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.011871992144733667:
                                                        if Q.lam2 > 7.711465514148585e-06:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 762.8125:
                                                            if Q.pt_6 > 51.046875:
                                                                if Q.sum_pt > 875.984375:
                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_dispersion > 0.4058472216129303:
                                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.020438908599317074:
                                                                    if Q.dr_0 > 0.008163684979081154:
                                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.009764391928911209:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 633.125:
                                                if Q.lam2 > 2.197216781496536e-05:
                                                    if Q.pt_dispersion > 0.43434663116931915:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.013359226286411285:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.008991328068077564:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.012459061108529568:
                                                        if Q.dr_0 > 0.011369501706212759:
                                                            if Q.log_sum_pt > 6.7844557762146:
                                                                return 'W'   # 42% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01582515798509121:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 1.3540545751311583e-05:
                                                        if Q.dr_0 > 0.01077882805839181:
                                                            if Q.centroid_offset > 0.010778397787362337:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.010033939965069294:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.934412479400635:
                                    if Q.width > 4.008108408015687e-05:
                                        if Q.centroid_offset > 0.0023159197298809886:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1055.875:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.986941814422607:
                                            if Q.centroid_offset > 0.0019232198828831315:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 5.241652727127075:
                                                    if Q.pt_7 > 38.21875:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_7 > 0.009021561592817307:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.004826997872442007:
                                        if Q.sum_pt > 974.328125:
                                            if Q.pt_7 > 36.359375:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 625.84375:
                                                if Q.lam2 > 2.9786558116029482e-05:
                                                    if Q.pt_7 > 38.984375:
                                                        if Q.dr_0 > 0.01276826998218894:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 4.13639354519546e-05:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top3 > 532.1875:
                                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.005510398419573903:
                                                            if Q.tau32 > 0.5035769045352936:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.005926316836848855:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.003678842098452151:
                                                        if Q.pt_7 > 39.984375:
                                                            if Q.mean_eta2 > 4.6703702537342906e-05:
                                                                if Q.dr_3 > 0.014191328547894955:
                                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 1.2048330972902477e-05:
                                                                        if Q.pt_dispersion > 0.42288829386234283:
                                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 1.5422993783431593e-05:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.010444986633956432:
                                                    if Q.lam2 > 4.697332042269409e-05:
                                                        if Q.dr_0 > 0.016426660120487213:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 757.390625:
                                                        if Q.lam2 > 2.529337416490307e-05:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 744.359375:
                                            if Q.lam2 > 7.205767906270921e-05:
                                                if Q.pt_7 > 38.8125:
                                                    if Q.log_sum_pt > 6.695451021194458:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.902884006500244:
                                                    if Q.girth2_top3 > 5.516854071174748e-05:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 758.703125:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.00493448437191546:
                                                            if Q.centroid_offset > 0.004038306185975671:
                                                                if Q.dr_0 > 0.0074865256901830435:
                                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 4.1915061956387945e-05:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 3.561673111107666e-05:
                                                    if Q.centroid_offset > 0.003472423064522445:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0020081757102161646:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.6125213503837585:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.988492488861084:
                                if Q.centroid_offset > 0.003343026852235198:
                                    if Q.pt_7 > 17.78125:
                                        if Q.girth2_top5 > 3.053559430554742e-05:
                                            if Q.max_dr > 0.12442228943109512:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005580039229243994:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 23.78125:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.06033522076904774:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 22.6953125:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.007432479877024889:
                                            if Q.pt_7 > 13.515625:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 24.6953125:
                                        if Q.girth2 > 3.962043956562411e-05:
                                            if Q.pt_7 > 30.34375:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0023559514665976167:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_7 > 0.015300690196454525:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 21.0:
                                            if Q.mass_top5 > 8.795521259307861:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.603636741638184:
                                    if Q.centroid_offset > 0.01297859800979495:
                                        if Q.log_sum_pt > 6.786955118179321:
                                            if Q.pt_7 > 18.6015625:
                                                if Q.log_sum_pt > 6.904013633728027:
                                                    if Q.pt_7 > 22.46875:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 913.796875:
                                                        if Q.z_7 > 0.03190181031823158:
                                                            if Q.max_dr > 0.02926691807806492:
                                                                return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi > 0.004366960376501083:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 27.078125:
                                                                if Q.mass_over_sum_pt > 0.007535481825470924:
                                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 935.62890625:
                                                    if Q.centroid_offset > 0.013975589536130428:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00014095277583692223:
                                                if Q.log_sum_pt > 6.734944581985474:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 30.7578125:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.0079790148884058:
                                                    if Q.dr_0 > 0.010915372520685196:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.03528504632413387:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.009682916570454836:
                                            if Q.sum_pt > 964.0234375:
                                                if Q.pt_7 > 24.390625:
                                                    if Q.sum_pt > 993.171875:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.7578125:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.927265167236328:
                                                        if Q.pt_7 > 17.328125:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 30.9140625:
                                                    if Q.lam2 > 3.4878776205005124e-05:
                                                        if Q.dr_0 > 0.01168148685246706:
                                                            if Q.log_sum_pt > 6.785552263259888:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.00010746922271209769:
                                                                    if Q.phi_7 > -0.008403778076171875:
                                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 569.6875:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.004212637664750218:
                                                        if Q.log_sum_pt > 6.834953546524048:
                                                            if Q.pt_7 > 22.3984375:
                                                                if Q.lam2 > 4.818065463041421e-05:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.011905522085726261:
                                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 806.8359375:
                                                if Q.log_sum_pt > 6.948814630508423:
                                                    if Q.centroid_offset > 0.004071059403941035:
                                                        if Q.pt_7 > 24.3203125:
                                                            if Q.lam1 > 4.42708933405811e-05:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 18.1328125:
                                                                if Q.mean_phi2 > 7.13021945557557e-05:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 9.969557140721008e-05:
                                                            if Q.pt_7 > 23.9140625:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1006.0859375:
                                                        if Q.centroid_offset > 0.00534566817805171:
                                                            if Q.pt_7 > 24.484375:
                                                                if Q.width > 0.00011480072134872898:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.72820782661438:
                                                            return 'q'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top5 > 0.9471074342727661:
                                                                if Q.girth2 > 9.138253153651021e-05:
                                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0011851056478917599:
                                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_5 > 30.7890625:
                                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 5.702562884835061e-05:
                                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.00204473698977381:
                                                    if Q.girth > 0.006156853167340159:
                                                        if Q.z_7 > 0.04341815784573555:
                                                            if Q.centroid_offset > 0.005974401254206896:
                                                                if Q.girth2_top2 > 0.0001096706764656119:
                                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_5 > 30.53125:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.023166362196207047:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_4 > 46.328125:
                                                        if Q.pt_5 > 31.5859375:
                                                            if Q.log_sum_pt > 6.622684478759766:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.018941359594464302:
                                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2_sq > 0.00011900131357833743:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 12.328125:
                                                                if Q.pt_5 > 33.703125:
                                                                    if Q.sum_pt > 781.78125:
                                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.008968944195657969:
                                        if Q.eccentricity > 0.788802981376648:
                                            if Q.dr_0 > 0.016095658764243126:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 7.539923171862029e-05:
                                                    if Q.centroid_offset > 0.009604015853255987:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04553358070552349:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.012937942519783974:
                                                            if Q.sum_pt_top5 > 615.125:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.041773999109864235:
                                                if Q.centroid_offset > 0.008200420532375574:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 626.8125:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.010227298364043236:
                                                        if Q.dr_0 > 0.01487840386107564:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.010896328836679459:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_6 > 29.203125:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 723.265625:
                                            if Q.centroid_offset > 0.006541972979903221:
                                                if Q.lam2 > 1.5438103673659498e-05:
                                                    if Q.z_6 > 0.043733397498726845:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.005923500983044505:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.0069854068569839:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.017390556633472443:
                                                if Q.dr_0 > 0.005990184843540192:
                                                    if Q.centroid_offset > 0.007519482169300318:
                                                        if Q.z_7 > 0.035176295787096024:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0029189969645813107:
                                                            if Q.pt_4 > 31.9609375:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 3.4494747524149716e-05:
                                                        if Q.eccentricity > 0.8460928499698639:
                                                            if Q.centroid_offset > 0.006556277861818671:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.010723287239670753:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_4 > 46.390625:
                                                                        return 'q'   # 45% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_5 > 0.025033111684024334:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.022896800190210342:
                    if Q.sum_pt_top5 > 485.734375:
                        if Q.centroid_offset > 0.029591670259833336:
                            if Q.centroid_offset > 0.04361565597355366:
                                if Q.z_7 > 0.044830018654465675:
                                    if Q.mass_over_sum_pt > 0.03785932622849941:
                                        if Q.centroid_offset > 0.053150469437241554:
                                            return 't'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.0017000431544147432:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_2 > 0.0751115158200264:
                                            if Q.pt_7 > 36.421875:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.07693108543753624:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.24621840566396713:
                                                if Q.pt_7 > 32.671875:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0035941957030445337:
                                                        return 't'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_2 > 91.5625:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eta_0 > -0.0023355484008789062:
                                                        if Q.mass_top5 > 3.144718885421753:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.9838185608386993:
                                        if Q.sum_pt_top5 > 510.78125:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2_sq > 0.0014323454815894365:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.037434034049510956:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 48% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt_sq > 0.0008402852108702064:
                                    if Q.lam2 > 0.00013602541730506346:
                                        if Q.centroid_offset > 0.0333882924169302:
                                            if Q.planar_flow > 0.5749183595180511:
                                                if Q.mass_over_sum_pt_sq > 0.001506432774476707:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 26.8828125:
                                                    if Q.pt_7 > 37.328125:
                                                        if Q.sum_pt > 668.203125:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0018862823490053415:
                                                if Q.pt_6 > 29.3671875:
                                                    if Q.tau21 > 0.33179737627506256:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 646.1640625:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 36% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.01865245122462511:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.1953125:
                                            if Q.centroid_offset > 0.035637641325592995:
                                                if Q.max_dr > 0.13649356365203857:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.0017127085011452436:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 4.015652302769013e-05:
                                                            if Q.log_sum_pt > 6.450340986251831:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 35% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.03905520774424076:
                                                                return 'Z'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 47.1875:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1587493196129799:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 4.928120688418858e-05:
                                                            if Q.centroid_offset > 0.032486118376255035:
                                                                if Q.e2 > 0.016574286855757236:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.0327878724783659:
                                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.932939052581787:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.4467854499816895:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 38% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 29.1953125:
                                        if Q.pt_7 > 52.171875:
                                            if Q.max_dr > 0.042576609179377556:
                                                if Q.lam2 > 2.2207186702871695e-05:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 56.90625:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.095931449730415e-05:
                                                    if Q.pt_7 > 57.015625:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 617.3671875:
                                                if Q.centroid_offset > 0.031695131212472916:
                                                    if Q.centroid_offset > 0.04055679030716419:
                                                        if Q.pt_2 > 101.21875:
                                                            if Q.eccentricity > 0.988735020160675:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.025454336777329445:
                                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.4969871044158936:
                                                                if Q.pt_7 > 45.734375:
                                                                    if Q.lam2 > 6.649608621955849e-05:
                                                                        if Q.pt_2 > 93.3125:
                                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.20250026881694794:
                                                                    if Q.pt_2 > 104.53125:
                                                                        if Q.dr_3 > 0.04045500047504902:
                                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.max_dr > 0.046273352578282356:
                                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.9764909744262695:
                                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 658.9453125:
                                                        if Q.lam1 > 0.0011573350639082491:
                                                            if Q.eccentricity > 0.9585395753383636:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 34.328125:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.planar_flow > 0.07862046733498573:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.11870745196938515:
                                                            if Q.C2 > 0.012263291515409946:
                                                                if Q.girth2_top3 > 0.0012535951100289822:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.03502679988741875:
                                                    if Q.pt_2 > 84.21875:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 542.828125:
                                            if Q.pt_7 > 27.1796875:
                                                if Q.log_sum_pt > 6.518686294555664:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_2 > 91.625:
                                                        return 'g'   # 43% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.008938302751630545:
                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 24.28125:
                                                        if Q.lam2 > 5.469141251523979e-05:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.0409191008657217:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.17513827234506607:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.1757570207118988:
                                if Q.lam1 > 0.001366849581245333:
                                    if Q.pt_7 > 25.96875:
                                        if Q.pt_7 > 46.484375:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.0010714668314903975:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.4949939250946045:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.03769482485949993:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00023238440189743415:
                                                return 'g'   # 27% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 663.4609375:
                                        if Q.z_7 > 0.06559516116976738:
                                            if Q.z_top5 > 0.7790473997592926:
                                                return 'g'   # 41% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02690511103719473:
                                                    if Q.max_dr > 0.04097684100270271:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02747170254588127:
                                                if Q.pt_7 > 31.9375:
                                                    if Q.log_sum_pt > 6.5488903522491455:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02811251860111952:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 30% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0006143574137240648:
                                                    if Q.planar_flow > 0.5214493274688721:
                                                        if Q.tau21 > 0.5031823813915253:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.02571746800094843:
                                                                return 'W'   # 42% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_2 > 94.6875:
                                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.78125:
                                                            if Q.girth2_top2 > 0.0008941709529608488:
                                                                return 'W'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 689.8515625:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.18316581100225449:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.54033088684082:
                                                        if Q.centroid_offset > 0.024044214747846127:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0008698948950041085:
                                            if Q.log_sum_pt > 6.446066856384277:
                                                if Q.eccentricity > 0.9291059374809265:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.001077879045624286:
                                                        if Q.D2 > 1.020404577255249:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 26.359375:
                                    if Q.sum_pt_top5 > 535.09375:
                                        if Q.pt_7 > 48.390625:
                                            if Q.centroid_offset > 0.026553120464086533:
                                                if Q.C2 > 0.005904106423258781:
                                                    return 'g'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.1234886646270752:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.027757836505770683:
                                                if Q.pt_7 > 39.796875:
                                                    if Q.sum_pt > 714.359375:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 679.65625:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.1827242746949196:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 31.578125:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.02937591541558504:
                                            if Q.mass_over_sum_pt > 0.036353159695863724:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 35.015625:
                                                    if Q.centroid_offset > 0.0250325258821249:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.483608961105347:
                                                if Q.LHA > 0.17705423384904861:
                                                    if Q.planar_flow > 0.1234324723482132:
                                                        return 'g'   # 39% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0031542181968688965:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.1904296651482582:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 529.90625:
                                        return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 24.02134132385254:
                            if Q.width > 0.004747697617858648:
                                if Q.tau21 > 0.22838520258665085:
                                    if Q.sum_pt > 436.6640625:
                                        if Q.centroid_offset > 0.049275027588009834:
                                            if Q.pt_7 > 35.5625:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.3850117474794388:
                                                    if Q.lam1 > 0.00596949178725481:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.7091246247291565:
                                                if Q.pt_7 > 30.9609375:
                                                    if Q.D2 > 1.5512736439704895:
                                                        if Q.girth2 > 0.005387321347370744:
                                                            return 'g'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.005465554539114237:
                                                            return 'Z'   # 36% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.32779401540756226:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 35.390625:
                                            if Q.max_dr > 0.10085448995232582:
                                                if Q.centroid_offset > 0.036555685102939606:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.6456420719623566:
                                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.040001051500439644:
                                        if Q.pt_7 > 43.296875:
                                            return 't'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.033833980560303:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_7 > 0.09438126161694527:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0003285276470705867:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.8046875:
                                                if Q.max_dr > 0.12302243337035179:
                                                    return 't'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 27.743252754211426:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 32% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 390.328125:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03444527089595795:
                                    if Q.lam2 > 0.0001123455076594837:
                                        if Q.planar_flow > 0.5266201198101044:
                                            if Q.tau21 > 0.19055591523647308:
                                                if Q.mass > 27.714491844177246:
                                                    return 'Z'   # 35% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 33.953125:
                                                if Q.centroid_offset > 0.046322932466864586:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_3 > 0.03660250082612038:
                                                        if Q.e2 > 0.02469662856310606:
                                                            if Q.mass > 27.581761360168457:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 35% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.24629579484462738:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.12779533118009567:
                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.040830790996551514:
                                                if Q.m012 > 1.394811749458313:
                                                    if Q.centroid_offset > 0.043868545442819595:
                                                        return 'Z'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 29.0234375:
                                        if Q.mass > 26.23923110961914:
                                            if Q.max_dr > 0.15848740190267563:
                                                if Q.girth2 > 0.0036710062995553017:
                                                    if Q.n_dr_0_0p05 > 5.5:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.19035913795232773:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.5333627462387085:
                                                        if Q.pt_7 > 45.8125:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top2 > 0.001573209068737924:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0005255946889519691:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 423.046875:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.001994326477870345:
                                                        if Q.pt_7 > 32.65625:
                                                            if Q.centroid_offset > 0.024722048081457615:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.2111639678478241:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 38% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 540.1953125:
                                            if Q.pt_7 > 27.015625:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.17587339133024216:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 583.671875:
                                if Q.centroid_offset > 0.03044667560607195:
                                    if Q.centroid_offset > 0.04392710328102112:
                                        if Q.z_7 > 0.052333805710077286:
                                            if Q.pt_dispersion > 0.38870659470558167:
                                                if Q.centroid_offset > 0.049114298075437546:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.5580267906188965:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.03414464555680752:
                                            if Q.z_7 > 0.0846060961484909:
                                                if Q.sum_pt_top3 > 315.15625:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.8126071095466614:
                                                    if Q.D2 > 1.776097595691681:
                                                        if Q.pt_2 > 82.15625:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_2 > 76.75:
                                                            if Q.sum_pt > 603.0625:
                                                                if Q.max_dr > 0.046070558950304985:
                                                                    if Q.pt_6 > 53.34375:
                                                                        if Q.mass_over_sum_pt_sq > 0.0001558009462314658:
                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top3 > 0.0016949152923189104:
                                                                    if Q.sum_pt_top5 > 461.234375:
                                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 612.859375:
                                                if Q.e2 > 0.004669007146731019:
                                                    if Q.pt_7 > 49.171875:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.03673502616584301:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.001813500712160021:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.0017122183926403522:
                                        if Q.z_7 > 0.061527194455266:
                                            if Q.tau21 > 0.23164864629507065:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.044702861458063126:
                                            if Q.planar_flow > 0.1166258230805397:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.030021163634955883:
                                                    if Q.sum_pt > 615.921875:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 609.953125:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.06855425983667374:
                                    if Q.z_7 > 0.06814109906554222:
                                        if Q.tau21 > 0.2792615294456482:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 483.671875:
                                            if Q.lam1 > 0.005467080511152744:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.21807878464460373:
                                        if Q.mass > 21.49907398223877:
                                            if Q.lam2 > 7.011427078396082e-05:
                                                if Q.lam2 > 0.0003228451532777399:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.030365895479917526:
                                                        if Q.pt_6 > 38.140625:
                                                            if Q.centroid_offset > 0.038946421816945076:
                                                                if Q.lam1 > 0.003853330039419234:
                                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top2 > 0.0023835627362132072:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.027146339416503906:
                                                    if Q.log_sum_pt > 6.291676759719849:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi > -0.012128946837037802:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03552968241274357:
                                                if Q.log_sum_pt > 6.317333936691284:
                                                    if Q.z_7 > 0.07334312796592712:
                                                        if Q.centroid_offset > 0.0489556472748518:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_dispersion > 0.3892935812473297:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mean_phi2 > 0.0009775935031939298:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.008677256293594837:
                                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04411211796104908:
                                                        if Q.tau21 > 0.2895726412534714:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_6 > 41.953125:
                                                                if Q.LHA > 0.2662419378757477:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass_over_sum_pt_sq > 0.0008775762980803847:
                                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.051670435816049576:
                                                            return 't'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.03117178939282894:
                                            if Q.pt_7 > 37.3125:
                                                if Q.lam2 > 0.0005475408397614956:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0001049284910550341:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.04327804781496525:
                                                            return 'Z'   # 45% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_6 > 0.08529873192310333:
                                                                return 'g'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.1439138948917389:
                                                    if Q.dr01 > 0.010431003756821156:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 20.164286613464355:
                                                            if Q.centroid_offset > 0.04029553197324276:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03983812779188156:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 540.890625:
                        if Q.centroid_offset > 0.004053136566653848:
                            if Q.mass_over_sum_pt > 0.017126908525824547:
                                if Q.z_7 > 0.056332020089030266:
                                    if Q.lam2 > 3.497690522635821e-05:
                                        if Q.girth2 > 0.0020667299395427108:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.005653529427945614:
                                                if Q.centroid_offset > 0.01982214394956827:
                                                    if Q.z_7 > 0.06298103928565979:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.3140076696872711:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06352701783180237:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.1118146181106567:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.007924396079033613:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.2144498825073242:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 47.21875:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.06582706794142723:
                                            if Q.centroid_offset > 0.006966616492718458:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 50.0:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.016915222629904747:
                                                if Q.width > 0.001648765231948346:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.011201940011233091:
                                                    if Q.centroid_offset > 0.01156299328431487:
                                                        return 'g'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.8919907212257385:
                                        if Q.dr_0 > 0.012143988162279129:
                                            if Q.centroid_offset > 0.01816628687083721:
                                                if Q.mass > 25.34074592590332:
                                                    if Q.z_7 > 0.03888104110956192:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04430173896253109:
                                                        if Q.centroid_offset > 0.02126707136631012:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 381.34375:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.031217578798532486:
                                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.042645858600735664:
                                                    if Q.eccentricity > 0.9710212349891663:
                                                        if Q.C2 > 0.031563905999064445:
                                                            if Q.z_7 > 0.0480225645005703:
                                                                return 'g'   # 47% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.047307247295975685:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top2 > 0.0004323776811361313:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.01664222590625286:
                                                        if Q.centroid_offset > 0.012307124678045511:
                                                            if Q.pt_7 > 33.3125:
                                                                if Q.eccentricity > 0.9516662359237671:
                                                                    if Q.sum_pt_top3 > 435.59375:
                                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.010132080875337124:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 4.2819607187993824e-05:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.00040422675374429673:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.019218942150473595:
                                            if Q.mass > 25.15913677215576:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 686.15625:
                                                    if Q.girth2_top2 > 0.0006593704747501761:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.007106204051524401:
                                                if Q.z_7 > 0.04372924752533436:
                                                    if Q.eccentricity > 0.8246075510978699:
                                                        if Q.sum_pt_top3 > 445.4375:
                                                            if Q.mass_top3 > 3.3523415327072144:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.021294563077390194:
                                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.020090607926249504:
                                                        if Q.eccentricity > 0.7927573025226593:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01566009735688567:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.011195233557373285:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.013792972080409527:
                                                                return 'q'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.013339108787477016:
                                                    if Q.eccentricity > 0.6760818958282471:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.04489566572010517:
                                    if Q.centroid_offset > 0.020529957488179207:
                                        if Q.C2 > 0.004513531690463424:
                                            if Q.planar_flow > 0.2554510235786438:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 50.28125:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.549427270889282:
                                                        if Q.dr_0 > 0.019091462716460228:
                                                            if Q.z_7 > 0.06222137436270714:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005749555304646492:
                                            if Q.mass_over_sum_pt > 0.013901473954319954:
                                                if Q.lam2 > 2.8567855224537198e-05:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 44.5:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.010975698474794626:
                                                            if Q.girth2 > 0.00038961834798101336:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.06374368071556091:
                                                    if Q.z_7 > 0.0524265393614769:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 752.53125:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.8710293173789978:
                                                                if Q.mass > 6.558216094970703:
                                                                    if Q.max_dr > 0.02368865627795458:
                                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.010237234644591808:
                                                if Q.z_7 > 0.06445543840527534:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.7433241307735443:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 432.78125:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 4.090772017661948e-05:
                                                    if Q.eccentricity > 0.9019382894039154:
                                                        if Q.dr_0 > 0.00734075577929616:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 588.53125:
                                                            if Q.z_7 > 0.050965091213583946:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.8211969137191772:
                                        if Q.sum_pt_top5 > 565.203125:
                                            if Q.dr_0 > 0.012106487527489662:
                                                if Q.centroid_offset > 0.016221781261265278:
                                                    if Q.z_7 > 0.036225421354174614:
                                                        if Q.dr_0 > 0.01848740130662918:
                                                            if Q.pt_2 > 80.875:
                                                                if Q.z_top5 > 0.848038911819458:
                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_6 > 0.061403874307870865:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.0079574584960938:
                                                        if Q.dr_2 > 0.010543684009462595:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.031509965658187866:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 706.53125:
                                            if Q.centroid_offset > 0.006343460641801357:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.02723692264407873:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.dr_0 > 0.010161080863326788:
                                if Q.z_7 > 0.06859253719449043:
                                    if Q.centroid_offset > 0.002607349306344986:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 2.833378857758362e-05:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.3704020380973816:
                                        if Q.mass > 22.83892822265625:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 683.828125:
                                                if Q.z_7 > 0.06105951406061649:
                                                    if Q.eccentricity > 0.7994354367256165:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.00026748918753582984:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.width > 4.117049138585571e-05:
                                    if Q.sum_pt_top5 > 571.953125:
                                        if Q.z_7 > 0.06626986339688301:
                                            if Q.centroid_offset > 0.0015366620500572026:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.7701043784618378:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.583754777908325:
                                                if Q.centroid_offset > 0.0029607063625007868:
                                                    if Q.z_7 > 0.056105541065335274:
                                                        if Q.lam2 > 1.5159326267166762e-05:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.6662268042564392:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 2.312031665496761e-05:
                                                        if Q.dr_0 > 0.004932394018396735:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 7.3007824420928955:
                                                    if Q.girth2_top2 > 4.617472586687654e-05:
                                                        if Q.pt_4 > 47.953125:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.043404147028923035:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 709.8125:
                                                        if Q.planar_flow > 0.6607290804386139:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.5484992265701294:
                                            if Q.z_5 > 0.09337466582655907:
                                                if Q.z_7 > 0.059067970141768456:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.006217621965333819:
                                                if Q.centroid_offset > 0.0020020557567477226:
                                                    if Q.pt_7 > 49.1875:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 3.205802568118088e-05:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_5 > 43.265625:
                                                                if Q.girth2_top5 > 8.898443047655746e-05:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 560.90625:
                                                    if Q.dr_0 > 0.003223237465135753:
                                                        if Q.phi_7 > -0.0033521652221679688:
                                                            if Q.dr_7 > 0.017187611665576696:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.587657690048218:
                                        return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.547995090484619:
                                            if Q.centroid_offset > 0.002337412443011999:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 3.475460835034028e-05:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_dispersion > 0.39369750022888184:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 2.958563436550321e-05:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 47% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 27.90214729309082:
                            if Q.centroid_offset > 0.017897718586027622:
                                if Q.pt_7 > 29.390625:
                                    if Q.z_dr_0p1_0p2 > 0.19173772633075714:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.18899666517972946:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.08247776702046394:
                                                if Q.mass > 28.434720993041992:
                                                    if Q.girth2 > 0.005099398549646139:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.10857699811458588:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.054133741185069084:
                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.019466416910290718:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_phi2 > 0.001332338957581669:
                                                        if Q.phi_7 > -0.025543212890625:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.019127200357615948:
                                                    if Q.pt_7 > 45.125:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top5 > 0.0042902149725705385:
                                    if Q.pt_7 > 31.5546875:
                                        if Q.mass_over_sum_pt_sq > 0.006136735435575247:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.037325458601117134:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.011774901300668716:
                                                    if Q.C2 > 0.024981550872325897:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10942325741052628:
                                                        if Q.max_pair_mass > 11.549719333648682:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 509.84375:
                                        if Q.tau21 > 0.23115438222885132:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.058530399575829506:
                                                if Q.centroid_offset > 0.007595626637339592:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.001753634656779468:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.015274159144610167:
                                            if Q.max_dr > 0.1193089410662651:
                                                if Q.pt_7 > 30.6484375:
                                                    if Q.mass > 29.368616104125977:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0032680120784789324:
                                                    if Q.pt_5 > 43.671875:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.034484343603253365:
                                                if Q.pt_4 > 47.46875:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 475.625:
                                                    if Q.C2 > 0.013559630140662193:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.8973737359046936:
                                                        if Q.girth > 0.03827219642698765:
                                                            if Q.mass > 29.48546600341797:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 1.0053434371948242:
                                if Q.sum_pt_top5 > 504.390625:
                                    if Q.tau21 > 0.47599175572395325:
                                        if Q.width > 3.523660416249186e-05:
                                            if Q.mass_top5 > 6.170742511749268:
                                                if Q.centroid_offset > 0.0038834639126434922:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.011019519995898008:
                                                        if Q.z_7 > 0.07269754260778427:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.497904539108276:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.030399692244827747:
                                                    if Q.planar_flow > 0.04791224002838135:
                                                        if Q.centroid_offset > 0.0025726709282025695:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_5 > 0.09842615947127342:
                                                                if Q.planar_flow > 0.434456929564476:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.8201041519641876:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 700.859375:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 2.6073680601257365e-05:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.4477321654558182:
                                            if Q.centroid_offset > 0.0021295264596119523:
                                                if Q.centroid_offset > 0.005200434476137161:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.01871267333626747:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.05770007707178593:
                                                if Q.centroid_offset > 0.004220650298520923:
                                                    if Q.centroid_offset > 0.007104696240276098:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.018709838390350342:
                                                            if Q.z_7 > 0.0658358745276928:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.01086795562878251:
                                                        if Q.mass > 18.766173362731934:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 48.578125:
                                                                if Q.dr_0 > 0.01538851484656334:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.011823778506368399:
                                                    if Q.centroid_offset > 0.007928552571684122:
                                                        if Q.z_7 > 0.049039121717214584:
                                                            if Q.centroid_offset > 0.010405322071164846:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_3 > 0.020383073017001152:
                                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.19649729877710342:
                                                                if Q.e2 > 0.00997126242145896:
                                                                    if Q.centroid_offset > 0.012749691493809223:
                                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.9301464557647705:
                                                                    if Q.dr_6 > 0.024164384230971336:
                                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.017533790320158005:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 4.7343950427602977e-05:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.002805943717248738:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 26.413575172424316:
                                        if Q.centroid_offset > 0.01934321317821741:
                                            if Q.e2 > 0.0195009708404541:
                                                if Q.pt_7 > 32.625:
                                                    if Q.dr_0 > 0.04719311371445656:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.2741353362798691:
                                                if Q.pt_6 > 34.6875:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0034498682944104075:
                                            if Q.planar_flow > 0.09796155616641045:
                                                if Q.z_7 > 0.026268143206834793:
                                                    if Q.sum_pt_top5 > 475.015625:
                                                        if Q.tau21 > 0.4196292906999588:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.0048642263282090425:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.01675447355955839:
                                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 453.71875:
                                                    if Q.centroid_offset > 0.013853958807885647:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.5991269946098328:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 44.890625:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_6 > 0.07218321412801743:
                                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top5 > 8.74577522277832:
                                                if Q.sum_pt_top5 > 443.53125:
                                                    if Q.dr_0 > 0.015681027434766293:
                                                        if Q.z_7 > 0.07545068487524986:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.3486332893371582:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.n_pt_above_50 > 4.5:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.1232933402061462:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 262.96875:
                                                            if Q.girth2_top2 > 0.0009176079474855214:
                                                                return 'q'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.005666545359417796:
                                    if Q.sum_pt_top3 > 352.453125:
                                        if Q.eccentricity > 0.9615620374679565:
                                            if Q.centroid_offset > 0.01239285385236144:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 47.203125:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.023264839313924313:
                                                        if Q.centroid_offset > 0.008222359698265791:
                                                            if Q.z_7 > 0.06012366898357868:
                                                                if Q.sum_pt_top3 > 371.78125:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 410.28125:
                                                if Q.eccentricity > 0.8991156816482544:
                                                    if Q.centroid_offset > 0.011310619302093983:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007384795928373933:
                                                    if Q.centroid_offset > 0.019613638520240784:
                                                        if Q.D2 > 0.7195990085601807:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.24200697243213654:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 281.21875:
                                                                if Q.centroid_offset > 0.012044749222695827:
                                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_dispersion > 0.4137086272239685:
                                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06455228105187416:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 3.0848972528474405e-05:
                                            if Q.LHA > 0.2676773965358734:
                                                if Q.pt_7 > 35.265625:
                                                    if Q.girth2 > 0.004203966585919261:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 446.953125:
                                                    if Q.z_7 > 0.062478529289364815:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009816121775656939:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.030396629124879837:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.06658870354294777:
                                                if Q.centroid_offset > 0.006645591929554939:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.07536975666880608:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.011542163789272308:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.17853888869285583:
                                                        if Q.mean_eta > -0.0028263978892937303:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 430.4375:
                                        if Q.z_7 > 0.07634837180376053:
                                            if Q.centroid_offset > 0.003137268009595573:
                                                if Q.planar_flow > 0.18234549462795258:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top5 > 0.7390008568763733:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.016827463172376156:
                                                    if Q.lam2 > 2.0644014512072317e-05:
                                                        if Q.mass > 18.13457202911377:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.017317602410912514:
                                                if Q.sum_pt > 592.953125:
                                                    if Q.pt_7 > 48.109375:
                                                        if Q.centroid_offset > 0.004211209947243333:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.851405143737793:
                                                        if Q.lam1 > 0.0006315503269433975:
                                                            if Q.sum_pt_top5 > 445.734375:
                                                                return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.450939178466797:
                                                    if Q.centroid_offset > 0.003729906748048961:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 270.3125:
                                            if Q.dr_0 > 0.021485932171344757:
                                                if Q.centroid_offset > 0.0038551808102056384:
                                                    if Q.mass_over_sum_pt_sq > 0.0016935680760070682:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top5 > 13.176732540130615:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.08424501866102219:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.421875:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
