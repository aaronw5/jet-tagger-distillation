"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 64.58% (the formula: 65.33%); same class as the formula for 92.01% of jets.  2607 leaves, depth 21.
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
            if Q.pt_7 > 24.3359375:
                if Q.lam2 > 0.0006050713127478957:
                    if Q.pt_7 > 46.453125:
                        if Q.centroid_offset > 0.017920104786753654:
                            return 'g'   # 69% of the training jets here get this class from the formula
                        else:
                            return 't'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.e2_sq > 0.011442582122981548:
                            return 't'   # 89% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.08361157402396202:
                                return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 56% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt > 941.359375:
                        return 'g'   # 98% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.013077473733574152:
                            if Q.pt_7 > 36.78125:
                                return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 26.8125:
                                    if Q.sum_pt_top5 > 787.953125:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.025441892445087433:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 50% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 133.16432189941406:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 39.828125:
                                    if Q.D2 > 0.2311614751815796:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 82% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 995.34765625:
                    return 'g'   # 72% of the training jets here get this class from the formula
                else:
                    if Q.C2 > 0.07176288962364197:
                        return 't'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.z_dr_0p2_0p4 > 0.023809473030269146:
                            return 'q'   # 81% of the training jets here get this class from the formula
                        else:
                            return 't'   # 68% of the training jets here get this class from the formula
        else:
            if Q.mass > 42.22504234313965:
                if Q.girth2 > 0.009855419397354126:
                    if Q.lam1 > 0.033716289326548576:
                        if Q.tau21 > 0.14597005397081375:
                            if Q.centroid_offset > 0.09908037632703781:
                                if Q.eccentricity > 0.9444245398044586:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.22174516320228577:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 61.9569206237793:
                                    if Q.pt_7 > 50.90625:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.04494989663362503:
                                            if Q.pt_7 > 39.203125:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2600296139717102:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.06837078556418419:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.08737979829311371:
                                                if Q.pt_7 > 42.484375:
                                                    if Q.tau21 > 0.19513912498950958:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.3260975480079651:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.08455083519220352:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 36.953125:
                                if Q.mass > 103.41161727905273:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.03668464533984661:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.09123599156737328:
                                            return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.053605202585458755:
                                    if Q.planar_flow > 0.021655308082699776:
                                        if Q.sum_pt > 358.265625:
                                            if Q.e2 > 0.0842021070420742:
                                                if Q.lam1 > 0.04083854332566261:
                                                    if Q.C2 > 0.026444800198078156:
                                                        if Q.sum_pt_top5 > 350.90625:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 26.4921875:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04991188459098339:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 102.92680358886719:
                                                return 'q'   # 41% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 126.61042022705078:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9842008948326111:
                                            if Q.lam1 > 0.04012974724173546:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.21744222939014435:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03398669883608818:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 50.890625:
                            if Q.lam2 > 0.0010767716448754072:
                                return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 113.0204963684082:
                                    if Q.tau32 > 0.37203195691108704:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.526121973991394:
                                        if Q.e2 > 0.06120065227150917:
                                            if Q.C2 > 0.07309944927692413:
                                                return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0240480350330472:
                                                    if Q.sum_pt_top5 > 461.09375:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.43268895149231:
                                                if Q.tau21 > 0.5158603191375732:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.8427853286266327:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 57.75:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 374.125:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.8456398248672485:
                                                    if Q.C2 > 0.06717995926737785:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.15450473874807358:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.02661432232707739:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 61.421875:
                                                if Q.mass > 88.2513656616211:
                                                    if Q.centroid_offset > 0.02160552516579628:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.685704946517944:
                                                    if Q.centroid_offset > 0.014991967007517815:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.717379570007324:
                                if Q.centroid_offset > 0.023540373891592026:
                                    if Q.planar_flow > 0.09297210723161697:
                                        if Q.z_7 > 0.052452340722084045:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.6015625:
                                            if Q.pt_7 > 38.96875:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.01799464412033558:
                                                    if Q.LHA > 0.3477974683046341:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.2567773908376694:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.02174829225987196:
                                        if Q.tau21 > 0.07941355928778648:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 41.0:
                                            if Q.z_dr_0p05_0p1 > 0.6462611854076385:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 51.68924140930176:
                                    if Q.z_7 > 0.02897490467876196:
                                        if Q.lam1 > 0.027400379069149494:
                                            if Q.e2 > 0.07824161648750305:
                                                if Q.sum_pt > 721.0078125:
                                                    if Q.tau21 > 0.08472997322678566:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 38.546875:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 62.64599609375:
                                                        if Q.pt_7 > 45.109375:
                                                            if Q.tau32 > 0.38168659806251526:
                                                                if Q.lam1 > 0.031203328631818295:
                                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.07206949591636658:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 0.7623323798179626:
                                                                return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.0472616758197546:
                                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 38.09375:
                                                    if Q.LHA > 0.4289858788251877:
                                                        if Q.lam1 > 0.0294240927323699:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.07069297134876251:
                                                                return 't'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04542280733585358:
                                                        if Q.e2 > 0.06607166677713394:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.07396605610847473:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 475.359375:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.028774751350283623:
                                                                return 'q'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0115339788608253:
                                                if Q.pt_7 > 45.671875:
                                                    if Q.e2 > 0.05359349027276039:
                                                        if Q.sum_pt_top3 > 469.28125:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.0254351869225502:
                                                                if Q.e2 > 0.0741506963968277:
                                                                    return 't'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.9738878905773163:
                                                            if Q.C2 > 0.07742524519562721:
                                                                return 't'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.000224008435907308:
                                                                    if Q.sum_pt_top2 > 264.4375:
                                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 785.265625:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 56.12882995605469:
                                                        if Q.e2 > 0.028657937422394753:
                                                            if Q.sum_pt > 752.2421875:
                                                                if Q.pt_7 > 43.203125:
                                                                    if Q.e2 > 0.049326078966259956:
                                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.024441417306661606:
                                                            if Q.girth2 > 0.027398107573390007:
                                                                return 't'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 27.625:
                                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0006708406435791403:
                                                                return 't'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.04451988823711872:
                                                                    if Q.pt_7 > 38.03125:
                                                                        if Q.width > 0.014122398104518652:
                                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.lam2 > 0.0002951946953544393:
                                                                                return 't'   # 81% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.15261350572109222:
                                                    if Q.pt_7 > 38.65625:
                                                        if Q.D2 > 0.9287894070148468:
                                                            if Q.C2 > 0.07133528590202332:
                                                                if Q.tau32 > 0.40746931731700897:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.0480497982352972:
                                                                    if Q.z_dr_0_0p05 > 0.15470296889543533:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.planar_flow > 0.6363871395587921:
                                                                            return 't'   # 75% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.04738491214811802:
                                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.010264720767736435:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.18936610966920853:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04851688630878925:
                                                            if Q.width > 0.010589470621198416:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.33122293651103973:
                                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0_0p05 > 0.04164671525359154:
                                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0243307463824749:
                                            if Q.tau32 > 0.385465070605278:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03425986133515835:
                                                if Q.z_4 > 0.031708721071481705:
                                                    if Q.lam1 > 0.021784492768347263:
                                                        if Q.e2 > 0.06786279007792473:
                                                            return 't'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.594190835952759:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.0954410694539547:
                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.0832874178886414:
                                                        if Q.girth2_top3 > 0.004185885190963745:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.038094405084848404:
                                        if Q.planar_flow > 0.12205936387181282:
                                            if Q.centroid_offset > 0.04106277599930763:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.4636748880147934:
                                                    if Q.mass > 45.319339752197266:
                                                        if Q.D2 > 1.3123685121536255:
                                                            return 't'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.04827193729579449:
                                                                if Q.width > 0.013529584277421236:
                                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_0 > 119.5:
                                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.2127051204442978:
                                                            if Q.C2 > 0.0777650848031044:
                                                                if Q.pt_7 > 27.0078125:
                                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.010862227994948626:
                                                        if Q.pt_7 > 27.0546875:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 45.2858772277832:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.8628809750080109:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.07807957381010056:
                                                            if Q.tau32 > 0.2768339216709137:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.08834976330399513:
                                                if Q.tau21 > 0.32910677790641785:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 5.872337818145752:
                                                        if Q.lam2 > 0.00034362360020168126:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.1948828399181366:
                                                                if Q.lam1 > 0.014962377492338419:
                                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top3 > 0.017703278921544552:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top2 > 159.625:
                                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 511.953125:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.04791687801480293:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.06372005119919777:
                                                            if Q.C2 > 0.07228875532746315:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.7422188520431519:
                                                                    return 't'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.010526083875447512:
                                                                return 't'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.07915031537413597:
                                                                    return 'g'   # 35% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.15085458010435104:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 98% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.04802870936691761:
                        if Q.tau32 > 0.4081386476755142:
                            if Q.pt_7 > 38.609375:
                                if Q.width > 0.009566505439579487:
                                    if Q.z_dr_0p05_0p1 > 0.5527028739452362:
                                        return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 58.027259826660156:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.15699338167905807:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 317.84375:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.04906637221574783:
                                            if Q.girth2 > 0.009440294932574034:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 417.5625:
                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 33.65625:
                                        if Q.n_dr_0_0p05 > 0.5:
                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11394069716334343:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 91% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 34.96875:
                                if Q.e2 > 0.0500531829893589:
                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 54.35521697998047:
                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 49.37586212158203:
                                    if Q.e2_sq > 0.009458765387535095:
                                        return 't'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 61% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 819.125:
                            if Q.D2 > 0.7996246218681335:
                                if Q.pt_7 > 25.8671875:
                                    return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 39.890625:
                                if Q.D2 > 1.118664264678955:
                                    if Q.eccentricity > 0.9077453911304474:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 65.38776779174805:
                                        if Q.z_dr_0p05_0p1 > 0.7060071229934692:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.023085711523890495:
                                    if Q.dr_7 > 0.03831023536622524:
                                        if Q.max_dr > 0.10722023248672485:
                                            if Q.mass > 66.68017196655273:
                                                if Q.pt_7 > 33.3125:
                                                    if Q.e2 > 0.0413201879709959:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.08216583728790283:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_0 > 231.5:
                                        return 'q'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.04874814487993717:
                    if Q.log_sum_pt > 5.7672951221466064:
                        if Q.mass > 34.8140983581543:
                            if Q.centroid_offset > 0.05771955847740173:
                                if Q.pt_7 > 41.359375:
                                    if Q.C2 > 0.040372252464294434:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                return 't'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.06834707781672478:
                                if Q.girth2 > 0.011217796709388494:
                                    if Q.width > 0.01297625619918108:
                                        return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.29378946125507355:
                                            if Q.sum_pt > 441.03125:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 340.65625:
                                        if Q.pt_7 > 41.65625:
                                            return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.27860091626644135:
                                                if Q.centroid_offset > 0.09070658311247826:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.5302224159240723:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_4 > 0.10161368176341057:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.8944534957408905:
                                    if Q.sum_pt > 372.96875:
                                        if Q.tau32 > 0.5188387334346771:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.22462282329797745:
                                        if Q.mass_over_sum_pt > 0.08308061957359314:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 100% of the training jets here get this class from the formula
                    else:
                        if Q.girth2 > 0.015230422839522362:
                            if Q.sum_pt > 279.0546875:
                                if Q.lam2 > 0.0012665981194004416:
                                    return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.4375803619623184:
                                    return 't'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.mass > 38.34031295776367:
                        if Q.centroid_offset > 0.021833867765963078:
                            if Q.C2 > 0.03931609354913235:
                                if Q.tau32 > 0.3902910649776459:
                                    if Q.z_dr_0p1_0p2 > 0.49004560708999634:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_7 > 0.13774312287569046:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.047513529658317566:
                                                if Q.lam2 > 0.00030397823138628155:
                                                    if Q.centroid_offset > 0.039563464000821114:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.07300766184926033:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.3671875:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                return 't'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.C2 > 0.03300073556602001:
                                if Q.tau32 > 0.26177090406417847:
                                    if Q.sum_pt > 418.25:
                                        return 'g'   # 41% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.0789904035627842:
                                            if Q.pt_7 > 28.359375:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 29.8359375:
                                    return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 5.965545654296875:
                            if Q.centroid_offset > 0.027582519687712193:
                                if Q.tau21 > 0.29667459428310394:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.tau32 > 0.16394180059432983:
                                if Q.mass > 36.58046531677246:
                                    if Q.tau21 > 0.601457953453064:
                                        return 't'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03984306566417217:
                                        if Q.LHA > 0.33027204871177673:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 28.1171875:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                return 't'   # 56% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.245094299316406:
            if Q.girth2 > 0.006661437917500734:
                if Q.centroid_offset > 0.028851264156401157:
                    if Q.max_dr > 0.12786699086427689:
                        if Q.C2 > 0.04651923105120659:
                            if Q.pt_7 > 36.328125:
                                if Q.D2 > 1.4354013204574585:
                                    if Q.pt_7 > 41.109375:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.053673384711146355:
                                            if Q.tau32 > 0.23302005976438522:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 45% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.15329559892416:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 456.3046875:
                                    if Q.sum_pt_top5 > 583.765625:
                                        if Q.lam2 > 0.0003532326518325135:
                                            if Q.sum_pt > 802.734375:
                                                return 'g'   # 28% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0071364156901836395:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.3195393085479736:
                                                if Q.log_sum_pt > 6.720502853393555:
                                                    return 'g'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 40% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 9.415158638148569e-05:
                                            if Q.max_dr > 0.1719178631901741:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03679715655744076:
                                                    if Q.lam1 > 0.006771471351385117:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 488.1953125:
                                                if Q.D2 > 2.627708673477173:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.263710021972656:
                                                    return 't'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.8873011469841003:
                                        return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.62872314453125:
                                if Q.lam2 > 0.0002590588264865801:
                                    return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.21477222442626953:
                                        if Q.pt_7 > 38.703125:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.0390625:
                                                if Q.log_sum_pt > 6.743542671203613:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.03600950725376606:
                                            if Q.pt_7 > 35.171875:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.22872549295425415:
                                    if Q.centroid_offset > 0.03655242919921875:
                                        if Q.z_7 > 0.07758121192455292:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.6640625:
                                            if Q.z_dr_0p1_0p2 > 0.20152242481708527:
                                                return 't'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00016329874051734805:
                                        if Q.log_sum_pt > 6.026054620742798:
                                            if Q.pt_7 > 45.765625:
                                                if Q.tau21 > 0.15224960446357727:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 98% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.007648436352610588:
                                            if Q.pt_7 > 45.21875:
                                                if Q.max_dr > 0.1914612352848053:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.030446426011621952:
                                                if Q.sum_pt > 568.15625:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.53949499130249:
                                                    if Q.sum_pt_top5 > 589.828125:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.040531009435653687:
                            if Q.tau21 > 0.2845359593629837:
                                if Q.eccentricity > 0.9502268135547638:
                                    return 't'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 44.671875:
                                    if Q.z_dr_0p05_0p1 > 0.874022901058197:
                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.473877415060997:
                                            return 'g'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.1943352222442627:
                                        if Q.sum_pt_top5 > 345.984375:
                                            return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 623.484375:
                                            if Q.centroid_offset > 0.047439321875572205:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.17721343040466309:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 531.3125:
                                if Q.pt_7 > 27.609375:
                                    if Q.z_dr_0p1_0p2 > 0.24014461785554886:
                                        if Q.centroid_offset > 0.03619664907455444:
                                            if Q.tau21 > 0.11308502405881882:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.007131605176255107:
                                                if Q.max_dr > 0.11820293590426445:
                                                    if Q.LHA > 0.33122238516807556:
                                                        return 't'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.037991685792803764:
                                            if Q.pt_7 > 35.484375:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 665.6796875:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.671875:
                                    if Q.max_dr > 0.10855606570839882:
                                        if Q.tau21 > 0.2157692313194275:
                                            if Q.mass > 37.905120849609375:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03400816023349762:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.06199305318295956:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005209489318076521:
                                            return 't'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 424.65625:
                                        if Q.C2 > 0.04240876063704491:
                                            return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.23070676624774933:
                        if Q.sum_pt > 787.2265625:
                            if Q.centroid_offset > 0.014073594473302364:
                                if Q.pt_7 > 36.125:
                                    return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.29238641262054443:
                                        if Q.girth2_top3 > 0.0018949317163787782:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0003162951616104692:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_4 > 0.0513536985963583:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.08436333760619164:
                                            if Q.lam2 > 8.304916264023632e-05:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 93.0330924987793:
                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.00825625378638506:
                                        if Q.dr_7 > 0.24575303494930267:
                                            return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 34.421875:
                                if Q.girth2 > 0.007831209804862738:
                                    if Q.tau21 > 0.1359696462750435:
                                        if Q.tau32 > 0.2435925379395485:
                                            if Q.z_7 > 0.06063346564769745:
                                                if Q.lam1 > 0.008170496672391891:
                                                    if Q.LHA > 0.2699641138315201:
                                                        return 't'   # 43% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.014710776973515749:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.24260059744119644:
                                                            return 't'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.010692326817661524:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.016803869046270847:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.507370710372925:
                                            if Q.girth2 > 0.00839729281142354:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02100292779505253:
                                        if Q.tau21 > 0.16641172766685486:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_7 > 0.2529626339673996:
                                            if Q.tau21 > 0.1547222062945366:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 266.046875:
                                                if Q.centroid_offset > 0.012237478978931904:
                                                    if Q.tau32 > 0.4790306091308594:
                                                        if Q.lam2 > 6.085611312300898e-05:
                                                            return 't'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 477.7578125:
                                    if Q.dr_7 > 0.035177821293473244:
                                        if Q.max_dr > 0.2520599514245987:
                                            if Q.pt_7 > 32.546875:
                                                if Q.z_7 > 0.049530548974871635:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.624791145324707:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.26547375321388245:
                                                    if Q.D2 > 2.0098989009857178:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.29024840891361237:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 29.078125:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.5033347606658936:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 426.0625:
                                        if Q.max_dr > 0.2788124978542328:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.girth2 > 0.006880460539832711:
                            if Q.log_sum_pt > 6.234900712966919:
                                if Q.width > 0.008886747527867556:
                                    if Q.e2 > 0.04707681015133858:
                                        if Q.pt_7 > 28.71875:
                                            if Q.log_sum_pt > 6.357381820678711:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.04938209615647793:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 35.859375:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 64.55215454101562:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 70.09017181396484:
                                            if Q.sum_pt > 900.0078125:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.28125:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 74.82258605957031:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 38.78125:
                                                if Q.e2 > 0.04229697398841381:
                                                    if Q.sum_pt > 624.6875:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.6559716463088989:
                                                            if Q.max_dr > 0.13737531751394272:
                                                                return 't'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.44493260979652405:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.18629786372184753:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.11848826333880424:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 56.993656158447266:
                                        if Q.mass > 96.13386917114258:
                                            if Q.z_7 > 0.029118355363607407:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.1788645163178444:
                                                if Q.centroid_offset > 0.01767238974571228:
                                                    if Q.log_sum_pt > 6.66752815246582:
                                                        if Q.pt_7 > 42.0:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 31.15625:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.007679544389247894:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.62984824180603:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 27.1875:
                                                            if Q.lam1 > 0.00849017221480608:
                                                                if Q.pt_7 > 39.484375:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.00796362804248929:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_0 > 299.5:
                                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0086275446228683:
                                                    if Q.pt_7 > 27.515625:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.6752800941467285:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.14118101447820663:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006984552834182978:
                                                        if Q.pt_7 > 24.5234375:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 711.0390625:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.06152470037341118:
                                                                    if Q.girth2 > 0.008133057970553637:
                                                                        return 't'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 713.71875:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.08289415389299393:
                                                                if Q.girth2_top3 > 0.007622090866789222:
                                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 27.7265625:
                                            if Q.girth2 > 0.007045523729175329:
                                                if Q.max_dr > 0.18358700722455978:
                                                    if Q.lam1 > 0.007778392173349857:
                                                        if Q.tau21 > 0.19955290108919144:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 39.375:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02102596126496792:
                                                            if Q.D2 > 1.1965959072113037:
                                                                if Q.pt_7 > 36.859375:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1475953459739685:
                                                        if Q.centroid_offset > 0.02292164135724306:
                                                            if Q.D2 > 0.7852155864238739:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_over_sum_pt > 0.08565809577703476:
                                                                    return 't'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 35.578125:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.5411728024482727:
                                                                    if Q.tau21 > 0.2186930626630783:
                                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt > 559.578125:
                                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.01290920376777649:
                                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top2 > 271.03125:
                                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0071739694103598595:
                                                            if Q.pt_7 > 31.0078125:
                                                                if Q.centroid_offset > 0.025478661060333252:
                                                                    if Q.max_dr > 0.11516518890857697:
                                                                        if Q.C2 > 0.017462962307035923:
                                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 610.546875:
                                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.13468950986862183:
                                                                    if Q.tau21 > 0.09649863839149475:
                                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.5053817331790924:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.7059156000614166:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 632.390625:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.012286592274904251:
                                                    if Q.z_dr_0p1_0p2 > 0.31073765456676483:
                                                        if Q.tau21 > 0.20635828375816345:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.08496370911598206:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10893891379237175:
                                                        if Q.z_dr_0p1_0p2 > 0.292116180062294:
                                                            if Q.z_dr_0p05_0p1 > 0.5652780532836914:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.7843405604362488:
                                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.22525829076766968:
                                                                if Q.z_dr_0p05_0p1 > 0.20750316977500916:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 5.6032427892205305e-05:
                                                            if Q.width > 0.006977307144552469:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.00013531912554753944:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.06472711265087128:
                                                if Q.pt_7 > 24.515625:
                                                    if Q.sum_pt > 547.4140625:
                                                        if Q.girth2 > 0.008216255344450474:
                                                            return 't'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 654.9921875:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 26.8984375:
                                                    if Q.z_7 > 0.04505131579935551:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.30099479854106903:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 32.359375:
                                    if Q.max_dr > 0.14354854822158813:
                                        if Q.tau21 > 0.17422085255384445:
                                            if Q.sum_pt > 396.25:
                                                if Q.max_dr > 0.19140832126140594:
                                                    if Q.centroid_offset > 0.016854574903845787:
                                                        if Q.pt_0 > 95.9375:
                                                            return 't'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 37.03125:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.017364145256578922:
                                                        if Q.tau32 > 0.39895111322402954:
                                                            if Q.log_sum_pt > 6.085850715637207:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0174269899725914:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 37.1875:
                                                    if Q.max_dr > 0.17199458181858063:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.0211239755153656:
                                            if Q.sum_pt > 392.40625:
                                                if Q.planar_flow > 0.8325150012969971:
                                                    if Q.pt_0 > 92.625:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0071244253776967525:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0001346132848993875:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.07747035101056099:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.008251276332885027:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11907690390944481:
                                                if Q.girth2 > 0.008695490192621946:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9835929274559021:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0071271569468081:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 401.8984375:
                                        if Q.tau21 > 0.23762667179107666:
                                            if Q.pt_7 > 28.7265625:
                                                if Q.max_dr > 0.1722700521349907:
                                                    if Q.sum_pt > 454.875:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02200643066316843:
                                                        return 't'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.117976665496826:
                                                    if Q.C2 > 0.053888674825429916:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.10564190149307251:
                                                if Q.e2 > 0.044368717819452286:
                                                    if Q.pt_7 > 30.3203125:
                                                        if Q.lam1 > 0.007866671774536371:
                                                            if Q.tau21 > 0.14344467967748642:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 29.6875:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04009445011615753:
                                if Q.lam2 > 0.00025957112666219473:
                                    if Q.mass > 32.93795394897461:
                                        if Q.D2 > 0.8281461298465729:
                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.6229440569877625:
                                                if Q.e2 > 0.044561175629496574:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 31.625:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 682.78125:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.673584461212158:
                                        if Q.max_dr > 0.10064943134784698:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.006741052260622382:
                                                if Q.width > 0.006814987398684025:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.009049438405781984:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.00905573833733797:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 866.234375:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.01236625388264656:
                                            if Q.sum_pt > 666.015625:
                                                if Q.girth2 > 0.006730257300660014:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.08042716979980469:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.7063938677310944:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006773393834009767:
                                                        if Q.lam2 > 0.0001359497691737488:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 60.086448669433594:
                                                if Q.z_dr_0p1_0p2 > 0.05979139357805252:
                                                    if Q.width > 0.0068029980175197124:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.01507678721100092:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9976920485496521:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.0067552507389336824:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006806436460465193:
                                                        if Q.log_sum_pt > 6.6201770305633545:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1396508663892746:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006806414341554046:
                                                        if Q.eccentricity > 0.9767773747444153:
                                                            if Q.e2 > 0.04067992977797985:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 694.6953125:
                                    if Q.z_dr_0p1_0p2 > 0.2656370848417282:
                                        if Q.z_dr_0p05_0p1 > 0.3711991459131241:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.25977572798728943:
                                        if Q.z_dr_0p05_0p1 > 0.6989441812038422:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 7.669942715438083e-05:
                                                return 't'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 27.4609375:
                                            if Q.e2 > 0.02941844053566456:
                                                if Q.mass > 38.80476951599121:
                                                    if Q.centroid_offset > 0.009327891282737255:
                                                        if Q.max_dr > 0.09909028187394142:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 49.326148986816406:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.20068155974149704:
                                                            if Q.z_dr_0_0p05 > 0.3185316324234009:
                                                                if Q.max_dr > 0.15623410791158676:
                                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 36.0625:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.474228143692017:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 79% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.15000691264867783:
                    if Q.girth2 > 0.004923145519569516:
                        if Q.centroid_offset > 0.010989651549607515:
                            if Q.mass > 45.482316970825195:
                                if Q.centroid_offset > 0.031709495931863785:
                                    if Q.pt_7 > 27.5546875:
                                        if Q.log_sum_pt > 6.799916982650757:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0002394710318185389:
                                                if Q.tau21 > 0.18017474561929703:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006071645068004727:
                                                    if Q.D2 > 2.276379108428955:
                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.044646717607975006:
                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.16754894703626633:
                                        if Q.max_dr > 0.29339560866355896:
                                            if Q.log_sum_pt > 6.608327388763428:
                                                if Q.lam2 > 0.00013647043670061976:
                                                    if Q.D2 > 4.468243598937988:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.19545616954565048:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.579977989196777:
                                                if Q.sum_pt > 998.1875:
                                                    if Q.pt_7 > 38.171875:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.005085369339212775:
                                                        if Q.lam2 > 0.0001606426521902904:
                                                            if Q.centroid_offset > 0.023403271101415157:
                                                                if Q.tau21 > 0.20817803591489792:
                                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 47% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.012331714387983084:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2_sq > 0.005150139797478914:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.19063309580087662:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.11954711005091667:
                                                            if Q.centroid_offset > 0.014029629062861204:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 1.127756953239441:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.012310232035815716:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.21576322615146637:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 25.2265625:
                                                    if Q.z_dr_0p05_0p1 > 0.04948563128709793:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00020553624199237674:
                                                            if Q.tau21 > 0.18848903477191925:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.11544273048639297:
                                                                if Q.girth2 > 0.005412028171122074:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.036931490525603294:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 588.75:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.2690019905567169:
                                            if Q.centroid_offset > 0.01420243177562952:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005633594235405326:
                                                    if Q.mass > 53.86904716491699:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.005997331580147147:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.016638170927762985:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005473400000482798:
                                                    if Q.max_dr > 0.15902022272348404:
                                                        if Q.girth > 0.061711620539426804:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.014616037253290415:
                                                        if Q.lam1 > 0.005130381789058447:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 27.8828125:
                                    if Q.mass > 37.13747787475586:
                                        if Q.centroid_offset > 0.014587417710572481:
                                            if Q.girth2 > 0.005864203674718738:
                                                if Q.e2 > 0.025961737148463726:
                                                    if Q.z_dr_0p05_0p1 > 0.07187742367386818:
                                                        if Q.tau21 > 0.16391663253307343:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 41.45534706115723:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04553624987602234:
                                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 6.390246926457621e-05:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0062591009773314:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.24159687012434006:
                                                    if Q.tau21 > 0.19122420251369476:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.018557299859821796:
                                                        if Q.z_dr_0p05_0p1 > 0.14654634147882462:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.427789688110352:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.14625492691993713:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.17691264301538467:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.width > 0.005170916439965367:
                                                                if Q.girth2 > 0.005640565184876323:
                                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.13677235692739487:
                                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.17478832602500916:
                                                if Q.tau21 > 0.14101628214120865:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.00599451526068151:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.22081352025270462:
                                            if Q.pt_7 > 31.1015625:
                                                if Q.LHA > 0.23883916437625885:
                                                    if Q.lam2 > 0.00047550130693707615:
                                                        if Q.mass > 32.80256652832031:
                                                            if Q.max_dr > 0.1926933377981186:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p1_0p2 > 0.12786294519901276:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.006078365491703153:
                                                            return 't'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 33.409217834472656:
                                                    if Q.width > 0.00573495333082974:
                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 37.765625:
                                                if Q.girth2 > 0.005643154727295041:
                                                    if Q.z_4 > 0.10768990591168404:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006039034342393279:
                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.048509448766708374:
                                                        if Q.pt_7 > 33.90625:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 488.09375:
                                                            return 't'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.473944902420044:
                                        if Q.e2 > 0.022400478832423687:
                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.0176407340914011:
                                                return 't'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 509.1328125:
                                            if Q.tau21 > 0.22523533552885056:
                                                if Q.z_7 > 0.03976891562342644:
                                                    if Q.max_dr > 0.18827864527702332:
                                                        if Q.mass > 37.1528205871582:
                                                            return 't'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.305390119552612:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 35.74130058288574:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0058754789642989635:
                                if Q.max_dr > 0.169140063226223:
                                    if Q.mass > 58.80994987487793:
                                        if Q.z_dr_0p1_0p2 > 0.17995136231184006:
                                            if Q.width > 0.006067771231755614:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.8828125:
                                            if Q.mass > 35.13447189331055:
                                                if Q.z_dr_0p1_0p2 > 0.1574498564004898:
                                                    if Q.girth2 > 0.006179455667734146:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.7777793705463409:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.024548844434320927:
                                                        if Q.width > 0.006055706180632114:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.004821743117645383:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 681.3203125:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.06700864806771278:
                                        if Q.centroid_offset > 0.00626008422113955:
                                            if Q.mass > 62.90322303771973:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.006380709353834391:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.4104281961917877:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 63.1854362487793:
                                                if Q.mass_over_sum_pt > 0.07916804030537605:
                                                    if Q.C2 > 0.015261778142303228:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.039704980328679085:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 71.27731704711914:
                                            if Q.lam1 > 0.005961438175290823:
                                                if Q.centroid_offset > 0.004304834408685565:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1593765765428543:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.1598910540342331:
                                                if Q.LHA > 0.26799823343753815:
                                                    if Q.centroid_offset > 0.005585997365415096:
                                                        if Q.log_sum_pt > 6.541297435760498:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 63.99856185913086:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.265968382358551:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.22699681669473648:
                                    if Q.max_dr > 0.2551043629646301:
                                        if Q.log_sum_pt > 6.639763355255127:
                                            if Q.centroid_offset > 0.0031531472923234105:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2961246222257614:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.0053972534369677305:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 28.3515625:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.04564284905791283:
                                            if Q.centroid_offset > 0.005475499900057912:
                                                if Q.lam1 > 0.005289212102070451:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009535443503409624:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.825901746749878:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.07496904954314232:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.24200545996427536:
                                                if Q.girth > 0.04412168823182583:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.03740102797746658:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 68.15685272216797:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.17467355728149414:
                                        if Q.girth2 > 0.005485421745106578:
                                            if Q.centroid_offset > 0.006298116873949766:
                                                if Q.mass > 56.2504940032959:
                                                    if Q.width > 0.005578343756496906:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1923057958483696:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.3655563592910767:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009557289071381092:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 68.91695404052734:
                                                    if Q.pt_7 > 19.8046875:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 75.45862579345703:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.004184676567092538:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00028075701266061515:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.005747310351580381:
                                                            if Q.centroid_offset > 0.00402287277393043:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00022655085194855928:
                                                if Q.z_dr_0_0p05 > 0.7513047158718109:
                                                    if Q.C2 > 0.05312552861869335:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.06381189450621605:
                                                        return 'Z'   # 43% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.948042154312134:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009932330343872309:
                                                        if Q.sum_pt_top3 > 577.03125:
                                                            if Q.z_7 > 0.025738392025232315:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0053511857986450195:
                                                            if Q.z_dr_0p1_0p2 > 0.12192286923527718:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.007136239670217037:
                                                                    if Q.sum_pt_top2 > 494.140625:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1022.08984375:
                                            if Q.pt_7 > 18.4453125:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.153679609298706:
                                                if Q.C2 > 0.06141286343336105:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009613389614969492:
                                                        if Q.e2_sq > 0.005525577813386917:
                                                            if Q.mass > 54.546586990356445:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.018207107670605183:
                            if Q.girth2 > 0.003664114628918469:
                                if Q.max_dr > 0.17055576294660568:
                                    if Q.mass > 40.12919807434082:
                                        if Q.log_sum_pt > 6.909581661224365:
                                            if Q.sum_pt_top2 > 696.0625:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 4.223224401473999:
                                                if Q.log_sum_pt > 6.662044525146484:
                                                    if Q.centroid_offset > 0.03367224894464016:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.0381020400673151:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 43% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.003838895936496556:
                                                    if Q.LHA > 0.2077868953347206:
                                                        if Q.max_dr > 0.18036195635795593:
                                                            if Q.sum_pt > 724.96875:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 23.71875:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau21 > 0.2446308135986328:
                                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2434815987944603:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02000936772674322:
                                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.06964090093970299:
                                                        if Q.centroid_offset > 0.02197176031768322:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.8828125:
                                            if Q.centroid_offset > 0.023791291750967503:
                                                if Q.girth2_top3 > 0.0013061213539913297:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 36% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.004281562054529786:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.04218036122620106:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0039049448678269982:
                                                            if Q.centroid_offset > 0.022048774175345898:
                                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 627.1875:
                                                if Q.max_dr > 0.1946098729968071:
                                                    if Q.centroid_offset > 0.03225846402347088:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.307294130325317:
                                                    if Q.tau21 > 0.2552906274795532:
                                                        return 'g'   # 30% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.024624339304864407:
                                        if Q.LHA > 0.25371597707271576:
                                            if Q.centroid_offset > 0.02563652116805315:
                                                if Q.pt_7 > 25.7421875:
                                                    if Q.max_dr > 0.1522512212395668:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.05676575191318989:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.4836671352386475:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.06077093817293644:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 39.75006675720215:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.030594097450375557:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00018395032384432852:
                                                        return 'Z'   # 38% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004386052489280701:
                                            if Q.max_dr > 0.15759926289319992:
                                                if Q.e2 > 0.029888546094298363:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.004590830998495221:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 707.78125:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0046735904179513454:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02070994209498167:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.021874171681702137:
                                                if Q.mass > 42.27053260803223:
                                                    if Q.max_dr > 0.154824897646904:
                                                        if Q.girth2 > 0.00394931435585022:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 53.674766540527344:
                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1600136235356331:
                                                        if Q.girth > 0.05282606743276119:
                                                            if Q.centroid_offset > 0.020846023224294186:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19974350184202194:
                                    if Q.girth2 > 0.002413027221336961:
                                        if Q.centroid_offset > 0.02000736352056265:
                                            if Q.mass > 35.81017303466797:
                                                if Q.max_dr > 0.20635976642370224:
                                                    if Q.log_sum_pt > 6.912909269332886:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.03190194442868233:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.549002408981323:
                                                                if Q.girth2 > 0.002681007026694715:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 4.348708152770996:
                                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.0011789390118792653:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.9483139514923096:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.23375368118286133:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 32.37654495239258:
                                                        if Q.centroid_offset > 0.022815306670963764:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.22694524377584457:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.0027592930709943175:
                                                    if Q.sum_pt_top3 > 491.65625:
                                                        if Q.width > 0.003476478159427643:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.019110657274723053:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02172738593071699:
                                            if Q.max_dr > 0.2100938931107521:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.2068522423505783:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.26322460174560547:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.23020153492689133:
                                                    if Q.centroid_offset > 0.020609410479664803:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.027724992483854294:
                                        if Q.centroid_offset > 0.032006630674004555:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.1718449965119362:
                                                if Q.girth > 0.04419751279056072:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 490.78125:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00014296998415375128:
                                                    if Q.dr_0 > 0.04534749686717987:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 35.97139549255371:
                                                        if Q.centroid_offset > 0.029170282185077667:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.024256552569568157:
                                            if Q.mass > 40.648786544799805:
                                                if Q.max_dr > 0.1749669536948204:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.08859981596469879:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00013459792535286397:
                                                    if Q.mass > 36.54315376281738:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.003469615592621267:
                                                        if Q.max_dr > 0.17235509306192398:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 51.09472846984863:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.02015361748635769:
                                if Q.max_dr > 0.24060555547475815:
                                    if Q.girth2_top3 > 0.0006298379448708147:
                                        if Q.centroid_offset > 0.011885764077305794:
                                            if Q.girth2 > 0.003308210871182382:
                                                if Q.max_dr > 0.26662780344486237:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.206684410572052:
                                                        if Q.centroid_offset > 0.013249326031655073:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2_sq > 0.0040832096710801125:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.015125984326004982:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.05390976183116436:
                                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2655467391014099:
                                                    if Q.lam1 > 0.0025470582768321037:
                                                        if Q.centroid_offset > 0.014929562341421843:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.016539092175662518:
                                                        if Q.e2_sq > 0.0024426691234111786:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.004246225813403726:
                                                if Q.max_dr > 0.2709488123655319:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.00978720048442483:
                                                        if Q.mass_over_sum_pt > 0.06653222814202309:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 24.6484375:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.05550684966146946:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 3.225624918937683:
                                                    if Q.mass > 55.820913314819336:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3045787215232849:
                                            if Q.girth > 0.02856468502432108:
                                                if Q.log_sum_pt > 6.655017614364624:
                                                    if Q.max_dr > 0.3221729248762131:
                                                        if Q.tau21 > 0.3886931985616684:
                                                            if Q.lam2 > 0.0002438546362100169:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.17925399541854858:
                                                            if Q.pt_7 > 12.98046875:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.02388603612780571:
                                                                if Q.z_4 > 0.06373242847621441:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 23.1328125:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 57.06705856323242:
                                                    if Q.pt_7 > 16.6953125:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.362042099237442:
                                                            if Q.pt_7 > 10.3984375:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.725703954696655:
                                                        if Q.girth > 0.026130964048206806:
                                                            if Q.max_dr > 0.36110472679138184:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.011134423315525055:
                                                                    if Q.mass > 46.92165756225586:
                                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.014793356880545616:
                                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0031849605729803443:
                                                            if Q.log_sum_pt > 6.652669668197632:
                                                                return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 744.578125:
                                                if Q.LHA > 0.1852441057562828:
                                                    if Q.max_dr > 0.27968303859233856:
                                                        if Q.pt_7 > 16.578125:
                                                            if Q.max_dr > 0.28762637078762054:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.012722205370664597:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.06447036191821098:
                                                            if Q.girth > 0.03577359765768051:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.013882394880056381:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.03860555961728096:
                                                                if Q.z_dr_0p2_0p4 > 0.06283622048795223:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 7.025336980819702:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 34.560829162597656:
                                                            if Q.centroid_offset > 0.015117165632545948:
                                                                if Q.max_dr > 0.28301775455474854:
                                                                    if Q.width > 0.002284388872794807:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.0035994144855067134:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 37.36935615539551:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.01024992298334837:
                                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.012680891435593367:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 27.671875:
                                                    if Q.mass > 38.9826545715332:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.03286319226026535:
                                                        if Q.sum_pt > 669.3359375:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 35.13629341125488:
                                        if Q.centroid_offset > 0.0147988460958004:
                                            if Q.width > 0.003998183645308018:
                                                if Q.max_dr > 0.18361067026853561:
                                                    if Q.width > 0.004305834416300058:
                                                        if Q.max_dr > 0.21089694648981094:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2313823103904724:
                                                                if Q.centroid_offset > 0.015839667059481144:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2 > 0.004546846263110638:
                                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.20364712923765182:
                                                            if Q.centroid_offset > 0.015785135328769684:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.22381385415792465:
                                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01677518244832754:
                                                                if Q.sum_pt_top5 > 665.828125:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.16816743463277817:
                                                        if Q.lam1 > 0.004395133350044489:
                                                            if Q.sum_pt_top5 > 576.921875:
                                                                if Q.centroid_offset > 0.01566249504685402:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.4095969498157501:
                                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0005071155028417706:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 59.87740898132324:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.004556230967864394:
                                                                    if Q.max_dr > 0.16058441251516342:
                                                                        if Q.LHA > 0.25613026320934296:
                                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.039859211072325706:
                                                    if Q.width > 0.003392926068045199:
                                                        if Q.max_dr > 0.23027346283197403:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.017264414578676224:
                                                                if Q.mass_over_sum_pt > 0.05786480754613876:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.00021606311929645017:
                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 907.5703125:
                                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1046.703125:
                                                        if Q.mass > 50.19669723510742:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 68.66564178466797:
                                                if Q.sum_pt > 1080.84375:
                                                    if Q.z_7 > 0.014632623177021742:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006867503048852086:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 38.60765838623047:
                                                    if Q.lam2 > 0.00044647253525909036:
                                                        if Q.z_dr_0p2_0p4 > 0.0247357077896595:
                                                            if Q.width > 0.0040284814313054085:
                                                                if Q.e2 > 0.02623418066650629:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.011405258905142546:
                                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.004477940732613206:
                                                                if Q.D2 > 1.9765360355377197:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.994950294494629:
                                                            if Q.pt_7 > 30.1484375:
                                                                if Q.eccentricity > 0.9866879284381866:
                                                                    if Q.mass > 59.93716049194336:
                                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p2_0p4 > 0.060720473527908325:
                                                                if Q.centroid_offset > 0.012483545113354921:
                                                                    if Q.width > 0.004316464764997363:
                                                                        if Q.e2_sq > 0.004457489820197225:
                                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.z_dr_0p2_0p4 > 0.07613791525363922:
                                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.sum_pt_top2 > 414.5625:
                                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 64.25753784179688:
                                                                        if Q.pt_7 > 28.84375:
                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.0017916916985996068:
                                                                    if Q.girth2_top3 > 0.00040932968840934336:
                                                                        if Q.centroid_offset > 0.013204309158027172:
                                                                            if Q.mass > 60.49872589111328:
                                                                                if Q.max_dr > 0.1690739169716835:
                                                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.e2_sq > 0.004529519937932491:
                                                                                    if Q.max_dr > 0.18673689663410187:
                                                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.max_dr > 0.18135587126016617:
                                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.lam1 > 0.002905367757193744:
                                                                                return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.z_7 > 0.04204101301729679:
                                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.mass > 40.90157127380371:
                                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.19263643771409988:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.0003057185385841876:
                                                        if Q.pt_7 > 24.4765625:
                                                            if Q.eccentricity > 0.9234307110309601:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top3 > 0.0007258830300997943:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_7 > 0.048647643998265266:
                                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.010002204217016697:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.03160943649709225:
                                                            if Q.eccentricity > 0.9818695187568665:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.00921573769301176:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.012917784973978996:
                                            if Q.sum_pt > 534.8671875:
                                                if Q.dr_0 > 0.014637768734246492:
                                                    if Q.centroid_offset > 0.014786162413656712:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 31.63564682006836:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.643962860107422:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.765625:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.037827061489224434:
                                                if Q.eccentricity > 0.9855976104736328:
                                                    if Q.centroid_offset > 0.008310517761856318:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.048387039452791214:
                                                        if Q.mass > 32.44475173950195:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_0 > 110.09375:
                                                            if Q.sum_pt_top5 > 607.09375:
                                                                if Q.eccentricity > 0.965734213590622:
                                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.011050961911678314:
                                                    if Q.z_dr_0_0p05 > 0.9602380990982056:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.044604817405343056:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 42.1884765625:
                                    if Q.girth > 0.015461104456335306:
                                        if Q.log_sum_pt > 7.006096124649048:
                                            return 'Z'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 4.0827195334713906e-05:
                                                if Q.sum_pt > 971.25:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9811470210552216:
                                            if Q.mass > 45.33012390136719:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 18.6171875:
                                        if Q.mass > 38.84885787963867:
                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1055.6640625:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 32.390625:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.024785758927464485:
                        if Q.girth2 > 0.00483433622866869:
                            if Q.max_dr > 0.1125669926404953:
                                if Q.z_dr_0p05_0p1 > 0.2545912563800812:
                                    if Q.mass > 35.618099212646484:
                                        if Q.mass > 44.821109771728516:
                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 27.6328125:
                                                if Q.max_dr > 0.12610553950071335:
                                                    if Q.LHA > 0.30114153027534485:
                                                        if Q.tau21 > 0.17726580053567886:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 34.046875:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.6272960305213928:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1338171362876892:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02901068702340126:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.006101624108850956:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0001281217555515468:
                                                                if Q.e2 > 0.032396506518125534:
                                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07085593044757843:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 24.90625:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.005868548294529319:
                                            if Q.tau21 > 0.29381293058395386:
                                                if Q.z_7 > 0.07083456963300705:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.765625:
                                                    if Q.centroid_offset > 0.040526287630200386:
                                                        return 't'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.609375:
                                                if Q.centroid_offset > 0.031352076679468155:
                                                    if Q.max_dr > 0.1277627944946289:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.02880961913615465:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13256515562534332:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.10044894367456436:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.944552093744278:
                                                    return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2395220473408699:
                                        return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.033720703795552254:
                                    if Q.mass > 38.30996131896973:
                                        if Q.lam2 > 3.3828475352493115e-05:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03778526559472084:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00014093975914875045:
                                            if Q.pt_7 > 32.6875:
                                                if Q.centroid_offset > 0.04783420450985432:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.5560785531997681:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 42% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.04156803339719772:
                                                if Q.girth > 0.07391267642378807:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.09736820682883263:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005329126259312034:
                                                    if Q.pt_7 > 33.21875:
                                                        if Q.e2 > 0.031478751450777054:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 49.376943588256836:
                                        if Q.lam2 > 8.116626122500747e-05:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.09878172725439072:
                                                if Q.width > 0.005955465137958527:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.028369147330522537:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00023510414030170068:
                                            if Q.C2 > 0.02189022209495306:
                                                if Q.girth2 > 0.006283218506723642:
                                                    if Q.dr_0 > 0.08623571693897247:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02902923710644245:
                                                        if Q.mass > 36.517757415771484:
                                                            if Q.z_dr_0p1_0p2 > 0.065023522824049:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.310428857803345:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 35.1875:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.07878567278385162:
                                                if Q.max_dr > 0.09639769420027733:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.10438622534275055:
                                                    if Q.LHA > 0.3051532208919525:
                                                        if Q.e2 > 0.03433595225214958:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 29.71875:
                                                            if Q.centroid_offset > 0.027148477733135223:
                                                                if Q.D2 > 0.7222206592559814:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.planar_flow > 0.056283898651599884:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9814760982990265:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.017057511024177074:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.03201071172952652:
                                if Q.max_dr > 0.11575163155794144:
                                    if Q.width > 0.003846663748845458:
                                        if Q.pt_7 > 24.6796875:
                                            if Q.centroid_offset > 0.033160848543047905:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.16466493159532547:
                                                    if Q.eccentricity > 0.9724485576152802:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 38% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9660178124904633:
                                            if Q.mass > 36.08504104614258:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03644704446196556:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.07596065476536751:
                                        if Q.tau21 > 0.15979185700416565:
                                            if Q.sum_pt > 721.6171875:
                                                if Q.pt_7 > 38.6875:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00020417050836840644:
                                                    if Q.sum_pt > 612.8125:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.644928216934204:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.560654878616333:
                                            if Q.centroid_offset > 0.0353271197527647:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 41.09214973449707:
                                    if Q.max_dr > 0.1301741600036621:
                                        if Q.lam1 > 0.003959371708333492:
                                            if Q.centroid_offset > 0.02768903225660324:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 47.99942970275879:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13691844791173935:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.028144991025328636:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00017141754506155849:
                                            if Q.C2 > 0.022996815852820873:
                                                if Q.e2 > 0.026824604719877243:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 50.17958641052246:
                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03000648319721222:
                                                    if Q.dr_0 > 0.05601515248417854:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.9630022048950195:
                                        if Q.centroid_offset > 0.02958262339234352:
                                            if Q.max_dr > 0.12951059639453888:
                                                if Q.lam1 > 0.004086402012035251:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.17524871230125427:
                                            if Q.max_dr > 0.13177210837602615:
                                                if Q.centroid_offset > 0.027215734124183655:
                                                    if Q.girth2 > 0.004253617953509092:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 504.53125:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00037962329224683344:
                                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.6737120151519775:
                                                    return 'Z'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.027255401946604252:
                                                if Q.eccentricity > 0.9335254728794098:
                                                    if Q.max_dr > 0.10512841492891312:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.41854028403759:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.11828305572271347:
                                                        return 'Z'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.0026827623369172215:
                            if Q.max_dr > 0.12343917414546013:
                                if Q.z_dr_0p05_0p1 > 0.6104320585727692:
                                    if Q.width > 0.006061566760763526:
                                        if Q.centroid_offset > 0.010702425613999367:
                                            if Q.e2 > 0.0352820698171854:
                                                if Q.girth2 > 0.006261587608605623:
                                                    if Q.sum_pt > 684.0859375:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 29.0390625:
                                                            if Q.D2 > 0.7311452031135559:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.006374184740707278:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.017628888599574566:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13228555023670197:
                                                            if Q.eccentricity > 0.9747186303138733:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 65.60940170288086:
                                                if Q.girth2 > 0.0064039574936032295:
                                                    if Q.centroid_offset > 0.0036833900958299637:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.08083625137805939:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005767879309132695:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 73.4766731262207:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13103751093149185:
                                                    if Q.girth2 > 0.006370398681610823:
                                                        if Q.sum_pt > 719.546875:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.09752563387155533:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.03792251646518707:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.036221591755747795:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.006504860240966082:
                                                        if Q.centroid_offset > 0.005970049183815718:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.016227740794420242:
                                            if Q.lam1 > 0.005136070074513555:
                                                if Q.e2 > 0.03266643360257149:
                                                    if Q.width > 0.005735931918025017:
                                                        if Q.e2 > 0.035271165892481804:
                                                            if Q.max_dr > 0.130002960562706:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0054190149530768394:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.029568086378276348:
                                                            if Q.sum_pt_top5 > 623.125:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 48.743900299072266:
                                                    if Q.centroid_offset > 0.020375018939375877:
                                                        if Q.girth2 > 0.004763098433613777:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.15166441351175308:
                                                            if Q.max_dr > 0.134330615401268:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.04395676590502262:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.004934185184538364:
                                                        if Q.centroid_offset > 0.02319445088505745:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.05485144816339016:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1320965215563774:
                                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.034146953374147415:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005762429675087333:
                                                    if Q.e2 > 0.03353982977569103:
                                                        if Q.girth2 > 0.00590477860532701:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.005553333321586251:
                                                        if Q.e2 > 0.032457711175084114:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 34.3098087310791:
                                        if Q.z_dr_0p05_0p1 > 0.3890502154827118:
                                            if Q.max_dr > 0.14172444492578506:
                                                if Q.tau21 > 0.1754135563969612:
                                                    if Q.girth > 0.07050299271941185:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01826830953359604:
                                                            if Q.LHA > 0.2732015997171402:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.015132967848330736:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0059513377491384745:
                                                            if Q.mass > 57.19968223571777:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.13542067259550095:
                                                    if Q.LHA > 0.2910705804824829:
                                                        if Q.centroid_offset > 0.007460690103471279:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01791357807815075:
                                                            if Q.mass > 49.649038314819336:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.005178514402359724:
                                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 66.41581344604492:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.0684531144797802:
                                                                    if Q.e2 > 0.03430142253637314:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.8430585861206055:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.006425382802262902:
                                                            if Q.D2 > 0.9198408424854279:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.016854051500558853:
                                                                if Q.girth2 > 0.005935985129326582:
                                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.004918325459584594:
                                                                        if Q.max_dr > 0.1332796961069107:
                                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1091.2578125:
                                                if Q.mass > 74.7542610168457:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.03181769140064716:
                                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00027063293964602053:
                                                    if Q.girth2 > 0.0061678655911237:
                                                        if Q.D2 > 1.135217308998108:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01630344893783331:
                                                                return 't'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.11569807305932045:
                                                            if Q.C2 > 0.05563865415751934:
                                                                if Q.mass > 38.35478210449219:
                                                                    if Q.LHA > 0.2691122144460678:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.058487189933657646:
                                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.021445922553539276:
                                                                    if Q.mass > 42.93630409240723:
                                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.2175564020872116:
                                                                        if Q.girth > 0.06427917629480362:
                                                                            if Q.centroid_offset > 0.014560990501195192:
                                                                                if Q.max_dr > 0.1374807432293892:
                                                                                    return 'Z'   # 50% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 75.02877807617188:
                                                        if Q.centroid_offset > 0.005289977649226785:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01752660609781742:
                                                            if Q.lam1 > 0.004888895666226745:
                                                                if Q.mass > 50.45772933959961:
                                                                    if Q.max_dr > 0.14279094338417053:
                                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.019136120565235615:
                                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.004722905578091741:
                                                                    if Q.e2_sq > 0.004335753154009581:
                                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.21509859710931778:
                                                                if Q.lam1 > 0.00634783529676497:
                                                                    if Q.z_dr_0p1_0p2 > 0.23285356163978577:
                                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt > 732.59375:
                                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 68.82437133789062:
                                                                        if Q.centroid_offset > 0.009651373140513897:
                                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 41.710243225097656:
                                                                            if Q.centroid_offset > 0.013658915646374226:
                                                                                if Q.lam1 > 0.005420401692390442:
                                                                                    if Q.max_dr > 0.14499742537736893:
                                                                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 100% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.pt_7 > 26.3125:
                                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.z_dr_0p1_0p2 > 0.15240435302257538:
                                                                                    return 't'   # 69% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 37.18854904174805:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.008596655447036028:
                                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.22034098207950592:
                                            if Q.pt_7 > 31.5625:
                                                if Q.C2 > 0.056242115795612335:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.011996258050203323:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.24009718000888824:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.271401882171631:
                                                    if Q.pt_7 > 25.90625:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.08214513212442398:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.428534269332886:
                                                return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 34.44369316101074:
                                    if Q.mass > 76.20438385009766:
                                        if Q.pt_7 > 25.671875:
                                            if Q.sum_pt > 1182.734375:
                                                if Q.tau32 > 0.6459895074367523:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 78.3415298461914:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0038508924189954996:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 34.328125:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 79.11038208007812:
                                                if Q.pt_7 > 14.765625:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.00648614508099854:
                                            if Q.e2 > 0.038368623703718185:
                                                if Q.eccentricity > 0.9690923690795898:
                                                    if Q.e2 > 0.039874257519841194:
                                                        if Q.sum_pt > 885.453125:
                                                            if Q.z_7 > 0.043198112398386:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.planar_flow > 0.010505971033126116:
                                                                    if Q.z_dr_0p1_0p2 > 0.01862992299720645:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01975639909505844:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 784.171875:
                                                                    if Q.girth2 > 0.006626585964113474:
                                                                        if Q.planar_flow > 0.010424954816699028:
                                                                            if Q.mass_over_sum_pt > 0.08127360045909882:
                                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.006604144116863608:
                                                            if Q.pt_0 > 222.6875:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_7 > 0.07467684522271156:
                                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 881.984375:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.015030449721962214:
                                                                    if Q.sum_pt > 676.1015625:
                                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.9828820526599884:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.0418930072337389:
                                                            if Q.mass > 60.327165603637695:
                                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0006306593713816255:
                                                                    if Q.pt_7 > 36.859375:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 291.59375:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.041065702214837074:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0032730625243857503:
                                                if Q.centroid_offset > 0.018720555119216442:
                                                    if Q.mass > 54.21177673339844:
                                                        if Q.max_dr > 0.10581662878394127:
                                                            if Q.width > 0.005707206204533577:
                                                                if Q.e2 > 0.03702125512063503:
                                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.021756322123110294:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 57.28944206237793:
                                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9644973576068878:
                                                                if Q.mass > 61.96690368652344:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.20454896241426468:
                                                            if Q.girth > 0.07526916265487671:
                                                                if Q.e2 > 0.03843394108116627:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 34.359375:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1177857555449009:
                                                                    if Q.LHA > 0.2956444174051285:
                                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.9718008935451508:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau21 > 0.09490776807069778:
                                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0002323903754586354:
                                                                if Q.tau21 > 0.09570303186774254:
                                                                    if Q.mass_over_sum_pt > 0.07368289679288864:
                                                                        if Q.max_dr > 0.10871337354183197:
                                                                            if Q.girth2 > 0.006158073432743549:
                                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau21 > 0.16988926380872726:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 687.5:
                                                                                if Q.centroid_offset > 0.0217480156570673:
                                                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 44.33834266662598:
                                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam2 > 0.000350260132108815:
                                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 7.018052101135254:
                                                        if Q.centroid_offset > 0.005994866602122784:
                                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 35.72087860107422:
                                                            if Q.centroid_offset > 0.014670164790004492:
                                                                if Q.mass > 63.44383239746094:
                                                                    if Q.max_dr > 0.11085297912359238:
                                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 65.83797836303711:
                                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.23987678438425064:
                                                                        if Q.girth > 0.07374345511198044:
                                                                            if Q.mass > 55.45409965515137:
                                                                                if Q.max_dr > 0.11282303929328918:
                                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.max_dr > 0.11879459023475647:
                                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 72.08065795898438:
                                                                    if Q.centroid_offset > 0.00793925765901804:
                                                                        if Q.z_7 > 0.03643728047609329:
                                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 37.66670608520508:
                                                                        return 'W'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 26.875:
                                                                            if Q.girth2 > 0.003595316899009049:
                                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.centroid_offset > 0.007541608763858676:
                                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.D2 > 0.6923363208770752:
                                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 28.0625:
                                                                if Q.width > 0.003900618525221944:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.009579925332218409:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 0.7349115014076233:
                                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 38% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.009777925908565521:
                                                    if Q.log_sum_pt > 7.0391809940338135:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.06989153102040291:
                                                            if Q.lam1 > 0.002823724295012653:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 38.70803451538086:
                                                        if Q.lam2 > 9.177672473015264e-05:
                                                            if Q.pt_7 > 38.828125:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 1065.9375:
                                                                return 'g'   # 43% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 42.79971885681152:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.003130681114271283:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 38% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.05216698348522186:
                                                            if Q.centroid_offset > 0.0067628067918121815:
                                                                if Q.lam2 > 6.279928493313491e-05:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.2009788453578949:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.011695284862071276:
                                        if Q.pt_7 > 28.8671875:
                                            if Q.girth > 0.04498039186000824:
                                                if Q.centroid_offset > 0.013935878407210112:
                                                    if Q.z_dr_0p1_0p2 > 0.3062451034784317:
                                                        return 'g'   # 39% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.0034237686777487397:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 32.22233200073242:
                                                            if Q.D2 > 0.8152614533901215:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 281.28125:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.14498933404684067:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 31.606953620910645:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004055476048961282:
                                            if Q.pt_7 > 31.1015625:
                                                if Q.width > 0.0044602868147194386:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.6048396825790405:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 389.078125:
                                                if Q.lam2 > 7.48480815673247e-05:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0035580757539719343:
                                                    if Q.D2 > 0.7562364041805267:
                                                        if Q.C2 > 0.03489513136446476:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.07464278489351273:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 37.171875:
                                if Q.centroid_offset > 0.012765196617692709:
                                    if Q.lam1 > 0.0018423103028908372:
                                        if Q.eccentricity > 0.9810864925384521:
                                            if Q.log_sum_pt > 6.868932485580444:
                                                if Q.e2_sq > 0.0018313711043447256:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 43.828125:
                                                if Q.lam1 > 0.0021799380192533135:
                                                    if Q.mass > 37.60090637207031:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016007483936846256:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.02490646205842495:
                                                    if Q.sum_pt_top5 > 720.734375:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.0595586784183979:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 906.8125:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.9845898747444153:
                                        if Q.mass > 47.238983154296875:
                                            if Q.eccentricity > 0.9944612979888916:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.002230632700957358:
                                                    if Q.girth > 0.045945778489112854:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 44.4375:
                                                if Q.centroid_offset > 0.010383975226432085:
                                                    if Q.max_dr > 0.09849075227975845:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 31.403297424316406:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 49.75:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 929.109375:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.08895501866936684:
                                                        if Q.centroid_offset > 0.006133890012279153:
                                                            if Q.mass > 34.78139114379883:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.0024934608954936266:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 828.234375:
                                                                return 'g'   # 45% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 43.171875:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9753432869911194:
                                                if Q.centroid_offset > 0.00796165643259883:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 36.490217208862305:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 0.001406406401656568:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 32.858001708984375:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.02486029639840126:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.008244646713137627:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.012786442879587412:
                                    if Q.centroid_offset > 0.014339829329401255:
                                        if Q.sum_pt > 1024.0546875:
                                            if Q.e2_sq > 0.0011717446614056826:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 33.26990509033203:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.5234166383743286:
                                                if Q.width > 0.002089532557874918:
                                                    return 'W'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0015713185421191156:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 40.3170166015625:
                                        if Q.width > 0.0017558695399202406:
                                            if Q.mass > 42.10727119445801:
                                                if Q.mass > 60.99400520324707:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 31.8984375:
                                                        if Q.width > 0.0022385198390111327:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.006784092867746949:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_4 > 0.06003380008041859:
                                                        return 'W'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1056.48828125:
                                            if Q.z_7 > 0.015202393289655447:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.009765439666807652:
                                                if Q.mass > 35.49508857727051:
                                                    if Q.mass > 37.6496467590332:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.011280361097306013:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.546875:
                                                        if Q.C2 > 0.02617247961461544:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.09081830829381943:
                                                                return 'g'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 933.1171875:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 589.609375:
                                                    if Q.pt_7 > 31.6171875:
                                                        if Q.planar_flow > 0.20974021404981613:
                                                            if Q.C2 > 0.02989030722528696:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.06850128620862961:
                                                                if Q.mass > 35.451913833618164:
                                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 39.005998611450195:
                                                            if Q.centroid_offset > 0.006949267815798521:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.02336139604449272:
                                                        if Q.z_7 > 0.05071869306266308:
                                                            if Q.centroid_offset > 0.005951630184426904:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top3 > 462.46875:
                if Q.centroid_offset > 0.01520640728995204:
                    if Q.centroid_offset > 0.02448699064552784:
                        if Q.e2_sq > 0.0004996637289877981:
                            if Q.centroid_offset > 0.03192843496799469:
                                if Q.pt_7 > 24.3515625:
                                    if Q.eccentricity > 0.9806686639785767:
                                        if Q.centroid_offset > 0.03919409401714802:
                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.043309418484568596:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04888810217380524:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.6876004338264465:
                                                if Q.log_sum_pt > 6.505550384521484:
                                                    if Q.pt_7 > 38.46875:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 43% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.597411870956421:
                                        if Q.width > 0.002079323399811983:
                                            if Q.pt_7 > 22.1796875:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.2498166859149933:
                                    if Q.centroid_offset > 0.027921326458454132:
                                        if Q.sum_pt > 717.09375:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 24.1796875:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 40.46875:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.001182943640742451:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 755.3203125:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 49% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.19350574165582657:
                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 42.953125:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.525292873382568:
                                                if Q.centroid_offset > 0.030213648453354836:
                                                    if Q.planar_flow > 0.07796672731637955:
                                                        if Q.sum_pt > 756.53125:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 23.8359375:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.04271246679127216:
                                if Q.pt_7 > 29.359375:
                                    if Q.LHA > 0.25755925476551056:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 42.171875:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9932011067867279:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.23563870787620544:
                                                    if Q.girth2_top3 > 0.0023234387626871467:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.9817518591880798:
                                        if Q.log_sum_pt > 6.674862384796143:
                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.03348229452967644:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.887783765792847:
                                    if Q.pt_7 > 36.625:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 906.40625:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.603933334350586:
                                        if Q.centroid_offset > 0.02634299173951149:
                                            if Q.pt_7 > 44.984375:
                                                if Q.lam2 > 3.116969674010761e-05:
                                                    if Q.max_dr > 0.041880154982209206:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.18155332654714584:
                                                    if Q.C2 > 0.019594614394009113:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 20.484375:
                                                            if Q.LHA > 0.21808244287967682:
                                                                if Q.pt_0 > 380.375:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.650460720062256:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 796.5234375:
                                                if Q.lam1 > 0.0009125778451561928:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 44.546875:
                                                        if Q.C2 > 0.006540245376527309:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.703514099121094:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 27.734375:
                                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.025501083582639694:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.21244361251592636:
                                                    if Q.mass_over_sum_pt > 0.016567393206059933:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 36.015625:
                                                        if Q.log_sum_pt > 6.642506837844849:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 533.625:
                                                            if Q.pt_7 > 27.171875:
                                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 25.4296875:
                                            if Q.centroid_offset > 0.028560781851410866:
                                                if Q.log_sum_pt > 6.5005621910095215:
                                                    if Q.mass > 10.631239414215088:
                                                        if Q.lam2 > 3.580638804123737e-05:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 39% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9295533001422882:
                                                    if Q.centroid_offset > 0.027289184741675854:
                                                        if Q.pt_7 > 31.90625:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.548688888549805:
                                                        if Q.centroid_offset > 0.026081060990691185:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.570652961730957:
                                                if Q.pt_7 > 23.21875:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 4.376702418085188e-05:
                                                    if Q.z_7 > 0.03431401588022709:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.328125:
                            if Q.centroid_offset > 0.018502225168049335:
                                if Q.mass > 6.761697769165039:
                                    if Q.pt_7 > 44.921875:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.04819142259657383:
                                            if Q.dr_0 > 0.023405064828693867:
                                                if Q.lam2 > 9.10504677449353e-05:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 935.0625:
                                        if Q.sum_pt > 976.96875:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9744776785373688:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.023507798090577126:
                                            if Q.sum_pt_top5 > 653.46875:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.0637308843433857:
                                                if Q.planar_flow > 0.0882631465792656:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.017055491916835308:
                                    if Q.z_7 > 0.050852734595537186:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 780.65625:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.006435913732275367:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.66852593421936:
                                if Q.centroid_offset > 0.017364585772156715:
                                    if Q.centroid_offset > 0.02274688519537449:
                                        if Q.sum_pt > 893.0625:
                                            if Q.lam1 > 0.0006818726542405784:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02328537590801716:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.1484375:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.2795882523059845:
                                                if Q.lam1 > 0.0008053827914409339:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.023397994227707386:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.723812580108643:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.023989330045878887:
                                                    if Q.log_sum_pt > 6.752224683761597:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.609375:
                                                        if Q.sum_pt_top5 > 703.484375:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1043.8203125:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.901957273483276:
                                                if Q.pt_7 > 28.0234375:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.859375:
                                                    if Q.e2 > 0.011226067785173655:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 3.8425348975579254e-05:
                                                            if Q.centroid_offset > 0.018525819294154644:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.70826530456543:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.00037897763832006603:
                                                            if Q.centroid_offset > 0.01774286199361086:
                                                                if Q.pt_7 > 18.890625:
                                                                    if Q.pt_7 > 24.4609375:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.019476271234452724:
                                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt > 800.9375:
                                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.17947911471128464:
                                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 853.79296875:
                                        if Q.sum_pt > 1011.0078125:
                                            if Q.z_7 > 0.01857816055417061:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 36.296875:
                                                if Q.max_dr > 0.02801353484392166:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 876.8828125:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0002918301324825734:
                                                        if Q.z_7 > 0.021996157243847847:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 34.265625:
                                            if Q.lam2 > 1.8749252376437653e-05:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 423.1875:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.3665569871664047:
                                                if Q.pt_7 > 22.5625:
                                                    if Q.sum_pt > 811.7734375:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016013724729418755:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01596961822360754:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0003083915216848254:
                                                    if Q.sum_pt > 831.625:
                                                        if Q.pt_7 > 25.5078125:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.01938918698579073:
                                    if Q.pt_7 > 25.2578125:
                                        if Q.sum_pt > 711.3046875:
                                            if Q.centroid_offset > 0.02029048465192318:
                                                if Q.pt_7 > 27.7109375:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0005613051180262119:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.613226890563965:
                                                    if Q.dr_0 > 0.020053147338330746:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 767.71875:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.17466972768306732:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.027119815349578857:
                                                if Q.LHA > 0.203687883913517:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 30.5625:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 36% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9338308572769165:
                                                    if Q.pt_7 > 29.0078125:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.6301679611206055:
                                            if Q.girth > 0.022706013172864914:
                                                if Q.pt_7 > 22.03125:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.9685520827770233:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.20266879349946976:
                                                if Q.sum_pt > 711.84375:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.035237932577729225:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.04319577291607857:
                                        if Q.dr_0 > 0.020558378659188747:
                                            if Q.mass > 26.447701454162598:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9176237881183624:
                                                    if Q.centroid_offset > 0.01843470986932516:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 400.9375:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.018641872331500053:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.046040890738368034:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.02716242242604494:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.03455965034663677:
                                            if Q.planar_flow > 0.38447389006614685:
                                                if Q.dr_0 > 0.020568852312862873:
                                                    if Q.sum_pt > 745.5390625:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 681.8203125:
                                                            if Q.LHA > 0.185768723487854:
                                                                return 'W'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01256368076428771:
                                                    if Q.mass > 26.773012161254883:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 39.515625:
                        if Q.centroid_offset > 0.004444011254236102:
                            if Q.pt_7 > 44.390625:
                                if Q.width > 4.7094365072553046e-05:
                                    if Q.pt_7 > 48.203125:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.005943771218881011:
                                            if Q.D2 > 1.1671150922775269:
                                                if Q.centroid_offset > 0.007554186275228858:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.4215114116668701:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.08520762622356415:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 12.167520999908447:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.20098301768302917:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.718522071838379:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.8520419895648956:
                                                if Q.log_sum_pt > 6.798986196517944:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.011146606411784887:
                                                        if Q.dr_0 > 0.011210970114916563:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 4.780835115525406e-05:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.830881834030151:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 391.625:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 51.109375:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.839576005935669:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.007948963437229395:
                                    if Q.eccentricity > 0.9030806422233582:
                                        if Q.centroid_offset > 0.009910530410706997:
                                            if Q.sum_pt > 824.1875:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9657330513000488:
                                                    if Q.centroid_offset > 0.012422476895153522:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 522.46875:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 896.328125:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 377.625:
                                                    if Q.D2 > 2.1390520334243774:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 1.320250430580927e-05:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 416.1875:
                                                return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 957.3125:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 1.747343867464224e-05:
                                            if Q.dr_0 > 0.014289960730820894:
                                                if Q.planar_flow > 0.5543202459812164:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005890858359634876:
                                                    if Q.sum_pt_top2 > 438.5:
                                                        if Q.lam2 > 4.6900106099201366e-05:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.006839490262791514:
                                                                return 'g'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 499.9375:
                                                        if Q.lam2 > 4.27179729740601e-05:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 463.3125:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_7 > 0.016282127238810062:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.5335831046104431:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 417.03125:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.00567631097510457:
                                                    if Q.eccentricity > 0.8679416179656982:
                                                        if Q.z_4 > 0.09749660640954971:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.05396995507180691:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.727055549621582:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.893069744110107:
                                if Q.mass > 5.2799036502838135:
                                    if Q.sum_pt > 1026.984375:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 3.2942542020464316e-05:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 49.125:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 4.566809548123274e-05:
                                                    if Q.centroid_offset > 0.002055976423434913:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1059.921875:
                                        if Q.pt_7 > 45.203125:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.001904528762679547:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 54.796875:
                                            if Q.centroid_offset > 0.0015176986926235259:
                                                if Q.width > 2.281963315908797e-05:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 9.830423550738487e-06:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 51.203125:
                                    if Q.width > 4.741107113659382e-05:
                                        if Q.centroid_offset > 0.0019323262968100607:
                                            if Q.pt_7 > 53.265625:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9158685505390167:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 1.7694093912723474e-05:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.059218887239694595:
                                                            return 'q'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.8129637241363525:
                                                if Q.z_4 > 0.09229385480284691:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.008600447326898575:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.010902849491685629:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 60.96875:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.09476325660943985:
                                                            if Q.lam2 > 1.2338668511802098e-05:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0037660442758351564:
                                            if Q.mass > 4.181682825088501:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 5.467971086502075:
                                                if Q.centroid_offset > 0.001984915812499821:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 3.864953941956628e-05:
                                        if Q.pt_7 > 44.90625:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.022356250323355198:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.5623487532138824:
                                                    if Q.centroid_offset > 0.0035478337667882442:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.003577383002266288:
                                            if Q.pt_7 > 46.328125:
                                                if Q.girth2 > 5.555800635193009e-05:
                                                    if Q.girth2_top3 > 6.997831223998219e-05:
                                                        if Q.sum_pt > 851.15625:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.1790530809084885e-05:
                                                    if Q.girth2_top3 > 8.323232759721577e-05:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 446.9375:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.8860185146331787:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.619823217391968:
                                                if Q.girth2 > 5.512061215995345e-05:
                                                    if Q.pt_7 > 45.828125:
                                                        if Q.centroid_offset > 0.002667224616743624:
                                                            if Q.pt_0 > 209.6875:
                                                                if Q.pt_7 > 48.796875:
                                                                    if Q.D2 > 1.4720996022224426:
                                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.58372762799263:
                                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.00011179973080288619:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0019851597025990486:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.988492488861084:
                            if Q.pt_7 > 21.6796875:
                                if Q.girth2 > 4.268777047400363e-05:
                                    if Q.centroid_offset > 0.0026812608120962977:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 25.234375:
                                            if Q.sum_pt > 1115.6171875:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 31.3515625:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.033671855926514:
                                                if Q.e2 > 0.0037888347869738936:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 35.703125:
                                        if Q.girth2 > 2.2769753741158638e-05:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 3.1820749427424744e-05:
                                            if Q.pt_7 > 31.328125:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1266.859375:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.0050361379981040955:
                                    if Q.pt_7 > 15.984375:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 18.3984375:
                                        if Q.centroid_offset > 0.003189725335687399:
                                            if Q.mass > 8.798603057861328:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 629.3515625:
                                if Q.centroid_offset > 0.01257301727309823:
                                    if Q.pt_7 > 33.859375:
                                        if Q.eccentricity > 0.8839117288589478:
                                            if Q.pt_7 > 36.109375:
                                                if Q.lam2 > 1.0861823284358252e-05:
                                                    if Q.sum_pt > 821.703125:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.01765669323503971:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 791.125:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.01219654455780983:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 902.203125:
                                            if Q.girth > 0.014611976686865091:
                                                if Q.sum_pt > 1019.640625:
                                                    if Q.pt_7 > 22.0703125:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.013923784252256155:
                                                        if Q.eccentricity > 0.9034990668296814:
                                                            if Q.sum_pt > 938.1640625:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.02481840830296278:
                                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_7 > 0.02792239934206009:
                                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 23.1796875:
                                                            if Q.planar_flow > 0.374600887298584:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 956.13671875:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.896923542022705:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 5.780645915365312e-05:
                                                if Q.pt_7 > 31.4609375:
                                                    if Q.dr_0 > 0.018041351810097694:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.726355075836182:
                                                        if Q.centroid_offset > 0.014231421053409576:
                                                            if Q.z_7 > 0.023230685852468014:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013347698375582695:
                                                                if Q.log_sum_pt > 6.759862422943115:
                                                                    return 'W'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.011327453888952732:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 881.296875:
                                                    if Q.centroid_offset > 0.014595446176826954:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 34.078125:
                                        if Q.centroid_offset > 0.007709667552262545:
                                            if Q.log_sum_pt > 6.852853536605835:
                                                if Q.log_sum_pt > 6.882853984832764:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.003656360669992864:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.3864985450927634e-05:
                                                    if Q.pt_7 > 37.265625:
                                                        if Q.dr_0 > 0.016469313763082027:
                                                            if Q.lam2 > 6.672033850918524e-05:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 3.2798903703223914e-05:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_0 > 280.0:
                                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 399.5:
                                                            if Q.centroid_offset > 0.011019314173609018:
                                                                if Q.planar_flow > 0.46836456656455994:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.006621569860726595:
                                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.01370155019685626:
                                                                return 'q'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 380.5625:
                                                        if Q.pt_7 > 37.515625:
                                                            if Q.lam2 > 1.893422995635774e-05:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 37.765625:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.920764207839966:
                                                if Q.girth2 > 4.6664770707138814e-05:
                                                    if Q.centroid_offset > 0.0035149749601259828:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 1034.921875:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.908482722181361e-05:
                                                    if Q.log_sum_pt > 6.8632471561431885:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0057295518927276134:
                                                            if Q.pt_7 > 36.640625:
                                                                if Q.dr_0 > 0.013283591251820326:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.007751756114885211:
                                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.898478031158447:
                                                        if Q.centroid_offset > 0.005339254625141621:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 768.9453125:
                                            if Q.log_sum_pt > 6.938523292541504:
                                                if Q.centroid_offset > 0.005165754118934274:
                                                    if Q.pt_7 > 26.84375:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 19.40625:
                                                            if Q.girth > 0.009735197760164738:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.8671875:
                                                        if Q.girth2 > 7.338081195484847e-05:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.009688550606369972:
                                                    if Q.log_sum_pt > 6.877078533172607:
                                                        if Q.z_7 > 0.02939177118241787:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 31.5703125:
                                                            if Q.lam2 > 4.09754993597744e-05:
                                                                if Q.dr_0 > 0.01230717170983553:
                                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 794.64453125:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0012573517742566764:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.056172000244259834:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 7.469168663024902:
                                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0019770670915022492:
                                                if Q.dr_0 > 0.004350424045696855:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006899749394506216:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.008966546040028334:
                                                    if Q.mass > 11.737618923187256:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 722.0625:
                                                            if Q.z_4 > 0.06416794657707214:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.028591019101440907:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_4 > 0.07493626326322556:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 18.2265625:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.4218936115503311:
                                    if Q.sum_pt_top5 > 593.484375:
                                        if Q.centroid_offset > 0.007755432510748506:
                                            if Q.z_7 > 0.043745096772909164:
                                                if Q.dr_0 > 0.017985230311751366:
                                                    if Q.centroid_offset > 0.01146793644875288:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_0 > 351.125:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009702961426228285:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.012128385249525309:
                                                                return 'q'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.014875359367579222:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 6.526392462546937e-05:
                                                        if Q.centroid_offset > 0.010630824137479067:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0023659043945372105:
                                                if Q.dr_0 > 0.005573052680119872:
                                                    if Q.z_7 > 0.04633018746972084:
                                                        if Q.centroid_offset > 0.005647054873406887:
                                                            if Q.dr_0 > 0.009938490577042103:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.574214696884155:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0032283220207318664:
                                                            if Q.sum_pt_top5 > 614.3125:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.004300412954762578:
                                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 722.4609375:
                                                    if Q.z_4 > 0.0786467157304287:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.606882333755493:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 3.907434438588098e-05:
                                                        if Q.pt_7 > 19.0234375:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2_sq > 0.00029858839116059244:
                                            if Q.centroid_offset > 0.007104187738150358:
                                                if Q.dr_0 > 0.01625345554202795:
                                                    if Q.z_7 > 0.04526069015264511:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.010204505175352097:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.02334923017770052:
                                                if Q.mass_over_sum_pt > 0.006098391721025109:
                                                    if Q.sum_pt_top5 > 576.046875:
                                                        if Q.dr_0 > 0.008399458602070808:
                                                            if Q.centroid_offset > 0.0071813620161265135:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.0050934734754264355:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.010456554591655731:
                                        if Q.max_dr > 0.1273227334022522:
                                            if Q.lam2 > 5.7153352827299386e-05:
                                                if Q.LHA > 0.17361485958099365:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.636972188949585:
                                                if Q.z_7 > 0.045922549441456795:
                                                    if Q.centroid_offset > 0.012798626441508532:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.008494725450873375:
                                            if Q.width > 0.0001414614889654331:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.004214756656438112:
                                                if Q.girth > 0.018144385889172554:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 585.71875:
                                                        if Q.centroid_offset > 0.002402993035502732:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 9.874598503112793:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.007858834927901626:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005445662187412381:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 600.28125:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top5 > 533.484375:
                    if Q.centroid_offset > 0.021947954781353474:
                        if Q.centroid_offset > 0.027416111901402473:
                            if Q.centroid_offset > 0.0443667508661747:
                                if Q.z_7 > 0.04543968848884106:
                                    if Q.pt_7 > 40.328125:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.052250223234295845:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p05_0p1 > 0.5693815648555756:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.551816701889038:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 2.6166280804318376e-05:
                                        if Q.mass > 18.0693998336792:
                                            return 't'   # 36% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 15.693216800689697:
                                    if Q.lam2 > 0.00010352027311455458:
                                        if Q.pt_7 > 43.015625:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.031869277358055115:
                                                if Q.pt_7 > 26.0859375:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.21748056262731552:
                                                    if Q.pt_7 > 28.2265625:
                                                        if Q.log_sum_pt > 6.5353546142578125:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.035512031987309456:
                                            if Q.pt_7 > 27.359375:
                                                if Q.mass_over_sum_pt > 0.03439143858850002:
                                                    if Q.lam2 > 3.980648943979759e-05:
                                                        return 'Z'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.5078125:
                                                if Q.LHA > 0.20827126502990723:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 27.5234375:
                                        if Q.pt_7 > 52.734375:
                                            if Q.mass > 5.096535921096802:
                                                if Q.lam2 > 2.2538009943673387e-05:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.04335080273449421:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.029455462470650673:
                                                if Q.centroid_offset > 0.04109048470854759:
                                                    if Q.pt_7 > 46.015625:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 3.417783955228515e-05:
                                                            if Q.e2_sq > 0.0001300664371228777:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.016239754855632782:
                                                        if Q.sum_pt_top5 > 574.4375:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 9.36808210099116e-05:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 701.125:
                                                    if Q.mass > 8.231742858886719:
                                                        if Q.pt_7 > 45.796875:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9671794772148132:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.012254185508936644:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.521723985671997:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 46% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 697.890625:
                                            if Q.pt_7 > 24.5078125:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 3.4496111766202375e-05:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.22014398127794266:
                                if Q.z_7 > 0.058471595868468285:
                                    if Q.z_7 > 0.06482159346342087:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.02871280163526535:
                                            if Q.centroid_offset > 0.025943368673324585:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 319.15625:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 692.8125:
                                        if Q.centroid_offset > 0.025282665155828:
                                            if Q.lam1 > 0.0008837643545120955:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 307.3125:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.0007668411999475211:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 709.0:
                                                    if Q.pt_7 > 40.375:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.033700967207551:
                                            if Q.pt_7 > 27.171875:
                                                return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.000755728455260396:
                                                if Q.sum_pt > 672.71875:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.06799961999058723:
                                    if Q.planar_flow > 0.08428292721509933:
                                        if Q.centroid_offset > 0.02463576663285494:
                                            if Q.e2 > 0.0038408502005040646:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 690.1953125:
                                        if Q.centroid_offset > 0.025681963190436363:
                                            if Q.log_sum_pt > 6.610170125961304:
                                                if Q.girth2 > 0.000803115515736863:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.18122538924217224:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.006293976679444313:
                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 2.1291087250574492e-05:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 811.328125:
                                                if Q.girth2 > 0.0006204369419720024:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.02022675797343254:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 586.9375:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.1827242746949196:
                                            if Q.pt_7 > 26.8359375:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02356825303286314:
                                                if Q.pt_7 > 31.5546875:
                                                    if Q.sum_pt_top5 > 546.859375:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.004252334823831916:
                            if Q.z_7 > 0.050569770857691765:
                                if Q.tau21 > 0.3878709524869919:
                                    if Q.centroid_offset > 0.01980004832148552:
                                        if Q.z_7 > 0.06139770708978176:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.591609001159668:
                                                if Q.eccentricity > 0.9234404861927032:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0005716516170650721:
                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 4.081812403455842e-05:
                                            if Q.z_7 > 0.05764981545507908:
                                                if Q.D2 > 1.0303481817245483:
                                                    if Q.tau21 > 0.4560948461294174:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.006022489862516522:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 47.421875:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.011285805609077215:
                                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.5202266871929169:
                                                    if Q.girth2 > 4.9938904339796863e-05:
                                                        if Q.centroid_offset > 0.0069730684626847506:
                                                            if Q.centroid_offset > 0.018825190141797066:
                                                                if Q.log_sum_pt > 6.618404865264893:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.011734291445463896:
                                                                if Q.planar_flow > 0.7687918245792389:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top3 > 423.28125:
                                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0069922893308103085:
                                                        if Q.planar_flow > 0.2639911323785782:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 8.852503776550293:
                                                                if Q.dr_0 > 0.014323446899652481:
                                                                    if Q.centroid_offset > 0.010463149286806583:
                                                                        if Q.C2 > 0.020956549793481827:
                                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.014440504368394613:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 1.3972166016174015e-05:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.06474797800183296:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.010085881687700748:
                                        if Q.girth2 > 0.001731008815113455:
                                            if Q.centroid_offset > 0.01701468415558338:
                                                if Q.e2 > 0.01909525040537119:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 41.015625:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.3870950574055314e-05:
                                                    if Q.planar_flow > 0.39540131390094757:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01478348346427083:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.023775418289005756:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01466173306107521:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.022166511043906212:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 46.765625:
                                            if Q.centroid_offset > 0.005747725022956729:
                                                if Q.pt_7 > 50.453125:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.019338291138410568:
                                                        if Q.C2 > 0.011490226723253727:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.015340818092226982:
                                                    if Q.pt_7 > 55.90625:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.04894779063761234:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.017130951397120953:
                                                if Q.lam2 > 7.267258115462027e-05:
                                                    if Q.pt_7 > 37.796875:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.3240928053855896:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.007337783230468631:
                                                            if Q.pt_7 > 42.859375:
                                                                if Q.sum_pt_top2 > 319.625:
                                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 3.5750259485212155e-05:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.009153782855719328:
                                                        if Q.centroid_offset > 0.006895724451169372:
                                                            if Q.sum_pt_top3 > 434.28125:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.015436396934092045:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.8920093476772308:
                                    if Q.centroid_offset > 0.01579262036830187:
                                        if Q.dr_0 > 0.021458142437040806:
                                            if Q.sum_pt > 720.21875:
                                                if Q.lam1 > 0.00042059416591655463:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0017801481299102306:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04422902502119541:
                                                        if Q.centroid_offset > 0.019743788987398148:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.18364200741052628:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.5045093446969986:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 743.234375:
                                                if Q.centroid_offset > 0.018991008400917053:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.578125:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.036049678921699524:
                                                    if Q.planar_flow > 0.13230015337467194:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.16127149015665054:
                                                            return 'W'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.015086156316101551:
                                            if Q.C2 > 0.03189858794212341:
                                                if Q.z_7 > 0.04237866587936878:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 7.168071031570435:
                                                    if Q.centroid_offset > 0.014542029239237309:
                                                        if Q.z_7 > 0.046920912340283394:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04622645862400532:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011016025673598051:
                                                if Q.z_7 > 0.044628679752349854:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 578.28125:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 575.6875:
                                                    if Q.z_7 > 0.046150682494044304:
                                                        if Q.dr_0 > 0.009781962260603905:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.006844997638836503:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.0107747633010149:
                                                        if Q.mass > 16.883434295654297:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.008319529239088297:
                                                            return 'g'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 552.734375:
                                                                return 'q'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00869713257998228:
                                        if Q.z_7 > 0.036140067502856255:
                                            if Q.LHA > 0.19780471920967102:
                                                if Q.centroid_offset > 0.0159454308450222:
                                                    if Q.pt_7 > 29.3671875:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 596.015625:
                                                    if Q.centroid_offset > 0.018170428462326527:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.043359024450182915:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 2.7235555535298772e-05:
                                                                if Q.e2_sq > 0.00028398017457220703:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.023141835816204548:
                                                        if Q.centroid_offset > 0.014871797990053892:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.855590283870697:
                                                            if Q.centroid_offset > 0.01266808807849884:
                                                                return 'g'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.01569309365004301:
                                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.562929153442383:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.8220008909702301:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 574.578125:
                                            if Q.z_7 > 0.043334463611245155:
                                                if Q.centroid_offset > 0.0054099662229418755:
                                                    if Q.dr_0 > 0.011382256634533405:
                                                        if Q.z_4 > 0.09984052181243896:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 1.340389917459106e-05:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.005456526065245271:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 6.0657583162537776e-05:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 703.3984375:
                                                    if Q.lam2 > 3.2274398108711466e-05:
                                                        if Q.dr_0 > 0.009537363424897194:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.006933393655344844:
                                                                return 'g'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.00010441687481943518:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.014459807425737381:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                        else:
                            if Q.z_7 > 0.06062503717839718:
                                if Q.girth2 > 4.127553984289989e-05:
                                    if Q.D2 > 1.1857044696807861:
                                        if Q.centroid_offset > 0.0016688068280927837:
                                            if Q.z_7 > 0.06814510002732277:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.012155514676123857:
                                                    if Q.pt_7 > 48.375:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 4.4364467612467706e-05:
                                                        if Q.planar_flow > 0.41760359704494476:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.03388162702322006:
                                                                return 'g'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.006904457230120897:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.07245013117790222:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.585373878479004:
                                                    if Q.lam2 > 1.2873491868958808e-05:
                                                        if Q.pt_7 > 50.265625:
                                                            if Q.centroid_offset > 0.0009739824454300106:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.005419061053544283:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.4944894313812256:
                                                            if Q.centroid_offset > 0.0011760903871618211:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.00902853300794959:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 61.359375:
                                            if Q.centroid_offset > 0.0014666896313428879:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.009659514296799898:
                                                if Q.pt_7 > 52.03125:
                                                    if Q.centroid_offset > 0.0025327226612716913:
                                                        if Q.D2 > 1.0697544813156128:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.10896848887205124:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0024573461851105094:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.596570014953613:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 721.09375:
                                        if Q.width > 3.6917794204782695e-05:
                                            if Q.z_7 > 0.07714474201202393:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 2.9953420380479656e-05:
                                            if Q.log_sum_pt > 6.562223196029663:
                                                return 'q'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 42% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 714.1875:
                                    if Q.z_7 > 0.05326477997004986:
                                        if Q.centroid_offset > 0.0032836844911798835:
                                            if Q.width > 4.661702041630633e-05:
                                                if Q.dr_0 > 0.009313175454735756:
                                                    if Q.eccentricity > 0.7789226770401001:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 1.987699306482682e-05:
                                                if Q.dr_0 > 0.006397159770131111:
                                                    if Q.C2 > 0.018072856590151787:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0019886416848748922:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.585535764694214:
                                            if Q.lam2 > 2.403898452030262e-05:
                                                if Q.D2 > 1.9229315519332886:
                                                    if Q.z_7 > 0.047116972506046295:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.005317030008882284:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.002475948422215879:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 42% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.009337707422673702:
                                        if Q.LHA > 0.14756353944540024:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 677.6875:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0031042483169585466:
                                                    if Q.dr_7 > 0.015622789040207863:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.529224872589111:
                                            if Q.width > 3.724066846189089e-05:
                                                if Q.z_7 > 0.05360397882759571:
                                                    if Q.eccentricity > 0.8707122802734375:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.005462870234623551:
                                                        if Q.planar_flow > 0.5697830617427826:
                                                            if Q.centroid_offset > 0.0028165722033008933:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 588.6875:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_0 > 184.9375:
                                                                return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0022853834088891745:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0002280440749018453:
                                                if Q.sum_pt_top5 > 554.1875:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.mass > 24.713354110717773:
                        if Q.centroid_offset > 0.018959971144795418:
                            if Q.girth2 > 0.004768843529745936:
                                if Q.sum_pt > 441.625:
                                    if Q.tau21 > 0.2945762425661087:
                                        if Q.centroid_offset > 0.049962855875492096:
                                            if Q.planar_flow > 0.1964692920446396:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 30.9375:
                                                if Q.lam2 > 0.0007239237893372774:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.02862060908228159:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04592951759696007:
                                            if Q.pt_7 > 41.40625:
                                                if Q.z_dr_0p05_0p1 > 0.8999302387237549:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 33.625:
                                                if Q.e2 > 0.024264763109385967:
                                                    if Q.lam2 > 0.00013952266453998163:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 27.8125:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.22277946025133133:
                                        if Q.max_dr > 0.10367557406425476:
                                            if Q.lam2 > 8.716482625459321e-05:
                                                if Q.pt_7 > 35.078125:
                                                    if Q.girth2 > 0.005809607915580273:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.27938133478164673:
                                                            return 'Z'   # 39% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005945406388491392:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 32.953125:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.2421875:
                                            if Q.centroid_offset > 0.02880894299596548:
                                                if Q.mass > 27.778005599975586:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.7018583416938782:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 391.859375:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03411571495234966:
                                    if Q.lam2 > 0.0001025624806061387:
                                        if Q.pt_7 > 26.9453125:
                                            if Q.z_dr_0_0p05 > 0.5482543408870697:
                                                if Q.sum_pt_top5 > 477.859375:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03889869339764118:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.19247034937143326:
                                                        if Q.max_dr > 0.11248617246747017:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9234459102153778:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0006605645467061549:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1276431456208229:
                                            if Q.lam1 > 0.003773468662984669:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.14272933453321457:
                                                    return 'Z'   # 39% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.043529532849788666:
                                                if Q.pt_7 > 38.09375:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 34% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 28.5078125:
                                        if Q.dr_0 > 0.02932089753448963:
                                            if Q.eccentricity > 0.9347212612628937:
                                                if Q.sum_pt > 510.9140625:
                                                    if Q.centroid_offset > 0.02094546053558588:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 27.158013343811035:
                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1337130293250084:
                                                        if Q.mass > 27.547987937927246:
                                                            return 'Z'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.05609252117574215:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 6.660054714302532e-05:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 27.313029289245605:
                                                    if Q.z_dr_0p05_0p1 > 0.32511086761951447:
                                                        if Q.pt_7 > 45.890625:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 554.3125:
                                                            if Q.dr_0 > 0.03362271748483181:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.021326838992536068:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 465.40625:
                                                        if Q.pt_7 > 42.96875:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p05_0p1 > 0.5259833037853241:
                                                            if Q.pt_7 > 41.921875:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 31.9375:
                                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0004707837651949376:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.5152648091316223:
                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 522.34375:
                                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.022752844728529453:
                                                if Q.eccentricity > 0.9342594146728516:
                                                    if Q.centroid_offset > 0.021877494640648365:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 551.796875:
                                            if Q.lam1 > 0.002461105352267623:
                                                if Q.pt_7 > 24.3828125:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 482.515625:
                                if Q.centroid_offset > 0.005205073859542608:
                                    if Q.tau21 > 0.2146480605006218:
                                        if Q.centroid_offset > 0.01722712628543377:
                                            if Q.mass > 27.380367279052734:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.022009755484759808:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.06663410365581512:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.009005526080727577:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 36.515625:
                                            if Q.centroid_offset > 0.008013223297894001:
                                                if Q.lam1 > 0.0020087406737729907:
                                                    if Q.centroid_offset > 0.01627397909760475:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.06440436094999313:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0128588886000216:
                                                if Q.sum_pt > 624.5234375:
                                                    return 'W'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.08811649680137634:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.021267473697662354:
                                        if Q.z_7 > 0.05606328696012497:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 28.81092357635498:
                                    if Q.centroid_offset > 0.015262039843946695:
                                        if Q.C2 > 0.051205024123191833:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 30.1640625:
                                                if Q.girth2_top3 > 0.0035147477174177766:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9325314462184906:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.1067084074020386:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.059883808717131615:
                                            if Q.pt_7 > 31.4453125:
                                                if Q.mass_over_sum_pt > 0.07833731546998024:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.08466248586773872:
                                                        if Q.dr_7 > 0.05074618570506573:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 360.5:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 454.78125:
                                                    if Q.centroid_offset > 0.010763044469058514:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 3.65249507012777e-05:
                                        if Q.girth > 0.055042218416929245:
                                            if Q.mass > 28.35108184814453:
                                                if Q.pt_7 > 32.875:
                                                    if Q.centroid_offset > 0.014335650019347668:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.013880388345569372:
                                                    if Q.mass > 27.3056640625:
                                                        if Q.pt_7 > 32.09375:
                                                            if Q.z_dr_0p05_0p1 > 0.64887934923172:
                                                                return 'W'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.003458782797679305:
                                            if Q.sum_pt_top5 > 432.9375:
                                                if Q.centroid_offset > 0.008247930090874434:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.0734521672129631:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.0809636227786541:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.eccentricity > 0.9491487741470337:
                            if Q.centroid_offset > 0.007798514794558287:
                                if Q.sum_pt_top5 > 485.734375:
                                    if Q.centroid_offset > 0.027178539894521236:
                                        if Q.centroid_offset > 0.04159916006028652:
                                            if Q.LHA > 0.2879396229982376:
                                                if Q.pt_7 > 35.734375:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.042399123311042786:
                                                    if Q.centroid_offset > 0.04413670115172863:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.04827072471380234:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 2.5708502107590903e-05:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 508.625:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 40% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.030733536928892136:
                                                if Q.pt_7 > 30.59375:
                                                    if Q.mass_over_sum_pt > 0.020877298898994923:
                                                        if Q.centroid_offset > 0.03751114755868912:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 41.9375:
                                                                return 'g'   # 42% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 617.21875:
                                                            if Q.LHA > 0.19852399080991745:
                                                                if Q.z_7 > 0.07701557874679565:
                                                                    if Q.max_dr > 0.04472875036299229:
                                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 6.8979997634887695:
                                                                        if Q.lam2 > 2.25392996071605e-05:
                                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 656.46875:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.03862173482775688:
                                                        if Q.mass_over_sum_pt > 0.021332143805921078:
                                                            if Q.pt_7 > 27.3203125:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.20086238533258438:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.4633705615997314:
                                                        if Q.z_7 > 0.06692152470350266:
                                                            if Q.sum_pt_top5 > 517.46875:
                                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.17934268712997437:
                                            if Q.centroid_offset > 0.011996830347925425:
                                                if Q.centroid_offset > 0.021791130304336548:
                                                    if Q.LHA > 0.20195944607257843:
                                                        if Q.pt_7 > 29.6328125:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.468402147293091:
                                                            if Q.z_7 > 0.07107960805296898:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 661.671875:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.055376624688506126:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.023472308181226254:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.01623465120792389:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 42.234375:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011247210204601288:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01719053927809:
                                                    if Q.pt_7 > 39.375:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0007625452999491245:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.28834168612957:
                                        if Q.eccentricity > 0.9902126789093018:
                                            if Q.z_7 > 0.0777430422604084:
                                                if Q.planar_flow > 0.013013152871280909:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 38.390625:
                                                    if Q.girth2_top3 > 0.006110550602898002:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 483.6484375:
                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.phi_1 > -0.01889801025390625:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 313.96875:
                                                if Q.max_dr > 0.08844355121254921:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 21.076647758483887:
                                            if Q.centroid_offset > 0.02385125309228897:
                                                if Q.lam2 > 6.310280514298938e-05:
                                                    if Q.lam1 > 0.0034346869215369225:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 22.22398853302002:
                                                            if Q.max_dr > 0.11462901160120964:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_7 > 0.06564744934439659:
                                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 29.5859375:
                                                        if Q.centroid_offset > 0.043578652665019035:
                                                            return 'g'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.03329838626086712:
                                                                if Q.log_sum_pt > 6.217104434967041:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2_top3 > 0.002963076578453183:
                                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 444.171875:
                                                if Q.LHA > 0.20122770965099335:
                                                    if Q.centroid_offset > 0.04289192892611027:
                                                        if Q.z_7 > 0.05137215740978718:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.988442987203598:
                                                                return 't'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 607.5:
                                                            if Q.centroid_offset > 0.03490769304335117:
                                                                if Q.girth2 > 0.0016428924864158034:
                                                                    if Q.z_dr_0p05_0p1 > 0.13129526376724243:
                                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2_sq > 5.456483813759405e-05:
                                                                    if Q.LHA > 0.2096520960330963:
                                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.001697277999483049:
                                                                if Q.mass > 15.037540435791016:
                                                                    if Q.lam2 > 5.0171018301625736e-05:
                                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 36.78125:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.013188124634325504:
                                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.22179166227579117:
                                                        if Q.tau21 > 0.35477420687675476:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 349.03125:
                                                                if Q.centroid_offset > 0.0123120853677392:
                                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.012899638619273901:
                                                            return 'g'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.21423587948083878:
                                                    if Q.log_sum_pt > 6.3284666538238525:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.01337489252910018:
                                                            if Q.z_7 > 0.04370670206844807:
                                                                if Q.mass > 16.958723068237305:
                                                                    if Q.sum_pt > 515.28125:
                                                                        if Q.lam1 > 0.0021109243389219046:
                                                                            if Q.lam2 > 5.636301648337394e-05:
                                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 437.5625:
                                    if Q.dr_0 > 0.01625403668731451:
                                        if Q.D2 > 1.5237950682640076:
                                            if Q.sum_pt_top3 > 356.28125:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.07599663734436035:
                                                if Q.centroid_offset > 0.004416809184476733:
                                                    if Q.C2 > 0.013242436107248068:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 53.171875:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 13.01718282699585:
                                                    if Q.max_dr > 0.07320882007479668:
                                                        if Q.z_7 > 0.06125735118985176:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.4275243282318115:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.011020047590136528:
                                            if Q.sum_pt_top5 > 489.890625:
                                                if Q.D2 > 1.5765761733055115:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.0782962292432785:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 12.21618366241455:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.06649226322770119:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.4775757193565369:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.9658523201942444:
                                        if Q.centroid_offset > 0.003951869904994965:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.16834868490695953:
                                                if Q.mass > 17.537970542907715:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.08222253620624542:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 506.546875:
                                                if Q.dr_0 > 0.01957832369953394:
                                                    if Q.e2 > 0.021028255112469196:
                                                        return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.17904651910066605:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 17.364852905273438:
                                                    if Q.lam2 > 4.392077426018659e-05:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_4 > 0.10135842487215996:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 486.609375:
                                if Q.centroid_offset > 0.031170197762548923:
                                    if Q.pt_7 > 29.3203125:
                                        if Q.max_dr > 0.06804406270384789:
                                            if Q.z_7 > 0.05382241867482662:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03525072894990444:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 46.46875:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 618.2421875:
                                                    if Q.z_7 > 0.06494321674108505:
                                                        if Q.C2 > 0.011586470529437065:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03508189506828785:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.0987290143966675:
                                        if Q.width > 2.7995378331979737e-05:
                                            if Q.LHA > 0.21442552655935287:
                                                if Q.pt_7 > 28.75:
                                                    if Q.planar_flow > 0.4389738440513611:
                                                        if Q.phi_1 > -0.0019087791442871094:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.009088637307286263:
                                                    if Q.centroid_offset > 0.006821918999776244:
                                                        if Q.planar_flow > 0.3799498379230499:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.011198253370821476:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.016613989137113094:
                                                                    if Q.z_7 > 0.06400897353887558:
                                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 1.554630994796753:
                                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.016885263845324516:
                                                            if Q.z_7 > 0.0604188684374094:
                                                                if Q.planar_flow > 0.4690581113100052:
                                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.3771542310714722:
                                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.06104490906000137:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.005133563885465264:
                                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.013205450493842363:
                                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top5 > 506.234375:
                                                                            if Q.centroid_offset > 0.0028536554891616106:
                                                                                return 'g'   # 78% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.028585869818925858:
                                                        if Q.sum_pt > 657.265625:
                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 3.780624865612481e-05:
                                                            if Q.z_7 > 0.026771174743771553:
                                                                if Q.tau21 > 0.4807320535182953:
                                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.007267124718055129:
                                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.dr_0 > 0.014299975708127022:
                                                                            if Q.z_7 > 0.06669509038329124:
                                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.545910358428955:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.518778562545776:
                                                if Q.log_sum_pt > 6.565682649612427:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.007851667702198029:
                                            if Q.girth > 0.04043292999267578:
                                                if Q.girth2_top3 > 0.0018675848259590566:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 419.1875:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.010833647102117538:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.17370669543743134:
                                                            if Q.z_7 > 0.06381602212786674:
                                                                return 'g'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.04637832194566727:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.017459223978221416:
                                                if Q.z_7 > 0.07618658989667892:
                                                    if Q.centroid_offset > 0.004234368680045009:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 9.882925033569336:
                                                    if Q.centroid_offset > 0.005929610226303339:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.07470010221004486:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.00033967860508710146:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.3205106556415558:
                                    if Q.centroid_offset > 0.034214990213513374:
                                        if Q.log_sum_pt > 6.385088920593262:
                                            if Q.lam2 > 0.0002929282054537907:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 47.34375:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00012797356612281874:
                                                        if Q.phi_1 > -0.0164031982421875:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.306759834289551:
                                                if Q.z_dr_0p05_0p1 > 0.35330405831336975:
                                                    if Q.centroid_offset > 0.04537699185311794:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00031066469091456383:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.1254580616950989:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.005169783951714635:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 447.8125:
                                                    if Q.dr_0 > 0.02045162208378315:
                                                        if Q.z_7 > 0.07071810960769653:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.006148009328171611:
                                        if Q.centroid_offset > 0.03020649403333664:
                                            if Q.mass > 21.207287788391113:
                                                if Q.max_dr > 0.0849757008254528:
                                                    if Q.z_dr_0p05_0p1 > 0.891478955745697:
                                                        if Q.centroid_offset > 0.05344044230878353:
                                                            return 't'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.035368412733078:
                                                        if Q.girth2 > 0.003545031533576548:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00042642631160560995:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 569.0625:
                                                    if Q.z_dr_0p05_0p1 > 0.3882249593734741:
                                                        if Q.LHA > 0.2566898316144943:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.007826254237443209:
                                                if Q.mass > 23.61601161956787:
                                                    if Q.centroid_offset > 0.023316138423979282:
                                                        if Q.max_dr > 0.08662672340869904:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 434.921875:
                                            if Q.dr_0 > 0.01690475083887577:
                                                if Q.C2 > 0.011272421106696129:
                                                    if Q.centroid_offset > 0.004508126759901643:
                                                        if Q.z_7 > 0.0646512359380722:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
