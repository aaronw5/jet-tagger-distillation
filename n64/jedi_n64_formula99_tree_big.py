"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 80.05% (the formula: 80.45%); same class as the formula for 94.63% of jets.  932 leaves, depth 17.
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
                            if Q.z_top20_slots > 0.9764512479305267:
                                return 'q'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 184.34141540527344:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.5552297830581665:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.051109399646520615:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.05019816569983959:
                                if Q.tau32 > 0.7033082544803619:
                                    if Q.girth2_top5 > 0.011607225053012371:
                                        if Q.mass > 172.8143539428711:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.01500893710181117:
                                                if Q.z_top50_slots > 0.9758247435092926:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 36.5:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 32% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.5883776247501373:
                                    if Q.girth > 0.11538562551140785:
                                        if Q.z_top50_slots > 0.9730723202228546:
                                            if Q.mass_top30 > 145.4321517944336:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.10456996411085129:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 148.52685546875:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top30_slots > 0.9949806928634644:
                                            return 'q'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.9956982135772705:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.09395112097263336:
                                        if Q.z_top50_slots > 0.9577785134315491:
                                            return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9706028401851654:
                                            if Q.mass_over_sum_pt > 0.13439955562353134:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.n_real_top50 > 39.5:
                            if Q.e2 > 0.03803045116364956:
                                if Q.tau32 > 0.5466742515563965:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 177.15985107421875:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.1400120034813881:
                                            if Q.z_top50_slots > 0.954947829246521:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.077959299087524:
                                return 'g'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top3 > 640.78125:
                                    return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 56% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.040633074939250946:
                        if Q.mass > 191.57489776611328:
                            if Q.log_sum_pt > 7.0731730461120605:
                                return 'g'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9671063125133514:
                                    if Q.D2 > 2.0399049520492554:
                                        return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.026075519621372223:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04711730778217316:
                                return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 178.5888442993164:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top40 > 0.01645046565681696:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9766803979873657:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.0345465075224638:
                            if Q.mass > 175.25045776367188:
                                return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.1265002265572548:
                                    if Q.z_top50_slots > 0.9690555334091187:
                                        return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.08345481380820274:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 58.5:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 51% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 90% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.03476156108081341:
                    if Q.e2 > 0.04258318990468979:
                        if Q.lam1 > 0.03234411031007767:
                            if Q.tau21 > 0.3124667704105377:
                                if Q.sum_pt > 1043.458984375:
                                    if Q.z_top50_slots > 0.9609233438968658:
                                        if Q.C2 > 0.14291159808635712:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.5264293551445007:
                                        if Q.z_top50_slots > 0.9469876289367676:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.4573024958372116:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9697490930557251:
                                    if Q.mass_top50 > 184.92730712890625:
                                        if Q.planar_flow > 0.2658484876155853:
                                            if Q.sum_pt > 1011.4296875:
                                                return 'g'   # 36% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1039.5137939453125:
                                                if Q.z_top50_slots > 0.9860529601573944:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.46222028136253357:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.23093950003385544:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.852943658828735:
                                                if Q.sum_pt > 995.5821533203125:
                                                    return 'q'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 904.00830078125:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 999.7333984375:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0349606778472662:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.009727940894663334:
                                if Q.sum_pt_top40 > 703.1240234375:
                                    if Q.sum_pt_top3 > 584.234375:
                                        if Q.mass_over_sum_pt_sq > 0.019541057758033276:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 997.146240234375:
                                                if Q.z_dr_0p2_0p4 > 0.11274601891636848:
                                                    if Q.tau21 > 0.5142765045166016:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.1780645027756691:
                                                            if Q.mass_over_sum_pt_sq > 0.0186310987919569:
                                                                return 't'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        return 't'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.950420618057251:
                                            if Q.sum_pt > 1053.096435546875:
                                                if Q.tau32 > 0.5778602063655853:
                                                    if Q.z_top50_slots > 0.9709924757480621:
                                                        if Q.dr_0 > 0.07823403924703598:
                                                            if Q.lam1 > 0.029829885810613632:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.1360185295343399:
                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.06352002173662186:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 182.9861068725586:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_top50 > 141.6480484008789:
                                                                if Q.tau32 > 0.8231624066829681:
                                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.010333584621548653:
                                                    if Q.sum_pt_top50 > 821.724609375:
                                                        if Q.girth > 0.10010212287306786:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top40 > 1003.238525390625:
                                                                if Q.girth2_top5 > 0.004269634839147329:
                                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.11222470179200172:
                                                                        return 't'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.047432124614715576:
                                                            return 't'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.11902457103133202:
                                                                if Q.mass_over_sum_pt > 0.14609605073928833:
                                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_top50_slots > 0.9705540835857391:
                                                                        return 't'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 3.5:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.950318813323975:
                                                if Q.tau32 > 0.5779730677604675:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9411463439464569:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 134.17008209228516:
                                                    if Q.tau32 > 0.6705085039138794:
                                                        if Q.sum_pt > 1022.8486328125:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 846.3681640625:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.027106687426567078:
                                                            return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.6261297464370728:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9741424322128296:
                                        return 't'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top30 > 99.71182632446289:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.sum_pt_top50 > 973.50390625:
                                        if Q.mass > 98.19168472290039:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.018652942031621933:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 88.01320266723633:
                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.944725036621094:
                            if Q.z_top50_slots > 0.9677032232284546:
                                if Q.sum_pt_top3 > 574.90625:
                                    if Q.z_dr_0p2_0p4 > 0.04400908201932907:
                                        if Q.z_top20_slots > 0.8878012299537659:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1067.4774169921875:
                                        if Q.tau32 > 0.5035877823829651:
                                            if Q.dr_0 > 0.0893140472471714:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 144.7144317626953:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top40 > 0.009720120579004288:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 45% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0575261190533638:
                                            if Q.tau32 > 0.5670660436153412:
                                                if Q.mass_top50 > 141.04200744628906:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 49.5:
                                                        if Q.dr_0 > 0.07460641488432884:
                                                            return 't'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top30_slots > 0.9619770646095276:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top30 > 0.009478701744228601:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.5092714726924896:
                                    return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.14318829774856567:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 768.064697265625:
                                if Q.sum_pt_top3 > 548.140625:
                                    if Q.sum_pt_top50 > 977.146240234375:
                                        if Q.z_dr_0p2_0p4 > 0.06650355085730553:
                                            if Q.tau32 > 0.4534441828727722:
                                                if Q.mass > 98.42814636230469:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.850315511226654:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9627520143985748:
                                        if Q.mass_over_sum_pt_sq > 0.009464744944125414:
                                            if Q.log_sum_pt > 6.897769927978516:
                                                if Q.girth2_top5 > 0.003686080453917384:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        if Q.z_top50_slots > 0.97605100274086:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.11348594352602959:
                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.7516360282897949:
                                                                    if Q.dr_0 > 0.07145671918988228:
                                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.09917636960744858:
                                                            if Q.mass_top30 > 89.40667343139648:
                                                                return 't'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 458.9375:
                                                        if Q.C2 > 0.12391713634133339:
                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p2_0p4 > 0.08153483271598816:
                                                                return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 797.5244140625:
                                                    return 't'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 102.96265029907227:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 954.5384521484375:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.90522575378418:
                                            if Q.tau32 > 0.5756949782371521:
                                                if Q.mass_top30 > 114.6688461303711:
                                                    if Q.z_top50_slots > 0.941064715385437:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.014349434524774551:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top40 > 119.18538665771484:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.0941450335085392:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 122.82132339477539:
                                                if Q.sum_pt_top50 > 864.115234375:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6130897104740143:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 966.0810546875:
                                                    if Q.e2 > 0.039967844262719154:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.5829666554927826:
                                                            if Q.dr_0 > 0.07066560909152031:
                                                                if Q.max_dr > 0.3212658017873764:
                                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 839.513671875:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.020632218569517136:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.12320207059383392:
                                    if Q.mass_over_sum_pt_sq > 0.028392042964696884:
                                        if Q.sum_pt_top50 > 732.650390625:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9715480208396912:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 750.8291015625:
                                                if Q.n_dr_0p2_0p4 > 13.5:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 760.368408203125:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 40% of the training jets here get this class from the formula
                else:
                    if Q.z_top50_slots > 0.9754540324211121:
                        if Q.sum_pt_top40 > 949.234375:
                            if Q.z_top30_slots > 0.9518505036830902:
                                if Q.z_dr_0_0p05 > 0.770574301481247:
                                    if Q.log_sum_pt > 6.966147184371948:
                                        if Q.sum_pt_top3 > 552.65625:
                                            if Q.n_particles > 50.5:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 462.0625:
                                            if Q.mass_top50 > 93.27505111694336:
                                                if Q.n_particles > 60.5:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1022.7696533203125:
                                                if Q.tau32 > 0.7293068766593933:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 970.455078125:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.898285388946533:
                                        if Q.z_dr_0p2_0p4 > 0.06298207864165306:
                                            if Q.sum_pt > 1065.6890869140625:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 446.03125:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.336721658706665:
                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.942647695541382:
                                    if Q.z_dr_0p2_0p4 > 0.04350392334163189:
                                        if Q.tau32 > 0.4570355713367462:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.02809706423431635:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1061.774658203125:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.04764286056160927:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.3333702087402344:
                                        if Q.sum_pt > 1007.855224609375:
                                            if Q.tau32 > 0.5448411405086517:
                                                if Q.sum_pt_top2 > 396.21875:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 61.5:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7076468169689178:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.027828986756503582:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 396.375:
                                                if Q.tau32 > 0.5216396749019623:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.026058536022901535:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 62.5:
                                                        if Q.tau32 > 0.7496205270290375:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 7.5:
                                            if Q.z_dr_0p2_0p4 > 0.059200093150138855:
                                                if Q.sum_pt_top3 > 423.375:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 97.68632888793945:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023076952435076237:
                                if Q.sum_pt_top40 > 737.2255859375:
                                    if Q.sum_pt_top40 > 937.3095703125:
                                        if Q.e2 > 0.025954311713576317:
                                            if Q.sum_pt_top3 > 580.1875:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.026805021800100803:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9805895090103149:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07568086683750153:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top40 > 0.009440755471587181:
                                    return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 63.5:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 40% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.029353326186537743:
                            if Q.log_sum_pt > 6.901112079620361:
                                if Q.tau32 > 0.4898948073387146:
                                    if Q.log_sum_pt > 6.922951698303223:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9640149176120758:
                                            if Q.dr_0 > 0.0696381963789463:
                                                return 't'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 18.5:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9628602862358093:
                                    if Q.sum_pt_top50 > 799.4267578125:
                                        if Q.log_sum_pt > 6.8888397216796875:
                                            if Q.e2 > 0.0318782664835453:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.5441127717494965:
                                        if Q.dr_0 > 0.08399391174316406:
                                            if Q.z_top50_slots > 0.9455567002296448:
                                                if Q.log_sum_pt > 6.8596508502960205:
                                                    if Q.lam1 > 0.01332058385014534:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 4.5:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 129.55992126464844:
                                                if Q.z_top50_slots > 0.9474378824234009:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.852937459945679:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 823.267578125:
                                                        if Q.mass_over_sum_pt_sq > 0.015177957247942686:
                                                            if Q.z_top50_slots > 0.9466590881347656:
                                                                return 't'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 764.076171875:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.014972304459661245:
                                if Q.sum_pt > 989.31298828125:
                                    return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9507319629192352:
                                        if Q.sum_pt_top50 > 796.8759765625:
                                            if Q.tau32 > 0.7187188267707825:
                                                if Q.sum_pt > 940.771484375:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top50 > 122.25429916381836:
                                            return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.966256856918335:
                                    if Q.sum_pt_top50 > 929.2431640625:
                                        if Q.sum_pt > 997.95068359375:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.3053302615880966:
                                                return 't'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.08199570700526237:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 96% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.976256847381592:
                    if Q.z_dr_0p2_0p4 > 0.005690349265933037:
                        if Q.log_sum_pt > 7.019988775253296:
                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.01727909315377474:
                                if Q.max_dr > 0.28304488956928253:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9703144133090973:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 95.39437866210938:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 82.52221298217773:
                                        if Q.max_dr > 0.3457719385623932:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 7.077538967132568:
                            if Q.mass > 93.28449249267578:
                                return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 81.05668258666992:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top40 > 0.0046270741149783134:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 95.25505828857422:
                                return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 82.36185836791992:
                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 62% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02443495485931635:
                        if Q.n_dr_0p2_0p4 > 7.5:
                            if Q.D2 > 1.6132236123085022:
                                if Q.log_sum_pt > 6.892338514328003:
                                    if Q.mass > 93.58331680297852:
                                        if Q.tau32 > 0.6002534627914429:
                                            if Q.D2 > 2.1962382793426514:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.955891847610474:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top40 > 0.0060686839278787374:
                                            if Q.z_top50_slots > 0.972879022359848:
                                                if Q.max_dr > 0.36836859583854675:
                                                    if Q.z_top50_slots > 0.9819633960723877:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6197776198387146:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.977630615234375:
                                        return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.09072207286953926:
                                    return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.3603467643260956:
                                        return 't'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 969.06396484375:
                                if Q.mass > 95.58185195922852:
                                    if Q.e2 > 0.02960700262337923:
                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 75.34086608886719:
                                        if Q.mass > 92.50194549560547:
                                            if Q.mass_top40 > 80.12435150146484:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.7275065183639526:
                                            if Q.dr_0 > 0.058734066784381866:
                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9723838269710541:
                                    return 't'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 2.5:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.31199921667575836:
                            if Q.mass_top30 > 65.44631958007812:
                                if Q.mass > 92.39786911010742:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top40 > 0.00629658717662096:
                                        if Q.sum_pt > 989.75830078125:
                                            if Q.girth2_top30 > 0.005916615482419729:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.6494426429271698:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.5777333676815033:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 59% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top40 > 72.19365310668945:
                                if Q.mass > 93.29387283325195:
                                    return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 4.665666580200195:
                                        if Q.tau32 > 0.7023230791091919:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 84% of the training jets here get this class from the formula
            else:
                if Q.mass > 98.54790496826172:
                    if Q.n_real_top50 > 39.5:
                        if Q.log_sum_pt > 7.001473426818848:
                            if Q.n_dr_0p2_0p4 > 5.5:
                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1230.194091796875:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.7058750987052917:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 7.5:
                                if Q.girth2_top5 > 0.00478395028039813:
                                    return 't'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top2 > 499.0:
                                        if Q.log_sum_pt > 6.984711647033691:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 94% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0p2_0p4 > 6.5:
                            if Q.log_sum_pt > 7.076107978820801:
                                if Q.n_real_top50 > 32.5:
                                    return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 12.5:
                                    return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0_0p05 > 0.7873814702033997:
                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 47% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1259.9564208984375:
                                return 'g'   # 44% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.mass_top50 > 85.86580657958984:
                        if Q.girth2_top20 > 0.0035427871625870466:
                            if Q.sum_pt_top50 > 963.1859130859375:
                                if Q.mass > 95.87205123901367:
                                    if Q.n_dr_0p2_0p4 > 10.5:
                                        if Q.girth2_top30 > 0.007001859834417701:
                                            if Q.z_dr_0p2_0p4 > 0.04804992116987705:
                                                if Q.lam2 > 0.0005713152349926531:
                                                    if Q.girth2_top40 > 0.008315124548971653:
                                                        if Q.sum_pt_top3 > 540.25:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0008561608556192368:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.004810601007193327:
                                                if Q.tau32 > 0.5485360026359558:
                                                    if Q.girth2_top40 > 0.007641479838639498:
                                                        return 't'   # 34% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0007965347031131387:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.005416144849732518:
                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 95.75008010864258:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 980.607177734375:
                                        if Q.n_dr_0p2_0p4 > 18.5:
                                            if Q.mass > 93.30462646484375:
                                                if Q.girth2_top30 > 0.0065785618498921394:
                                                    if Q.mass_over_sum_pt > 0.09289851412177086:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.0058968758676201105:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 86.4323616027832:
                                                if Q.girth2_top20 > 0.004500471753999591:
                                                    if Q.mass_over_sum_pt_sq > 0.008755175862461329:
                                                        if Q.girth2_top20 > 0.007469347212463617:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.38018985092639923:
                                                                if Q.tau21 > 0.3798281103372574:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 93.9083480834961:
                                                        if Q.n_dr_0p2_0p4 > 6.5:
                                                            if Q.girth2_top30 > 0.0057595642283558846:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_particles > 58.5:
                                                            if Q.max_dr > 0.38671447336673737:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.0814317911863327:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.04428535886108875:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.32581736147403717:
                                                            if Q.lam2 > 0.0005300506891217083:
                                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.z_dr_0p2_0p4 > 0.05496268533170223:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    if Q.D2 > 1.8844610452651978:
                                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.012650711927562952:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 5.5:
                                    if Q.z_dr_0_0p05 > 0.8390665352344513:
                                        if Q.n_dr_0p2_0p4 > 15.5:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.33513587713241577:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 938.7401123046875:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.365343376994133:
                                if Q.sum_pt > 1110.0357666015625:
                                    if Q.n_dr_0p2_0p4 > 6.5:
                                        if Q.n_particles > 48.5:
                                            if Q.max_dr > 0.3354652673006058:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.17878243327140808:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 92.81747817993164:
                                        return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 59.5:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 93.71124649047852:
                                    if Q.mass_top30 > 88.6093521118164:
                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.016560176387429237:
                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.2992691844701767:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.C2 > 0.04242449440062046:
                            if Q.girth2_top40 > 0.005356325069442391:
                                if Q.log_sum_pt > 6.87085223197937:
                                    if Q.mass > 85.36334609985352:
                                        if Q.girth2_top30 > 0.005075602093711495:
                                            if Q.sum_pt_top50 > 974.3916015625:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 9.5:
                                                    if Q.z_dr_0p2_0p4 > 0.05045555345714092:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.38157713413238525:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.805328845977783:
                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1014.611328125:
                                                if Q.max_dr > 0.3377996236085892:
                                                    if Q.lam2 > 0.0010340727749280632:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.025991076603531837:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3637622743844986:
                                                    if Q.D2 > 2.080514907836914:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 6.5:
                                                            return 't'   # 42% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.331433966755867:
                                        return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.0036433464847505093:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        if Q.z_top50_slots > 0.9996245801448822:
                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.537796825170517:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt > 0.08474580198526382:
                                if Q.n_dr_0p2_0p4 > 5.5:
                                    if Q.log_sum_pt > 6.905799388885498:
                                        if Q.mass > 85.77479553222656:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 41% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.28228169679641724:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 939.3609619140625:
                                        if Q.max_dr > 0.36938048899173737:
                                            if Q.mass > 85.1570816040039:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.02614403422921896:
                                    if Q.max_dr > 0.296394482254982:
                                        if Q.z_top30_slots > 0.9728155136108398:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 85.45121383666992:
                                                if Q.mass_over_sum_pt > 0.08017713576555252:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 41% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.4449602961540222:
                                                    return 'Z'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 85.31033325195312:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1048.5660400390625:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.24886777251958847:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 51.5:
                                        if Q.mass_over_sum_pt > 0.07683705538511276:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.mass_over_sum_pt > 0.0849158950150013:
                if Q.mass_over_sum_pt_sq > 0.00753265293315053:
                    if Q.mass_top30 > 66.09911346435547:
                        if Q.lam2 > 0.00029209516651462764:
                            if Q.z_dr_0p2_0p4 > 0.052502017468214035:
                                if Q.mass_top50 > 79.41117095947266:
                                    if Q.sum_pt_top50 > 947.306640625:
                                        if Q.C2 > 0.08303756266832352:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 58.5:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.1115325428545475:
                                                return 't'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top30_slots > 0.998534232378006:
                                                    if Q.mass_over_sum_pt_sq > 0.008428180124610662:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.008644762448966503:
                                        return 't'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 48.0:
                                            return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 946.5751953125:
                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt_sq > 0.008153848815709352:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 33.5:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top40 > 83.73870849609375:
                                return 'Z'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 65.80865859985352:
                                    return 'W'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 53% of the training jets here get this class from the formula
                    else:
                        if Q.D2 > 2.588046431541443:
                            if Q.n_particles > 60.5:
                                return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.7292757034301758:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 49% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.980586975812912:
                                return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 69% of the training jets here get this class from the formula
                else:
                    if Q.lam2 > 0.0005449870077427477:
                        if Q.D2 > 2.584362268447876:
                            if Q.sum_pt_top50 > 967.166748046875:
                                return 'Z'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top3 > 428.375:
                                    return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 4.5:
                                if Q.z_dr_0p2_0p4 > 0.05493026040494442:
                                    return 'W'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 83.31649017333984:
                                        if Q.C2 > 0.06119270250201225:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 83.30009078979492:
                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.8430562019348145:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 83.50281524658203:
                            if Q.C2 > 0.05186283588409424:
                                return 'Z'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.2855226844549179:
                                    return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 898.74365234375:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    if Q.girth > 0.06508507207036018:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                return 't'   # 64% of the training jets here get this class from the formula
            else:
                if Q.girth2_top20 > 0.0029952499317005277:
                    if Q.sum_pt_top50 > 948.9957275390625:
                        if Q.mass > 83.58809661865234:
                            if Q.D2 > 3.0743356943130493:
                                if Q.n_particles > 58.5:
                                    if Q.e2 > 0.023519473150372505:
                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.8483365774154663:
                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 84.09164810180664:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 77.23736572265625:
                                    if Q.planar_flow > 0.6816665828227997:
                                        if Q.planar_flow > 0.9005409181118011:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.02463877759873867:
                                            if Q.C2 > 0.044134512543678284:
                                                if Q.max_dr > 0.27443842589855194:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.005333576118573546:
                                        if Q.mass > 84.15914154052734:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9895964562892914:
                                            if Q.girth2_top20 > 0.004262731876224279:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 14.5:
                                if Q.mass_top40 > 74.90768051147461:
                                    if Q.lam2 > 0.000918490084586665:
                                        if Q.sum_pt_top40 > 980.26611328125:
                                            if Q.mass > 81.52416610717773:
                                                if Q.C2 > 0.09333035349845886:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0_0p05 > 0.8292622566223145:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 41% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 82.62108993530273:
                                            if Q.D2 > 4.519561052322388:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 970.117919921875:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 60.5:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.40057289600372314:
                                            if Q.mass > 74.62080383300781:
                                                if Q.sum_pt_top50 > 987.7503662109375:
                                                    if Q.n_dr_0p2_0p4 > 17.5:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 983.9407958984375:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 72.18085098266602:
                                    if Q.n_particles > 62.5:
                                        if Q.girth2_top20 > 0.004140696721151471:
                                            return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 80.79531478881836:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 4.567371845245361:
                                            if Q.mass > 82.39163970947266:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 964.7196044921875:
                                                if Q.girth2_top20 > 0.003687008284032345:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 81.83098983764648:
                                                        if Q.z_top50_slots > 0.999674528837204:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top30 > 0.004381655715405941:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0006888869102112949:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.LHA > 0.2617262303829193:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 81.481201171875:
                                        if Q.lam1 > 0.005282277707010508:
                                            return 'W'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 9.5:
                                            if Q.planar_flow > 0.3892272710800171:
                                                if Q.mass > 71.92625045776367:
                                                    if Q.sum_pt > 985.510009765625:
                                                        if Q.girth2_top40 > 0.004551534540951252:
                                                            if Q.n_particles > 63.5:
                                                                return 'g'   # 49% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top30 > 0.004622802371159196:
                                                        if Q.sum_pt > 996.8134765625:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.979827642440796:
                                                if Q.mass > 80.40584564208984:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        if Q.mass > 77.15321350097656:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.892925262451172:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.004814663203433156:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top20_slots > 0.8762781023979187:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.planar_flow > 0.25221243500709534:
                            if Q.LHA > 0.24181237071752548:
                                if Q.n_dr_0p2_0p4 > 5.5:
                                    if Q.C2 > 0.05174311622977257:
                                        if Q.sum_pt_top2 > 353.1875:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 934.5511474609375:
                                        if Q.z_top50_slots > 0.9639898538589478:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top50 > 73.28129577636719:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9986022114753723:
                                    if Q.mass_top30 > 74.13975524902344:
                                        return 'W'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 908.982177734375:
                                return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p2_0p4 > 0.0017388578271493316:
                                    if Q.z_dr_0_0p05 > 0.70644211769104:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 91% of the training jets here get this class from the formula
                else:
                    if Q.planar_flow > 0.40477268397808075:
                        if Q.log_sum_pt > 6.965287685394287:
                            if Q.e2 > 0.018984285183250904:
                                if Q.n_particles > 54.5:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 73.77200698852539:
                                        if Q.mass > 82.27883911132812:
                                            return 'g'   # 47% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6750978231430054:
                                                if Q.n_real_top50 > 40.5:
                                                    if Q.lam1 > 0.00336047844029963:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.528533935546875:
                                            if Q.n_particles > 41.5:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 40% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.n_real_top50 > 38.5:
                                    if Q.girth2_top30 > 0.0037432421231642365:
                                        if Q.mass > 80.38759231567383:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 5.795248985290527:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.0047281947918236256:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1326.68115234375:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top20_slots > 0.9678115844726562:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 57.5:
                                if Q.n_particles > 61.5:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.21442659199237823:
                                        if Q.mass_over_sum_pt_sq > 0.006343456683680415:
                                            return 'g'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.19794244319200516:
                                    if Q.mass > 82.62504577636719:
                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 986.638916015625:
                                            if Q.mass > 73.36276626586914:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.0037303095450624824:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 74.76576614379883:
                                        if Q.n_real_top50 > 45.5:
                                            if Q.mass > 80.57500839233398:
                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 992.546142578125:
                                                    if Q.lam1 > 0.0043329300824552774:
                                                        return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 12.5:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 82.24787139892578:
                            if Q.e2 > 0.020253089256584644:
                                return 'W'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top40 > 0.005720446119084954:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 80.04505920410156:
                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.01411773869767785:
                                if Q.planar_flow > 0.31231167912483215:
                                    if Q.mass_top40 > 73.65520095825195:
                                        return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top40 > 0.0043340378906577826:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 522.34375:
                                                return 'W'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1080.2088623046875:
                                    if Q.n_particles > 41.5:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 59% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 74% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 45.5:
                if Q.sum_pt > 1043.142333984375:
                    if Q.girth2_top30 > 0.0036832899786531925:
                        if Q.mass > 79.69818878173828:
                            return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top40 > 65.64867782592773:
                                if Q.z_dr_0p2_0p4 > 0.008130873553454876:
                                    if Q.tau32 > 0.6021642684936523:
                                        if Q.sum_pt_top2 > 486.6875:
                                            return 'q'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top30 > 0.0042026781011372805:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.5866662859916687:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 56% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1064.343505859375:
                            if Q.girth2_top30 > 0.003330855746753514:
                                if Q.n_particles > 57.5:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 53.5:
                                if Q.e2 > 0.018206236883997917:
                                    if Q.n_particles > 60.5:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.0009867773042060435:
                                    if Q.sum_pt_top2 > 428.5:
                                        if Q.tau32 > 0.8584360778331757:
                                            if Q.girth2_top20 > 0.0014845514670014381:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.tau32 > 0.7448329031467438:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 0.00011824336252175272:
                                            if Q.tau32 > 0.8686195313930511:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.79022789001465:
                            if Q.mass_over_sum_pt_sq > 0.006536588305607438:
                                return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 7.5:
                                    if Q.n_particles > 62.5:
                                        if Q.tau32 > 0.5856288373470306:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.0032380843767896295:
                                            if Q.log_sum_pt > 6.903154611587524:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 45% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.922990083694458:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top20_slots > 0.873786985874176:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 66.19049072265625:
                                        if Q.mass > 80.53207397460938:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.88712739944458:
                                            if Q.z_dr_0p2_0p4 > 0.008321854751557112:
                                                if Q.planar_flow > 0.4820674806833267:
                                                    if Q.tau32 > 0.6204931735992432:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top30 > 0.004290781915187836:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 70.52891159057617:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9726859629154205:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 62.5:
                                if Q.mass_top10 > 22.262697219848633:
                                    if Q.z_dr_0p2_0p4 > 0.01095005078241229:
                                        if Q.sum_pt_top3 > 434.578125:
                                            if Q.sum_pt > 1004.8348388671875:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 969.63525390625:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9720169901847839:
                                            if Q.tau32 > 0.7474168837070465:
                                                if Q.dr_0 > 0.03792084939777851:
                                                    if Q.z_dr_0p2_0p4 > 0.00549212796613574:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.932945013046265:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 61.22325325012207:
                                                if Q.sum_pt_top50 > 927.9140625:
                                                    if Q.z_top50_slots > 0.9486218690872192:
                                                        if Q.z_dr_0p2_0p4 > 0.006693462608382106:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6691858768463135:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.015194171108305454:
                                            if Q.sum_pt_top50 > 942.54296875:
                                                if Q.girth2_top20 > 0.0035238584969192743:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 63.23416709899902:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 16.93508815765381:
                                    if Q.n_dr_0p2_0p4 > 6.5:
                                        if Q.sum_pt_top40 > 915.8857421875:
                                            if Q.sum_pt > 1012.908203125:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 273.328125:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6941853165626526:
                                                        if Q.dr_0 > 0.028201390989124775:
                                                            if Q.z_dr_0p2_0p4 > 0.022832194343209267:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 70.2504653930664:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.026431958191096783:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1018.104736328125:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.8340738415718079:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        if Q.tau32 > 0.6801162958145142:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8317672312259674:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 909.3720703125:
                                                if Q.girth2_top20 > 0.0013061826466582716:
                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.930432081222534:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top10 > 11.137710094451904:
                            if Q.log_sum_pt > 6.822614669799805:
                                if Q.lam1 > 0.003927669022232294:
                                    if Q.sum_pt > 987.209228515625:
                                        if Q.girth2_top20 > 0.0033731881994754076:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top20_slots > 0.8957912027835846:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.004369109170511365:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.026881407015025616:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.3524601310491562:
                                            if Q.mass_top40 > 67.70828628540039:
                                                if Q.planar_flow > 0.5100003480911255:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 953.9033203125:
                                                if Q.girth > 0.06262477859854698:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1021.5855712890625:
                                        if Q.girth2_top20 > 0.0010534704197198153:
                                            if Q.n_particles > 54.5:
                                                if Q.girth2_top20 > 0.001556095143314451:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        if Q.tau32 > 0.7264724373817444:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 309.125:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.tau32 > 0.690789133310318:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.sum_pt_top2 > 433.6875:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00043391088547650725:
                                                        if Q.tau32 > 0.7645359635353088:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_particles > 49.5:
                                                                return 'g'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 53.5:
                                            if Q.girth2_top20 > 0.001274359179660678:
                                                if Q.mass_top30 > 58.02984046936035:
                                                    if Q.z_top30_slots > 0.9420565962791443:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.sum_pt_top2 > 350.515625:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top40 > 994.37158203125:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.8275929689407349:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.0007179241220001131:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    if Q.sum_pt_top2 > 273.0625:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.sum_pt_top2 > 298.96875:
                                        if Q.tau32 > 0.8411693871021271:
                                            if Q.tau21 > 0.44142383337020874:
                                                if Q.n_particles > 49.5:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top30 > 0.006185119738802314:
                                                return 't'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.6666586101055145:
                                            if Q.mass_top10 > 28.875123977661133:
                                                if Q.mass_top40 > 65.17551803588867:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.047738462686538696:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top10 > 17.803220748901367:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 241.484375:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 732.367919921875:
                                        if Q.tau21 > 0.2862232178449631:
                                            if Q.girth2_top30 > 0.000973594665993005:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 39% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 5.5:
                                if Q.sum_pt_top3 > 472.1875:
                                    if Q.log_sum_pt > 6.923426866531372:
                                        if Q.girth2_top20 > 0.0008040850516408682:
                                            if Q.sum_pt_top3 > 542.484375:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9997185468673706:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top10 > 10.321775436401367:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9997098445892334:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 5.839918776473496e-05:
                                                if Q.sum_pt_top40 > 949.926025390625:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9996799528598785:
                                        if Q.sum_pt_top40 > 913.93994140625:
                                            if Q.sum_pt > 1013.092529296875:
                                                if Q.girth2_top20 > 0.001602581818588078:
                                                    return 'q'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 9.207057155435905e-05:
                                                    if Q.tau32 > 0.8737220764160156:
                                                        if Q.n_dr_0p2_0p4 > 7.5:
                                                            if Q.e2 > 0.011498593259602785:
                                                                return 'q'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.8392283320426941:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 370.1875:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.7111323177814484:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.0006795304070692509:
                                            if Q.log_sum_pt > 6.834955453872681:
                                                if Q.sum_pt_top40 > 993.8251953125:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top20 > 0.0016793080139905214:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top50 > 958.4588623046875:
                                                            if Q.dr_0 > 0.005963780917227268:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.939023971557617:
                                    if Q.dr_0 > 0.014566697180271149:
                                        return 'q'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 47.5:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 906.3505859375:
                                        if Q.log_sum_pt > 6.920902490615845:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                if Q.planar_flow > 0.9356267750263214:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 57% of the training jets here get this class from the formula
            else:
                if Q.log_sum_pt > 7.005598783493042:
                    if Q.n_particles > 35.5:
                        if Q.girth2_top20 > 0.0012839192058891058:
                            if Q.n_real_top50 > 40.5:
                                if Q.sum_pt_top2 > 558.21875:
                                    if Q.sum_pt_top50 > 1137.4351806640625:
                                        if Q.e2 > 0.016410685144364834:
                                            return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 456.75:
                                    if Q.sum_pt_top40 > 1237.6522216796875:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.n_real_top50 > 38.5:
                                if Q.tau32 > 0.7178767323493958:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 41.5:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1147.052001953125:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 1197.4432373046875:
                                    if Q.tau32 > 0.6598040759563446:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top10 > 12.786488056182861:
                                        if Q.sum_pt_top2 > 423.375:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 653.1875:
                                            if Q.sum_pt_top50 > 1128.2152099609375:
                                                if Q.tau32 > 0.802990049123764:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.n_real_top50 > 31.5:
                            if Q.sum_pt_top40 > 1265.4287109375:
                                if Q.e2 > 0.009602298960089684:
                                    if Q.tau32 > 0.714376300573349:
                                        if Q.sum_pt_top50 > 1358.4298095703125:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 486.6875:
                                    if Q.girth2_top5 > 5.2632123697549105e-05:
                                        return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 1191.7833251953125:
                                            if Q.tau32 > 0.8961449265480042:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.11613724008202553:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0001235112504218705:
                                            if Q.sum_pt > 1168.6890869140625:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0002245202413178049:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.2044103145599365:
                                if Q.n_real_top50 > 27.5:
                                    if Q.girth > 0.013255878817290068:
                                        return 'q'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 1621.319091796875:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.003118723980151117:
                                    return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 98% of the training jets here get this class from the formula
                else:
                    if Q.lam1 > 0.003603111137636006:
                        if Q.lam2 > 0.00035261810990050435:
                            if Q.sum_pt > 987.73486328125:
                                if Q.lam1 > 0.003912289394065738:
                                    if Q.planar_flow > 0.36918793618679047:
                                        if Q.tau32 > 0.6309210062026978:
                                            if Q.C2 > 0.038175612688064575:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 67.53345489501953:
                                        return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.601635068655014:
                                            if Q.D2 > 1.0742190480232239:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 1.1560149192810059:
                                    if Q.sum_pt_top3 > 349.40625:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.8431458473205566:
                                            if Q.z_dr_0p2_0p4 > 0.029290092177689075:
                                                if Q.z_top30_slots > 0.9908996224403381:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 877.1978759765625:
                                        if Q.sum_pt_top50 > 953.322509765625:
                                            return 'q'   # 39% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 64% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 974.886474609375:
                                if Q.planar_flow > 0.2133009061217308:
                                    if Q.mass > 65.0866813659668:
                                        if Q.C2 > 0.06473572179675102:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.23966574668884277:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.08599627390503883:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 64.53638076782227:
                                    if Q.z_dr_0p2_0p4 > 0.0014264553901739419:
                                        if Q.C2 > 0.04001184552907944:
                                            if Q.D2 > 2.6252260208129883:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.9607253968715668:
                                        return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 45% of the training jets here get this class from the formula
                    else:
                        if Q.n_particles > 40.5:
                            if Q.sum_pt_top50 > 1051.2225341796875:
                                if Q.girth2_top20 > 0.0010719100828282535:
                                    if Q.sum_pt_top3 > 605.5625:
                                        if Q.girth2_top20 > 0.0012953045079484582:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1074.89453125:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            if Q.log_sum_pt > 6.978382349014282:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 501.875:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        if Q.tau32 > 0.7436729967594147:
                                            if Q.sum_pt_top2 > 512.875:
                                                if Q.log_sum_pt > 6.98637318611145:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 7.049688065308146e-05:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.969886779785156:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.664447009563446:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 1076.16943359375:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1075.7784423828125:
                                            if Q.tau21 > 0.6700255274772644:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 9.178879737854004:
                                    if Q.sum_pt > 849.393798828125:
                                        if Q.log_sum_pt > 6.947863340377808:
                                            if Q.sum_pt_top2 > 402.875:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    if Q.girth2_top20 > 0.0009562712803017348:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.sum_pt_top3 > 368.25:
                                                return 'q'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.0018461909494362772:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 836.677978515625:
                                        if Q.sum_pt > 1036.3131103515625:
                                            if Q.sum_pt_top3 > 536.21875:
                                                if Q.tau32 > 0.9279562532901764:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 445.125:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    if Q.girth2_top5 > 9.749570745043457e-05:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.8380386829376221:
                                                            if Q.n_real_top50 > 41.5:
                                                                return 'g'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 353.71875:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            if Q.tau32 > 0.7792421281337738:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.21068278700113297:
                                if Q.sum_pt > 1068.6004638671875:
                                    if Q.n_particles > 36.5:
                                        if Q.sum_pt_top2 > 452.0:
                                            if Q.girth2_top5 > 5.177275124879088e-05:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 651.3125:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1365717649459839:
                                                if Q.n_dr_0p2_0p4 > 3.5:
                                                    if Q.sum_pt_top3 > 432.96875:
                                                        return 'q'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    if Q.tau32 > 0.8032529950141907:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 401.59375:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_real_top50 > 33.5:
                                            if Q.sum_pt_top2 > 415.6875:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 7.108538193278946e-05:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 4.5:
                                                        if Q.tau32 > 0.8294797539710999:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.0033689545234665275:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 224.046875:
                                        if Q.planar_flow > 0.2921026796102524:
                                            if Q.n_particles > 38.5:
                                                if Q.sum_pt > 713.14892578125:
                                                    if Q.log_sum_pt > 6.957838296890259:
                                                        if Q.girth2_top5 > 2.9585317861346994e-05:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 59.813140869140625:
                                                if Q.C2 > 0.08252066001296043:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 54.03442192077637:
                                    if Q.C2 > 0.07892308384180069:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.046672796830534935:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 91% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
