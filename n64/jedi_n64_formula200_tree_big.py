"""JEDI-linear jet tagger, 64 particles, 3 features: the formula with the fewest quantities (31) at the network's accuracy: ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 80.35% (the formula: 81.07%); same class as the formula for 94.63% of jets.  1596 leaves, depth 21.
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
                            if Q.tau21 > 0.49494652450084686:
                                if Q.z_top50_slots > 0.9672821164131165:
                                    if Q.sum_pt_top40 > 1135.6953125:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 189.7772445678711:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0029183782171458006:
                                        if Q.e2 > 0.053090931847691536:
                                            if Q.mass > 186.5312728881836:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04736979305744171:
                                if Q.tau32 > 0.702892005443573:
                                    if Q.e2 > 0.054335420951247215:
                                        if Q.mass > 173.60240936279297:
                                            if Q.tau32 > 0.8355779349803925:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.20332684367895126:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.065663814544678:
                                                if Q.tau21 > 0.15904965996742249:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.05798704922199249:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.28392088413238525:
                                            if Q.e2 > 0.04832485131919384:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1151.699462890625:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.012952537275850773:
                                                    if Q.e2 > 0.05241967737674713:
                                                        if Q.tau32 > 0.8582432866096497:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.011408533900976181:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.1026155836880207:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0018937826971523464:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 7.098966598510742:
                                            if Q.e2 > 0.04996410198509693:
                                                if Q.tau21 > 0.25906839966773987:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.6474721133708954:
                                    if Q.mass > 118.38384628295898:
                                        if Q.lam2 > 0.003728926181793213:
                                            if Q.e2 > 0.041471049189567566:
                                                if Q.z_top50_slots > 0.9662809371948242:
                                                    return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.04412606358528137:
                                                if Q.tau32 > 0.7471466362476349:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top30 > 119.32767486572266:
                                                        if Q.girth2_top5 > 0.017241718247532845:
                                                            return 't'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9970026910305023:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8007335364818573:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.04305613227188587:
                                        if Q.sum_pt > 1207.4248046875:
                                            if Q.lam2 > 0.0031663381960242987:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.003851482877507806:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 134.28279876708984:
                                                    if Q.mass_top50 > 148.44869995117188:
                                                        if Q.z_top50_slots > 0.9618565440177917:
                                                            if Q.girth2_top5 > 0.014214730821549892:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0019206590950489044:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.006319463020190597:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 127.23397827148438:
                                                if Q.mass_over_sum_pt_sq > 0.01658679824322462:
                                                    if Q.z_top50_slots > 0.9561513662338257:
                                                        if Q.lam2 > 0.0036500708665698767:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.5295439064502716:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_particles > 60.5:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 196.10855102539062:
                            if Q.log_sum_pt > 7.084420919418335:
                                if Q.sum_pt > 1267.650390625:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.023308186791837215:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.46983009576797485:
                                    if Q.z_top50_slots > 0.9578204154968262:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9701602458953857:
                                        if Q.tau21 > 0.3606148958206177:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04438859224319458:
                                if Q.mass > 185.80514526367188:
                                    if Q.e2 > 0.051463423296809196:
                                        if Q.e2 > 0.05971015803515911:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.2387300729751587:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1384.6341552734375:
                                        if Q.mass > 176.64554595947266:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 178.46058654785156:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 7.222259998321533:
                                        if Q.tau32 > 0.3185676336288452:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.4207487404346466:
                                            if Q.mass > 163.12491607666016:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_particles > 60.5:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.002739913063123822:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9613251388072968:
                                                    if Q.lam1 > 0.01443484053015709:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 138.91193389892578:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top40 > 751.96044921875:
                        if Q.lam1 > 0.0323406457901001:
                            if Q.tau21 > 0.36105668544769287:
                                if Q.e2 > 0.050561435520648956:
                                    if Q.log_sum_pt > 6.964576005935669:
                                        if Q.mass_top30 > 179.40765380859375:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top30_slots > 0.7672846019268036:
                                        return 't'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.918093919754028:
                                    if Q.z_top50_slots > 0.9771109521389008:
                                        if Q.sum_pt > 1056.062744140625:
                                            if Q.tau32 > 0.6748769581317902:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 423.8125:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.6461300849914551:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9609409868717194:
                                        if Q.lam1 > 0.035482872277498245:
                                            if Q.lam2 > 0.003075636108405888:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top40 > 182.93719482421875:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.975429117679596:
                                                        if Q.tau21 > 0.23367831110954285:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7119348347187042:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.08927256241440773:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 931.4951171875:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 906.64599609375:
                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.03460335545241833:
                                            if Q.lam2 > 0.0037852406967431307:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.7711129188537598:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.00946466950699687:
                                if Q.e2 > 0.04672575555741787:
                                    if Q.mass > 182.76719665527344:
                                        if Q.tau21 > 0.3162347972393036:
                                            if Q.e2 > 0.05528286099433899:
                                                if Q.log_sum_pt > 6.969415187835693:
                                                    if Q.tau32 > 0.6491972506046295:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.956400394439697:
                                                    if Q.tau21 > 0.44986702501773834:
                                                        if Q.mass > 191.87305450439453:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1030.43603515625:
                                                        if Q.tau32 > 0.5430116653442383:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1037.49560546875:
                                                if Q.n_particles > 61.5:
                                                    if Q.z_top50_slots > 0.9732411205768585:
                                                        if Q.tau21 > 0.26075397431850433:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.1771702691912651:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 39% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0015670909197069705:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.7285579144954681:
                                            if Q.sum_pt_top50 > 999.00439453125:
                                                if Q.tau21 > 0.2208927422761917:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.0548785999417305:
                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 478.96875:
                                                            if Q.mass_over_sum_pt_sq > 0.01619057822972536:
                                                                if Q.girth2_top20 > 0.019252289086580276:
                                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 830.411376953125:
                                                    return 't'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.607399582862854:
                                                if Q.log_sum_pt > 6.951663494110107:
                                                    if Q.z_top50_slots > 0.9699918329715729:
                                                        if Q.girth2_top5 > 0.006018361775204539:
                                                            if Q.e2 > 0.04881310649216175:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top30 > 119.3962173461914:
                                                                    if Q.mass_top50 > 143.09312438964844:
                                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.27373577654361725:
                                                            if Q.mass_top50 > 152.14739990234375:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 836.000244140625:
                                                        if Q.sum_pt_top3 > 622.625:
                                                            if Q.girth2_top5 > 0.009481721557676792:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.014364872593432665:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top5 > 0.008119587786495686:
                                                            if Q.z_top50_slots > 0.9637156128883362:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.26871514320373535:
                                                                    if Q.mass > 143.83255767822266:
                                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top50 > 850.2314453125:
                                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top50_slots > 0.9838529825210571:
                                                                return 't'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.92807936668396:
                                        if Q.tau32 > 0.5447275936603546:
                                            if Q.z_top50_slots > 0.980771005153656:
                                                if Q.z_dr_0p2_0p4 > 0.09297554567456245:
                                                    if Q.sum_pt_top3 > 518.40625:
                                                        if Q.tau21 > 0.3681998699903488:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.966758728027344:
                                                                if Q.z_top50_slots > 0.9992314279079437:
                                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.971416711807251:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top5 > 0.009355322923511267:
                                                                return 't'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 154.53685760498047:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.015550641342997551:
                                                                        if Q.n_particles > 60.5:
                                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.log_sum_pt > 6.940251588821411:
                                                                            if Q.n_particles > 56.5:
                                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt_top3 > 420.4375:
                                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.974598407745361:
                                                        if Q.z_top50_slots > 0.9916015863418579:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.08858213573694229:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 5.5:
                                                            return 't'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 468.75:
                                                                return 'Z'   # 39% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0028935596346855164:
                                                    if Q.log_sum_pt > 6.951810598373413:
                                                        if Q.mass_top40 > 134.7164764404297:
                                                            if Q.z_top50_slots > 0.9456363320350647:
                                                                if Q.z_dr_0_0p05 > 0.054685940966010094:
                                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2_top5 > 0.016301306895911694:
                                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04104803688824177:
                                                            return 't'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.6555452048778534:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1055.5634765625:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top50_slots > 0.9665591418743134:
                                                            if Q.mass > 157.17645263671875:
                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.08926106616854668:
                                                                    if Q.max_dr > 0.37801745533943176:
                                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.log_sum_pt > 6.9391725063323975:
                                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.15353422611951828:
                                                                if Q.LHA > 0.4212620109319687:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2_top5 > 0.015444282442331314:
                                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9450514912605286:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    if Q.tau32 > 0.4283480793237686:
                                                        if Q.e2 > 0.04144408367574215:
                                                            return 't'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top50_slots > 0.9718104004859924:
                                                                if Q.tau21 > 0.5032923519611359:
                                                                    return 't'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p2_0p4 > 0.08576257154345512:
                                                                        return 'q'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_over_sum_pt > 0.1441369131207466:
                                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1057.1201171875:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1048.4912109375:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 799.357421875:
                                            if Q.sum_pt_top50 > 994.934326171875:
                                                if Q.sum_pt_top3 > 552.53125:
                                                    if Q.z_dr_0p2_0p4 > 0.04845113307237625:
                                                        if Q.tau32 > 0.5470487177371979:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.005008841631934047:
                                                        if Q.mass_over_sum_pt_sq > 0.009896705858409405:
                                                            if Q.max_dr > 0.3020395189523697:
                                                                if Q.lam1 > 0.0148399299941957:
                                                                    if Q.sum_pt_top3 > 358.375:
                                                                        if Q.e2 > 0.044092271476984024:
                                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                                return 't'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.39406508207321167:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 400.0625:
                                                                if Q.mass > 112.45479202270508:
                                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.014664892107248306:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 814.626953125:
                                                    if Q.log_sum_pt > 6.898407936096191:
                                                        if Q.sum_pt_top3 > 431.65625:
                                                            if Q.tau21 > 0.40728676319122314:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.10309760272502899:
                                                                    if Q.e2 > 0.04344670660793781:
                                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.3039148896932602:
                                                                if Q.tau32 > 0.7629651427268982:
                                                                    if Q.z_top50_slots > 0.9517321586608887:
                                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.1278006210923195:
                                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.04164299741387367:
                                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 19.5:
                                                        if Q.tau32 > 0.6856690943241119:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.13281378895044327:
                                                if Q.tau32 > 0.5872259736061096:
                                                    if Q.lam2 > 0.006393622141331434:
                                                        if Q.z_dr_0p05_0p1 > 0.2829393297433853:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.08442777022719383:
                                                            if Q.max_dr > 0.302744597196579:
                                                                if Q.dr_0 > 0.08803794905543327:
                                                                    if Q.z_top50_slots > 0.953376293182373:
                                                                        return 't'   # 59% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.026706455275416374:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.018901314586400986:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.3022768944501877:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.8886096477508545:
                                    if Q.n_dr_0p2_0p4 > 8.5:
                                        if Q.n_particles > 40.5:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.mass_top50 > 99.65289306640625:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 999.68798828125:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.0027560219168663025:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.046109966933727264:
                            if Q.sum_pt_top40 > 678.603515625:
                                if Q.girth > 0.11500459536910057:
                                    if Q.z_top30_slots > 0.982628583908081:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.03571435809135437:
                                            if Q.e2 > 0.07070637494325638:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 180.44313049316406:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 709.3046875:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.16099964082241058:
                                                    if Q.e2 > 0.06129981949925423:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.050488030537962914:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 734.96044921875:
                                        return 't'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9787032306194305:
                                    if Q.mass > 115.56820678710938:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.4656500816345215:
                                            if Q.mass_over_sum_pt_sq > 0.02254265733063221:
                                                return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 109.61106872558594:
                                        if Q.tau21 > 0.5710879862308502:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3878982663154602:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.055594755336642265:
                                            if Q.sum_pt > 679.205078125:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top50 > 134.48687744140625:
                                if Q.tau21 > 0.5573651492595673:
                                    return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 63% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.1219686008989811:
                                    if Q.e2 > 0.042780496180057526:
                                        if Q.sum_pt_top40 > 650.8291015625:
                                            if Q.lam2 > 0.005821192637085915:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.09270716458559036:
                                                    if Q.z_top50_slots > 0.9619855582714081:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 76% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1029.9437255859375:
                    if Q.n_particles > 60.5:
                        if Q.tau32 > 0.424565926194191:
                            if Q.log_sum_pt > 6.961099863052368:
                                if Q.C2 > 0.14336220175027847:
                                    if Q.log_sum_pt > 7.010406970977783:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 13.5:
                                            if Q.mass_top30 > 116.95892715454102:
                                                if Q.sum_pt_top40 > 990.421875:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 146.46446228027344:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.03917320817708969:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.994873046875:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.08281992748379707:
                                            if Q.z_top50_slots > 0.9834198355674744:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.12166188284754753:
                                                    if Q.z_top50_slots > 0.9453246891498566:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9831056296825409:
                                    if Q.girth2_top5 > 0.004300017841160297:
                                        if Q.dr_0 > 0.0547361895442009:
                                            if Q.lam2 > 0.0017446456477046013:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.007071541389450431:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.759192168712616:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 541.71875:
                                            return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.004417737713083625:
                                        if Q.tau32 > 0.5275579392910004:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.15359706431627274:
                                            if Q.z_top50_slots > 0.9448257982730865:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03663318045437336:
                                                if Q.z_top50_slots > 0.9768204987049103:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top30_slots > 0.7933385670185089:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.515190452337265:
                                                        if Q.mass_top50 > 106.0948715209961:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.03400414064526558:
                                if Q.log_sum_pt > 7.12626051902771:
                                    if Q.mass > 173.96484375:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.12476912885904312:
                                            return 't'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 18.5:
                                        if Q.tau32 > 0.37052100896835327:
                                            if Q.lam2 > 0.003960276022553444:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9803239405155182:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 154.30477142333984:
                                                return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.96599268913269:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9790769219398499:
                                        return 't'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 74% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.979100465774536:
                            if Q.tau32 > 0.536823570728302:
                                if Q.n_particles > 35.5:
                                    if Q.mass > 108.40874862670898:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 595.25:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.035125305876135826:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.035021256655454636:
                                    if Q.sum_pt_top40 > 1216.59765625:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 45.5:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top3 > 547.5625:
                                if Q.n_dr_0p2_0p4 > 10.5:
                                    if Q.sum_pt > 1062.495849609375:
                                        if Q.sum_pt_top3 > 638.25:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.0972430408000946:
                                            if Q.z_dr_0p2_0p4 > 0.0687238872051239:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.007187062641605735:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 51% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top5 > 0.003908021724782884:
                                    if Q.lam2 > 0.0009082856704480946:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 101.2775764465332:
                                            if Q.mass > 110.48156356811523:
                                                return 'g'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6477420926094055:
                                        if Q.sum_pt > 1045.124755859375:
                                            if Q.z_top50_slots > 0.9983577132225037:
                                                if Q.girth2_top5 > 0.0019987099803984165:
                                                    return 't'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9978269934654236:
                                                if Q.mass > 100.95024871826172:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.031156012788414955:
                                                    return 'q'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.4397248923778534:
                                            if Q.mass_over_sum_pt_sq > 0.013030592817813158:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.009521606843918562:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 40% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02890962827950716:
                        if Q.log_sum_pt > 6.902575254440308:
                            if Q.z_top50_slots > 0.9760033786296844:
                                if Q.sum_pt_top3 > 426.71875:
                                    if Q.mass_over_sum_pt_sq > 0.009595312178134918:
                                        if Q.lam2 > 0.004689503228291869:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.0651228092610836:
                                                if Q.tau21 > 0.6401041448116302:
                                                    return 't'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 559.28125:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.03613496758043766:
                                                        return 't'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 43.5:
                                            if Q.e2 > 0.03208528272807598:
                                                if Q.z_dr_0p2_0p4 > 0.041753169149160385:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.07453075423836708:
                                        if Q.tau21 > 0.4247235357761383:
                                            if Q.mass_over_sum_pt > 0.11510622873902321:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1014.573974609375:
                                                    if Q.tau32 > 0.5701713263988495:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.11323954537510872:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00741982483305037:
                                                if Q.z_dr_0p05_0p1 > 0.35655559599399567:
                                                    if Q.sum_pt_top3 > 301.9375:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 100.90987014770508:
                                                    if Q.sum_pt_top3 > 335.578125:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.032648177817463875:
                                                            if Q.sum_pt_top3 > 299.96875:
                                                                return 'q'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 7.5:
                                            if Q.girth2_top5 > 0.004702514968812466:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.006584734423086047:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 47% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.009394107386469841:
                                                if Q.max_dr > 0.37536779046058655:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1011.3843994140625:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 141.7034683227539:
                                    if Q.tau21 > 0.40946343541145325:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9576260447502136:
                                            return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.967891663312912:
                                        if Q.dr_0 > 0.07425926625728607:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                if Q.sum_pt > 1016.47607421875:
                                                    return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6046259105205536:
                                                if Q.log_sum_pt > 6.912818908691406:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.032591335475444794:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.11895904690027237:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.005939899478107691:
                                            if Q.tau21 > 0.6515760719776154:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1009.0615234375:
                                                if Q.z_top50_slots > 0.9351072609424591:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9576873481273651:
                                                    if Q.dr_0 > 0.08681541308760643:
                                                        if Q.e2 > 0.03433161415159702:
                                                            return 't'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 32% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0028582047671079636:
                                                            return 't'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 765.974609375:
                                if Q.z_top50_slots > 0.9596991539001465:
                                    if Q.sum_pt_top40 > 960.240234375:
                                        if Q.z_dr_0_0p05 > 0.8041257560253143:
                                            if Q.mass_over_sum_pt > 0.09801829606294632:
                                                if Q.sum_pt_top3 > 462.84375:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.08767501637339592:
                                                if Q.sum_pt > 988.6787109375:
                                                    if Q.dr_0 > 0.0449466984719038:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.870744377374649:
                                            return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.886556625366211:
                                                if Q.e2 > 0.031841935589909554:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 959.6328125:
                                                        if Q.sum_pt_top3 > 372.75:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top30 > 91.77660751342773:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 799.80859375:
                                                    if Q.z_top50_slots > 0.9681280851364136:
                                                        if Q.sum_pt_top3 > 584.3125:
                                                            if Q.z_dr_0_0p05 > 0.8150190711021423:
                                                                if Q.lam2 > 0.0009723340044729412:
                                                                    return 't'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.030913252383470535:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.872217893600464:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.017681392841041088:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.03252648934721947:
                                        if Q.z_top50_slots > 0.9399732053279877:
                                            if Q.sum_pt_top50 > 924.03515625:
                                                if Q.mass_over_sum_pt_sq > 0.02148208301514387:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.033758606761693954:
                                                        if Q.lam2 > 0.0023426710395142436:
                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top5 > 0.012037774547934532:
                                                                return 't'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.009566884022206068:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.743398904800415:
                                                        if Q.z_top30_slots > 0.8093792796134949:
                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.035664865747094154:
                                                return 't'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 959.482421875:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.5368222892284393:
                                                        if Q.lam1 > 0.02044160384684801:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_top50 > 118.82730484008789:
                                                                return 't'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.34138017892837524:
                                            if Q.log_sum_pt > 6.867988586425781:
                                                if Q.lam2 > 0.005466431379318237:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    if Q.girth2_top5 > 0.006718471879139543:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top50_slots > 0.947086364030838:
                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.00746541959233582:
                                                        if Q.log_sum_pt > 6.8409223556518555:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9471027851104736:
                                                if Q.dr_0 > 0.08471260964870453:
                                                    if Q.sum_pt_top40 > 854.162109375:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt_sq > 0.01495630107820034:
                                    if Q.log_sum_pt > 6.707402944564819:
                                        if Q.mass_over_sum_pt_sq > 0.01844058372080326:
                                            if Q.tau32 > 0.6483940482139587:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.010895656887441874:
                                                    if Q.sum_pt_top3 > 145.546875:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.5411218404769897:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 727.482421875:
                                        if Q.e2 > 0.031164701096713543:
                                            return 't'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                    else:
                        if Q.z_top50_slots > 0.9793778657913208:
                            if Q.sum_pt_top40 > 943.0625:
                                if Q.sum_pt_top3 > 385.125:
                                    if Q.n_particles > 60.5:
                                        if Q.log_sum_pt > 6.9264466762542725:
                                            if Q.dr_0 > 0.035168685019016266:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.028265023604035378:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 480.46875:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 63.5:
                                        if Q.tau32 > 0.6517215371131897:
                                            if Q.D2 > 2.996355414390564:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.6287302672863007:
                                                return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 982.8480224609375:
                                            if Q.sum_pt > 1015.1065673828125:
                                                if Q.z_dr_0p05_0p1 > 0.09314340725541115:
                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.02258624415844679:
                                    return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6644653677940369:
                                        if Q.z_top50_slots > 0.9919426739215851:
                                            return 't'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9669782817363739:
                                if Q.log_sum_pt > 6.89242959022522:
                                    if Q.tau32 > 0.6205752491950989:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.28884634375572205:
                                            return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.023722580634057522:
                                        if Q.girth2_top5 > 0.0028397178975865245:
                                            if Q.D2 > 2.8429986238479614:
                                                if Q.sum_pt > 958.60107421875:
                                                    if Q.z_top50_slots > 0.972764402627945:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.5825782120227814:
                                                if Q.z_top50_slots > 0.9741794764995575:
                                                    if Q.girth > 0.07281388342380524:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 929.6318359375:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top40 > 79.6113052368164:
                                                            if Q.D2 > 4.324603080749512:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.17422789335250854:
                                    if Q.max_dr > 0.38622190058231354:
                                        return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.025498411618173122:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.11764157190918922:
                                        if Q.tau32 > 0.6709461510181427:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 931.8193359375:
                                                if Q.tau32 > 0.46290990710258484:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top40 > 92.05704498291016:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.979418992996216:
                    if Q.z_dr_0p2_0p4 > 0.0032225524773821235:
                        if Q.sum_pt > 1118.7900390625:
                            if Q.z_dr_0p2_0p4 > 0.008089417591691017:
                                return 'g'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1153.7728271484375:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top30_slots > 0.8854607343673706:
                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.006757491268217564:
                                if Q.e2 > 0.023396477103233337:
                                    if Q.mass > 95.49390029907227:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top30 > 74.48269653320312:
                                            if Q.log_sum_pt > 6.990644931793213:
                                                if Q.max_dr > 0.37375159561634064:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.32390762865543365:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.996394157409668:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9718137383460999:
                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 95.19766616821289:
                            return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023087256588041782:
                                return 'Z'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.018930175341665745:
                                    if Q.lam2 > 0.0006973635463509709:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02336365170776844:
                        if Q.mass_over_sum_pt > 0.09147834777832031:
                            if Q.e2 > 0.029284872114658356:
                                if Q.log_sum_pt > 6.889760494232178:
                                    if Q.n_dr_0p2_0p4 > 8.5:
                                        if Q.mass > 97.12925338745117:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.7406274676322937:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.3326420336961746:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 50% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9746612906455994:
                                    if Q.sum_pt_top50 > 978.8466796875:
                                        if Q.mass > 95.92044448852539:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.011591293383389711:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.29194045066833496:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top40 > 76.38047790527344:
                                if Q.sum_pt > 982.1964111328125:
                                    if Q.mass > 92.4471549987793:
                                        if Q.mass_top30 > 76.07029342651367:
                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.006339772138744593:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top40 > 83.28080749511719:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.08206707239151001:
                                            if Q.max_dr > 0.3624967634677887:
                                                if Q.log_sum_pt > 6.901137113571167:
                                                    if Q.C2 > 0.035836346447467804:
                                                        if Q.girth2_top20 > 0.004830519203096628:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top50_slots > 0.9817988872528076:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 9.5:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3462218791246414:
                                                return 'W'   # 43% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.31789176166057587:
                                        if Q.z_dr_0p2_0p4 > 0.012051372323185205:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.3445963114500046:
                                    if Q.mass_top30 > 66.11537551879883:
                                        if Q.mass_over_sum_pt_sq > 0.007921433541923761:
                                            return 'g'   # 47% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.952306032180786:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8447714745998383:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9779577255249023:
                                            if Q.tau32 > 0.7441187798976898:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 38% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 72.98254776000977:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.2859497517347336:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.30817796289920807:
                            if Q.z_top50_slots > 0.9806720614433289:
                                if Q.girth > 0.05596010573208332:
                                    if Q.mass_over_sum_pt > 0.0897374115884304:
                                        if Q.mass > 94.18990325927734:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 71.397705078125:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 76.93646621704102:
                                            if Q.sum_pt > 1049.5394287109375:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.37953953444957733:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 506.625:
                                        if Q.log_sum_pt > 6.932770729064941:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 23.5:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.001032242609653622:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.017421837896108627:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.49283352494239807:
                                    if Q.e2 > 0.020792248658835888:
                                        if Q.lam2 > 0.00169883988564834:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 68.47321319580078:
                                                if Q.mass > 90.60805130004883:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top30 > 61.96601104736328:
                                if Q.mass > 93.29387283325195:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.265475258231163:
                                        if Q.z_top50_slots > 0.9565406739711761:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.08772144466638565:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 78.98262405395508:
                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.56694793701172:
                    if Q.girth2_top20 > 0.006970660528168082:
                        if Q.mass > 104.47787475585938:
                            if Q.n_dr_0p2_0p4 > 3.5:
                                if Q.mass > 109.61774826049805:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.708421528339386:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 39% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.03414694406092167:
                            if Q.mass > 102.92605209350586:
                                if Q.tau32 > 0.6198186576366425:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 111.06240844726562:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 45% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.C2 > 0.04330706410109997:
                                        return 't'   # 47% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 34.5:
                                if Q.sum_pt > 1080.6802978515625:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top30_slots > 0.9576283693313599:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 51% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top50 > 963.7841796875:
                        if Q.girth2_top20 > 0.0038280574372038245:
                            if Q.mass > 86.29475021362305:
                                if Q.sum_pt_top50 > 978.3707275390625:
                                    if Q.girth2_top20 > 0.005103359697386622:
                                        if Q.mass > 86.91515731811523:
                                            if Q.mass > 97.073974609375:
                                                if Q.girth2_top20 > 0.006407557055354118:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top40 > 93.65980911254883:
                                                        if Q.D2 > 3.641642689704895:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_particles > 38.5:
                                                                if Q.LHA > 0.25848618149757385:
                                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.008510388433933258:
                                                            return 't'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.09365453943610191:
                                                    if Q.girth2_top20 > 0.007489197654649615:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 3.0075619220733643:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p2_0p4 > 0.04260902665555477:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                                    if Q.sum_pt_top50 > 1002.4346923828125:
                                                                        if Q.n_dr_0p2_0p4 > 11.5:
                                                                            return 't'   # 64% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 4.78329610824585:
                                                        if Q.mass_over_sum_pt_sq > 0.008260935544967651:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top50 > 988.169921875:
                                                            if Q.girth2_top20 > 0.006328303134068847:
                                                                if Q.n_dr_0p2_0p4 > 10.5:
                                                                    if Q.girth > 0.07707216218113899:
                                                                        if Q.sum_pt > 1003.4072265625:
                                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 54% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 94.95742797851562:
                                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                                        if Q.mass_over_sum_pt > 0.08445372059941292:
                                                                            if Q.girth > 0.076293233782053:
                                                                                return 't'   # 52% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 87.45080947875977:
                                                                        if Q.mass_over_sum_pt > 0.0923527367413044:
                                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.max_dr > 0.3843412548303604:
                                                                                if Q.C2 > 0.03416406363248825:
                                                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.lam2 > 0.001206156041007489:
                                                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.C2 > 0.03188631311058998:
                                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.max_dr > 0.32668572664260864:
                                                                                if Q.lam1 > 0.006143356906250119:
                                                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                                if Q.z_dr_0p2_0p4 > 0.051346512511372566:
                                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.35803332924842834:
                                                                        if Q.tau32 > 0.7068771421909332:
                                                                            if Q.lam2 > 0.0007006033847574145:
                                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1049.3729248046875:
                                                if Q.max_dr > 0.35016609728336334:
                                                    if Q.C2 > 0.04372549057006836:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.3165217489004135:
                                                        if Q.C2 > 0.034836942330002785:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.6299901008605957:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.04098961502313614:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 6.5:
                                                            if Q.sum_pt > 995.5118408203125:
                                                                if Q.D2 > 0.8452266156673431:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 43% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.3680395781993866:
                                                                if Q.mass_over_sum_pt > 0.08393699303269386:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 95.17791748046875:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                if Q.sum_pt > 1122.8216552734375:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6725517213344574:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 5.967196941375732:
                                                if Q.sum_pt > 1062.76806640625:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.001124656933825463:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 92.47543716430664:
                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                        if Q.log_sum_pt > 7.009323358535767:
                                                            if Q.mass_top30 > 88.85087966918945:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_over_sum_pt_sq > 0.006381656741723418:
                                                                    if Q.tau32 > 0.7783016562461853:
                                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 4.742070198059082:
                                                                return 'g'   # 36% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 87.25716018676758:
                                                        if Q.max_dr > 0.3678880035877228:
                                                            if Q.sum_pt > 1113.376953125:
                                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                                    if Q.n_particles > 43.5:
                                                                        if Q.tau32 > 0.6655476689338684:
                                                                            if Q.dr_0 > 0.02897057682275772:
                                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.046137621626257896:
                                                            if Q.sum_pt > 1085.0780029296875:
                                                                if Q.mass_top50 > 85.24251174926758:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.32654501497745514:
                                                                if Q.z_top50_slots > 0.9982687532901764:
                                                                    if Q.C2 > 0.032846832647919655:
                                                                        if Q.girth > 0.0631646141409874:
                                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 1135.4471435546875:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        if Q.z_dr_0p2_0p4 > 0.05483941175043583:
                                            if Q.D2 > 3.6656335592269897:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                if Q.max_dr > 0.3483903110027313:
                                                    if Q.tau32 > 0.6962864995002747:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.029675711877644062:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.35404686629772186:
                                                    if Q.sum_pt_top3 > 484.5:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.05318184196949005:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p2_0p4 > 0.0060045840218663216:
                                                                return 't'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.08283007889986038:
                                    if Q.D2 > 1.3479517698287964:
                                        if Q.max_dr > 0.43234729766845703:
                                            if Q.D2 > 2.1243830919265747:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.007110970094799995:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 978.244873046875:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.36932511627674103:
                                                    if Q.z_dr_0p2_0p4 > 0.05234287492930889:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.354832261800766:
                                            if Q.mass_over_sum_pt > 0.08519725129008293:
                                                if Q.sum_pt_top50 > 986.6575927734375:
                                                    if Q.n_dr_0p2_0p4 > 4.5:
                                                        if Q.D2 > 0.7667326033115387:
                                                            if Q.mass_top50 > 85.6759262084961:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.0032470397418364882:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.0066928251180797815:
                                                    if Q.D2 > 0.8664701581001282:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 44.5:
                                                        if Q.mass > 85.83769226074219:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3079695403575897:
                                                if Q.mass_over_sum_pt_sq > 0.007165322313085198:
                                                    if Q.sum_pt_top50 > 978.84033203125:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 85.3487663269043:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.330394983291626:
                                        if Q.mass_over_sum_pt > 0.07925134152173996:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 85.41861724853516:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.902980089187622:
                                                    if Q.girth2_top20 > 0.004613190423697233:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.29175300896167755:
                                            if Q.D2 > 1.407488465309143:
                                                if Q.mass_over_sum_pt > 0.08176974952220917:
                                                    if Q.n_dr_0p2_0p4 > 8.5:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3705250769853592:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 85.8014144897461:
                                                        if Q.max_dr > 0.37298035621643066:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 85.30880737304688:
                                                if Q.max_dr > 0.2160276621580124:
                                                    if Q.C2 > 0.03615882433950901:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.08069157227873802:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.030654994770884514:
                                                    if Q.max_dr > 0.23187324404716492:
                                                        if Q.mass_over_sum_pt_sq > 0.006518093636259437:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.9649951457977295:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.z_top50_slots > 0.9999295473098755:
                                        if Q.sum_pt_top40 > 1130.705322265625:
                                            if Q.lam2 > 0.00032301769533660263:
                                                if Q.D2 > 5.708764553070068:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 42.5:
                                                        if Q.max_dr > 0.37729182839393616:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1227.3568115234375:
                                                                return 'g'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.8158747255802155:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.0024913642555475235:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 6.101858139038086:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 7.01194429397583:
                                            if Q.max_dr > 0.3200581669807434:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 13.5:
                                                if Q.n_particles > 55.5:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.0006569182442035526:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.36760710179805756:
                                                    if Q.mass_over_sum_pt_sq > 0.006441408535465598:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 86.08575439453125:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            if Q.mass_top30 > 79.14235305786133:
                                                if Q.mass > 93.4659194946289:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 81.94237518310547:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt_sq > 0.0082471314817667:
                                    if Q.sum_pt_top3 > 381.390625:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 6.466017007827759:
                                        if Q.lam2 > 0.0011014380143024027:
                                            if Q.tau32 > 0.7572256922721863:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 41% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 59.5:
                                            if Q.girth2_top5 > 0.0005640400049742311:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 993.124267578125:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3858272433280945:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.32537876069545746:
                            if Q.e2 > 0.021583122201263905:
                                if Q.sum_pt_top40 > 955.2099609375:
                                    if Q.z_dr_0p2_0p4 > 0.06619098410010338:
                                        if Q.LHA > 0.23656360059976578:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.007855557836592197:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0_0p05 > 0.8146336376667023:
                                        if Q.girth2_top20 > 0.007091479375958443:
                                            if Q.sum_pt > 943.740234375:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0017279582098126411:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 59% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 935.43359375:
                                return 'Z'   # 92% of the training jets here get this class from the formula
                            else:
                                return 't'   # 66% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.51444244384766:
            if Q.sum_pt_top50 > 961.5888671875:
                if Q.girth2_top20 > 0.0030367206782102585:
                    if Q.mass_over_sum_pt > 0.08326738327741623:
                        if Q.D2 > 1.9302263855934143:
                            if Q.mass > 83.10474395751953:
                                if Q.sum_pt_top50 > 978.3199462890625:
                                    if Q.D2 > 2.188099503517151:
                                        if Q.n_dr_0p2_0p4 > 17.5:
                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.04824492335319519:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.005125029711052775:
                                                    if Q.sum_pt_top50 > 990.2255859375:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.023997300304472446:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 35% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 3.026057481765747:
                                    if Q.n_particles > 39.5:
                                        if Q.mass_top30 > 69.12742233276367:
                                            if Q.girth > 0.047752495855093:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0010884598013944924:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.08412507548928261:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.007178739877417684:
                                if Q.max_dr > 0.3239309638738632:
                                    if Q.sum_pt_top50 > 979.1630859375:
                                        if Q.mass_over_sum_pt_sq > 0.00723998318426311:
                                            if Q.lam2 > 0.0003559391916496679:
                                                if Q.D2 > 1.2292702198028564:
                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.4414884150028229:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.06549637019634247:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 83.30611801147461:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 49% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 968.1217041015625:
                                    if Q.max_dr > 0.2730086147785187:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.log_sum_pt > 6.902079105377197:
                                                if Q.z_top30_slots > 0.9681011140346527:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.2924070209264755:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.3088058829307556:
                                                if Q.girth2_top20 > 0.006587797775864601:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 84.05706787109375:
                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.1099472045898438:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        return 't'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt_sq > 0.007025745464488864:
                                            return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0p2_0p4 > 13.5:
                            if Q.C2 > 0.09647204726934433:
                                if Q.mass > 81.5353889465332:
                                    if Q.mass_top40 > 82.63314437866211:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1053.29296875:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 18.5:
                                                return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.0808052234351635:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 47% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 75.84906768798828:
                                        if Q.sum_pt_top40 > 987.184814453125:
                                            return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0007102583476807922:
                                            if Q.z_top30_slots > 0.9698404669761658:
                                                if Q.mass_top30 > 72.08559799194336:
                                                    if Q.z_dr_0_0p05 > 0.9137451648712158:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9962787926197052:
                                                    return 'q'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.901029109954834:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 83.37408447265625:
                                    if Q.D2 > 3.0080946683883667:
                                        if Q.mass_over_sum_pt > 0.0799519419670105:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.0039140209555625916:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.40835028886795044:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 981.9990234375:
                                        if Q.mass_top40 > 72.90604782104492:
                                            if Q.mass_over_sum_pt > 0.08266114443540573:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 7.0032057762146:
                                                    if Q.n_particles > 44.5:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.004329947056248784:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9994034171104431:
                                                    if Q.n_dr_0p2_0p4 > 14.5:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 76.95121383666992:
                                            if Q.n_dr_0p2_0p4 > 17.5:
                                                return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.04328733868896961:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 63.5:
                                if Q.mass > 82.24599075317383:
                                    if Q.e2 > 0.02457493171095848:
                                        if Q.mass_top40 > 73.1469497680664:
                                            if Q.sum_pt_top40 > 1022.580078125:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 82.83835983276367:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1056.4046630859375:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.32451727986335754:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 40% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.0038631072966381907:
                                        return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.981220006942749:
                                            if Q.max_dr > 0.30021676421165466:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.020761821419000626:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.8499976396560669:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 83.87530136108398:
                                    if Q.D2 > 2.6423048973083496:
                                        if Q.mass_over_sum_pt_sq > 0.006529460661113262:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 3.7439050674438477:
                                                if Q.mass_top30 > 74.9022216796875:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3451549559831619:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.006777643924579024:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 72.24909591674805:
                                                if Q.e2 > 0.027779788710176945:
                                                    if Q.max_dr > 0.272898867726326:
                                                        return 'W'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.08142353221774101:
                                                            if Q.LHA > 0.29251760244369507:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.004374189069494605:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 71.86931991577148:
                                        if Q.C2 > 0.11085241660475731:
                                            if Q.mass > 82.31698608398438:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 983.8729248046875:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 974.279296875:
                                                if Q.girth2_top20 > 0.003654903848655522:
                                                    if Q.mass_over_sum_pt > 0.0821499302983284:
                                                        if Q.D2 > 3.534404754638672:
                                                            if Q.mass > 82.3575439453125:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.1004951260983944:
                                                            if Q.mass_top50 > 82.56935501098633:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top50 > 982.2169189453125:
                                                                if Q.n_particles > 59.5:
                                                                    if Q.sum_pt_top40 > 1054.724609375:
                                                                        if Q.z_dr_0p2_0p4 > 0.0061690243892371655:
                                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                                    if Q.D2 > 1.7517605423927307:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.girth2_top20 > 0.005102604161947966:
                                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 82.73873901367188:
                                                        if Q.n_particles > 49.5:
                                                            if Q.sum_pt_top40 > 1109.7900390625:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top40 > 73.58757019042969:
                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top50 > 1066.24853515625:
                                                                if Q.max_dr > 0.3325193524360657:
                                                                    if Q.sum_pt_top3 > 553.03125:
                                                                        if Q.sum_pt_top3 > 742.46875:
                                                                            return 'q'   # 48% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.n_dr_0p2_0p4 > 4.5:
                                                                            if Q.log_sum_pt > 7.026953935623169:
                                                                                return 'g'   # 93% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.tau32 > 0.8603172898292542:
                                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 998.2611083984375:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.004638154758140445:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0005257547891233116:
                                                    if Q.max_dr > 0.3508494645357132:
                                                        if Q.girth > 0.058375366032123566:
                                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top40 > 964.738037109375:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_top30_slots > 0.9673627018928528:
                                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top20 > 0.004414560738950968:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0006620454369112849:
                                            if Q.log_sum_pt > 6.89874792098999:
                                                if Q.n_dr_0p2_0p4 > 10.5:
                                                    if Q.lam1 > 0.003971450496464968:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.3536912798881531:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.05477915517985821:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.00397496996447444:
                                                if Q.log_sum_pt > 6.890685796737671:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.004870675737038255:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.0028277377132326365:
                                                    if Q.C2 > 0.06509596854448318:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top20 > 0.0035047882702201605:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                else:
                    if Q.log_sum_pt > 6.967252016067505:
                        if Q.mass_top30 > 69.5692253112793:
                            if Q.lam2 > 0.00043799213017337024:
                                if Q.e2 > 0.018714895471930504:
                                    if Q.mass > 83.09563446044922:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3807297796010971:
                                            if Q.mass_top40 > 76.38834762573242:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 80.58673858642578:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.003798560705035925:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 36.5:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    if Q.mass > 83.49214172363281:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top30 > 72.79481506347656:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.1391026973724365:
                                                if Q.tau32 > 0.8116037547588348:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.00045605312334373593:
                                if Q.girth2_top20 > 0.0022307607578113675:
                                    if Q.n_particles > 54.5:
                                        if Q.mass > 80.40987396240234:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.29221267998218536:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 1112.076416015625:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00045757577754557133:
                                                if Q.mass_top40 > 72.6142578125:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 9.5:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 661.09375:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1101.10693359375:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 45.5:
                                            if Q.girth2_top5 > 0.0005217278085183352:
                                                if Q.mass_over_sum_pt_sq > 0.005585324484854937:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 35% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 1166.96875:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 73.50193786621094:
                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.020237098447978497:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 81.58495330810547:
                            if Q.mass_top50 > 80.79388809204102:
                                if Q.mass_over_sum_pt > 0.08006176725029945:
                                    if Q.girth2_top5 > 0.00029765363433398306:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 0.0007577422366011888:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 13.5:
                                if Q.lam1 > 0.004340866580605507:
                                    if Q.sum_pt_top50 > 995.145751953125:
                                        if Q.n_particles > 57.5:
                                            if Q.girth > 0.045679377391934395:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.10804149508476257:
                                                return 'q'   # 45% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top30_slots > 0.9678517282009125:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 51.5:
                                        if Q.e2 > 0.0189497210085392:
                                            return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.04304459132254124:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.003952851518988609:
                                                if Q.sum_pt_top40 > 1006.8946533203125:
                                                    if Q.dr_0 > 0.01504947617650032:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 32% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.004003352951258421:
                                    if Q.sum_pt_top50 > 980.627197265625:
                                        return 'W'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 43% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 73.10440444946289:
                                        if Q.tau32 > 0.7371609807014465:
                                            if Q.mass_top40 > 71.8331527709961:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 40% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 9.5:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 65% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.02524426206946373:
                    if Q.n_dr_0p2_0p4 > 1.5:
                        if Q.sum_pt_top50 > 717.5264892578125:
                            if Q.mass_over_sum_pt > 0.08349178731441498:
                                if Q.z_top50_slots > 0.9754826128482819:
                                    if Q.z_dr_0_0p05 > 0.7929919958114624:
                                        if Q.mass_top50 > 76.41348648071289:
                                            if Q.girth2_top20 > 0.008502470329403877:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.849911689758301:
                                                    if Q.mass_over_sum_pt > 0.08592681586742401:
                                                        if Q.lam2 > 0.0004700428689830005:
                                                            return 't'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.49232354760169983:
                                            if Q.mass_over_sum_pt_sq > 0.007479548919945955:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 64.64370727539062:
                                        if Q.log_sum_pt > 6.86694073677063:
                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 74.5042724609375:
                                    if Q.lam2 > 0.00045448400487657636:
                                        if Q.tau21 > 0.22488922625780106:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.856563091278076:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.34477949142456055:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0_0p05 > 0.8139013648033142:
                                        if Q.mass > 73.66690063476562:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.30532264709472656:
                                            if Q.log_sum_pt > 6.863817453384399:
                                                if Q.n_dr_0p2_0p4 > 3.5:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.856163740158081:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 64% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9994847476482391:
                                return 'q'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.046647923067212105:
                                    return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                    else:
                        if Q.mass_over_sum_pt_sq > 0.007119766436517239:
                            if Q.mass > 83.47762298583984:
                                return 'Z'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19441823661327362:
                                    return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 81.0532455444336:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 53% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 92% of the training jets here get this class from the formula
                else:
                    if Q.z_top30_slots > 0.9729074835777283:
                        if Q.mass_over_sum_pt > 0.08330774679780006:
                            if Q.z_dr_0p2_0p4 > 0.06738518178462982:
                                return 't'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 5.694576263427734:
                                    return 't'   # 53% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.00032569120230618864:
                                if Q.sum_pt_top3 > 445.53125:
                                    return 'q'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 0.0004506797413341701:
                                        return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.019959996454417706:
                            if Q.z_top50_slots > 0.9868762195110321:
                                if Q.z_dr_0_0p05 > 0.8044313490390778:
                                    if Q.mass_top30 > 66.29884719848633:
                                        if Q.z_top50_slots > 0.9986713826656342:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 49% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.28064870834350586:
                                    if Q.D2 > 3.366426706314087:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 81.58774185180664:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 41% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 81.82711029052734:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 86% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 45.5:
                if Q.sum_pt > 1045.9378662109375:
                    if Q.girth2_top20 > 0.003222178900614381:
                        if Q.mass_top40 > 65.78133392333984:
                            if Q.mass > 79.64737701416016:
                                return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.3006512522697449:
                                    if Q.C2 > 0.04373203031718731:
                                        if Q.sum_pt > 1080.396728515625:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.022571059875190258:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.004279830027371645:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 62.5:
                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.7141790986061096:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 42% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.971696138381958:
                            if Q.n_particles > 51.5:
                                if Q.girth2_top20 > 0.0023151178611442447:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.0017630489310249686:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 57.5:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0009400304115843028:
                                                return 'q'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.0011262425687164068:
                                    if Q.log_sum_pt > 7.018747091293335:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            if Q.sum_pt_top3 > 637.265625:
                                                if Q.tau32 > 0.8779822289943695:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.6039938628673553:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.22002100944519043:
                                        if Q.z_dr_0p2_0p4 > 0.0020208375062793493:
                                            return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1111.9501953125:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.0002730099658947438:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 61% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 53.5:
                                if Q.mass_top30 > 57.040199279785156:
                                    if Q.z_dr_0p2_0p4 > 0.01197573309764266:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 44% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.018609163351356983:
                                        if Q.lam2 > 0.0005886110884603113:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.0010084316600114107:
                                    if Q.sum_pt_top3 > 499.875:
                                        if Q.girth2_top5 > 0.0002311529460712336:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 50.5:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 629.9375:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.tau32 > 0.7856250405311584:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0007259926351252943:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00036849413299933076:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.7725091874599457:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.74402046203613:
                            if Q.mass_over_sum_pt_sq > 0.006308590993285179:
                                if Q.D2 > 2.1490159034729004:
                                    if Q.tau32 > 0.6517595946788788:
                                        if Q.D2 > 2.774089813232422:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9838668704032898:
                                                return 't'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 953.029296875:
                                            if Q.max_dr > 0.3673534542322159:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9811860620975494:
                                        if Q.sum_pt > 940.4732666015625:
                                            return 'q'   # 35% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.33544546365737915:
                                    if Q.mass_top40 > 66.91460418701172:
                                        if Q.n_dr_0p2_0p4 > 10.5:
                                            if Q.n_dr_0p2_0p4 > 14.5:
                                                if Q.z_top50_slots > 0.9937276840209961:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1001.74267578125:
                                                    if Q.n_particles > 61.5:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 965.85595703125:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 34% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 62.5:
                                            if Q.z_dr_0p2_0p4 > 0.005938323680311441:
                                                if Q.tau32 > 0.6004688143730164:
                                                    if Q.e2 > 0.021926009096205235:
                                                        if Q.z_top50_slots > 0.9799031615257263:
                                                            if Q.log_sum_pt > 6.921274900436401:
                                                                return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 983.8477783203125:
                                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.293525218963623:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0009450464567635208:
                                                if Q.sum_pt_top50 > 970.1416015625:
                                                    if Q.z_top30_slots > 0.9331232607364655:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.906699180603027:
                                                            if Q.mass_top40 > 64.65219497680664:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 413.5:
                                                    return 'q'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 71.18313598632812:
                                        if Q.log_sum_pt > 6.883711099624634:
                                            if Q.mass_top40 > 65.97433090209961:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.23967592418193817:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 46% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 63.5:
                                if Q.mass_top30 > 52.53584861755371:
                                    if Q.max_dr > 0.2980705797672272:
                                        if Q.mass_over_sum_pt_sq > 0.0049101728945970535:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.022370888851583004:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 942.4169921875:
                                            if Q.mass_top50 > 67.24164962768555:
                                                if Q.z_top50_slots > 0.9570225477218628:
                                                    if Q.tau32 > 0.8464406728744507:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9738753736019135:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.0033837617374956608:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.00968343811109662:
                                        if Q.sum_pt_top3 > 504.453125:
                                            if Q.LHA > 0.21297404915094376:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.8572511672973633:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.8086764514446259:
                                                if Q.sum_pt_top3 > 292.390625:
                                                    if Q.dr_0 > 0.03432788699865341:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.925670146942139:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 948.10693359375:
                                                        if Q.e2 > 0.015408760868012905:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.014373783953487873:
                                    if Q.sum_pt > 966.0384521484375:
                                        if Q.sum_pt > 1023.138427734375:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                if Q.sum_pt_top3 > 507.125:
                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6866275072097778:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top5 > 0.0009537310979794711:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.031920356675982475:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 373.125:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.021147286519408226:
                                                if Q.tau32 > 0.7038362324237823:
                                                    if Q.sum_pt_top3 > 406.125:
                                                        if Q.dr_0 > 0.016751241870224476:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.2309160903096199:
                                                            if Q.mass > 74.95871353149414:
                                                                return 'W'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 332.84375:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0038118321681395173:
                                                            return 'W'   # 39% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8194176852703094:
                                                    if Q.girth2_top20 > 0.0014947691233828664:
                                                        if Q.n_dr_0p2_0p4 > 7.5:
                                                            if Q.sum_pt_top3 > 320.328125:
                                                                return 'q'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_particles > 60.5:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.5702995955944061:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.girth2_top5 > 0.002962928148917854:
                                                if Q.mass_top30 > 52.179765701293945:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.005538842407986522:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.girth2_top20 > 0.0009118611342273653:
                                            if Q.sum_pt_top3 > 498.5:
                                                if Q.sum_pt > 1022.308349609375:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0004937508492730558:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.02269431296736002:
                                            if Q.sum_pt_top3 > 272.640625:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 367.1875:
                                                if Q.LHA > 0.17567147314548492:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8448680937290192:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.927260160446167:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.000636186683550477:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top40 > 935.4697265625:
                            if Q.girth2_top20 > 0.0007474646554328501:
                                if Q.mass_top50 > 69.37508010864258:
                                    if Q.n_dr_0p2_0p4 > 9.5:
                                        if Q.mass_top50 > 71.97030639648438:
                                            if Q.sum_pt_top50 > 987.5731201171875:
                                                if Q.n_dr_0p2_0p4 > 14.5:
                                                    if Q.lam2 > 0.0007697085966356099:
                                                        if Q.log_sum_pt > 6.919698476791382:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.00425891182385385:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 448.046875:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 970.36083984375:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.004211260937154293:
                                                if Q.log_sum_pt > 6.908733129501343:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 1027.7962646484375:
                                                    if Q.sum_pt_top40 > 1027.580078125:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 990.6822509765625:
                                            if Q.mass_top50 > 71.00743865966797:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.0035958747612312436:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.061995845288038254:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.004212790634483099:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 52.5:
                                        if Q.LHA > 0.18804530054330826:
                                            if Q.sum_pt_top50 > 1028.806884765625:
                                                if Q.z_dr_0p2_0p4 > 0.00825946219265461:
                                                    if Q.sum_pt_top3 > 425.390625:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7311416268348694:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    if Q.sum_pt_top3 > 393.96875:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7522481977939606:
                                                            if Q.dr_0 > 0.02197624184191227:
                                                                if Q.mass_top30 > 47.06296157836914:
                                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 458.0625:
                                                if Q.sum_pt_top50 > 1030.2100830078125:
                                                    if Q.LHA > 0.17154283821582794:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.15452811866998672:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top20 > 0.0011437368229962885:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    if Q.sum_pt_top50 > 1016.462890625:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.887969493865967:
                                                            if Q.girth2_top20 > 0.001248981337994337:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.8780773282051086:
                                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_top50_slots > 0.9977891743183136:
                                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.940103769302368:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.8432473540306091:
                                                            if Q.tau21 > 0.6107262969017029:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.003841966507025063:
                                            if Q.log_sum_pt > 6.906954050064087:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    if Q.D2 > 2.936482787132263:
                                                        if Q.n_dr_0p2_0p4 > 10.5:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.001040173345245421:
                                                if Q.sum_pt_top40 > 949.39501953125:
                                                    if Q.sum_pt > 1020.4525146484375:
                                                        if Q.sum_pt_top3 > 423.65625:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                                if Q.n_dr_0p2_0p4 > 12.5:
                                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.log_sum_pt > 6.940093994140625:
                                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.5372785925865173:
                                                        if Q.tau32 > 0.7806656956672668:
                                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                                if Q.sum_pt_top3 > 426.625:
                                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.937181711196899:
                                                    if Q.lam2 > 0.0005614261899609119:
                                                        if Q.sum_pt_top3 > 530.25:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau32 > 0.7783550918102264:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 973.6611328125:
                                                        if Q.log_sum_pt > 6.91929292678833:
                                                            if Q.mass_over_sum_pt_sq > 0.0019024336943402886:
                                                                if Q.girth2_top20 > 0.0008838122885208577:
                                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top3 > 478.28125:
                                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau32 > 0.8767908215522766:
                                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0005793447780888528:
                                                            if Q.sum_pt_top3 > 456.71875:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.z_top50_slots > 0.9997240900993347:
                                        if Q.girth2_top5 > 9.219414641847834e-05:
                                            if Q.sum_pt_top3 > 422.34375:
                                                if Q.tau32 > 0.7729410231113434:
                                                    if Q.log_sum_pt > 6.932328939437866:
                                                        if Q.z_top30_slots > 0.9748432636260986:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 970.6561279296875:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 1022.430419921875:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 973.950927734375:
                                                        if Q.log_sum_pt > 6.916604042053223:
                                                            if Q.lam2 > 0.000524056755239144:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.14586256444454193:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top50 > 41.59354591369629:
                                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 515.40625:
                                                if Q.log_sum_pt > 6.932837247848511:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 978.241455078125:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.747124582529068:
                                                    if Q.sum_pt_top3 > 422.1875:
                                                        if Q.sum_pt > 1018.5198974609375:
                                                            return 'g'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.884534597396851:
                                                                if Q.lam2 > 0.0005676962027791888:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.6911291182041168:
                                            if Q.girth2_top20 > 0.0004093688621651381:
                                                if Q.sum_pt_top3 > 508.6875:
                                                    if Q.sum_pt > 1025.44775390625:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.8640364110469818:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0005643697804771364:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 41.8609504699707:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1001.4443359375:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.000765766657423228:
                                                    return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1023.53369140625:
                                        if Q.lam2 > 0.0004149726592004299:
                                            if Q.girth2_top20 > 0.0005517090321518481:
                                                if Q.lam2 > 0.0005732607387471944:
                                                    if Q.sum_pt_top50 > 1033.7872314453125:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 408.21875:
                                                    if Q.tau32 > 0.8750798404216766:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00010050427954411134:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005621844902634621:
                                            if Q.e2 > 0.009560981299728155:
                                                if Q.sum_pt_top40 > 960.81005859375:
                                                    if Q.z_top50_slots > 0.999642938375473:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top50 > 1003.91064453125:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 6.926304195076227e-05:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 389.0625:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0003270566085120663:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top5 > 0.00107179005863145:
                                if Q.mass_top30 > 53.81949996948242:
                                    if Q.z_dr_0p2_0p4 > 0.051542529836297035:
                                        return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 66.556396484375:
                                            if Q.e2 > 0.02415785938501358:
                                                if Q.z_top30_slots > 0.9197856187820435:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.2855919003486633:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    if Q.tau21 > 0.454219788312912:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        if Q.log_sum_pt > 6.819656133651733:
                                            if Q.tau32 > 0.7863912284374237:
                                                if Q.sum_pt_top3 > 323.90625:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.3718620240688324:
                                                if Q.tau32 > 0.7167114913463593:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9997052252292633:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.002876469399780035:
                                                    return 'q'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top3 > 478.6875:
                                    if Q.girth2_top5 > 0.00024334766931133345:
                                        if Q.z_top50_slots > 0.9994998872280121:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 575.578125:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 4.5:
                                        if Q.tau32 > 0.6588573455810547:
                                            if Q.tau21 > 0.5334626138210297:
                                                if Q.max_dr > 0.5881698727607727:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 922.39501953125:
                                                        if Q.girth2_top20 > 0.0011258201557211578:
                                                            if Q.lam1 > 0.0028564739041030407:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top30_slots > 0.9721249043941498:
                                                    if Q.sum_pt_top3 > 373.625:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0005207682261243463:
                                                if Q.lam1 > 0.003113292157649994:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.01965456549078226:
                                            if Q.sum_pt_top3 > 281.1875:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.849196195602417:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.7444636523723602:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1124.5762939453125:
                    if Q.n_particles > 33.5:
                        if Q.n_particles > 37.5:
                            if Q.e2 > 0.012033026665449142:
                                if Q.n_dr_0p2_0p4 > 2.5:
                                    if Q.sum_pt > 1199.0977783203125:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 633.125:
                                            if Q.lam2 > 0.0005341033102013171:
                                                if Q.sum_pt > 1150.6953125:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 40.5:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0004259476118022576:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 535.1875:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 7.088094472885132:
                                        if Q.dr_0 > 0.018698930740356445:
                                            return 'q'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.019424455240368843:
                                if Q.girth2_top20 > 0.0006237290217541158:
                                    if Q.sum_pt_top50 > 1255.7191162109375:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 547.875:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.0035193219082430005:
                                                if Q.tau32 > 0.8506515622138977:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.34308600425720215:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top50 > 1204.4793701171875:
                                    if Q.tau32 > 0.6615529358386993:
                                        if Q.max_dr > 0.2862464189529419:
                                            if Q.log_sum_pt > 7.1639814376831055:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 705.1875:
                                                    if Q.tau32 > 0.8981855809688568:
                                                        return 'g'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0003709220763994381:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 655.8125:
                                        if Q.lam2 > 0.00035624740121420473:
                                            if Q.girth2_top20 > 0.0003587126120692119:
                                                if Q.z_top30_slots > 0.9977510571479797:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 1148.6600341796875:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 2.5:
                                            if Q.n_particles > 34.5:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0003016110131284222:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 6.908875366207212e-05:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                    else:
                        if Q.n_particles > 30.5:
                            if Q.log_sum_pt > 7.094985723495483:
                                if Q.log_sum_pt > 7.207504034042358:
                                    if Q.tau32 > 0.737594336271286:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 643.6875:
                                        if Q.lam2 > 0.00028012516850139946:
                                            if Q.girth2_top20 > 0.0008675298595335335:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.0002693835413083434:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.0005370481230784208:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.009228990413248539:
                                            return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top3 > 572.65625:
                                    if Q.sum_pt > 1161.5953369140625:
                                        if Q.lam2 > 0.0003692351747304201:
                                            if Q.girth2_top20 > 0.0007542124949395657:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 7.652987915207632e-05:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.33834218978881836:
                                            return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.214495897293091:
                                if Q.n_particles > 27.5:
                                    if Q.dr_0 > 0.003919483860954642:
                                        if Q.n_particles > 29.5:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.0030071354703977704:
                                    return 'W'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 28.5:
                                        if Q.sum_pt_top3 > 606.3125:
                                            if Q.sum_pt_top50 > 1235.9666748046875:
                                                if Q.tau32 > 0.8634211122989655:
                                                    if Q.lam2 > 0.00023503097327193245:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 99% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 39.5:
                        if Q.sum_pt > 1058.0333251953125:
                            if Q.girth2_top20 > 0.0009541187318973243:
                                if Q.sum_pt_top3 > 545.9375:
                                    if Q.girth2_top5 > 9.31790127651766e-05:
                                        if Q.n_particles > 43.5:
                                            if Q.sum_pt_top40 > 1072.24853515625:
                                                if Q.sum_pt_top3 > 640.125:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.8272659778594971:
                                            if Q.lam2 > 0.0006917603895999491:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        if Q.sum_pt_top50 > 1080.084716796875:
                                            if Q.tau32 > 0.7829625308513641:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 1094.4327392578125:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.009467069525271654:
                                                if Q.sum_pt_top3 > 420.96875:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    if Q.sum_pt_top3 > 613.5625:
                                        if Q.lam2 > 0.0006059283623471856:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 42.5:
                                                if Q.LHA > 0.1268184781074524:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1100.845947265625:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1070.8863525390625:
                                            if Q.tau32 > 0.7406620979309082:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00010011130143539049:
                                                if Q.n_particles > 41.5:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.6820977628231049:
                                        if Q.tau32 > 0.80821892619133:
                                            if Q.max_dr > 0.3193529397249222:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.8499957323074341:
                                            if Q.z_top30_slots > 0.9776801466941833:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 924.43798828125:
                                if Q.lam1 > 0.0037350638303905725:
                                    if Q.sum_pt_top40 > 990.36083984375:
                                        if Q.lam1 > 0.004044928587973118:
                                            if Q.D2 > 1.1589043736457825:
                                                if Q.e2 > 0.021152561530470848:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 4.097711801528931:
                                                if Q.max_dr > 0.3811313211917877:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top30 > 62.313926696777344:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.3725187182426453:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 965.752685546875:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.0004011363780591637:
                                        if Q.girth2_top20 > 0.0008114966913126409:
                                            if Q.LHA > 0.26517172157764435:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top40 > 64.7845344543457:
                                                    if Q.n_dr_0p2_0p4 > 8.5:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 3.608275532722473:
                                                            return 'W'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1041.5242919921875:
                                                if Q.lam2 > 0.0005141305446159095:
                                                    if Q.n_particles > 41.5:
                                                        if Q.sum_pt_top3 > 518.28125:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 957.95166015625:
                                                    if Q.mass_top50 > 41.67191505432129:
                                                        if Q.girth2_top5 > 8.639038787805475e-05:
                                                            if Q.sum_pt_top3 > 343.25:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.log_sum_pt > 6.926555871963501:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 512.4375:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top50 > 1028.2037353515625:
                                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top3 > 388.09375:
                                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0005803868407383561:
                                                        if Q.sum_pt_top3 > 479.09375:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.011908289976418018:
                                                                return 'q'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0004356430727057159:
                                            if Q.log_sum_pt > 6.9431304931640625:
                                                if Q.girth2_top20 > 0.0003019748255610466:
                                                    if Q.sum_pt_top3 > 508.21875:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.1388211026787758:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 970.6820068359375:
                                                    if Q.n_particles > 43.5:
                                                        if Q.girth2_top5 > 0.00010235931767965667:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 449.9375:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top40 > 1004.4969482421875:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top40 > 1019.8641357421875:
                                                            if Q.tau32 > 0.8356730937957764:
                                                                if Q.sum_pt_top3 > 499.5625:
                                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 479.625:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 1040.250732421875:
                                                if Q.sum_pt_top3 > 433.59375:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 4.5:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.0008587540942244232:
                                    if Q.sum_pt_top3 > 379.953125:
                                        if Q.girth2_top5 > 0.00013604776177089661:
                                            if Q.LHA > 0.2634037137031555:
                                                return 't'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 417.5625:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.0005217962025199085:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.001036814646795392:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 495.921875:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0007443455688189715:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.4426850080490112:
                                            if Q.tau32 > 0.7530433833599091:
                                                if Q.lam2 > 0.0005671308899763972:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.000809173536254093:
                                                    if Q.e2 > 0.023430165834724903:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 63.298879623413086:
                                                if Q.D2 > 1.4548283219337463:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.mass_top30 > 39.76018524169922:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0004675167438108474:
                                        if Q.sum_pt_top3 > 442.359375:
                                            if Q.LHA > 0.13416333496570587:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 9.081112875719555e-05:
                                            if Q.sum_pt_top3 > 324.71875:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.lam1 > 0.003404597518965602:
                            if Q.sum_pt > 989.4381103515625:
                                if Q.D2 > 4.480108976364136:
                                    if Q.n_dr_0p2_0p4 > 11.5:
                                        if Q.lam1 > 0.0038691358640789986:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0036605162313207984:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                if Q.girth > 0.027169507928192616:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top30_slots > 0.9991735517978668:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.003925392869859934:
                                        if Q.lam2 > 0.00046242287498898804:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.004099855897948146:
                                                if Q.C2 > 0.059402236714959145:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0027166942600160837:
                                            if Q.lam1 > 0.004012060351669788:
                                                if Q.log_sum_pt > 6.916100740432739:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.0323626883327961:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 1.105211079120636:
                                    if Q.sum_pt_top3 > 343.34375:
                                        if Q.log_sum_pt > 6.877596855163574:
                                            if Q.girth2_top20 > 0.004266156814992428:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.028499561361968517:
                                                if Q.mass_top40 > 68.24175643920898:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 64.66412353515625:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                if Q.tau32 > 0.8391595184803009:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.734116554260254:
                                        if Q.z_dr_0p2_0p4 > 0.0006619731429964304:
                                            if Q.log_sum_pt > 6.874289512634277:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top3 > 254.40625:
                                if Q.n_particles > 36.5:
                                    if Q.sum_pt_top40 > 1068.907470703125:
                                        if Q.sum_pt_top3 > 571.8125:
                                            if Q.girth2_top5 > 5.734754631703254e-05:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 653.90625:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00041082945244852453:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00013008285168325529:
                                                if Q.sum_pt_top3 > 436.0:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.3496586084365845:
                                                        if Q.sum_pt_top40 > 1085.76220703125:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 3.5:
                                                    if Q.tau32 > 0.8282498419284821:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.00040212093153968453:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 1086.766357421875:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 911.5291748046875:
                                            if Q.log_sum_pt > 6.959018230438232:
                                                if Q.sum_pt_top3 > 522.34375:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 8.491904009133577e-05:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0004057185724377632:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.1418672651052475:
                                                if Q.sum_pt_top50 > 705.868896484375:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 36% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 457.40625:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00037422438617795706:
                                                        if Q.girth2_top5 > 6.330393080133945e-05:
                                                            if Q.tau32 > 0.8674745857715607:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.0029169406043365598:
                                        if Q.girth > 0.022894815541803837:
                                            if Q.girth2_top20 > 0.003936421126127243:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 734.5799560546875:
                                                    if Q.log_sum_pt > 6.9230194091796875:
                                                        if Q.lam1 > 0.0031810287619009614:
                                                            if Q.tau21 > 0.3933788388967514:
                                                                if Q.lam1 > 0.0032684290781617165:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                if Q.lam2 > 0.0003639777278294787:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top20 > 0.003187578869983554:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 993.942138671875:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.608351230621338:
                                            if Q.girth2_top20 > 0.002534975530579686:
                                                if Q.lam2 > 0.00020959111134288833:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 6.915953159332275:
                                                        if Q.z_dr_0p2_0p4 > 0.021860252134501934:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top30_slots > 0.9980103671550751:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.985127210617065:
                                                        if Q.sum_pt_top3 > 535.90625:
                                                            return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top5 > 9.90241824183613e-05:
                                                                return 'q'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.0002497195382602513:
                                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.807791709899902:
                                                            if Q.sum_pt > 1064.0897216796875:
                                                                if Q.sum_pt_top3 > 454.21875:
                                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.girth2_top5 > 7.910796921350993e-05:
                                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 100% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top30_slots > 0.9988445937633514:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.15082066506147385:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 380.90625:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    if Q.lam2 > 0.0004661311104428023:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 88% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
