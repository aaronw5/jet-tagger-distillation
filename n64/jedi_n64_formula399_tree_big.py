"""JEDI-linear jet tagger, 64 particles, 3 features: the simplified formula closest to the start formula (within 0.1 point on validation jets): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 80.51% (the formula: 81.41%); same class as the formula for 94.09% of jets.  934 leaves, depth 19.
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
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_9=pt[9],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
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
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def decide(Q):
    if Q.mass > 84.85325241088867:
        if Q.mass_over_sum_pt > 0.09500768035650253:
            if Q.e2 > 0.038742437958717346:
                if Q.log_sum_pt > 7.0071799755096436:
                    if Q.tau32 > 0.5237221121788025:
                        if Q.mass > 178.43948364257812:
                            if Q.tau21 > 0.41505739092826843:
                                if Q.z_top50_slots > 0.9677960276603699:
                                    if Q.sum_pt > 1270.333251953125:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04588715359568596:
                                if Q.lam2 > 0.001433116034604609:
                                    if Q.lam2 > 0.0023236306151375175:
                                        return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.05086647532880306:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6180581748485565:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top30 > 1142.2548828125:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.7361796498298645:
                                            if Q.n_dr_0p05_0p1 > 15.5:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.04147559031844139:
                                                    if Q.n_particles > 58.5:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top10 > 846.09375:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.973728358745575:
                                                if Q.z_dr_0p05_0p1 > 0.6431933641433716:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.003590135835111141:
                                    if Q.girth2 > 0.01725427247583866:
                                        if Q.z_top50_slots > 0.9598753452301025:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.n_pt_above_10 > 13.5:
                                        if Q.C2 > 0.12296987697482109:
                                            if Q.mass_top30 > 130.13045501708984:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 51% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 196.10855102539062:
                            if Q.e2 > 0.06605730205774307:
                                if Q.tau21 > 0.40244260430336:
                                    if Q.sum_pt > 1206.0439453125:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 202.5630645751953:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.053631288930773735:
                                        if Q.tau21 > 0.4806682914495468:
                                            return 't'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04425579681992531:
                                if Q.mass > 182.85150909423828:
                                    if Q.e2 > 0.05194684676826:
                                        if Q.C2 > 0.0834641233086586:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.4322737604379654:
                                            if Q.z_top50_slots > 0.9620202481746674:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.42787033319473267:
                                    if Q.mass_over_sum_pt > 0.130686417222023:
                                        if Q.mass > 177.75643920898438:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.mass > 179.19293975830078:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                else:
                    if Q.tau32 > 0.576105922460556:
                        if Q.sum_pt > 1042.14453125:
                            if Q.lam1 > 0.028610888868570328:
                                if Q.tau21 > 0.47814735770225525:
                                    return 't'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 462.03125:
                                        if Q.log_sum_pt > 6.978548526763916:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.046362193301320076:
                                    if Q.z_dr_0p05_0p1 > 0.6997401416301727:
                                        if Q.pt_dispersion > 0.35130828619003296:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1089.453125:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 179.32582092285156:
                                            if Q.sum_pt > 1068.476318359375:
                                                if Q.tau21 > 0.3842567503452301:
                                                    if Q.z_top50_slots > 0.9618770480155945:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0013110206928104162:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 439.4375:
                                                    if Q.z_dr_0p2_0p4 > 0.06115787662565708:
                                                        if Q.log_sum_pt > 6.978931903839111:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.010474300477653742:
                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9809885025024414:
                                                        if Q.max_dr > 0.28724272549152374:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 62.5:
                                        if Q.lam2 > 0.002608651528134942:
                                            if Q.e2 > 0.04248875752091408:
                                                if Q.mass > 177.61014556884766:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1062.607177734375:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9774425327777863:
                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.08096372336149216:
                                            if Q.pt_dispersion > 0.3649474233388901:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.978299140930176:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top5_slots > 0.5418290793895721:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.sum_pt_top3 > 553.671875:
                                                    if Q.planar_flow > 0.4102924019098282:
                                                        return 't'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.009785564616322517:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 723.50390625:
                                if Q.sum_pt_top3 > 553.09375:
                                    if Q.log_sum_pt > 6.891619443893433:
                                        if Q.z_dr_0p1_0p2 > 0.1555098071694374:
                                            if Q.girth2 > 0.009749213699251413:
                                                if Q.girth2_top3 > 0.005936009809374809:
                                                    if Q.lam1 > 0.02905149944126606:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0018053847597911954:
                                                if Q.mass_top5 > 53.78029441833496:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.06049572862684727:
                                                    return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top50 > 0.010332734324038029:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top5 > 3.9270509481430054:
                                            if Q.max_dr > 0.32473134994506836:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.031971486285328865:
                                        if Q.tau21 > 0.35528695583343506:
                                            if Q.z_top50_slots > 0.9487932026386261:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.2089717537164688:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9699790775775909:
                                                if Q.sum_pt_top50 > 971.5498046875:
                                                    if Q.tau21 > 0.25281526148319244:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9533912837505341:
                                                    if Q.lam1 > 0.035011544823646545:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 1009.20751953125:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.009359845891594887:
                                            if Q.sum_pt > 1000.0675048828125:
                                                if Q.girth2_top5 > 0.005210217786952853:
                                                    if Q.e2 > 0.04324923828244209:
                                                        if Q.lam1 > 0.029588050208985806:
                                                            if Q.tau21 > 0.20659618824720383:
                                                                if Q.sum_pt > 1031.181640625:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 1016.331298828125:
                                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.2863353490829468:
                                                                if Q.z_dr_0p05_0p1 > 0.6954756081104279:
                                                                    if Q.planar_flow > 0.23189670592546463:
                                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_1st > 0.18641722947359085:
                                                                            if Q.z_dr_0_0p05 > 0.0015568764647468925:
                                                                                return 'q'   # 69% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top2 > 388.09375:
                                                                        if Q.e2 > 0.05172739550471306:
                                                                            return 't'   # 93% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.planar_flow > 0.24559882283210754:
                                                                                return 't'   # 85% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'q'   # 46% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 149.3660430908203:
                                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.n_real_top50 > 46.5:
                                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.34655532240867615:
                                                            if Q.sum_pt_top3 > 465.71875:
                                                                if Q.lam1 > 0.010786234401166439:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top50_slots > 0.9698787331581116:
                                                                if Q.girth2_top20 > 0.009833957068622112:
                                                                    if Q.z_1st > 0.12258442863821983:
                                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.001428912510164082:
                                                        if Q.width > 0.02269232366234064:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.15727299451828003:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p1_0p2 > 0.1872701793909073:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top30 > 771.802734375:
                                                    if Q.girth2_top5 > 0.005406948970630765:
                                                        if Q.n_dr_0p2_0p4 > 1.5:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top40 > 0.011307820677757263:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top50 > 987.1656494140625:
                                                            if Q.lam2 > 0.0016260892734862864:
                                                                return 't'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 983.2230224609375:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.003298527910374105:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 608.25390625:
                                    if Q.e2 > 0.04744107276201248:
                                        if Q.lam1 > 0.035113897174596786:
                                            if Q.tau21 > 0.5215277075767517:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0069462307728827:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 797.21484375:
                                                if Q.girth > 0.16305390745401382:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.058811794966459274:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top30 > 647.84765625:
                            if Q.mass_over_sum_pt_sq > 0.009563890751451254:
                                if Q.mass > 191.06275177001953:
                                    if Q.tau21 > 0.33418506383895874:
                                        if Q.e2 > 0.056746164336800575:
                                            if Q.sum_pt > 1076.951904296875:
                                                if Q.z_top50_slots > 0.9696547985076904:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.981825828552246:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.957413911819458:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.1778145655989647:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.044427210465073586:
                                        return 't'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.964507341384888:
                                            if Q.n_dr_0p2_0p4 > 11.5:
                                                return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9772226512432098:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.45744624733924866:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 808.873046875:
                                                if Q.sum_pt_top30 > 967.671875:
                                                    if Q.z_dr_0p1_0p2 > 0.11402452737092972:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0015098127187229693:
                                                            if Q.sum_pt_top3 > 592.9375:
                                                                return 'q'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 956.71875:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top5 > 38.820112228393555:
                                return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.z_top20_slots > 0.7349534034729004:
                                    return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 696.525390625:
                                        return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1038.2291259765625:
                    if Q.n_particles > 55.5:
                        if Q.tau32 > 0.4231249690055847:
                            if Q.sum_pt > 1074.3668212890625:
                                return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9848356246948242:
                                    if Q.girth2_top3 > 0.0057242710608989:
                                        return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top2 > 432.46875:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6504928171634674:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.07510743662714958:
                                                    return 't'   # 39% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 131.67235565185547:
                                        if Q.tau32 > 0.6325021982192993:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top50 > 0.01505257235839963:
                                if Q.z_top50_slots > 0.9630928039550781:
                                    return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.13796966522932053:
                                        if Q.mass > 177.52716064453125:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.98167285323143:
                                    if Q.z_dr_0p1_0p2 > 0.1643831431865692:
                                        return 't'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1086.097900390625:
                            if Q.pt_9 > 21.34375:
                                return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top50 > 1131.364013671875:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top10 > 951.0703125:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.pt_dispersion > 0.3331304043531418:
                                return 'q'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p1_0p2 > 0.12730418145656586:
                                    return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.964958906173706:
                                        return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.028727149590849876:
                        if Q.sum_pt > 988.6695556640625:
                            if Q.z_top50_slots > 0.9766022264957428:
                                if Q.z_dr_0p1_0p2 > 0.1585603505373001:
                                    if Q.sum_pt_top2 > 371.53125:
                                        if Q.mass > 98.4509162902832:
                                            if Q.n_dr_0p1_0p2 > 19.5:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 41% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 7.5:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.009383811615407467:
                                                if Q.max_dr > 0.3704424351453781:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 98.83872985839844:
                                        if Q.pt_dispersion > 0.26616905629634857:
                                            if Q.C2 > 0.14915551990270615:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top5_slots > 0.5734696388244629:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.12147370725870132:
                                                        return 't'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.001600167015567422:
                                                if Q.girth2_top5 > 0.004755201982334256:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 130.48783111572266:
                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.923641204833984:
                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.931265115737915:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005563195154536515:
                                            if Q.e2 > 0.035067373886704445:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.009214980993419886:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 141.57955932617188:
                                    if Q.tau32 > 0.6493726968765259:
                                        if Q.log_sum_pt > 6.918794393539429:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9681256711483002:
                                        if Q.sum_pt_top40 > 939.4296875:
                                            if Q.tau32 > 0.718759298324585:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.00610455428250134:
                                                    if Q.max_dr > 0.36271098256111145:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.45870010554790497:
                                            if Q.sum_pt > 994.90625:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03397788107395172:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 119.2587776184082:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 766.423828125:
                                if Q.z_top50_slots > 0.9633165895938873:
                                    if Q.sum_pt_top30 > 936.32470703125:
                                        if Q.z_dr_0_0p05 > 0.7936128079891205:
                                            if Q.mass_over_sum_pt_sq > 0.009434557054191828:
                                                if Q.tau32 > 0.5343530476093292:
                                                    if Q.sum_pt_top40 > 955.517578125:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.09459931030869484:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_9 > 22.046875:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.03148762136697769:
                                            if Q.z_dr_0_0p05 > 0.8626042902469635:
                                                return 'q'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.881756067276001:
                                                if Q.sum_pt_top3 > 421.046875:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9756948053836823:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.033088164404034615:
                                        if Q.z_top50_slots > 0.9428299963474274:
                                            if Q.girth > 0.10561053454875946:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 965.97119140625:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 767.8203125:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.5573297739028931:
                                                    if Q.e2 > 0.03551951423287392:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top50_slots > 0.9333809614181519:
                                                            return 't'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3405406326055527:
                                            if Q.girth2_top5 > 0.005137348081916571:
                                                if Q.sum_pt > 960.9775390625:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.5647582709789276:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.12255881354212761:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 744.6669921875:
                                        if Q.e2 > 0.030771725811064243:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                    else:
                        if Q.z_top50_slots > 0.9842063784599304:
                            if Q.sum_pt_top40 > 946.5439453125:
                                if Q.pt_dispersion > 0.28453338146209717:
                                    return 'q'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1008.18212890625:
                                        if Q.n_particles > 56.5:
                                            if Q.girth2_top5 > 0.0012041670270264149:
                                                if Q.tau32 > 0.7232823371887207:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.12213030830025673:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.004466027719900012:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.023436645045876503:
                                    if Q.max_dr > 0.48178812861442566:
                                        return 'q'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9929121136665344:
                                        return 't'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9754172265529633:
                                if Q.e2 > 0.023828725330531597:
                                    if Q.sum_pt_top30 > 882.234375:
                                        if Q.pt_dispersion > 0.252415731549263:
                                            if Q.z_dr_0_0p05 > 0.7390759885311127:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1018.35009765625:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top10 > 21.79201030731201:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 106.26579284667969:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.018539669923484325:
                                    if Q.tau21 > 0.6752403974533081:
                                        return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.438857078552246:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 984.1806640625:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 77.76479721069336:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.sum_pt > 1072.51220703125:
                    if Q.z_dr_0p2_0p4 > 0.0032225524773821235:
                        if Q.log_sum_pt > 7.017247200012207:
                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.005946426186710596:
                                if Q.e2 > 0.023396477103233337:
                                    if Q.mass > 95.49390029907227:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top30 > 0.005522560095414519:
                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9719784557819366:
                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1245.4775390625:
                            if Q.mass > 92.90767669677734:
                                return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 79.65656280517578:
                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 95.26943969726562:
                                return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 68.62350082397461:
                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02443495485931635:
                        if Q.sum_pt > 982.4052734375:
                            if Q.mass > 94.34076690673828:
                                if Q.mass_top30 > 76.51801681518555:
                                    if Q.mass > 97.79297256469727:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 68.29390716552734:
                                    if Q.max_dr > 0.3684229701757431:
                                        if Q.sum_pt > 993.804443359375:
                                            if Q.girth2_top30 > 0.006409280234947801:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 91.13174438476562:
                                                    if Q.mass_top40 > 81.64868927001953:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.06694604456424713:
                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.32509712874889374:
                                        if Q.z_top50_slots > 0.9747491180896759:
                                            if Q.tau32 > 0.7457468509674072:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.31835953891277313:
                                if Q.e2 > 0.027215569280087948:
                                    return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9774201512336731:
                                        return 't'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 961.27294921875:
                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 60% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.31468693912029266:
                            if Q.z_top50_slots > 0.9806741774082184:
                                if Q.LHA > 0.23812180012464523:
                                    if Q.max_dr > 0.37582333385944366:
                                        if Q.lam2 > 0.001150808995589614:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 93.26172637939453:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_9 > 15.390625:
                                        if Q.girth2_top15 > 0.0042755575850605965:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.4931105077266693:
                                    if Q.mass_top30 > 64.48244094848633:
                                        if Q.mass > 90.3221664428711:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 78.65564346313477:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 47% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9559215009212494:
                                if Q.mass > 91.7824935913086:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p2_0p4 > 0.006978678982704878:
                                    if Q.max_dr > 0.24097146838903427:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 81% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.28694534301758:
                    if Q.log_sum_pt > 7.031641721725464:
                        if Q.n_dr_0p2_0p4 > 4.5:
                            return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.133136749267578:
                                return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.girth2_top20 > 0.0071361500304192305:
                            if Q.n_dr_0p2_0p4 > 6.5:
                                if Q.girth2_top3 > 0.00643787975423038:
                                    return 't'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.030726746656000614:
                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 89% of the training jets here get this class from the formula
                        else:
                            if Q.z_top5_slots > 0.6559404134750366:
                                if Q.girth2_top30 > 0.007399301044642925:
                                    return 'q'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
                else:
                    if Q.mass_top50 > 86.53750991821289:
                        if Q.girth2_top20 > 0.0038607511669397354:
                            if Q.sum_pt_top50 > 973.6759033203125:
                                if Q.D2 > 4.7593536376953125:
                                    if Q.mass > 92.27801895141602:
                                        if Q.girth2_top40 > 0.007284537656232715:
                                            if Q.n_dr_0p2_0p4 > 16.5:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 15.5:
                                            if Q.log_sum_pt > 6.905303239822388:
                                                if Q.z_dr_0p2_0p4 > 0.061114368960261345:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 96.98636245727539:
                                        if Q.girth2_top20 > 0.0064019698183983564:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                if Q.girth2_top30 > 0.006589032709598541:
                                                    if Q.n_dr_0p2_0p4 > 13.5:
                                                        if Q.girth2_top30 > 0.00727075757458806:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top30 > 0.005723301786929369:
                                            if Q.sum_pt_top50 > 985.89892578125:
                                                if Q.mass > 86.94502258300781:
                                                    if Q.n_dr_0p2_0p4 > 23.5:
                                                        if Q.mass > 93.37844467163086:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.008832832798361778:
                                                            if Q.girth2_top20 > 0.006937683327123523:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 1.7542251348495483:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top30 > 1055.478271484375:
                                                        if Q.z_dr_0p2_0p4 > 0.030075903981924057:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.3130827099084854:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.z_dr_0p2_0p4 > 0.04619820974767208:
                                                        if Q.D2 > 3.8488165140151978:
                                                            return 'q'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 7.5:
                                                            if Q.max_dr > 0.35803332924842834:
                                                                return 't'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                if Q.mass > 92.68731307983398:
                                                    if Q.mass > 94.01050567626953:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top30 > 0.005227292655035853:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.038766760379076004:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top30 > 0.005095307948067784:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 87.71007537841797:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.35087503492832184:
                                                        if Q.D2 > 1.1173434853553772:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.34887444972991943:
                                    if Q.lam2 > 0.00047547715075779706:
                                        if Q.girth > 0.06295016780495644:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                if Q.D2 > 1.8308452367782593:
                                                    if Q.sum_pt_top50 > 966.076904296875:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.861424684524536:
                                                if Q.D2 > 3.5206586122512817:
                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 961.16845703125:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_real_top50 > 31.5:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.846046209335327:
                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.3076027184724808:
                                if Q.log_sum_pt > 6.992799758911133:
                                    if Q.n_dr_0p2_0p4 > 6.5:
                                        if Q.tau32 > 0.4757426530122757:
                                            if Q.girth2_top30 > 0.004812574712559581:
                                                if Q.mass > 92.69016647338867:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.000825586001155898:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 93.39033126831055:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 6.4644224643707275:
                                        if Q.log_sum_pt > 6.942215442657471:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 40% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.008233145345002413:
                                            if Q.sum_pt_top10 > 675.6171875:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top40 > 0.006273310165852308:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 93.71084213256836:
                                    if Q.mass_top5 > 10.00930404663086:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.mass_over_sum_pt > 0.08240700513124466:
                            if Q.sum_pt_top50 > 963.772216796875:
                                if Q.D2 > 1.3962942957878113:
                                    if Q.D2 > 5.532696962356567:
                                        if Q.n_dr_0p2_0p4 > 15.5:
                                            if Q.z_top50_slots > 0.9949380457401276:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 981.554443359375:
                                            if Q.n_dr_0p2_0p4 > 18.5:
                                                if Q.sum_pt_top50 > 1003.1246337890625:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top30 > 0.006894240388646722:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.527384638786316:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.35645194351673126:
                                        if Q.girth2 > 0.007191742770373821:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.sum_pt > 996.6024169921875:
                                                    if Q.D2 > 0.9132562875747681:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 86.11717224121094:
                                                if Q.tau21 > 0.2423155978322029:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top30 > 990.068359375:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.7505226731300354:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3142002522945404:
                                            if Q.width > 0.0071712699718773365:
                                                if Q.log_sum_pt > 6.885485887527466:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 85.6048583984375:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.31176629662513733:
                                    if Q.e2 > 0.022278619930148125:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 45% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.847999095916748:
                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 59% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 1.8003619313240051:
                                if Q.girth2_top30 > 0.00474092410877347:
                                    if Q.D2 > 2.452801465988159:
                                        if Q.girth2_top30 > 0.005305036902427673:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.020978638902306557:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3714265674352646:
                                            if Q.mass > 86.31727981567383:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 85.32005310058594:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.2890917509794235:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.28851184248924255:
                                    if Q.n_particles > 52.5:
                                        if Q.girth2_top30 > 0.005051382351666689:
                                            if Q.mass > 86.03764343261719:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top15 > 0.00355069269426167:
                                            if Q.mass > 86.29625701904297:
                                                if Q.max_dr > 0.3564542979001999:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 85.61466979980469:
                                        if Q.C2 > 0.02117217145860195:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top10 > 847.828125:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 69% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.45967102050781:
            if Q.girth2 > 0.006993504939600825:
                if Q.log_sum_pt > 6.881679058074951:
                    if Q.D2 > 1.9282103776931763:
                        if Q.mass_top30 > 62.21279525756836:
                            if Q.n_dr_0p2_0p4 > 14.5:
                                if Q.sum_pt_top40 > 987.556396484375:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 48% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 79% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 71% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.31106625497341156:
                            if Q.width > 0.007174941478297114:
                                if Q.sum_pt > 989.8363037109375:
                                    return 'W'   # 38% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.34151114523410797:
                                        return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 976.046875:
                                    return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 83.59348678588867:
                                return 'Z'   # 95% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 50% of the training jets here get this class from the formula
                else:
                    if Q.D2 > 3.2682838439941406:
                        if Q.mass_top30 > 67.81707763671875:
                            if Q.sum_pt_top40 > 956.688232421875:
                                if Q.n_dr_0p2_0p4 > 14.5:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 77.11709594726562:
                                    return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top3 > 429.3125:
                                return 't'   # 37% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top40 > 710.99853515625:
                            if Q.max_dr > 0.2726328819990158:
                                if Q.z_dr_0_0p05 > 0.826786071062088:
                                    if Q.mass_top50 > 76.56512451171875:
                                        if Q.sum_pt_top30 > 942.6990966796875:
                                            if Q.mass_top40 > 82.28550720214844:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9683734476566315:
                                        if Q.max_dr > 0.4955512583255768:
                                            if Q.sum_pt_top40 > 937.263427734375:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.0018708878778852522:
                                                if Q.mass_top20 > 50.5340690612793:
                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9809892475605011:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.007115938933566213:
                                                    if Q.mass > 83.03678131103516:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 82.39320755004883:
                                    if Q.z_dr_0p2_0p4 > 0.002941470476798713:
                                        return 't'   # 46% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.00715815438888967:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 63% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top2 > 241.90625:
                                return 'q'   # 56% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 52% of the training jets here get this class from the formula
            else:
                if Q.girth2_top20 > 0.0032219437416642904:
                    if Q.sum_pt_top50 > 965.0179443359375:
                        if Q.C2 > 0.09220858290791512:
                            if Q.mass > 82.32430267333984:
                                if Q.mass_top50 > 82.97179412841797:
                                    if Q.girth2_top40 > 0.005352341337129474:
                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.005152292083948851:
                                        if Q.D2 > 4.052947521209717:
                                            if Q.n_dr_0p2_0p4 > 14.5:
                                                return 'q'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top50 > 0.006515562301501632:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 13.5:
                                    if Q.log_sum_pt > 6.908708095550537:
                                        if Q.D2 > 5.656208276748657:
                                            if Q.n_dr_0p2_0p4 > 16.5:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.004065715009346604:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top40 > 0.005064172670245171:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.38730470836162567:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.006558129331097007:
                                        if Q.C2 > 0.10447195172309875:
                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0037156192120164633:
                                            if Q.sum_pt > 980.3070068359375:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.000818167463876307:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 9.5:
                                                if Q.pt_9 > 12.37109375:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top40 > 72.50880432128906:
                                if Q.mass_over_sum_pt > 0.08227086067199707:
                                    if Q.D2 > 2.638716220855713:
                                        if Q.mass > 83.29878616333008:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 3.8201589584350586:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 977.785400390625:
                                            if Q.z_dr_0p2_0p4 > 0.011259545106440783:
                                                if Q.mass > 83.7085952758789:
                                                    if Q.max_dr > 0.3492288738489151:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.D2 > 1.3050451278686523:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 83.8036003112793:
                                        if Q.D2 > 3.1018893718719482:
                                            if Q.girth2_top30 > 0.005265020066872239:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 62.5:
                                                if Q.girth2_top30 > 0.0053315728437155485:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 17.5:
                                            if Q.sum_pt_top40 > 993.104248046875:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 975.7513427734375:
                                                if Q.n_particles > 63.5:
                                                    if Q.mass > 82.60203170776367:
                                                        if Q.girth2_top30 > 0.005060733761638403:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    if Q.D2 > 1.3532834649085999:
                                                        if Q.planar_flow > 0.2856399565935135:
                                                            if Q.C2 > 0.05702507682144642:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 82.24678421020508:
                                    if Q.n_particles > 63.5:
                                        if Q.girth2_top50 > 0.0062652286142110825:
                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top20_slots > 0.909429281949997:
                                        if Q.planar_flow > 0.31856945157051086:
                                            if Q.mass_top40 > 71.06605529785156:
                                                if Q.sum_pt_top50 > 986.0513916015625:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 990.10009765625:
                                                    if Q.tau21 > 0.3551569879055023:
                                                        if Q.sum_pt_top2 > 546.875:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top40 > 0.004478711634874344:
                                            if Q.sum_pt > 979.7677001953125:
                                                if Q.mass > 80.87408828735352:
                                                    if Q.girth2_top50 > 0.0053155431523919106:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.009170750621706247:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.31309595704078674:
                            if Q.z_dr_0_0p05 > 0.8138881921768188:
                                if Q.planar_flow > 0.2510802894830704:
                                    if Q.mass_top20 > 69.68191146850586:
                                        return 'W'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top5_slots > 0.614812046289444:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 33% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p2_0p4 > 0.0022089419653639197:
                                    if Q.sum_pt > 965.8631591796875:
                                        return 'W'   # 32% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.1440461054444313:
                                            if Q.tau32 > 0.5762490332126617:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0009373294305987656:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 61% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.002006669295951724:
                                if Q.z_dr_0_0p05 > 0.7632027268409729:
                                    return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.8573668003082275:
                                        return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 56% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.planar_flow > 0.41170477867126465:
                        if Q.sum_pt > 1075.8192138671875:
                            if Q.mass_top30 > 68.01529312133789:
                                if Q.e2 > 0.018292993307113647:
                                    if Q.max_dr > 0.34607014060020447:
                                        if Q.mass > 83.22188949584961:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 71.2741584777832:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top10 > 969.1328125:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top40 > 35.5:
                                        if Q.max_dr > 0.3533656448125839:
                                            if Q.log_sum_pt > 7.035356760025024:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top20_slots > 0.9505569040775299:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top20 > 0.002541986061260104:
                                    if Q.mass > 80.88020706176758:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.291890487074852:
                                            if Q.girth2_top40 > 0.004315941594541073:
                                                if Q.girth2_top3 > 0.0006367386959027499:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top5_slots > 0.7575917541980743:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 14.5:
                                if Q.z_top5_slots > 0.628993421792984:
                                    if Q.D2 > 4.877321481704712:
                                        if Q.z_top30_slots > 0.960628867149353:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 75.17959594726562:
                                            return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9943632483482361:
                                        if Q.girth2_top5 > 0.00047812079719733447:
                                            if Q.log_sum_pt > 6.907766342163086:
                                                if Q.mass > 82.3165283203125:
                                                    return 'Z'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top5_slots > 0.5604726672172546:
                                                if Q.log_sum_pt > 6.927404880523682:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 81.639892578125:
                                    if Q.girth2_top50 > 0.006107485154643655:
                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top50 > 80.04613876342773:
                                            return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 73.45200729370117:
                                        if Q.sum_pt_top50 > 978.8790283203125:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 36% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 546.4375:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.9046711921691895:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 50% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 82.65948486328125:
                            if Q.mass_top20 > 70.3990249633789:
                                return 'W'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top40 > 0.005737028317525983:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6997499763965607:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 48% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.3156609982252121:
                                if Q.girth2_top40 > 0.004479676717892289:
                                    return 'W'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 69.63544082641602:
                                        return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.016469310969114304:
                                            if Q.tau21 > 0.36797352135181427:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 93% of the training jets here get this class from the formula
        else:
            if Q.n_real_top50 > 44.5:
                if Q.sum_pt > 1052.7176513671875:
                    if Q.girth2_top30 > 0.0035438460763543844:
                        if Q.mass_top40 > 65.03044128417969:
                            if Q.mass > 80.73271560668945:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top40 > 0.00443697813898325:
                                    if Q.e2 > 0.01883033849298954:
                                        return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 72.15937042236328:
                                        if Q.tau32 > 0.7105074226856232:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_phi > -7.317660129046999e-05:
                                                if Q.girth2_top30 > 0.0037921753246337175:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_9 > 21.75:
                                            return 'g'   # 42% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.log_sum_pt > 6.9849631786346436:
                            if Q.planar_flow > 0.3167423605918884:
                                if Q.n_real_top50 > 49.5:
                                    return 'g'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top15 > 0.0008156641270034015:
                                        if Q.sum_pt_top30 > 1103.7021484375:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                if Q.sum_pt_top3 > 708.15625:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9996465444564819:
                                if Q.girth2_top15 > 0.0006732186302542686:
                                    if Q.sum_pt_top2 > 475.4375:
                                        return 'q'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.sum_pt > 1064.85009765625:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0005517669196706265:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 97% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 56.5:
                        if Q.mass_top30 > 56.160959243774414:
                            if Q.girth2 > 0.006285723997280002:
                                if Q.D2 > 2.7580090761184692:
                                    if Q.max_dr > 0.2251645103096962:
                                        if Q.tau32 > 0.7167128920555115:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.026157939806580544:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006556135835126042:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 73.57707595825195:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 42% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.981181412935257:
                                        if Q.log_sum_pt > 6.889266490936279:
                                            return 'q'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.0424183439463377:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.027167762629687786:
                                            return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.33544546365737915:
                                    if Q.mass_top40 > 65.73928451538086:
                                        if Q.n_dr_0p2_0p4 > 11.5:
                                            if Q.LHA > 0.1930958405137062:
                                                if Q.z_top5_slots > 0.6567037999629974:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 45% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.904685974121094:
                                                if Q.mass > 80.31637191772461:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 72.73786163330078:
                                                        if Q.girth2_top15 > 0.0016348767676390707:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 49% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 968.273681640625:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 74.65809631347656:
                                            if Q.tau32 > 0.6395782232284546:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 975.8651123046875:
                                                if Q.log_sum_pt > 6.926318883895874:
                                                    return 'g'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top20_slots > 0.8001815974712372:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.6217066049575806:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 71.40892791748047:
                                        if Q.sum_pt > 976.24267578125:
                                            if Q.z_top50_slots > 0.9469934105873108:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 44% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 62.5:
                                if Q.mass_top20 > 37.737083435058594:
                                    if Q.width > 0.0047656206879764795:
                                        if Q.max_dr > 0.3001291900873184:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 942.220703125:
                                                if Q.mass_top50 > 67.60336685180664:
                                                    if Q.z_top50_slots > 0.9507394134998322:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1020.764404296875:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 965.02294921875:
                                                if Q.z_dr_0p2_0p4 > 0.007940502371639013:
                                                    if Q.z_1st > 0.16526315361261368:
                                                        if Q.girth2_top3 > 0.0006775865913368762:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9751294851303101:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top20 > 30.436716079711914:
                                        if Q.girth2 > 0.0034137278562411666:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1017.57373046875:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 930.0986328125:
                                                    if Q.tau32 > 0.817412942647934:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 17.003875732421875:
                                    if Q.log_sum_pt > 6.866481065750122:
                                        if Q.sum_pt_top50 > 1016.693359375:
                                            if Q.tau32 > 0.7317779660224915:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.sum_pt_top2 > 431.3125:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.028295435942709446:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.206381194293499:
                                                    if Q.mass > 69.28722763061523:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.008973462507128716:
                                                if Q.tau32 > 0.7207809090614319:
                                                    if Q.tau21 > 0.6655612885951996:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 397.84375:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.02834650594741106:
                                                                return 'q'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01584533229470253:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top20 > 41.034332275390625:
                                            if Q.mass_top50 > 64.66033172607422:
                                                return 't'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.005523683736100793:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top15 > 0.0006668926507700235:
                                        if Q.sum_pt > 959.8331298828125:
                                            if Q.log_sum_pt > 6.928351163864136:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0006405697495210916:
                                                    if Q.sum_pt_top2 > 385.0:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 0.000593363685766235:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top20 > 23.349926948547363:
                            if Q.sum_pt_top40 > 926.70654296875:
                                if Q.mass_top50 > 70.22299575805664:
                                    if Q.z_top20_slots > 0.8972160518169403:
                                        return 'q'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 990.1116943359375:
                                            return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.7087087035179138:
                                                return 'q'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top15 > 0.0007793061959091574:
                                        if Q.sum_pt > 1032.267822265625:
                                            if Q.n_particles > 51.5:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.sum_pt_top3 > 549.21875:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7531904578208923:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 966.21435546875:
                                                if Q.mass > 68.12324523925781:
                                                    if Q.pt_dispersion > 0.267982080578804:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 3.2957743406295776:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9997554421424866:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 7.5:
                                                            if Q.pt_9 > 31.5625:
                                                                if Q.girth2_top3 > 0.0005395713960751891:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.3397156596183777:
                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                        if Q.z_top5_slots > 0.6046583950519562:
                                                            return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.016301078721880913:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top15 > 0.002464874880388379:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.999879777431488:
                                            if Q.sum_pt > 1029.813720703125:
                                                if Q.lam2 > 0.0006229745922610164:
                                                    if Q.pt_dispersion > 0.365434393286705:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.8680853843688965:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_1st > 0.23372989892959595:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.939908504486084:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 9.038828284246847e-05:
                                                    if Q.sum_pt_top50 > 979.303466796875:
                                                        if Q.sum_pt > 1011.1937255859375:
                                                            if Q.tau32 > 0.8638031184673309:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.0031369454227387905:
                                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 514.75:
                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 26.981669425964355:
                                    if Q.mass_top50 > 64.59712982177734:
                                        if Q.z_dr_0p2_0p4 > 0.04083515703678131:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 827.5185546875:
                                            if Q.tau32 > 0.7830647528171539:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.003945542266592383:
                                                return 't'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.sum_pt_top2 > 399.625:
                                            if Q.girth2_top3 > 0.00018940533482236788:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6695098876953125:
                                                if Q.lam2 > 0.0005411519668996334:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top30 > 840.69140625:
                                                    return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                        else:
                            if Q.lam2 > 0.0005047474114689976:
                                if Q.sum_pt_top40 > 951.84375:
                                    if Q.girth2_top15 > 0.0002645990753080696:
                                        if Q.n_particles > 51.5:
                                            if Q.LHA > 0.17972327023744583:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                    if Q.z_1st > 0.34944501519203186:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 1018.1048583984375:
                                                if Q.pt_9 > 21.3359375:
                                                    if Q.tau32 > 0.8363303542137146:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_1st > 0.19794011861085892:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.8315332233905792:
                                                    if Q.sum_pt_top2 > 394.65625:
                                                        return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 6.5:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.3522571176290512:
                                            if Q.sum_pt > 1016.3758544921875:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_real_top50 > 49.5:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top30 > 988.124755859375:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 4.290566682815552:
                                                        return 'q'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top20 > 15.331752300262451:
                                    if Q.log_sum_pt > 6.85376238822937:
                                        if Q.log_sum_pt > 6.93271279335022:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                if Q.girth2_top5 > 7.536018893006258e-05:
                                                    if Q.n_particles > 50.5:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0004189546743873507:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.00010909625780186616:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_real_top50 > 49.5:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0002877314982470125:
                                        if Q.n_real_top50 > 47.5:
                                            if Q.tau32 > 0.7751288414001465:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 967.22265625:
                                                if Q.sum_pt > 1031.55712890625:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top20 > 0.00016197968943743035:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 1024.126953125:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top50 > 1116.3072509765625:
                    if Q.n_particles > 33.5:
                        if Q.girth2_top15 > 0.0005367645644582808:
                            if Q.n_real_top50 > 38.5:
                                if Q.log_sum_pt > 7.077098369598389:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_dispersion > 0.4209514558315277:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0034679585369303823:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 1193.81982421875:
                                    if Q.tau32 > 0.8021562695503235:
                                        if Q.planar_flow > 0.42905496060848236:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top15 > 0.0007414705469273031:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_dispersion > 0.3544498533010483:
                                        return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            return 'g'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.n_real_top50 > 36.5:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 1204.583984375:
                                    return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 3.805484175682068:
                                        if Q.pt_9 > 23.2421875:
                                            if Q.lam2 > 0.0004069871793035418:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top10 > 8.227584838867188:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.372733399271965:
                                            if Q.max_dr > 0.38116101920604706:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.n_real_top40 > 29.5:
                            if Q.sum_pt_top40 > 1213.6473388671875:
                                if Q.pt_9 > 22.8203125:
                                    if Q.girth2_top5 > 8.844446711009368e-05:
                                        if Q.n_dr_0p2_0p4 > 2.5:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_9 > 31.1796875:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 30.785625457763672:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1330.4090576171875:
                                        return 'g'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.911454051733017:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 436.8125:
                                    return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.000196356100786943:
                                        if Q.girth2_top5 > 9.631142893340439e-05:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top30 > 65.55248641967773:
                                return 'W'   # 59% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 97% of the training jets here get this class from the formula
                else:
                    if Q.n_real_top40 > 38.5:
                        if Q.sum_pt > 1069.248046875:
                            if Q.girth2_top15 > 0.0003318620438221842:
                                if Q.pt_dispersion > 0.3929590731859207:
                                    return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.planar_flow > 0.6769154965877533:
                                            if Q.tau32 > 0.7133272886276245:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_dispersion > 0.32134735584259033:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.7820379436016083:
                                            if Q.sum_pt_top2 > 337.125:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 516.4375:
                                    if Q.sum_pt > 1086.5791015625:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top50 > 0.0007948996790219098:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.7907520234584808:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 901.23046875:
                                if Q.girth2_top15 > 0.00019470023835310712:
                                    if Q.lam1 > 0.003738011815585196:
                                        if Q.log_sum_pt > 6.896570205688477:
                                            if Q.mass_top10 > 20.901726722717285:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.537304401397705:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 969.9029541015625:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1051.0028076171875:
                                            if Q.girth2_top20 > 0.0006652020383626223:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00044474340393207967:
                                                    if Q.sum_pt_top2 > 385.3125:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 1041.378662109375:
                                        if Q.pt_dispersion > 0.3785066157579422:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00035393520374782383:
                                                if Q.tau21 > 0.7061126828193665:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_particles > 40.5:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 955.08837890625:
                                            if Q.lam2 > 0.0004510989674599841:
                                                if Q.pt_dispersion > 0.3044019490480423:
                                                    if Q.sum_pt > 1026.4228515625:
                                                        if Q.sum_pt_top2 > 420.90625:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.8354827463626862:
                                                        if Q.LHA > 0.12578276544809341:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_real_top50 > 40.5:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top20 > 23.49809169769287:
                                    if Q.sum_pt_top30 > 717.78173828125:
                                        if Q.girth > 0.05703452602028847:
                                            if Q.sum_pt > 801.13720703125:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_1st > 0.22983967512845993:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    if Q.tau32 > 0.8487933874130249:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.000488497142214328:
                                        if Q.tau32 > 0.7407234311103821:
                                            if Q.sum_pt_top2 > 401.3125:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top10 > 9.412740707397461:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.lam1 > 0.0032328892266377807:
                            if Q.sum_pt_top30 > 979.8662109375:
                                if Q.z_dr_0_0p05 > 0.9248844385147095:
                                    if Q.n_dr_0p2_0p4 > 10.5:
                                        if Q.planar_flow > 0.3367619663476944:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.06527505069971085:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_1st > 0.3705940246582031:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0035353160928934813:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                if Q.pt_9 > 21.484375:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 66.87892150878906:
                                        if Q.lam2 > 0.0002846514544216916:
                                            if Q.girth > 0.03672112710773945:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top5_slots > 0.8150405585765839:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top10 > 20.387700080871582:
                                            if Q.planar_flow > 0.15925147384405136:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 65.24583435058594:
                                                return 'W'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.052952276542782784:
                                    if Q.mass_top30 > 64.94873046875:
                                        if Q.z_dr_0p2_0p4 > 0.0006619731429964304:
                                            if Q.C2 > 0.04729722626507282:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.08229618147015572:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top10 > 618.21875:
                                if Q.planar_flow > 0.24427923560142517:
                                    if Q.n_real_top50 > 35.5:
                                        if Q.sum_pt_top50 > 1076.5439453125:
                                            if Q.pt_dispersion > 0.3763107657432556:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 8.401155719184317e-05:
                                                    if Q.tau32 > 0.8388267755508423:
                                                        if Q.lam2 > 0.00031559808121528476:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0002562021982157603:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 894.22998046875:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top15 > 0.0002377615965087898:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 3.288723349571228:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.015517937950789928:
                                        if Q.C2 > 0.08839572966098785:
                                            if Q.sum_pt_top30 > 992.9508056640625:
                                                if Q.pt_9 > 11.18359375:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.15066402405500412:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_top30_slots > 0.9992038607597351:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.7726839780807495:
                                        if Q.mass_top10 > 7.694080114364624:
                                            if Q.mass_top30 > 29.44676113128662:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top10 > 511.0234375:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 63% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
