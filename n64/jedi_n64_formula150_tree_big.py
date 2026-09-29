"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 80.36% (the formula: 81.02%); same class as the formula for 94.30% of jets.  931 leaves, depth 17.
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
                            if Q.log_sum_pt > 7.037416219711304:
                                return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.024325110018253326:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.05234529264271259:
                                        if Q.planar_flow > 0.4642770141363144:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.047336241230368614:
                                if Q.lam2 > 0.0014328158576972783:
                                    if Q.e2 > 0.05072072707116604:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.618948221206665:
                                            if Q.mass_top10 > 86.91726303100586:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.05389290675520897:
                                        if Q.mass_top50 > 171.05247497558594:
                                            if Q.tau32 > 0.6033962070941925:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1190.838134765625:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.6687709093093872:
                                            if Q.sum_pt_top30 > 1153.7607421875:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_particles > 40.5:
                                                    if Q.dr_0 > 0.1117454394698143:
                                                        if Q.girth2_top15 > 0.017810741439461708:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 33% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.6616984009742737:
                                    if Q.mass > 118.38384628295898:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9986846745014191:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.04323302023112774:
                                        if Q.z_top50_slots > 0.9660884737968445:
                                            if Q.sum_pt_top30 > 1153.984375:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 152.4620361328125:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 135.1864242553711:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top50 > 0.021321885287761688:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.13075994700193405:
                                            if Q.z_top50_slots > 0.9627405405044556:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9942085146903992:
                                                if Q.mass_top20 > 95.25376510620117:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 188.2408676147461:
                            if Q.log_sum_pt > 7.089750528335571:
                                if Q.tau32 > 0.2966904640197754:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1251.0123291015625:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9676743447780609:
                                    if Q.tau21 > 0.3720589578151703:
                                        if Q.lam1 > 0.03291686438024044:
                                            return 'g'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.5447897017002106:
                                        if Q.z_top50_slots > 0.953311413526535:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.044231826439499855:
                                if Q.mass > 181.2772445678711:
                                    if Q.e2 > 0.05117662623524666:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.48019278049469:
                                            return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 178.46058654785156:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.41493403911590576:
                                        if Q.girth2_top50 > 0.01949719339609146:
                                            return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9744074046611786:
                                                return 't'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                else:
                    if Q.lam1 > 0.03233633004128933:
                        if Q.tau21 > 0.35214319825172424:
                            if Q.e2 > 0.05962803214788437:
                                if Q.sum_pt_top30 > 590.578125:
                                    return 't'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.5219308733940125:
                                    return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 785.90234375:
                                        if Q.mass > 208.62850952148438:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9697490930557251:
                                if Q.sum_pt_top50 > 980.03515625:
                                    if Q.log_sum_pt > 6.96143651008606:
                                        if Q.z_top50_slots > 0.984166294336319:
                                            return 'q'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9772050976753235:
                                            if Q.tau21 > 0.2635049372911453:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.035132862627506256:
                                        if Q.planar_flow > 0.26373566687107086:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1009.0283203125:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.03560861013829708:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top20_slots > 0.6763126850128174:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.04702729731798172:
                            if Q.sum_pt_top30 > 626.35546875:
                                if Q.girth2_top50 > 0.00960738817229867:
                                    if Q.tau32 > 0.5855484008789062:
                                        if Q.mass > 178.92652130126953:
                                            if Q.sum_pt > 1034.28662109375:
                                                if Q.planar_flow > 0.5485918819904327:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9734527468681335:
                                                        if Q.tau21 > 0.2168898805975914:
                                                            return 't'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_particles > 62.5:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.14680564403533936:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 684.3125:
                                                if Q.log_sum_pt > 6.9065141677856445:
                                                    if Q.z_dr_0p1_0p2 > 0.019425611943006516:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1056.5009765625:
                                                    if Q.z_top50_slots > 0.9651383757591248:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4274246543645859:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top30 > 747.630859375:
                                                        if Q.max_dr > 0.2859419733285904:
                                                            if Q.sum_pt_top3 > 525.59375:
                                                                if Q.girth2_top5 > 0.005472529912367463:
                                                                    return 't'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top30 > 998.1826171875:
                                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.01147108431905508:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top20 > 85.29544448852539:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top20 > 91.37947082519531:
                                    return 't'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 59% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1012.11572265625:
                                if Q.tau32 > 0.5698172748088837:
                                    if Q.z_top50_slots > 0.9772111475467682:
                                        if Q.sum_pt_top3 > 548.0:
                                            if Q.z_dr_0p2_0p4 > 0.09834115579724312:
                                                if Q.C2 > 0.10777805745601654:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.009565642569214106:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.09731396287679672:
                                                if Q.log_sum_pt > 6.971316576004028:
                                                    if Q.tau32 > 0.7428041994571686:
                                                        if Q.z_dr_0p2_0p4 > 0.04980098269879818:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 1075.4739990234375:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.004994755145162344:
                                                        if Q.n_dr_0p2_0p4 > 6.5:
                                                            if Q.z_dr_0p2_0p4 > 0.06940650939941406:
                                                                if Q.girth2_top5 > 0.007944581098854542:
                                                                    return 't'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 0.001169555471278727:
                                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 39% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top40 > 986.44921875:
                                                                return 't'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0016772027011029422:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 337.125:
                                                                if Q.sum_pt > 1044.35888671875:
                                                                    return 't'   # 44% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.950751066207886:
                                            if Q.lam2 > 0.0034374836832284927:
                                                if Q.z_top50_slots > 0.9466476440429688:
                                                    if Q.girth2_top40 > 0.020885363221168518:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 146.9325408935547:
                                                if Q.sum_pt_top40 > 891.0390625:
                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.967111736536026:
                                                    if Q.max_dr > 0.3339293897151947:
                                                        if Q.tau32 > 0.7388154864311218:
                                                            if Q.z_dr_0p05_0p1 > 0.5353308618068695:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9449861347675323:
                                        if Q.mass > 100.95809936523438:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1048.4912109375:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 795.744140625:
                                    if Q.width > 0.009532171767205:
                                        if Q.log_sum_pt > 6.902179956436157:
                                            if Q.sum_pt_top3 > 577.21875:
                                                if Q.z_dr_0p1_0p2 > 0.13901455700397491:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.tau21 > 0.302078515291214:
                                                        if Q.z_top50_slots > 0.9389924705028534:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 158.33211517333984:
                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3195965588092804:
                                                            if Q.pt_9 > 22.859375:
                                                                return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top5 > 0.006048226961866021:
                                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.989774614572525:
                                                        if Q.girth2_top50 > 0.010005352552980185:
                                                            return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.029914631508290768:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 977.296630859375:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.003557326039299369:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 750.83203125:
                                        if Q.girth2_top50 > 0.015901469625532627:
                                            if Q.tau32 > 0.5872259736061096:
                                                if Q.z_dr_0p2_0p4 > 0.08442777022719383:
                                                    if Q.lam2 > 0.006950395414605737:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_top50_slots > 0.9639517962932587:
                                                            return 't'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 128.83795166015625:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9574316442012787:
                                                if Q.dr_7 > 0.0806679017841816:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.006776658119633794:
                                                    if Q.sum_pt_top50 > 720.09765625:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1035.702880859375:
                    if Q.n_particles > 59.5:
                        if Q.tau32 > 0.40873438119888306:
                            if Q.log_sum_pt > 6.967809200286865:
                                return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9847830832004547:
                                    if Q.z_dr_0p1_0p2 > 0.18924088776111603:
                                        return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0036412719637155533:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 480.40625:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.036224354058504105:
                                        if Q.z_top50_slots > 0.9713164269924164:
                                            if Q.lam2 > 0.003291841712780297:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.03409704007208347:
                                if Q.n_dr_0p2_0p4 > 17.5:
                                    if Q.mass > 176.0698013305664:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9852893054485321:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.989217758178711:
                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.030579392798244953:
                                        return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1076.89306640625:
                            if Q.tau32 > 0.4562455862760544:
                                if Q.pt_9 > 17.3203125:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 47.5:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 1151.6749267578125:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.031788015738129616:
                                    return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 58% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top3 > 545.9375:
                                if Q.z_dr_0p1_0p2 > 0.15476097911596298:
                                    return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 102.29251098632812:
                                        return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top15 > 0.006287503754720092:
                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.033628473058342934:
                                    if Q.planar_flow > 0.2669486403465271:
                                        return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 104.42501449584961:
                                            return 'g'   # 46% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.13402368873357773:
                                        return 't'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1059.9693603515625:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9983052313327789:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                else:
                    if Q.z_top50_slots > 0.9715259969234467:
                        if Q.log_sum_pt > 6.895995140075684:
                            if Q.sum_pt_top3 > 426.71875:
                                if Q.mass_over_sum_pt_sq > 0.009441971313208342:
                                    if Q.z_dr_0p1_0p2 > 0.17021885514259338:
                                        if Q.n_dr_0p1_0p2 > 20.5:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.6400137841701508:
                                            if Q.girth2_top40 > 0.011334936134517193:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.17038896679878235:
                                                if Q.tau21 > 0.3792213201522827:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 12.5:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 100.32259750366211:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.028265709057450294:
                                        if Q.mass_top20 > 79.11470794677734:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03294628672301769:
                                                return 't'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 42% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.030681317672133446:
                                    if Q.mass_over_sum_pt_sq > 0.009459440130740404:
                                        if Q.max_dr > 0.35677748918533325:
                                            if Q.e2 > 0.032989636063575745:
                                                if Q.sum_pt > 1003.5352783203125:
                                                    if Q.z_top50_slots > 0.9755996465682983:
                                                        return 't'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.98064985871315:
                                                    if Q.z_dr_0p2_0p4 > 0.04590239003300667:
                                                        if Q.lam2 > 0.0016317182453349233:
                                                            return 't'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 7.5:
                                                if Q.z_dr_0p2_0p4 > 0.07281417399644852:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 8.5:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 63.5:
                                        if Q.log_sum_pt > 6.919607162475586:
                                            if Q.tau32 > 0.6083576679229736:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.035923805087804794:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.813920259475708:
                                                return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top20 > 46.32996940612793:
                                                    if Q.z_dr_0p1_0p2 > 0.1428820565342903:
                                                        return 't'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top50 > 972.57421875:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.05904131010174751:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 43% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023077521473169327:
                                if Q.sum_pt_top40 > 763.94921875:
                                    if Q.sum_pt_top30 > 936.2314453125:
                                        if Q.z_dr_0_0p05 > 0.8016902208328247:
                                            if Q.log_sum_pt > 6.878774881362915:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0_0p05 > 0.860071450471878:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.028129123151302338:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 977.8677978515625:
                                                if Q.sum_pt_top2 > 292.09375:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9764237701892853:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top40 > 0.008328303694725037:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.013524547684937716:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.n_particles > 63.5:
                                    if Q.girth2_top50 > 0.01074919244274497:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top30 > 909.283203125:
                                        return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.020483952946960926:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 54% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.03076271992176771:
                            if Q.log_sum_pt > 6.900224208831787:
                                if Q.C2 > 0.13360699266195297:
                                    if Q.z_top50_slots > 0.9475379288196564:
                                        if Q.mass_top50 > 141.57981872558594:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.0358443483710289:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.01680684182792902:
                                        if Q.tau32 > 0.6224900484085083:
                                            if Q.z_top50_slots > 0.9652422070503235:
                                                if Q.dr_0 > 0.08672529086470604:
                                                    return 'q'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3087352216243744:
                                                if Q.e2 > 0.033825598657131195:
                                                    return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 766.4453125:
                                    if Q.z_top50_slots > 0.9544270932674408:
                                        if Q.log_sum_pt > 6.872648239135742:
                                            if Q.C2 > 0.12067579478025436:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9637077152729034:
                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.10556676983833313:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.6237609386444092:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 868.955078125:
                                                return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.017650192603468895:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.5418263077735901:
                                            if Q.log_sum_pt > 6.858627557754517:
                                                if Q.mass_over_sum_pt_sq > 0.025561219081282616:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9379500150680542:
                                                    if Q.girth2_top3 > 0.011180450208485126:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.12217145040631294:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.017570498399436474:
                                                        if Q.lam1 > 0.021130813285708427:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.12320207059383392:
                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top50 > 0.01419768063351512:
                                if Q.z_top50_slots > 0.9561841785907745:
                                    if Q.log_sum_pt > 6.885793685913086:
                                        if Q.tau32 > 0.5614609122276306:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.746830701828003:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6709032654762268:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.86456036567688:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 697.09375:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9656890332698822:
                                    if Q.log_sum_pt > 6.872551441192627:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.08183551207184792:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.028555065393447876:
                                        if Q.log_sum_pt > 6.859707593917847:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9615895748138428:
                                                return 't'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.log_sum_pt > 6.977758884429932:
                    if Q.z_dr_0p2_0p4 > 0.0032225524773821235:
                        if Q.sum_pt > 1115.7109375:
                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.010981891304254532:
                                if Q.e2 > 0.023158361203968525:
                                    if Q.mass > 95.49390029907227:
                                        return 'g'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.6982638537883759:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 96.61059188842773:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 82.01659774780273:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 95.19766616821289:
                            return 'g'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.023144333623349667:
                                if Q.mass_top50 > 83.64300918579102:
                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 40% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 77% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.0233558164909482:
                        if Q.log_sum_pt > 6.885926008224487:
                            if Q.mass > 94.4185905456543:
                                if Q.e2 > 0.030892335809767246:
                                    if Q.mass > 97.6885871887207:
                                        return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 95.7932243347168:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top50 > 89.97330856323242:
                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 76.39021682739258:
                                    if Q.width > 0.008678868878632784:
                                        if Q.mass_top40 > 81.09676742553711:
                                            if Q.max_dr > 0.367232084274292:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3624926209449768:
                                            if Q.sum_pt > 995.4327392578125:
                                                if Q.girth2_top40 > 0.0063880193047225475:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 91.10406494140625:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.3428032100200653:
                                        if Q.z_top50_slots > 0.9750592112541199:
                                            if Q.tau32 > 0.8220914304256439:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9755214750766754:
                                if Q.n_dr_0p2_0p4 > 4.5:
                                    if Q.e2 > 0.02609643805772066:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 967.534423828125:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.30198322236537933:
                                    if Q.e2 > 0.028655909933149815:
                                        return 't'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.864378213882446:
                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.32099248468875885:
                            if Q.z_top50_slots > 0.9797081351280212:
                                if Q.girth > 0.05488033406436443:
                                    if Q.mass_over_sum_pt_sq > 0.008035881910473108:
                                        if Q.mass > 94.18104934692383:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 977.98388671875:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top50 > 0.006518556736409664:
                                            if Q.mass > 91.25447082519531:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 977.89453125:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 36% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 522.46875:
                                        if Q.log_sum_pt > 6.9331629276275635:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.4632102847099304:
                                    return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 92.0693588256836:
                                return 'g'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.891602993011475:
                                    if Q.girth2_top40 > 0.005993568105623126:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.9692962467670441:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 81% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.28694534301758:
                    if Q.girth2_top30 > 0.007852394133806229:
                        if Q.mass > 104.26414489746094:
                            if Q.n_dr_0p2_0p4 > 4.5:
                                return 'g'   # 72% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top20 > 0.007715016137808561:
                                return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.058374738320708275:
                                    return 'Z'   # 41% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.n_particles > 34.5:
                            if Q.log_sum_pt > 7.039498329162598:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.03170427680015564:
                                    if Q.mass > 100.58744812011719:
                                        return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_9 > 18.1875:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 57% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 6.5:
                                return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top50 > 965.404052734375:
                        if Q.girth2_top20 > 0.0036466537276282907:
                            if Q.mass > 86.01634216308594:
                                if Q.girth2_top20 > 0.004849403165280819:
                                    if Q.sum_pt_top50 > 978.467041015625:
                                        if Q.mass > 86.6754379272461:
                                            if Q.mass > 97.0673828125:
                                                if Q.girth2_top20 > 0.006407557055354118:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9997647106647491:
                                                        if Q.e2 > 0.025679669342935085:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 37% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt_sq > 0.00825112173333764:
                                                            return 'Z'   # 34% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.09368055686354637:
                                                    if Q.D2 > 3.4284331798553467:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top20 > 0.007395275868475437:
                                                            return 'Z'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top3 > 0.005073087522760034:
                                                                return 't'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1049.529541015625:
                                                if Q.max_dr > 0.3410409241914749:
                                                    if Q.D2 > 1.5948597192764282:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.6436814069747925:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.z_dr_0p2_0p4 > 0.05478723160922527:
                                                if Q.D2 > 3.6941423416137695:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3603324145078659:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 94.36486434936523:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            if Q.girth2_top50 > 0.007983146701008081:
                                                if Q.C2 > 0.08294704556465149:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 6.124965190887451:
                                            if Q.log_sum_pt > 6.969897985458374:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 978.12109375:
                                                if Q.girth2_top30 > 0.005294312024489045:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 53.5:
                                                        if Q.tau32 > 0.7563298940658569:
                                                            if Q.sum_pt_top30 > 1025.0859375:
                                                                if Q.n_dr_0p2_0p4 > 4.5:
                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 86.79161834716797:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top5 > 0.002621783409267664:
                                                                return 'W'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 43% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.046930862590670586:
                                    if Q.width > 0.006726397201418877:
                                        if Q.sum_pt_top50 > 981.420654296875:
                                            if Q.max_dr > 0.5554502606391907:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3489864021539688:
                                                if Q.z_dr_0p2_0p4 > 0.056787049397826195:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.531818151473999:
                                            if Q.girth2_top40 > 0.005413510603830218:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.337388351559639:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.007090584607794881:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            if Q.sum_pt > 1000.9705810546875:
                                                if Q.e2 > 0.03707317262887955:
                                                    return 'W'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.004077777033671737:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.2924395203590393:
                                            if Q.girth2 > 0.006835640873759985:
                                                if Q.girth2_top20 > 0.005584938684478402:
                                                    if Q.max_dr > 0.3612336963415146:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 85.21842193603516:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.006292975274845958:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 85.3256607055664:
                                                    if Q.max_dr > 0.2392696738243103:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1081.9466552734375:
                                if Q.n_dr_0p2_0p4 > 6.5:
                                    if Q.planar_flow > 0.28416426479816437:
                                        if Q.e2 > 0.016867412254214287:
                                            if Q.max_dr > 0.340421199798584:
                                                if Q.n_particles > 48.5:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 91.72719192504883:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 93.30990982055664:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top15 > 0.001410479366313666:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 85.86057662963867:
                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.998532772064209:
                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 5.7667248249053955:
                                    if Q.lam2 > 0.0010734338429756463:
                                        if Q.sum_pt > 1035.060791015625:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 40% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 93.60990905761719:
                                        return 'g'   # 43% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.32737329602241516:
                            if Q.mass_top20 > 81.46444702148438:
                                if Q.sum_pt_top40 > 953.4752197265625:
                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.02306303381919861:
                                    if Q.sum_pt_top40 > 956.9605712890625:
                                        if Q.z_dr_0p2_0p4 > 0.06619098410010338:
                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 40% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 937.155029296875:
                                return 'Z'   # 92% of the training jets here get this class from the formula
                            else:
                                return 't'   # 65% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.sum_pt_top50 > 963.5062255859375:
                if Q.girth2_top20 > 0.0029040308436378837:
                    if Q.width > 0.0069456028286367655:
                        if Q.D2 > 2.232545256614685:
                            if Q.mass > 83.11318588256836:
                                if Q.sum_pt_top40 > 976.4013671875:
                                    if Q.LHA > 0.23388820886611938:
                                        if Q.sum_pt_top30 > 966.4736328125:
                                            return 'W'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 14.5:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 67% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 3.2613638639450073:
                                    if Q.girth2_top30 > 0.006613131612539291:
                                        return 'Z'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0010884598013944924:
                                        return 't'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                        else:
                            if Q.mass_over_sum_pt_sq > 0.007178036263212562:
                                if Q.max_dr > 0.3239309638738632:
                                    if Q.sum_pt_top50 > 980.8868408203125:
                                        return 't'   # 42% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.06365092471241951:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 36% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 83.27912902832031:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 39% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.3102119266986847:
                                    if Q.sum_pt_top30 > 970.353759765625:
                                        return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 994.417236328125:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.C2 > 0.03856109268963337:
                                                    return 'W'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 83.31779479980469:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.D2 > 5.152032852172852:
                            if Q.mass > 82.41828155517578:
                                if Q.girth2_top40 > 0.005535159958526492:
                                    if Q.n_particles > 57.5:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 13.5:
                                    if Q.mass_top50 > 74.50794982910156:
                                        if Q.sum_pt_top50 > 995.3673095703125:
                                            if Q.girth2_top30 > 0.0043028127402067184:
                                                if Q.mass > 81.56261444091797:
                                                    return 'W'   # 35% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 76.88652801513672:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0p2_0p4 > 0.04697451926767826:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 43% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.3225919306278229:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.4752044230699539:
                                        if Q.mass_top50 > 73.4464111328125:
                                            if Q.girth2_top30 > 0.004204391269013286:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 10.5:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 83.61030197143555:
                                if Q.e2 > 0.02602249290794134:
                                    if Q.D2 > 1.7933131456375122:
                                        if Q.lam2 > 0.0017232351237908006:
                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.22257596254348755:
                                                if Q.mass_top40 > 76.11443710327148:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3713137358427048:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.006777643924579024:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 72.58365249633789:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.0054203474428504705:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 82.29148483276367:
                                        if Q.D2 > 2.780056118965149:
                                            if Q.girth2_top40 > 0.006235747132450342:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top30 > 83.87688446044922:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 84.16607666015625:
                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 3.6951321363449097:
                                                            return 'Z'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top40 > 0.006672141375020146:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 72.2908935546875:
                                    if Q.sum_pt_top50 > 979.304931640625:
                                        if Q.n_particles > 62.5:
                                            if Q.girth2_top20 > 0.004014679696410894:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 81.2259292602539:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 16.5:
                                                if Q.mass_over_sum_pt > 0.0821087546646595:
                                                    return 'q'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top50 > 75.48627090454102:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.08213067427277565:
                                                    if Q.girth > 0.04595556855201721:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 82.68693161010742:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 5.5:
                                            if Q.D2 > 1.280935525894165:
                                                if Q.n_dr_0p2_0p4 > 14.5:
                                                    if Q.D2 > 3.555156111717224:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 82.24678421020508:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 10.5:
                                            if Q.sum_pt_top50 > 985.296142578125:
                                                if Q.z_top20_slots > 0.9208500385284424:
                                                    if Q.planar_flow > 0.36206191778182983:
                                                        if Q.mass_top40 > 70.67879486083984:
                                                            if Q.pt_9 > 14.32421875:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1064.6812744140625:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.050698962062597275:
                                                    return 't'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.99307656288147:
                                                if Q.z_dr_0p2_0p4 > 0.004417738411575556:
                                                    if Q.n_particles > 62.5:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.36309507489204407:
                                                            if Q.girth2_top30 > 0.0039006866281852126:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 987.7587890625:
                                                    if Q.mass_top50 > 71.31953811645508:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.31856945157051086:
                                                            if Q.C2 > 0.06405546143651009:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.005550606641918421:
                                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.z_dr_0p2_0p4 > 0.033768076449632645:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top2 > 311.5:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                else:
                    if Q.planar_flow > 0.420708030462265:
                        if Q.sum_pt > 1059.768310546875:
                            if Q.e2 > 0.017443770542740822:
                                if Q.n_particles > 52.5:
                                    if Q.z_dr_0p2_0p4 > 0.001369442034047097:
                                        if Q.n_particles > 57.5:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.04352623037993908:
                                                if Q.tau32 > 0.7704562842845917:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 70.25203704833984:
                                        if Q.mass > 82.58760070800781:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top40 > 0.004301339387893677:
                                            return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                if Q.tau32 > 0.6722978055477142:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.n_particles > 39.5:
                                    if Q.girth2_top30 > 0.003948149969801307:
                                        if Q.mass > 80.38759231567383:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.10246973857283592:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 42% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 5.355832576751709:
                                if Q.n_particles > 57.5:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 75.0108642578125:
                                        if Q.n_dr_0p2_0p4 > 14.5:
                                            if Q.tau32 > 0.8615899085998535:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 42% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 11.5:
                                            if Q.pt_9 > 29.1953125:
                                                return 'g'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 81.97967910766602:
                                    if Q.mass_top50 > 79.18371200561523:
                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 73.44771575927734:
                                        if Q.n_particles > 58.5:
                                            if Q.dr_0 > 0.02722529135644436:
                                                return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0_0p05 > 0.7895027101039886:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 12.5:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 1001.45068359375:
                                                return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 82.24787139892578:
                            if Q.mass_top20 > 65.7188720703125:
                                return 'W'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top50 > 0.006209982326254249:
                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 80.04505920410156:
                                        return 'W'   # 47% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.014024483971297741:
                                return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1067.60009765625:
                                    if Q.mass_top30 > 71.24090194702148:
                                        return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0_0p05 > 0.9152258634567261:
                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 81% of the training jets here get this class from the formula
            else:
                if Q.mass_over_sum_pt > 0.08445882424712181:
                    if Q.z_top50_slots > 0.9785110950469971:
                        if Q.e2 > 0.0209367573261261:
                            if Q.sum_pt_top40 > 717.513671875:
                                if Q.max_dr > 0.2615366578102112:
                                    if Q.z_dr_0_0p05 > 0.8262901902198792:
                                        if Q.mass_top50 > 76.10275268554688:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.00757516291923821:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.000643600826151669:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 51% of the training jets here get this class from the formula
                            else:
                                if Q.dr_7 > 0.07492806762456894:
                                    return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top50 > 41.5:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9942107796669006:
                                return 't'   # 54% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.24362901598215103:
                            if Q.e2 > 0.028652017004787922:
                                if Q.sum_pt_top40 > 701.240234375:
                                    return 't'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 63% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.3130234032869339:
                        if Q.z_dr_0_0p05 > 0.824813038110733:
                            if Q.mass_top20 > 69.68191146850586:
                                if Q.n_dr_0p2_0p4 > 11.5:
                                    return 'q'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9991076588630676:
                                    if Q.girth2 > 0.006713027134537697:
                                        return 't'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 10.5:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 73.87197875976562:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.0021898975828662515:
                                if Q.z_top50_slots > 0.9827152788639069:
                                    if Q.sum_pt_top30 > 949.615234375:
                                        return 't'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.5138185918331146:
                                            return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top15 > 0.005058080889284611:
                                        return 'W'   # 36% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top50 > 953.8070068359375:
                                    return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 74.64873886108398:
                                        return 'W'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0p2_0p4 > 1.5:
                            if Q.log_sum_pt > 6.849742889404297:
                                if Q.z_top50_slots > 0.9697283208370209:
                                    return 'W'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 42% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 370.625:
                                    return 'W'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 93% of the training jets here get this class from the formula
        else:
            if Q.n_real_top50 > 45.5:
                if Q.log_sum_pt > 6.952676296234131:
                    if Q.girth2_top30 > 0.003544405219145119:
                        if Q.e2 > 0.021714946255087852:
                            if Q.z_dr_0p2_0p4 > 0.0036606525536626577:
                                if Q.tau32 > 0.7239186465740204:
                                    if Q.max_dr > 0.30845803022384644:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1078.1912841796875:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 76% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 89% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 60.5:
                                if Q.z_dr_0p2_0p4 > 0.0023701980244368315:
                                    if Q.girth2_top40 > 0.004454102599993348:
                                        if Q.mass > 79.68721771240234:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.019592588767409325:
                                                return 'W'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9511592984199524:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 501.0625:
                                    return 'q'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.03988838940858841:
                                        return 'W'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1066.0294189453125:
                            if Q.e2 > 0.019046797417104244:
                                if Q.n_particles > 50.5:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 51% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 51.5:
                                if Q.mass_top10 > 30.136130332946777:
                                    if Q.sum_pt_top2 > 409.0625:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.007393570384010673:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 56.5:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.03408958949148655:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top15 > 0.0008073339704424143:
                                    if Q.pt_9 > 26.609375:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 668.125:
                                        return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0002943663712358102:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 59% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top20 > 44.60230255126953:
                            if Q.log_sum_pt > 6.882007598876953:
                                if Q.girth2_top30 > 0.004073749762028456:
                                    if Q.mass > 80.70491027832031:
                                        if Q.tau32 > 0.6248345077037811:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.904637813568115:
                                            if Q.girth2_top40 > 0.004565595183521509:
                                                if Q.z_top20_slots > 0.8734774589538574:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top30 > 0.004751213360577822:
                                                if Q.max_dr > 0.3179499953985214:
                                                    if Q.sum_pt_top50 > 961.63427734375:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9747860431671143:
                                                    if Q.girth2 > 0.0057690623216331005:
                                                        return 'g'   # 38% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.2958403080701828:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 73.87557220458984:
                                        if Q.max_dr > 0.35314059257507324:
                                            if Q.n_particles > 61.5:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9791167080402374:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.934020280838013:
                                            if Q.tau32 > 0.7839995920658112:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 45% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.00034152576699852943:
                                                if Q.z_top50_slots > 0.9775887429714203:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9784954786300659:
                                    if Q.e2 > 0.02141141425818205:
                                        if Q.log_sum_pt > 6.85346531867981:
                                            if Q.n_particles > 62.5:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.048425570130348206:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 65.01419830322266:
                                                    if Q.z_top50_slots > 0.981303334236145:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.1065581776201725:
                                        return 't'   # 40% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 63.5:
                                if Q.mass_top20 > 40.3996467590332:
                                    if Q.log_sum_pt > 6.886542081832886:
                                        if Q.max_dr > 0.28844960033893585:
                                            if Q.z_top50_slots > 0.9766597151756287:
                                                if Q.dr_0 > 0.03446330316364765:
                                                    if Q.sum_pt > 1020.7421875:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_9 > 23.4296875:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 0.0005326898535713553:
                                                            return 'q'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top40 > 0.004200497176498175:
                                                if Q.z_top50_slots > 0.9532219171524048:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top20 > 32.47174835205078:
                                        if Q.sum_pt_top50 > 960.767578125:
                                            if Q.log_sum_pt > 6.922009706497192:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 67.97091293334961:
                                                    if Q.max_dr > 0.2982388734817505:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.8084443509578705:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 0.0007179296226240695:
                                                            if Q.z_top50_slots > 0.9753198325634003:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top3 > 0.0005921579140704125:
                                    if Q.sum_pt_top50 > 954.1904296875:
                                        if Q.mass_top50 > 70.33358383178711:
                                            if Q.girth2_top15 > 0.0011216167476959527:
                                                if Q.mass_top50 > 73.43189239501953:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1027.3017578125:
                                                if Q.dr_0 > 0.03588815964758396:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 304.140625:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 7.5:
                                                    if Q.tau32 > 0.7203855216503143:
                                                        if Q.sum_pt_top2 > 250.8125:
                                                            return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.03461674228310585:
                                                                return 'q'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top2 > 402.8125:
                                        if Q.mass_top20 > 28.67210865020752:
                                            if Q.z_dr_0p2_0p4 > 0.022532706148922443:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 0.00033750952570699155:
                                            if Q.sum_pt > 967.7244873046875:
                                                if Q.log_sum_pt > 6.925526142120361:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.sum_pt_top3 > 415.875:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top40 > 942.18017578125:
                            if Q.girth2_top15 > 0.0005189494404476136:
                                if Q.mass_top50 > 70.28935623168945:
                                    if Q.z_top20_slots > 0.9003070592880249:
                                        if Q.planar_flow > 0.402377650141716:
                                            return 'q'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 9.5:
                                            if Q.mass_top50 > 72.63249588012695:
                                                if Q.log_sum_pt > 6.901909589767456:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.897817611694336:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 52.5:
                                        if Q.girth2_top3 > 0.0004977072239853442:
                                            if Q.girth2_top20 > 0.0036106633488088846:
                                                if Q.sum_pt > 996.6475830078125:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.939455032348633:
                                                    if Q.girth2_top3 > 0.0008349174750037491:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_9 > 30.1875:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top20 > 0.0007761297456454486:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0006527435325551778:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.932972431182861:
                                                if Q.pt_9 > 23.75:
                                                    if Q.max_dr > 0.32862523198127747:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.0002609837974887341:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.885580778121948:
                                                    if Q.pt_9 > 26.828125:
                                                        if Q.n_dr_0p2_0p4 > 5.5:
                                                            if Q.log_sum_pt > 6.921661615371704:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_7 > 0.015748457051813602:
                                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.0034916953882202506:
                                            if Q.z_dr_0p2_0p4 > 0.002516724169254303:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.00017506929725641385:
                                                if Q.sum_pt > 1028.607666015625:
                                                    if Q.mass_top10 > 18.62724018096924:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_9 > 31.1328125:
                                                            if Q.lam2 > 0.0006324349378701299:
                                                                return 'g'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.870069742202759:
                                                        return 'q'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.36871348321437836:
                                                            return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 448.4375:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 1016.972900390625:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0p2_0p4 > 7.5:
                                                            if Q.mass_top20 > 37.20216178894043:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9997548162937164:
                                    if Q.girth2_top5 > 9.216255421051756e-05:
                                        if Q.log_sum_pt > 6.938964605331421:
                                            if Q.lam2 > 0.00036553242534864694:
                                                if Q.sum_pt_top3 > 526.78125:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0004372004623292014:
                                                if Q.pt_9 > 30.59375:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.tau32 > 0.7941755652427673:
                                                            if Q.max_dr > 0.3539870083332062:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top40 > 956.80029296875:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 478.59375:
                                            if Q.log_sum_pt > 6.932837247848511:
                                                if Q.sum_pt_top2 > 498.5:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top50 > 35.56272888183594:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 1018.58544921875:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 974.081787109375:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.sum_pt_top2 > 404.1875:
                                            if Q.sum_pt_top50 > 1031.8475341796875:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 8.274154970422387e-05:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6895316243171692:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1024.84912109375:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 8.289444667752832e-05:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top10 > 17.768568992614746:
                                if Q.mass_top50 > 66.72921371459961:
                                    if Q.girth > 0.048848073929548264:
                                        if Q.sum_pt_top30 > 678.044921875:
                                            if Q.log_sum_pt > 6.855309247970581:
                                                return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top2 > 418.1875:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.sum_pt_top2 > 323.96875:
                                            if Q.girth2_top3 > 0.0005306575039867312:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 499.9375:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.9997071921825409:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.36541178822517395:
                                                if Q.tau32 > 0.6330226361751556:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 47% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 894.46240234375:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 4.5:
                                    if Q.sum_pt_top3 > 529.03125:
                                        if Q.mass_top30 > 46.029048919677734:
                                            return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau32 > 0.6742591857910156:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 752.3212890625:
                                                if Q.girth2_top3 > 0.00037446917849592865:
                                                    return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.018708910793066025:
                                        return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0035611633211374283:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 54% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1113.7164306640625:
                    if Q.n_real_top50 > 33.5:
                        if Q.n_particles > 36.5:
                            if Q.girth2_top15 > 0.0005365803081076592:
                                if Q.log_sum_pt > 7.095804452896118:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top50 > 40.5:
                                        if Q.n_dr_0p2_0p4 > 2.5:
                                            if Q.sum_pt_top2 > 537.875:
                                                if Q.girth2_top3 > 0.0002297127721249126:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_9 > 25.71875:
                                            if Q.z_dr_0p2_0p4 > 0.004242592258378863:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 7.572451795567758e-05:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 1204.91357421875:
                                if Q.girth2_top15 > 0.000694269489031285:
                                    if Q.tau32 > 0.8513505160808563:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 493.375:
                                    if Q.tau32 > 0.8635483980178833:
                                        if Q.mass_top10 > 11.47141981124878:
                                            return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.6369909048080444:
                                        if Q.max_dr > 0.3379467874765396:
                                            if Q.planar_flow > 0.5438899099826813:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 48% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 72% of the training jets here get this class from the formula
                    else:
                        if Q.n_real_top40 > 31.5:
                            if Q.log_sum_pt > 7.102043390274048:
                                if Q.girth2_top5 > 4.7401830670423806e-05:
                                    return 'q'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 440.3125:
                                    return 'q'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 3.3334821462631226:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top50 > 65.55248641967773:
                                if Q.lam2 > 0.00022895284200785682:
                                    return 'q'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1362.8680419921875:
                                    if Q.n_real_top40 > 27.5:
                                        if Q.pt_9 > 31.1171875:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top40 > 29.5:
                                        if Q.pt_9 > 28.0703125:
                                            if Q.C2 > 0.06207180209457874:
                                                if Q.log_sum_pt > 7.06335711479187:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 99% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 39.5:
                        if Q.log_sum_pt > 6.965435743331909:
                            if Q.girth2_top15 > 0.0004007776005892083:
                                if Q.sum_pt_top2 > 454.21875:
                                    if Q.girth2_top20 > 0.001271193556021899:
                                        return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 1079.1146240234375:
                                            if Q.tau32 > 0.7938719391822815:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.tau32 > 0.6640720367431641:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 0.00034908992529381067:
                                            if Q.mass_over_sum_pt > 0.05917138233780861:
                                                return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 516.59375:
                                    if Q.log_sum_pt > 6.9779040813446045:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 8.590668949182145e-05:
                                            return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 933.7120361328125:
                                if Q.girth2_top15 > 0.0001860713673522696:
                                    if Q.lam1 > 0.003941758535802364:
                                        if Q.z_dr_0p2_0p4 > 0.004666745895519853:
                                            if Q.sum_pt > 995.2662353515625:
                                                if Q.z_top20_slots > 0.9238611459732056:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.05362290143966675:
                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.889952659606934:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.9488136768341064:
                                            if Q.girth2_top20 > 0.0007763239846099168:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00048149775830097497:
                                                    if Q.sum_pt_top2 > 378.25:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 959.5810546875:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.32345838844776154:
                                                    if Q.girth2_top5 > 0.00010050075070466846:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 531.0625:
                                                            return 'q'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1036.072021484375:
                                        if Q.lam2 > 0.00047704807366244495:
                                            if Q.sum_pt_top2 > 467.125:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_real_top50 > 43.5:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top30 > 954.65380859375:
                                            if Q.lam2 > 0.0004188886523479596:
                                                if Q.sum_pt_top2 > 337.4375:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_particles > 41.5:
                                                        if Q.girth2_top5 > 8.83968677953817e-05:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top50 > 0.0012224590755067766:
                                                if Q.girth2_top5 > 8.999487181426957e-05:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 334.0625:
                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 9.292800903320312:
                                    if Q.sum_pt_top30 > 739.9990234375:
                                        if Q.girth > 0.05672936141490936:
                                            if Q.pt_9 > 27.1875:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 296.3125:
                                                if Q.mass_top20 > 22.976670265197754:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00065817084396258:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p2_0p4 > 5.5:
                                                    if Q.girth2_top3 > 0.0003196910402039066:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top10 > 16.16418170928955:
                                            if Q.tau32 > 0.8684881031513214:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top2 > 329.28125:
                                        if Q.mass_top20 > 28.220582962036133:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0006664270767942071:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005035547656007111:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0001133558434958104:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top30 > 64.39311981201172:
                            if Q.lam2 > 0.00038030183350201696:
                                if Q.n_dr_0p2_0p4 > 9.5:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.005125750787556171:
                                        if Q.z_dr_0p2_0p4 > 0.028689391911029816:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top30 > 66.96259689331055:
                                            if Q.C2 > 0.06207128241658211:
                                                if Q.sum_pt > 989.0780029296875:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 972.0711669921875:
                                    if Q.n_dr_0p2_0p4 > 11.5:
                                        return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.2739337980747223:
                                            if Q.tau21 > 0.30525198578834534:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.05259039252996445:
                                        if Q.z_dr_0p2_0p4 > 0.0007729747449047863:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.20894677937030792:
                                if Q.sum_pt_top30 > 695.1693115234375:
                                    if Q.n_particles > 36.5:
                                        if Q.sum_pt_top50 > 1075.51806640625:
                                            if Q.sum_pt_top2 > 499.3125:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.0001560117889312096:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0002455991198075935:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 806.8006591796875:
                                                if Q.girth > 0.058453649282455444:
                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top15 > 0.0004430692351888865:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0032906820997595787:
                                            if Q.lam2 > 0.0002789285354083404:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.401507705450058:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top30_slots > 0.9989815354347229:
                                        return 'q'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top2 > 342.6875:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 3.5:
                                                if Q.mass > 36.52252388000488:
                                                    if Q.z_top30_slots > 0.9957643151283264:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top30 > 52.21282768249512:
                                    if Q.sum_pt_top40 > 990.37841796875:
                                        if Q.D2 > 3.035421133041382:
                                            if Q.n_dr_0p2_0p4 > 8.5:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top30 > 62.45152473449707:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.24664609134197235:
                                            return 't'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 86% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
