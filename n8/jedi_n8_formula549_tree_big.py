"""JEDI-linear jet tagger, 8 particles, 3 features: the simplest formula at the network's accuracy (from the 931-term tuned formula): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 64.80% (the formula: 65.80%); same class as the formula for 90.90% of jets.  2758 leaves, depth 23.
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
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_7=z[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_2=phi[2],
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
        n_pt_above_10=sum(1 for x in pt if x > 10),
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
    if Q.girth2 > 0.009139599744230509:
        if Q.log_sum_pt > 6.777339696884155:
            if Q.pt_7 > 38.546875:
                if Q.tau21 > 0.34063927829265594:
                    if Q.e2_sq > 0.010796689428389072:
                        return 't'   # 80% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 68% of the training jets here get this class from the formula
                else:
                    if Q.log_sum_pt > 6.821635484695435:
                        return 'g'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.009309391025453806:
                            return 'g'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.lam1 > 0.013945923186838627:
                                return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                return 't'   # 67% of the training jets here get this class from the formula
            else:
                if Q.log_sum_pt > 6.8576741218566895:
                    if Q.pt_7 > 16.8046875:
                        if Q.tau32 > 0.2469859942793846:
                            if Q.mass > 109.12617111206055:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.C2 > 0.014513354282826185:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1007.5703125:
                                return 'g'   # 48% of the training jets here get this class from the formula
                            else:
                                return 't'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 11.359375:
                            if Q.mass > 121.0008430480957:
                                return 'g'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 56% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 88% of the training jets here get this class from the formula
                else:
                    if Q.lam1 > 0.021876834332942963:
                        return 'g'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.03288239240646362:
                            if Q.z_dr_0p05_0p1 > 0.8107420206069946:
                                if Q.z_7 > 0.0293041979894042:
                                    if Q.dr01 > 0.006034574471414089:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 48% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.27325429022312164:
                                        if Q.phi_1 > -0.02082061767578125:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                return 't'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.pt_6 > 24.5703125:
                                return 't'   # 39% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 93% of the training jets here get this class from the formula
        else:
            if Q.width > 0.009906836785376072:
                if Q.mass > 46.41265869140625:
                    if Q.lam1 > 0.03249704651534557:
                        if Q.tau21 > 0.1414710283279419:
                            if Q.centroid_offset > 0.09908037632703781:
                                if Q.tau32 > 0.44849734008312225:
                                    return 'g'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.5001501142978668:
                                        if Q.pt_7 > 29.8671875:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 132.36934661865234:
                                    if Q.pt_7 > 37.953125:
                                        if Q.lam1 > 0.03918827511370182:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 216.34375:
                                        if Q.tau32 > 0.43805329501628876:
                                            if Q.planar_flow > 0.07270998507738113:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.39896124601364136:
                                                    return 'g'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.27515555918216705:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 40.015625:
                                if Q.mass > 98.37372207641602:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.08775456622242928:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.995910108089447:
                                    if Q.z_7 > 0.05191278085112572:
                                        if Q.LHA > 0.4773465096950531:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 246.109375:
                                                if Q.z_dr_0p1_0p2 > 0.6468873918056488:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 127.18372344970703:
                                            return 'q'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.040564367547631264:
                                        if Q.planar_flow > 0.05292041040956974:
                                            if Q.mean_eta > -0.020831365138292313:
                                                if Q.pt_dispersion > 0.37691211700439453:
                                                    return 't'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.05313125438988209:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 667.65625:
                                            if Q.pt_7 > 31.9453125:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.m012 > 25.713601112365723:
                                                if Q.planar_flow > 0.05210020951926708:
                                                    return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.036757923662662506:
                                                        if Q.pt_7 > 34.9375:
                                                            return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.48717930912971497:
                                                    if Q.D2 > 0.27824366092681885:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.lam2 > 0.0009297120850533247:
                            if Q.width > 0.011184143368154764:
                                if Q.mass > 51.26698112487793:
                                    return 't'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02829443383961916:
                                        return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.4385666698217392:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.3845970332622528:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.05052926205098629:
                                    if Q.width > 0.010424891486763954:
                                        if Q.max_pair_mass > 14.825917720794678:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.5825295150279999:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.07338108122348785:
                                        if Q.tau21 > 0.270881712436676:
                                            return 'g'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 47.546875:
                                if Q.mass > 113.01369857788086:
                                    if Q.tau32 > 0.28740641474723816:
                                        if Q.mass > 118.51958847045898:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt1_dr01 > 34.33860778808594:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 0.5075474381446838:
                                        if Q.e2 > 0.05422535538673401:
                                            if Q.C2 > 0.06854630634188652:
                                                return 't'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.048124438151717186:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 35.40780067443848:
                                                        if Q.tau32 > 0.5767937898635864:
                                                            return 't'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 680.421875:
                                                                if Q.girth2_top3 > 0.014951827935874462:
                                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.598402500152588:
                                                if Q.dr_0 > 0.04414417780935764:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.038592178374528885:
                                                    if Q.lam1 > 0.013040535617619753:
                                                        if Q.lam2 > 0.0003886400518240407:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.08505590632557869:
                                                                return 't'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 56.734375:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.016301589086651802:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0271706935018301:
                                            if Q.sum_pt_top3 > 298.84375:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.705409526824951:
                                                if Q.D2 > 0.29512283205986023:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01549146231263876:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 100.21984481811523:
                                                    if Q.dr01 > 0.025424519553780556:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.07485617324709892:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.041263069957494736:
                                                        if Q.mass > 89.87008285522461:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.m012 > 22.774517059326172:
                                    if Q.mass > 128.03314971923828:
                                        if Q.tau32 > 0.2807425707578659:
                                            if Q.pt_7 > 32.5625:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 134.4178466796875:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 5.941486120223999:
                                            if Q.z_5 > 0.02870748285204172:
                                                if Q.mass > 110.7004280090332:
                                                    if Q.pt_7 > 39.203125:
                                                        if Q.tau32 > 0.2664194107055664:
                                                            if Q.max_dr > 0.24941160529851913:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.018635897897183895:
                                                                    if Q.eccentricity > 0.9956377446651459:
                                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.03864465467631817:
                                                            if Q.lam2 > 0.00023272421822184697:
                                                                return 't'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.010009380988776684:
                                                        if Q.D2 > 0.6558997333049774:
                                                            if Q.mass > 58.89285659790039:
                                                                if Q.C2 > 0.06544282659888268:
                                                                    return 't'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 42.421875:
                                                                        if Q.D2 > 1.037771224975586:
                                                                            return 'g'   # 65% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.04477175883948803:
                                                                    if Q.pt_7 > 37.8125:
                                                                        if Q.width > 0.012630455195903778:
                                                                            return 't'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.tau32 > 0.3522174656391144:
                                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 40% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.tau32 > 0.43298986554145813:
                                                                            if Q.mean_phi > 0.017025962471961975:
                                                                                return 't'   # 100% of the training jets here get this class from the formula
                                                                            else:
                                                                                if Q.sum_pt_top3 > 262.46875:
                                                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.mass_top5 > 38.96647262573242:
                                                                                        if Q.dr_1 > 0.09790080785751343:
                                                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            return 't'   # 62% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 't'   # 73% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.028682641685009003:
                                                                if Q.e2 > 0.07763554155826569:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.log_sum_pt > 6.268634796142578:
                                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.14935302734375:
                                                            if Q.pt_6 > 42.4375:
                                                                if Q.tau32 > 0.434810608625412:
                                                                    return 'g'   # 40% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.03363417647778988:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_4 > 20.5:
                                                        return 't'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 5.85634708404541:
                                                if Q.D2 > 0.5916080474853516:
                                                    if Q.z_dr_0_0p05 > 0.11369131878018379:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 0.0003829097549896687:
                                                            return 't'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.020541825331747532:
                                                    if Q.tau32 > 0.45798613131046295:
                                                        if Q.C2 > 0.03994059003889561:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.03687071241438389:
                                        if Q.tau32 > 0.2662169635295868:
                                            if Q.pt_7 > 38.171875:
                                                if Q.D2 > 1.1908183097839355:
                                                    if Q.lam2 > 0.00030571768002118915:
                                                        if Q.tau21 > 0.22300364077091217:
                                                            if Q.girth2 > 0.01212936220690608:
                                                                if Q.sum_pt_top5 > 356.203125:
                                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04929087311029434:
                                                            return 't'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_pair_mass > 1.5886833667755127:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.phi_1 > 0.054107666015625:
                                                                    return 't'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.e2 > 0.034285612404346466:
                                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam2 > 0.00010395100980531424:
                                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 592.171875:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 59.204383850097656:
                                                            if Q.centroid_offset > 0.06713876873254776:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.9906911849975586:
                                                                    if Q.centroid_offset > 0.029412899166345596:
                                                                        if Q.tau32 > 0.48419971764087677:
                                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.022921569645404816:
                                                                if Q.tau21 > 0.197525255382061:
                                                                    if Q.mean_eta2 > 0.007971744751557708:
                                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.C2 > 0.04531027749180794:
                                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 485.15625:
                                                    if Q.z_dr_0p2_0p4 > 0.08871739357709885:
                                                        if Q.eccentricity > 0.9834494590759277:
                                                            if Q.centroid_offset > 0.04212651960551739:
                                                                if Q.eccentricity > 0.9916376173496246:
                                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9832225739955902:
                                                            if Q.pt_7 > 29.96875:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth > 0.08921859413385391:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top2 > 502.90625:
                                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 549.109375:
                                                                if Q.e2 > 0.03211953863501549:
                                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.211462497711182:
                                                        if Q.lam2 > 0.00015924145554890856:
                                                            return 't'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.05406174249947071:
                                                                if Q.sum_pt_top5 > 530.125:
                                                                    if Q.dr_3 > 0.10096441209316254:
                                                                        return 't'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_2 > 0.10894317552447319:
                                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 51.909170150756836:
                                                                            if Q.phi_7 > 0.014995574951171875:
                                                                                return 't'   # 52% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.044013453647494316:
                                                            if Q.C2 > 0.05402901954948902:
                                                                if Q.mass > 55.20967674255371:
                                                                    if Q.centroid_offset > 0.07107476517558098:
                                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.017720894888043404:
                                                                        if Q.max_dr > 0.22708889842033386:
                                                                            if Q.mass_top5 > 5.416706562042236:
                                                                                if Q.z_dr_0_0p05 > 0.39115436375141144:
                                                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.lam2 > 0.0007397913432214409:
                                                                                        return 't'   # 61% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 80% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.C2 > 0.07917079702019691:
                                                                            return 't'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.phi_1 > -0.058868408203125:
                                                                if Q.D2 > 1.1634542346000671:
                                                                    if Q.z_top5 > 0.7659624516963959:
                                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_4 > 37.8125:
                                                                        if Q.z_dr_0p1_0p2 > 0.8178126215934753:
                                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p2_0p4 > 0.08232417702674866:
                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.01242868835106492:
                                                return 't'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 39.234375:
                                                    if Q.C2 > 0.09497178345918655:
                                                        return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.21737178415060043:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.024350750260055065:
                                            if Q.mass > 108.32182693481445:
                                                if Q.pt_7 > 35.375:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 40% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9960939884185791:
                                                    if Q.D2 > 0.48764532804489136:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi > -0.03256675973534584:
                                                            if Q.lam1 > 0.02737169712781906:
                                                                if Q.max_dr > 0.21444446593523026:
                                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 771.7421875:
                                                if Q.pt_7 > 40.109375:
                                                    if Q.C2 > 0.02263379655778408:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.37974822521209717:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.04624916799366474:
                                                    if Q.sum_pt_top2 > 459.703125:
                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.24131543189287186:
                                                            if Q.planar_flow > 0.021213078871369362:
                                                                return 't'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 38.21875:
                                                                    if Q.mass > 71.56124496459961:
                                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 99% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.034152233973145485:
                        if Q.log_sum_pt > 5.81950306892395:
                            if Q.C2 > 0.04409065283834934:
                                if Q.lam2 > 0.0016081480425782502:
                                    if Q.girth2 > 0.012420170474797487:
                                        return 't'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 292.15625:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.mean_eta > -0.009729328099638224:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.07594805583357811:
                                        if Q.lam2 > 0.000127180821436923:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_5 > 36.765625:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.2572724372148514:
                                            if Q.z_dr_0_0p05 > 0.24672669917345047:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 288.140625:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0008354152669198811:
                                                        if Q.tau32 > 0.3602458983659744:
                                                            if Q.log_sum_pt > 5.981794834136963:
                                                                return 't'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_dr_0_0p05 > 0.5:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 36.0625:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_5 > 34.921875:
                                                                    return 't'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.015983350574970245:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9919707179069519:
                                                    if Q.LHA > 0.3489881157875061:
                                                        return 'g'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 87% of the training jets here get this class from the formula
                            else:
                                return 't'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.pt_4 > 27.5390625:
                                if Q.lam2 > 0.002462068689055741:
                                    if Q.lam1 > 0.009946424048393965:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.04443258047103882:
                                        if Q.centroid_offset > 0.05056813918054104:
                                            if Q.tau32 > 0.4698530286550522:
                                                if Q.dr_0 > 0.10944430902600288:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 27.0234375:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 74% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top5 > 28.525423049926758:
                                    return 't'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.C2 > 0.03951165825128555:
                            if Q.mass > 41.45612335205078:
                                if Q.lam2 > 0.001783880579750985:
                                    if Q.pt_5 > 28.0:
                                        if Q.tau32 > 0.3582700937986374:
                                            if Q.mean_phi2 > 0.007349108578637242:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.21850263327360153:
                                        if Q.tau21 > 0.6136941015720367:
                                            return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.023096544668078423:
                                                if Q.mean_phi2 > 0.005342096788808703:
                                                    if Q.girth2_top3 > 0.009067279286682606:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.07962431386113167:
                                            if Q.pt_6 > 33.546875:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 37.42132568359375:
                                    if Q.centroid_offset > 0.019664174877107143:
                                        if Q.tau21 > 0.20940343290567398:
                                            if Q.lam2 > 0.0025170313892886043:
                                                return 't'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                        else:
                            if Q.pt_5 > 29.640625:
                                if Q.mass > 37.76174545288086:
                                    return 't'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.030626104213297367:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.max_pair_mass > 16.44151782989502:
                                    return 't'   # 56% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 83% of the training jets here get this class from the formula
            else:
                if Q.e2 > 0.04764731973409653:
                    if Q.tau21 > 0.14471519738435745:
                        if Q.log_sum_pt > 6.157475471496582:
                            if Q.centroid_offset > 0.033962052315473557:
                                return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 31.625:
                                    if Q.z_dr_0p05_0p1 > 0.6642431020736694:
                                        if Q.girth2 > 0.009426658973097801:
                                            if Q.pt_6 > 44.28125:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.009517140686511993:
                                            if Q.pt_6 > 42.09375:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_5 > 0.07665940746665001:
                                                    if Q.eta_7 > 0.01782989501953125:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top3 > 0.009831487201154232:
                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 36.38095664978027:
                                if Q.pt_6 > 35.25:
                                    if Q.tau32 > 0.4702243506908417:
                                        return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.25847578048706055:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.03830323740839958:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.width > 0.009362242184579372:
                            if Q.pt_7 > 42.765625:
                                if Q.width > 0.009655485861003399:
                                    if Q.z_dr_0p05_0p1 > 0.5207825899124146:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.11898661032319069:
                                        if Q.tau21 > 0.07285704091191292:
                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.11167202889919281:
                                    if Q.tau21 > 0.11694115400314331:
                                        if Q.mass_top5 > 41.06594657897949:
                                            if Q.z_top5 > 0.8063444197177887:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 33.75:
                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.pt_6 > 38.609375:
                                if Q.girth2_top5 > 0.008394225966185331:
                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 707.59375:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.11577211320400238:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 53% of the training jets here get this class from the formula
                else:
                    if Q.C2 > 0.03853870369493961:
                        if Q.z_7 > 0.06192115321755409:
                            if Q.centroid_offset > 0.030554247088730335:
                                if Q.sum_pt_top2 > 142.09375:
                                    if Q.dr_0 > 0.0806034691631794:
                                        if Q.z_dr_0p05_0p1 > 0.7836087644100189:
                                            return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr01 > 0.04692289046943188:
                                                return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.819749116897583:
                                            if Q.max_pair_mass > 1.5500012040138245:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.24111726135015488:
                                    return 'g'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 47.12318229675293:
                                        if Q.pt1_dr01 > 1.6066410541534424:
                                            return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 36.5625:
                                if Q.log_sum_pt > 6.522758960723877:
                                    return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 63% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.201776027679443:
                                    if Q.lam2 > 0.00011682940748869441:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_0 > 228.25:
                                            if Q.e2 > 0.028466593474149704:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.05490165576338768:
                                        if Q.mean_eta > 0.005125068826600909:
                                            return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 72% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 75.84713363647461:
                            if Q.mass_top5 > 62.81814384460449:
                                if Q.eta_7 > 0.020721435546875:
                                    return 't'   # 55% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 74% of the training jets here get this class from the formula
                            else:
                                return 't'   # 55% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04535564035177231:
                                if Q.z_dr_0_0p05 > 0.03928683325648308:
                                    if Q.pt_7 > 39.484375:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 484.75:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.009333847556263208:
                                        return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 42.265625:
                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.765625:
                                    if Q.sum_pt_top3 > 373.53125:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 5.930938005447388:
                                        if Q.D2 > 0.9668562710285187:
                                            if Q.lam2 > 4.1202667489415035e-05:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 37.234375:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 52% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.245094299316406:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.027192377485334873:
                    if Q.max_dr > 0.14311284571886063:
                        if Q.C2 > 0.04650634340941906:
                            if Q.pt_7 > 36.203125:
                                if Q.max_dr > 0.1757609024643898:
                                    if Q.sum_pt > 665.359375:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top5 > 3.722250819206238:
                                            if Q.girth2_top2 > 0.0024166707880795:
                                                if Q.girth > 0.07623636722564697:
                                                    if Q.tau32 > 0.2445949912071228:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03582625836133957:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.08117963001132011:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 583.34375:
                                    if Q.lam2 > 0.00030999140290077776:
                                        if Q.sum_pt > 849.3828125:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_5 > 0.03788406774401665:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 58% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.763581991195679:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 503.234375:
                                        if Q.planar_flow > 0.04565221630036831:
                                            if Q.max_dr > 0.17135003209114075:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.039880046620965004:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.9302090406417847:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.8440075218677521:
                                            if Q.tau21 > 0.31657539308071136:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 30.1953125:
                                                if Q.e2 > 0.03279547207057476:
                                                    if Q.mass > 36.69034385681152:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 838.6875:
                                if Q.pt_7 > 28.8046875:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 44% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0003418350388528779:
                                    return 't'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.2000608891248703:
                                        if Q.centroid_offset > 0.04103732295334339:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.007912972941994667:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 1.4736995697021484:
                                            if Q.sum_pt_top3 > 493.8125:
                                                if Q.lam2 > 5.866791616426781e-05:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 41.703125:
                                                    if Q.mass > 49.575103759765625:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.007593109039589763:
                                                return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.15918301790952682:
                                                    if Q.centroid_offset > 0.03217103146016598:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.18090158700942993:
                                                            return 't'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03495190106332302:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.037078382447361946:
                            if Q.tau21 > 0.2107367143034935:
                                if Q.log_sum_pt > 6.163345813751221:
                                    if Q.pt_7 > 39.09375:
                                        if Q.width > 0.007873937953263521:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.007373024011030793:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.2811947464942932:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.007440144661813974:
                                    if Q.max_dr > 0.1089118979871273:
                                        return 't'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.042890748009085655:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.006738333031535149:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 34.609375:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.00024150682293111458:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04750247299671173:
                                            return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.20294246822595596:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 33.421875:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p1_0p2 > 0.2385910451412201:
                                if Q.tau21 > 0.215524323284626:
                                    if Q.mass > 36.29891586303711:
                                        if Q.pt_7 > 32.515625:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 43% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0002856451756088063:
                                        if Q.tau21 > 0.14318832755088806:
                                            if Q.pt_7 > 37.171875:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.030255134217441082:
                                            if Q.D2 > 0.3367857038974762:
                                                if Q.pt_7 > 35.203125:
                                                    if Q.LHA > 0.3325899839401245:
                                                        return 't'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 37.0:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.11974674463272095:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 31.8828125:
                                    if Q.log_sum_pt > 6.1119513511657715:
                                        if Q.girth2 > 0.008534498512744904:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.306283101439476:
                                                if Q.tau21 > 0.14070520550012589:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 36% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 529.796875:
                                        if Q.girth2 > 0.007947712205350399:
                                            if Q.pt_7 > 28.3671875:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 451.90625:
                                            if Q.pt_7 > 29.71875:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 48% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.006914052646607161:
                        if Q.max_dr > 0.20981992781162262:
                            if Q.centroid_offset > 0.010773746762424707:
                                if Q.log_sum_pt > 6.664279460906982:
                                    if Q.max_dr > 0.2770105302333832:
                                        if Q.pt_7 > 34.953125:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.09700707346200943:
                                                return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3374049961566925:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 24.59375:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 47% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.019326966255903244:
                                            if Q.pt_7 > 35.40625:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.09362659603357315:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.05721266567707062:
                                        if Q.C2 > 0.0473988875746727:
                                            if Q.max_dr > 0.2530026137828827:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.183568954467773:
                                                    if Q.pt_7 > 35.953125:
                                                        if Q.z_7 > 0.07194063439965248:
                                                            if Q.tau32 > 0.3072418123483658:
                                                                return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 365.8125:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.04639839939773083:
                                            if Q.C2 > 0.060484202578663826:
                                                if Q.C2 > 0.0920809842646122:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.2411804273724556:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 688.5625:
                                                    if Q.girth2 > 0.007702864706516266:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_7 > 0.027444323524832726:
                                                if Q.log_sum_pt > 6.62427282333374:
                                                    if Q.max_dr > 0.254159152507782:
                                                        return 't'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 42% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 673.7109375:
                                    if Q.width > 0.00825625378638506:
                                        if Q.max_dr > 0.27106139063835144:
                                            if Q.pt_7 > 36.3125:
                                                return 'g'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.720012664794922:
                                                    return 'q'   # 31% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 785.1484375:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top5 > 0.8276373445987701:
                                                    return 't'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.09023157134652138:
                                            return 't'   # 38% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_7 > 0.28590282797813416:
                                                if Q.sum_pt_top2 > 454.625:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 787.0703125:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 25.609375:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 541.203125:
                                        if Q.z_7 > 0.05143407918512821:
                                            if Q.max_dr > 0.2797178775072098:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.008575224317610264:
                                                    return 't'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.24389474838972092:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_4 > 47.453125:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.087999105453491:
                                if Q.mass > 53.49141311645508:
                                    if Q.width > 0.008786667603999376:
                                        if Q.e2 > 0.04363331198692322:
                                            if Q.e2 > 0.04757891222834587:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 35.984375:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.576534748077393:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1147061213850975:
                                                            if Q.n_dr_0p1_0p2 > 2.5:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top5 > 56.800058364868164:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.010953844990581274:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 37.5625:
                                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.993659257888794:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.022623839788138866:
                                                if Q.max_dr > 0.15708794444799423:
                                                    if Q.girth2 > 0.008218543604016304:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 5.898910967516713e-05:
                                                            if Q.tau21 > 0.155593603849411:
                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.008389647118747234:
                                                        if Q.e2 > 0.04438787326216698:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.007000066339969635:
                                                    if Q.max_dr > 0.18007688969373703:
                                                        if Q.width > 0.00824839947745204:
                                                            if Q.log_sum_pt > 6.639957427978516:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.11374182626605034:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_top5 > 0.7828002870082855:
                                                                        return 't'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.019580076448619366:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 678.375:
                                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 28.6640625:
                                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.008627574425190687:
                                                            if Q.e2 > 0.044729121029376984:
                                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_4 > 53.796875:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.007161796558648348:
                                                                return 'Z'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 59.32656669616699:
                                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0_0p05 > 0.07155352458357811:
                                                                        if Q.max_dr > 0.11585508286952972:
                                                                            if Q.z_dr_0p1_0p2 > 0.24309583008289337:
                                                                                if Q.n_dr_0_0p05 > 1.5:
                                                                                    if Q.mean_phi > -0.001418957777787:
                                                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 699.265625:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.006905771093443036:
                                                            if Q.centroid_offset > 0.007013343507423997:
                                                                if Q.z_dr_0p1_0p2 > 0.304141029715538:
                                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 27.6796875:
                                        if Q.girth2 > 0.007083080243319273:
                                            if Q.max_dr > 0.15856285393238068:
                                                if Q.tau21 > 0.1827382594347:
                                                    if Q.centroid_offset > 0.020996239967644215:
                                                        if Q.dr_0 > 0.06105882301926613:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.2772608995437622:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.018232800997793674:
                                                        if Q.e2_sq > 0.007209376199170947:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 8.086388334049843e-05:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 37.25:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.008286416996270418:
                                                                return 't'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.1331930011510849:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.008655416779220104:
                                                    if Q.e2 > 0.04675234481692314:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 53.96875:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 30.6015625:
                                                        if Q.width > 0.0072699058800935745:
                                                            if Q.centroid_offset > 0.02031971514225006:
                                                                if Q.max_dr > 0.13942061364650726:
                                                                    if Q.tau21 > 0.15788863599300385:
                                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.eccentricity > 0.9813854396343231:
                                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p1_0p2 > 0.24287017434835434:
                                                                        if Q.tau21 > 0.10711048915982246:
                                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.eccentricity > 0.9776242077350616:
                                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 57% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.5707274377346039:
                                                                if Q.centroid_offset > 0.007747799390926957:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.tau32 > 0.5876200497150421:
                                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.planar_flow > 0.04909418150782585:
                                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 0.8074790835380554:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top5 > 461.515625:
                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_dr_0p1_0p2 > 0.4044411927461624:
                                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.lam2 > 0.00014552669745171443:
                                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 484.2734375:
                                                            if Q.dr_0 > 0.07022532820701599:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.5854582488536835:
                                                                    if Q.m012 > 26.72110939025879:
                                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.20870191603899002:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011791516561061144:
                                                if Q.z_dr_0p1_0p2 > 0.3015894442796707:
                                                    if Q.C2 > 0.03561815060675144:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.08496370911598206:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.11469431221485138:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00021589503739960492:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.04086880385875702:
                                                        if Q.dr_0 > 0.09367188811302185:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9837526381015778:
                                                                if Q.girth2 > 0.0069645356852561235:
                                                                    if Q.centroid_offset > 0.005652307532727718:
                                                                        if Q.sum_pt > 621.640625:
                                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt_top5 > 432.53125:
                                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.028551779687404633:
                                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_0 > 113.46875:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 24.0703125:
                                            if Q.sum_pt_top5 > 431.6875:
                                                if Q.girth2_top2 > 0.002794862608425319:
                                                    if Q.width > 0.0082058678381145:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.18102209270000458:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.06364568322896957:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.2545039504766464:
                                                    return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 32.828125:
                                    if Q.tau21 > 0.15225475281476974:
                                        if Q.max_dr > 0.17933640629053116:
                                            if Q.planar_flow > 0.16932716965675354:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 40% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 5.955108165740967:
                                                if Q.girth2 > 0.007167004747316241:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.12781425565481186:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.14265945553779602:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_eta > 0.001022425334667787:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.08136146515607834:
                                            if Q.pt_7 > 38.625:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_0 > 93.21875:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.tau21 > 0.18002185225486755:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.040092840790748596:
                            if Q.lam2 > 0.00025957112666219473:
                                if Q.dr_0 > 0.0931284911930561:
                                    if Q.lam2 > 0.00034289997711312026:
                                        if Q.mass > 46.43575477600098:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.006789947627112269:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 32.66198921203613:
                                        if Q.z_dr_0p1_0p2 > 0.2969927340745926:
                                            if Q.mass > 51.08442306518555:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 0.9184796810150146:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 45% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.12379781901836395:
                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.006777352653443813:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.04280569963157177:
                                                        if Q.log_sum_pt > 6.47099232673645:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr01 > 0.14848282933235168:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 64.25715637207031:
                                    if Q.width > 0.006766933482140303:
                                        if Q.girth2_top5 > 0.007369426777586341:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.006830722093582153:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.08271316066384315:
                                                    if Q.sum_pt_top2 > 424.8125:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 865.65625:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010231659281998873:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 37.265625:
                                                    if Q.mass > 67.98991775512695:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top3 > 0.006695061223581433:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 456.6875:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.01192340673878789:
                                        if Q.log_sum_pt > 6.499575614929199:
                                            if Q.width > 0.006734242895618081:
                                                if Q.centroid_offset > 0.016158397309482098:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.03951514698565006:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.006852437974885106:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.07793280854821205:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.006505213910713792:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.0067103467881679535:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.7355172336101532:
                                                if Q.dr_0 > 0.08910141885280609:
                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.3198496848344803:
                                                    if Q.centroid_offset > 0.01748037990182638:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.006297184387221932:
                                                        if Q.lam2 > 0.00014744657528353855:
                                                            if Q.z_7 > 0.07216287776827812:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.08439257368445396:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.006794988876208663:
                                            if Q.log_sum_pt > 6.5997960567474365:
                                                if Q.girth2_top5 > 0.006991623900830746:
                                                    if Q.tau21 > 0.09324272722005844:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.14271000772714615:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 691.265625:
                                                        if Q.girth2_top5 > 0.0066609010100364685:
                                                            if Q.tau32 > 0.6640228629112244:
                                                                if Q.dr01 > 0.16673137992620468:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.14039763063192368:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.005047593265771866:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 737.78125:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 1.9388178586959839:
                                if Q.mass > 58.300703048706055:
                                    if Q.centroid_offset > 0.01580371893942356:
                                        return 'q'   # 48% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.05177597515285015:
                                        if Q.z_dr_0p05_0p1 > 0.08693400025367737:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 35% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.508687496185303:
                                    if Q.e2 > 0.03869481198489666:
                                        if Q.z_dr_0p05_0p1 > 0.6187694668769836:
                                            if Q.girth2 > 0.0067127791699022055:
                                                return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 794.296875:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.006552303675562143:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 63.95870399475098:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.02430106047540903:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1033.2109375:
                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.03696271404623985:
                                                if Q.z_dr_0p05_0p1 > 0.099772859364748:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.14981473982334137:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.0305501576513052:
                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 62.88977241516113:
                                                        return 'Z'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 26.3359375:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p1_0p2 > 0.25208035111427307:
                                        if Q.girth > 0.07554281502962112:
                                            if Q.m01 > 19.908422470092773:
                                                return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.047366293147206306:
                                                return 't'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.02960838470607996:
                                            if Q.pt_7 > 27.4609375:
                                                if Q.z_dr_0p05_0p1 > 0.16552969068288803:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.20938795804977417:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.388286352157593:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1473577842116356:
                    if Q.girth2 > 0.005036225542426109:
                        if Q.centroid_offset > 0.010997515171766281:
                            if Q.centroid_offset > 0.03173349052667618:
                                if Q.max_dr > 0.1873420998454094:
                                    if Q.sum_pt_top5 > 581.859375:
                                        if Q.pt_7 > 37.671875:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.327808976173401:
                                                if Q.lam2 > 0.00022870628163218498:
                                                    if Q.z_top5 > 0.8919108510017395:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00013521083019440994:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.33909304440021515:
                                            if Q.sum_pt_top3 > 345.171875:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.47229790687561:
                                                if Q.pt_7 > 38.515625:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.04929388500750065:
                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.043666306883096695:
                                        if Q.sum_pt_top5 > 560.359375:
                                            return 'q'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.006018025102093816:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mean_phi > 0.0024213746073655784:
                                                    return 't'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.006078311009332538:
                                            if Q.centroid_offset > 0.03705338574945927:
                                                if Q.C2 > 0.040979379788041115:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 613.515625:
                                    if Q.max_dr > 0.29290881752967834:
                                        if Q.sum_pt > 741.2421875:
                                            if Q.centroid_offset > 0.020864679478108883:
                                                if Q.pt_7 > 29.0546875:
                                                    return 'g'   # 30% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0006430884532164782:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.005436028121039271:
                                            if Q.max_dr > 0.24085644632577896:
                                                if Q.centroid_offset > 0.02052431460469961:
                                                    if Q.mass > 54.73478317260742:
                                                        if Q.centroid_offset > 0.025683452375233173:
                                                            return 'q'   # 37% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04654432460665703:
                                                            if Q.pt_7 > 29.296875:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 734.5703125:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 26.265625:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.000813546939752996:
                                                    if Q.mass > 51.57265090942383:
                                                        if Q.girth2 > 0.005593636306002736:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.15802259743213654:
                                                                if Q.centroid_offset > 0.013550959527492523:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 22.4296875:
                                                            if Q.centroid_offset > 0.014872270170599222:
                                                                if Q.max_dr > 0.20217034965753555:
                                                                    if Q.centroid_offset > 0.02263334859162569:
                                                                        if Q.D2 > 1.7793254256248474:
                                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_dr_0p05_0p1 > 0.04348511062562466:
                                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam2 > 0.000141377211548388:
                                                                            return 't'   # 54% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1613810658454895:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.width > 0.005849521374329925:
                                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 49% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.4883546382188797:
                                                        return 't'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.1366085410118103:
                                                if Q.centroid_offset > 0.015734425745904446:
                                                    if Q.sum_pt_top5 > 531.140625:
                                                        if Q.centroid_offset > 0.017435552552342415:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1552872210741043:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.019359806552529335:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.17293116450309753:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.013483134098351002:
                                                            if Q.max_dr > 0.15960507094860077:
                                                                if Q.mass > 57.371870040893555:
                                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03317885287106037:
                                                    if Q.width > 0.0052193026058375835:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.0007515489123761654:
                                                        if Q.centroid_offset > 0.013285670429468155:
                                                            if Q.D2 > 3.1919363737106323:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam1 > 0.0050220664124935865:
                                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.494088053703308:
                                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.016640723682940006:
                                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.18865413963794708:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.005295505048707128:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.phi_1 > 0.010463714599609375:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 45% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.06927120685577393:
                                        if Q.max_dr > 0.22003918886184692:
                                            if Q.sum_pt > 512.8828125:
                                                if Q.centroid_offset > 0.024262010119855404:
                                                    if Q.mass_top5 > 5.617968320846558:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.24277672171592712:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 490.484375:
                                                if Q.pt_7 > 27.2734375:
                                                    if Q.centroid_offset > 0.013275498524308205:
                                                        if Q.centroid_offset > 0.018331734463572502:
                                                            if Q.tau21 > 0.14546099305152893:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eccentricity > 0.9843264520168304:
                                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2_sq > 0.005226944573223591:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.1709626466035843:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2_sq > 0.0058691424783319235:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.5995821952819824:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.17893998324871063:
                                                        return 't'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.03995175287127495:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 30.8359375:
                                                    if Q.girth2 > 0.005408599739894271:
                                                        if Q.max_dr > 0.1969752237200737:
                                                            return 'g'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.00567163759842515:
                                                        return 't'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 32.71875:
                                            if Q.C2 > 0.042357491329312325:
                                                if Q.girth2_top3 > 0.0012583023053593934:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 47% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0010294672683812678:
                                                    if Q.max_dr > 0.17227549850940704:
                                                        if Q.planar_flow > 0.04120813123881817:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 549.6875:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 488.1875:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.17563309520483017:
                                if Q.width > 0.005610994296148419:
                                    if Q.mass > 56.15493583679199:
                                        if Q.girth2 > 0.00581318000331521:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.027963309548795223:
                                                if Q.D2 > 1.1734967827796936:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006056769751012325:
                                                        if Q.girth > 0.058222223073244095:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 68.91556930541992:
                                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005180601961910725:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.030222478322684765:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.m01 > 1.0734903216362:
                                                            if Q.sum_pt > 943.203125:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.640625:
                                            if Q.C2 > 0.04031324200332165:
                                                if Q.mass > 41.12510681152344:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_3 > 0.03321359120309353:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.005914317211136222:
                                                    if Q.mass > 44.71842575073242:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.02849564142525196:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.0011118566617369652:
                                                if Q.sum_pt_top5 > 496.0625:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 40% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 66% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.5027176141738892:
                                        if Q.mass > 47.75173759460449:
                                            if Q.centroid_offset > 0.00519929314032197:
                                                if Q.dr_0 > 0.015880372375249863:
                                                    if Q.mass > 54.33851623535156:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 30.96875:
                                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 40% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.0052508101798594:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.009008526802063:
                                                        if Q.dr_0 > 0.01754461694508791:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.2113518714904785:
                                                if Q.girth > 0.04342150315642357:
                                                    if Q.girth2_top3 > 0.0018175874138250947:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.05144745856523514:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 36% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.22732066363096237:
                                            if Q.centroid_offset > 0.006298898486420512:
                                                if Q.centroid_offset > 0.009392383974045515:
                                                    return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.005405323579907417:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.24065788835287094:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2_sq > 0.005441914079710841:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.009763359557837248:
                                                if Q.lam1 > 0.0052318274974823:
                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 72.57498168945312:
                                                    if Q.pt_7 > 16.4140625:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 74.97235107421875:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.005484540248289704:
                                                        if Q.centroid_offset > 0.006174708250910044:
                                                            if Q.max_dr > 0.1926106959581375:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 67.84177017211914:
                                                            if Q.centroid_offset > 0.0072762127965688705:
                                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.00603402778506279:
                                    if Q.centroid_offset > 0.005022869678214192:
                                        if Q.mass > 58.53314018249512:
                                            if Q.max_dr > 0.1498899832367897:
                                                if Q.dr_2 > 0.039266763255000114:
                                                    if Q.girth2 > 0.006193377776071429:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.16162476688623428:
                                                            return 'Z'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.007091948296874762:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.18531335145235062:
                                                if Q.LHA > 0.27880530059337616:
                                                    if Q.max_dr > 0.1528583988547325:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009124443866312504:
                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.0376365277916193:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.16157711297273636:
                                            if Q.width > 0.006146302679553628:
                                                if Q.mass > 65.32922744750977:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.7662197351455688:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 72.49267959594727:
                                                if Q.centroid_offset > 0.0024256103206425905:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.006415758049115539:
                                                    if Q.z_dr_0p1_0p2 > 0.21143993735313416:
                                                        if Q.sum_pt > 783.890625:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 72.7567024230957:
                                        if Q.pt_7 > 20.59375:
                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 77.22766876220703:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.005845865234732628:
                                            if Q.centroid_offset > 0.007848719134926796:
                                                if Q.max_dr > 0.16527920216321945:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0059113719034940004:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.04470329359173775:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.5815497040748596:
                                                if Q.max_dr > 0.1673768013715744:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 34.5554141998291:
                                                    if Q.centroid_offset > 0.009473908692598343:
                                                        if Q.girth2 > 0.005595809780061245:
                                                            if Q.max_dr > 0.16514407098293304:
                                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.018207107670605183:
                            if Q.lam1 > 0.003343206364661455:
                                if Q.max_dr > 0.1685866415500641:
                                    if Q.mass > 36.795921325683594:
                                        if Q.z_dr_0p1_0p2 > 0.05715266056358814:
                                            if Q.girth > 0.05013234354555607:
                                                if Q.centroid_offset > 0.020061801187694073:
                                                    if Q.lam1 > 0.003896624082699418:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.02319361362606287:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0_0p05 > 5.5:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.004377203295007348:
                                                        if Q.mass > 45.39959907531738:
                                                            return 'Z'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.08663881942629814:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.17775805294513702:
                                                            if Q.eta_7 > 0.005229949951171875:
                                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.022992713376879692:
                                                    if Q.max_dr > 0.17354366183280945:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.003822589758783579:
                                                        if Q.max_dr > 0.18612543493509293:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.019176535308361053:
                                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03252033144235611:
                                                if Q.max_dr > 0.2261962592601776:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.039665523916482925:
                                                        return 'q'   # 35% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.906709671020508:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top5 > 0.0007612720073666424:
                                                        if Q.max_dr > 0.2680809646844864:
                                                            if Q.sum_pt > 737.4453125:
                                                                if Q.centroid_offset > 0.024882283061742783:
                                                                    if Q.pt_7 > 21.375:
                                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_1 > 0.0369623526930809:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 677.0625:
                                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 25.5234375:
                                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.24256916344165802:
                                            if Q.max_dr > 0.19787847995758057:
                                                if Q.sum_pt_top3 > 433.4375:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.01955240685492754:
                                                        return 'Z'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02201743610203266:
                                                    if Q.centroid_offset > 0.04123099707067013:
                                                        return 'q'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 26.7890625:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt_sq > 0.0029045625124126673:
                                                                return 't'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 39% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.19089990109205246:
                                                if Q.pt_7 > 28.1484375:
                                                    if Q.LHA > 0.22038427740335464:
                                                        return 'Z'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02617923729121685:
                                        if Q.girth > 0.05248860456049442:
                                            if Q.LHA > 0.25833994150161743:
                                                if Q.centroid_offset > 0.04281475953757763:
                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 26.9453125:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 649.1953125:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.15381630510091782:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 36.875:
                                                        if Q.centroid_offset > 0.029430773109197617:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 735.671875:
                                                if Q.mean_phi2 > 0.0018700932268984616:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.00439949007704854:
                                            if Q.centroid_offset > 0.021565438248217106:
                                                if Q.dr_5 > 0.03613451309502125:
                                                    if Q.max_dr > 0.15594550967216492:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.004701812518760562:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.023808861151337624:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1557328850030899:
                                                    if Q.lam1 > 0.004546158714219928:
                                                        if Q.sum_pt > 662.15625:
                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_4 > 0.08212842047214508:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 47.66643714904785:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.023295212537050247:
                                                if Q.girth2 > 0.003863360616378486:
                                                    if Q.max_dr > 0.16063034534454346:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 43.4926815032959:
                                                            if Q.dr_0 > 0.04517676122486591:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.809141159057617:
                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00035272100649308413:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.16103409230709076:
                                                            if Q.girth2 > 0.003994735423475504:
                                                                if Q.centroid_offset > 0.021555383689701557:
                                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19978925585746765:
                                    if Q.centroid_offset > 0.021441803313791752:
                                        if Q.pt_5 > 20.8515625:
                                            if Q.log_sum_pt > 6.531150817871094:
                                                if Q.lam1 > 0.0025734694208949804:
                                                    if Q.centroid_offset > 0.033816417679190636:
                                                        return 'g'   # 31% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.022923593409359455:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.22834992408752441:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_5 > 0.04722006246447563:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.21468045562505722:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.06676913797855377:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.35306739807129:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 40% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 40.708635330200195:
                                            if Q.max_dr > 0.2320319041609764:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.003102890099398792:
                                                    if Q.centroid_offset > 0.018656007014214993:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 44.9221305847168:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.23417334258556366:
                                                if Q.girth2 > 0.0021463691955432296:
                                                    if Q.centroid_offset > 0.019985816441476345:
                                                        if Q.pt_7 > 19.46875:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.26477135717868805:
                                                            return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 642.875:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.028182193636894226:
                                                    if Q.z_dr_0p2_0p4 > 0.03256280533969402:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.002938699792139232:
                                                        if Q.centroid_offset > 0.020270006731152534:
                                                            return 'Z'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.029111566953361034:
                                        if Q.max_dr > 0.172550231218338:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0323286447674036:
                                                if Q.eccentricity > 0.8247499167919159:
                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 36.5900764465332:
                                                    return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.21387837827205658:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00022882809571456164:
                                            if Q.centroid_offset > 0.02413860522210598:
                                                if Q.mass > 35.98686408996582:
                                                    if Q.max_dr > 0.1630450263619423:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.17262475192546844:
                                                        return 'Z'   # 43% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.059797728434205055:
                                                    return 'W'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.18971849232912064:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 988.90625:
                                                if Q.girth > 0.036545876413583755:
                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.027243142016232014:
                                                    if Q.max_dr > 0.17733852565288544:
                                                        if Q.mass_over_sum_pt_sq > 0.0021546990610659122:
                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.girth > 0.021229864098131657:
                                if Q.max_dr > 0.2197299674153328:
                                    if Q.width > 0.00327483844012022:
                                        if Q.centroid_offset > 0.012613121885806322:
                                            if Q.mass > 45.66777992248535:
                                                if Q.girth2 > 0.003996365703642368:
                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 2.7859444618225098:
                                                        if Q.girth2 > 0.003475203411653638:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 3.8290354013442993:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.015603174455463886:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01448037289083004:
                                                            if Q.lam1 > 0.003568667802028358:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 38.6607551574707:
                                                    if Q.LHA > 0.2131180688738823:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.24276254326105118:
                                                            if Q.sum_pt > 735.703125:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.2740905284881592:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 3.5000548362731934:
                                                        return 'g'   # 30% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.053254423663020134:
                                                if Q.lam1 > 0.003685102565214038:
                                                    if Q.mass > 47.68144989013672:
                                                        if Q.centroid_offset > 0.005778530612587929:
                                                            if Q.dr_1 > 0.01303991349413991:
                                                                if Q.max_dr > 0.30131784081459045:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_5 > 0.050007518380880356:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.008126304019242525:
                                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.34991635382175446:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.004709442146122456:
                                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.06644012778997421:
                                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 30.96875:
                                                            if Q.girth2 > 0.004741066135466099:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_1 > 0.021644589491188526:
                                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 4.9318084716796875:
                                                        if Q.girth2_top2 > 0.0002893608616432175:
                                                            if Q.lam1 > 0.0032355289440602064:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 836.53125:
                                                                if Q.tau32 > 0.47609901428222656:
                                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.010520500130951405:
                                                            if Q.max_dr > 0.25490377843379974:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.004554860759526491:
                                                    if Q.centroid_offset > 0.008832794614136219:
                                                        if Q.max_dr > 0.23889929056167603:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 1.7922370716405567e-05:
                                                                if Q.centroid_offset > 0.010954238008707762:
                                                                    return 'Z'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.2628510594367981:
                                                            if Q.centroid_offset > 0.005744770634919405:
                                                                if Q.mass > 64.23207473754883:
                                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top2 > 509.34375:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_1 > 0.01888587698340416:
                                                                    if Q.lam1 > 0.004793609259650111:
                                                                        return 'Z'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 67.41336059570312:
                                                                if Q.z_7 > 0.014671863988041878:
                                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 60.13537788391113:
                                                        if Q.pt_5 > 38.59375:
                                                            if Q.e2 > 0.0177029836922884:
                                                                if Q.pt_7 > 27.2109375:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.0087772230617702:
                                                                if Q.max_dr > 0.2768442630767822:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam1 > 0.004192001419141889:
                                                                        if Q.max_dr > 0.24012281745672226:
                                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0041068848222494125:
                                                            if Q.D2 > 2.508945345878601:
                                                                if Q.centroid_offset > 0.00864548608660698:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.011913727037608624:
                                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 40.55649375915527:
                                                                if Q.D2 > 3.1824172735214233:
                                                                    if Q.centroid_offset > 0.009663721080869436:
                                                                        if Q.girth2 > 0.0037058035377413034:
                                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 36.06550407409668:
                                            if Q.centroid_offset > 0.01545076398178935:
                                                if Q.width > 0.0026600839337334037:
                                                    if Q.max_dr > 0.2549193948507309:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.003209782182238996:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 7.025003433227539:
                                                    if Q.C2 > 0.061461715027689934:
                                                        return 'Z'   # 45% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 29.046875:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 39.40801239013672:
                                                        if Q.C2 > 0.0613558255136013:
                                                            if Q.width > 0.002985421451739967:
                                                                if Q.centroid_offset > 0.011156081687659025:
                                                                    if Q.e2 > 0.015076772775501013:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.013583495747298002:
                                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.girth2_top5 > 0.0005214890697970986:
                                                                                return 'W'   # 50% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 9.7890625:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 948.138671875:
                                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01106997486203909:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top5 > 0.9147890508174896:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.012292200699448586:
                                                if Q.pt_6 > 16.5078125:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 916.24609375:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 542.34375:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 32% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.001930810627527535:
                                        if Q.log_sum_pt > 6.971162796020508:
                                            if Q.mass > 65.0066032409668:
                                                if Q.pt_6 > 20.28125:
                                                    if Q.log_sum_pt > 7.152155160903931:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0052495854906737804:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 7.006305694580078:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 42.4375:
                                                    return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.010061409324407578:
                                                        if Q.mass > 59.06011390686035:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.05169534124433994:
                                                if Q.LHA > 0.22504500299692154:
                                                    if Q.centroid_offset > 0.013351718429476023:
                                                        if Q.centroid_offset > 0.01448138803243637:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.2119147852063179:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.010590886697173119:
                                                            if Q.z_dr_0_0p05 > 0.8987219333648682:
                                                                if Q.e2_sq > 0.004643339663743973:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01524149114266038:
                                                        if Q.width > 0.0034839658765122294:
                                                            if Q.sum_pt > 733.8046875:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9534662365913391:
                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.2181638851761818:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 34.45332145690918:
                                                    if Q.centroid_offset > 0.015241702552884817:
                                                        if Q.width > 0.004407386993989348:
                                                            if Q.max_dr > 0.17557194083929062:
                                                                if Q.mass > 49.6636905670166:
                                                                    if Q.max_dr > 0.18581487983465195:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.lam1 > 0.0045634249690920115:
                                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.1944286897778511:
                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.z_dr_0_0p05 > 0.8693302571773529:
                                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 57.38642883300781:
                                                                    if Q.max_dr > 0.16421835869550705:
                                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.16229012608528137:
                                                                        if Q.centroid_offset > 0.01704689208418131:
                                                                            if Q.tau21 > 0.11740279197692871:
                                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 52.979604721069336:
                                                                if Q.z_dr_0p1_0p2 > 0.08875685185194016:
                                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.width > 0.0041043872479349375:
                                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt1_dr01 > 0.7330779433250427:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.004055120283737779:
                                                                    if Q.max_dr > 0.18170872330665588:
                                                                        if Q.D2 > 2.0709742307662964:
                                                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0024438488762825727:
                                                            if Q.mass > 68.66884994506836:
                                                                if Q.centroid_offset > 0.006855611689388752:
                                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.lam2 > 0.00048094458179548383:
                                                                    if Q.max_dr > 0.1828368604183197:
                                                                        if Q.girth2 > 0.004493636079132557:
                                                                            if Q.e2 > 0.029744400642812252:
                                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 87% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 39.938560485839844:
                                                                        if Q.width > 0.004748070612549782:
                                                                            if Q.centroid_offset > 0.01223958469927311:
                                                                                if Q.max_dr > 0.18443862348794937:
                                                                                    if Q.z_4 > 0.07756931707262993:
                                                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                                                else:
                                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 25.7890625:
                                                                            return 'W'   # 94% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.sum_pt_top3 > 491.765625:
                                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 't'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.010093980934470892:
                                                                if Q.pt_7 > 21.03125:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 37.83325958251953:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 40.73057746887207:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 28.5078125:
                                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 2.9771734476089478:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 26.1484375:
                                                        if Q.centroid_offset > 0.009277474135160446:
                                                            if Q.n_dr_0p1_0p2 > 1.5:
                                                                if Q.tau21 > 0.3248380273580551:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 44% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 31.32347011566162:
                                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.014469542540609837:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_1 > 100.90625:
                                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.003052285290323198:
                                                                if Q.pt_7 > 34.140625:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 39% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 562.9609375:
                                                            if Q.D2 > 2.891038179397583:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.011594646144658327:
                                            if Q.centroid_offset > 0.013557357247918844:
                                                if Q.max_dr > 0.16264311224222183:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 22.46875:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 33.30115509033203:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 20.59375:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 983.7890625:
                                                if Q.max_dr > 0.17727217078208923:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9795301258563995:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 25.1640625:
                                                    if Q.z_7 > 0.04443824291229248:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 36.05800819396973:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt_sq > 0.001954963430762291:
                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 42.65251350402832:
                                    if Q.girth > 0.017695018090307713:
                                        if Q.dr_5 > 0.062066687270998955:
                                            if Q.D2 > 7.4204185009002686:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 49.30007553100586:
                                            return 'W'   # 41% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 13.79296875:
                                                if Q.girth > 0.016289416700601578:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00963910948485136:
                                        if Q.mass > 35.764822006225586:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011871113441884518:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.012574997264891863:
                                            return 'g'   # 32% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 91% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.0028181112138554454:
                        if Q.centroid_offset > 0.024551325477659702:
                            if Q.width > 0.005047633312642574:
                                if Q.max_dr > 0.10757903382182121:
                                    if Q.z_dr_0p1_0p2 > 0.2451564446091652:
                                        if Q.tau21 > 0.24863416701555252:
                                            if Q.mass > 34.674442291259766:
                                                if Q.C2 > 0.04800223186612129:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.043308086693286896:
                                            if Q.phi_1 > -0.0355072021484375:
                                                if Q.log_sum_pt > 6.507522344589233:
                                                    return 'g'   # 33% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6143167912960052:
                                                        return 't'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.04705122672021389:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 58% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 36.10127067565918:
                                                if Q.max_dr > 0.11825541034340858:
                                                    if Q.lam2 > 0.00034013704862445593:
                                                        if Q.C2 > 0.03490916080772877:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.phi_0 > -0.02246856689453125:
                                                                return 't'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.07205544412136078:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.029151182621717453:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 46.636226654052734:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 29.953125:
                                                    if Q.centroid_offset > 0.027523922733962536:
                                                        if Q.e2 > 0.03572582267224789:
                                                            if Q.girth2 > 0.006087720626965165:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.11665258929133415:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.29659661650657654:
                                                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.12238320335745811:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.2247818112373352:
                                                        if Q.pt_7 > 27.7109375:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03398290276527405:
                                        if Q.centroid_offset > 0.04773957096040249:
                                            if Q.log_sum_pt > 6.490281105041504:
                                                return 'g'   # 39% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.18743817508220673:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.29969531297683716:
                                                    if Q.eccentricity > 0.8314220607280731:
                                                        if Q.pt_7 > 28.0390625:
                                                            if Q.D2 > 0.4336520731449127:
                                                                if Q.log_sum_pt > 6.277494668960571:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.z_7 > 0.0724416933953762:
                                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 7.428257595165633e-05:
                                                        if Q.sum_pt > 540.4375:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.07626210898160934:
                                            if Q.max_dr > 0.09594078734517097:
                                                if Q.dr_0 > 0.08486882597208023:
                                                    if Q.planar_flow > 0.08142149820923805:
                                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9726960062980652:
                                                    if Q.LHA > 0.3141586631536484:
                                                        if Q.pt_2 > 80.9375:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.342860698699951:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.3118774741888046:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9710288345813751:
                                                if Q.centroid_offset > 0.030296881683170795:
                                                    if Q.max_dr > 0.0979674905538559:
                                                        if Q.mass > 39.36276626586914:
                                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.15241722762584686:
                                                    if Q.sum_pt > 632.703125:
                                                        if Q.centroid_offset > 0.028248989954590797:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.029006912373006344:
                                                            if Q.dr_0 > 0.073787372559309:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.109375:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 43% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03194448910653591:
                                    if Q.max_dr > 0.1209828071296215:
                                        if Q.LHA > 0.2634136527776718:
                                            if Q.z_7 > 0.034209564328193665:
                                                if Q.log_sum_pt > 6.660865068435669:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.044618481770157814:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.1307913064956665:
                                                            return 'Z'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0_0p05 > 0.11246989667415619:
                                                                if Q.centroid_offset > 0.03436649590730667:
                                                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.25535738468170166:
                                                if Q.tau21 > 0.2190234139561653:
                                                    if Q.sum_pt > 686.609375:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0001156622929556761:
                                                    return 'Z'   # 31% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9672569334506989:
                                            if Q.centroid_offset > 0.03749408572912216:
                                                if Q.max_dr > 0.1074555367231369:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 720.0625:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.13165736198425293:
                                                    if Q.z_dr_0_0p05 > 0.04732688330113888:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 40.612165451049805:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9797010719776154:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.1648697778582573:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.20362061262130737:
                                                if Q.max_dr > 0.1055169589817524:
                                                    if Q.tau21 > 0.38047677278518677:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 43.203125:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.643025159835815:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_eta > -0.02929520048201084:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 38% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.12950148433446884:
                                        if Q.LHA > 0.26668499410152435:
                                            if Q.centroid_offset > 0.026321508921682835:
                                                if Q.mass > 42.410207748413086:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02867517899721861:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 6.88007530698087e-05:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 47.30991744995117:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.06266011670231819:
                                                        if Q.max_dr > 0.13725272566080093:
                                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0037411233643069863:
                                                if Q.centroid_offset > 0.0286412900313735:
                                                    if Q.max_dr > 0.14036286622285843:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.0022977329790592194:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 5.532432987820357e-05:
                                                            if Q.C2 > 0.030486307106912136:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 7.586034189444035e-05:
                                                    if Q.tau21 > 0.17785004526376724:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.13245393335819244:
                                            if Q.tau21 > 0.16797345131635666:
                                                if Q.C2 > 0.021031946875154972:
                                                    if Q.width > 0.004823101218789816:
                                                        if Q.e2 > 0.030056177638471127:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 43% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02599805872887373:
                                                    if Q.max_dr > 0.08713381737470627:
                                                        return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.2871709316968918:
                                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.560632228851318:
                                                                return 'Z'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 51.36610794067383:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.1966920867562294:
                                                    return 'W'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 40.84660339355469:
                                                        if Q.centroid_offset > 0.028712534345686436:
                                                            if Q.max_dr > 0.1183643527328968:
                                                                if Q.LHA > 0.2777194529771805:
                                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006298912689089775:
                                if Q.e2 > 0.037772081792354584:
                                    if Q.eccentricity > 0.9686770141124725:
                                        if Q.mass > 76.14661026000977:
                                            if Q.pt_7 > 24.5859375:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 627.6875:
                                                    if Q.mean_eta2 > 0.004171241540461779:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.039114343002438545:
                                                if Q.centroid_offset > 0.012812173925340176:
                                                    if Q.girth2_top2 > 0.005699615925550461:
                                                        if Q.girth2 > 0.006605290342122316:
                                                            if Q.e2 > 0.040757978335022926:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 0.006518859416246414:
                                                            if Q.e2 > 0.03978862799704075:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 71.30450057983398:
                                                        if Q.pt_7 > 44.3125:
                                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.14131496101617813:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.11147135496139526:
                                                                if Q.mass > 65.56956100463867:
                                                                    if Q.girth2 > 0.006550885736942291:
                                                                        if Q.z_dr_0p05_0p1 > 0.6710840165615082:
                                                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0065415294375270605:
                                                    if Q.girth > 0.07351180911064148:
                                                        if Q.sum_pt_top5 > 587.375:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.013294082134962082:
                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.011931135319173336:
                                                        if Q.sum_pt_top2 > 345.875:
                                                            if Q.width > 0.00642180466093123:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 0.5326777994632721:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.006481882883235812:
                                                                    return 'Z'   # 48% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 453.375:
                                                            if Q.girth2 > 0.00640168646350503:
                                                                if Q.centroid_offset > 0.005732125137001276:
                                                                    return 'Z'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.040899449959397316:
                                            if Q.max_dr > 0.12795965373516083:
                                                if Q.pt_0 > 150.375:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.0420075748115778:
                                                        if Q.sum_pt_top2 > 193.0:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.0065585949923843145:
                                                    if Q.girth2_top3 > 0.006559106055647135:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.04248766228556633:
                                                            if Q.C2 > 0.02964585367590189:
                                                                if Q.n_dr_0_0p05 > 0.5:
                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01533029368147254:
                                                        if Q.mass > 51.03112602233887:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 0.9614301025867462:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.25216829776763916:
                                                    if Q.tau32 > 0.5201111435890198:
                                                        if Q.mass > 50.86678504943848:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 608.375:
                                                            return 'Z'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.006437958450987935:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 438.71875:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top3 > 0.006257844157516956:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.008424242027103901:
                                        if Q.n_dr_0_0p05 > 3.5:
                                            return 'W'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.489394426345825:
                                                if Q.z_dr_0p1_0p2 > 0.27723369002342224:
                                                    if Q.width > 0.006426915060728788:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 30.09375:
                                                    if Q.centroid_offset > 0.016462350264191628:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.12556981295347214:
                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 58.614999771118164:
                                            if Q.centroid_offset > 0.004008265910670161:
                                                if Q.z_dr_0p05_0p1 > 0.22822997719049454:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.834780693054199:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1059.015625:
                                    if Q.mass > 72.85363388061523:
                                        if Q.pt_7 > 16.5625:
                                            if Q.mass > 86.67378997802734:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 7.038440942764282:
                                            if Q.centroid_offset > 0.006946127396076918:
                                                if Q.eccentricity > 0.9909171462059021:
                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 69.23347854614258:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010224475525319576:
                                                if Q.mass > 64.02660751342773:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 35.28334999084473:
                                        if Q.centroid_offset > 0.016994492150843143:
                                            if Q.girth2 > 0.0054193115793168545:
                                                if Q.max_dr > 0.1195785142481327:
                                                    if Q.e2 > 0.03513326309621334:
                                                        if Q.width > 0.005875925766304135:
                                                            if Q.dr_4 > 0.04596921242773533:
                                                                if Q.z_dr_0p1_0p2 > 0.2656048387289047:
                                                                    return 't'   # 34% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.13763776421546936:
                                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.005647983169183135:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.1275429055094719:
                                                                if Q.e2 > 0.0305701345205307:
                                                                    if Q.eccentricity > 0.9606082439422607:
                                                                        if Q.width > 0.0055247575510293245:
                                                                            return 'Z'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 66% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 573.53125:
                                                                    return 'Z'   # 60% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.10806320607662201:
                                                        if Q.girth > 0.07442738488316536:
                                                            if Q.mass > 48.83810234069824:
                                                                if Q.width > 0.005996399559080601:
                                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.02300017699599266:
                                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 781.40625:
                                                                if Q.pt_4 > 66.59375:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 61.269927978515625:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00021526798082049936:
                                                                if Q.tau21 > 0.12802431732416153:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.LHA > 0.30915918946266174:
                                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.eccentricity > 0.9438982605934143:
                                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2 > 0.006080971332266927:
                                                                    if Q.e2 > 0.036417627707123756:
                                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 50.50185012817383:
                                                    if Q.max_dr > 0.12997838109731674:
                                                        if Q.lam1 > 0.004928740439936519:
                                                            if Q.centroid_offset > 0.019594410434365273:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau32 > 0.690276175737381:
                                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.max_dr > 0.13776472955942154:
                                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.02065360452979803:
                                                                if Q.width > 0.004662821535021067:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt > 907.5625:
                                                            if Q.pt_7 > 46.71875:
                                                                return 'Z'   # 43% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00011396054105716757:
                                                                if Q.tau21 > 0.12895801663398743:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.019135852344334126:
                                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.251587450504303:
                                                        if Q.pt_6 > 39.984375:
                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.13115457445383072:
                                                            if Q.girth2 > 0.0049760539550334215:
                                                                if Q.centroid_offset > 0.02023714128881693:
                                                                    if Q.e2 > 0.029751070775091648:
                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0002162093442166224:
                                                                if Q.tau21 > 0.1302337422966957:
                                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.02138673886656761:
                                                                        if Q.max_dr > 0.10743793100118637:
                                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 99% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.003194811404682696:
                                                if Q.mass > 74.85727310180664:
                                                    if Q.pt_7 > 31.1640625:
                                                        if Q.mass_top3 > 46.86880302429199:
                                                            if Q.centroid_offset > 0.003934880718588829:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.00556927127763629:
                                                            if Q.dr_1 > 0.07765050232410431:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 79.2916374206543:
                                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.13049067556858063:
                                                        if Q.girth > 0.0685935765504837:
                                                            if Q.centroid_offset > 0.011982825119048357:
                                                                if Q.girth2 > 0.0058981431648135185:
                                                                    if Q.e2 > 0.03663969971239567:
                                                                        if Q.D2 > 0.9501869976520538:
                                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 75% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt > 757.6328125:
                                                                        if Q.lam1 > 0.005640640156343579:
                                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 65% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.009308333974331617:
                                                                    if Q.mass > 62.668697357177734:
                                                                        if Q.width > 0.006009164033457637:
                                                                            return 'Z'   # 100% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.width > 0.006024746689945459:
                                                                            if Q.sum_pt_top3 > 459.09375:
                                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 71.58354187011719:
                                                                        if Q.centroid_offset > 0.005034143105149269:
                                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth > 0.06666967272758484:
                                                                if Q.centroid_offset > 0.011922187637537718:
                                                                    if Q.max_dr > 0.1414506658911705:
                                                                        if Q.z_7 > 0.05109792947769165:
                                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 83% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass > 43.94818115234375:
                                                                    return 'W'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_7 > 25.1015625:
                                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt > 680.4609375:
                                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0035682846792042255:
                                                            if Q.log_sum_pt > 6.923882484436035:
                                                                if Q.centroid_offset > 0.007620552554726601:
                                                                    if Q.pt_6 > 52.859375:
                                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.11717884242534637:
                                                                    if Q.girth > 0.071958739310503:
                                                                        if Q.e2 > 0.034664930775761604:
                                                                            if Q.centroid_offset > 0.010826254729181528:
                                                                                if Q.mass > 58.014699935913086:
                                                                                    if Q.girth2 > 0.006192982429638505:
                                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        if Q.z_dr_0_0p05 > 0.010723616927862167:
                                                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            if Q.girth2 > 0.006024503614753485:
                                                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                                                            else:
                                                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                                                else:
                                                                                    if Q.centroid_offset > 0.01476466003805399:
                                                                                        if Q.e2_sq > 0.005761699751019478:
                                                                                            if Q.max_dr > 0.12096109613776207:
                                                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                                                            else:
                                                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                                                        else:
                                                                                            return 'W'   # 100% of the training jets here get this class from the formula
                                                                                    else:
                                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 98% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_6 > 22.984375:
                                                                        return 'W'   # 100% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 48.03965187072754:
                                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 't'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.029498237185180187:
                                                                if Q.pt_7 > 44.328125:
                                                                    if Q.eccentricity > 0.9817343354225159:
                                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.007927434518933296:
                                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.00847030384466052:
                                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.mass > 43.980634689331055:
                                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.z_top5 > 0.8174013793468475:
                                                                                return 'q'   # 47% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.025554931722581387:
                                                    if Q.z_7 > 0.05489586852490902:
                                                        if Q.lam1 > 0.00295383355114609:
                                                            if Q.centroid_offset > 0.006524105789139867:
                                                                if Q.phi_1 > -0.0071392059326171875:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top5 > 29.640501976013184:
                                                                    return 'W'   # 47% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.009729116223752499:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 46.68210411071777:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.0048276681918650866:
                                                                    if Q.pt_7 > 29.3671875:
                                                                        if Q.mass > 41.58139228820801:
                                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 36% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.008127602748572826:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 42.89889717102051:
                                                            return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 35.09375:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.012573959771543741:
                                            if Q.pt_7 > 28.28125:
                                                if Q.z_dr_0p1_0p2 > 0.24032286554574966:
                                                    if Q.z_dr_0p05_0p1 > 0.14632827043533325:
                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_6 > 41.375:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.0033473974326625466:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.016219141893088818:
                                                            if Q.e2 > 0.027013693004846573:
                                                                if Q.z_top5 > 0.7638700902462006:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass > 32.82053565979004:
                                                                return 'W'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.7195527851581573:
                                                                    return 'W'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.220015048980713:
                                                    if Q.pt_7 > 23.3515625:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 32.53252983093262:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0038317227736115456:
                                                if Q.pt_7 > 28.765625:
                                                    if Q.e2 > 0.03484794311225414:
                                                        return 'W'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 0.8097005486488342:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 34.4375:
                                                                if Q.lam1 > 0.003994784550741315:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top5 > 391.1875:
                                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 33.51391410827637:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1064387857913971:
                                                    if Q.width > 0.0033326081465929747:
                                                        if Q.mass > 32.445587158203125:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top5 > 0.763177901506424:
                                                                return 'g'   # 86% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 5.203647924645338e-05:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top2 > 321.71875:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.035878755152225494:
                                                            if Q.girth2_top2 > 0.002299028681591153:
                                                                return 'W'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.051934994757175446:
                                                                if Q.girth2_top3 > 0.0040266914293169975:
                                                                    if Q.mass > 33.20019721984863:
                                                                        return 'W'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.01454621460288763:
                            if Q.pt_7 > 42.421875:
                                if Q.log_sum_pt > 6.811536550521851:
                                    if Q.planar_flow > 0.05483945831656456:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 57.890625:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p1_0p2 > 0.05191100016236305:
                                                return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.0843331515789032:
                                        if Q.lam1 > 0.002166955266147852:
                                            if Q.z_dr_0_0p05 > 0.84659543633461:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 49.0625:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.min_pair_mass > 0.8507368266582489:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.07137828320264816:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 960.4296875:
                                    if Q.planar_flow > 0.09330982342362404:
                                        if Q.pt_7 > 34.40625:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.962774991989136:
                                            return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.01663117576390505:
                                        if Q.centroid_offset > 0.02809916716068983:
                                            if Q.lam2 > 0.00019169213192071766:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 19.8046875:
                                                if Q.lam1 > 0.0021693173330277205:
                                                    return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.09673546999692917:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.25:
                                                            if Q.log_sum_pt > 6.691909074783325:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00017227828357135877:
                                                                return 'W'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.020227964036166668:
                                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 32.91314125061035:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p05_0p1 > 0.054795993492007256:
                                            if Q.pt_7 > 34.140625:
                                                if Q.e2_sq > 0.0020388260018080473:
                                                    if Q.sum_pt_top2 > 340.6875:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00014666569040855393:
                                                    return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 36.55825614929199:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 32.36418342590332:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 29.046875:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 38.484375:
                                if Q.planar_flow > 0.04630817100405693:
                                    if Q.pt_7 > 43.296875:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.979725569486618:
                                            if Q.sum_pt > 818.609375:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 397.9375:
                                                    if Q.pt1_dr01 > 1.2460168600082397:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.013331456575542688:
                                                return 'W'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top2 > 12.862561225891113:
                                                    if Q.mass > 32.31557846069336:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.10523030534386635:
                                        if Q.dr_0 > 0.02286790031939745:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 41.761919021606445:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 53.05750274658203:
                                            if Q.sum_pt > 1323.0625:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_5 > 0.07623705640435219:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 44.015625:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.810948848724365:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.002485377714037895:
                                                        return 'g'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 77.8125:
                                                            return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 42.458099365234375:
                                    if Q.lam1 > 0.0023792312713339925:
                                        if Q.centroid_offset > 0.005598719231784344:
                                            return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 47.940216064453125:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 21.9921875:
                                            if Q.max_dr > 0.12338728085160255:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 51.495521545410156:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mean_phi2 > 0.0014096794766373932:
                                                        return 'W'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 690.5:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.9110777378082275:
                                        if Q.pt_7 > 20.671875:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.004872893914580345:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 545.765625:
                                            if Q.centroid_offset > 0.010891763493418694:
                                                if Q.mass > 37.04108810424805:
                                                    if Q.z_dr_0p1_0p2 > 0.04009465128183365:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.02094913460314274:
                                                        if Q.max_dr > 0.10913436859846115:
                                                            if Q.pt_6 > 31.390625:
                                                                if Q.mass > 33.519296646118164:
                                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 47% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.03306545875966549:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 47% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 32.859375:
                                                    if Q.C2 > 0.02899025846272707:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.808588027954102:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p1_0p2 > 0.08480257168412209:
                                                                if Q.sum_pt_top2 > 382.71875:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.1181732565164566:
                                                        if Q.centroid_offset > 0.006592474645003676:
                                                            return 'W'   # 36% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_pair_mass > 15.667678833007812:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.07026323676109314:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 61% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top3 > 465.09375:
                if Q.centroid_offset > 0.01596675906330347:
                    if Q.centroid_offset > 0.025867613963782787:
                        if Q.lam1 > 0.0016755242832005024:
                            if Q.centroid_offset > 0.035405440255999565:
                                if Q.pt_7 > 28.8359375:
                                    if Q.log_sum_pt > 6.670468807220459:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04708095081150532:
                                            if Q.pt_7 > 34.75:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.003206756664440036:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.03467937000095844:
                                                if Q.width > 0.002997842733748257:
                                                    return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 24.2734375:
                                        if Q.centroid_offset > 0.04217457212507725:
                                            if Q.log_sum_pt > 6.658411741256714:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 24.90625:
                                    if Q.eccentricity > 0.9700749218463898:
                                        return 'W'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.030772512778639793:
                                            if Q.z_dr_0p1_0p2 > 0.038530582562088966:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 25.5:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.03732970915734768:
                                                return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.3188377767801285:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 740.25:
                                        if Q.centroid_offset > 0.027752167545259:
                                            if Q.z_top5 > 0.9127073287963867:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.846237421035767:
                                if Q.pt_7 > 36.453125:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.000900941202417016:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 735.7421875:
                                    if Q.centroid_offset > 0.027734292671084404:
                                        if Q.pt_5 > 74.25:
                                            if Q.z_7 > 0.04181777499616146:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_5 > 22.046875:
                                                if Q.pt_7 > 52.328125:
                                                    if Q.mass_top5 > 3.440803647041321:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.018194456584751606:
                                                        if Q.pt_7 > 38.9375:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.029742931947112083:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top2 > 437.875:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 23.2578125:
                                                            if Q.z_dr_0_0p05 > 0.9407710433006287:
                                                                if Q.sum_pt > 755.4453125:
                                                                    return 'Z'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.030093271285295486:
                                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.sum_pt_top5 > 626.171875:
                                                                            if Q.dr_6 > 0.02938284631818533:
                                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 40.78125:
                                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.658010959625244:
                                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_6 > 27.6875:
                                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.678292989730835:
                                            if Q.pt_7 > 44.921875:
                                                if Q.mass > 6.800292730331421:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 2.6892720460891724:
                                                    if Q.planar_flow > 0.06363832578063011:
                                                        if Q.lam1 > 0.0010814368724822998:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_pt_above_50 > 3.5:
                                                        if Q.girth > 0.026527058333158493:
                                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 833.9609375:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 39.96875:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 21.640625:
                                                    if Q.mass_top5 > 3.4699281454086304:
                                                        if Q.C2 > 0.01781900692731142:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.036713361740112305:
                                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 25.3984375:
                                        if Q.centroid_offset > 0.030065924860537052:
                                            if Q.log_sum_pt > 6.518951416015625:
                                                if Q.C2 > 0.017829827964305878:
                                                    if Q.eccentricity > 0.9109838902950287:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 36% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.028014598414301872:
                                                if Q.mass_top5 > 3.600900411605835:
                                                    if Q.pt_4 > 46.640625:
                                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top2 > 0.0010693537769839168:
                                                            return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 682.0859375:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 76% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.515625:
                            if Q.mass > 5.77167820930481:
                                if Q.log_sum_pt > 6.709742546081543:
                                    if Q.pt_7 > 43.359375:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 907.609375:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.01840322930365801:
                                                if Q.e2 > 0.004802468465641141:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.01923337485641241:
                                        if Q.pt_7 > 44.0625:
                                            return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.020928784273564816:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 933.109375:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.0005344080564100295:
                                        if Q.log_sum_pt > 6.692122459411621:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.017913387157022953:
                                            if Q.pt_7 > 47.75:
                                                if Q.planar_flow > 0.08457637950778008:
                                                    if Q.log_sum_pt > 6.710597515106201:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 865.9375:
                                                    if Q.LHA > 0.16027242690324783:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.977045863866806:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.661725044250488:
                                if Q.centroid_offset > 0.017542580142617226:
                                    if Q.centroid_offset > 0.023820617236196995:
                                        if Q.log_sum_pt > 6.759509086608887:
                                            if Q.D2 > 2.864166021347046:
                                                if Q.sum_pt_top5 > 837.921875:
                                                    return 'g'   # 31% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 34.203125:
                                                    if Q.mass > 6.739497423171997:
                                                        return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 798.375:
                                                        if Q.sum_pt > 927.0703125:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_5 > 0.024082759395241737:
                                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0006242167146410793:
                                                            if Q.phi_0 > 0.0031862258911132812:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top5 > 3.70734441280365:
                                                if Q.D2 > 1.7289571166038513:
                                                    if Q.dr_0 > 0.029153269715607166:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_4 > 0.026553312316536903:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.024487554095685482:
                                                        if Q.girth2_top2 > 0.0007952240121085197:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 36.21875:
                                                    if Q.log_sum_pt > 6.709550857543945:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau21 > 0.4300140142440796:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.11340466514229774:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.913791656494141:
                                            if Q.log_sum_pt > 7.003689765930176:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 1.8975230887008365e-05:
                                                    if Q.z_7 > 0.02460028324276209:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 18.9140625:
                                                if Q.mass_top5 > 5.393652439117432:
                                                    if Q.pt_7 > 34.671875:
                                                        if Q.centroid_offset > 0.02207990735769272:
                                                            return 'W'   # 42% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.04942895285785198:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_7 > 37.46875:
                                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4066944122314453:
                                                            if Q.log_sum_pt > 6.709628105163574:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.020528269931674004:
                                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.C2 > 0.018208307214081287:
                                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.02108587045222521:
                                                                if Q.girth2_top2 > 0.0008592107624281198:
                                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.pt_1 > 158.9375:
                                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.LHA > 0.18991325050592422:
                                                                    if Q.lam2 > 7.578709482913837e-05:
                                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 825.140625:
                                                        if Q.pt_7 > 36.046875:
                                                            if Q.centroid_offset > 0.02252264227718115:
                                                                if Q.D2 > 1.8613572120666504:
                                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01957786362618208:
                                                            return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.03030153177678585:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.phi_7 > 0.011295318603515625:
                                                                    return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.756282806396484:
                                                    if Q.pt_7 > 15.99609375:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.11580224335193634:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.814667224884033:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_top5 > 0.9381692111492157:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.019791338592767715:
                                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.04008466377854347:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01940335053950548:
                                                            if Q.log_sum_pt > 6.692421913146973:
                                                                if Q.z_top5 > 0.9233970046043396:
                                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.7654128074646:
                                        if Q.log_sum_pt > 6.917922019958496:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 20.5234375:
                                                if Q.pt_7 > 35.984375:
                                                    if Q.max_dr > 0.028162180446088314:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.819734811782837:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.037442704662680626:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.4140625:
                                            if Q.eccentricity > 0.9396080076694489:
                                                if Q.log_sum_pt > 6.729189157485962:
                                                    if Q.C2 > 0.006062235217541456:
                                                        return 'q'   # 45% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.7120308876037598:
                                                        if Q.max_dr > 0.07369932904839516:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.min_pair_mass > 0.5721469223499298:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 36.9375:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.703111410140991:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.6577386260032654:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.15323998779058456:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.5025286078453064:
                                                    if Q.phi_1 > -0.0018482208251953125:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 26.2578125:
                                    if Q.centroid_offset > 0.019515296444296837:
                                        if Q.log_sum_pt > 6.562962532043457:
                                            if Q.pt_7 > 29.6640625:
                                                if Q.z_dr_0p05_0p1 > 0.0455434825271368:
                                                    if Q.tau21 > 0.4795946627855301:
                                                        if Q.z_7 > 0.04506721347570419:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.021982459351420403:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.6246771812438965:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4349643737077713:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.021999279037117958:
                                                if Q.girth > 0.035187287256121635:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 28.625:
                                                        if Q.centroid_offset > 0.02341895643621683:
                                                            return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.2204877957701683:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.2817414402961731:
                                            if Q.sum_pt > 738.234375:
                                                if Q.z_top5 > 0.862909585237503:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.15173859894275665:
                                                        if Q.centroid_offset > 0.017076564021408558:
                                                            if Q.tau21 > 0.5410239100456238:
                                                                if Q.log_sum_pt > 6.626407146453857:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.021757179871201515:
                                                    if Q.sum_pt_top3 > 488.625:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 49.734375:
                                                            return 'W'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.014868366066366434:
                                                if Q.max_dr > 0.11873895302414894:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.828125:
                                                        if Q.girth2_top2 > 0.0004099830402992666:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.006053632125258446:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01869152393192053:
                                                            if Q.girth2_top3 > 0.00042490560736041516:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.021739989519119263:
                                        if Q.pt_6 > 23.4140625:
                                            if Q.log_sum_pt > 6.6035096645355225:
                                                if Q.pt_7 > 23.8046875:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth > 0.0329146571457386:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_7 > 0.035821953788399696:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 22.826176643371582:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 601.234375:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 493.78125:
                                            if Q.mass > 24.750011444091797:
                                                if Q.log_sum_pt > 6.591491460800171:
                                                    if Q.pt_7 > 21.6328125:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.014551247470080853:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 48% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.8655283749103546:
                                                if Q.dr_0 > 0.01736189890652895:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 41.640625:
                        if Q.centroid_offset > 0.00406828336417675:
                            if Q.pt_7 > 44.421875:
                                if Q.girth2 > 4.9956910515902564e-05:
                                    if Q.pt_6 > 50.421875:
                                        if Q.planar_flow > 0.04451124556362629:
                                            if Q.pt_7 > 47.234375:
                                                return 'g'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005535979522392154:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top3 > 511.53125:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_2 > 120.96875:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.783409833908081:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.007219112711027265:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 2.2289124899543822e-05:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.810088872909546:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.005020349053665996:
                                                        if Q.pt_7 > 47.890625:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.005783629138022661:
                                                                if Q.pt_4 > 61.328125:
                                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 917.78125:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 53.265625:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.006969738285988569:
                                    if Q.eccentricity > 0.9720185101032257:
                                        if Q.log_sum_pt > 6.755766153335571:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 3.8701783418655396:
                                            if Q.log_sum_pt > 6.711873531341553:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.3216201066970825:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_dispersion > 0.45294827222824097:
                                                        return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.00010469904373167083:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.863199234008789:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 2.2668604287900962e-05:
                                            if Q.pt_2 > 106.875:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01655243430286646:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.010576363187283278:
                                                        return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 429.5625:
                                                            return 'q'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 615.890625:
                                                if Q.centroid_offset > 0.005570153007283807:
                                                    if Q.pt_6 > 52.1875:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.898178339004517:
                                if Q.sum_pt > 1048.421875:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 3.4653927286854014e-05:
                                        if Q.pt_6 > 51.671875:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 5.2162742576911114e-05:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 56.3125:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0019887324888259172:
                                                if Q.m012 > 1.7139566540718079:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 6.588555812835693:
                                    if Q.pt_7 > 49.515625:
                                        if Q.lam2 > 8.933754543249961e-06:
                                            if Q.pt_7 > 53.921875:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.002647273475304246:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 8.942262649536133:
                                                        if Q.pt_6 > 54.34375:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top3 > 3.203718370059505e-05:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 58.421875:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.833586931228638:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.82938814163208:
                                            if Q.girth > 0.008602426387369633:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0025507148820906878:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 2.840730667230673e-05:
                                                if Q.C2 > 0.013079165015369654:
                                                    if Q.pt_6 > 49.71875:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.0005401922389864922:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 45.453125:
                                                        if Q.pt_5 > 59.171875:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_dispersion > 0.3867254704236984:
                                                    if Q.pt_7 > 46.515625:
                                                        if Q.centroid_offset > 0.002367021981626749:
                                                            if Q.lam2 > 1.1902250207640463e-05:
                                                                if Q.pt_5 > 56.328125:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 52.921875:
                                        if Q.girth2 > 3.897851456713397e-05:
                                            if Q.centroid_offset > 0.002035845071077347:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 938.90625:
                                                if Q.girth > 0.004702651407569647:
                                                    if Q.pt_7 > 63.03125:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 4.836001062358264e-05:
                                            if Q.pt_7 > 51.171875:
                                                if Q.centroid_offset > 0.0028765314491465688:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 99% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.006408276502043009:
                            if Q.pt_7 > 34.140625:
                                if Q.sum_pt > 919.453125:
                                    if Q.sum_pt > 983.734375:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.00012086687638657168:
                                            if Q.lam2 > 1.4807002116867807e-05:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.mean_eta2 > 0.0001390136530972086:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 37.625:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 2.196421701228246e-05:
                                        if Q.eccentricity > 0.8958697319030762:
                                            if Q.dr_0 > 0.010789146181195974:
                                                if Q.centroid_offset > 0.012951253447681665:
                                                    if Q.dr_0 > 0.020040616393089294:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 38.703125:
                                                        if Q.log_sum_pt > 6.707263231277466:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.6359274983406067:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 525.09375:
                                                if Q.pt_7 > 37.296875:
                                                    if Q.lam2 > 4.310506301408168e-05:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.008185272105038166:
                                                            if Q.pt_7 > 39.78125:
                                                                return 'g'   # 97% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_2 > 109.28125:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.027216872200369835:
                                                        if Q.centroid_offset > 0.009284109808504581:
                                                            if Q.dr_0 > 0.014153616037219763:
                                                                if Q.lam2 > 0.00010869064135476947:
                                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.009274875279515982:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007709062658250332:
                                                    if Q.dr_0 > 0.01607282366603613:
                                                        if Q.centroid_offset > 0.010816403664648533:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.5550736784934998:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.010755322873592377:
                                                        if Q.dr_1 > 0.01163407601416111:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.665703296661377:
                                                            return 'q'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 760.453125:
                                            if Q.centroid_offset > 0.01163992052897811:
                                                if Q.pt_7 > 38.671875:
                                                    if Q.sum_pt > 845.171875:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.01350352168083191:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 1.341834968116018e-05:
                                                        if Q.pt_5 > 53.453125:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.004942499566823244:
                                                    if Q.z_top5 > 0.8152085244655609:
                                                        if Q.pt_7 > 38.265625:
                                                            if Q.centroid_offset > 0.008604067843407393:
                                                                if Q.pt_dispersion > 0.4451419562101364:
                                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 4.492402791976929:
                                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_4 > 0.013948758598417044:
                                                            return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 1.609852370165754e-05:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.003617287380620837:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2_sq > 0.0001698144551482983:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.010544692631810904:
                                                    if Q.sum_pt > 733.953125:
                                                        if Q.pt_dispersion > 0.427215576171875:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 1014.0:
                                    if Q.pt_7 > 19.3359375:
                                        if Q.log_sum_pt > 6.949556589126587:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.007366833509877324:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.011458330787718296:
                                            if Q.log_sum_pt > 6.979770660400391:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1119.22265625:
                                                if Q.pt_7 > 15.5703125:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 635.015625:
                                        if Q.centroid_offset > 0.013442893046885729:
                                            if Q.log_sum_pt > 6.790575981140137:
                                                if Q.pt_7 > 19.125:
                                                    if Q.centroid_offset > 0.014530261047184467:
                                                        if Q.eccentricity > 0.9356074631214142:
                                                            if Q.log_sum_pt > 6.8370890617370605:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_top5 > 0.8761128187179565:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.843825101852417:
                                                            if Q.e2 > 0.0041864675004035234:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_phi2 > 4.681997415900696e-05:
                                                                if Q.pt_4 > 52.796875:
                                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.864660263061523:
                                                        if Q.z_7 > 0.014569487422704697:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_7 > 0.1495787426829338:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.010987499263137579:
                                                    if Q.lam2 > 0.0001237095711985603:
                                                        if Q.pt_7 > 29.6015625:
                                                            return 'W'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 843.3359375:
                                                                return 'W'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 28.5234375:
                                                            if Q.planar_flow > 0.5191508829593658:
                                                                if Q.z_7 > 0.0379809346050024:
                                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.03125951625406742:
                                                        if Q.e2 > 0.004624157911166549:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 28.5703125:
                                                if Q.log_sum_pt > 6.857784986495972:
                                                    if Q.width > 0.0001309017388848588:
                                                        if Q.mass_top5 > 4.2282774448394775:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_2 > 120.96875:
                                                                return 'g'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00010620077955536544:
                                                        if Q.dr_0 > 0.013250511139631271:
                                                            if Q.log_sum_pt > 6.7467241287231445:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.0059298144187778234:
                                                            if Q.lam2 > 2.995378690684447e-05:
                                                                if Q.sum_pt > 918.734375:
                                                                    if Q.centroid_offset > 0.008985409047454596:
                                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.01067884685471654:
                                                                        return 'q'   # 95% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.011055173352360725:
                                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                                        else:
                                                                            if Q.tau21 > 0.42681051790714264:
                                                                                return 'q'   # 91% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 650.4375:
                                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.913196474313736:
                                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.centroid_offset > 0.009345432743430138:
                                                                            return 'g'   # 57% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam1 > 0.00013530291471397504:
                                                                if Q.tau21 > 0.5234408378601074:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 1.7480879250797443e-05:
                                                    if Q.sum_pt > 948.6015625:
                                                        if Q.centroid_offset > 0.01143234595656395:
                                                            if Q.pt_7 > 20.875:
                                                                if Q.lam2 > 3.113692218903452e-05:
                                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 1.2797939777374268:
                                                                if Q.pt_7 > 22.53125:
                                                                    if Q.lam2 > 3.5326360375620425e-05:
                                                                        if Q.centroid_offset > 0.008312429301440716:
                                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.013585683889687061:
                                            if Q.eccentricity > 0.8490492701530457:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.03698194958269596:
                                                    if Q.dr_0 > 0.019598711282014847:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.011557636316865683:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 488.5625:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.026889020577073097:
                                                if Q.planar_flow > 0.27720557153224945:
                                                    if Q.centroid_offset > 0.008619886357337236:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.010302437469363213:
                                                            if Q.LHA > 0.11960667744278908:
                                                                return 'q'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_5 > 49.84375:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.008976988028734922:
                                                        if Q.sum_pt_top5 > 594.140625:
                                                            if Q.centroid_offset > 0.01090652821585536:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.00669264793396:
                                if Q.pt_7 > 23.2734375:
                                    if Q.girth2_top5 > 2.893961482186569e-05:
                                        if Q.centroid_offset > 0.0021677121985703707:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.0078125:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_4 > 0.05944441445171833:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 34.75:
                                            if Q.sum_pt_top5 > 982.984375:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1326.609375:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.001305078447330743:
                                                    if Q.sum_pt_top3 > 844.4375:
                                                        if Q.pt_7 > 26.296875:
                                                            if Q.girth2_top5 > 1.8591800653666724e-05:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 7.264068126678467:
                                        if Q.D2 > 5.163179874420166:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.0037733607459813356:
                                            if Q.pt_7 > 20.0:
                                                if Q.mass_top5 > 4.0054731369018555:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 21.1796875:
                                                if Q.mass > 9.871303081512451:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 620.109375:
                                    if Q.log_sum_pt > 6.9594948291778564:
                                        if Q.pt_7 > 27.875:
                                            if Q.girth > 0.005974895786494017:
                                                if Q.centroid_offset > 0.0022707574535161257:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 35.109375:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.001985202427022159:
                                                    if Q.pt_7 > 34.953125:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 7.306239422177896e-05:
                                                if Q.z_7 > 0.018164588138461113:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 24.0078125:
                                                    if Q.centroid_offset > 0.004032263299450278:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 100% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 37.296875:
                                            if Q.sum_pt > 991.140625:
                                                if Q.girth2_top3 > 3.68584951502271e-05:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.00505673885345459:
                                                    if Q.lam2 > 2.5773478228074964e-05:
                                                        if Q.dr_0 > 0.005821356782689691:
                                                            if Q.lam1 > 9.392572974320501e-05:
                                                                if Q.dr_0 > 0.011745356488972902:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.m012 > 2.465216279029846:
                                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 645.3359375:
                                                if Q.mass_top3 > 9.421442985534668:
                                                    if Q.sum_pt > 991.73828125:
                                                        if Q.pt_7 > 25.9453125:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 31.5234375:
                                                        if Q.log_sum_pt > 6.906066417694092:
                                                            if Q.girth2_top5 > 7.008992542978376e-05:
                                                                if Q.centroid_offset > 0.00400770315900445:
                                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005106325261294842:
                                                    if Q.dr_0 > 0.007246462162584066:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.017863412387669086:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.009273812174797058:
                                        if Q.planar_flow > 0.5441637635231018:
                                            if Q.centroid_offset > 0.0048460159450769424:
                                                if Q.z_7 > 0.04688395373523235:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.003907300648279488:
                                            if Q.z_7 > 0.03358414024114609:
                                                if Q.lam2 > 8.583926046412671e-06:
                                                    if Q.dr_0 > 0.00683476310223341:
                                                        if Q.sum_pt_top5 > 605.75:
                                                            if Q.sum_pt_top3 > 489.25:
                                                                return 'q'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 572.078125:
                                                if Q.pt_4 > 33.390625:
                                                    if Q.lam2 > 1.871917902462883e-05:
                                                        if Q.sum_pt_top5 > 599.25:
                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.04232978820800781:
                                                                if Q.girth2_top2 > 3.5093617043457925e-05:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.phi_1 > -0.00038063526153564453:
                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 8.453503131866455:
                                                    return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
            else:
                if Q.sum_pt_top3 > 395.734375:
                    if Q.centroid_offset > 0.01910274662077427:
                        if Q.centroid_offset > 0.02864647377282381:
                            if Q.max_dr > 0.057816244661808014:
                                if Q.mass > 21.68713665008545:
                                    if Q.centroid_offset > 0.03793897293508053:
                                        if Q.girth2 > 0.004490741062909365:
                                            if Q.sum_pt > 630.1953125:
                                                return 'g'   # 41% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 25.3125:
                                                if Q.eccentricity > 0.9738811552524567:
                                                    if Q.max_dr > 0.12347596511244774:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.040224168449640274:
                                                            return 'W'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.001404973678290844:
                                                        return 'Z'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 23.703125:
                                            if Q.eccentricity > 0.9859784841537476:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.22127988189458847:
                                                    if Q.lam1 > 0.0019353878451511264:
                                                        if Q.centroid_offset > 0.03548573888838291:
                                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p05_0p1 > 0.636655867099762:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 55% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04686555080115795:
                                        if Q.z_7 > 0.044060004875063896:
                                            return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 534.546875:
                                                if Q.mass_top5 > 3.4645644426345825:
                                                    return 'g'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 29.96875:
                                            if Q.C2 > 0.021747938357293606:
                                                if Q.centroid_offset > 0.03250824846327305:
                                                    if Q.tau21 > 0.365475669503212:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.3091766834259033:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.626096487045288:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_4 > 44.53125:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.480248212814331:
                                                if Q.z_7 > 0.03686991147696972:
                                                    if Q.mass_top5 > 3.005908489227295:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 38% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 29.7734375:
                                    if Q.pt_5 > 72.46875:
                                        if Q.mass > 5.039375066757202:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0014116681413725019:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04320298880338669:
                                            if Q.mean_phi > -0.016416160855442286:
                                                if Q.centroid_offset > 0.04515138082206249:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.473649740219116:
                                                if Q.centroid_offset > 0.031631600111722946:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 38.25:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9657140374183655:
                                                            if Q.sum_pt_top5 > 582.46875:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.20361533015966415:
                                                                return 'W'   # 46% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.001243098988197744:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.030980320647358894:
                                                        return 'W'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 534.109375:
                                        if Q.pt_7 > 25.4296875:
                                            if Q.sum_pt > 687.9140625:
                                                if Q.dr_3 > 0.03832251578569412:
                                                    return 'Z'   # 37% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.19971781224012375:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_2 > 111.28125:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_over_sum_pt > 0.00707262777723372:
                                                        if Q.mass_over_sum_pt > 0.012994097080081701:
                                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 3.442238084971905e-05:
                                                return 'g'   # 31% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                        else:
                            if Q.eccentricity > 0.9717588424682617:
                                if Q.pt_7 > 26.65625:
                                    if Q.pt_7 > 48.453125:
                                        if Q.mass_top5 > 3.883767247200012:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.024925583973526955:
                                                if Q.sum_pt_top5 > 580.09375:
                                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 664.3046875:
                                            if Q.centroid_offset > 0.027200250886380672:
                                                if Q.sum_pt > 724.953125:
                                                    if Q.pt_7 > 39.796875:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.0827602855861187:
                                                    if Q.C2 > 0.006076843943446875:
                                                        if Q.LHA > 0.17189081758260727:
                                                            return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.18818449974060059:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 567.90625:
                                        if Q.z_top5 > 0.8501749932765961:
                                            return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.18573378771543503:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 48.546875:
                                    if Q.mass > 4.79403281211853:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.38330167531967163:
                                            return 'W'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 680.921875:
                                        if Q.mass_top5 > 4.253896713256836:
                                            if Q.pt_7 > 41.34375:
                                                if Q.D2 > 1.1454524993896484:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.023949788883328438:
                                                        return 'Z'   # 40% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.102165937423706:
                                                    if Q.centroid_offset > 0.02179075963795185:
                                                        if Q.mean_phi > -0.006360169500112534:
                                                            if Q.phi_0 > 0.02109527587890625:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.eta_7 > -0.01839447021484375:
                                                                    if Q.mass_top5 > 5.254673004150391:
                                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.573352575302124:
                                                            if Q.mean_eta2 > 0.0003045348566956818:
                                                                return 'g'   # 49% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.02601609844714403:
                                                        return 'Z'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.021194705739617348:
                                                if Q.z_top5 > 0.8555618226528168:
                                                    return 'q'   # 25% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 44.203125:
                                                        if Q.centroid_offset > 0.026084343902766705:
                                                            return 'Z'   # 49% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.588647127151489:
                                                    if Q.max_dr > 0.02768534515053034:
                                                        if Q.dr_7 > 0.018748493865132332:
                                                            return 'W'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.020529617555439472:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.035343265160918236:
                                            if Q.pt_7 > 26.9296875:
                                                if Q.mass > 23.796175003051758:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 505.453125:
                                                        if Q.tau21 > 0.2811902165412903:
                                                            if Q.LHA > 0.2116497904062271:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.4470579624176025:
                                                    return 'W'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.4198777377605438:
                                                if Q.log_sum_pt > 6.487779140472412:
                                                    if Q.centroid_offset > 0.023117555305361748:
                                                        if Q.pt_7 > 31.8046875:
                                                            if Q.pt_2 > 98.25:
                                                                return 'g'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top5 > 4.653017044067383:
                                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 649.546875:
                                                    if Q.centroid_offset > 0.021252033300697803:
                                                        if Q.z_dr_0_0p05 > 0.9392130672931671:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.003971492638811469:
                            if Q.eccentricity > 0.9304787218570709:
                                if Q.pt_7 > 41.890625:
                                    if Q.centroid_offset > 0.006845457013696432:
                                        if Q.planar_flow > 0.07097247987985611:
                                            if Q.pt_7 > 46.703125:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.010083307977765799:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.019765449687838554:
                                                        return 'q'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.017616071738302708:
                                                return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 46.34375:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.01128931250423193:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 48.328125:
                                            if Q.planar_flow > 0.07711131125688553:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 21.10092544555664:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.016316687688231468:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 9.120710274146404e-06:
                                                    return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 9.892611980438232:
                                        if Q.dr_0 > 0.01077683875337243:
                                            if Q.z_dr_0p1_0p2 > 0.0448638629168272:
                                                if Q.z_7 > 0.05088425241410732:
                                                    if Q.centroid_offset > 0.01541276415809989:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 3.0754757062823046e-05:
                                                        if Q.dr_0 > 0.022332487627863884:
                                                            if Q.max_dr > 0.13056374341249466:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.016080106608569622:
                                                    if Q.z_7 > 0.05440320633351803:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.012657229322940111:
                                                            if Q.sum_pt_top5 > 524.25:
                                                                return 'q'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.017936497926712036:
                                                        if Q.centroid_offset > 0.012185633182525635:
                                                            if Q.sum_pt_top5 > 522.578125:
                                                                if Q.z_top5 > 0.8105741441249847:
                                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.eccentricity > 0.9716029167175293:
                                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 65% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.06461839005351067:
                                                                if Q.pt_7 > 39.734375:
                                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam2 > 4.462482320377603e-05:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_7 > 0.055567849427461624:
                                                                if Q.centroid_offset > 0.006443392485380173:
                                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top5 > 549.90625:
                                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.008093221113085747:
                                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 616.25:
                                                return 'q'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 737.0703125:
                                            if Q.centroid_offset > 0.010741015430539846:
                                                if Q.z_7 > 0.04285556823015213:
                                                    if Q.centroid_offset > 0.01757385302335024:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.002611433621495962:
                                                            return 'g'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.00569448689930141:
                                                if Q.z_7 > 0.04325907491147518:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 694.265625:
                                                        if Q.dr_0 > 0.015189365018159151:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.050507474690675735:
                                    if Q.width > 3.821169048023876e-05:
                                        if Q.D2 > 1.1383823156356812:
                                            if Q.z_7 > 0.05700008198618889:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.005022573517635465:
                                                    if Q.dr_0 > 0.017701241187751293:
                                                        if Q.centroid_offset > 0.008903241250663996:
                                                            return 'g'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.8026758432388306:
                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.width > 5.7864648624672554e-05:
                                                            if Q.planar_flow > 0.4060879498720169:
                                                                return 'g'   # 99% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.006982064805924892:
                                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.dr_0 > 0.01167667517438531:
                                                                        return 'q'   # 52% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.614872217178345:
                                                        if Q.eccentricity > 0.6687133610248566:
                                                            if Q.dr_7 > 0.012542226817458868:
                                                                return 'g'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.011809705290943384:
                                                            return 'q'   # 53% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0070539843291044235:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 48.046875:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.013697541784495115:
                                                        if Q.centroid_offset > 0.005406948504969478:
                                                            if Q.eccentricity > 0.840169221162796:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_pt_above_50 > 7.5:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 759.765625:
                                        if Q.centroid_offset > 0.008681373670697212:
                                            if Q.z_7 > 0.04301144927740097:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 2.683964976313291e-05:
                                                    if Q.dr_0 > 0.015306615736335516:
                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 7.809392991475761e-05:
                                                if Q.z_7 > 0.042743291705846786:
                                                    if Q.dr_0 > 0.007259915350005031:
                                                        if Q.pt_dispersion > 0.39141401648521423:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 627.34375:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.005576937226578593:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 9.915812492370605:
                                            if Q.planar_flow > 0.5130250453948975:
                                                if Q.centroid_offset > 0.007695483509451151:
                                                    if Q.dr_0 > 0.0202370872721076:
                                                        if Q.z_7 > 0.0408290158957243:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.013413225300610065:
                                                        if Q.sum_pt_top3 > 432.1875:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.01358628086745739:
                                                    if Q.max_dr > 0.08358057960867882:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 540.46875:
                                                            if Q.centroid_offset > 0.01535707339644432:
                                                                if Q.z_7 > 0.043624019250273705:
                                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt_top2 > 338.5625:
                                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.009416094981133938:
                                                                        if Q.planar_flow > 0.35801415145397186:
                                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'q'   # 86% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.011333493515849113:
                                                                return 'g'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.dr_0 > 0.020334296859800816:
                                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 570.296875:
                                                        if Q.centroid_offset > 0.006428764201700687:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.005297530442476273:
                                                if Q.z_7 > 0.038072213530540466:
                                                    return 'g'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.549516677856445:
                                                        if Q.lam2 > 2.6316000003134832e-05:
                                                            return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 593.828125:
                                                    if Q.dr_0 > 0.005570894805714488:
                                                        return 'q'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.z_7 > 0.06254271045327187:
                                if Q.width > 4.310017357056495e-05:
                                    if Q.D2 > 1.1926871538162231:
                                        if Q.centroid_offset > 0.0020559453405439854:
                                            if Q.lam2 > 1.0424815627629869e-05:
                                                if Q.dr_0 > 0.013877423945814371:
                                                    if Q.pt_7 > 50.765625:
                                                        return 'g'   # 100% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.007343450328335166:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.06692254543304443:
                                                if Q.lam2 > 9.314952421846101e-06:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.0047783765476197:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 1.4880741673550801e-05:
                                                    if Q.D2 > 1.4249672889709473:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 53.078125:
                                            if Q.pt_4 > 80.625:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 1.8275464753969572e-05:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_dispersion > 0.38568317890167236:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.01186341606080532:
                                                return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0015674029709771276:
                                                    if Q.mean_phi > 0.0007407140510622412:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.0023385758977383375:
                                        if Q.girth2 > 2.9353145691857208e-05:
                                            if Q.z_7 > 0.07364313676953316:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.625137805938721:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 573.21875:
                                    if Q.z_7 > 0.05326417274773121:
                                        if Q.girth2 > 4.4149437599116936e-05:
                                            if Q.dr_0 > 0.004980945028364658:
                                                if Q.pt_7 > 49.859375:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.0168072571977973:
                                                        if Q.C2 > 0.021754691377282143:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.0036413645138964057:
                                                            if Q.girth2_top2 > 6.938026490388438e-05:
                                                                return 'q'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.002141023986041546:
                                                    if Q.log_sum_pt > 6.6356799602508545:
                                                        if Q.girth2 > 7.861084668547846e-05:
                                                            return 'g'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.612293004989624:
                                                        return 'q'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.57944655418396:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.004939224570989609:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0019351771916262805:
                                                    if Q.lam1 > 3.5204704545321874e-05:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.dr_0 > 0.010350127704441547:
                                        if Q.D2 > 1.729086995124817:
                                            if Q.phi_1 > 0.003185272216796875:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 3.78206095774658e-05:
                                            if Q.dr_0 > 0.00648145517334342:
                                                if Q.centroid_offset > 0.0023059555096551776:
                                                    if Q.lam2 > 2.028069502557628e-05:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_7 > 0.050375230610370636:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.059643346816301346:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_5 > 0.06332112103700638:
                                                            return 'q'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 561.34375:
                                                    if Q.centroid_offset > 0.0020696745486930013:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.003561468329280615:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.0263841450214386:
                                                        if Q.planar_flow > 0.47595740854740143:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.002883990644477308:
                                                                return 'g'   # 100% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top2 > 3.0257975595304742e-05:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 544.71875:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 55% of the training jets here get this class from the formula
                else:
                    if Q.centroid_offset > 0.023215947672724724:
                        if Q.mass > 24.064191818237305:
                            if Q.girth2 > 0.004747697617858648:
                                if Q.tau21 > 0.29780687391757965:
                                    if Q.log_sum_pt > 6.028805255889893:
                                        if Q.centroid_offset > 0.04750493913888931:
                                            if Q.pt_7 > 36.53125:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.4020244926214218:
                                                    if Q.sum_pt_top5 > 400.78125:
                                                        return 't'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.0078125:
                                                if Q.D2 > 1.5512736439704895:
                                                    if Q.z_dr_0p05_0p1 > 0.4493129104375839:
                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.1278320476412773:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 34.25:
                                            if Q.dr_2 > 0.08021937683224678:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.12176765128970146:
                                                    return 'g'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04681120440363884:
                                        if Q.e2 > 0.034638622775673866:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.0520580280572176:
                                                return 't'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.phi_1 > -0.0285186767578125:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 49% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 31.625:
                                            if Q.width > 0.0060400268994271755:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p05_0p1 > 0.4300507754087448:
                                                    if Q.centroid_offset > 0.03106084279716015:
                                                        if Q.pt_7 > 33.75:
                                                            return 'Z'   # 60% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 28.45719051361084:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 28.72346019744873:
                                                if Q.e2 > 0.03402787819504738:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.22680512070655823:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9309608638286591:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03452405892312527:
                                    if Q.max_dr > 0.12696631997823715:
                                        if Q.max_dr > 0.17204953730106354:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.2544267922639847:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 35% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 9.790135300136171e-05:
                                            if Q.tau21 > 0.22618641704320908:
                                                if Q.centroid_offset > 0.037547649815678596:
                                                    if Q.min_pair_mass > 0.5194680094718933:
                                                        if Q.mass > 27.538954734802246:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mean_phi > 0.008929971605539322:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 39% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.1363430619239807:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 31.9609375:
                                                    if Q.centroid_offset > 0.04532588832080364:
                                                        return 'g'   # 41% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 35% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.04488410986959934:
                                                return 'Z'   # 48% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.14515338838100433:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 26.84397792816162:
                                        if Q.max_dr > 0.1665225401520729:
                                            if Q.max_dr > 0.18176688253879547:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.24169060587882996:
                                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 28.6015625:
                                                if Q.z_dr_0p1_0p2 > 0.17353734374046326:
                                                    return 'g'   # 46% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 50.15625:
                                                        if Q.lam1 > 0.0024881282588467:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.030445050448179245:
                                                            if Q.max_dr > 0.14542841166257858:
                                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.tau21 > 0.1586832031607628:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.lam2 > 7.085272955009714e-05:
                                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.00022750904463464394:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 544.8125:
                                            if Q.lam2 > 0.00010315282270312309:
                                                if Q.pt_7 > 44.046875:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.07355661690235138:
                                                        if Q.pt_6 > 33.234375:
                                                            if Q.z_7 > 0.06142290122807026:
                                                                if Q.dr_0 > 0.040579721331596375:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 31.3828125:
                                                if Q.mean_phi > 0.009996695909649134:
                                                    if Q.planar_flow > 0.10743096843361855:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.04615280590951443:
                                                        if Q.C2 > 0.027324811555445194:
                                                            return 'W'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.10649577900767326:
                                                                return 'g'   # 52% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 25.0129976272583:
                                                            if Q.lam2 > 0.00018927717610495165:
                                                                return 'g'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.3833746910095215:
                                if Q.centroid_offset > 0.030531599186360836:
                                    if Q.centroid_offset > 0.045311134308576584:
                                        if Q.pt_7 > 32.640625:
                                            if Q.n_dr_0_0p05 > 4.5:
                                                if Q.z_5 > 0.09963130950927734:
                                                    return 'g'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.m012 > 1.4946443438529968:
                                                        return 'g'   # 75% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_eta > -0.0136718126013875:
                                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top3 > 0.0044618414249271154:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top3 > 1.6526456475257874:
                                            if Q.z_5 > 0.09507372602820396:
                                                if Q.max_dr > 0.04197024181485176:
                                                    if Q.D2 > 1.1369296312332153:
                                                        if Q.pt_5 > 64.84375:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.036499619483947754:
                                                                if Q.max_dr > 0.05269490368664265:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.002091921865940094:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_pt_above_50 > 7.5:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 5.007521867752075:
                                                        return 'g'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.016265736892819405:
                                                    if Q.planar_flow > 0.5594079494476318:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_dispersion > 0.3857952505350113:
                                                            if Q.e2_sq > 0.0006994019204284996:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 41% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.84375:
                                                        if Q.mean_phi2 > 0.000135537120513618:
                                                            if Q.sum_pt > 610.671875:
                                                                if Q.max_dr > 0.05044964887201786:
                                                                    if Q.pt_2 > 99.28125:
                                                                        if Q.D2 > 1.2779818773269653:
                                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 53% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.D2 > 1.3739420771598816:
                                                                            if Q.pt_dispersion > 0.4015302360057831:
                                                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                                                            else:
                                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.girth2_top2 > 0.0012476597912609577:
                                                                    if Q.tau21 > 0.5015529096126556:
                                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'Z'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.C2 > 0.00656413147225976:
                                                                return 'g'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.2176546454429626:
                                                            return 'g'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 32.03125:
                                                if Q.z_5 > 0.10649678856134415:
                                                    if Q.e2 > 0.003608852857723832:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 47% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.017285761423408985:
                                                        if Q.lam2 > 4.2446099541848525e-05:
                                                            if Q.pt_dispersion > 0.3833579272031784:
                                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.03220893442630768:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.427901268005371:
                                                                if Q.pt_7 > 38.71875:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 58% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.33467763662338257:
                                                    if Q.sum_pt > 647.984375:
                                                        return 'Z'   # 36% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 48% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.08204999566078186:
                                        if Q.tau21 > 0.36874084174633026:
                                            if Q.e2 > 0.00429104664362967:
                                                if Q.LHA > 0.19174925237894058:
                                                    if Q.z_7 > 0.06363732740283012:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.481696844100952:
                                                            if Q.mass_top5 > 4.256057977676392:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 546.75:
                                                        if Q.mass_top5 > 4.165947437286377:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 641.125:
                                                    if Q.pt_7 > 49.71875:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_over_sum_pt > 0.006652528652921319:
                                                            if Q.girth2_top3 > 0.0008444087870884687:
                                                                return 'W'   # 73% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mean_eta2 > 0.0003511827817419544:
                                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.03822081722319126:
                                                if Q.z_5 > 0.09604111313819885:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.02786578331142664:
                                                    if Q.e2 > 0.012520749121904373:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 620.9609375:
                                                        if Q.z_7 > 0.06673384830355644:
                                                            return 'g'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.phi_0 > -0.0064945220947265625:
                                                                if Q.log_sum_pt > 6.475284814834595:
                                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 624.5625:
                                            if Q.mass_top5 > 3.040535092353821:
                                                if Q.mean_phi2 > 0.00016548967687413096:
                                                    if Q.z_7 > 0.06650744006037712:
                                                        if Q.log_sum_pt > 6.523057460784912:
                                                            return 'Z'   # 42% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.6131720542907715:
                                                    return 'W'   # 36% of the training jets here get this class from the formula
                                                else:
                                                    if Q.m012 > 1.331202208995819:
                                                        if Q.sum_pt_top5 > 526.90625:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.LHA > 0.186237633228302:
                                                                return 'W'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.0014006219571456313:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.006298965774476528:
                                    if Q.sum_pt_top5 > 277.015625:
                                        if Q.z_7 > 0.06783299148082733:
                                            if Q.width > 0.008286431431770325:
                                                return 't'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.31012317538261414:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 529.765625:
                                        if Q.centroid_offset > 0.03188561834394932:
                                            if Q.pt_7 > 40.015625:
                                                if Q.width > 0.002318647922948003:
                                                    if Q.tau21 > 0.345604345202446:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top3 > 1.9637500047683716:
                                                            return 'g'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9737970530986786:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03461980260908604:
                                                        if Q.C2 > 0.01173302624374628:
                                                            if Q.mass_top3 > 1.7068860530853271:
                                                                if Q.C2 > 0.013767002150416374:
                                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_4 > 0.1098596639931202:
                                                                if Q.mean_phi2 > 0.0003441945882514119:
                                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.0703965350985527:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.3032606691122055:
                                                    if Q.mass > 3.4447619915008545:
                                                        if Q.mass > 17.98184585571289:
                                                            if Q.lam2 > 8.940505358623341e-05:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 23.4609375:
                                                                if Q.tau21 > 0.3961821645498276:
                                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.centroid_offset > 0.04513924941420555:
                                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                                    else:
                                                                        if Q.pt_7 > 36.328125:
                                                                            return 'Z'   # 54% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.21866122633218765:
                                                            if Q.max_dr > 0.05131139047443867:
                                                                return 't'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9787485599517822:
                                                        if Q.centroid_offset > 0.046949904412031174:
                                                            return 'g'   # 40% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.m012 > 1.370509922504425:
                                                            if Q.pt_7 > 38.25:
                                                                return 'Z'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_2 > 78.09375:
                                                                return 'g'   # 48% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.32636840641498566:
                                                return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.mean_phi > -0.0115941665135324:
                                                    if Q.centroid_offset > 0.025241785682737827:
                                                        if Q.mass_top3 > 0.9606982469558716:
                                                            return 'g'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top2 > 0.0013753834064118564:
                                                        if Q.centroid_offset > 0.02668329607695341:
                                                            return 'W'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 21.516201972961426:
                                            if Q.mass_top5 > 3.8687989711761475:
                                                if Q.centroid_offset > 0.035596514120697975:
                                                    if Q.girth2 > 0.0041635066736489534:
                                                        if Q.C2 > 0.03327836096286774:
                                                            return 'g'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_1 > 0.08315467461943626:
                                                                return 'g'   # 96% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.005044281017035246:
                                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9630885422229767:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.546875:
                                                    if Q.phi_1 > -0.01665496826171875:
                                                        return 'g'   # 49% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 42.296875:
                                                if Q.centroid_offset > 0.03497806191444397:
                                                    if Q.girth > 0.04811259172856808:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau21 > 0.4217544198036194:
                                                            if Q.log_sum_pt > 6.249099969863892:
                                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam1 > 0.005080120638012886:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.037215154618024826:
                                                        if Q.tau21 > 0.2564883977174759:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_top3 > 1.7150766253471375:
                                                                return 'g'   # 98% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.03737592697143555:
                                                                    if Q.pt_7 > 32.109375:
                                                                        if Q.sum_pt_top5 > 350.078125:
                                                                            return 'Z'   # 36% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.003055960522033274:
                            if Q.mass > 27.450894355773926:
                                if Q.centroid_offset > 0.01806551869958639:
                                    if Q.pt_7 > 29.0234375:
                                        if Q.D2 > 0.7459652721881866:
                                            if Q.max_dr > 0.17838674783706665:
                                                return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p1_0p2 > 0.2129024937748909:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 49.921875:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.08252441510558128:
                                                            return 'W'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_1 > 0.05382169969379902:
                                                                return 'W'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02171616442501545:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.019284998066723347:
                                                    if Q.dr_2 > 0.05525483377277851:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.2326829507946968:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 54% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.0038152706110849977:
                                        if Q.pt_7 > 30.0703125:
                                            if Q.tau32 > 0.47794605791568756:
                                                if Q.e2 > 0.03657182864844799:
                                                    if Q.max_dr > 0.12520373612642288:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 29.794666290283203:
                                                        return 'W'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.02714244555681944:
                                                            if Q.dr_0 > 0.04438212886452675:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 37.234375:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 18.772422790527344:
                                                        if Q.tau21 > 0.36948496103286743:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.015271486714482307:
                                            if Q.max_dr > 0.12061291933059692:
                                                if Q.planar_flow > 0.12860094755887985:
                                                    if Q.girth2_top3 > 0.0009480576263740659:
                                                        return 'W'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 497.640625:
                                                if Q.D2 > 0.734001100063324:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 41.921875:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 0.9887267649173737:
                                    if Q.mass > 25.177520751953125:
                                        if Q.centroid_offset > 0.019220713526010513:
                                            if Q.z_dr_0_0p05 > 0.9085613489151001:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 34.578125:
                                                    if Q.girth2_top5 > 0.003006494021974504:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mean_phi > 0.0074494448490440845:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.min_pair_mass > 0.5568247735500336:
                                                                return 'g'   # 76% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth > 0.05575278587639332:
                                                if Q.pt_7 > 31.5625:
                                                    if Q.girth2_top2 > 0.0025709806941449642:
                                                        if Q.e2_sq > 0.00412596226669848:
                                                            return 'W'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 516.984375:
                                            if Q.centroid_offset > 0.020150144584476948:
                                                if Q.e2 > 0.0033213242422789335:
                                                    if Q.sum_pt_top5 > 539.171875:
                                                        if Q.pt_7 > 48.234375:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt > 0.007964398711919785:
                                                                if Q.dr_3 > 0.024944709613919258:
                                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 50% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.4714842438697815:
                                                    if Q.girth2 > 4.082487248524558e-05:
                                                        if Q.z_7 > 0.04760606959462166:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.933322548866272:
                                                                if Q.dr_7 > 0.020980087108910084:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_4 > 84.09375:
                                                            return 'q'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 41.984375:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.eccentricity > 0.9115206599235535:
                                                            if Q.dr_0 > 0.015022935345768929:
                                                                if Q.centroid_offset > 0.008029411546885967:
                                                                    if Q.z_7 > 0.04946567863225937:
                                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.005101234884932637:
                                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.centroid_offset > 0.004350256407633424:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 52% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 349.859375:
                                                if Q.D2 > 1.1189388632774353:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.004961992148309946:
                                                        if Q.planar_flow > 0.22013608366250992:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 373.625:
                                                                return 'q'   # 53% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.015987548977136612:
                                                            return 'q'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00576383457519114:
                                        if Q.sum_pt_top3 > 348.46875:
                                            if Q.centroid_offset > 0.011717356275767088:
                                                if Q.centroid_offset > 0.01974630542099476:
                                                    if Q.z_4 > 0.09379870817065239:
                                                        if Q.z_7 > 0.06773031875491142:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.tau21 > 0.23336020857095718:
                                                                return 'g'   # 56% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.1430084928870201:
                                                    if Q.eccentricity > 0.9053470194339752:
                                                        if Q.z_7 > 0.06356095895171165:
                                                            return 'g'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.052405839785933495:
                                                                return 'g'   # 87% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.008360379841178656:
                                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top5 > 0.7521249055862427:
                                                        if Q.dr_0 > 0.023782656528055668:
                                                            if Q.mass_over_sum_pt_sq > 0.0017153649241663516:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 83% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 323.3125:
                                            if Q.pt_7 > 50.671875:
                                                if Q.pt_6 > 61.375:
                                                    return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.02144591510295868:
                                                        return 'q'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.017317602410912514:
                                                    if Q.max_dr > 0.0641746073961258:
                                                        if Q.lam2 > 3.9566490158904344e-05:
                                                            return 'g'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top3 > 351.59375:
                                                            return 'q'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.eccentricity > 0.9421156346797943:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.0038609150797128677:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 291.296875:
                                                if Q.lam2 > 1.7456583918828983e-05:
                                                    if Q.dr_0 > 0.026281189173460007:
                                                        if Q.mass_over_sum_pt > 0.0422525480389595:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.planar_flow > 0.21417837589979172:
                                                                return 'g'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_7 > 0.07528982684016228:
                                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 1.0841488242149353:
                                if Q.sum_pt_top5 > 535.890625:
                                    if Q.width > 4.214576802041847e-05:
                                        if Q.z_7 > 0.06542716920375824:
                                            if Q.lam2 > 1.0019130058935843e-05:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.0068257341627031565:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.3833845257759094:
                                                if Q.sum_pt > 736.75:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.planar_flow > 0.42223721742630005:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 3.103532799286768e-05:
                                            if Q.dr_0 > 0.004283533431589603:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_7 > 0.00808615144342184:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.008104179985821247:
                                        if Q.D2 > 1.2451140880584717:
                                            if Q.z_7 > 0.03024839237332344:
                                                if Q.width > 3.630065839388408e-05:
                                                    if Q.eccentricity > 0.9219814538955688:
                                                        if Q.sum_pt_top5 > 463.953125:
                                                            if Q.D2 > 1.4713897109031677:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.pt_dispersion > 0.39702609181404114:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.dr_0 > 0.013205731753259897:
                                                            if Q.sum_pt_top5 > 459.625:
                                                                if Q.z_7 > 0.06770012900233269:
                                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.D2 > 1.3457916378974915:
                                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 98% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.494793176651001:
                                                        if Q.centroid_offset > 0.002229624311439693:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top5 > 506.359375:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 44% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 310.78125:
                                                if Q.dr_0 > 0.010812914464622736:
                                                    if Q.z_7 > 0.07317142933607101:
                                                        if Q.centroid_offset > 0.001555786409880966:
                                                            return 'g'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 265.78125:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.phi_2 > 0.000462263822555542:
                                                                return 'g'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mean_phi > 0.00038356824370566756:
                                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top3 > 341.375:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top3 > 293.75:
                                    if Q.z_7 > 0.07684792578220367:
                                        if Q.dr_0 > 0.014453074894845486:
                                            if Q.C2 > 0.014326899778097868:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.0025558475172147155:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.0644182413816452:
                                            if Q.z_7 > 0.06015482172369957:
                                                return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.01361513789743185:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.3037557303905487:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 488.875:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 273.65625:
                                        if Q.max_dr > 0.05433087609708309:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.02152117621153593:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 90% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
