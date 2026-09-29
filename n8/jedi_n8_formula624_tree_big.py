"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 64.70% (the formula: 65.58%); same class as the formula for 92.27% of jets.  975 leaves, depth 18.
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
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_top5_slots=sum(pt[:5]) / tot,
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_1=phi[1],
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
    if Q.girth2 > 0.00906778173521161:
        if Q.width > 0.0096655348315835:
            if Q.sum_pt > 883.203125:
                if Q.log_sum_pt > 6.900636196136475:
                    return 'g'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 38.484375:
                        if Q.planar_flow > 0.3714803159236908:
                            return 't'   # 90% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 46.640625:
                                return 'g'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.010264887940138578:
                                    return 'g'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 71% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.03703739680349827:
                            if Q.lam1 > 0.028369351290166378:
                                return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                return 't'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.eccentricity > 0.9848182201385498:
                                return 'q'   # 56% of the training jets here get this class from the formula
                            else:
                                return 't'   # 84% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 360.234375:
                    if Q.lam1 > 0.033669132739305496:
                        if Q.pt_7 > 40.390625:
                            if Q.tau21 > 0.14220944792032242:
                                return 't'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 76% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.08339220657944679:
                                if Q.tau21 > 0.05465678684413433:
                                    return 't'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.49963007867336273:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 51% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 42.91367149353027:
                            if Q.pt_7 > 49.953125:
                                if Q.sum_pt > 775.6875:
                                    if Q.lam2 > 0.0011623288155533373:
                                        return 't'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.17807049304246902:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 823.953125:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.026477103121578693:
                                        if Q.tau21 > 0.1268257051706314:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.009997160639613867:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.024724353104829788:
                                                return 't'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.717549562454224:
                                    if Q.pt_7 > 40.046875:
                                        if Q.lam2 > 0.0002762141957646236:
                                            return 't'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.02136851940304041:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.20228908210992813:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.03234642557799816:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 46% of the training jets here get this class from the formula
                                else:
                                    if Q.width > 0.010241396725177765:
                                        if Q.lam2 > 0.0005846793646924198:
                                            return 't'   # 100% of the training jets here get this class from the formula
                                        else:
                                            if Q.LHA > 0.4468107968568802:
                                                if Q.e2 > 0.06913314014673233:
                                                    if Q.pt_7 > 44.328125:
                                                        if Q.mass > 115.9941635131836:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 36.109375:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.05489739216864109:
                                            if Q.e2 > 0.04806985892355442:
                                                if Q.centroid_offset > 0.018157752230763435:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_5 > 41.109375:
                                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 98% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.022491701878607273:
                                if Q.sum_pt > 390.5546875:
                                    return 't'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0008007169235497713:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.05672547034919262:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 40.76988410949707:
                                    return 't'   # 54% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 84% of the training jets here get this class from the formula
                else:
                    if Q.mass > 41.03298759460449:
                        if Q.eccentricity > 0.9353410303592682:
                            if Q.log_sum_pt > 5.68408465385437:
                                return 't'   # 71% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 70% of the training jets here get this class from the formula
                        else:
                            return 't'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.052264975383877754:
                            if Q.sum_pt > 311.6953125:
                                return 't'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.planar_flow > 0.6731796860694885:
                                    return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 37.42132568359375:
                                if Q.planar_flow > 0.28819866478443146:
                                    return 't'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 85% of the training jets here get this class from the formula
        else:
            if Q.e2 > 0.04606887698173523:
                if Q.pt_7 > 36.671875:
                    if Q.centroid_offset > 0.03190459683537483:
                        return 't'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.z_dr_0p05_0p1 > 0.6488564312458038:
                            if Q.girth2 > 0.009470171760767698:
                                return 't'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 42.125:
                                    return 'Z'   # 79% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 62% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 85% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt > 407.25:
                        if Q.mass > 64.5759048461914:
                            if Q.girth2 > 0.009393270127475262:
                                return 't'   # 57% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0_0p05 > 0.04606982693076134:
                                if Q.centroid_offset > 0.020292827859520912:
                                    return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 30.0078125:
                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                return 't'   # 87% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 94% of the training jets here get this class from the formula
            else:
                if Q.log_sum_pt > 6.009772300720215:
                    if Q.sum_pt > 805.46875:
                        if Q.log_sum_pt > 6.900705575942993:
                            return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.01238954858854413:
                                if Q.pt_7 > 44.1875:
                                    return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.02964075095951557:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 53% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 59% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.014215842820703983:
                            return 't'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.pt_6 > 43.609375:
                                if Q.e2 > 0.04187103360891342:
                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 68% of the training jets here get this class from the formula
                            else:
                                return 't'   # 85% of the training jets here get this class from the formula
                else:
                    return 'g'   # 71% of the training jets here get this class from the formula
    else:
        if Q.mass > 31.293143272399902:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.027192377485334873:
                    if Q.max_dr > 0.1433788537979126:
                        if Q.log_sum_pt > 6.659394025802612:
                            if Q.pt_7 > 31.234375:
                                if Q.sum_pt > 900.625:
                                    return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.e2_sq > 0.007035717600956559:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 57% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.02410678192973137:
                                    return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.089416742324829:
                                if Q.pt_7 > 47.046875:
                                    if Q.C2 > 0.030713336542248726:
                                        return 'g'   # 48% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 7.702916991547681e-05:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 2.2594906091690063:
                                            if Q.sum_pt_top3 > 475.125:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.16331307590007782:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.3201422840356827:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.tau21 > 0.2932930141687393:
                                    return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 65% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.037238309159874916:
                            if Q.width > 0.007442181697115302:
                                if Q.sum_pt_top5 > 319.859375:
                                    return 't'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.00025703851133584976:
                                    if Q.z_7 > 0.0810956358909607:
                                        return 'g'   # 42% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt_sq > 0.004733311245217919:
                                        if Q.pt_6 > 33.421875:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.pt_7 > 34.140625:
                                if Q.width > 0.00815593870356679:
                                    if Q.dr_0 > 0.08337106555700302:
                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 88% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 563.8828125:
                                    if Q.width > 0.007692904444411397:
                                        if Q.pt_7 > 27.5703125:
                                            return 'Z'   # 55% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0004845828952966258:
                                            return 't'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 319.703125:
                                        return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.006930230185389519:
                        if Q.z_dr_0p2_0p4 > 0.027957831509411335:
                            if Q.centroid_offset > 0.013856560457497835:
                                if Q.sum_pt > 837.8203125:
                                    if Q.pt_6 > 39.515625:
                                        return 'g'   # 66% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.27853691577911377:
                                            return 'q'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 39.057350158691406:
                                        if Q.pt_7 > 44.390625:
                                            return 'g'   # 40% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0072987324092537165:
                                                return 't'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.2327083721756935:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 671.3828125:
                                    if Q.width > 0.00825625378638506:
                                        if Q.max_dr > 0.25292959809303284:
                                            if Q.sum_pt > 883.8359375:
                                                return 'g'   # 35% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.09615190699696541:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 36.703125:
                                        if Q.width > 0.008357202634215355:
                                            if Q.tau21 > 0.19990301132202148:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.062330722808838:
                                            if Q.pt_7 > 29.8203125:
                                                if Q.max_dr > 0.24072035402059555:
                                                    if Q.tau21 > 0.30177390575408936:
                                                        return 'g'   # 41% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2_sq > 0.007880508434027433:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 100% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 39.66495895385742:
                                if Q.mass > 53.479087829589844:
                                    if Q.log_sum_pt > 6.990575551986694:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.022117850370705128:
                                            if Q.max_dr > 0.15708794444799423:
                                                if Q.girth2 > 0.007877039723098278:
                                                    return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.008389647118747234:
                                                    if Q.max_dr > 0.13773686438798904:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.008627574425190687:
                                                if Q.e2 > 0.043593158945441246:
                                                    if Q.pt_7 > 28.3203125:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.499527454376221:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.011337348259985447:
                                                        return 't'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.00705037428997457:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.e2 > 0.043367454782128334:
                                                        if Q.z_dr_0_0p05 > 0.04144522547721863:
                                                            if Q.C2 > 0.03478996083140373:
                                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 28.9453125:
                                        if Q.girth2 > 0.007178057683631778:
                                            if Q.max_dr > 0.1556517407298088:
                                                if Q.centroid_offset > 0.01865073014050722:
                                                    if Q.tau21 > 0.21477072685956955:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.921875:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.4941622018814087:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 36.859375:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.008277651388198137:
                                                        if Q.e2 > 0.04586697369813919:
                                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.09650604799389839:
                                                if Q.e2 > 0.04115315154194832:
                                                    if Q.eccentricity > 0.9603946208953857:
                                                        if Q.z_dr_0_0p05 > 0.05878135748207569:
                                                            return 'W'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.010772995185106993:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 0.7964732944965363:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.z_dr_0_0p05 > 0.024372568354010582:
                                                            if Q.e2 > 0.04126502387225628:
                                                                return 'W'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 26.5859375:
                                            if Q.log_sum_pt > 6.295467853546143:
                                                return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 35.890625:
                                    if Q.max_dr > 0.1625383496284485:
                                        return 'Z'   # 40% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.1674121543765068:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0002646994689712301:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.024770975112915:
                                        if Q.tau21 > 0.1851458176970482:
                                            if Q.centroid_offset > 0.0191932562738657:
                                                return 't'   # 46% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                    else:
                        if Q.e2 > 0.04040752165019512:
                            if Q.log_sum_pt > 6.680403709411621:
                                if Q.width > 0.006802528165280819:
                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_5 > 34.75:
                                        return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.00034387721098028123:
                                    if Q.dr_0 > 0.08983195945620537:
                                        return 'W'   # 50% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 0.9278418719768524:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.043506285175681114:
                                                if Q.dr01 > 0.15807824581861496:
                                                    return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.011943037621676922:
                                        if Q.girth > 0.08188793808221817:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.00619269791059196:
                                                if Q.e2 > 0.04124005138874054:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.0068249444011598825:
                                            if Q.log_sum_pt > 6.599530220031738:
                                                if Q.z_dr_0_0p05 > 0.04620295576751232:
                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.14119643718004227:
                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 1.916097640991211:
                                if Q.centroid_offset > 0.014544691424816847:
                                    return 't'   # 62% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 71% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 720.859375:
                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.009270553011447191:
                                        if Q.z_dr_0p05_0p1 > 0.15642865747213364:
                                            if Q.e2 > 0.029970474541187286:
                                                if Q.pt_7 > 27.9921875:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.444613218307495:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.15172725170850754:
                                                if Q.centroid_offset > 0.017797494307160378:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.03742886707186699:
                                            if Q.z_dr_0p05_0p1 > 0.594417929649353:
                                                if Q.width > 0.006783336633816361:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.027254333719611168:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.14520669728517532:
                    if Q.width > 0.005255302879959345:
                        if Q.centroid_offset > 0.008072602096945047:
                            if Q.centroid_offset > 0.030012542381882668:
                                if Q.max_dr > 0.18721305578947067:
                                    if Q.log_sum_pt > 6.587582349777222:
                                        if Q.pt_7 > 33.5:
                                            return 'g'   # 65% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.14211831986904144:
                                                return 't'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.03918434679508209:
                                        if Q.pt_7 > 47.78125:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.005930241430178285:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_6 > 28.75:
                                            if Q.z_dr_0_0p05 > 0.553771048784256:
                                                return 't'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 46.18492126464844:
                                    if Q.centroid_offset > 0.012664586305618286:
                                        if Q.max_dr > 0.2688940167427063:
                                            if Q.sum_pt > 767.3203125:
                                                if Q.centroid_offset > 0.024106080643832684:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.019063892774283886:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.0009201339271385223:
                                                if Q.width > 0.005575522547587752:
                                                    return 'Z'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.14665093272924423:
                                                        if Q.centroid_offset > 0.016089584678411484:
                                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1755961999297142:
                                            if Q.width > 0.0055154478177428246:
                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.20706281065940857:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.811098098754883:
                                                        return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.0058479090221226215:
                                                if Q.LHA > 0.2645101696252823:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 70.10128784179688:
                                                    return 'Z'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 28.3515625:
                                        if Q.n_dr_0p05_0p1 > 1.5:
                                            if Q.centroid_offset > 0.014778312295675278:
                                                if Q.log_sum_pt > 6.2008891105651855:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.765625:
                                                        return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.width > 0.005974187282845378:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.15865810960531235:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.1709585189819336:
                                                if Q.pt_7 > 33.265625:
                                                    if Q.girth2_top2 > 0.0010978076606988907:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.014395688660442829:
                                                    if Q.C2 > 0.03495264612138271:
                                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.19862522929906845:
                                if Q.girth2 > 0.00580770312808454:
                                    return 'Z'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.26260851323604584:
                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.004703656537458301:
                                            if Q.mass_over_sum_pt > 0.07432857155799866:
                                                return 'Z'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.05762346833944321:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.006041594780981541:
                                    if Q.max_dr > 0.1663462296128273:
                                        if Q.centroid_offset > 0.004023659275844693:
                                            if Q.LHA > 0.25106382369995117:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 47% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.006481793941929936:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.00467501487582922:
                                            if Q.mass > 66.36206436157227:
                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0063743554055690765:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 74.81714248657227:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.04159700684249401:
                                                    return 'Z'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 78.88769912719727:
                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.02080067340284586:
                            if Q.width > 0.0033396052895113826:
                                if Q.max_dr > 0.16844581067562103:
                                    if Q.mass > 36.824424743652344:
                                        if Q.centroid_offset > 0.03564519062638283:
                                            if Q.max_dr > 0.2462681233882904:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 22.6171875:
                                                if Q.centroid_offset > 0.023215938359498978:
                                                    if Q.log_sum_pt > 6.909581661224365:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.20993199199438095:
                                                            if Q.max_dr > 0.2613813579082489:
                                                                if Q.sum_pt > 734.15625:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 77% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.1983766406774521:
                                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2 > 0.004059100290760398:
                                                            if Q.z_dr_0p05_0p1 > 0.5302468836307526:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 725.4765625:
                                                    if Q.z_dr_0p2_0p4 > 0.014622590970247984:
                                                        return 'Z'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.052557796239852905:
                                                            return 'Z'   # 75% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 27.3515625:
                                            if Q.LHA > 0.24221985042095184:
                                                if Q.centroid_offset > 0.04076281934976578:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.18899617344141006:
                                                    if Q.lam2 > 0.00019414261623751372:
                                                        return 't'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.465925931930542:
                                                return 'Z'   # 51% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02626609429717064:
                                        if Q.LHA > 0.2532113641500473:
                                            if Q.pt_7 > 25.2109375:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.49020767211914:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03000643290579319:
                                                return 'Z'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 37.9627742767334:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                    else:
                                        if Q.width > 0.004666833905503154:
                                            if Q.dr_0 > 0.060036471113562584:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.022392352111637592:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2 > 0.004965316271409392:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.004062557592988014:
                                                if Q.max_dr > 0.1565016731619835:
                                                    if Q.centroid_offset > 0.023314346559345722:
                                                        if Q.dr_0 > 0.05059806816279888:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19993796944618225:
                                    if Q.lam1 > 0.00245972559787333:
                                        if Q.centroid_offset > 0.023571521043777466:
                                            return 'Z'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2_sq > 0.002280959510244429:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.23960591852664948:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.024457630701363087:
                                            return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.25296100974082947:
                                                return 'Z'   # 39% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.028612219728529453:
                                        if Q.max_dr > 0.1731800138950348:
                                            if Q.LHA > 0.22808361053466797:
                                                return 'Z'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 59% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.033672863617539406:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.922149658203125:
                                            return 'Z'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.02645150013267994:
                                                if Q.mass > 35.51425552368164:
                                                    if Q.max_dr > 0.17906950414180756:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.20089343190193176:
                                if Q.width > 0.00354658963624388:
                                    if Q.centroid_offset > 0.01263404544442892:
                                        if Q.girth2 > 0.003989100223407149:
                                            if Q.mass > 38.96207046508789:
                                                if Q.girth2_top3 > 0.00044943083776161075:
                                                    if Q.z_dr_0p1_0p2 > 0.019924962893128395:
                                                        if Q.centroid_offset > 0.015567162074148655:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.mass_over_sum_pt_sq > 0.004415662260726094:
                                                                return 'Z'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.D2 > 2.7395323514938354:
                                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 55% of the training jets here get this class from the formula
                                            else:
                                                if Q.LHA > 0.22468385100364685:
                                                    return 'Z'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.23980719596147537:
                                                if Q.centroid_offset > 0.01412623981013894:
                                                    return 'Z'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p2_0p4 > 0.040578316897153854:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.01630992628633976:
                                                    if Q.z_dr_0p1_0p2 > 0.01579460548236966:
                                                        return 'W'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.27983614802360535:
                                            if Q.centroid_offset > 0.008332104422152042:
                                                if Q.girth2 > 0.003993304446339607:
                                                    if Q.log_sum_pt > 6.569952487945557:
                                                        return 'Z'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 4.329071998596191:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3250609040260315:
                                                    if Q.log_sum_pt > 6.660769939422607:
                                                        if Q.girth2 > 0.004049548413604498:
                                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.18771544843912125:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.599562168121338:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.004604009911417961:
                                                if Q.centroid_offset > 0.009047037456184626:
                                                    if Q.max_dr > 0.21786190569400787:
                                                        if Q.mass > 41.284799575805664:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 57% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.C2 > 0.04265310801565647:
                                                            return 'Z'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.eccentricity > 0.9486404955387115:
                                                        if Q.mass > 41.878265380859375:
                                                            if Q.log_sum_pt > 6.942017078399658:
                                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.24080874770879745:
                                                                    if Q.centroid_offset > 0.007344065699726343:
                                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 39.22752571105957:
                                                    if Q.mass > 67.95873260498047:
                                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.010523588862270117:
                                                            if Q.max_dr > 0.24274342507123947:
                                                                if Q.girth2 > 0.004120335215702653:
                                                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 90% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 32.375:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.girth > 0.020703423768281937:
                                        if Q.centroid_offset > 0.018201254308223724:
                                            if Q.mass > 42.03307342529297:
                                                if Q.max_dr > 0.2274361476302147:
                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_6 > 30.171875:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.0027973270043730736:
                                                    if Q.max_dr > 0.22521232813596725:
                                                        return 'Z'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 35.4110164642334:
                                                if Q.sum_pt > 1213.7421875:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.2920732796192169:
                                                        if Q.LHA > 0.18317458778619766:
                                                            if Q.tau21 > 0.32468220591545105:
                                                                return 'W'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.681398630142212:
                                                                if Q.dr_7 > 0.34080059826374054:
                                                                    if Q.girth > 0.026731835678219795:
                                                                        if Q.planar_flow > 0.045787978917360306:
                                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                                        else:
                                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.02530273888260126:
                                                            if Q.centroid_offset > 0.01549550611525774:
                                                                if Q.mass_over_sum_pt_sq > 0.0029978854581713676:
                                                                    if Q.max_dr > 0.24417097866535187:
                                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.mass > 50.26076698303223:
                                                                        return 'Z'   # 56% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_6 > 18.9765625:
                                                                return 'W'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.sum_pt > 944.59765625:
                                                                    return 'W'   # 68% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_5 > 20.3125:
                                                    if Q.centroid_offset > 0.011338855605572462:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 586.5:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 42.26090049743652:
                                            if Q.eccentricity > 0.9769396185874939:
                                                if Q.log_sum_pt > 7.075069427490234:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.008116059005260468:
                                                if Q.mass > 35.538564682006836:
                                                    return 'W'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                            else:
                                if Q.girth > 0.030567455105483532:
                                    if Q.centroid_offset > 0.016672552563250065:
                                        if Q.girth2 > 0.004564800299704075:
                                            if Q.max_dr > 0.1728484183549881:
                                                if Q.z_dr_0_0p05 > 0.43020203709602356:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.15554209053516388:
                                                    if Q.width > 0.004956692457199097:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2 > 0.003973027924075723:
                                                if Q.max_dr > 0.17995089292526245:
                                                    if Q.centroid_offset > 0.018449987284839153:
                                                        return 'Z'   # 59% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 35.11530876159668:
                                            if Q.girth2 > 0.002664131228812039:
                                                if Q.mass > 74.19783401489258:
                                                    if Q.z_dr_0_0p05 > 0.8539016544818878:
                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 17.34375:
                                                    if Q.centroid_offset > 0.009827292058616877:
                                                        return 'W'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass > 41.645503997802734:
                                                            return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0_0p05 > 0.9269869029521942:
                                                        return 'W'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.010495327413082123:
                                                if Q.z_top5 > 0.8891595304012299:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.14254703372716904:
                                                        return 'g'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top5 > 568.1875:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 23.7578125:
                                        if Q.centroid_offset > 0.011265887878835201:
                                            if Q.sum_pt > 1081.671875:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.0019271335913799703:
                                                if Q.sum_pt_top3 > 483.9375:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.953831672668457:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04408160224556923:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.013563146349042654:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 45.286203384399414:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                else:
                    if Q.width > 0.002735531306825578:
                        if Q.centroid_offset > 0.02699857112020254:
                            if Q.girth2 > 0.005164559464901686:
                                if Q.max_dr > 0.10768583416938782:
                                    if Q.z_dr_0p1_0p2 > 0.23630648106336594:
                                        return 't'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.04918944463133812:
                                            return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_6 > 30.6328125:
                                                if Q.e2 > 0.03675310127437115:
                                                    return 'Z'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.00556324515491724:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.028746758587658405:
                                                            if Q.lam2 > 0.0001681435460341163:
                                                                return 'Z'   # 79% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.12052466347813606:
                                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.430789947509766:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.LHA > 0.307331919670105:
                                        if Q.mass_over_sum_pt > 0.06352714821696281:
                                            if Q.LHA > 0.3118113577365875:
                                                return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.034464381635189056:
                                                    if Q.lam2 > 0.00028491053672041744:
                                                        return 'Z'   # 55% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.035701220855116844:
                                            if Q.pt1_dr01 > 7.55197286605835:
                                                return 't'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.9658401906490326:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 44.21608543395996:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.034616585820913315:
                                    if Q.max_dr > 0.11604591831564903:
                                        if Q.LHA > 0.26564744114875793:
                                            return 'Z'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 42% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.000123439691378735:
                                            if Q.sum_pt > 782.25:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 49% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.041651083156466484:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 84% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.12715133279561996:
                                        if Q.width > 0.004196318332105875:
                                            if Q.centroid_offset > 0.029338201507925987:
                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.10768495500087738:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0002574566315161064:
                                            if Q.tau21 > 0.13966305553913116:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.395808935165405:
                                                    return 'Z'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 870.203125:
                                                return 'g'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006295684026554227:
                                if Q.e2 > 0.03774021193385124:
                                    if Q.centroid_offset > 0.012770693749189377:
                                        if Q.e2 > 0.03973015956580639:
                                            if Q.D2 > 1.0905981063842773:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.000185389922989998:
                                                    if Q.girth2 > 0.0064917695708572865:
                                                        if Q.e2 > 0.04154557175934315:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.width > 0.006443687016144395:
                                                return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.98406782746315:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 52% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 73.0212516784668:
                                            if Q.pt_7 > 35.734375:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 81.3937873840332:
                                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.005917423404753208:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.11263030394911766:
                                                if Q.z_dr_0p05_0p1 > 0.6635142266750336:
                                                    if Q.girth2 > 0.006550839403644204:
                                                        if Q.e2 > 0.03917275555431843:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00845292815938592:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.004378900863230228:
                                            if Q.mass > 62.227237701416016:
                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 35.3755989074707:
                                    if Q.centroid_offset > 0.018821639008820057:
                                        if Q.girth2 > 0.005573660135269165:
                                            if Q.max_dr > 0.11558401584625244:
                                                if Q.e2 > 0.035418299958109856:
                                                    if Q.width > 0.005762831540778279:
                                                        if Q.dr_0 > 0.07448025792837143:
                                                            return 'W'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.D2 > 0.806583434343338:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 44% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.005825062748044729:
                                                        return 'Z'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.031417179852724075:
                                                            if Q.sum_pt > 689.296875:
                                                                return 'Z'   # 74% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.1040075346827507:
                                                    if Q.girth > 0.07486328482627869:
                                                        if Q.e2 > 0.03720923513174057:
                                                            return 'W'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2 > 0.006057732738554478:
                                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.13711783289909363:
                                                if Q.girth2 > 0.004975039046257734:
                                                    if Q.centroid_offset > 0.022140976041555405:
                                                        return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.879789590835571:
                                                    return 'g'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.00319025048520416:
                                            if Q.mass > 79.39935302734375:
                                                if Q.z_7 > 0.025273984298110008:
                                                    if Q.log_sum_pt > 7.093625068664551:
                                                        return 'g'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.12683618813753128:
                                                    if Q.girth > 0.06967301666736603:
                                                        if Q.centroid_offset > 0.011965323705226183:
                                                            if Q.girth2 > 0.0059059064369648695:
                                                                if Q.e2 > 0.03527306392788887:
                                                                    if Q.D2 > 1.067392110824585:
                                                                        return 'Z'   # 85% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 'W'   # 63% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.0330901313573122:
                                                                    return 'W'   # 95% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.e2 > 0.03495732694864273:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.width > 0.006100977538153529:
                                                                    return 'Z'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.06814848631620407:
                                                            if Q.e2 > 0.0325104221701622:
                                                                return 'W'   # 94% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.max_dr > 0.13978800177574158:
                                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.0035682846792042255:
                                                        if Q.mass > 76.27666091918945:
                                                            if Q.z_7 > 0.04048208333551884:
                                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 89% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 99% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.e2 > 0.025639403611421585:
                                                if Q.pt_7 > 42.390625:
                                                    if Q.lam2 > 1.9348916794115212e-05:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 46.596689224243164:
                                                        return 'W'   # 93% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.011008069384843111:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.008127602748572826:
                                                    return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass > 42.89889717102051:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 35.890625:
                                                            return 'W'   # 57% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.012573959771543741:
                                        if Q.pt_7 > 29.2890625:
                                            if Q.centroid_offset > 0.016298227943480015:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.07427185773849487:
                                                    return 'W'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 380.125:
                                                if Q.pt_6 > 25.8359375:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 28.3828125:
                                                    return 'g'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2 > 0.004208576632663608:
                                            if Q.pt_7 > 32.421875:
                                                if Q.LHA > 0.24957308173179626:
                                                    return 'W'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 43% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 510.59375:
                                                if Q.z_6 > 0.05564095079898834:
                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass > 34.386295318603516:
                                                    if Q.D2 > 0.805768609046936:
                                                        return 'W'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.953125:
                            if Q.centroid_offset > 0.015903531573712826:
                                if Q.log_sum_pt > 6.8048741817474365:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p05_0p1 > 0.19639194756746292:
                                        return 'g'   # 65% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0020852411398664117:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 44.296875:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.0730564184486866:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.735222816467285:
                                            if Q.width > 0.00227554130833596:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013564847875386477:
                                if Q.centroid_offset > 0.01699134148657322:
                                    return 'W'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 33.709938049316406:
                                        if Q.pt_6 > 25.9921875:
                                            if Q.mass_over_sum_pt_sq > 0.0015153493732213974:
                                                return 'W'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 998.078125:
                                    if Q.lam1 > 0.002152568311430514:
                                        return 'W'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 22.6015625:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.035742282867432:
                                                return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 32.703125:
                                        if Q.lam2 > 6.811947605456226e-05:
                                            if Q.pt_7 > 39.296875:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.4741986095905304:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.113709706813097:
                                                if Q.mass > 37.25317192077637:
                                                    return 'W'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 42.458099365234375:
                                            if Q.centroid_offset > 0.008085708599537611:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 94% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 603.484375:
                if Q.centroid_offset > 0.015207459684461355:
                    if Q.centroid_offset > 0.024315545335412025:
                        if Q.mass_over_sum_pt > 0.02637974638491869:
                            if Q.lam2 > 0.00031454172858502716:
                                if Q.pt_7 > 40.359375:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 45% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.03383960202336311:
                                    if Q.pt_6 > 24.1796875:
                                        if Q.pt_7 > 37.390625:
                                            return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.03829108737409115:
                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_6 > 20.2265625:
                                        if Q.pt_7 > 45.4375:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 22.368958473205566:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 9.91025663097389e-05:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 64% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.04271823912858963:
                                if Q.pt_7 > 26.3515625:
                                    if Q.pt_2 > 84.71875:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.2468499019742012:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 68% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.765625:
                                    if Q.mass > 5.639124870300293:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.03186960704624653:
                                            return 'g'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 877.546875:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.02646953985095024:
                                        if Q.pt_6 > 19.234375:
                                            if Q.sum_pt > 940.703125:
                                                if Q.pt_7 > 29.8671875:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt > 0.013731143902987242:
                                                    if Q.centroid_offset > 0.0276877349242568:
                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.5203097760677338:
                                                            return 'Z'   # 88% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.553098440170288:
                                                        if Q.LHA > 0.22348416596651077:
                                                            if Q.sum_pt > 799.5625:
                                                                return 'g'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt > 0.006934895180165768:
                                            if Q.sum_pt > 774.515625:
                                                if Q.pt_7 > 27.4375:
                                                    if Q.sum_pt > 954.421875:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.lam1 > 0.0008059275278355926:
                                                            return 'W'   # 50% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.0007530654838774353:
                                                        return 'W'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth > 0.0267237089574337:
                                                            return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 58% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 24.609375:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 783.234375:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.050879642367362976:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.025385779328644276:
                                                        return 'Z'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.796875:
                            if Q.centroid_offset > 0.019444081000983715:
                                if Q.pt_7 > 47.578125:
                                    if Q.mass > 5.0022053718566895:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.022622741758823395:
                                            return 'Z'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 50.546875:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 7.299920320510864:
                                        if Q.log_sum_pt > 6.703514099121094:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.02211096417158842:
                                            if Q.sum_pt > 802.75:
                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.809401273727417:
                                                return 'g'   # 40% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.017947672866284847:
                                    if Q.pt_7 > 45.046875:
                                        return 'g'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 5.725728273391724:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.0177337983623147:
                                if Q.pt_6 > 23.4609375:
                                    if Q.mass_over_sum_pt > 0.01635571848601103:
                                        if Q.centroid_offset > 0.020869430154561996:
                                            if Q.max_dr > 0.10757295042276382:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9136112630367279:
                                                    if Q.log_sum_pt > 6.597710132598877:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.8536303341388702:
                                                if Q.max_dr > 0.12658735364675522:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 33.296875:
                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.7781243324279785:
                                                            return 'W'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.dr_0 > 0.01862298045307398:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 52% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 33.1875:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.023086861707270145:
                                            if Q.sum_pt > 810.6328125:
                                                if Q.pt_7 > 29.3828125:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 917.09375:
                                                        return 'Z'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1025.953125:
                                                return 'g'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 745.1171875:
                                                    if Q.log_sum_pt > 6.848651885986328:
                                                        if Q.centroid_offset > 0.02160696964710951:
                                                            return 'Z'   # 73% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.020023901015520096:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 626.40625:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 809.6875:
                                        if Q.pt_6 > 19.3125:
                                            if Q.log_sum_pt > 6.74235987663269:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.020034292712807655:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.834520578384399:
                                                return 'W'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 32.203125:
                                    if Q.log_sum_pt > 6.668635129928589:
                                        if Q.pt_7 > 36.984375:
                                            if Q.mass > 5.817806005477905:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.0004115902120247483:
                                                if Q.eccentricity > 0.8163245320320129:
                                                    if Q.max_dr > 0.1419757977128029:
                                                        return 'W'   # 84% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.923820734024048:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.726860761642456:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.01641602162271738:
                                                            return 'W'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.pt_7 > 29.21875:
                                                                return 'W'   # 57% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.17060398310422897:
                                            if Q.lam2 > 0.0001506572007201612:
                                                return 'W'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 646.296875:
                                                return 'q'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.76430869102478:
                                        if Q.pt_6 > 22.4921875:
                                            if Q.sum_pt_top5 > 918.671875:
                                                return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.846216440200806:
                                                return 'W'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 24.5859375:
                                            if Q.sum_pt > 738.890625:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.00043208397983107716:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 44.671875:
                        if Q.girth > 0.006905150134116411:
                            if Q.pt_7 > 48.515625:
                                if Q.pt_7 > 51.421875:
                                    return 'g'   # 99% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.00380421441514045:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 8.426655769348145:
                                            if Q.lam2 > 2.560491702752188e-05:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.746135711669922:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.centroid_offset > 0.005600777454674244:
                                    return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 947.671875:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 2.7267747100268025e-05:
                                            if Q.max_dr > 0.027991710230708122:
                                                if Q.pt_7 > 45.453125:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.890163898468018:
                                if Q.pt_7 > 48.859375:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.942428350448608:
                                        return 'g'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 52.890625:
                                    if Q.centroid_offset > 0.0028486496303230524:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.005427329335361719:
                                            if Q.log_sum_pt > 6.856297492980957:
                                                return 'g'   # 100% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.006064897868782282:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 957.828125:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 800.796875:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.004634361248463392:
                                        if Q.pt_7 > 48.375:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 97% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.008630370255559683:
                            if Q.pt_7 > 35.671875:
                                if Q.pt_7 > 40.046875:
                                    if Q.centroid_offset > 0.009612669236958027:
                                        if Q.LHA > 0.14686747640371323:
                                            if Q.lam2 > 5.939772927376907e-05:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.717608213424683:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.4740957021713257:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 881.65625:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 903.65625:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.012740723323076963:
                                            if Q.girth2_top2 > 0.00031447481887880713:
                                                if Q.eccentricity > 0.8674091398715973:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.655440330505371:
                                                if Q.lam2 > 6.0185580878169276e-05:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0002455974172335118:
                                                    return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.934070110321045:
                                    if Q.pt_7 > 20.8203125:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 635.75:
                                        if Q.centroid_offset > 0.013065237551927567:
                                            if Q.sum_pt > 887.1484375:
                                                if Q.pt_6 > 28.3515625:
                                                    if Q.centroid_offset > 0.014197447337210178:
                                                        return 'W'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_7 > 33.140625:
                                                            return 'g'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.log_sum_pt > 6.8408849239349365:
                                                                return 'W'   # 55% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 31.7578125:
                                                    if Q.dr_0 > 0.015069592278450727:
                                                        return 'q'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2 > 0.0050688209012150764:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.871115684509277:
                                                if Q.pt_7 > 28.7578125:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.14071307331323624:
                                            if Q.planar_flow > 0.6936856210231781:
                                                if Q.z_7 > 0.04164084605872631:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.04016965255141258:
                                                if Q.log_sum_pt > 6.645933389663696:
                                                    return 'q'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.011188582982867956:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_5 > 47.765625:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.039064645767212:
                                if Q.pt_7 > 26.1875:
                                    if Q.pt_7 > 33.1875:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top5 > 3.554322574927937e-05:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 7.133146047592163:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 7.219055652618408:
                                        if Q.pt_7 > 11.9296875:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.LHA > 0.10626206547021866:
                                            if Q.pt_6 > 27.46875:
                                                return 'g'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 626.703125:
                                    if Q.pt_7 > 39.234375:
                                        if Q.sum_pt > 983.78125:
                                            if Q.LHA > 0.08794671669602394:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.982558250427246:
                                                    if Q.girth2 > 3.864702557621058e-05:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 4.4246689867577516e-05:
                                                if Q.sum_pt > 862.40625:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007069471757858992:
                                                    if Q.log_sum_pt > 6.825238943099976:
                                                        return 'g'   # 88% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.968901872634888:
                                            if Q.pt_7 > 31.2421875:
                                                if Q.LHA > 0.09193841740489006:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 9.360775948152877e-05:
                                                    if Q.pt_7 > 24.8125:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 99% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.005033661145716906:
                                        if Q.dr_0 > 0.010393915697932243:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_5 > 56.84375:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.006476350361481309:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.006779128219932318:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 92% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.022137119434773922:
                    if Q.log_sum_pt > 6.4331159591674805:
                        if Q.centroid_offset > 0.02702755481004715:
                            if Q.girth2 > 0.0017588083283044398:
                                if Q.e2_sq > 0.0008995103416964412:
                                    if Q.lam2 > 0.0002759691997198388:
                                        if Q.eccentricity > 0.7390800714492798:
                                            if Q.girth2 > 0.004479620140045881:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 41.796875:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.030826504342257977:
                                                        return 'Z'   # 50% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.040281545370817184:
                                            if Q.centroid_offset > 0.050268057733774185:
                                                return 't'   # 59% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.1548583209514618:
                                                if Q.e2_sq > 0.001278239011298865:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.03223790228366852:
                                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 22.953125:
                                                    return 'W'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 42% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04398087412118912:
                                        if Q.z_7 > 0.03654264844954014:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 45.234375:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0004243917064741254:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top2 > 331.3125:
                                                    if Q.z_top5 > 0.8589090704917908:
                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.042342409491539:
                                                        return 'g'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 66% of the training jets here get this class from the formula
                            else:
                                if Q.mass_over_sum_pt > 0.018211975693702698:
                                    if Q.centroid_offset > 0.03104056790471077:
                                        if Q.lam2 > 0.00033179989259224385:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9099626243114471:
                                            if Q.pt_7 > 47.0:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 73% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_7 > 0.06705853715538979:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.8479118943214417:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 57% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_7 > 26.5234375:
                                        if Q.pt_7 > 51.796875:
                                            if Q.width > 0.0014181238948367536:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.488872528076172:
                                                    if Q.log_sum_pt > 6.64472508430481:
                                                        if Q.girth2 > 0.001029597595334053:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.029288881458342075:
                                                return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 642.6875:
                                                    if Q.e2 > 0.005420038942247629:
                                                        if Q.planar_flow > 0.2853243947029114:
                                                            return 'Z'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.512650489807129:
                                            return 'Z'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0013900174526497722:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 674.59375:
                                if Q.pt_7 > 48.234375:
                                    if Q.C2 > 0.006950237788259983:
                                        if Q.planar_flow > 0.07128122076392174:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.024699261412024498:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.025819014757871628:
                                        if Q.lam1 > 0.000747068232158199:
                                            return 'W'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 63% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005040457472205162:
                                            return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.630167722702026:
                                                return 'Z'   # 62% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.535354852676392:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.17019132524728775:
                                                        return 'W'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                            else:
                                if Q.LHA > 0.20741083472967148:
                                    if Q.lam1 > 0.0018944745534099638:
                                        if Q.pt_6 > 28.890625:
                                            return 'W'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_7 > 0.06679749861359596:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.483250617980957:
                                        if Q.LHA > 0.1827242746949196:
                                            if Q.C2 > 0.007693014573305845:
                                                return 'g'   # 52% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 63% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 24.497224807739258:
                            if Q.sum_pt > 455.09375:
                                if Q.centroid_offset > 0.03888905793428421:
                                    if Q.centroid_offset > 0.05002483166754246:
                                        return 't'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 30.3828125:
                                            if Q.lam2 > 0.0005108237382955849:
                                                return 'g'   # 61% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.planar_flow > 0.15610095858573914:
                                        if Q.mass > 28.261351585388184:
                                            if Q.pt_7 > 29.671875:
                                                if Q.centroid_offset > 0.03353147767484188:
                                                    return 'Z'   # 43% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.026320180855691433:
                                                        if Q.max_dr > 0.16852086037397385:
                                                            return 'g'   # 35% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.5040980577468872:
                                                if Q.pt_7 > 31.7421875:
                                                    if Q.eccentricity > 0.8100016415119171:
                                                        return 'W'   # 66% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 67% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.1470663994550705:
                                            if Q.girth2 > 0.003756417427212:
                                                if Q.pt_7 > 34.046875:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_7 > 27.9140625:
                                                    return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.02177517395466566:
                                                if Q.sum_pt > 510.921875:
                                                    if Q.pt_6 > 28.84375:
                                                        return 'W'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 40% of the training jets here get this class from the formula
                                                else:
                                                    if Q.width > 0.003999976208433509:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.2466816902160645:
                                                            return 'W'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 34.828125:
                                    if Q.centroid_offset > 0.045261433348059654:
                                        return 't'   # 50% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.04929259233176708:
                                        return 'g'   # 61% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0_0p05 > 0.800049901008606:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.07878440991044044:
                                if Q.girth > 0.08647424355149269:
                                    return 't'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 55% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.3731865882873535:
                                    if Q.centroid_offset > 0.030290158465504646:
                                        if Q.centroid_offset > 0.04336604103446007:
                                            return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 32.140625:
                                                if Q.pt_7 > 50.140625:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00029433341114781797:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.LHA > 0.2067134901881218:
                                                            if Q.mass_over_sum_pt_sq > 0.001109384116716683:
                                                                return 'W'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.centroid_offset > 0.040760353207588196:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt > 609.859375:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0017773102736100554:
                                            if Q.z_dr_0_0p05 > 0.9208435118198395:
                                                return 'W'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 21.263364791870117:
                                        if Q.e2 > 0.01798511203378439:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.05416908487677574:
                                                return 't'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.eccentricity > 0.9625682234764099:
                                                    if Q.centroid_offset > 0.043404584750533104:
                                                        return 'Z'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.centroid_offset > 0.027197287417948246:
                                                            if Q.pt_7 > 29.421875:
                                                                return 'W'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 54% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 554.078125:
                        if Q.mass_over_sum_pt > 0.01695671770721674:
                            if Q.z_7 > 0.059098176658153534:
                                if Q.pt_7 > 49.390625:
                                    return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.1894450187683105:
                                        return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.009387976955622435:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 4.5059969124849886e-05:
                                                if Q.pt_7 > 45.46875:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.eccentricity > 0.8677972555160522:
                                    if Q.dr_0 > 0.012248802464455366:
                                        if Q.centroid_offset > 0.015552370343357325:
                                            if Q.max_dr > 0.13385938853025436:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.04732782207429409:
                                                    if Q.planar_flow > 0.20646730810403824:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    if Q.LHA > 0.1728411614894867:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.3380481004714966:
                                                if Q.z_7 > 0.0483667217195034:
                                                    if Q.lam2 > 3.244469189667143e-05:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.007592395413666964:
                                            return 'g'   # 89% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.007714478066191077:
                                                if Q.sum_pt_top5 > 578.640625:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.centroid_offset > 0.015146933495998383:
                                        if Q.LHA > 0.17720628529787064:
                                            if Q.pt_7 > 25.578125:
                                                return 'W'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.01565519068390131:
                                            if Q.z_7 > 0.05051450431346893:
                                                if Q.centroid_offset > 0.009044204838573933:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.007380012888461351:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.00016247358144028112:
                                                    return 'q'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.0040909829549491405:
                                if Q.centroid_offset > 0.01967553049325943:
                                    if Q.sum_pt > 702.2265625:
                                        if Q.z_7 > 0.06480089947581291:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.z_7 > 0.057697001844644547:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.613855838775635:
                                            if Q.centroid_offset > 0.0069840538781136274:
                                                if Q.centroid_offset > 0.017843235284090042:
                                                    return 'W'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_7 > 0.04335377737879753:
                                                        return 'g'   # 87% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 8.727453231811523:
                                                if Q.lam2 > 4.142325997236185e-05:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.007813658099621534:
                                                        if Q.sum_pt_top5 > 575.25:
                                                            return 'q'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 69% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.024496634490787983:
                                                    return 'g'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.z_7 > 0.06313971057534218:
                                    if Q.log_sum_pt > 6.702228784561157:
                                        if Q.girth > 0.005976437823846936:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.5808775424957275:
                                        if Q.z_7 > 0.05618455074727535:
                                            if Q.sum_pt > 747.859375:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.D2 > 1.2956942319869995:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.dr_0 > 0.010638607200235128:
                                            return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top5 > 584.15625:
                                                if Q.pt_5 > 45.03125:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 27.834721565246582:
                            if Q.centroid_offset > 0.017317375168204308:
                                if Q.pt_7 > 32.390625:
                                    if Q.D2 > 1.6395787596702576:
                                        return 'W'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 29.750361442565918:
                                            return 'W'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 0.0015402332646772265:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 43% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top5 > 497.34375:
                                    if Q.C2 > 0.025578533299267292:
                                        return 'g'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_7 > 45.265625:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00010477983232703991:
                                                return 'g'   # 64% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top2 > 0.0006268828001338989:
                                                    return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.004314090358093381:
                                        if Q.pt_7 > 35.046875:
                                            if Q.z_dr_0p05_0p1 > 0.21971385180950165:
                                                if Q.mass > 30.376627922058105:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 50% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.centroid_offset > 0.013813862577080727:
                                            if Q.max_dr > 0.1360492929816246:
                                                if Q.pt_7 > 35.6875:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top5 > 506.796875:
                                if Q.mass_top5 > 10.3717041015625:
                                    if Q.z_7 > 0.0641789436340332:
                                        if Q.D2 > 0.8445455133914948:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.pt_7 > 48.546875:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.centroid_offset > 0.007293808739632368:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.planar_flow > 0.2463376820087433:
                                            if Q.centroid_offset > 0.01141044357791543:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.dr_0 > 0.019000142812728882:
                                                    if Q.z_7 > 0.05867212638258934:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top5 > 528.140625:
                                                            return 'q'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.dr_0 > 0.014942038804292679:
                                                if Q.centroid_offset > 0.016120835207402706:
                                                    if Q.z_7 > 0.05047115497291088:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    if Q.D2 > 1.4322952032089233:
                                                        return 'g'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 11.145750522613525:
                                        if Q.z_7 > 0.053005898371338844:
                                            if Q.D2 > 1.0886252522468567:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_7 > 0.06809327006340027:
                                                    return 'g'   # 98% of the training jets here get this class from the formula
                                                else:
                                                    if Q.centroid_offset > 0.006937784142792225:
                                                        return 'g'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 5.955332926532719e-05:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.000265267924987711:
                                                    if Q.sum_pt_top5 > 536.890625:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 0.831730991601944:
                                    return 'g'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top5 > 472.34375:
                                        if Q.z_7 > 0.06909306347370148:
                                            return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.centroid_offset > 0.011942985467612743:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.15587791055440903:
                                                    return 'g'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 99% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 0
    print('class:', classify(pt, eta, phi))
