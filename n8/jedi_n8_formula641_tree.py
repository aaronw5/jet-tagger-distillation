"""JEDI-linear jet tagger, 8 particles, 3 features: the tuned formula (start): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 62.57% (the formula: 65.56%); same class as the formula for 85.07% of jets.  56 leaves, depth 10.
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
            return 'g'   # 62% of the training jets here get this class from the formula
        else:
            if Q.mass > 45.279863357543945:
                return 't'   # 94% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.03272750414907932:
                    return 't'   # 82% of the training jets here get this class from the formula
                else:
                    return 'g'   # 59% of the training jets here get this class from the formula
    else:
        if Q.mass > 30.866321563720703:
            if Q.girth2 > 0.006653153337538242:
                if Q.centroid_offset > 0.02942987810820341:
                    return 't'   # 56% of the training jets here get this class from the formula
                else:
                    if Q.max_dr > 0.21452561765909195:
                        if Q.centroid_offset > 0.012098278384655714:
                            return 't'   # 45% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 74% of the training jets here get this class from the formula
                    else:
                        return 'Z'   # 90% of the training jets here get this class from the formula
            else:
                if Q.max_dr > 0.1473577842116356:
                    if Q.girth2 > 0.004858389729633927:
                        if Q.centroid_offset > 0.011311578564345837:
                            return 'Z'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.17949608713388443:
                                if Q.girth2 > 0.005485784262418747:
                                    return 'Z'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 1.5421990156173706:
                                        return 'Z'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.width > 0.00601978600025177:
                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 90% of the training jets here get this class from the formula
                    else:
                        if Q.centroid_offset > 0.01820436865091324:
                            if Q.width > 0.0035425531677901745:
                                return 'Z'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.19975372403860092:
                                    return 'Z'   # 65% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 81% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.0017601520521566272:
                                if Q.max_dr > 0.22487571090459824:
                                    if Q.girth2 > 0.0035206981701776385:
                                        if Q.centroid_offset > 0.012668156065046787:
                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 77% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 87% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 63% of the training jets here get this class from the formula
                else:
                    if Q.girth2 > 0.0026776777813211083:
                        if Q.centroid_offset > 0.024785758927464485:
                            if Q.width > 0.00483433622866869:
                                if Q.max_dr > 0.11201531067490578:
                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 55% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 69% of the training jets here get this class from the formula
                        else:
                            if Q.girth2 > 0.006397217977792025:
                                if Q.e2 > 0.03773641400039196:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 38.515625:
                            return 'g'   # 67% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.013291803188621998:
                                return 'W'   # 81% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 65% of the training jets here get this class from the formula
        else:
            if Q.sum_pt_top5 > 596.453125:
                if Q.centroid_offset > 0.015240416396409273:
                    if Q.centroid_offset > 0.02524277288466692:
                        if Q.girth2 > 0.0016775433905422688:
                            return 'W'   # 30% of the training jets here get this class from the formula
                        else:
                            return 'Z'   # 72% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 40.515625:
                            return 'g'   # 65% of the training jets here get this class from the formula
                        else:
                            if Q.centroid_offset > 0.018050862476229668:
                                return 'W'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt > 840.8515625:
                                    return 'W'   # 71% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 61% of the training jets here get this class from the formula
                else:
                    if Q.pt_7 > 42.296875:
                        if Q.centroid_offset > 0.004032833967357874:
                            return 'g'   # 87% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.891260623931885:
                                return 'g'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.pt_7 > 50.046875:
                                    if Q.width > 4.453965448192321e-05:
                                        return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 94% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.pt_7 > 33.796875:
                            if Q.centroid_offset > 0.007019019220024347:
                                return 'g'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.934412479400635:
                                    return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 93% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 6.988492488861084:
                                if Q.centroid_offset > 0.003343026852235198:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'q'   # 96% of the training jets here get this class from the formula
            else:
                if Q.centroid_offset > 0.022896800190210342:
                    if Q.sum_pt_top5 > 485.734375:
                        if Q.centroid_offset > 0.029591670259833336:
                            if Q.centroid_offset > 0.04361565597355366:
                                return 'g'   # 69% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 58% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 52% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 24.02134132385254:
                            return 'W'   # 38% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 88% of the training jets here get this class from the formula
                else:
                    if Q.sum_pt_top5 > 540.890625:
                        if Q.centroid_offset > 0.004053136566653848:
                            return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            return 'q'   # 52% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 92% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    print('class:', classify(pt, eta, phi))
