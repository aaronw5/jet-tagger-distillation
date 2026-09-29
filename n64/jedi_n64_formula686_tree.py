"""JEDI-linear jet tagger, 64 particles, 3 features: the tuned formula (start): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 79.19% (the formula: 81.46%); same class as the formula for 91.31% of jets.  55 leaves, depth 8.
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_6=pt[6],
        pt_7=pt[7],
        pt_9=pt[9],
        z_0=z[0],
        z_4=z[4],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
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
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_5=eta[5],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def decide(Q):
    if Q.mass > 84.84941864013672:
        if Q.girth2 > 0.009016547817736864:
            if Q.e2 > 0.038742437958717346:
                if Q.log_sum_pt > 7.009220838546753:
                    if Q.tau32 > 0.5237221121788025:
                        return 'g'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 192.17066192626953:
                            return 'g'   # 74% of the training jets here get this class from the formula
                        else:
                            return 't'   # 91% of the training jets here get this class from the formula
                else:
                    if Q.tau32 > 0.5762497186660767:
                        if Q.sum_pt > 1038.5926513671875:
                            if Q.lam1 > 0.028610888868570328:
                                return 'g'   # 74% of the training jets here get this class from the formula
                            else:
                                return 't'   # 66% of the training jets here get this class from the formula
                        else:
                            return 't'   # 89% of the training jets here get this class from the formula
                    else:
                        return 't'   # 98% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1038.2291259765625:
                    return 'g'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.028827364556491375:
                        if Q.sum_pt > 988.6719970703125:
                            if Q.z_top50_slots > 0.9766022264957428:
                                if Q.z_dr_0p1_0p2 > 0.1585603505373001:
                                    return 't'   # 63% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 67% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 766.423828125:
                                return 't'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 78% of the training jets here get this class from the formula
                    else:
                        if Q.z_top50_slots > 0.9839461445808411:
                            if Q.sum_pt_top40 > 943.3115234375:
                                return 'q'   # 69% of the training jets here get this class from the formula
                            else:
                                return 't'   # 63% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 82% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.sum_pt > 1072.51220703125:
                    return 'g'   # 95% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02443495485931635:
                        if Q.log_sum_pt > 6.89000391960144:
                            return 'Z'   # 82% of the training jets here get this class from the formula
                        else:
                            return 't'   # 53% of the training jets here get this class from the formula
                    else:
                        return 'g'   # 71% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.28694534301758:
                    return 'g'   # 78% of the training jets here get this class from the formula
                else:
                    if Q.mass_top50 > 86.53249740600586:
                        if Q.girth2_top20 > 0.0038607511669397354:
                            return 'Z'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.33268189430236816:
                                return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 85% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top30 > 1015.1318359375:
                            if Q.D2 > 1.7925812602043152:
                                if Q.girth2_top30 > 0.004865220515057445:
                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 963.772216796875:
                                return 'Z'   # 80% of the training jets here get this class from the formula
                            else:
                                return 't'   # 76% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.width > 0.006974134128540754:
                if Q.sum_pt > 974.260986328125:
                    return 'Z'   # 46% of the training jets here get this class from the formula
                else:
                    return 't'   # 74% of the training jets here get this class from the formula
            else:
                if Q.girth2_top20 > 0.0032217807602137327:
                    if Q.sum_pt_top50 > 968.90185546875:
                        if Q.C2 > 0.08916794881224632:
                            if Q.mass > 82.09088897705078:
                                return 'Z'   # 61% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 81% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 98% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0p2_0p4 > 2.5:
                            return 't'   # 49% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 87% of the training jets here get this class from the formula
                else:
                    if Q.eccentricity > 0.8680699169635773:
                        return 'W'   # 81% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1075.8192138671875:
                            return 'g'   # 75% of the training jets here get this class from the formula
                        else:
                            return 'W'   # 54% of the training jets here get this class from the formula
        else:
            if Q.n_real_top50 > 44.5:
                if Q.sum_pt > 1052.7176513671875:
                    return 'g'   # 96% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.160959243774414:
                            if Q.mass_over_sum_pt > 0.07926744222640991:
                                return 'g'   # 66% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 60% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 87% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top15 > 19.13690948486328:
                            if Q.sum_pt_top40 > 926.7423095703125:
                                if Q.mass > 70.74204635620117:
                                    return 'W'   # 47% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 954.576171875:
                                if Q.n_particles > 51.5:
                                    return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 55% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 93% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1117.7225341796875:
                    if Q.n_particles > 33.5:
                        return 'g'   # 81% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 90% of the training jets here get this class from the formula
                else:
                    if Q.n_real_top50 > 38.5:
                        if Q.sum_pt > 1063.7198486328125:
                            return 'g'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 901.9462890625:
                                return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 48% of the training jets here get this class from the formula
                    else:
                        return 'q'   # 97% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
