"""JEDI-linear jet tagger, 64 particles, 3 features: the tuned formula (start): ONE tree of if-statements on the jet quantities that gives the class directly.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t).

1. quantities(): physics quantities of the particles (the same as in the formula).
2. decide():     one tree; each test is "quantity > threshold".
                 Grown on the entire training set (595,000 jets), each jet labelled with the formula's class;
                 each leaf notes the share of its training jets that the formula puts in its class.

Test set (50,000 jets): accuracy 80.52% (the formula: 81.46%); same class as the formula for 94.05% of jets.  939 leaves, depth 17.
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
                        if Q.mass > 178.91419219970703:
                            if Q.tau21 > 0.5012564957141876:
                                if Q.z_top50_slots > 0.9672821164131165:
                                    if Q.log_sum_pt > 7.120833158493042:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04732930101454258:
                                if Q.lam2 > 0.0012377405655570328:
                                    if Q.tau21 > 0.25661537051200867:
                                        return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.05158821679651737:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top15 > 109.25737762451172:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top10 > 848.375:
                                        return 'g'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 58.5:
                                            if Q.tau32 > 0.7384912371635437:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 55% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p05_0p1 > 15.5:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0035948718432337046:
                                    if Q.mass_top50 > 149.43779754638672:
                                        if Q.z_top50_slots > 0.9660064578056335:
                                            return 't'   # 92% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.043819986283779144:
                                        if Q.mass_top15 > 91.40533828735352:
                                            if Q.lam2 > 0.0015054333489388227:
                                                if Q.max_dr > 0.3755233436822891:
                                                    return 't'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                    else:
                        if Q.mass > 192.17066192626953:
                            if Q.tau21 > 0.47448399662971497:
                                if Q.z_top50_slots > 0.9645468294620514:
                                    if Q.log_sum_pt > 7.190066576004028:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 87% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.06928620859980583:
                                        return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.04168423265218735:
                                if Q.tau32 > 0.42511333525180817:
                                    if Q.mass > 182.78397369384766:
                                        if Q.e2 > 0.05213656462728977:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top15 > 1093.81640625:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 90% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9510844349861145:
                                        return 't'   # 98% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.42263348400592804:
                                    if Q.mass_over_sum_pt > 0.13944830745458603:
                                        return 't'   # 71% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 73% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                else:
                    if Q.tau32 > 0.5762497186660767:
                        if Q.sum_pt > 1038.5926513671875:
                            if Q.lam1 > 0.028610888868570328:
                                if Q.eccentricity > 0.7901439666748047:
                                    if Q.sum_pt_top3 > 462.03125:
                                        if Q.log_sum_pt > 6.97193455696106:
                                            return 'g'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.046362193301320076:
                                    if Q.z_dr_0p05_0p1 > 0.6642658114433289:
                                        if Q.sum_pt_top3 > 580.890625:
                                            if Q.z_dr_0p2_0p4 > 0.06841681897640228:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1084.875244140625:
                                                return 'g'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.001197390432935208:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_dr_0p1_0p2 > 0.10580171272158623:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 44% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 179.32582092285156:
                                            if Q.C2 > 0.11918198317289352:
                                                if Q.z_top50_slots > 0.9533074498176575:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 53% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 1070.11328125:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.001312936597969383:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9809885025024414:
                                                    if Q.pt_9 > 21.984375:
                                                        if Q.sum_pt > 1086.429931640625:
                                                            return 't'   # 70% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.2883395701646805:
                                                                return 't'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 48% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 49% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 63.5:
                                        if Q.lam2 > 0.002792365732602775:
                                            if Q.log_sum_pt > 6.983834981918335:
                                                if Q.e2 > 0.043530143797397614:
                                                    return 't'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1069.34521484375:
                                                if Q.C2 > 0.12877491861581802:
                                                    return 't'   # 58% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.977878600358963:
                                                    if Q.dr_0 > 0.08947351202368736:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.08202289044857025:
                                            if Q.pt_dispersion > 0.3649474233388901:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.971143484115601:
                                                    return 'g'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top5_slots > 0.5431699156761169:
                                                        return 'q'   # 77% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.31061218678951263:
                                                if Q.sum_pt_top3 > 567.90625:
                                                    if Q.girth2_top50 > 0.009440097957849503:
                                                        return 't'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 61% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 727.28515625:
                                if Q.lam1 > 0.031971486285328865:
                                    if Q.planar_flow > 0.26630210876464844:
                                        if Q.e2 > 0.06464950740337372:
                                            return 't'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9513590931892395:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.38021938502788544:
                                                    return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top50_slots > 0.975193202495575:
                                            if Q.mass_top50 > 180.43772888183594:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top10_slots > 0.5041592419147491:
                                                if Q.LHA > 0.4715813100337982:
                                                    return 'g'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 1008.92431640625:
                                                        return 'g'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 566.359375:
                                        if Q.log_sum_pt > 6.890968084335327:
                                            if Q.z_dr_0p1_0p2 > 0.1555098071694374:
                                                if Q.e2_sq > 0.009729481302201748:
                                                    if Q.max_pair_mass > 34.38485145568848:
                                                        if Q.mass_top15 > 140.15330505371094:
                                                            return 'q'   # 51% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 90% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.20532090961933136:
                                                    if Q.lam2 > 0.0013330142828635871:
                                                        return 't'   # 91% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam1 > 0.009351613000035286:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 46% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.32473134994506836:
                                                return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 45% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2_sq > 0.009343732614070177:
                                            if Q.girth2_top5 > 0.005789983551949263:
                                                if Q.sum_pt > 1003.1624755859375:
                                                    if Q.e2 > 0.04446897655725479:
                                                        if Q.lam1 > 0.029477103613317013:
                                                            if Q.tau21 > 0.20659618824720383:
                                                                return 't'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.mass_top50 > 178.3883819580078:
                                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_dr_0p05_0p1 > 0.6975333392620087:
                                                                if Q.pt_4 > 45.71875:
                                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.max_dr > 0.3420378416776657:
                                                            if Q.sum_pt_top2 > 360.34375:
                                                                if Q.lam1 > 0.01079660514369607:
                                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 75% of the training jets here get this class from the formula
                                                            else:
                                                                return 't'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top30_slots > 0.8610022068023682:
                                                                if Q.girth2_top20 > 0.010597020387649536:
                                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.04125823266804218:
                                                                    return 't'   # 71% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top30 > 771.69921875:
                                                        if Q.n_dr_0p2_0p4 > 1.5:
                                                            return 't'   # 99% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.girth2_top20 > 0.01067696139216423:
                                                                return 't'   # 93% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 984.24267578125:
                                                    if Q.z_dr_0p2_0p4 > 0.0691596195101738:
                                                        if Q.mass_over_sum_pt > 0.14824797213077545:
                                                            return 't'   # 86% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.sum_pt_top3 > 398.84375:
                                                                return 'q'   # 84% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.e2 > 0.04333074204623699:
                                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top30 > 781.756103515625:
                                                        if Q.sum_pt_top50 > 981.10546875:
                                                            if Q.e2 > 0.043570466339588165:
                                                                return 't'   # 92% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0p05_0p1 > 0.7164851129055023:
                                                                    return 'q'   # 78% of the training jets here get this class from the formula
                                                                else:
                                                                    return 't'   # 66% of the training jets here get this class from the formula
                                                        else:
                                                            return 't'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.890836000442505:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    return 't'   # 50% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.3228265792131424:
                                                    return 't'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 610.751953125:
                                    if Q.e2 > 0.04475056566298008:
                                        if Q.lam1 > 0.031550804153084755:
                                            if Q.tau21 > 0.4557828903198242:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9465680718421936:
                                                    return 't'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 856.90625:
                                            if Q.mass_over_sum_pt > 0.17637615650892258:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 183.21875:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top40_slots > 0.8616543114185333:
                                                    return 't'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 72% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top30 > 647.84765625:
                            if Q.mass_over_sum_pt > 0.09770118817687035:
                                if Q.mass > 191.06275177001953:
                                    if Q.tau21 > 0.33418506383895874:
                                        if Q.e2 > 0.055583324283361435:
                                            return 't'   # 95% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1078.1103515625:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9279801845550537:
                                                    return 't'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 67% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.957413911819458:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                        else:
                                            if Q.planar_flow > 0.1778145655989647:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.04414956457912922:
                                        return 't'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 1073.3336181640625:
                                            if Q.tau32 > 0.4303828030824661:
                                                if Q.max_dr > 0.34767912328243256:
                                                    return 't'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 808.873046875:
                                                if Q.sum_pt_top30 > 968.76318359375:
                                                    if Q.sum_pt_top3 > 646.59375:
                                                        return 'q'   # 63% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.planar_flow > 0.19370242953300476:
                                                            return 't'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 54% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 96% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top50 > 995.697265625:
                                    return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 78% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top5 > 36.37182426452637:
                                return 't'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top20 > 521.2265625:
                                    return 't'   # 59% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 80% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1038.2291259765625:
                    if Q.n_particles > 55.5:
                        if Q.tau32 > 0.4161815494298935:
                            if Q.log_sum_pt > 6.978979587554932:
                                return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.n_particles > 62.5:
                                    if Q.e2 > 0.03668497875332832:
                                        return 'g'   # 68% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.3720057010650635:
                                            if Q.e2 > 0.030248328112065792:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_6 > 33.359375:
                                        if Q.girth2_top3 > 0.003350410610437393:
                                            if Q.lam2 > 0.0014784862287342548:
                                                return 't'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 38% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.6372641921043396:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 48% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 70% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top40 > 0.013913295231759548:
                                if Q.z_top50_slots > 0.9581478834152222:
                                    if Q.mass > 176.90402221679688:
                                        return 'g'   # 72% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_over_sum_pt > 0.14622363448143005:
                                        return 't'   # 61% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9869278967380524:
                                    if Q.z_dr_0p1_0p2 > 0.14806129038333893:
                                        return 't'   # 70% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 75% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1080.9970703125:
                            if Q.pt_7 > 24.4921875:
                                if Q.tau32 > 0.45159219205379486:
                                    return 'g'   # 90% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 45% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top50 > 1131.364013671875:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 681.53125:
                                        return 'q'   # 93% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 54% of the training jets here get this class from the formula
                        else:
                            if Q.pt_dispersion > 0.3331304043531418:
                                return 'q'   # 87% of the training jets here get this class from the formula
                            else:
                                if Q.z_dr_0p1_0p2 > 0.12730418145656586:
                                    return 't'   # 64% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1047.4859619140625:
                                        if Q.tau32 > 0.49840009212493896:
                                            return 'g'   # 52% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.028827364556491375:
                        if Q.sum_pt > 988.6719970703125:
                            if Q.z_top50_slots > 0.9766022264957428:
                                if Q.z_dr_0p1_0p2 > 0.1585603505373001:
                                    if Q.pt_7 > 30.8203125:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt > 0.09688498452305794:
                                                if Q.max_dr > 0.36925292015075684:
                                                    return 't'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.001895986613817513:
                                            return 't'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top2 > 0.00838353531435132:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_dispersion > 0.266769140958786:
                                        if Q.mass > 98.87337493896484:
                                            if Q.C2 > 0.14874069392681122:
                                                return 't'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top5_slots > 0.5904616713523865:
                                                    return 'q'   # 96% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.12147370725870132:
                                                        return 't'   # 64% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 81% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 16.5:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2 > 0.009349598549306393:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_over_sum_pt_sq > 0.009530448354780674:
                                            if Q.lam2 > 0.0014142144937068224:
                                                if Q.girth2_top5 > 0.004755201982334256:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.C2 > 0.13220225274562836:
                                                        return 't'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 61% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 58% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 141.56890869140625:
                                    if Q.tau32 > 0.6633106172084808:
                                        if Q.lam2 > 0.002284812508150935:
                                            return 't'   # 65% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9681256711483002:
                                        if Q.sum_pt_top40 > 941.6591796875:
                                            if Q.tau32 > 0.718759298324585:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.00610455428250134:
                                                    return 't'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.max_dr > 0.3542994409799576:
                                                if Q.e2 > 0.0327787920832634:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 38% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.e2 > 0.0361781045794487:
                                            if Q.sum_pt > 999.998046875:
                                                if Q.D2 > 3.3269288539886475:
                                                    return 't'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 766.423828125:
                                if Q.z_top50_slots > 0.961214542388916:
                                    if Q.sum_pt_top30 > 936.32470703125:
                                        if Q.z_dr_0p1_0p2 > 0.09762674942612648:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top3 > 415.53125:
                                                if Q.mass > 95.84103775024414:
                                                    if Q.sum_pt_top40 > 958.8817138671875:
                                                        return 'q'   # 78% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.884334325790405:
                                            if Q.sum_pt_top3 > 421.125:
                                                if Q.e2 > 0.03194653429090977:
                                                    return 't'   # 69% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.03163523226976395:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top40_slots > 0.9403635561466217:
                                                        return 't'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 57% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 838.09375:
                                                return 't'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_over_sum_pt_sq > 0.018709314987063408:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.03196057677268982:
                                        if Q.tau32 > 0.5462651252746582:
                                            if Q.log_sum_pt > 6.873031377792358:
                                                if Q.mass_top50 > 129.9297637939453:
                                                    return 't'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.11394835263490677:
                                                        return 't'   # 53% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 74% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.34991657733917236:
                                            if Q.log_sum_pt > 6.877562761306763:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top5 > 0.00815505301579833:
                                                    return 't'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.6225368082523346:
                                                        return 'g'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.014705067500472069:
                                    return 'g'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 769.5751953125:
                                        return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 64% of the training jets here get this class from the formula
                    else:
                        if Q.z_top50_slots > 0.9839461445808411:
                            if Q.sum_pt_top40 > 943.3115234375:
                                if Q.pt_dispersion > 0.2845582216978073:
                                    if Q.n_particles > 60.5:
                                        if Q.LHA > 0.23006388545036316:
                                            return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 93% of the training jets here get this class from the formula
                                else:
                                    if Q.log_sum_pt > 6.919938087463379:
                                        if Q.n_particles > 61.5:
                                            return 'g'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 48% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p1_0p2 > 0.12879440188407898:
                                            return 't'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top5 > 0.0008819138165563345:
                                                if Q.sum_pt_top50 > 975.42041015625:
                                                    return 'q'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 47% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9953512251377106:
                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                            else:
                                if Q.e2 > 0.0240139439702034:
                                    if Q.max_dr > 0.5141500234603882:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6914538145065308:
                                        if Q.z_top50_slots > 0.9935274422168732:
                                            return 't'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 63% of the training jets here get this class from the formula
                        else:
                            if Q.e2 > 0.024272220209240913:
                                if Q.z_top50_slots > 0.9714282155036926:
                                    if Q.log_sum_pt > 6.887784957885742:
                                        if Q.pt_9 > 24.6640625:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1014.27587890625:
                                                return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top10 > 0.00332936504855752:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.5793003141880035:
                                                if Q.lam2 > 0.004036137601360679:
                                                    return 't'   # 68% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 82% of the training jets here get this class from the formula
                                else:
                                    if Q.C2 > 0.17393627762794495:
                                        return 't'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth > 0.11096563935279846:
                                            if Q.z_top50_slots > 0.9408584237098694:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 2.644737720489502:
                                                return 'g'   # 90% of the training jets here get this class from the formula
                                            else:
                                                if Q.log_sum_pt > 6.8918092250823975:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top50_slots > 0.964751273393631:
                                                        return 't'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 93% of the training jets here get this class from the formula
        else:
            if Q.n_particles > 62.5:
                if Q.sum_pt > 1072.51220703125:
                    if Q.z_dr_0p2_0p4 > 0.0032225524773821235:
                        if Q.sum_pt > 1115.85498046875:
                            return 'g'   # 99% of the training jets here get this class from the formula
                        else:
                            if Q.z_dr_0p2_0p4 > 0.005946426186710596:
                                if Q.e2 > 0.023396477103233337:
                                    if Q.mass > 95.49390029907227:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 84.62195205688477:
                                            return 'Z'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top40 > 0.005614653695374727:
                                                if Q.mass > 91.30117797851562:
                                                    return 'g'   # 90% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.7816509902477264:
                                                        return 'g'   # 81% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 70% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 96% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9710497558116913:
                                    return 'Z'   # 88% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1239.396728515625:
                            if Q.mass > 92.90767669677734:
                                return 'g'   # 100% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top40 > 79.65656280517578:
                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 94% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.95526322722435:
                                if Q.mass > 94.64977264404297:
                                    return 'g'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.00471565849147737:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 64% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 89% of the training jets here get this class from the formula
                else:
                    if Q.e2 > 0.02443495485931635:
                        if Q.log_sum_pt > 6.89000391960144:
                            if Q.mass > 93.9399185180664:
                                if Q.mass_top30 > 76.47164535522461:
                                    if Q.mass > 98.04782104492188:
                                        return 'g'   # 83% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 12.5:
                                            return 'g'   # 33% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 82% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top30 > 0.005827347515150905:
                                    if Q.width > 0.008265165612101555:
                                        if Q.mass_top40 > 78.37572479248047:
                                            if Q.n_dr_0p2_0p4 > 9.5:
                                                if Q.C2 > 0.05143386870622635:
                                                    return 'Z'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 75.60923385620117:
                                            if Q.mass > 92.4604263305664:
                                                if Q.girth2_top30 > 0.006434518378227949:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.007835212163627148:
                                                return 'g'   # 76% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.6693825125694275:
                                        if Q.girth2_top40 > 0.0059040633495897055:
                                            if Q.mass > 90.22840118408203:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top30 > 65.28209686279297:
                                                    return 'Z'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 66% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 77% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.29668375849723816:
                                if Q.e2 > 0.027721384540200233:
                                    return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9769369661808014:
                                        return 't'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 6.868258476257324:
                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 60% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.31468693912029266:
                            if Q.z_top50_slots > 0.979811429977417:
                                if Q.girth2_top5 > 0.0010193554917350411:
                                    if Q.e2_sq > 0.008204325567930937:
                                        if Q.sum_pt_top3 > 417.15625:
                                            return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top30 > 0.004848645301535726:
                                            if Q.sum_pt > 1048.8326416015625:
                                                return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_9 > 15.3828125:
                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.4931105077266693:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.z_top50_slots > 0.9559215009212494:
                                if Q.mass > 91.82611846923828:
                                    return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 951.2666015625:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 56% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.24055462330579758:
                                    return 'g'   # 78% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 73% of the training jets here get this class from the formula
            else:
                if Q.mass > 99.28694534301758:
                    if Q.girth2_top20 > 0.0069860732182860374:
                        if Q.mass > 105.25511932373047:
                            if Q.n_dr_0p2_0p4 > 4.5:
                                return 'g'   # 87% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 60% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 10.5:
                                if Q.pt_7 > 24.6796875:
                                    return 'Z'   # 38% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 84% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 84% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1124.5501708984375:
                            if Q.n_dr_0p2_0p4 > 4.5:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.log_sum_pt > 7.133136749267578:
                                    return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top30 > 0.00731772743165493:
                                if Q.sum_pt_top2 > 464.5625:
                                    return 'q'   # 73% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 49% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 85% of the training jets here get this class from the formula
                else:
                    if Q.mass_top50 > 86.53249740600586:
                        if Q.girth2_top20 > 0.0038607511669397354:
                            if Q.sum_pt_top50 > 973.7825927734375:
                                if Q.n_dr_0p2_0p4 > 17.5:
                                    if Q.mass > 94.93874740600586:
                                        if Q.girth2_top30 > 0.007146735209971666:
                                            if Q.D2 > 3.2230504751205444:
                                                return 'q'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 57% of the training jets here get this class from the formula
                                    else:
                                        if Q.C2 > 0.10049169138073921:
                                            if Q.e2_sq > 0.008198515977710485:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.022673146799206734:
                                                    return 'Z'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.005566728068515658:
                                        if Q.mass > 87.24390411376953:
                                            if Q.mass > 96.97949981689453:
                                                if Q.girth2_top20 > 0.0064019698183983564:
                                                    return 'Z'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top40 > 93.66156768798828:
                                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 988.3460693359375:
                                                    if Q.D2 > 4.862040758132935:
                                                        if Q.mass > 94.70732879638672:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                                        else:
                                                            return 'Z'   # 89% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 6.5:
                                                        if Q.z_dr_0p2_0p4 > 0.04619820974767208:
                                                            return 'Z'   # 95% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.max_dr > 0.35803332924842834:
                                                                if Q.n_dr_0p2_0p4 > 8.5:
                                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                                else:
                                                                    if Q.sum_pt_top50 > 983.0330810546875:
                                                                        return 'Z'   # 77% of the training jets here get this class from the formula
                                                                    else:
                                                                        return 't'   # 72% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 1055.478271484375:
                                                if Q.D2 > 1.6743345260620117:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                                else:
                                                    if Q.max_dr > 0.3185137063264847:
                                                        return 'W'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.36353784799575806:
                                                    if Q.D2 > 1.4419627785682678:
                                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.e2_sq > 0.007163971196860075:
                                                            if Q.n_dr_0p2_0p4 > 6.5:
                                                                if Q.girth2_top5 > 0.0060121905989944935:
                                                                    return 't'   # 67% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'Z'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'Z'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 53% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 6.5:
                                            if Q.mass > 92.68731307983398:
                                                if Q.n_particles > 43.5:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 66% of the training jets here get this class from the formula
                                            else:
                                                if Q.C2 > 0.038448844105005264:
                                                    if Q.D2 > 5.994119167327881:
                                                        return 'g'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 42% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 87.71007537841797:
                                                if Q.mass > 94.51457595825195:
                                                    if Q.mass_top30 > 88.58021926879883:
                                                        return 'Z'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 99% of the training jets here get this class from the formula
                                            else:
                                                if Q.max_dr > 0.38310734927654266:
                                                    return 'W'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 91% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 3.5:
                                    if Q.z_dr_0p2_0p4 > 0.047628724947571754:
                                        if Q.lam2 > 0.0008119656995404512:
                                            if Q.sum_pt_top50 > 961.1339111328125:
                                                if Q.D2 > 3.540404796600342:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.024488158524036407:
                                                    return 't'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 50% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top40 > 944.2337646484375:
                                                return 'Z'   # 87% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 56% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.3162461668252945:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                return 't'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 960.637451171875:
                                                    if Q.girth2_top20 > 0.007511239731684327:
                                                        return 'Z'   # 73% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 950.74609375:
                                        return 'Z'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.33547182381153107:
                                            return 't'   # 76% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 96% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.33268189430236816:
                                if Q.log_sum_pt > 6.960397005081177:
                                    if Q.n_dr_0p2_0p4 > 6.5:
                                        if Q.girth2_top30 > 0.005095877451822162:
                                            if Q.mass > 91.8729476928711:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top3 > 0.0005268894601613283:
                                                    return 'Z'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top30 > 79.09103393554688:
                                            return 'Z'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2 > 0.008155670948326588:
                                        if Q.pt_dispersion > 0.2668848931789398:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 84% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 93.65275955200195:
                                    if Q.mass_top5 > 11.68628215789795:
                                        return 'Z'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 77% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top10 > 0.0003694935148814693:
                                        return 'Z'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 50% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt_top30 > 1015.1318359375:
                            if Q.D2 > 1.7925812602043152:
                                if Q.girth2_top30 > 0.004865220515057445:
                                    if Q.D2 > 2.512758731842041:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 85.66533279418945:
                                            return 'Z'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam1 > 0.006162071833387017:
                                                return 'Z'   # 92% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 70% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top40 > 81.85797882080078:
                                        if Q.tau32 > 0.774233877658844:
                                            return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.3195868134498596:
                                    if Q.z_top50_slots > 0.9987444877624512:
                                        if Q.mass > 86.04413604736328:
                                            if Q.lam1 > 0.006454849615693092:
                                                if Q.z_dr_0_0p05 > 0.0632256418466568:
                                                    return 'Z'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 83% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top30 > 0.005074671236798167:
                                            return 'Z'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top50 > 85.6198844909668:
                                        if Q.mass_top2 > 46.0338134765625:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 80% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top30 > 1034.2525634765625:
                                            if Q.mass_top15 > 68.8727035522461:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 75% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top50 > 963.772216796875:
                                if Q.C2 > 0.04754587076604366:
                                    if Q.D2 > 5.95058274269104:
                                        if Q.lam2 > 0.0010447310050949454:
                                            if Q.z_top50_slots > 0.9938589632511139:
                                                return 'q'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.889137506484985:
                                            if Q.girth2_top40 > 0.0058009501080960035:
                                                if Q.n_dr_0p2_0p4 > 22.5:
                                                    return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.0524146743118763:
                                                    return 'Z'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top30 > 0.006894240388646722:
                                                return 'Z'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.e2 > 0.02735460363328457:
                                                    return 't'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    if Q.max_dr > 0.34874463081359863:
                                        if Q.e2_sq > 0.00721598113887012:
                                            if Q.sum_pt_top50 > 992.0194091796875:
                                                if Q.D2 > 0.9132848978042603:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 4.5:
                                                        return 't'   # 42% of the training jets here get this class from the formula
                                                    else:
                                                        return 'Z'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.0032591326162219048:
                                                    if Q.pt_dispersion > 0.38576996326446533:
                                                        return 'Z'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 93% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 85.93941497802734:
                                                if Q.girth2_top40 > 0.005866304971277714:
                                                    return 'Z'   # 59% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.02222766913473606:
                                                    return 'Z'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 92% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.3110775947570801:
                                    if Q.e2 > 0.022278619930148125:
                                        if Q.sum_pt_top30 > 941.5157470703125:
                                            if Q.girth2_top20 > 0.007484300294891:
                                                return 'Z'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 94% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 46% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 84% of the training jets here get this class from the formula
    else:
        if Q.mass_top40 > 69.55105972290039:
            if Q.width > 0.006974134128540754:
                if Q.sum_pt > 974.260986328125:
                    if Q.D2 > 1.9328188300132751:
                        if Q.mass_top30 > 61.79167556762695:
                            if Q.mass > 83.4296989440918:
                                if Q.n_dr_0p2_0p4 > 18.5:
                                    return 'q'   # 45% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 85% of the training jets here get this class from the formula
                            else:
                                if Q.D2 > 2.3906995058059692:
                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 40% of the training jets here get this class from the formula
                        else:
                            return 'g'   # 75% of the training jets here get this class from the formula
                    else:
                        if Q.mass_over_sum_pt_sq > 0.007202488137409091:
                            if Q.max_dr > 0.34048689901828766:
                                if Q.z_dr_0p2_0p4 > 0.01895362325012684:
                                    return 'Z'   # 58% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 73% of the training jets here get this class from the formula
                            else:
                                return 'Z'   # 85% of the training jets here get this class from the formula
                        else:
                            if Q.max_dr > 0.30546824634075165:
                                if Q.sum_pt_top40 > 975.1190185546875:
                                    if Q.n_dr_0p2_0p4 > 5.5:
                                        if Q.sum_pt_top40 > 990.62890625:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.D2 > 1.1536754965782166:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                else:
                                    return 't'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 83.68060302734375:
                                    return 'Z'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 65% of the training jets here get this class from the formula
                else:
                    if Q.D2 > 2.729994535446167:
                        if Q.mass_top30 > 68.0019645690918:
                            if Q.sum_pt_top40 > 956.5394287109375:
                                if Q.n_dr_0p2_0p4 > 15.5:
                                    return 'q'   # 87% of the training jets here get this class from the formula
                                else:
                                    return 'Z'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 77.40347290039062:
                                    if Q.z_dr_0_0p05 > 0.8541611433029175:
                                        if Q.pt_9 > 19.21875:
                                            if Q.z_top40_slots > 0.9963470101356506:
                                                return 't'   # 75% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 46% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.00932273082435131:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 81% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 65% of the training jets here get this class from the formula
                        else:
                            if Q.D2 > 4.011471509933472:
                                return 'g'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top40 > 797.701171875:
                                    if Q.z_top50_slots > 0.9746637642383575:
                                        return 't'   # 66% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 71% of the training jets here get this class from the formula
                    else:
                        if Q.max_dr > 0.26411646604537964:
                            if Q.z_dr_0p2_0p4 > 0.07487567141652107:
                                if Q.sum_pt_top50 > 798.79541015625:
                                    if Q.sum_pt_top30 > 944.588134765625:
                                        if Q.mass_top40 > 82.09294509887695:
                                            return 'Z'   # 64% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 72% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 78.16305923461914:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 49% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top2 > 243.5625:
                                        return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 54% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 655.478515625:
                                    if Q.z_top50_slots > 0.9683734476566315:
                                        if Q.z_dr_0p2_0p4 > 0.001446452282834798:
                                            if Q.max_dr > 0.4955512583255768:
                                                if Q.mass_over_sum_pt_sq > 0.007701306138187647:
                                                    return 't'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 49% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 83.03678131103516:
                                                return 'Z'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 46% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 82.97179412841797:
                                return 'Z'   # 86% of the training jets here get this class from the formula
                            else:
                                if Q.girth2 > 0.007276209071278572:
                                    return 't'   # 69% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top30_slots > 0.9970333576202393:
                                        return 'W'   # 91% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 52% of the training jets here get this class from the formula
            else:
                if Q.girth2_top20 > 0.0032217807602137327:
                    if Q.sum_pt_top50 > 968.90185546875:
                        if Q.C2 > 0.08916794881224632:
                            if Q.mass > 82.09088897705078:
                                if Q.mass_top50 > 82.97039413452148:
                                    if Q.n_dr_0p2_0p4 > 18.5:
                                        return 'q'   # 53% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 3.5719916820526123:
                                            if Q.girth2_top40 > 0.005352341337129474:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 44% of the training jets here get this class from the formula
                                        else:
                                            return 'Z'   # 51% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.005040996940806508:
                                        if Q.D2 > 4.540440797805786:
                                            if Q.n_dr_0p2_0p4 > 17.5:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'Z'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_over_sum_pt_sq > 0.006497091148048639:
                                                return 'Z'   # 53% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 85% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 59% of the training jets here get this class from the formula
                            else:
                                if Q.n_dr_0p2_0p4 > 13.5:
                                    if Q.D2 > 5.40408182144165:
                                        if Q.sum_pt_top50 > 1013.1025390625:
                                            if Q.z_top5_slots > 0.7636236250400543:
                                                if Q.lam1 > 0.004153782734647393:
                                                    if Q.tau32 > 0.6682923138141632:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 60% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.log_sum_pt > 6.904904127120972:
                                            if Q.mass_over_sum_pt > 0.07991349697113037:
                                                return 'Z'   # 34% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top40 > 0.005134166916832328:
                                                    return 'W'   # 89% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 49% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top50 > 0.006423471728339791:
                                        if Q.n_dr_0p2_0p4 > 8.5:
                                            return 'Z'   # 47% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.004070996074005961:
                                            return 'W'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 9.5:
                                                if Q.mass_top50 > 74.44154739379883:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_7 > 21.546875:
                                                        return 'W'   # 61% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 89% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 63.5:
                                if Q.mass > 82.5799674987793:
                                    if Q.girth2_top30 > 0.004990513203665614:
                                        if Q.mass_top40 > 73.04777145385742:
                                            if Q.girth2_top50 > 0.0063181305304169655:
                                                return 'Z'   # 60% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 68% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 60% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 88% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top30 > 0.00430380436591804:
                                        if Q.mass > 81.27629089355469:
                                            if Q.mass_top50 > 76.22718811035156:
                                                return 'W'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 59% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 98% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass > 80.11484909057617:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.7186584174633026:
                                                return 'g'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 98% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top50 > 72.23726654052734:
                                    if Q.mass_over_sum_pt > 0.08227335289120674:
                                        if Q.D2 > 2.638716220855713:
                                            if Q.mass > 83.58233261108398:
                                                return 'Z'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top10 > 0.0008122498693410307:
                                                    return 'W'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'Z'   # 85% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.011508875992149115:
                                                if Q.mass_top40 > 80.15670013427734:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt > 993.04345703125:
                                                        return 'W'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 't'   # 56% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top20 > 0.003991771722212434:
                                            if Q.mass > 83.88108825683594:
                                                if Q.D2 > 3.075273275375366:
                                                    return 'Z'   # 72% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 96% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 975.751220703125:
                                                    return 'W'   # 100% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top20 > 0.005315030692145228:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.D2 > 1.5015031099319458:
                                                            return 'W'   # 81% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                                return 't'   # 89% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 83.15483093261719:
                                                if Q.eccentricity > 0.8847745954990387:
                                                    if Q.D2 > 3.2532156705856323:
                                                        return 'Z'   # 62% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 91% of the training jets here get this class from the formula
                                                else:
                                                    if Q.girth2_top50 > 0.006140120094642043:
                                                        return 'Z'   # 47% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top30 > 72.87109756469727:
                                                            return 'W'   # 55% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 982.2271728515625:
                                                    if Q.mass_top40 > 72.93021011352539:
                                                        return 'W'   # 97% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.girth2_top40 > 0.004591114120557904:
                                                            return 'W'   # 91% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                                if Q.tau32 > 0.7528456151485443:
                                                                    return 'g'   # 61% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                            else:
                                                                return 'W'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top5 > 629.0:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 53% of the training jets here get this class from the formula
                                else:
                                    if Q.eccentricity > 0.9047297239303589:
                                        return 'W'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.004037446342408657:
                                            if Q.tau21 > 0.35395365953445435:
                                                if Q.sum_pt_top50 > 984.760498046875:
                                                    if Q.mass_top40 > 71.21468734741211:
                                                        return 'W'   # 95% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.sum_pt_top2 > 525.125:
                                                            return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top30 > 71.09530258178711:
                                                    return 'W'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    if Q.mass_top5 > 3.543891191482544:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0p2_0p4 > 2.5:
                            if Q.z_dr_0_0p05 > 0.8138881921768188:
                                if Q.eccentricity > 0.9272895455360413:
                                    if Q.n_dr_0p2_0p4 > 11.5:
                                        return 'q'   # 44% of the training jets here get this class from the formula
                                    else:
                                        return 'W'   # 89% of the training jets here get this class from the formula
                                else:
                                    if Q.e2 > 0.025302414782345295:
                                        if Q.mass_top40 > 73.46708297729492:
                                            return 'W'   # 71% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 72% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.31309595704078674:
                                    if Q.log_sum_pt > 6.87086033821106:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.D2 > 2.1465762853622437:
                                                return 'W'   # 27% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau32 > 0.8518623411655426:
                                                return 't'   # 63% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 958.5662841796875:
                                        return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 't'   # 50% of the training jets here get this class from the formula
                        else:
                            if Q.mass > 81.03396987915039:
                                return 'Z'   # 52% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.34424275159835815:
                                    if Q.sum_pt_top20 > 884.302734375:
                                        return 'W'   # 92% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top50 > 958.63916015625:
                                            return 'W'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 67% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 94% of the training jets here get this class from the formula
                else:
                    if Q.eccentricity > 0.8680699169635773:
                        if Q.mass > 82.65948486328125:
                            if Q.mass_top20 > 70.3990249633789:
                                return 'W'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top40 > 0.005468237213790417:
                                    return 'Z'   # 74% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 73% of the training jets here get this class from the formula
                        else:
                            if Q.planar_flow > 0.3156609982252121:
                                if Q.girth2_top40 > 0.004414305556565523:
                                    if Q.sum_pt_top50 > 986.8284912109375:
                                        return 'W'   # 89% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 31% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top30 > 69.56355285644531:
                                        return 'W'   # 79% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.37098030745983124:
                                            return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 52% of the training jets here get this class from the formula
                            else:
                                return 'W'   # 93% of the training jets here get this class from the formula
                    else:
                        if Q.sum_pt > 1075.8192138671875:
                            if Q.mass_top30 > 68.20998764038086:
                                if Q.e2 > 0.018714895471930504:
                                    if Q.mass > 83.23384857177734:
                                        return 'g'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.34607014060020447:
                                            if Q.mass_top30 > 71.2741584777832:
                                                return 'W'   # 75% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top20 > 1051.810546875:
                                                    return 'g'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 75% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 80.58219909667969:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.D2 > 6.174113750457764:
                                            if Q.n_real_top40 > 35.5:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 53% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top30 > 0.0035598023096099496:
                                    if Q.n_particles > 54.5:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_pt_above_10 > 12.5:
                                            if Q.girth2_top3 > 0.0006544987263623625:
                                                return 'W'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top50 > 39.5:
                                        if Q.z_dr_0p2_0p4 > 0.0030716429464519024:
                                            return 'g'   # 96% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_particles > 63.5:
                                                return 'g'   # 97% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 54% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 52% of the training jets here get this class from the formula
                        else:
                            if Q.n_dr_0p2_0p4 > 14.5:
                                if Q.z_top10_slots > 0.8053672313690186:
                                    if Q.D2 > 4.946976184844971:
                                        if Q.log_sum_pt > 6.932679653167725:
                                            if Q.z_top10_slots > 0.8576928675174713:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.5400883257389069:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top40 > 73.155029296875:
                                            return 'W'   # 78% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 59% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.9943611025810242:
                                        if Q.sum_pt > 999.92041015625:
                                            if Q.girth > 0.041507115587592125:
                                                if Q.mass > 82.71773910522461:
                                                    return 'Z'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9991876482963562:
                                                    return 'W'   # 41% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 62% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 79% of the training jets here get this class from the formula
                            else:
                                if Q.mass > 81.67353057861328:
                                    if Q.girth2_top50 > 0.006077564088627696:
                                        return 'Z'   # 58% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 80% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 980.7099609375:
                                        if Q.z_top10_slots > 0.8235359787940979:
                                            if Q.mass_top40 > 71.89775085449219:
                                                return 'W'   # 80% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 63% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 71.7986946105957:
                                                return 'W'   # 95% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top15_slots > 0.8341468870639801:
                                                    if Q.girth2_top10 > 0.001697152096312493:
                                                        return 'q'   # 60% of the training jets here get this class from the formula
                                                    else:
                                                        return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 84% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 33% of the training jets here get this class from the formula
        else:
            if Q.n_real_top50 > 44.5:
                if Q.sum_pt > 1052.7176513671875:
                    if Q.girth2_top30 > 0.0035288315266370773:
                        if Q.z_top50_slots > 0.9972058236598969:
                            if Q.sum_pt_top3 > 556.734375:
                                return 'q'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top20 > 909.37890625:
                                    return 'g'   # 50% of the training jets here get this class from the formula
                                else:
                                    return 'W'   # 82% of the training jets here get this class from the formula
                        else:
                            if Q.girth2_top40 > 0.004230038728564978:
                                if Q.mass > 79.64737701416016:
                                    return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top40 > 0.004457754781469703:
                                        if Q.e2 > 0.01879869680851698:
                                            return 'W'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.max_dr > 0.30871325731277466:
                                            return 'g'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top50_slots > 0.9706637561321259:
                                                return 'W'   # 100% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 65% of the training jets here get this class from the formula
                            else:
                                return 'g'   # 89% of the training jets here get this class from the formula
                    else:
                        if Q.n_dr_0_0p05 > 17.5:
                            if Q.sum_pt > 1083.412841796875:
                                return 'g'   # 99% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9997527301311493:
                                    if Q.girth2_top15 > 0.0006825495220255107:
                                        if Q.n_dr_0p2_0p4 > 4.5:
                                            if Q.sum_pt_top2 > 473.96875:
                                                return 'q'   # 85% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.6235380470752716:
                                                    if Q.lam2 > 0.0005079166148789227:
                                                        return 'g'   # 85% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 56% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 83% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 91% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                        else:
                            if Q.log_sum_pt > 7.007084369659424:
                                return 'g'   # 96% of the training jets here get this class from the formula
                            else:
                                if Q.z_top50_slots > 0.9997665882110596:
                                    if Q.pt_6 > 37.125:
                                        return 'g'   # 70% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top15 > 0.0008718508761376143:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 61% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 84% of the training jets here get this class from the formula
                else:
                    if Q.n_particles > 57.5:
                        if Q.mass_top30 > 56.160959243774414:
                            if Q.mass_over_sum_pt > 0.07926744222640991:
                                if Q.D2 > 2.7580090761184692:
                                    if Q.max_dr > 0.2262929379940033:
                                        if Q.tau32 > 0.7167128920555115:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 978.421875:
                                                return 'W'   # 42% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'Z'   # 39% of the training jets here get this class from the formula
                                else:
                                    if Q.z_top50_slots > 0.982549250125885:
                                        if Q.mass_top15 > 40.34163284301758:
                                            return 't'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 47% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 62% of the training jets here get this class from the formula
                            else:
                                if Q.max_dr > 0.33544546365737915:
                                    if Q.mass_top40 > 65.78070068359375:
                                        if Q.n_dr_0p2_0p4 > 11.5:
                                            if Q.girth2_top10 > 0.000840107531985268:
                                                if Q.z_top20_slots > 0.8666459619998932:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 55% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 991.8983154296875:
                                                if Q.mass > 80.37286758422852:
                                                    return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top20_slots > 0.8729671239852905:
                                                        return 'W'   # 45% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.mass_top20 > 38.48668098449707:
                                                            return 'W'   # 92% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 34% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_particles > 62.5:
                                            if Q.tau32 > 0.5965811014175415:
                                                if Q.lam2 > 0.0006751113396603614:
                                                    return 'g'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 44% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 56% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 970.728515625:
                                                if Q.z_top10_slots > 0.5979727208614349:
                                                    if Q.sum_pt > 1019.0550537109375:
                                                        return 'g'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 53% of the training jets here get this class from the formula
                                else:
                                    if Q.mass > 71.40892791748047:
                                        if Q.log_sum_pt > 6.883075475692749:
                                            if Q.z_top50_slots > 0.94731205701828:
                                                return 'W'   # 94% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 54% of the training jets here get this class from the formula
                                        else:
                                            return 'W'   # 45% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 53% of the training jets here get this class from the formula
                        else:
                            if Q.mass_top10 > 19.51738739013672:
                                if Q.n_particles > 62.5:
                                    if Q.sum_pt_top50 > 952.880859375:
                                        if Q.girth2_top30 > 0.0036881831474602222:
                                            if Q.max_dr > 0.3041759878396988:
                                                if Q.z_top50_slots > 0.9830073714256287:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top50_slots > 0.9496577084064484:
                                                    if Q.mass_top50 > 67.0599479675293:
                                                        return 'W'   # 86% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 44% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1016.539794921875:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.010269755497574806:
                                                    if Q.sum_pt_top3 > 434.515625:
                                                        if Q.dr_0 > 0.01346293417736888:
                                                            return 'q'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 96% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.7285919785499573:
                                                            return 'g'   # 94% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.z_top50_slots > 0.979941338300705:
                                                                return 'q'   # 49% of the training jets here get this class from the formula
                                                            else:
                                                                return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.z_top40_slots > 0.9315432608127594:
                                                        if Q.tau32 > 0.853150337934494:
                                                            return 'g'   # 67% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 86% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top50 > 952.32080078125:
                                        if Q.sum_pt_top40 > 993.046875:
                                            if Q.pt_7 > 26.2734375:
                                                if Q.tau32 > 0.6865884959697723:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 58% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 71.67296600341797:
                                                if Q.z_top30_slots > 0.893199622631073:
                                                    return 'q'   # 42% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau32 > 0.860889196395874:
                                                    if Q.z_dr_0p2_0p4 > 0.008784984704107046:
                                                        return 'g'   # 56% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top15 > 39.902530670166016:
                                            return 't'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.tau32 > 0.6723323166370392:
                                    if Q.lam2 > 0.0005937489331699908:
                                        return 'g'   # 97% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 0.00036289842682890594:
                                            if Q.n_particles > 63.5:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top50 > 1016.4541015625:
                                                    return 'g'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    if Q.log_sum_pt > 6.867136001586914:
                                                        return 'q'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 84% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 95% of the training jets here get this class from the formula
                                else:
                                    if Q.n_particles > 63.5:
                                        return 'g'   # 89% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top3 > 0.0007785155321471393:
                                            if Q.lam1 > 0.0034532264107838273:
                                                return 'g'   # 48% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top2 > 380.5:
                                                return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 80% of the training jets here get this class from the formula
                    else:
                        if Q.mass_top15 > 19.13690948486328:
                            if Q.sum_pt_top40 > 926.7423095703125:
                                if Q.mass > 70.74204635620117:
                                    if Q.z_top15_slots > 0.8327580690383911:
                                        if Q.z_top10_slots > 0.8048343062400818:
                                            return 'q'   # 86% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt > 1004.3641357421875:
                                                if Q.girth2_top30 > 0.0037718121893703938:
                                                    return 'W'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 51% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt > 970.636962890625:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 54% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt > 984.853759765625:
                                            if Q.mass_top40 > 67.31815338134766:
                                                return 'W'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_top40_slots > 0.9795464277267456:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.C2 > 0.05112852156162262:
                                                return 'W'   # 33% of the training jets here get this class from the formula
                                            else:
                                                return 't'   # 71% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1032.408203125:
                                        if Q.n_particles > 51.5:
                                            if Q.girth2_top15 > 0.0009959553135558963:
                                                if Q.pt_6 > 39.90625:
                                                    if Q.n_dr_0p2_0p4 > 5.5:
                                                        return 'g'   # 74% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 71% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                                            else:
                                                if Q.z_dr_0p2_0p4 > 0.005361997755244374:
                                                    if Q.pt_7 > 25.828125:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top20 > 0.0009793421486392617:
                                                if Q.pt_dispersion > 0.3181568831205368:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                                else:
                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                        return 'g'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.0006020066502969712:
                                                    return 'g'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top40 > 952.185791015625:
                                            if Q.z_top50_slots > 0.9997554421424866:
                                                if Q.mass_top50 > 68.21330642700195:
                                                    if Q.z_top5_slots > 0.6035524010658264:
                                                        return 'q'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.log_sum_pt > 6.919406414031982:
                                                            return 'W'   # 61% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 97% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top15 > 0.0008239738526754081:
                                                    if Q.pt_7 > 36.390625:
                                                        if Q.dr_0 > 0.01465195370838046:
                                                            return 'q'   # 87% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.0008809391583781689:
                                                                return 'g'   # 59% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 72% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 94% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_6 > 35.921875:
                                                        if Q.n_dr_0p2_0p4 > 5.5:
                                                            if Q.tau32 > 0.6729869246482849:
                                                                if Q.tau21 > 0.590720146894455:
                                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'q'   # 63% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 77% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                        else:
                                            if Q.tau21 > 0.3459220975637436:
                                                if Q.mass_top15 > 22.855234146118164:
                                                    if Q.n_dr_0p2_0p4 > 7.5:
                                                        if Q.sum_pt_top3 > 400.71875:
                                                            return 'q'   # 79% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 52% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.0005899856332689524:
                                                        if Q.sum_pt_top2 > 375.25:
                                                            return 'q'   # 68% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 83% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 87% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth > 0.050487278029322624:
                                                    return 't'   # 81% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 76% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top10 > 27.118226051330566:
                                    if Q.mass_top50 > 64.63678741455078:
                                        if Q.z_dr_0p2_0p4 > 0.033794036135077477:
                                            if Q.sum_pt_top5 > 542.984375:
                                                return 'q'   # 71% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 't'   # 78% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top5 > 449.078125:
                                            return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                return 'g'   # 47% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                else:
                                    if Q.lam2 > 0.0005476337391883135:
                                        if Q.sum_pt_top2 > 355.625:
                                            if Q.girth2_top3 > 0.00019524723757058382:
                                                return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top3 > 548.6875:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 4.5:
                                                if Q.tau32 > 0.7050521373748779:
                                                    return 'g'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.dr_0 > 0.03245190903544426:
                                                        return 'q'   # 46% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 71% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top15 > 25.23215961456299:
                                                    return 'q'   # 77% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 69% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 72% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top40 > 954.576171875:
                                if Q.n_particles > 51.5:
                                    if Q.girth2_top20 > 0.0007355147681664675:
                                        if Q.dr_0 > 0.012621862348169088:
                                            if Q.lam1 > 0.0041254141833633184:
                                                return 'W'   # 65% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 64% of the training jets here get this class from the formula
                                        else:
                                            if Q.log_sum_pt > 6.923563718795776:
                                                return 'g'   # 83% of the training jets here get this class from the formula
                                            else:
                                                if Q.pt_6 > 45.578125:
                                                    return 'g'   # 83% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 64% of the training jets here get this class from the formula
                                    else:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            if Q.tau32 > 0.6666597425937653:
                                                if Q.n_dr_0_0p05 > 38.5:
                                                    return 'q'   # 62% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 91% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top5 > 9.374666115036234e-05:
                                        if Q.sum_pt > 1031.613037109375:
                                            if Q.pt_7 > 27.734375:
                                                if Q.lam2 > 0.0003589777334127575:
                                                    if Q.girth2_top30 > 0.002888899762183428:
                                                        return 'W'   # 54% of the training jets here get this class from the formula
                                                    else:
                                                        return 'g'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00044040601642336696:
                                                if Q.girth2_top20 > 0.0003829656488960609:
                                                    if Q.lam1 > 0.004096233053132892:
                                                        return 'W'   # 68% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_9 > 30.9453125:
                                                            if Q.girth2_top20 > 0.0006278944783844054:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                            else:
                                                                if Q.z_dr_0_0p05 > 0.9396992027759552:
                                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                                                else:
                                                                    return 'g'   # 58% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 88% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.7992296814918518:
                                                        if Q.pt_9 > 30.9140625:
                                                            return 'g'   # 82% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 51% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 70% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top50 > 60.96187400817871:
                                                    return 'W'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_top5_slots > 0.6610846519470215:
                                            if Q.log_sum_pt > 6.932631015777588:
                                                if Q.sum_pt_top5 > 756.734375:
                                                    return 'q'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 81% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top20 > 16.66836166381836:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.9036674797534943:
                                                        return 'g'   # 79% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.00030578189762309194:
                                                if Q.n_dr_0p2_0p4 > 6.5:
                                                    if Q.tau32 > 0.8433516025543213:
                                                        return 'g'   # 92% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.n_particles > 46.5:
                                                            return 'g'   # 84% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 52% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top50 > 1019.8660888671875:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.pt_dispersion > 0.29310885071754456:
                                                            return 'q'   # 76% of the training jets here get this class from the formula
                                                        else:
                                                            return 'g'   # 62% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                            else:
                                if Q.lam2 > 0.0003837336989818141:
                                    if Q.sum_pt_top40 > 922.54931640625:
                                        if Q.mass_top10 > 11.296099662780762:
                                            if Q.lam2 > 0.0005744721565861255:
                                                return 'g'   # 79% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 70% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 98% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 60% of the training jets here get this class from the formula
            else:
                if Q.sum_pt > 1117.7225341796875:
                    if Q.n_particles > 33.5:
                        if Q.girth2_top15 > 0.0005367645644582808:
                            if Q.n_real_top50 > 39.5:
                                if Q.sum_pt > 1144.8658447265625:
                                    if Q.z_dr_0p2_0p4 > 0.0014118471881374717:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 50% of the training jets here get this class from the formula
                                else:
                                    if Q.pt_9 > 23.28125:
                                        if Q.n_dr_0p2_0p4 > 3.5:
                                            return 'g'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 77% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 66% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top20 > 1164.203125:
                                    if Q.n_dr_0p2_0p4 > 2.5:
                                        if Q.pt_dispersion > 0.4002145230770111:
                                            if Q.sum_pt_top40 > 1295.7528076171875:
                                                return 'g'   # 82% of the training jets here get this class from the formula
                                            else:
                                                if Q.planar_flow > 0.8008994162082672:
                                                    return 'g'   # 70% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 87% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top3 > 649.5:
                                        return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.z_dr_0p2_0p4 > 0.0030446507735177875:
                                            if Q.lam2 > 0.00035475892946124077:
                                                return 'g'   # 69% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 69% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.n_particles > 35.5:
                                if Q.n_real_top40 > 37.5:
                                    return 'g'   # 97% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top40 > 1157.79345703125:
                                        return 'g'   # 93% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_3 > 74.53125:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 73% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top30 > 1204.583984375:
                                    if Q.pt_2 > 80.34375:
                                        return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 61% of the training jets here get this class from the formula
                                else:
                                    if Q.mass_top5 > 4.112050771713257:
                                        return 'q'   # 76% of the training jets here get this class from the formula
                                    else:
                                        if Q.pt_dispersion > 0.3849831372499466:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 79% of the training jets here get this class from the formula
                    else:
                        if Q.n_real_top50 > 31.5:
                            if Q.log_sum_pt > 7.1013853549957275:
                                if Q.girth2_top5 > 4.7401830670423806e-05:
                                    return 'q'   # 60% of the training jets here get this class from the formula
                                else:
                                    return 'g'   # 82% of the training jets here get this class from the formula
                            else:
                                if Q.pt_6 > 48.71875:
                                    if Q.girth2_top5 > 6.440750803449191e-05:
                                        return 'q'   # 82% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.00019441302720224485:
                                            return 'g'   # 73% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 84% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 92% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt > 1356.5771484375:
                                if Q.n_particles > 27.5:
                                    if Q.pt_7 > 54.984375:
                                        return 'g'   # 71% of the training jets here get this class from the formula
                                    else:
                                        if Q.m012 > 3.473687171936035:
                                            return 'q'   # 89% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 49% of the training jets here get this class from the formula
                                else:
                                    return 'q'   # 89% of the training jets here get this class from the formula
                            else:
                                if Q.lam1 > 0.0030523017048835754:
                                    return 'W'   # 67% of the training jets here get this class from the formula
                                else:
                                    if Q.n_real_top40 > 28.5:
                                        if Q.pt_4 > 87.65625:
                                            if Q.pt_9 > 26.6875:
                                                if Q.lam2 > 0.0002870175230782479:
                                                    return 'g'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 72% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 91% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 99% of the training jets here get this class from the formula
                else:
                    if Q.n_real_top50 > 38.5:
                        if Q.sum_pt > 1063.7198486328125:
                            if Q.girth2_top15 > 0.0004814740823348984:
                                if Q.pt_6 > 40.328125:
                                    if Q.lam2 > 0.00048091493954416364:
                                        if Q.sum_pt > 1085.039794921875:
                                            if Q.tau32 > 0.6619839370250702:
                                                return 'g'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 68% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top40_slots > 0.9993376731872559:
                                                return 'q'   # 78% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top20 > 0.0011439252411946654:
                                                    return 'q'   # 60% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 92% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 75% of the training jets here get this class from the formula
                                else:
                                    if Q.girth2_top20 > 0.0010456604068167508:
                                        return 'q'   # 95% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam2 > 0.0005113114893902093:
                                            return 'g'   # 51% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                            else:
                                if Q.sum_pt_top2 > 510.03125:
                                    if Q.sum_pt_top50 > 1084.5166015625:
                                        if Q.sum_pt_top3 > 757.71875:
                                            return 'q'   # 85% of the training jets here get this class from the formula
                                        else:
                                            return 'g'   # 74% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 79% of the training jets here get this class from the formula
                                else:
                                    if Q.n_dr_0p2_0p4 > 3.5:
                                        return 'g'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.tau21 > 0.7466153204441071:
                                            return 'g'   # 81% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 68% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top30 > 901.9462890625:
                                if Q.girth2_top15 > 0.00019470023835310712:
                                    if Q.lam1 > 0.003738011815585196:
                                        if Q.n_dr_0_0p05 > 20.5:
                                            if Q.sum_pt_top50 > 980.2100830078125:
                                                return 'W'   # 77% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 76% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top10 > 0.0038746176287531853:
                                                if Q.sum_pt > 970.0155029296875:
                                                    return 'q'   # 57% of the training jets here get this class from the formula
                                                else:
                                                    return 't'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top10 > 0.0006830245838500559:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                                else:
                                                    if Q.sum_pt_top30 > 988.642578125:
                                                        return 'W'   # 76% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top15 > 0.00034752232022583485:
                                            return 'q'   # 97% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top50 > 1041.03857421875:
                                                if Q.n_particles > 42.5:
                                                    return 'g'   # 55% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                            else:
                                                if Q.sum_pt_top40 > 961.49755859375:
                                                    return 'q'   # 95% of the training jets here get this class from the formula
                                                else:
                                                    if Q.tau32 > 0.8615171015262604:
                                                        if Q.pt_dispersion > 0.36624006927013397:
                                                            return 'q'   # 93% of the training jets here get this class from the formula
                                                        else:
                                                            if Q.lam2 > 0.00040209175494965166:
                                                                return 'g'   # 78% of the training jets here get this class from the formula
                                                            else:
                                                                return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 85% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt > 1035.4752197265625:
                                        if Q.z_top40_slots > 0.9998802542686462:
                                            if Q.pt_9 > 27.4609375:
                                                if Q.lam2 > 0.0003540964826242998:
                                                    return 'g'   # 82% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 84% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_dr_0p2_0p4 > 0.003851112793199718:
                                                if Q.sum_pt_top3 > 611.46875:
                                                    return 'q'   # 73% of the training jets here get this class from the formula
                                                else:
                                                    return 'g'   # 78% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 74% of the training jets here get this class from the formula
                                    else:
                                        if Q.sum_pt_top30 > 947.8779296875:
                                            if Q.z_top40_slots > 0.9992575943470001:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.lam2 > 0.000383522201445885:
                                                    if Q.pt_dispersion > 0.3093526065349579:
                                                        return 'q'   # 80% of the training jets here get this class from the formula
                                                    else:
                                                        if Q.tau32 > 0.829591304063797:
                                                            return 'g'   # 64% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 85% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 92% of the training jets here get this class from the formula
                                        else:
                                            if Q.z_top40_slots > 0.9997332394123077:
                                                return 'q'   # 66% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 75% of the training jets here get this class from the formula
                            else:
                                if Q.mass_top15 > 19.397384643554688:
                                    if Q.sum_pt_top2 > 299.625:
                                        if Q.girth2_top3 > 0.0030051006469875574:
                                            return 't'   # 56% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 88% of the training jets here get this class from the formula
                                    else:
                                        if Q.mass_top15 > 41.87945365905762:
                                            return 't'   # 61% of the training jets here get this class from the formula
                                        else:
                                            if Q.n_dr_0p2_0p4 > 5.5:
                                                if Q.tau32 > 0.7139376401901245:
                                                    return 'g'   # 63% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 68% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                                else:
                                    if Q.tau32 > 0.7180951237678528:
                                        if Q.sum_pt_top5 > 600.875:
                                            if Q.girth2_top5 > 6.860812572995201e-05:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 77% of the training jets here get this class from the formula
                                        else:
                                            if Q.lam2 > 0.0004323648608988151:
                                                return 'g'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.tau21 > 0.6590141355991364:
                                                    return 'g'   # 75% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 79% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 67% of the training jets here get this class from the formula
                    else:
                        if Q.lam1 > 0.0032328892266377807:
                            if Q.sum_pt_top30 > 990.76171875:
                                if Q.z_dr_0_0p05 > 0.9191054105758667:
                                    if Q.n_dr_0p2_0p4 > 10.5:
                                        if Q.lam1 > 0.003822560072876513:
                                            return 'W'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.girth2_top15 > 0.00108694180380553:
                                                return 'q'   # 86% of the training jets here get this class from the formula
                                            else:
                                                return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        if Q.eccentricity > 0.9233573973178864:
                                            return 'W'   # 94% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass_top40 > 63.08951950073242:
                                                if Q.pt_4 > 40.15625:
                                                    return 'W'   # 79% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 45% of the training jets here get this class from the formula
                                            else:
                                                if Q.n_dr_0p1_0p2 > 2.5:
                                                    return 'q'   # 80% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 65% of the training jets here get this class from the formula
                                else:
                                    if Q.lam1 > 0.004038718529045582:
                                        return 'W'   # 59% of the training jets here get this class from the formula
                                    else:
                                        if Q.girth2_top10 > 0.0008047690789680928:
                                            return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            if Q.mass > 64.96173858642578:
                                                return 'W'   # 67% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 83% of the training jets here get this class from the formula
                            else:
                                if Q.girth2_top5 > 0.0033305794931948185:
                                    if Q.tau32 > 0.7606441974639893:
                                        if Q.girth2_top3 > 0.0038201927673071623:
                                            return 't'   # 77% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 47% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 63% of the training jets here get this class from the formula
                                else:
                                    if Q.sum_pt_top20 > 979.3779296875:
                                        return 'W'   # 55% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 88% of the training jets here get this class from the formula
                        else:
                            if Q.sum_pt_top10 > 618.2578125:
                                if Q.planar_flow > 0.23875875771045685:
                                    if Q.n_real_top40 > 35.5:
                                        if Q.log_sum_pt > 6.981511116027832:
                                            if Q.sum_pt_top3 > 600.09375:
                                                return 'q'   # 93% of the training jets here get this class from the formula
                                            else:
                                                if Q.girth2_top10 > 0.00010030785415438004:
                                                    if Q.lam2 > 0.0003869297361234203:
                                                        if Q.pt_9 > 40.046875:
                                                            return 'g'   # 85% of the training jets here get this class from the formula
                                                        else:
                                                            return 'q'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 86% of the training jets here get this class from the formula
                                                else:
                                                    if Q.lam2 > 0.00023097788653103635:
                                                        return 'g'   # 82% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 75% of the training jets here get this class from the formula
                                        else:
                                            if Q.sum_pt_top30 > 894.22998046875:
                                                return 'q'   # 98% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top15 > 15.463991641998291:
                                                    return 'q'   # 93% of the training jets here get this class from the formula
                                                else:
                                                    if Q.pt_6 > 37.265625:
                                                        return 'g'   # 65% of the training jets here get this class from the formula
                                                    else:
                                                        return 'q'   # 73% of the training jets here get this class from the formula
                                    else:
                                        if Q.lam1 > 0.0026639042189344764:
                                            if Q.girth2_top15 > 0.0004119107034057379:
                                                return 'q'   # 94% of the training jets here get this class from the formula
                                            else:
                                                if Q.mass_top30 > 56.341678619384766:
                                                    return 'W'   # 74% of the training jets here get this class from the formula
                                                else:
                                                    return 'q'   # 90% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 100% of the training jets here get this class from the formula
                                else:
                                    if Q.z_dr_0p2_0p4 > 0.015517937950789928:
                                        if Q.C2 > 0.08828232064843178:
                                            if Q.sum_pt_top20 > 975.4959716796875:
                                                if Q.z_top10_slots > 0.9677812457084656:
                                                    return 'q'   # 61% of the training jets here get this class from the formula
                                                else:
                                                    return 'W'   # 82% of the training jets here get this class from the formula
                                            else:
                                                return 'q'   # 96% of the training jets here get this class from the formula
                                        else:
                                            return 'q'   # 78% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 97% of the training jets here get this class from the formula
                            else:
                                if Q.z_top30_slots > 0.9995734393596649:
                                    if Q.sum_pt_top5 > 324.84375:
                                        return 'q'   # 90% of the training jets here get this class from the formula
                                    else:
                                        return 'g'   # 60% of the training jets here get this class from the formula
                                else:
                                    if Q.D2 > 2.3447901010513306:
                                        if Q.mass_top10 > 11.681162357330322:
                                            return 'q'   # 67% of the training jets here get this class from the formula
                                        else:
                                            if Q.eccentricity > 0.7201952338218689:
                                                return 'q'   # 57% of the training jets here get this class from the formula
                                            else:
                                                return 'g'   # 88% of the training jets here get this class from the formula
                                    else:
                                        return 'q'   # 82% of the training jets here get this class from the formula


def classify(pt, eta, phi):
    return decide(quantities(pt, eta, phi))


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    print('class:', classify(pt, eta, phi))
